#include <iostream>
#include <memory>
#include <chrono>
#include <random>
#include <algorithm>
#include <string>
#include <type_traits>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

namespace gemm_mpi {

enum class Precision { SINGLE = 'S', DOUBLE = 'D', HALF = 'H' };

template<typename T>
struct CublasTraits;

template<>
struct CublasTraits<float> {
    static constexpr auto gemm_func = &cublasSgemm;
    static constexpr auto math_mode = CUBLAS_TF32_TENSOR_OP_MATH;
    static constexpr auto mpi_type = MPI_FLOAT;
};

template<>
struct CublasTraits<double> {
    static constexpr auto gemm_func = &cublasDgemm;
    static constexpr auto math_mode = CUBLAS_DEFAULT_MATH;
    static constexpr auto mpi_type = MPI_DOUBLE;
};

template<>
struct CublasTraits<__half> {
    static constexpr auto gemm_func = &cublasHgemm;
    static constexpr auto math_mode = CUBLAS_TENSOR_OP_MATH;
    //MPI doesn't have a native data type for __half, use MPI_UINT16_T 
    static constexpr auto mpi_type = MPI_UINT16_T;
};

template<typename T>
class GemmMPI {
private:
    int mpi_rank_;
    int mpi_size_;
    int N_;
    int local_N_;
    
    // Host matrices (rank 0 only)
    T* full_A_host_;
    T* B_host_;
    T* full_C_host_;
    
    // GPU matrices
    T* full_A_gpu_;
    T* B_gpu_;
    T* full_C_gpu_;
    T* local_A_gpu_;
    T* local_C_gpu_;
    
    cublasHandle_t cublas_handle_;

public:
    GemmMPI(int N) 
        : N_(N), full_A_host_(nullptr), B_host_(nullptr), full_C_host_(nullptr),
          full_A_gpu_(nullptr), B_gpu_(nullptr), full_C_gpu_(nullptr),
          local_A_gpu_(nullptr), local_C_gpu_(nullptr), cublas_handle_(nullptr) {
        
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);
        local_N_ = N_ / mpi_size_;
        
        if (N_ % mpi_size_ != 0) {
            if (mpi_rank_ == 0) {
                std::cerr << "Matrix size N must be divisible by number of MPI processes" << std::endl;
            }
            MPI_Finalize();
            exit(1);
        }
        
        // Set GPU device based on rank
        cudaError_t err = cudaSetDevice(mpi_rank_);
        if (err != cudaSuccess) {
            std::cerr << "Rank " << mpi_rank_ << ": cudaSetDevice failed: " 
                      << cudaGetErrorString(err) << std::endl;
            MPI_Finalize();
            exit(1);
        }
    }
    
    ~GemmMPI() {
        cleanup();
    }
    
    bool initialize() {
        return allocate_host_matrices() && allocate_gpu_matrices() && setup_cublas();
    }
        
    bool distribute_data() {
        // Handle __half type with custom MPI communication
        MPI_Scatter(full_A_gpu_, local_N_ * N_, CublasTraits<T>::mpi_type,
                    local_A_gpu_, local_N_ * N_, CublasTraits<T>::mpi_type,
                    0, MPI_COMM_WORLD);
        
        MPI_Bcast(B_gpu_, N_ * N_, CublasTraits<T>::mpi_type, 0, MPI_COMM_WORLD);
        
        MPI_Scatter(full_C_gpu_, local_N_ * N_, CublasTraits<T>::mpi_type,
                    local_C_gpu_, local_N_ * N_, CublasTraits<T>::mpi_type,
                    0, MPI_COMM_WORLD);
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        cudaDeviceSynchronize();
        return true;
    }
    
    bool compute_gemm(int repeats, T alpha = T(1.0), T beta = T(0.0)) {
        T* temp_C = nullptr;
        cudaError_t err = cudaMalloc(&temp_C, sizeof(T) * local_N_ * N_);
        handle_cuda_error(err, "cudaMalloc temp_C");
        
        cublasStatus_t status = cublasSetMatrix(local_N_, N_, sizeof(T), local_C_gpu_, local_N_, temp_C, local_N_);
        handle_cublas_error(status, "cublasSetMatrix");
        
        for (int r = 0; r < repeats; ++r) {
            status = CublasTraits<T>::gemm_func(cublas_handle_,
                                                CUBLAS_OP_N, CUBLAS_OP_N,
                                                local_N_, N_, N_,
                                                &alpha,
                                                local_A_gpu_, local_N_,
                                                B_gpu_, N_,
                                                &beta,
                                                temp_C, local_N_);
            handle_cublas_error(status, "gemm computation");
        }
        
        cudaDeviceSynchronize();
        
        status = cublasGetMatrix(local_N_, N_, sizeof(T), temp_C, local_N_,
                               local_C_gpu_, local_N_);
        handle_cublas_error(status, "cublasGetMatrix");
        
        cudaFree(temp_C);
        return true;
    }
    
    bool gather_results() {
        MPI_Barrier(MPI_COMM_WORLD);
        if (mpi_rank_ == 0) {
            MPI_Gather(local_C_gpu_, local_N_ * N_, CublasTraits<T>::mpi_type,
                       full_C_host_, local_N_ * N_, CublasTraits<T>::mpi_type,
                       0, MPI_COMM_WORLD);
        } else {
            MPI_Gather(local_C_gpu_, local_N_ * N_, CublasTraits<T>::mpi_type,
                       nullptr, 0, CublasTraits<T>::mpi_type,
                       0, MPI_COMM_WORLD);
        }
        
        return true;
    }

private:
    bool allocate_host_matrices() {
        if (mpi_rank_ == 0) {
            // Use pinned memory for better transfer performance
            cudaError_t err = cudaMallocHost(&full_A_host_, sizeof(T) * N_ * N_);
            handle_cuda_error(err, "cudaMallocHost full_A_host");
            
            err = cudaMallocHost(&B_host_, sizeof(T) * N_ * N_);
            handle_cuda_error(err, "cudaMallocHost B_host");
            
            err = cudaMallocHost(&full_C_host_, sizeof(T) * N_ * N_);
            handle_cuda_error(err, "cudaMallocHost full_C_host");
            
            // Initialize matrices with random values
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
            
            for (int i = 0; i < N_ * N_; ++i) {
                full_A_host_[i] = static_cast<T>(dis(gen));
                B_host_[i] = static_cast<T>(dis(gen));
                full_C_host_[i] = static_cast<T>(dis(gen));
            }
        }
        return true;
    }
    
    bool allocate_gpu_matrices() {
        if (mpi_rank_ == 0) {
            // Allocate full matrices on rank 0
            cudaError_t err = cudaMalloc(&full_A_gpu_, sizeof(T) * N_ * N_);
            handle_cuda_error(err, "cudaMalloc full_A_gpu");
            
            err = cudaMalloc(&B_gpu_, sizeof(T) * N_ * N_);
            handle_cuda_error(err, "cudaMalloc B_gpu");
            
            err = cudaMalloc(&full_C_gpu_, sizeof(T) * N_ * N_);
            handle_cuda_error(err, "cudaMalloc full_C_gpu");
            
            // Copy from host to device
            err = cudaMemcpy(full_A_gpu_, full_A_host_, sizeof(T) * N_ * N_, cudaMemcpyHostToDevice);
            handle_cuda_error(err, "cudaMemcpy A host to device");
            
            err = cudaMemcpy(B_gpu_, B_host_, sizeof(T) * N_ * N_, cudaMemcpyHostToDevice);
            handle_cuda_error(err, "cudaMemcpy B host to device");
            
            err = cudaMemcpy(full_C_gpu_, full_C_host_, sizeof(T) * N_ * N_, cudaMemcpyHostToDevice);
            handle_cuda_error(err, "cudaMemcpy C host to device");
        } else {
            // Non-root ranks only need B matrix
            cudaError_t err = cudaMalloc(&B_gpu_, sizeof(T) * N_ * N_);
            handle_cuda_error(err, "cudaMalloc B_gpu");
        }
        
        // Allocate local matrices on all ranks
        cudaError_t err = cudaMalloc(&local_A_gpu_, sizeof(T) * local_N_ * N_);
        handle_cuda_error(err, "cudaMalloc local_A_gpu");
        
        err = cudaMalloc(&local_C_gpu_, sizeof(T) * local_N_ * N_);
        handle_cuda_error(err, "cudaMalloc local_C_gpu");
        
        cudaDeviceSynchronize();
        return true;
    }
    
    bool setup_cublas() {
        cublasStatus_t status = cublasCreate(&cublas_handle_);
        handle_cublas_error(status, "cublasCreate");
        
        status = cublasSetMathMode(cublas_handle_, CublasTraits<T>::math_mode);
        handle_cublas_error(status, "cublasSetMathMode");
        
        return true;
    }
            
    void cleanup() {
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
            cublas_handle_ = nullptr;
        }
        
        cudaFree(full_A_gpu_);
        cudaFree(B_gpu_);
        cudaFree(full_C_gpu_);
        cudaFree(local_A_gpu_);
        cudaFree(local_C_gpu_);
        
        if (mpi_rank_ == 0) {
            cudaFreeHost(full_A_host_);
            cudaFreeHost(B_host_);
            cudaFreeHost(full_C_host_);
        }
        
        // Reset pointers
        full_A_gpu_ = B_gpu_ = full_C_gpu_ = local_A_gpu_ = local_C_gpu_ = nullptr;
        full_A_host_ = B_host_ = full_C_host_ = nullptr;
    }
    
    void handle_cuda_error(cudaError_t error, const std::string& operation) {
        if (error != cudaSuccess) {
            std::cerr << "Rank " << mpi_rank_ << ": " << operation 
                      << " failed: " << cudaGetErrorString(error) << std::endl;
            MPI_Finalize();
            exit(1);
        }
    }
    
    void handle_cublas_error(cublasStatus_t status, const std::string& operation) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Rank " << mpi_rank_ << ": " << operation 
                      << " failed with cuBLAS error " << status << std::endl;
            MPI_Finalize();
            exit(1);
        }
    }
};

} // namespace gemm_mpi

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    if (argc != 6) {
        if (mpi_rank == 0) {
            std::cout << "Usage: " << argv[0] << " <N> <repeats> <alpha> <beta> <precision(S/D/H)>" << std::endl;
            std::cout << "  N: Matrix size (NxN)" << std::endl;
            std::cout << "  repeats: Number of GEMM iterations" << std::endl;
            std::cout << "  alpha, beta: GEMM coefficients" << std::endl;
            std::cout << "  precision: S(ingle), D(ouble), or H(alf)" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    int N = std::atoi(argv[1]);
    int repeats = std::atoi(argv[2]);
    double alpha = std::atof(argv[3]);
    double beta = std::atof(argv[4]);
    char precision = argv[5][0];
    
    if (mpi_rank == 0) {
        std::cout << "Starting distributed GEMM with:" << std::endl;
        std::cout << "  Matrix size: " << N << "x" << N << std::endl;
        std::cout << "  MPI processes: " << mpi_size << std::endl;
        std::cout << "  Repeats: " << repeats << std::endl;
        std::cout << "  Precision: " << precision << std::endl;
    }
    
    // Check number of available GPUs
    int num_gpus;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err != cudaSuccess || num_gpus == 0) {
        std::cerr << "Rank " << mpi_rank << ": No CUDA-capable devices found: " 
                  << cudaGetErrorString(err) << std::endl;
        MPI_Finalize();
        return 1;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        switch (precision) {
            case 'S': {
                gemm_mpi::GemmMPI<float> gemm(N);
                if (gemm.initialize() && gemm.distribute_data() && 
                    gemm.compute_gemm(repeats, static_cast<float>(alpha), static_cast<float>(beta)) && 
                    gemm.gather_results()) {
                    if (mpi_rank == 0) {
                        std::cout << "Single precision GEMM completed successfully" << std::endl;
                    }
                }
                break;
            }
            case 'D': {
                gemm_mpi::GemmMPI<double> gemm(N);
                if (gemm.initialize() && gemm.distribute_data() && 
                    gemm.compute_gemm(repeats, alpha, beta) && gemm.gather_results()) {
                    if (mpi_rank == 0) {
                        std::cout << "Double precision GEMM completed successfully" << std::endl;
                    }
                }
                break;
            }
            case 'H': {
                gemm_mpi::GemmMPI<__half> gemm(N);
                if (gemm.initialize() && gemm.distribute_data() && 
                    gemm.compute_gemm(repeats, static_cast<__half>(alpha), static_cast<__half>(beta)) && 
                    gemm.gather_results()) {
                    if (mpi_rank == 0) {
                        std::cout << "Half precision GEMM completed successfully" << std::endl;
                    }
                }
                break;
            }
            default:
                if (mpi_rank == 0) {
                    std::cerr << "Invalid precision. Use S, D, or H" << std::endl;
                }
                MPI_Finalize();
                return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Rank " << mpi_rank << ": Exception: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (mpi_rank == 0) {
        std::cout << "Total execution time: " << duration.count() << " ms" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}