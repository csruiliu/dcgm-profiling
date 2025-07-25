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

namespace gemm {

enum class Precision { SINGLE = 'S', DOUBLE = 'D', HALF = 'H' };

template<typename T>
struct CublasTraits;

template<>
struct CublasTraits<float> {
  static constexpr auto gemm_func = &cublasSgemm;
  static constexpr auto math_mode = CUBLAS_TF32_TENSOR_OP_MATH;
};

template<>
struct CublasTraits<double> {
  static constexpr auto gemm_func = &cublasDgemm;
  static constexpr auto math_mode = CUBLAS_DEFAULT_MATH;
};

template<>
struct CublasTraits<__half> {
  static constexpr auto gemm_func = &cublasHgemm;
  static constexpr auto math_mode = CUBLAS_TENSOR_OP_MATH;
};

template<typename T>
class Gemm {
private:
  int N_;

  // Matrices on Host
  T* h_matrixA; 
  T* h_matrixB;
  T* h_matrixC;

  // Matrices on Device
  T* d_matrixA;
  T* d_matrixB;
  T* d_matrixC;

  cublasHandle_t cublas_handle_;

public:
  Gemm(int N): N_(N) {}

  ~Gemm() {
      cleanup();
  }


  bool initialize() {
    return allocate_host_matrices() && allocate_gpu_matrices() && setup_cublas();
  }

  bool compute_gemm(int repeats, T alpha = T(1.0), T beta = T(0.0)) {
    for (int r = 0; r < repeats; ++r) {
        cublasStatus_t status = CublasTraits<T>::gemm_func(cublas_handle_,
                                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                                           N_, N_, N_,
                                                           &alpha,
                                                           d_matrixA, N_, d_matrixB, N_,
                                                           &beta,
                                                           d_matrixC, N_);
        handle_cublas_error(status, "gemm computation");
    }
        
    cudaDeviceSynchronize();  
    return true;
  }

  bool gather_results() {    
    cudaError_t err = cudaMemcpy(h_matrixC, d_matrixC, sizeof(T) * N_ * N_, cudaMemcpyDeviceToHost);
    handle_cuda_error(err, "cudaMemcpy C host to device");

    cudaDeviceSynchronize();
    return true;
  }

private:
    bool allocate_host_matrices() {
      std::cout << "Allocating Matrics on Host" << std::endl;
      // Use pinned memory for better transfer performance
      cudaError_t err = cudaMallocHost(&h_matrixA, sizeof(T) * N_ * N_);
      handle_cuda_error(err, "cudaMallocHost h_matrixA");
      
      err = cudaMallocHost(&h_matrixB, sizeof(T) * N_ * N_);
      handle_cuda_error(err, "cudaMallocHost h_matrixB");
      
      err = cudaMallocHost(&h_matrixC, sizeof(T) * N_ * N_);
      handle_cuda_error(err, "cudaMallocHost h_matrixC");
      
      // Initialize matrices with random values
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
      
      for (int i = 0; i < N_ * N_; ++i) {
          h_matrixA[i] = static_cast<T>(dis(gen));
          h_matrixB[i] = static_cast<T>(dis(gen));
          h_matrixC[i] = static_cast<T>(dis(gen));
      }
      return true;
    }

    bool allocate_gpu_matrices() {
      // Allocate matrices on host
      cudaError_t err = cudaMalloc(&d_matrixA, sizeof(T) * N_ * N_);
      handle_cuda_error(err, "cudaMalloc d_matrixA");
      
      err = cudaMalloc(&d_matrixB, sizeof(T) * N_ * N_);
      handle_cuda_error(err, "cudaMalloc d_matrixB");
      
      err = cudaMalloc(&d_matrixC, sizeof(T) * N_ * N_);
      handle_cuda_error(err, "cudaMalloc d_matrixC");
      
      // Copy from host to device
      err = cudaMemcpy(d_matrixA, h_matrixA, sizeof(T) * N_ * N_, cudaMemcpyHostToDevice);
      handle_cuda_error(err, "cudaMemcpy A host to device");
      
      err = cudaMemcpy(d_matrixB, h_matrixB, sizeof(T) * N_ * N_, cudaMemcpyHostToDevice);
      handle_cuda_error(err, "cudaMemcpy B host to device");
      
      err = cudaMemcpy(d_matrixC, h_matrixC, sizeof(T) * N_ * N_, cudaMemcpyHostToDevice);
      handle_cuda_error(err, "cudaMemcpy C host to device");

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
      
      cudaFree(d_matrixA);
      cudaFree(d_matrixB);
      cudaFree(d_matrixC);
      cudaFreeHost(h_matrixA);
      cudaFreeHost(h_matrixB);
      cudaFreeHost(h_matrixC);
            
      // Reset pointers
      d_matrixA = d_matrixB = d_matrixC = nullptr;
      h_matrixA = h_matrixB = h_matrixC = nullptr;
    }

    void handle_cuda_error(cudaError_t error, const std::string& operation) {
      if (error != cudaSuccess) {
          std::cerr << operation << " failed: " << cudaGetErrorString(error) << std::endl;
          exit(1);
      }
    }
    
    void handle_cublas_error(cublasStatus_t status, const std::string& operation) {
      if (status != CUBLAS_STATUS_SUCCESS) {
          std::cerr << operation << " failed with cuBLAS error " << status << std::endl;
          exit(1);
      }
    }
};

} // namespace gemm

int main(int argc, char* argv[]) {

  if (argc != 6) {
    std::cout << "Usage: " << argv[0] << " <N> <repeats> <alpha> <beta> <precision(S/D/H)>" << std::endl;
    std::cout << "  N: Matrix size (NxN)" << std::endl;
    std::cout << "  repeats: Number of GEMM iterations" << std::endl;
    std::cout << "  alpha, beta: GEMM coefficients" << std::endl;
    std::cout << "  precision: S(ingle), D(ouble), or H(alf)" << std::endl;
    return 1;
  }
  
  int N = std::atoi(argv[1]);
  int repeats = std::atoi(argv[2]);
  double alpha = std::atof(argv[3]);
  double beta = std::atof(argv[4]);
  char precision = argv[5][0];

  std::cout << "Starting distributed GEMM with:" << std::endl;
  std::cout << "  Matrix size: " << N << "x" << N << std::endl;
  std::cout << "  Repeats: " << repeats << std::endl;
  std::cout << "  Precision: " << precision << std::endl;

  auto start_time = std::chrono::high_resolution_clock::now();

  try {
    switch (precision) {
      case 'S': {
        gemm::Gemm<float> gemm(N);
        if (gemm.initialize() &&  
            gemm.compute_gemm(repeats, static_cast<float>(alpha), static_cast<float>(beta)) &&
            gemm.gather_results()) {
            std::cout << "Single precision GEMM completed successfully" << std::endl;
        }
        break;
      }
      case 'D': {
        gemm::Gemm<double> gemm(N);
        if (gemm.initialize() &&  
            gemm.compute_gemm(repeats, alpha, beta) &&
            gemm.gather_results()) {
            std::cout << "Double precision GEMM completed successfully" << std::endl;
        }
        break;
      }
      case 'H': {
        gemm::Gemm<__half> gemm(N);
        if (gemm.initialize() &&  
            gemm.compute_gemm(repeats, static_cast<__half>(alpha), static_cast<__half>(beta)) &&
            gemm.gather_results()) {
            std::cout << "Half precision GEMM completed successfully" << std::endl;
        }
        break;
      }
      default:
        std::cerr << "Invalid precision. Use S, D, or H" << std::endl;
        return 1;
    }
  } catch (const std::exception& e) {
    std::cerr <<"Exception: " << e.what() << std::endl;
    return 1;
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  std::cout << "Total execution time: " << duration.count() << " ms" << std::endl;

  return 0;
}