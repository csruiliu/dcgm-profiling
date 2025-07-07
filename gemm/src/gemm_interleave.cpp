#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <memory>

// Constants
const int NUM_TRANSFER_MATRICES = 2;
const double TOTAL_RUNTIME_MS = 60000.0;
const double CYCLE_RUNTIME_MS = 1000.0;

// Precision control - change this type to switch precision
// 'float' for single precision
// 'double' for double precision
using precision_t = float; 

// Helper function to get precision name
template<typename T>
const char* get_precision_name();

template<>
const char* get_precision_name<float>() { return "SINGLE"; }

template<>
const char* get_precision_name<double>() { return "DOUBLE"; }

// Timer
class Timer {
    std::chrono::steady_clock::time_point start;
public:
    Timer() { reset(); }
    void reset() { start = std::chrono::steady_clock::now(); }
    double elapsed_ms() const { 
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); 
    }
};

// Template specializations for CUBLAS operations
template<typename T>
cublasStatus_t cublas_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k, const T *alpha,
                          const T *A, int lda, const T *B, int ldb,
                          const T *beta, T *C, int ldc);

template<>
cublasStatus_t cublas_gemm<float>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                 int m, int n, int k, const float *alpha,
                                 const float *A, int lda, const float *B, int ldb,
                                 const float *beta, float *C, int ldc) {
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
cublasStatus_t cublas_gemm<double>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                  int m, int n, int k, const double *alpha,
                                  const double *A, int lda, const double *B, int ldb,
                                  const double *beta, double *C, int ldc) {
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<typename T>
void generate_random_matrix(T* matrix, int size, int seed) {
    srand(seed);
    for (int i = 0; i < size * size; i++) {
        matrix[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    }
}

// Template-based execution context
template<typename T>
struct ExecutionContext {
    // Host matrices
    std::unique_ptr<T*[]> h_transfer_matrices;
    std::unique_ptr<T[]> h_gemm_matrix;
    
    // Device matrices
    T *d_transfer_matrix;
    T *d_gemm_matrixA, *d_gemm_matrixB, *d_gemm_matrixC;
    
    cublasHandle_t handle;
    T alpha, beta;
    
    // Statistics
    int total_transfer_count;
    int total_gemm_count;
    
    ExecutionContext() : total_transfer_count(0), total_gemm_count(0) {
        alpha = T(1.0);
        beta = T(0.1);
    }
};

template<typename T>
bool initialize_context(ExecutionContext<T>& ctx, int transfer_matrix_size, int gemm_matrix_size, size_t transfer_matrix_bytes, size_t gemm_matrix_bytes) {
    // Create cuBLAS handle
    cublasStatus_t status = cublasCreate(&ctx.handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR: creating cuBLAS handle\n");
        return false;
    }
    cublasSetMathMode(ctx.handle, CUBLAS_DEFAULT_MATH);

    // Allocate host memory for transfer matrices
    ctx.h_transfer_matrices = std::make_unique<T*[]>(NUM_TRANSFER_MATRICES);
    for (int i = 0; i < NUM_TRANSFER_MATRICES; i++) {
        ctx.h_transfer_matrices[i] = new T[transfer_matrix_size * transfer_matrix_size];
        generate_random_matrix(ctx.h_transfer_matrices[i], transfer_matrix_size, i * 1000);
    }

    // Allocate and initialize GEMM matrix
    ctx.h_gemm_matrix = std::make_unique<T[]>(gemm_matrix_size * gemm_matrix_size);    
    generate_random_matrix(ctx.h_gemm_matrix.get(), gemm_matrix_size, 12345);

    // Check if we have enough memory before proceeding
    printf("Each transfer matrix requires %.2f GB\n", transfer_matrix_bytes / (1000.0 * 1000.0 * 1000.0));
    printf("Total host memory for transfer matrices: %.2f GB\n", NUM_TRANSFER_MATRICES * gemm_matrix_bytes / (1000.0 * 1000.0 * 1000.0));
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    // Only one transfer matrix needed
    size_t required_gpu_mem = transfer_matrix_bytes + 3 * gemm_matrix_bytes; 
    printf("Required GPU memory Allocation: %.2f GB, Available: %.2f GB\n", 
           required_gpu_mem / (1000.0 * 1000.0 * 1000.0),
           free_mem / (1000.0 * 1000.0 * 1000.0));
    
    if (required_gpu_mem > free_mem) {
        printf("Error: Insufficient GPU memory!\n");
        return false;
    }

    // Allocate device matrices
    cudaMalloc(&ctx.d_transfer_matrix, transfer_matrix_bytes);
    cudaMalloc(&ctx.d_gemm_matrixA, gemm_matrix_bytes);
    cudaMalloc(&ctx.d_gemm_matrixB, gemm_matrix_bytes);
    cudaMalloc(&ctx.d_gemm_matrixC, gemm_matrix_bytes);
    
    // Initialize GEMM matrices
    cudaMemcpy(ctx.d_gemm_matrixA, ctx.h_gemm_matrix.get(), gemm_matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.d_gemm_matrixB, ctx.h_gemm_matrix.get(), gemm_matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.d_gemm_matrixC, ctx.h_gemm_matrix.get(), gemm_matrix_bytes, cudaMemcpyHostToDevice);

    return true;
}

template<typename T>
void cleanup_context(ExecutionContext<T>& ctx) {
    for (int i = 0; i < NUM_TRANSFER_MATRICES; i++) {
        delete[] ctx.h_transfer_matrices[i];
    }
    
    cudaFree(ctx.d_transfer_matrix);
    cudaFree(ctx.d_gemm_matrixA);
    cudaFree(ctx.d_gemm_matrixB);
    cudaFree(ctx.d_gemm_matrixC);
    cublasDestroy(ctx.handle);
}

template<typename T>
void interleaved_execution(int transfer_matrix_size, int gemm_matrix_size) {
    Timer timer;
    
    printf("Using %s precision GEMM operations\n", get_precision_name<precision_t>());

    const size_t transfer_matrix_bytes = transfer_matrix_size * transfer_matrix_size * sizeof(T);
    const size_t gemm_matrix_bytes = gemm_matrix_size * gemm_matrix_size * sizeof(T);
    const long long FLOPS_PER_GEMM = (2LL * gemm_matrix_size * gemm_matrix_size * gemm_matrix_size) + (3LL * gemm_matrix_size * gemm_matrix_size);

    // Initialize execution context
    ExecutionContext<precision_t> ctx;
    if (!initialize_context(ctx, transfer_matrix_size, gemm_matrix_size, transfer_matrix_bytes, gemm_matrix_bytes)) {
        return;
    }

    double preparation_time = timer.elapsed_ms();
    printf("\nTotal preparation time: %.2f seconds\n", preparation_time / 1000.0);

    // Main execution loop
    timer.reset();
    int cycle = 0;
    
    Timer cycle_timer;
    while (timer.elapsed_ms() < TOTAL_RUNTIME_MS) {
        printf("cycle %d: ", ++cycle);
        
        int transfer_count = 0;
        int gemm_count = 0;
        
        // Interleaved pattern: many tiny operations throughout entire 1 second
        int op_index = 0;
        
        cycle_timer.reset();
        while (cycle_timer.elapsed_ms() < CYCLE_RUNTIME_MS) {
            // 1:1 ratio but many more operations total
            bool do_transfer = (op_index % 2 == 0);  
            int matrix_index = 0;

            if (do_transfer) {
                // Transfer operation    
                cudaMemcpy(ctx.d_transfer_matrix, ctx.h_transfer_matrices[matrix_index], transfer_matrix_bytes, cudaMemcpyHostToDevice);
                transfer_count++;
                matrix_index = transfer_count % NUM_TRANSFER_MATRICES;
            } else {
                // GEMM operation
                cublasStatus_t gemm_status = cublas_gemm<T>(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                           gemm_matrix_size, gemm_matrix_size, gemm_matrix_size,
                                                           &ctx.alpha, ctx.d_gemm_matrixA, gemm_matrix_size,
                                                           ctx.d_gemm_matrixB, gemm_matrix_size,
                                                           &ctx.beta, ctx.d_gemm_matrixC, gemm_matrix_size);
                if (gemm_status != CUBLAS_STATUS_SUCCESS) {
                    printf("GEMM operation failed\n");
                    break;
                }
                gemm_count++;
                
                // every 100 gemm operations, switch matrix A and B
                if (gemm_count % 100 == 0) {
                    T* temp = ctx.d_gemm_matrixA;
                    ctx.d_gemm_matrixA = ctx.d_gemm_matrixC;
                    ctx.d_gemm_matrixC = temp;
                }
            }
            
            op_index++;
        }
        
        cudaDeviceSynchronize();
        double total_cycle_time = cycle_timer.elapsed_ms();
        
        // Calculate actual metrics
        double total_data_transferred = (double)transfer_count * transfer_matrix_bytes;
        double total_bandwidth = total_data_transferred / (total_cycle_time / 1000.0) / (1000.0 * 1000.0 * 1000.0);
        double total_flops = (double)gemm_count * FLOPS_PER_GEMM;
        double total_gflops = total_flops / (total_cycle_time / 1000.0) / (1000.0 * 1000.0 * 1000.0);
        
        printf("Total Data Transfers: %d times, Total GEMM %d ops, BW: %.1f GB/s, Perf: %.1f GFLOPS, Time: %.1fms\n",
               transfer_count, gemm_count, total_bandwidth, total_gflops, total_cycle_time);
    }
    
    double total_time = timer.elapsed_ms();
    printf("\nTotal runtime: %.2f seconds\n", total_time / 1000.0);
    
    // Cleanup
    cleanup_context(ctx);
}

int main(int argc, char* argv[]) {
    // Default values
    int transfer_matrix_size = 1024;
    int gemm_matrix_size = 2048;
    
    // Parse command line arguments
    if (argc >= 2) {
        transfer_matrix_size = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        gemm_matrix_size = std::atoi(argv[2]);
    }
    
    if (transfer_matrix_size <= 0 || gemm_matrix_size <= 0) {
        printf("Usage: %s [transfer_matrix_size] [gemm_matrix_size]\n", argv[0]);
        printf("Example: %s 1024 2048\n", argv[0]);
        return 1;
    }

    printf("Using transfer matrix size: %d x %d\n", transfer_matrix_size, transfer_matrix_size);
    printf("Using GEMM matrix size: %d x %d\n", gemm_matrix_size, gemm_matrix_size);
    printf("Using %s precision GEMM operations\n", get_precision_name<precision_t>());

    interleaved_execution<precision_t>(transfer_matrix_size, gemm_matrix_size);
    return 0;
}