#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <memory>

// Constants
const int NUM_MEMORY_MATRICES = 4;
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

// Execution context
template<typename T>
struct ExecutionContext {
    // GPU matrices for pure memory operations (GPU-to-GPU copies)
    T* d_source_matrices[NUM_MEMORY_MATRICES];  // Source matrices on GPU
    T* d_transfer_matrix;                         // Destination matrix for GPU copies
    
    // GPU matrices for pure computation (GEMM operations)
    T* d_gemm_matrixA;
    T* d_gemm_matrixB; 
    T* d_gemm_matrixC;
    
    // CUBLAS handle and parameters
    cublasHandle_t handle;
    T alpha, beta;
    
    // Host matrix for initial setup only
    std::unique_ptr<T[]> h_gemm_matrix;
};

// Random matrix generation
template<typename T>
void generate_random_matrix(T* matrix, int size, int seed) {
    srand(seed);
    for (int i = 0; i < size * size; i++) {
        matrix[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX) * 2.0 - 1.0;
    }
}

template<typename T>
bool initialize_context(ExecutionContext<T>& ctx, int transfer_matrix_size, int gemm_matrix_size) {
    size_t transfer_matrix_bytes = transfer_matrix_size * transfer_matrix_size * sizeof(T);
    size_t gemm_matrix_bytes = gemm_matrix_size * gemm_matrix_size * sizeof(T);
    
    // Initialize CUBLAS
    cublasCreate(&ctx.handle);
    ctx.alpha = static_cast<T>(1.0);
    ctx.beta = static_cast<T>(0.0);
    
    // Allocate and initialize GEMM matrix (used for initial setup only)
    ctx.h_gemm_matrix = std::make_unique<T[]>(gemm_matrix_size * gemm_matrix_size);    
    generate_random_matrix(ctx.h_gemm_matrix.get(), gemm_matrix_size, 12345);

    // Check memory requirements
    printf("Each transfer matrix requires %.2f GB\n", transfer_matrix_bytes / (1000.0 * 1000.0 * 1000.0));
    printf("Total GPU memory for source matrices: %.2f GB\n", NUM_MEMORY_MATRICES * transfer_matrix_bytes / (1000.0 * 1000.0 * 1000.0));
    
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    // Calculate required GPU memory: source matrices + destination + 3 GEMM matrices
    size_t required_gpu_mem = (NUM_MEMORY_MATRICES + 1) * transfer_matrix_bytes + 3 * gemm_matrix_bytes; 
    printf("Required GPU memory allocation: %.2f GB, Available: %.2f GB\n", 
           required_gpu_mem / (1000.0 * 1000.0 * 1000.0),
           free_mem / (1000.0 * 1000.0 * 1000.0));
    
    if (required_gpu_mem > free_mem) {
        printf("Error: Insufficient GPU memory!\n");
        return false;
    }

    // Allocate GPU source matrices for memory operations
    for (int i = 0; i < NUM_MEMORY_MATRICES; i++) {
        cudaMalloc(&ctx.d_source_matrices[i], transfer_matrix_bytes);
        // Initialize each source matrix with different patterns
        generate_random_matrix(ctx.h_gemm_matrix.get(), transfer_matrix_size, 12345 + i);
        cudaMemcpy(ctx.d_source_matrices[i], ctx.h_gemm_matrix.get(), transfer_matrix_bytes, cudaMemcpyHostToDevice);
    }
    
    // Allocate destination matrix for memory operations
    cudaMalloc(&ctx.d_transfer_matrix, transfer_matrix_bytes);
    
    // Allocate device matrices for computation operations
    cudaMalloc(&ctx.d_gemm_matrixA, gemm_matrix_bytes);
    cudaMalloc(&ctx.d_gemm_matrixB, gemm_matrix_bytes);
    cudaMalloc(&ctx.d_gemm_matrixC, gemm_matrix_bytes);
    
    // Initialize GEMM matrices (one-time setup)
    generate_random_matrix(ctx.h_gemm_matrix.get(), gemm_matrix_size, 54321);
    cudaMemcpy(ctx.d_gemm_matrixA, ctx.h_gemm_matrix.get(), gemm_matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.d_gemm_matrixB, ctx.h_gemm_matrix.get(), gemm_matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.d_gemm_matrixC, ctx.h_gemm_matrix.get(), gemm_matrix_bytes, cudaMemcpyHostToDevice);

    return true;
}

template<typename T>
void cleanup_context(ExecutionContext<T>& ctx) {
    // Free GPU source matrices
    for (int i = 0; i < NUM_MEMORY_MATRICES; i++) {
        cudaFree(ctx.d_source_matrices[i]);
    }
    
    // Free other GPU matrices
    cudaFree(ctx.d_transfer_matrix);
    cudaFree(ctx.d_gemm_matrixA);
    cudaFree(ctx.d_gemm_matrixB);
    cudaFree(ctx.d_gemm_matrixC);
    
    // Destroy CUBLAS handle
    cublasDestroy(ctx.handle);
}

template<typename T>
void interleaved_execution(int transfer_matrix_size, int gemm_matrix_size) {
    Timer timer;

    printf("Using %s precision GEMM operations\n", get_precision_name<precision_t>());
    printf("Starting interleaved execution: Pure GPU memory operations + Pure computation\n");
    printf("Transfer matrix size: %d x %d\n", transfer_matrix_size, transfer_matrix_size);
    printf("GEMM matrix size: %d x %d\n", gemm_matrix_size, gemm_matrix_size);
    printf("Cycle duration: %.1f ms\n", CYCLE_RUNTIME_MS);
    printf("Total runtime: %.1f seconds\n", TOTAL_RUNTIME_MS / 1000.0);

    ExecutionContext<T> ctx;
    if (!initialize_context(ctx, transfer_matrix_size, gemm_matrix_size)) {
        return;
    }

    const size_t transfer_matrix_bytes = transfer_matrix_size * transfer_matrix_size * sizeof(T);
    const size_t gemm_matrix_bytes = gemm_matrix_size * gemm_matrix_size * sizeof(T);
    const long long FLOPS_PER_GEMM = (2LL * gemm_matrix_size * gemm_matrix_size * gemm_matrix_size) + (3LL * gemm_matrix_size * gemm_matrix_size);

    int cycle_memory_count = 0;
    int cycle_computation_count = 0;
    int matrix_index = 0;
    int op_index = 0;

    Timer cycle_timer;
    Timer total_timer;
    
    printf("\n--- Starting Execution ---\n");

    while (total_timer.elapsed_ms() < TOTAL_RUNTIME_MS) {
        cycle_timer.reset();

        // Execute operations for one cycle (1 second)
        while (cycle_timer.elapsed_ms() < CYCLE_RUNTIME_MS) {
            // Alternate between memory operations and computation operations
            bool do_transfer = (op_index % 2 == 0);

            if (do_transfer) {
                // PURE GPU MEMORY OPERATION - GPU-to-GPU copy only
                cudaMemcpy(ctx.d_transfer_matrix, ctx.d_source_matrices[matrix_index], 
                          transfer_matrix_bytes, cudaMemcpyDeviceToDevice);
                cycle_memory_count++;
                matrix_index = cycle_memory_count % NUM_MEMORY_MATRICES;
            } else {
                // PURE COMPUTATION OPERATION - GEMM only, no memory transfers
                cublasStatus_t gemm_status = cublas_gemm<T>(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                           gemm_matrix_size, gemm_matrix_size, gemm_matrix_size,
                                                           &ctx.alpha, ctx.d_gemm_matrixA, gemm_matrix_size,
                                                           ctx.d_gemm_matrixB, gemm_matrix_size,
                                                           &ctx.beta, ctx.d_gemm_matrixC, gemm_matrix_size);
                if (gemm_status != CUBLAS_STATUS_SUCCESS) {
                    printf("GEMM operation failed\n");
                    break;
                }
                cycle_computation_count++;
            }
            op_index++;
        }

        // Synchronize to ensure all operations in this cycle are complete
        cudaDeviceSynchronize();

        double total_cycle_time = cycle_timer.elapsed_ms();
        
        // Calculate actual metrics
        double total_data_transferred = (double)cycle_memory_count * transfer_matrix_bytes;
        double total_bandwidth = total_data_transferred / (total_cycle_time / 1000.0) / (1000.0 * 1000.0 * 1000.0);
        double total_flops = (double)cycle_computation_count * FLOPS_PER_GEMM;
        double total_gflops = total_flops / (total_cycle_time / 1000.0) / (1000.0 * 1000.0 * 1000.0);
        
        printf("Total GPU Memory Ops: %d, Total Computation: %d, BW: %.1f GB/s, Perf: %.1f GFLOPS, Time: %.1fms\n",
                cycle_memory_count, cycle_computation_count, total_bandwidth, total_gflops, total_cycle_time);

        cycle_memory_count = 0;
        cycle_computation_count = 0;
    }
    
    double total_time = total_timer.elapsed_ms();
    printf("\nTotal runtime: %.2f seconds\n", total_time / 1000.0);
    
    cleanup_context(ctx);
}


int main(int argc, char* argv[]) {
    int transfer_matrix_size = 8192;  // Default transfer matrix size
    int gemm_matrix_size = 4096;      // Default GEMM matrix size
    
    if (argc >= 2) {
        transfer_matrix_size = atoi(argv[1]);
    }
    if (argc >= 3) {
        gemm_matrix_size = atoi(argv[2]);
    }
    
    printf("=== GPU Memory Operations + Pure Computation Interleaving ===\n");
    printf("Transfer matrix size: %d x %d\n", transfer_matrix_size, transfer_matrix_size);
    printf("GEMM matrix size: %d x %d\n", gemm_matrix_size, gemm_matrix_size);
    
    interleaved_execution<precision_t>(transfer_matrix_size, gemm_matrix_size);
    
    return 0;
}