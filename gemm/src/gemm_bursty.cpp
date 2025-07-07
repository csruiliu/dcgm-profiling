#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <vector>

// Constants
const int NUM_TRANSFER_MATRICES = 2;
const double TOTAL_RUNTIME_MS = 60000.0;

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

// Execution pattern configuration
enum OperationType {
    OP_TRANSFER,
    OP_COMPUTE
};

struct ExecutionPhase {
    OperationType type;
    double duration_ms;
};

// Define your execution patterns here
ExecutionPhase EXEC_PATTERN[] = {
    {OP_TRANSFER, 100.0},
    {OP_COMPUTE, 400.0},
    {OP_TRANSFER, 100.0},
    {OP_COMPUTE, 400.0},
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
    
    //Constant
    int transfer_matrix_size;
    int gemm_matrix_size;

    ExecutionContext() : total_transfer_count(0), total_gemm_count(0) {
        alpha = T(1.0);
        beta = T(0.1);
    }
};

template<typename T>
void generate_random_matrix(T* matrix, int size, int seed) {
    srand(seed);
    for (int i = 0; i < size * size; i++) {
        matrix[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    }
}

// Template functions for execution phases
template<typename T>
void execute_transfer_phase(ExecutionContext<T>* ctx, double duration_ms) {
    Timer transfer_timer;
    int phase_transfers = 0;
    int matrix_index = 0;
    
    while (transfer_timer.elapsed_ms() < duration_ms) {
        cudaMemcpy(ctx->d_transfer_matrix, 
                  ctx->h_transfer_matrices[matrix_index], 
                  ctx->transfer_matrix_size * ctx->transfer_matrix_size * sizeof(T), 
                  cudaMemcpyHostToDevice);

        phase_transfers++;
        matrix_index = phase_transfers % NUM_TRANSFER_MATRICES;
    }
    ctx->total_transfer_count += phase_transfers;
}

template<typename T>
void execute_compute_phase(ExecutionContext<T>* ctx, double duration_ms) {
    Timer compute_timer;
    int phase_gemm = 0;
    
    while (compute_timer.elapsed_ms() < duration_ms) {
        cublas_gemm<T>(ctx->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                      ctx->gemm_matrix_size, ctx->gemm_matrix_size, ctx->gemm_matrix_size,
                      &ctx->alpha,
                      ctx->d_gemm_matrixA, ctx->gemm_matrix_size,
                      ctx->d_gemm_matrixB, ctx->gemm_matrix_size,
                      &ctx->beta,
                      ctx->d_gemm_matrixC, ctx->gemm_matrix_size);
        cudaDeviceSynchronize();
        phase_gemm++;
    }
    
    ctx->total_gemm_count += phase_gemm;
}

template<typename T>
void execute_cycle(ExecutionContext<T>* ctx, ExecutionPhase* pattern, int pattern_length, double cycle_duration) {
    ctx->total_transfer_count = 0;
    ctx->total_gemm_count = 0;
    
    for (int i = 0; i < pattern_length; i++) {
        if (pattern[i].type == OP_TRANSFER) {
            execute_transfer_phase<T>(ctx, pattern[i].duration_ms);
        } else {
            execute_compute_phase<T>(ctx, pattern[i].duration_ms);
        }
    }
}

template<typename T>
bool initialize_context(ExecutionContext<T>& ctx, int transfer_matrix_size, int gemm_matrix_size, size_t transfer_matrix_bytes, size_t gemm_matrix_bytes) {
    ctx.transfer_matrix_size = transfer_matrix_size;
    ctx.gemm_matrix_size = gemm_matrix_size;
    
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
void bursty_execution(int transfer_matrix_size, int gemm_matrix_size) {
    Timer timer;
    
    printf("Using %s precision GEMM operations\n", get_precision_name<precision_t>());

    const size_t transfer_matrix_bytes = transfer_matrix_size * transfer_matrix_size * sizeof(T);
    const size_t gemm_matrix_bytes = gemm_matrix_size * gemm_matrix_size * sizeof(T);
    const long long FLOPS_PER_GEMM = (2LL * gemm_matrix_size * gemm_matrix_size * gemm_matrix_size) + (3LL * gemm_matrix_size * gemm_matrix_size);

    // Choose your execution pattern here
    ExecutionPhase* current_pattern = EXEC_PATTERN;
    int pattern_length = sizeof(EXEC_PATTERN) / sizeof(ExecutionPhase);
    
    printf("Using execution pattern with %d phases:\n", pattern_length);
    for (int i = 0; i < pattern_length; i++) {
        printf("  Phase %d: %s (%.0fms)\n", i + 1,
               current_pattern[i].type == OP_TRANSFER ? "Transfer" : "Compute",
               current_pattern[i].duration_ms);
    }
    printf("\n");

    // Calculate cycle duration
    double cycle_duration = 0.0;
    for (int i = 0; i < pattern_length; i++) {
        cycle_duration += current_pattern[i].duration_ms;
    }
    printf("Total predefined cycle duration: %.0fms\n", cycle_duration);    

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
        printf("Cycle %d: ", ++cycle);
        cycle_timer.reset();    
        execute_cycle(&ctx, current_pattern, pattern_length, cycle_duration);
        double cycle_time = cycle_timer.elapsed_ms();
        
        // Calculate actual metrics
        double total_data_transferred = (double)ctx.total_transfer_count * transfer_matrix_bytes;
        double total_bandwidth = total_data_transferred / (cycle_time / 1000.0) / (1000.0 * 1000.0 * 1000.0);
        double total_flops = (double)ctx.total_gemm_count * FLOPS_PER_GEMM;
        double total_gflops = total_flops / (cycle_time / 1000.0) / (1000.0 * 1000.0 * 1000.0);
        
        printf("Total Data Transfers: %d times, Total GEMM %d ops, BW: %.1f GB/s, Perf: %.1f GFLOPS, Time: %.1fms\n",
               ctx.total_transfer_count, ctx.total_gemm_count, total_bandwidth, total_gflops, cycle_time);
    
        printf("\n");
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

    bursty_execution<precision_t>(transfer_matrix_size, gemm_matrix_size);    
    return 0;
}
