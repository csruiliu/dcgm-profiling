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
const int NUM_GPU_MEMORY_MATRICES = 4;  // Multiple matrices for GPU memory operations
const double TOTAL_RUNTIME_MS = 60000.0;

// Precision control - change this type to switch precision
using precision_t = float; 

// Helper function to get precision name
template<typename T>
const char* get_precision_name();

template<>
const char* get_precision_name<float>() { return "SINGLE"; }

template<>
const char* get_precision_name<double>() { return "DOUBLE"; }

// Simple but robust timer 
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
    OP_MEMORY_OP,
    OP_COMPUTE
};

struct ExecutionPhase {
    OperationType type;
    double duration_ms;
};

// Define your execution patterns here
ExecutionPhase EXEC_PATTERN[] = {
    {OP_MEMORY_OP, 100.0},
    {OP_COMPUTE, 400.0},
    {OP_MEMORY_OP, 100.0},
    {OP_COMPUTE, 400.0},
};

template<typename T>
struct ExecutionContext {
    // Pure GPU memory operation matrices
    int memory_matrix_size;
    T** d_memory_matrices;  // Multiple matrices for GPU memory operations
    T* d_temp_matrix;       // Temporary matrix for memory operations
    
    // Pure compute matrices (separate from memory operation matrices)
    int compute_matrix_size;
    T* d_compute_matrixA;
    T* d_compute_matrixB;
    T* d_compute_matrixC;
    
    T alpha, beta;

    // Counters
    int total_memory_ops_count;
    int total_compute_count;

    cublasHandle_t handle;
    ExecutionContext() : total_memory_ops_count(0), total_compute_count(0) {
        alpha = T(1.0);
        beta = T(0.1);
    }
};

template<typename T>
void generate_random_matrix(T* matrix, int size, int seed) {
    srand(seed);
    for (int i = 0; i < size * size; i++) {
        matrix[i] = static_cast<T>(rand()) / RAND_MAX;
    }
}

// Specializations for cuBLAS GEMM
template<typename T>
cublasStatus_t cublas_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n, int k, const T* alpha,
                          const T* A, int lda, const T* B, int ldb,
                          const T* beta, T* C, int ldc);

template<>
cublasStatus_t cublas_gemm<float>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                 int m, int n, int k, const float* alpha,
                                 const float* A, int lda, const float* B, int ldb,
                                 const float* beta, float* C, int ldc) {
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
cublasStatus_t cublas_gemm<double>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                  int m, int n, int k, const double* alpha,
                                  const double* A, int lda, const double* B, int ldb,
                                  const double* beta, double* C, int ldc) {
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<typename T>
bool initialize_context(ExecutionContext<T>& ctx, int memory_matrix_size, int compute_matrix_size, size_t memory_matrix_bytes, size_t compute_matrix_bytes) {
    ctx.memory_matrix_size = memory_matrix_size;
    ctx.compute_matrix_size = compute_matrix_size;
    
    // Create cuBLAS handle
    cublasStatus_t status = cublasCreate(&ctx.handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR: creating cuBLAS handle\n");
        return false;
    }
    cublasSetMathMode(ctx.handle, CUBLAS_DEFAULT_MATH);

    // Allocate GPU memory for memory bandwidth operations
    ctx.d_memory_matrices = new T*[NUM_GPU_MEMORY_MATRICES];
    for (int i = 0; i < NUM_GPU_MEMORY_MATRICES; i++) {
        cudaMalloc(&ctx.d_memory_matrices[i], memory_matrix_bytes);
        
        // Initialize with some data
        T* h_temp = new T[memory_matrix_size * memory_matrix_size];
        generate_random_matrix(h_temp, memory_matrix_size, i * 1000);
        cudaMemcpy(ctx.d_memory_matrices[i], h_temp, memory_matrix_bytes, cudaMemcpyHostToDevice);
        delete[] h_temp;
    }
    
    // Allocate temporary matrix for memory operations
    cudaMalloc(&ctx.d_temp_matrix, memory_matrix_bytes);
    
    // Allocate GPU memory for pure compute operations (separate matrices)
    cudaMalloc(&ctx.d_compute_matrixA, compute_matrix_bytes);
    cudaMalloc(&ctx.d_compute_matrixB, compute_matrix_bytes);
    cudaMalloc(&ctx.d_compute_matrixC, compute_matrix_bytes);
    
    // Initialize compute matrices
    T* h_compute_matrix = new T[compute_matrix_size * compute_matrix_size];
    generate_random_matrix(h_compute_matrix, compute_matrix_size, 12345);
    cudaMemcpy(ctx.d_compute_matrixA, h_compute_matrix, compute_matrix_bytes, cudaMemcpyHostToDevice);
    
    generate_random_matrix(h_compute_matrix, compute_matrix_size, 54321);
    cudaMemcpy(ctx.d_compute_matrixB, h_compute_matrix, compute_matrix_bytes, cudaMemcpyHostToDevice);
    
    // Initialize result matrix to zeros
    cudaMemset(ctx.d_compute_matrixC, 0, compute_matrix_bytes);
    
    delete[] h_compute_matrix;
    
    printf("Initialized GPU memory matrices: %d x %d (%zu bytes each)\n", 
           memory_matrix_size, memory_matrix_size, memory_matrix_bytes);
    printf("Initialized compute matrices: %d x %d (%zu bytes each)\n", 
           compute_matrix_size, compute_matrix_size, compute_matrix_bytes);
    
    return true;
}

template<typename T>
void execute_pure_gpu_memory_phase(ExecutionContext<T>* ctx, double duration_ms) {
    Timer memory_timer;
    int phase_memory_ops = 0;
    int matrix_src = 0, matrix_dst = 1;
    int operation_type = 0;
    
    const size_t matrix_bytes = ctx->memory_matrix_size * ctx->memory_matrix_size * sizeof(T);
    
    printf("  Pure GPU Memory Phase: zero-computation memory operations for %.0fms\n", duration_ms);
    
    while (memory_timer.elapsed_ms() < duration_ms) {
        switch (operation_type % 3) {
            case 0:
                // matrix to temp memcpy
                cudaMemcpy(ctx->d_temp_matrix, ctx->d_memory_matrices[matrix_src], 
                           matrix_bytes, cudaMemcpyDeviceToDevice);
                break;
                                
            case 1:
                // temp to matrix memcpy
                cudaMemcpy(ctx->d_memory_matrices[matrix_dst], ctx->d_temp_matrix, 
                           matrix_bytes, cudaMemcpyDeviceToDevice);
                break;
                
            case 2:
                // Direct matrix-to-matrix copy
                cudaMemcpy(ctx->d_memory_matrices[matrix_dst], ctx->d_memory_matrices[matrix_src], 
                           matrix_bytes, cudaMemcpyDeviceToDevice);
                break;
        }
        
        phase_memory_ops++;
        operation_type++;
        matrix_src = (matrix_src + 1) % NUM_GPU_MEMORY_MATRICES;
        matrix_dst = (matrix_dst + 1) % NUM_GPU_MEMORY_MATRICES;
    }
    
    ctx->total_memory_ops_count += phase_memory_ops;
    printf("    Completed %d zero-computation memory operations\n", phase_memory_ops);
}

template<typename T>
void execute_pure_compute_phase(ExecutionContext<T>* ctx, double duration_ms) {
    Timer compute_timer;
    int phase_compute = 0;
    
    printf("  Pure Compute Phase: performing GEMM operations for %.0fms\n", duration_ms);
    
    while (compute_timer.elapsed_ms() < duration_ms) {
        // Pure GEMM computation - no memory transfers, just computation
        cublas_gemm<T>(ctx->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                      ctx->compute_matrix_size, ctx->compute_matrix_size, ctx->compute_matrix_size,
                      &ctx->alpha,
                      ctx->d_compute_matrixA, ctx->compute_matrix_size,
                      ctx->d_compute_matrixB, ctx->compute_matrix_size,
                      &ctx->beta,
                      ctx->d_compute_matrixC, ctx->compute_matrix_size);
        
        cudaDeviceSynchronize();
        phase_compute++;
    }
    
    ctx->total_compute_count += phase_compute;
    printf("    Completed %d GEMM operations\n", phase_compute);
}

template<typename T>
void execute_cycle(ExecutionContext<T>* ctx, ExecutionPhase* pattern, int pattern_length) {
    ctx->total_memory_ops_count = 0;
    ctx->total_compute_count = 0;
    
    for (int i = 0; i < pattern_length; i++) {
        if (pattern[i].type == OP_MEMORY_OP) {
            execute_pure_gpu_memory_phase<T>(ctx, pattern[i].duration_ms);
        } else {
            execute_pure_compute_phase<T>(ctx, pattern[i].duration_ms);
        }
    }
}

template<typename T>
void cleanup_context(ExecutionContext<T>& ctx) {
    for (int i = 0; i < NUM_GPU_MEMORY_MATRICES; i++) {
        cudaFree(ctx.d_memory_matrices[i]);
    }
    delete[] ctx.d_memory_matrices;
    
    cudaFree(ctx.d_temp_matrix);
    cudaFree(ctx.d_compute_matrixA);
    cudaFree(ctx.d_compute_matrixB);
    cudaFree(ctx.d_compute_matrixC);
    cublasDestroy(ctx.handle);
}

template<typename T>
void bursty_execution(int memory_matrix_size, int compute_matrix_size) {
    // Preparation
    Timer timer;
    
    printf("Using %s precision operations\n", get_precision_name<T>());

    const size_t memory_matrix_bytes = memory_matrix_size * memory_matrix_size * sizeof(T);
    const size_t compute_matrix_bytes = compute_matrix_size * compute_matrix_size * sizeof(T);
    const long long FLOPS_PER_GEMM = (2LL * compute_matrix_size * compute_matrix_size * compute_matrix_size) + 
                                     (3LL * compute_matrix_size * compute_matrix_size);

    // Choose your execution pattern here
    ExecutionPhase* current_pattern = EXEC_PATTERN;
    int pattern_length = sizeof(EXEC_PATTERN) / sizeof(ExecutionPhase);

    printf("Using execution pattern with %d phases:\n", pattern_length);
    for (int i = 0; i < pattern_length; i++) {
        printf("  Phase %d: %s (%.0fms)\n", i + 1,
               current_pattern[i].type == OP_MEMORY_OP ? "Pure GPU Memory Ops" : "Pure Compute",
               current_pattern[i].duration_ms);
    }
    printf("\n");

    ExecutionContext<T> ctx;
    if (!initialize_context(ctx, memory_matrix_size, compute_matrix_size, memory_matrix_bytes, compute_matrix_bytes)) {
        printf("Failed to initialize execution context\n");
        return;
    }

    double cycle_duration = 0;
    for (int i = 0; i < pattern_length; i++) {
        cycle_duration += current_pattern[i].duration_ms;
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
        execute_cycle(&ctx, current_pattern, pattern_length);
        double cycle_time = cycle_timer.elapsed_ms();
        
        // Calculate actual metrics
        double total_memory_data = (double)ctx.total_memory_ops_count * memory_matrix_bytes;
        double total_bandwidth = total_memory_data / (cycle_time / 1000.0) / (1000.0 * 1000.0 * 1000.0);
        double total_flops = (double)ctx.total_compute_count * FLOPS_PER_GEMM;
        double total_gflops = total_flops / (cycle_time / 1000.0) / (1000.0 * 1000.0 * 1000.0);
        
        printf("Total GPU Memory Ops: %d, Total Computation: %d, BW: %.1f GB/s, Perf: %.1f GFLOPS, Time: %.1fms\n",
               ctx.total_memory_ops_count, ctx.total_compute_count, total_bandwidth, total_gflops, cycle_time);
    }
    
    double total_time = timer.elapsed_ms();
    printf("\nTotal runtime: %.2f seconds\n", total_time / 1000.0);
    
    // Cleanup
    cleanup_context(ctx);
}

int main(int argc, char* argv[]) {
    int memory_matrix_size = 2048;  // Size for GPU memory bandwidth operations
    int compute_matrix_size = 1024; // Size for pure compute operations
    
    if (argc >= 2) {
        memory_matrix_size = atoi(argv[1]);
    }
    if (argc >= 3) {
        compute_matrix_size = atoi(argv[2]);
    }
    
    printf("GPU Memory Bandwidth & Pure Compute Benchmark\n");
    printf("Memory matrix size: %d x %d\n", memory_matrix_size, memory_matrix_size);
    printf("Compute matrix size: %d x %d\n", compute_matrix_size, compute_matrix_size);
    printf("==================================================\n");
    
    bursty_execution<precision_t>(memory_matrix_size, compute_matrix_size);
    
    return 0;
}