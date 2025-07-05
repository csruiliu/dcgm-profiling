#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// Constants
const int TRANSFER_MATRIX_SIZE = 16384;
const int GEMM_MATRIX_SIZE = 4096;
const int NUM_MATRICES = 2;
const double TOTAL_RUNTIME_MS = 60000.0;
const double PHASE_RUNTIME_MS = 250.0;
const long long TRANSFER_MATRIX_BYTES = (long long)TRANSFER_MATRIX_SIZE * TRANSFER_MATRIX_SIZE * sizeof(float);
const long long GEMM_MATRIX_BYTES = (long long)GEMM_MATRIX_SIZE * GEMM_MATRIX_SIZE * sizeof(float);
const long long FLOPS_PER_GEMM = (2LL * GEMM_MATRIX_SIZE * GEMM_MATRIX_SIZE * GEMM_MATRIX_SIZE) + (3LL * GEMM_MATRIX_SIZE * GEMM_MATRIX_SIZE);

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
    {OP_TRANSFER, 250.0},
    {OP_COMPUTE, 250.0},
    {OP_TRANSFER, 250.0},
    {OP_COMPUTE, 250.0}
};

// Global execution context
struct ExecutionContext {
    float** h_transfer_matrices;
    float* h_gemm_matrix;
    float *d_transfer_matrix;
    float *d_gemm_matrixA, *d_gemm_matrixB, *d_gemm_matrixC;
    cublasHandle_t handle;
    float alpha, beta;
    
    // Statistics
    int total_transfer_count;
    int total_gemm_count;
    double total_transfer_time;
    double total_compute_time;
};

double get_milliseconds() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0;
}

void generate_random_matrix(float* matrix, int size, int seed) {
    srand(seed);
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

// Execute transfer phase
void execute_transfer_phase(ExecutionContext* ctx, double duration_ms) {
    double phase_start = get_milliseconds();
    int phase_transfers = 0;
    
    while ((get_milliseconds() - phase_start) < duration_ms) {
        int matrix_idx = phase_transfers % NUM_MATRICES;
        
        double transfer_start = get_milliseconds();
        cudaMemcpy(ctx->d_transfer_matrix, ctx->h_transfer_matrices[matrix_idx], 
                  TRANSFER_MATRIX_BYTES, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        double transfer_time = get_milliseconds() - transfer_start;
        
        ctx->total_transfer_time += transfer_time;
        phase_transfers++;
    }
    ctx->total_transfer_count += phase_transfers;
}

// Execute compute phase
void execute_compute_phase(ExecutionContext* ctx, double duration_ms) {
    double phase_start = get_milliseconds();
    int phase_gemm = 0;
    
    while ((get_milliseconds() - phase_start) < duration_ms) {
        double compute_start = get_milliseconds();
        cublasSgemm(ctx->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   GEMM_MATRIX_SIZE, GEMM_MATRIX_SIZE, GEMM_MATRIX_SIZE,
                   &ctx->alpha,
                   ctx->d_gemm_matrixA, GEMM_MATRIX_SIZE,
                   ctx->d_gemm_matrixB, GEMM_MATRIX_SIZE,
                   &ctx->beta,
                   ctx->d_gemm_matrixC, GEMM_MATRIX_SIZE);
        cudaDeviceSynchronize();
        double compute_time = get_milliseconds() - compute_start;
        
        ctx->total_compute_time += compute_time;
        phase_gemm++;
    }
    ctx->total_gemm_count += phase_gemm;
}

// Execute a cycle with the given pattern
void execute_cycle(ExecutionContext* ctx, ExecutionPhase* pattern, int pattern_length) {
    // Reset statistics for this cycle
    ctx->total_transfer_count = 0;
    ctx->total_gemm_count = 0;
    ctx->total_transfer_time = 0.0;
    ctx->total_compute_time = 0.0;
    
    for (int i = 0; i < pattern_length; i++) {        
        if (pattern[i].type == OP_TRANSFER) {
            execute_transfer_phase(ctx, pattern[i].duration_ms);
        } else {
            execute_compute_phase(ctx, pattern[i].duration_ms);
        }
    }
}


int main() {
    double preparation_start = get_milliseconds();
    
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

    // Check if we have enough memory before proceeding
    printf("Each transfer matrix requires %.2f GB\n", TRANSFER_MATRIX_BYTES / (1000.0 * 1000.0 * 1000.0));
    printf("Total host memory for transfer matrices: %.2f GB\n", NUM_MATRICES * TRANSFER_MATRIX_BYTES / (1000.0 * 1000.0 * 1000.0));

    // Initialize execution context
    ExecutionContext ctx = {0};
    cublasCreate(&ctx.handle);
    ctx.alpha = 1.0f;
    ctx.beta = 0.1f;

    // Check GPU memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t required_gpu_mem = TRANSFER_MATRIX_BYTES + 3 * GEMM_MATRIX_BYTES; // Only one transfer matrix needed
    printf("Required GPU memory Allocation: %.2f GB, Available: %.2f GB\n", 
           required_gpu_mem / (1000.0 * 1000.0 * 1000.0),
           free_mem / (1000.0 * 1000.0 * 1000.0));
    
    if (required_gpu_mem > free_mem) {
        printf("Error: Insufficient GPU memory!\n");
        return -1;
    }
    
    // Allocate host matrices
    ctx.h_transfer_matrices = (float**)malloc(NUM_MATRICES * sizeof(float*));
    for (int i = 0; i < NUM_MATRICES; i++) {
        ctx.h_transfer_matrices[i] = (float*)malloc(TRANSFER_MATRIX_BYTES);
        if (!ctx.h_transfer_matrices[i]) {
            printf("Error: Failed to allocate host memory for matrix %d\n", i);
            return -1;
        }
        generate_random_matrix(ctx.h_transfer_matrices[i], TRANSFER_MATRIX_SIZE, i);
    }

    // GEMM matrix
    ctx.h_gemm_matrix = (float*)malloc(GEMM_MATRIX_BYTES);
    if (!ctx.h_gemm_matrix) {
        printf("Error: Failed to allocate GEMM matrix memory\n");
        return -1;
    }
    generate_random_matrix(ctx.h_gemm_matrix, GEMM_MATRIX_SIZE, 42);

    // Allocate device matrices
    cudaMalloc(&ctx.d_transfer_matrix, TRANSFER_MATRIX_BYTES);
    cudaMalloc(&ctx.d_gemm_matrixA, GEMM_MATRIX_BYTES);
    cudaMalloc(&ctx.d_gemm_matrixB, GEMM_MATRIX_BYTES);
    cudaMalloc(&ctx.d_gemm_matrixC, GEMM_MATRIX_BYTES);
    
    // Initialize GEMM matrices
    cudaMemcpy(ctx.d_gemm_matrixA, ctx.h_gemm_matrix, GEMM_MATRIX_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.d_gemm_matrixB, ctx.h_gemm_matrix, GEMM_MATRIX_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.d_gemm_matrixC, ctx.h_gemm_matrix, GEMM_MATRIX_BYTES, cudaMemcpyHostToDevice);
        
    double preparation_time = get_milliseconds() - preparation_start;
    printf("\nTotal preparation time: %.2f seconds\n", preparation_time / 1000.0);
    
    // Main execution loop
    double total_start = get_milliseconds();
    int cycle = 0;
    
    while ((get_milliseconds() - total_start) < TOTAL_RUNTIME_MS) {
        double cycle_start = get_milliseconds();
        printf("Cycle %d: ", ++cycle);
        
        execute_cycle(&ctx, current_pattern, pattern_length);
        
        double cycle_time = get_milliseconds() - cycle_start;
        
        // Calculate actual metrics
        double total_data_transferred = (double)ctx.total_transfer_count * TRANSFER_MATRIX_BYTES;
        double total_bandwidth = total_data_transferred / (cycle_time / 1000.0) / (1000.0 * 1000.0 * 1000.0);
        double total_flops = (double)ctx.total_gemm_count * FLOPS_PER_GEMM;
        double total_gflops = total_flops / (cycle_time / 1000.0) / (1000.0 * 1000.0 * 1000.0);
        
        printf("Total Data Transfers: %d times, Total GEMM %d ops, BW: %.1f GB/s, Perf: %.1f GFLOPS, Time: %.1fms\n",
               ctx.total_transfer_count, ctx.total_gemm_count, total_bandwidth, total_gflops, cycle_time);
    }
    
    double total_time = get_milliseconds() - total_start;
    printf("\nTotal runtime: %.2f seconds\n", total_time / 1000.0);
    
    // Cleanup
    for (int i = 0; i < NUM_MATRICES; i++) {
        free(ctx.h_transfer_matrices[i]);
    }
    free(ctx.h_transfer_matrices);
    free(ctx.h_gemm_matrix);
    cudaFree(ctx.d_transfer_matrix);
    cudaFree(ctx.d_gemm_matrixA);
    cudaFree(ctx.d_gemm_matrixB);
    cudaFree(ctx.d_gemm_matrixC);
    cublasDestroy(ctx.handle);
    
    return 0;
}
