#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <cstdlib>
#include <cmath>

// Constants
const int TRANSFER_MATRIX_SIZE = 512;
const int GEMM_MATRIX_SIZE = 2048;
const int NUM_MATRICES = 2;
const double TOTAL_RUNTIME_MS = 60000.0;
const double CYCLE_RUNTIME_MS = 1000.0;
const long long TRANSFER_MATRIX_BYTES = (long long)TRANSFER_MATRIX_SIZE * TRANSFER_MATRIX_SIZE * sizeof(float);
const long long GEMM_MATRIX_BYTES = (long long)GEMM_MATRIX_SIZE * GEMM_MATRIX_SIZE * sizeof(float);
const long long FLOPS_PER_GEMM = (2LL * GEMM_MATRIX_SIZE * GEMM_MATRIX_SIZE * GEMM_MATRIX_SIZE) + (3LL * GEMM_MATRIX_SIZE * GEMM_MATRIX_SIZE);


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

int main() {
    double preparation_start = get_milliseconds();

    printf("Transfer matrix size: %dx%d (%.3f MB each)\n", 
           TRANSFER_MATRIX_SIZE, TRANSFER_MATRIX_SIZE, (double)TRANSFER_MATRIX_BYTES/(1000.0 * 1000.0));
    printf("GEMM matrix size: %dx%d (%.6f GFLOPS each)\n", 
           GEMM_MATRIX_SIZE, GEMM_MATRIX_SIZE, (double)FLOPS_PER_GEMM / 1000.0 * 1000.0 * 1000.0);
    printf("Total predefined runtime: %.1f seconds\n\n", TOTAL_RUNTIME_MS / 1000.0);
    
    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR: creating cuBLAS handle\n");
        return 1;
    }
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    // Allocate matrices for transfers
    float* h_base_matrices[NUM_MATRICES];
    
    for (int i = 0; i < NUM_MATRICES; i++) {
        h_base_matrices[i] = (float*)malloc(TRANSFER_MATRIX_BYTES);
        generate_random_matrix(h_base_matrices[i], TRANSFER_MATRIX_SIZE, i * 1000);
    }
    
    float* d_transfer_matrixA, *d_transfer_matrixB, *d_transfer_matrixC;
    cudaMalloc(&d_transfer_matrixA, TRANSFER_MATRIX_BYTES);
    cudaMalloc(&d_transfer_matrixB, TRANSFER_MATRIX_BYTES);
    cudaMalloc(&d_transfer_matrixC, TRANSFER_MATRIX_BYTES);

    // Allocate matrices for GEMM operations
    float* d_gemm_matrixA, *d_gemm_matrixB, *d_gemm_matrixC;
    cudaMalloc(&d_gemm_matrixA, GEMM_MATRIX_BYTES);
    cudaMalloc(&d_gemm_matrixB, GEMM_MATRIX_BYTES);
    cudaMalloc(&d_gemm_matrixC, GEMM_MATRIX_BYTES);

    // Initialize GEMM matrices
    float* h_gemm_matrix = (float*)malloc(GEMM_MATRIX_BYTES);
    generate_random_matrix(h_gemm_matrix, GEMM_MATRIX_SIZE, 12345);
    cudaMemcpy(d_gemm_matrixA, h_gemm_matrix, GEMM_MATRIX_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gemm_matrixB, h_gemm_matrix, GEMM_MATRIX_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gemm_matrixC, h_gemm_matrix, GEMM_MATRIX_BYTES, cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.1f;
    
    double preparation_time = get_milliseconds() - preparation_start;
    printf("\nTotal preparation time: %.2f seconds\n", preparation_time / 1000.0);

    double total_start = get_milliseconds();
    int cycle = 0;
    
    while ((get_milliseconds() - total_start) < TOTAL_RUNTIME_MS) {
        double second_start = get_milliseconds();
        printf("cycle %d: ", ++cycle);
        
        int transfer_count = 0;
        int gemm_count = 0;
        
        // Interleaved pattern: many tiny operations throughout entire 1 second
        int op_index = 0;
        while ((get_milliseconds() - second_start) < CYCLE_RUNTIME_MS) {
            // 1:1 ratio but many more operations total
            bool do_transfer = (op_index % 2 == 0);  
            
            if (do_transfer) {
                // Transfer operation
                int base_matrix_idx = transfer_count % NUM_MATRICES;
                
                float* target_matrix;
                if (transfer_count % 3 == 0) target_matrix = d_transfer_matrixA;
                else if (transfer_count % 3 == 1) target_matrix = d_transfer_matrixB;
                else target_matrix = d_transfer_matrixC;
                
                cudaMemcpy(target_matrix, h_base_matrices[base_matrix_idx], TRANSFER_MATRIX_BYTES, cudaMemcpyHostToDevice);
                transfer_count++;
            } else {
                // GEMM operation
                cublasStatus_t gemm_status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                        GEMM_MATRIX_SIZE, GEMM_MATRIX_SIZE, GEMM_MATRIX_SIZE,
                                                        &alpha, d_gemm_matrixA, GEMM_MATRIX_SIZE,
                                                        d_gemm_matrixB, GEMM_MATRIX_SIZE,
                                                        &beta, d_gemm_matrixC, GEMM_MATRIX_SIZE);
                if (gemm_status != CUBLAS_STATUS_SUCCESS) {
                    printf("GEMM operation failed\n");
                    break;
                }
                gemm_count++;
                
                // every 100 gemm operations, switch matrix A and B
                if (gemm_count % 100 == 0) {
                    float* temp = d_gemm_matrixA;
                    d_gemm_matrixA = d_gemm_matrixC;
                    d_gemm_matrixC = temp;
                }
            }
            
            op_index++;
            
            // Break if we're approaching 1 second limit
            if ((get_milliseconds() - second_start) > CYCLE_RUNTIME_MS) break;
        }
        
        cudaDeviceSynchronize();
        double total_second_time = get_milliseconds() - second_start;
        
        // Calculate actual metrics
        double total_data_transferred = (double)transfer_count * TRANSFER_MATRIX_BYTES;
        double total_bandwidth = total_data_transferred / (total_second_time / 1000.0) / (1000.0 * 1000.0 * 1000.0);
        double total_flops = (double)gemm_count * FLOPS_PER_GEMM;
        double total_gflops = total_flops / (total_second_time / 1000.0) / (1000.0 * 1000.0 * 1000.0);
        
        printf("Total Data Transfers: %d times, Total GEMM %d ops, BW: %.1f GB/s, Perf: %.1f GFLOPS, Time: %.1fms\n",
               transfer_count, gemm_count, total_bandwidth, total_gflops, total_second_time);
    }
    
    double total_time = get_milliseconds() - total_start;
    printf("\nTotal runtime: %.2f seconds\n", total_time / 1000.0);
    
    // Cleanup
    for (int i = 0; i < NUM_MATRICES; i++) {
        free(h_base_matrices[i]);
    }
    free(h_gemm_matrix);
    cudaFree(d_transfer_matrixA);
    cudaFree(d_transfer_matrixB);
    cudaFree(d_transfer_matrixC);
    cudaFree(d_gemm_matrixA);
    cudaFree(d_gemm_matrixB);
    cudaFree(d_gemm_matrixC);
    cublasDestroy(handle);
    
    return 0;
}