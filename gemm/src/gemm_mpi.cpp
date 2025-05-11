#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <mpi.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cuda_fp16.h>

// Uncomment for pageable host memory allocation
#define USE_PINNED_MEMORY
// Uncomment to use std::chrono for timing:
#define TIMING_CUDA_EVENTS
// Number of copies for sending
#define NUM_COPIES 10
// Sleep time in milliseconds
#define SLEEP_TIME 5000
// Number of GPUs in a node
#define NUM_NODE_GPUS 4

// ------------------------------------------------------- //
// Function: get_seconds
// ------------------------------------------------------- //
double get_seconds() {
  struct timeval now;
  gettimeofday(&now, NULL);
  const double seconds = (double)now.tv_sec;
  const double usec = (double)now.tv_usec;
  return seconds + (usec * 1.0e-6);
}

#define PRECISION 'S'
#include "calc_gemm_mpi.cpp"
#undef PRECISION

#define PRECISION 'D'
#include "calc_gemm_mpi.cpp"
#undef PRECISION

#define PRECISION 'H'
#include "calc_gemm_mpi.cpp"
#undef PRECISION

int main(int argc, char *argv[]) {
  // Default parameters
  int N = 4096;
  int repeats = 100;
  double alpha = 1.0;
  double beta = 1.0;
  char prec = 'D';

  // Initialize MPI
  MPI_Init(&argc, &argv);
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  
  // Arguments Parsing
  if (argc > 1) {
    N = atoi(argv[1]);
    if (mpi_rank == 0) printf("Matrix size: %d\n", N);
  } else if (mpi_rank == 0) printf("Matrix size defaulted to %d\n", N);

  if (argc > 2) {
    repeats = atoi(argv[2]);
    if (mpi_rank == 0) printf("Repeat multiply %d times\n", repeats);
  } else if (mpi_rank == 0) printf("Repeat multiply defaulted to %d\n", repeats);

  if (argc > 3) alpha = atof(argv[3]);
  if (mpi_rank == 0) printf("Alpha = %f\n", alpha);

  if (argc > 4) beta = atof(argv[4]);
  if (mpi_rank == 0) printf("Beta = %f\n", beta);

  if (argc > 5) {
    if (argv[5][0] == 'S') prec = 'S';
    else if (argv[5][0] == 'D') prec = 'D';
    else if (argv[5][0] == 'H') prec = 'H';
    else if (mpi_rank == 0) printf("Precision '%s' not recognized, using default '%c'\n", argv[5], prec);
  } else if (mpi_rank == 0) printf("Precision defaulted to %c\n", prec);
  if (mpi_rank == 0) printf("Precision = %c\n", prec);
  
  // Check number of MPI processes
  if (mpi_size != NUM_NODE_GPUS) {
    if (mpi_rank == 0)
    std::cerr << "This program assumes "<< NUM_NODE_GPUS <<" MPI processes" << std::endl;
    MPI_Finalize();
    return 1;
  } 
  std::cout << "I am rank " << mpi_rank << " of " << mpi_size << std::endl;
  
  // Check number of available GPUs
  int num_gpus;
  cudaError_t err = cudaGetDeviceCount(&num_gpus);
  if (err != cudaSuccess || num_gpus == 0) {
    std::cerr << "Rank " << mpi_rank << ": No CUDA-capable devices found: " << cudaGetErrorString(err) << std::endl;
    MPI_Finalize();
    return 1;
  }
  std::cout << "Rank " << mpi_rank << ": Detected " << num_gpus << " GPUs" << std::endl;
  
  // Set GPU device based on rank
  err = cudaSetDevice(mpi_rank);
  if (err != cudaSuccess) {
    std::cerr << "Rank " << mpi_rank << ": cudaSetDevice failed: " << cudaGetErrorString(err) << std::endl;
    MPI_Finalize();
    return 1;
  }
  std::cout << "Rank " << mpi_rank << ": cudaSetDevice sucessfully" << std::endl;

  // Compute local size
  int local_N = N / mpi_size;
  if (local_N * mpi_size != N) {
    if (mpi_rank == 0) printf("N must be divisible by %d\n", mpi_size);
    MPI_Finalize();
    return 1;
  }

  double time_taken;
  int sizeof_gemm_t;

  switch (prec) {
    case 'S': {
      float *full_A_host = nullptr, *full_B_host = nullptr, *full_C_host = nullptr;
      float *local_A_host, *local_C_host, *B_host;
      float *local_A_gpu, *local_B_gpu, *local_C_gpu;

      // Allocate local host buffers
      local_A_host = (float *)malloc(sizeof(float) * local_N * N);
      local_C_host = (float *)malloc(sizeof(float) * local_N * N);
      B_host = (float *)malloc(sizeof(float) * N * N);
      
      // Rank 0 allocates and initializes full matrices
      if (mpi_rank == 0) {
        alloc_gemm(N, &full_A_host, &full_B_host, &full_C_host);
      }

      // Distribute data
      MPI_Scatter(full_A_host, local_N * N, MPI_FLOAT, local_A_host, local_N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Scatter(full_C_host, local_N * N, MPI_FLOAT, local_C_host, local_N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Bcast(B_host, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

      // Allocate GPU memory
      cudaMalloc(&local_A_gpu, sizeof(float) * local_N * N);
      cudaMalloc(&local_C_gpu, sizeof(float) * local_N * N);
      cudaMalloc(&local_B_gpu, sizeof(float) * N * N);

      // Transfer to GPU
      cudaMemcpy(local_A_gpu, local_A_host, sizeof(float) * local_N * N, cudaMemcpyHostToDevice);
      cudaMemcpy(local_C_gpu, local_C_host, sizeof(float) * local_N * N, cudaMemcpyHostToDevice);
      cudaMemcpy(local_B_gpu, B_host, sizeof(float) * N * N, cudaMemcpyHostToDevice);

      // Perform computation
      time_taken = calc_gemm(repeats, local_N, N, N, alpha, beta, local_A_gpu, local_B_gpu, local_C_gpu);

      // Transfer result back to host
      cudaMemcpy(local_C_host, local_C_gpu, sizeof(float) * local_N * N, cudaMemcpyDeviceToHost);

      // Gather results on rank 0
      if (mpi_rank == 0) {
        MPI_Gather(local_C_host, local_N * N, MPI_FLOAT, full_C_host, local_N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        check_gemm(N, full_A_host, full_B_host, full_C_host);
      } else {
        MPI_Gather(local_C_host, local_N * N, MPI_FLOAT, nullptr, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
      }

      // Cleanup
      free(local_A_host);
      free(local_C_host);
      free(B_host);
      cudaFree(local_A_gpu);
      cudaFree(local_B_gpu);
      cudaFree(local_C_gpu);
      if (mpi_rank == 0) {
        free_gemm(full_A_host, full_B_host, full_C_host);
      }
      sizeof_gemm_t = sizeof(float);
      break;
    }
    case 'D': {
      double *full_A_host = nullptr, *full_B_host = nullptr, *full_C_host = nullptr;
      double *local_A_host, *local_C_host, *B_host;
      double *local_A_gpu, *local_B_gpu, *local_C_gpu;

      if (mpi_rank == 0) {
        alloc_gemm(N, &full_A_host, &full_B_host, &full_C_host);
      }

      local_A_host = (double *)malloc(sizeof(double) * local_N * N);
      local_C_host = (double *)malloc(sizeof(double) * local_N * N);
      B_host = (double *)malloc(sizeof(double) * N * N);

      MPI_Scatter(full_A_host, local_N * N, MPI_DOUBLE, local_A_host, local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Scatter(full_C_host, local_N * N, MPI_DOUBLE, local_C_host, local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(B_host, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      cudaMalloc(&local_A_gpu, sizeof(double) * local_N * N);
      cudaMalloc(&local_C_gpu, sizeof(double) * local_N * N);
      cudaMalloc(&local_B_gpu, sizeof(double) * N * N);

      cudaMemcpy(local_A_gpu, local_A_host, sizeof(double) * local_N * N, cudaMemcpyHostToDevice);
      cudaMemcpy(local_C_gpu, local_C_host, sizeof(double) * local_N * N, cudaMemcpyHostToDevice);
      cudaMemcpy(local_B_gpu, B_host, sizeof(double) * N * N, cudaMemcpyHostToDevice);

      time_taken = calc_gemm(repeats, local_N, N, N, alpha, beta, local_A_gpu, local_B_gpu, local_C_gpu);

      cudaMemcpy(local_C_host, local_C_gpu, sizeof(double) * local_N * N, cudaMemcpyDeviceToHost);

      if (mpi_rank == 0) {
        MPI_Gather(local_C_host, local_N * N, MPI_DOUBLE, full_C_host, local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        check_gemm(N, full_A_host, full_B_host, full_C_host);
      } else {
        MPI_Gather(local_C_host, local_N * N, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      }

      free(local_A_host);
      free(local_C_host);
      free(B_host);
      cudaFree(local_A_gpu);
      cudaFree(local_B_gpu);
      cudaFree(local_C_gpu);
      if (mpi_rank == 0) {
        free_gemm(full_A_host, full_B_host, full_C_host);
      }
      sizeof_gemm_t = sizeof(double);
      break;
    }
    case 'H': {
      __half *full_A_host = nullptr, *full_B_host = nullptr, *full_C_host = nullptr;
      __half *local_A_host, *local_C_host, *B_host;
      __half *local_A_gpu, *local_B_gpu, *local_C_gpu;

      if (mpi_rank == 0) {
        alloc_gemm(N, &full_A_host, &full_B_host, &full_C_host);
      }

      local_A_host = (__half *)malloc(sizeof(__half) * local_N * N);
      local_C_host = (__half *)malloc(sizeof(__half) * local_N * N);
      B_host = (__half *)malloc(sizeof(__half) * N * N);

      int byte_count = local_N * N * sizeof(__half);
      MPI_Scatter(full_A_host, byte_count, MPI_BYTE, local_A_host, byte_count, MPI_BYTE, 0, MPI_COMM_WORLD);
      MPI_Scatter(full_C_host, byte_count, MPI_BYTE, local_C_host, byte_count, MPI_BYTE, 0, MPI_COMM_WORLD);
      MPI_Bcast(B_host, N * N * sizeof(__half), MPI_BYTE, 0, MPI_COMM_WORLD);

      cudaMalloc(&local_A_gpu, sizeof(__half) * local_N * N);
      cudaMalloc(&local_C_gpu, sizeof(__half) * local_N * N);
      cudaMalloc(&local_B_gpu, sizeof(__half) * N * N);

      cudaMemcpy(local_A_gpu, local_A_host, sizeof(__half) * local_N * N, cudaMemcpyHostToDevice);
      cudaMemcpy(local_C_gpu, local_C_host, sizeof(__half) * local_N * N, cudaMemcpyHostToDevice);
      cudaMemcpy(local_B_gpu, B_host, sizeof(__half) * N * N, cudaMemcpyHostToDevice);

      time_taken = calc_gemm(repeats, local_N, N, N, alpha, beta, local_A_gpu, local_B_gpu, local_C_gpu);

      cudaMemcpy(local_C_host, local_C_gpu, sizeof(__half) * local_N * N, cudaMemcpyDeviceToHost);

      if (mpi_rank == 0) {
        MPI_Gather(local_C_host, byte_count, MPI_BYTE, full_C_host, byte_count, MPI_BYTE, 0, MPI_COMM_WORLD);
        check_gemm(N, full_A_host, full_B_host, full_C_host);
      } else {
        MPI_Gather(local_C_host, byte_count, MPI_BYTE, nullptr, 0, MPI_BYTE, 0, MPI_COMM_WORLD);
      }

      free(local_A_host);
      free(local_C_host);
      free(B_host);
      cudaFree(local_A_gpu);
      cudaFree(local_B_gpu);
      cudaFree(local_C_gpu);
      if (mpi_rank == 0) {
        free_gemm(full_A_host, full_B_host, full_C_host);
      }
      sizeof_gemm_t = sizeof(__half);
      break;
    }
    default:
      if (mpi_rank == 0) printf("Invalid precision\n");
      MPI_Finalize();
      return 1;
  }

  // Compute maximum time across all ranks
  double max_time;
  MPI_Reduce(&time_taken, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // Output results on rank 0
  if (mpi_rank == 0) {
    printf("\n===============================================================\n");
    double N_dbl = (double)N;
    double matrix_memory = (3 * N_dbl * N_dbl) * ((double)sizeof_gemm_t);
    printf("Memory for Matrices: %f MB\n", matrix_memory / (1024 * 1024));
    printf("Multiply time: %f seconds\n", max_time);
    const double flops_computed = (N_dbl * N_dbl * N_dbl * 2.0 * (double)repeats) + (N_dbl * N_dbl * 3 * (double)repeats);
    printf("GFLOP/s rate: %f GF/s\n", (flops_computed / max_time) / 1.0e9);
    printf("===============================================================\n\n");
  }

  MPI_Finalize();
  return 0;
}