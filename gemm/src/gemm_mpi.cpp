#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <mpi.h>
#include <sys/time.h>
#include <thread>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cuda_fp16.h>

// Uncomment for pageable host memory allocation
#define USE_PINNED_MEMORY
// Uncomment to use std::chrono for timing:
//#define TIMING_CUDA_EVENTS
// Number of copies for sending
#define NUM_COPIES 10
// Sleep time in milliseconds
#define SLEEP_TIME 42000
// Number of GPUs in a node
#define NUM_NODE_GPUS 4

// ------------------------------------------------------- //
// Function: get_seconds for current time
// ------------------------------------------------------- //
double get_milliseconds() {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  
  const double seconds = (double)now.tv_sec;
  const double nanosec = (double)now.tv_nsec;
  
  // Convert nanoseconds to milliseconds
  return (seconds * 1000.0) + (nanosec / 1000000.0);  
}

// ------------------------------------------------------- //
// Function to make both CPU and GPU idle
// ------------------------------------------------------- //
inline void sleep_cpu_gpu_idle(int milliseconds) {
  // Capture timestamp before sleep
  double sleep_start_time = get_milliseconds();
  
  // Increment the sleep counter (declare as extern in this file)
  extern int sleep_occurrences;
  extern double sleep_timestamps[];
  sleep_timestamps[sleep_occurrences] = sleep_start_time;
  sleep_occurrences++;

  // Ensure all GPU operations are complete first
  cudaDeviceSynchronize();
  
  // Now just sleep the CPU - GPU will be naturally idle
  std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
  
  // GPU remains idle during this time since no kernels are launched
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
  // Global starting time
  double global_start_time = get_milliseconds();

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
  if (argc != 6) {
    if (mpi_rank == 0) {
        printf("Usage: %s <N> <repeats> <alpha> <beta> <precision: S|D|H>\n", argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  N = atoi(argv[1]);
  repeats = atoi(argv[2]);
  alpha = atof(argv[3]);
  beta = atof(argv[4]);
  prec = argv[5][0];
  
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

  int sizeof_gemm_t;

  switch (prec) {
    case 'S': {
      // Full A, B, C matrics on host of rank 0 MPI process
      float *full_A_host = nullptr, *B_host = nullptr, *full_C_host = nullptr;
      // Full A, B, C matrics on gpu of rank 0 MPI process
      float *full_A_gpu, *B_gpu, *full_C_gpu;      
      // A, C matrics on GPU of each MPI process
      float *local_A_gpu, *local_C_gpu;
      
      // Rank 0 allocates and initializes full matrices on host
      if (mpi_rank == 0) {
        alloc_gemm_host(N, &full_A_host, &B_host, &full_C_host);
        alloc_gemm_gpu(mpi_rank, N, N, &full_A_gpu, &B_gpu, &full_C_gpu);
        
        // Copy data from host to GPU        
        cudaError_t cudaErr;
        cudaErr = cudaMemcpy(full_A_gpu, full_A_host, N * N * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaErr != cudaSuccess) {
          std::cerr << "Rank " << mpi_rank << ": cudaMemcpy full_A_host to local_A_gpu failed:" << cudaGetErrorString(cudaErr) << std::endl;
          MPI_Finalize();
          return 1;
        }
        cudaErr = cudaMemcpy(B_gpu, B_host, N * N * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaErr != cudaSuccess) {
          std::cerr << "Rank " << mpi_rank << ": cudaMemcpy full_B_host to local_B_gpu failed:" << cudaGetErrorString(cudaErr) << std::endl;
          MPI_Finalize();
          return 1;
        }
        cudaErr = cudaMemcpy(full_C_gpu, full_C_host, N * N * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaErr != cudaSuccess) {
          std::cerr << "Rank " << mpi_rank << ": cudaMemcpy full_C_host to local_C_gpu failed:" << cudaGetErrorString(cudaErr) << std::endl;
          MPI_Finalize();
          return 1;
        }
        
        // 
        cudaErr = cudaMalloc(&local_A_gpu, local_N * N * sizeof(float));
        if (cudaErr != cudaSuccess) {
          std::cerr << "Rank " << mpi_rank << "cudaMalloc local_A_gpu failed: " << cudaGetErrorString(cudaErr);
          MPI_Finalize();
          exit(1);
        }
        cudaErr = cudaMalloc(&local_C_gpu, local_N * N * sizeof(float));
        if (cudaErr != cudaSuccess) {
          std::cerr << "Rank " << mpi_rank << "cudaMalloc local_C_gpu failed: " << cudaGetErrorString(cudaErr);
          MPI_Finalize();
          exit(1);
        }
      } else {
        // Allocate Memory on GPU of each MPI process
        alloc_gemm_gpu(mpi_rank, local_N, N, &local_A_gpu, &B_gpu, &local_C_gpu);
      }

      cudaDeviceSynchronize();

      // Distribute data
      MPI_Scatter(full_A_gpu, local_N * N, MPI_FLOAT, local_A_gpu, local_N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Scatter(full_C_gpu, local_N * N, MPI_FLOAT, local_C_gpu, local_N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Bcast(B_gpu, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
      
      cudaDeviceSynchronize(); 

      // Perform computation
      calc_gemm(mpi_rank, repeats, local_N, N, N, alpha, beta, local_A_gpu, B_gpu, local_C_gpu);
      
      cudaDeviceSynchronize();     

      // Gather results on rank 0
      if (mpi_rank == 0) {
        MPI_Gather(local_C_gpu, local_N * N, MPI_FLOAT, full_C_host, local_N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
      } else {
        MPI_Gather(local_C_gpu, local_N * N, MPI_FLOAT, nullptr, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
      }
      
      cudaDeviceSynchronize();

      // Cleanup
      cudaFree(local_A_gpu);
      cudaFree(B_gpu);
      cudaFree(local_C_gpu);
      if (mpi_rank == 0) {
        cudaFreeHost(full_A_host);
        cudaFreeHost(B_host);
        cudaFreeHost(full_C_host);
        cudaFree(full_A_gpu);
        cudaFree(full_C_gpu);
      }
      sizeof_gemm_t = sizeof(float);
      break;
    }
    case 'D': {
      // Full A, B, C matrics on host of rank 0 MPI process
      double *full_A_host = nullptr, *B_host = nullptr, *full_C_host = nullptr;
      // Full A, B, C matrics on gpu of rank 0 MPI process
      double *full_A_gpu, *B_gpu, *full_C_gpu;      
      // A, C matrics on GPU of each MPI process
      double *local_A_gpu, *local_C_gpu;
      
      // Rank 0 allocates and initializes full matrices on host using cudaMallocHost
      if (mpi_rank == 0) {
        alloc_gemm_host(N, &full_A_host, &B_host, &full_C_host);
        alloc_gemm_gpu(mpi_rank, N, N, &full_A_gpu, &B_gpu, &full_C_gpu);
        
        // Copy from host to GPU        
        cudaError_t cudaErr;
        cudaErr = cudaMemcpy(full_A_gpu, full_A_host, N * N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaErr != cudaSuccess) {
          std::cerr << "Rank " << mpi_rank << ": cudaMemcpy full_A_host to local_A_gpu failed:" << cudaGetErrorString(cudaErr) << std::endl;
          MPI_Finalize();
          return 1;
        }
        cudaErr = cudaMemcpy(B_gpu, B_host, N * N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaErr != cudaSuccess) {
          std::cerr << "Rank " << mpi_rank << ": cudaMemcpy full_B_host to local_B_gpu failed:" << cudaGetErrorString(cudaErr) << std::endl;
          MPI_Finalize();
          return 1;
        }
        cudaErr = cudaMemcpy(full_C_gpu, full_C_host, N * N * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaErr != cudaSuccess) {
          std::cerr << "Rank " << mpi_rank << ": cudaMemcpy full_C_host to local_C_gpu failed:" << cudaGetErrorString(cudaErr) << std::endl;
          MPI_Finalize();
          return 1;
        }
        
        cudaErr = cudaMalloc(&local_A_gpu, local_N * N * sizeof(double));
        if (cudaErr != cudaSuccess) {
          std::cerr << "Rank " << mpi_rank << "cudaMalloc local_A_gpu failed: " << cudaGetErrorString(cudaErr);
          MPI_Finalize();
          exit(1);
        }
        cudaErr = cudaMalloc(&local_C_gpu, local_N * N * sizeof(double));
        if (cudaErr != cudaSuccess) {
          std::cerr << "Rank " << mpi_rank << "cudaMalloc local_C_gpu failed: " << cudaGetErrorString(cudaErr);
          MPI_Finalize();
          exit(1);
        }
      } else {
        // Allocate Memory on GPU of each MPI process
        alloc_gemm_gpu(mpi_rank, local_N, N, &local_A_gpu, &B_gpu, &local_C_gpu);
      }
        
      cudaDeviceSynchronize();
      
      // Distribute data
      MPI_Scatter(full_A_gpu, local_N * N, MPI_DOUBLE, local_A_gpu, local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Scatter(full_C_gpu, local_N * N, MPI_DOUBLE, local_C_gpu, local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(B_gpu, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      cudaDeviceSynchronize();
      // Perform computation
      calc_gemm(mpi_rank, repeats, local_N, N, N, alpha, beta, local_A_gpu, B_gpu, local_C_gpu);

      cudaDeviceSynchronize();

      // Gather results on rank 0
      if (mpi_rank == 0) {
        MPI_Gather(local_C_gpu, local_N * N, MPI_DOUBLE, full_C_host, local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      } else {
        MPI_Gather(local_C_gpu, local_N * N, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      }

      cudaDeviceSynchronize();

      // Cleanup
      cudaFree(local_A_gpu);
      cudaFree(B_gpu);
      cudaFree(local_C_gpu);
      if (mpi_rank == 0) {
        cudaFreeHost(full_A_host);
        cudaFreeHost(B_host);
        cudaFreeHost(full_C_host);
        cudaFree(full_A_gpu);
        cudaFree(full_C_gpu);
      }
      sizeof_gemm_t = sizeof(double);
      break;
    }
    case 'H': {
      // Full A, B, C matrics on host of rank 0 MPI process
      __half *full_A_host = nullptr, *B_host = nullptr, *full_C_host = nullptr;
      // Full A, B, C matrics on gpu of rank 0 MPI process
      __half *full_A_gpu, *B_gpu, *full_C_gpu;      
      // A, C matrics on GPU of each MPI process
      __half *local_A_gpu, *local_C_gpu;
      
      // Rank 0 allocates and initializes full matrices on host using cudaMallocHost
      if (mpi_rank == 0) {
        alloc_gemm_host(N, &full_A_host, &B_host, &full_C_host);
        alloc_gemm_gpu(mpi_rank, N, N, &full_A_gpu, &B_gpu, &full_C_gpu);
        
        // Copy from host to GPU        
        cudaError_t cudaErr;
        cudaErr = cudaMemcpy(full_A_gpu, full_A_host, sizeof(__half) * N * N, cudaMemcpyHostToDevice);
        if (cudaErr != cudaSuccess) {
          std::cerr << "Rank " << mpi_rank << ": cudaMemcpy full_A_host to local_A_gpu failed:" << cudaGetErrorString(cudaErr) << std::endl;
          MPI_Finalize();
          return 1;
        }
        cudaErr = cudaMemcpy(B_gpu, B_host, sizeof(__half) * N * N, cudaMemcpyHostToDevice);
        if (cudaErr != cudaSuccess) {
          std::cerr << "Rank " << mpi_rank << ": cudaMemcpy full_B_host to local_B_gpu failed:" << cudaGetErrorString(cudaErr) << std::endl;
          MPI_Finalize();
          return 1;
        }
        cudaErr = cudaMemcpy(full_C_gpu, full_C_host, sizeof(__half) * N * N, cudaMemcpyHostToDevice);
        if (cudaErr != cudaSuccess) {
          std::cerr << "Rank " << mpi_rank << ": cudaMemcpy full_C_host to local_C_gpu failed:" << cudaGetErrorString(cudaErr) << std::endl;
          MPI_Finalize();
          return 1;
        }

        cudaErr = cudaMalloc(&local_A_gpu, local_N * N * sizeof(__half));
        if (cudaErr != cudaSuccess) {
          std::cerr << "Rank " << mpi_rank << "cudaMalloc local_A_gpu failed: " << cudaGetErrorString(cudaErr);
          MPI_Finalize();
          exit(1);
        }
        cudaErr = cudaMalloc(&local_C_gpu, local_N * N * sizeof(__half));
        if (cudaErr != cudaSuccess) {
          std::cerr << "Rank " << mpi_rank << "cudaMalloc local_C_gpu failed: " << cudaGetErrorString(cudaErr);
          MPI_Finalize();
          exit(1);
        }
      } else {
        // Allocate Memory on GPU of each MPI process
        alloc_gemm_gpu(mpi_rank, local_N, N, &local_A_gpu, &B_gpu, &local_C_gpu);
      }
      
      cudaDeviceSynchronize();

      // Distribute data
      MPI_Scatter(full_A_gpu, local_N * N, MPI_UINT16_T, local_A_gpu, local_N * N, MPI_UINT16_T, 0, MPI_COMM_WORLD);
      MPI_Scatter(full_C_gpu, local_N * N, MPI_UINT16_T, local_C_gpu, local_N * N, MPI_UINT16_T, 0, MPI_COMM_WORLD);
      MPI_Bcast(B_gpu, local_N * N, MPI_UINT16_T, 0, MPI_COMM_WORLD);

      cudaDeviceSynchronize();
      // Perform computation
      calc_gemm(mpi_rank, repeats, local_N, N, N, alpha, beta, local_A_gpu, B_gpu, local_C_gpu);

      cudaDeviceSynchronize();
      
      // Gather results on rank 0
      if (mpi_rank == 0) {
        MPI_Gather(local_C_gpu, local_N * N, MPI_UINT16_T, full_C_host, local_N * N, MPI_UINT16_T, 0, MPI_COMM_WORLD);
      } else {
        MPI_Gather(local_C_gpu, local_N * N, MPI_UINT16_T, nullptr, 0, MPI_UINT16_T, 0, MPI_COMM_WORLD);
      }
      cudaDeviceSynchronize();

      // Cleanup
      cudaFree(local_A_gpu);
      cudaFree(B_gpu);
      cudaFree(local_C_gpu);
      if (mpi_rank == 0) {
        cudaFreeHost(full_A_host);
        cudaFreeHost(B_host);
        cudaFreeHost(full_C_host);
        cudaFree(full_A_gpu);
        cudaFree(full_C_gpu);
      }
      sizeof_gemm_t = sizeof(__half);
      break;
    }
    default:
      if (mpi_rank == 0) printf("Invalid precision\n");
      MPI_Finalize();
      return 1;
  }

  // Global ending time
  double global_end_time = get_milliseconds();
  double overall_time = global_end_time - global_start_time;

  // Output results on rank 0
  if (mpi_rank == 0) {
    printf("===============================================================\n");
    double N_dbl = (double)N;
    double matrix_memory = (3 * N_dbl * N_dbl) * ((double)sizeof_gemm_t);
    printf("Memory for Matrices:  %f MB\n", (matrix_memory / (1000 * 1000)));
    printf("Overll runtime time: %f milliseconds\n", overall_time);
    printf("===============================================================\n");
  }

  MPI_Finalize();
  return 0;
}