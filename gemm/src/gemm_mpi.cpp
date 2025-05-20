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
#define SLEEP_TIME 5000
// Number of GPUs in a node
#define NUM_NODE_GPUS 4

// ------------------------------------------------------- //
// Timer function to profile time
// ------------------------------------------------------- //
struct Timer {
#ifdef TIMING_CUDA_EVENTS
    cudaEvent_t start, stop;

    Timer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~Timer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void record_start() { cudaEventRecord(start); }
    void record_stop() { cudaEventRecord(stop); }
    double elapsed() {
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms / 1000.0;
    }
#else
    std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;

    void record_start() { start = std::chrono::high_resolution_clock::now(); }
    void record_stop() { stop = std::chrono::high_resolution_clock::now(); }
    double elapsed() {
        return std::chrono::duration<double>(stop - start).count();
    }
#endif
};

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

  Timer timer;
  int sizeof_gemm_t;

  double host_gpu_cpy_time;
  double mpi_matmul_time;
  double mpi_scatter_time;
  double mpi_gather_time;

  switch (prec) {
    case 'S': {
      // Full A, B, C matrics on host of rank 0 MPI process
      float *full_A_host = nullptr, *B_host = nullptr, *full_C_host = nullptr;
      // Full A, B, C matrics on gpu of rank 0 MPI process
      float *full_A_gpu, *B_gpu, *full_C_gpu;      
      // A, C matrics on GPU of each MPI process
      float *local_A_gpu, *local_C_gpu;
      
      // Rank 0 allocates and initializes full matrices on host using cudaMallocHost
      if (mpi_rank == 0) {
        alloc_gemm_host(N, &full_A_host, &B_host, &full_C_host);
        alloc_gemm_gpu(mpi_rank, N, N, &full_A_gpu, &B_gpu, &full_C_gpu);
        
        
        timer.record_start();
        // Copy from host to GPU        
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
        timer.record_stop();
        host_gpu_cpy_time = timer.elapsed();
        
        cudaDeviceSynchronize();
        std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));

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

      // Distribute data
      timer.record_start();
      MPI_Scatter(full_A_gpu, local_N * N, MPI_FLOAT, local_A_gpu, local_N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Scatter(full_C_gpu, local_N * N, MPI_FLOAT, local_C_gpu, local_N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
      MPI_Bcast(B_gpu, N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
      timer.record_stop();
      mpi_scatter_time = timer.elapsed();
      
      cudaDeviceSynchronize();
      std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
      // Perform computation
      mpi_matmul_time = calc_gemm(mpi_rank, repeats, local_N, N, N, alpha, beta, local_A_gpu, B_gpu, local_C_gpu);
      
      cudaDeviceSynchronize();
      std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));      

      // Gather results on rank 0
      timer.record_start();
      if (mpi_rank == 0) {
        MPI_Gather(local_C_gpu, local_N * N, MPI_FLOAT, full_C_host, local_N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        check_gemm(N, full_A_host, B_host, full_C_host);
      } else {
        MPI_Gather(local_C_gpu, local_N * N, MPI_FLOAT, nullptr, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
      }
      timer.record_stop();
      mpi_gather_time = timer.elapsed();

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
        
        timer.record_start();
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
        timer.record_stop();
        host_gpu_cpy_time = timer.elapsed();

        cudaDeviceSynchronize();
        std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
        
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
            
      // Distribute data
      timer.record_start();
      MPI_Scatter(full_A_gpu, local_N * N, MPI_DOUBLE, local_A_gpu, local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Scatter(full_C_gpu, local_N * N, MPI_DOUBLE, local_C_gpu, local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(B_gpu, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      timer.record_stop();
      mpi_scatter_time = timer.elapsed();

      cudaDeviceSynchronize();
      std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
      // Perform computation
      mpi_matmul_time = calc_gemm(mpi_rank, repeats, local_N, N, N, alpha, beta, local_A_gpu, B_gpu, local_C_gpu);

      cudaDeviceSynchronize();
      std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));

      // Gather results on rank 0
      timer.record_start();
      if (mpi_rank == 0) {
        MPI_Gather(local_C_gpu, local_N * N, MPI_DOUBLE, full_C_host, local_N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        check_gemm(N, full_A_host, B_host, full_C_host);
      } else {
        MPI_Gather(local_C_gpu, local_N * N, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      }
      timer.record_stop();
      mpi_gather_time = timer.elapsed();

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
        
        timer.record_start();
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
        timer.record_stop();
        host_gpu_cpy_time = timer.elapsed();

        cudaDeviceSynchronize();
        std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));

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
      
      // Distribute data
      timer.record_start();
      MPI_Scatter(full_A_gpu, local_N * N, MPI_UINT16_T, local_A_gpu, local_N * N, MPI_UINT16_T, 0, MPI_COMM_WORLD);
      MPI_Scatter(full_C_gpu, local_N * N, MPI_UINT16_T, local_C_gpu, local_N * N, MPI_UINT16_T, 0, MPI_COMM_WORLD);
      MPI_Bcast(B_gpu, local_N * N, MPI_UINT16_T, 0, MPI_COMM_WORLD);
      /*  
      # Use MPI_BYTE as transfer unit for case 'H'
      int byte_count = local_N * N * sizeof(__half);
      MPI_Scatter(full_A_gpu, byte_count, MPI_BYTE, local_A_gpu, byte_count, MPI_BYTE, 0, MPI_COMM_WORLD);
      MPI_Scatter(full_C_gpu, byte_count, MPI_BYTE, local_C_gpu, byte_count, MPI_BYTE, 0, MPI_COMM_WORLD);
      MPI_Bcast(B_gpu, N * N * sizeof(__half), MPI_BYTE, 0, MPI_COMM_WORLD);
      */
      timer.record_stop();
      mpi_scatter_time = timer.elapsed();

      cudaDeviceSynchronize();
      std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
      // Perform computation
      mpi_matmul_time = calc_gemm(mpi_rank, repeats, local_N, N, N, alpha, beta, local_A_gpu, B_gpu, local_C_gpu);

      cudaDeviceSynchronize();
      std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
      
      // Gather results on rank 0
      timer.record_start();
      if (mpi_rank == 0) {
        MPI_Gather(local_C_gpu, local_N * N, MPI_UINT16_T, full_C_host, local_N * N, MPI_UINT16_T, 0, MPI_COMM_WORLD);
        check_gemm(N, full_A_host, B_host, full_C_host);
      } else {
        MPI_Gather(local_C_gpu, local_N * N, MPI_UINT16_T, nullptr, 0, MPI_UINT16_T, 0, MPI_COMM_WORLD);
      }
      /* 
      # Use MPI_BYTE as transfer unit for case 'H'
      if (mpi_rank == 0) {
        MPI_Gather(local_C_gpu, byte_count, MPI_BYTE, full_C_host, byte_count, MPI_BYTE, 0, MPI_COMM_WORLD);
        check_gemm(N, full_A_host, B_host, full_C_host);
      } else {
        MPI_Gather(local_C_gpu, byte_count, MPI_BYTE, nullptr, 0, MPI_BYTE, 0, MPI_COMM_WORLD);
      }
      */
      timer.record_stop();
      mpi_gather_time = timer.elapsed();

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

  // Compute maximum time across all ranks
  double max_time_matmul = 0.0;
  double max_time_data_scatter = 0.0;
  double max_time_data_gather = 0.0;
  MPI_Reduce(&mpi_matmul_time, &max_time_matmul, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&mpi_scatter_time, &max_time_data_scatter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&mpi_gather_time, &max_time_data_gather, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // Output results on rank 0
  if (mpi_rank == 0) {
    std::cout << "===============================================================" << std::endl;
    double N_dbl = (double)N;
    double matrix_memory = (3 * N_dbl * N_dbl) * ((double)sizeof_gemm_t);
    std::cout << "Memory for Matrices:" << matrix_memory / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Time of Copying data from Host to one GPU: " << host_gpu_cpy_time << " seconds" << std::endl;
    std::cout << "Multiply time: " << max_time_matmul << " seconds" << std::endl;
    std::cout << "NVLink data scatter time: " << max_time_data_scatter << " seconds" << std::endl;
    std::cout << "NVLink data gather time: " << max_time_data_gather << " seconds" << std::endl;
    const double flops_computed = (N_dbl * N_dbl * N_dbl * 2.0 * (double)repeats) + (N_dbl * N_dbl * 3 * (double)repeats);
    std::cout << "GFLOP/s rate:" << (flops_computed / max_time_matmul) / 1.0e9 << " GF/s" << std::endl;
    std::cout << "===============================================================" << std::endl;
  }

  MPI_Finalize();
  return 0;
}