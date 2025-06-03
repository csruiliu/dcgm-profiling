#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <time.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cublasLt.h>
#include <cuda_fp16.h>

// Sleep time in milliseconds
#define SLEEP_TIME 40000

// Add this global variable to counter sleep calls
int sleep_occurrences = 0;

// Assuming max 20 sleep calls, adjust as needed
double sleep_timestamps[20]; 

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
#include "calc_gemm.cpp"
#undef PRECISION

#define PRECISION 'D'
#include "calc_gemm.cpp"
#undef PRECISION

#define PRECISION 'H'
#include "calc_gemm.cpp"
#undef PRECISION

int main(int argc, char *argv[]) {

  double global_start_time = get_milliseconds();

  int N = 4096;
  int repeats = 100;
  double alpha = 1.0;
  double beta = 1.0;
  char prec = 'D';

  // ------------------------------------------------------- //
  // Arguments Parsing
  // ------------------------------------------------------- //
  if (argc != 6) {
    printf("Usage: %s <N> <repeats> <alpha> <beta> <precision: S|D|H>\n", argv[0]);
    return 1;
  }

  N = atoi(argv[1]);
  repeats = atoi(argv[2]);
  alpha = atof(argv[3]);
  beta = atof(argv[4]);
  prec = argv[5][0];

  printf("Matrix size input by command line: %d\n", N);
  printf("Repeat multiply %d times.\n", repeats);
  printf("Alpha =    %f\n", alpha);
  printf("Beta  =    %f\n", beta);
  printf("Precision =    %c\n", prec);

  double alloc_time, device_alloc_time, host_to_device_time, gemm_time, device_to_host_time, free_time, device_free_time, overall_time;
  int sizeof_gemm_t;
  //int status;

  switch (prec) {
    case 'S': {
      float *matrixA, *matrixB, *matrixC;
      alloc_gemm(N, &matrixA, &matrixB, &matrixC, &alloc_time);
      calc_gemm(repeats, N, alpha, beta, matrixA, matrixB, matrixC,
                &device_alloc_time, &host_to_device_time, &gemm_time, &device_to_host_time, &device_free_time);
      //status = check_gemm(N, matrixA, matrixB, matrixC);
      free_gemm(matrixA, matrixB, matrixC, &free_time);
      sizeof_gemm_t = sizeof(float);
      break;
    }
    case 'D': {
      double *matrixA, *matrixB, *matrixC;
      alloc_gemm(N, &matrixA, &matrixB, &matrixC, &alloc_time);
      calc_gemm(repeats, N, alpha, beta, matrixA, matrixB, matrixC,
                &device_alloc_time, &host_to_device_time, &gemm_time, &device_to_host_time, &device_free_time);
      //status = check_gemm(N, matrixA, matrixB, matrixC);
      free_gemm(matrixA, matrixB, matrixC, &free_time);
      sizeof_gemm_t = sizeof(double);
      break;
    }
    case 'H': {
      __half *matrixA, *matrixB, *matrixC;
      alloc_gemm(N, &matrixA, &matrixB, &matrixC, &alloc_time);
      calc_gemm(repeats, N, alpha, beta, matrixA, matrixB, matrixC,
                &device_alloc_time, &host_to_device_time, &gemm_time, &device_to_host_time, &device_free_time);
      //status = check_gemm(N, matrixA, matrixB, matrixC);
      free_gemm(matrixA, matrixB, matrixC, &free_time);
      sizeof_gemm_t = sizeof(__half);
      break;
    }
  }

  printf("\n");
  printf("===============================================================\n");
  double N_dbl = (double)N;
  double matrix_memory = (3 * N_dbl * N_dbl) * ((double)sizeof_gemm_t);
  printf("Memory for Matrices:  %f MB\n", (matrix_memory / (1000 * 1000)));
  printf("Time for host memory allocation and initialization: %f milliseconds\n", alloc_time);
  printf("Time for device memory allocation: %f milliseconds\n", device_alloc_time);
  printf("Time for host to device data transfer: %f milliseconds\n", host_to_device_time);
  printf("Stage 1: %f seconds\n", (alloc_time + device_alloc_time + host_to_device_time) / 1000);
  printf("Time for GEMM operations: %f milliseconds\n", gemm_time);
  printf("Stage 2: %f seconds\n", gemm_time / 1000);
  printf("Time for device to host data transfer: %f milliseconds\n", device_to_host_time);
  printf("Time for free device memory: %f milliseconds\n", device_free_time);
  printf("Time for free host memory: %f milliseconds\n", free_time);
  printf("Stage 3: %f seconds\n", (device_to_host_time + device_free_time + free_time) / 1000);
  const double flops_computed = (N_dbl * N_dbl * N_dbl * 2.0 * (double)repeats) + (N_dbl * N_dbl * 3 * (double)repeats);
  printf("GFLOP/s rate:         %f GF/s\n", (flops_computed / (gemm_time / 1000.0)) / 1.0e9);  // Convert back to seconds for FLOPS calculation
  printf("===============================================================\n");

  // Sleep analysis section
  printf("===============================================================\n");
  printf("SLEEP ANALYSIS:\n");
  printf("Sleep function called %d times\n", sleep_occurrences);
  printf("Total sleep time: %f milliseconds\n", (double)(sleep_occurrences * SLEEP_TIME));
  printf("Sleep timestamps:\n");
  for (int i = 0; i < sleep_occurrences; i++) {
      printf("  Sleep #%d: %f milliseconds\n", i + 1, sleep_timestamps[i] - global_start_time);
  }
  printf("===============================================================\n");
  printf("\n");

  double global_end_time = get_milliseconds();
  overall_time = global_end_time - global_start_time;
  printf("===============================================================\n");
  printf("Overll runtime time: %f milliseconds\n", overall_time);
  printf("===============================================================\n");
  return 0;
}
