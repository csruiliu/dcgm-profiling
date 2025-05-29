#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <sys/time.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cublasLt.h>
#include <cuda_fp16.h>

// Sleep time in milliseconds
#define SLEEP_TIME 5000

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
#include "calc_gemm.cpp"
#undef PRECISION

#define PRECISION 'D'
#include "calc_gemm.cpp"
#undef PRECISION

#define PRECISION 'H'
#include "calc_gemm.cpp"
#undef PRECISION

int main(int argc, char *argv[]) {

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

  double alloc_time, device_alloc_time, host_to_device_time, gemm_time, device_to_host_time;
  int sizeof_gemm_t;
  int status;

  switch (prec) {
    case 'S': {
      float *matrixA, *matrixB, *matrixC;
      alloc_gemm(N, &matrixA, &matrixB, &matrixC, &alloc_time);
      // std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
      calc_gemm(repeats, N, alpha, beta, matrixA, matrixB, matrixC,
                &device_alloc_time, &host_to_device_time, &gemm_time, &device_to_host_time);
      status = check_gemm(N, matrixA, matrixB, matrixC);
      free_gemm(matrixA, matrixB, matrixC);
      sizeof_gemm_t = sizeof(float);
      break;
    }
    case 'D': {
      double *matrixA, *matrixB, *matrixC;
      alloc_gemm(N, &matrixA, &matrixB, &matrixC, &alloc_time);
      //std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
      calc_gemm(repeats, N, alpha, beta, matrixA, matrixB, matrixC,
                &device_alloc_time, &host_to_device_time, &gemm_time, &device_to_host_time);
      status = check_gemm(N, matrixA, matrixB, matrixC);
      free_gemm(matrixA, matrixB, matrixC);
      sizeof_gemm_t = sizeof(double);
      break;
    }
    case 'H': {
      __half *matrixA, *matrixB, *matrixC;
      alloc_gemm(N, &matrixA, &matrixB, &matrixC, &alloc_time);
      //std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_TIME));
      calc_gemm(repeats, N, alpha, beta, matrixA, matrixB, matrixC,
                &device_alloc_time, &host_to_device_time, &gemm_time, &device_to_host_time);
      status = check_gemm(N, matrixA, matrixB, matrixC);
      free_gemm(matrixA, matrixB, matrixC);
      sizeof_gemm_t = sizeof(__half);
      break;
    }
  }

  printf("\n");
  printf("===============================================================\n");
  double N_dbl = (double)N;
  double matrix_memory = (3 * N_dbl * N_dbl) * ((double)sizeof_gemm_t);
  printf("Memory for Matrices:  %f MB\n", (matrix_memory / (1024 * 1024)));
  printf("Time for host memory allocation and initialization: %f seconds\n", alloc_time);
  printf("Time for device memory allocation: %f seconds\n", device_alloc_time);
  printf("Time for host to device data transfer: %f seconds\n", host_to_device_time);
  printf("Time for GEMM operations: %f seconds\n", gemm_time);
  printf("Time for device to host data transfer: %f seconds\n", device_to_host_time);
  const double flops_computed = (N_dbl * N_dbl * N_dbl * 2.0 * (double)repeats) + (N_dbl * N_dbl * 3 * (double)repeats);
  printf("GFLOP/s rate:         %f GF/s\n", (flops_computed / gemm_time) / 1.0e9);
  printf("===============================================================\n");
  printf("\n");

  return 0;
}
