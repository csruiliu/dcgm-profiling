#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <sys/time.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cublasLt.h>
#include <cuda_fp16.h>

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

  double time_taken;
  int sizeof_gemm_t;
  int status;

  switch (prec) {
    case 'S': {
      float *matrixA, *matrixB, *matrixC;
      alloc_gemm(N, &matrixA, &matrixB, &matrixC);
      time_taken = calc_gemm(repeats, N, alpha, beta, matrixA, matrixB, matrixC);
      status = check_gemm(N, matrixA, matrixB, matrixC);
      free_gemm(matrixA, matrixB, matrixC);
      sizeof_gemm_t = sizeof(float);
      break;
    }
    case 'D': {
      double *matrixA, *matrixB, *matrixC;
      alloc_gemm(N, &matrixA, &matrixB, &matrixC);
      time_taken = calc_gemm(repeats, N, alpha, beta, matrixA, matrixB, matrixC);
      status = check_gemm(N, matrixA, matrixB, matrixC);
      free_gemm(matrixA, matrixB, matrixC);
      sizeof_gemm_t = sizeof(double);
      break;
    }
    case 'H': {
      __half *matrixA, *matrixB, *matrixC;
      alloc_gemm(N, &matrixA, &matrixB, &matrixC);
      time_taken = calc_gemm(repeats, N, alpha, beta, matrixA, matrixB, matrixC);
      status = check_gemm(N, matrixA, matrixB, matrixC);
      free_gemm(matrixA, matrixB, matrixC);
      sizeof_gemm_t = sizeof(__half);
      break;
    }
  }

  printf("\n");
  printf("===============================================================\n");

  double N_dbl = (double) N;
  double matrix_memory = (3 * N_dbl * N_dbl) * ((double)sizeof_gemm_t);

  printf("Memory for Matrices:  %f MB\n", (matrix_memory / (1024 * 1024)));

  printf("Multiply time:        %f seconds\n", time_taken);

  const double flops_computed = (N_dbl * N_dbl * N_dbl * 2.0 * (double)(repeats)) + (N_dbl * N_dbl * 3 * (double)(repeats));

  // mprintf("FLOPs computed:       %f\n", flops_computed);
  printf("GFLOP/s rate:         %f GF/s\n", (flops_computed / time_taken) / 1.0e9);

  printf("===============================================================\n");
  printf("\n");

  return 0;
}
