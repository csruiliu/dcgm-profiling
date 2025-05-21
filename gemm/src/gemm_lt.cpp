#include <stdio.h>
#include <stdlib.h>
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
#include "calc_gemm_lt.cpp"
#undef PRECISION

#define PRECISION 'D'
#include "calc_gemm_lt.cpp"
#undef PRECISION

#define PRECISION 'H'
#include "calc_gemm_lt.cpp"
#undef PRECISION

#define PRECISION 'I'
#include "calc_gemm_lt.cpp"
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
    printf("Usage: %s <N> <repeats> <alpha> <beta> <precision: S|D|H|I>\n", argv[0]);
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
    case ('S'): {
      float *matrixA, *matrixB, *matrixC, *matrixD;
      alloc_gemm(N, &matrixA, &matrixB, &matrixC, &matrixD);
      time_taken = calc_gemm(repeats, N, alpha, beta, matrixA, matrixB, matrixC, matrixD);
      status = check_gemm(N, matrixA, matrixB, matrixC, matrixD);
      free_gemm(matrixA, matrixB, matrixC, matrixD);
      sizeof_gemm_t = sizeof(float);
      break;
    }
    case ('D'): {
      double *matrixA, *matrixB, *matrixC, *matrixD;
      alloc_gemm(N, &matrixA, &matrixB, &matrixC, &matrixD);
      time_taken = calc_gemm(repeats, N, alpha, beta, matrixA, matrixB, matrixC, matrixD);
      status = check_gemm(N, matrixA, matrixB, matrixC, matrixD);
      free_gemm(matrixA, matrixB, matrixC, matrixD);
      sizeof_gemm_t = sizeof(double);
      break;
    }
    case ('H'): {
      __half *matrixA, *matrixB, *matrixC, *matrixD;
      alloc_gemm(N, &matrixA, &matrixB, &matrixC, &matrixD);
      time_taken = calc_gemm(repeats, N, alpha, beta, matrixA, matrixB, matrixC, matrixD);
      status = check_gemm(N, matrixA, matrixB, matrixC, matrixD);
      free_gemm(matrixA, matrixB, matrixC, matrixD);
      sizeof_gemm_t = sizeof(__half);
      break;
    }
    case ('I'): {
      int8_t *matrixA;
      int8_t *matrixB;
      int32_t *matrixC;
      alloc_gemm_int(N, &matrixA, &matrixB, &matrixC);
      // time_taken = calc_gemm_int(repeats, N, alpha, beta, matrixA, matrixB, matrixC);
      // status = check_gemm_int(N, matrixA, matrixB, matrixC);
      // free_gemm_int(matrixA, matrixB, matrixC);
      sizeof_gemm_t = sizeof(int32_t);
      break;
    }
  }


  // ------------------------------------------------------- //
  // Print results
  // ------------------------------------------------------- //
  printf("\n");
  printf("===============================================================\n");

  double N_dbl = (double)N;
  double matrix_memory = (3 * N_dbl * N_dbl) * ((double)sizeof_gemm_t);

  printf("Memory for Matrices:  %f MB\n", (matrix_memory / (1024 * 1024)));

  printf("Multiply time:        %f seconds\n", time_taken);

  const double ops_computed = ((N_dbl * N_dbl * N_dbl * 2.0) + (N_dbl * N_dbl * 3.0)) * (double)(repeats);

  if (prec == 'I') {
    // mprintf("FLOPs computed:       %f\n", flops_computed);
    printf("TOP/s rate:         %f TOP/s\n", (ops_computed / time_taken) / 1.0e12);
  } else {
    // mprintf("FLOPs computed:       %f\n", flops_computed);
    printf("GFLOP/s rate:         %f GF/s\n", (ops_computed / time_taken) / 1.0e9);
  }

  printf("===============================================================\n");
  printf("\n");
  
  return 0;
}
