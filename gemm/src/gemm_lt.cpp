#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cublasLt.h>
#include <cuda_fp16.h>

#ifdef USE_MPI
#include <mpi.h>
#define mprintf      \
  if (mpi_rank == 0) \
  printf
#else
#define mprintf printf
#endif
int mpi_thread = 0;
int mpi_rank = 0;
int mpi_size = 1;

// ------------------------------------------------------- //
// Function: get_seconds
// ------------------------------------------------------- //
double get_seconds()
{

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

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

int main(int argc, char *argv[])
{

#ifdef USE_MPI
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif

  int N = 4096;
  int repeats = 100;
  double alpha = 1.0;
  double beta = 1.0;
  char prec = 'D';

  if (true)
  { // argv parse

    // argv[1] is the matrix size
    if (argc > 1)
    {
      N = atoi(argv[1]);
      mprintf("Matrix size input by command line: %d\n", N);
    }
    else
    {
      mprintf("Matrix size defaulted to %d\n", N);
    }

    // argv[2] is the number of trials
    if (argc > 2)
    {
      repeats = atoi(argv[2]);
      mprintf("Repeat multiply %d times.\n", repeats);
    }
    else
    {
      mprintf("Repeat multiply defaulted to %d\n", repeats);
    }

    // argv[3] is alpha
    if (argc > 3)
    {
      alpha = atof(argv[3]);
    }
    mprintf("Alpha =    %f\n", alpha);

    // argv[4] is beta
    if (argc > 4)
    {
      beta = atof(argv[4]);
    }
    mprintf("Beta  =    %f\n", beta);

    // argv[5] is precision
    if (argc > 5)
    {
      if (argv[5][0] == 'S')
        prec = 'S';
      else if (argv[5][0] == 'D')
        prec = 'D';
      else if (argv[5][0] == 'H')
        prec = 'H';
      else if (argv[5][0] == 'I')
        prec = 'I';
      else
      {
        mprintf("Reqested precision (%s) not recognized. "
                "Using default (%c).\n",
                argv[5], prec);
      }
    }
    else
    {
      mprintf("Precision defaulted to %c\n", prec);
    }
    mprintf("Precision =    %c\n", prec);
  }

  double time_taken;
  int sizeof_gemm_t;
  int status;
  switch (prec)
  {
  case ('S'):
  {
    float *matrixA, *matrixB, *matrixC, *matrixD;
    alloc_gemm(N, &matrixA, &matrixB, &matrixC, &matrixD);
    time_taken = calc_gemm(repeats, N, alpha, beta, matrixA, matrixB, matrixC, matrixD);
    status = check_gemm(N, matrixA, matrixB, matrixC, matrixD);
    free_gemm(matrixA, matrixB, matrixC, matrixD);
    sizeof_gemm_t = sizeof(float);
    break;
  }
  case ('D'):
  {
    double *matrixA, *matrixB, *matrixC, *matrixD;
    alloc_gemm(N, &matrixA, &matrixB, &matrixC, &matrixD);
    time_taken = calc_gemm(repeats, N, alpha, beta, matrixA, matrixB, matrixC, matrixD);
    status = check_gemm(N, matrixA, matrixB, matrixC, matrixD);
    free_gemm(matrixA, matrixB, matrixC, matrixD);
    sizeof_gemm_t = sizeof(double);
    break;
  }
  case ('H'):
  {
    __half *matrixA, *matrixB, *matrixC, *matrixD;
    alloc_gemm(N, &matrixA, &matrixB, &matrixC, &matrixD);
    time_taken = calc_gemm(repeats, N, alpha, beta, matrixA, matrixB, matrixC, matrixD);
    status = check_gemm(N, matrixA, matrixB, matrixC, matrixD);
    free_gemm(matrixA, matrixB, matrixC, matrixD);
    sizeof_gemm_t = sizeof(__half);
    break;
  }
  case ('I'):
  {
    int8_t *matrixA; 
    int8_t *matrixB; 
    int32_t *matrixC; 
    alloc_gemm_int(N, &matrixA, &matrixB, &matrixC);
    time_taken = calc_gemm_int(repeats, N, alpha, beta, matrixA, matrixB, matrixC);
    // time_taken = calc_gemm_int(repeats, N, alpha, beta, matrixA, matrixB, matrixC, matrixD);
    // status = check_gemm_int(N, matrixA, matrixB, matrixC, matrixD);
    free_gemm_int(matrixA, matrixB, matrixC);
    sizeof_gemm_t = sizeof(int32_t);
    break;
  }
  }

  if (true)
  { // Print results

    mprintf("\n");
    mprintf("===============================================================\n");

    double N_dbl = (double)N;
    double matrix_memory = (3 * N_dbl * N_dbl) * ((double)sizeof_gemm_t);

    mprintf("Memory for Matrices:  %f MB\n", (matrix_memory / (1024 * 1024)));

    mprintf("Multiply time:        %f seconds\n", time_taken);

    const double ops_computed =
          ((N_dbl * N_dbl * N_dbl * 2.0) + (N_dbl * N_dbl * 3.0)) * (double)(repeats) * (double)(mpi_size);

    if (prec == 'I')
    {
      // mprintf("FLOPs computed:       %f\n", flops_computed);
      mprintf("TOP/s rate:         %f TOP/s\n",
              (ops_computed / time_taken) / 1.0e12);
    }
    else
    {
      // mprintf("FLOPs computed:       %f\n", flops_computed);
      mprintf("GFLOP/s rate:         %f GF/s\n",
              (ops_computed / time_taken) / 1.0e9);
    }

    mprintf("===============================================================\n");
    mprintf("\n");
  }

#ifdef USE_MPI
  MPI_Finalize();
#endif

  return 0;
}

