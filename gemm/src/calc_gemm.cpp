
#if (PRECISION == 'S')
#define gemm_t float
#define gemm_f cublasSgemm
#define cumode CUBLAS_TF32_TENSOR_OP_MATH
#elif (PRECISION == 'D')
#define gemm_t double
#define gemm_f cublasDgemm
#define cumode CUBLAS_DEFAULT_MATH
#elif (PRECISION == 'H')
#define gemm_t __half
#define gemm_f cublasHgemm
#define cumode CUBLAS_DEFAULT_MATH
#else
#error "Must define PRECISION before including calc_gemm.cpp"
#endif

static inline void
alloc_gemm(int N, gemm_t **pmatrixA, gemm_t **pmatrixB, gemm_t **pmatrixC) {
  printf("Allocating Matrices...\n");
  gemm_t *__restrict__ matrixA = (gemm_t *)malloc(sizeof(gemm_t) * N * N);
  gemm_t *__restrict__ matrixB = (gemm_t *)malloc(sizeof(gemm_t) * N * N);
  gemm_t *__restrict__ matrixC = (gemm_t *)malloc(sizeof(gemm_t) * N * N);

  printf("Allocation complete, populating with values...\n");
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      matrixA[i * N + j] = drand48() - 0.5;
      matrixB[i * N + j] = drand48() - 0.5;
      matrixC[i * N + j] = drand48() - 0.5;
    }
  }

  *pmatrixA = matrixA;
  *pmatrixB = matrixB;
  *pmatrixC = matrixC;
  return;
}

static inline long long int
check_gemm(int N, gemm_t *matrixA, gemm_t *matrixB, gemm_t *matrixC) {
  printf("Calculating matrix check...\n");

  // double final_sum = 0;
  long long int count = 0;
#pragma omp parallel for reduction(+ : final_sum, count)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      // final_sum += matrixC[i*N + j];
      // printf("matrixC[i*N + j] %f \n",__half2float(matrixC[i*N + j]));
      count++;
    }
  }

  return 0;
}

static inline void
free_gemm(gemm_t *matrixA, gemm_t *matrixB, gemm_t *matrixC) {
  free(matrixA);
  free(matrixB);
  free(matrixC);
  return;
}

static inline double
calc_gemm(int repeats, int N, double dalpha, double dbeta,
          gemm_t *matrixA, gemm_t *matrixB, gemm_t *matrixC) {

  printf("Performing multiplication...\n");

  gemm_t alpha = dalpha;
  gemm_t beta = dbeta;

  cudaError_t errorA, errorB, errorC;
  gemm_t *d_matrixA, *d_matrixB, *d_matrixC;
  errorA = cudaMalloc((void **)&d_matrixA, N * N * sizeof(gemm_t));
  errorB = cudaMalloc((void **)&d_matrixB, N * N * sizeof(gemm_t));
  errorC = cudaMalloc((void **)&d_matrixC, N * N * sizeof(gemm_t));
  if (errorA != cudaSuccess || errorB != cudaSuccess || errorC != cudaSuccess) {
    printf("ERROR: allocating device matrices\n");
    exit(1);
  }

  cublasStatus_t status;
  cublasHandle_t handle;
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("ERROR: creating a device handle\n");
    exit(1);
  }
  status = cublasSetMathMode(handle, cumode);

  cublasStatus_t statusA, statusB, statusC;
  statusA = cublasSetMatrix(N, N, sizeof(gemm_t), matrixA, N, d_matrixA, N);
  statusB = cublasSetMatrix(N, N, sizeof(gemm_t), matrixB, N, d_matrixB, N);
  statusC = cublasSetMatrix(N, N, sizeof(gemm_t), matrixC, N, d_matrixC, N);
  if (statusA != CUBLAS_STATUS_SUCCESS || statusB != CUBLAS_STATUS_SUCCESS || statusC != CUBLAS_STATUS_SUCCESS) {
    printf("ERROR: intializing device matrices\n");
    exit(1);
  }

  // Repeat multiple times
  const double start = get_seconds();
  for (int r = 0; r < repeats; r++) {
    cublasStatus_t matmulStatus;
    matmulStatus = gemm_f(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                          &alpha, d_matrixA, N, d_matrixB, N, &beta, d_matrixC, N);
    if (matmulStatus != CUBLAS_STATUS_SUCCESS) {
      printf("MatMul Function failed with error %d\n", matmulStatus);
      exit(1);
    }
  }
  cudaDeviceSynchronize();
  const double end = get_seconds();

  cublasGetMatrix(N, N, sizeof(gemm_t), d_matrixC, N, matrixC, N);
  cudaFree(d_matrixA);
  cudaFree(d_matrixB);
  cudaFree(d_matrixC);
  cublasDestroy(handle);

  return (end - start);
}

#undef gemm_t
#undef gemm_f
#undef cumode
