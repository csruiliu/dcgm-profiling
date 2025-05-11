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
  gemm_t *matrixA = (gemm_t *)malloc(sizeof(gemm_t) * N * N);
  gemm_t *matrixB = (gemm_t *)malloc(sizeof(gemm_t) * N * N);
  gemm_t *matrixC = (gemm_t *)malloc(sizeof(gemm_t) * N * N);

  printf("Allocation complete, populating with values...\n");
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      matrixA[i * N + j] = drand48() - 0.5; // Row-major storage
      matrixB[i * N + j] = drand48() - 0.5;
      matrixC[i * N + j] = drand48() - 0.5;
    }
  }

  *pmatrixA = matrixA;
  *pmatrixB = matrixB;
  *pmatrixC = matrixC;
}

static inline long long int
check_gemm(int N, gemm_t *matrixA, gemm_t *matrixB, gemm_t *matrixC) {
  printf("Calculating matrix check...\n");
  long long int count = 0;
#pragma omp parallel for reduction(+ : count)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
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
}

static inline double
calc_gemm(int repeats, int M, int N, int K, double dalpha, double dbeta,
          gemm_t *d_A, gemm_t *d_B, gemm_t *d_C) {
  gemm_t alpha = (gemm_t)dalpha;
  gemm_t beta = (gemm_t)dbeta;

  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("ERROR: creating cublas handle\n");
    exit(1);
  }
  status = cublasSetMathMode(handle, cumode);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("ERROR: setting math mode\n");
    exit(1);
  }

  // For row-major matrices, adjust gemm call with transpose operations
  // C_rowmajor (M x N) = alpha * A_rowmajor (M x K) * B_rowmajor (K x N) + beta * C_rowmajor
  // Compute C^T = alpha * B^T * A^T + beta * C^T in column-major
  const double start = get_seconds();
  for (int r = 0; r < repeats; r++) {
    cublasStatus_t matmulStatus = gemm_f(handle,
                                         CUBLAS_OP_T,   // op(B) = B^T
                                         CUBLAS_OP_T,   // op(A) = A^T
                                         N,             // m = N (rows of C^T)
                                         M,             // n = M (cols of C^T)
                                         K,             // k = K
                                         &alpha,
                                         d_B, N,        // B_rowmajor (K x N), ldB = N
                                         d_A, K,        // A_rowmajor (M x K), ldA = K
                                         &beta,
                                         d_C, N);       // C_rowmajor (M x N), ldC = N
    if (matmulStatus != CUBLAS_STATUS_SUCCESS) {
      printf("MatMul failed with error %d\n", matmulStatus);
      exit(1);
    }
  }
  cudaDeviceSynchronize();
  const double end = get_seconds();

  cublasDestroy(handle);
  return (end - start);
}

#undef gemm_t
#undef gemm_f
#undef cumode