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
alloc_gemm_host(int N, gemm_t **pA_host, gemm_t **pB_host, gemm_t **pC_host) {
  printf("Allocating Matrices on Host...\n");
  
  gemm_t *matrixA, *matrixB, *matrixC;
  cudaError_t err_alloc;

  err_alloc = cudaMallocHost(&matrixA, sizeof(gemm_t) * N * N);
  if (err_alloc != cudaSuccess) {
    printf("cudaMalloc Matrix A failed: %s\n", cudaGetErrorString(err_alloc));
    MPI_Finalize();
    exit(1);
  }
  err_alloc = cudaMallocHost(&matrixB, sizeof(gemm_t) * N * N);
  if (err_alloc != cudaSuccess) {
    printf("cudaMalloc Matrix B failed: %s\n", cudaGetErrorString(err_alloc));
    MPI_Finalize();
    exit(1);
  }
  err_alloc = cudaMallocHost(&matrixC, sizeof(gemm_t) * N * N);
  if (err_alloc != cudaSuccess) {
    printf("cudaMalloc Matrix C failed: %s\n", cudaGetErrorString(err_alloc));
    MPI_Finalize();
    exit(1);
  }
  
  printf("Allocation complete, populating with values...\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      matrixA[i * N + j] = drand48() - 0.5;
      matrixB[i * N + j] = drand48() - 0.5;
      matrixC[i * N + j] = drand48() - 0.5;
    }
  }
  
  *pA_host = matrixA;
  *pB_host = matrixB;
  *pC_host = matrixC;
  return;
}

static inline void
alloc_gemm_gpu(int rank, int M, int N, gemm_t **pA_gpu, gemm_t **pB_gpu, gemm_t **pC_gpu) {
  std::cout << "Rank " << rank << ": Allocating and initializing matrices on GPU..." << std::endl;
  gemm_t *matrixA, *matrixB, *matrixC;
  
  cudaError_t err;

  err = cudaMalloc(&matrixA, sizeof(gemm_t) * M * N);
  if (err != cudaSuccess) {
    printf("cudaMalloc A_gpu failed: %s\n", cudaGetErrorString(err));
    MPI_Finalize();
    exit(1);
  }
  err = cudaMalloc(&matrixB, sizeof(gemm_t) * N * N);
  if (err != cudaSuccess) {
    printf("cudaMalloc B_gpu failed: %s\n", cudaGetErrorString(err));
    MPI_Finalize();
    exit(1);
  }
  err = cudaMalloc(&matrixC, sizeof(gemm_t) * M * N);
  if (err != cudaSuccess) {
    printf("cudaMalloc C_gpu failed: %s\n", cudaGetErrorString(err));
    MPI_Finalize();
    exit(1);
  }

  *pA_gpu = matrixA;
  *pB_gpu = matrixB;
  *pC_gpu = matrixC;
  return;
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

static inline double
calc_gemm(int rank, int repeats, int M, int N, int K, double dalpha, double dbeta, gemm_t *d_A, gemm_t *d_B, gemm_t *d_C) {
  std::cout << "Rank " << rank << ": Performing multiplication..." << std::endl;
  
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