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
alloc_gemm(int N, gemm_t **pmatrixA, gemm_t **pmatrixB, gemm_t **pmatrixC, double *alloc_time) {
  printf("Allocating Matrices...\n");
  double start = get_milliseconds();
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
  double end = get_milliseconds();
  *alloc_time = end - start;

  *pmatrixA = matrixA;
  *pmatrixB = matrixB;
  *pmatrixC = matrixC;
}

static inline long long int
check_gemm(int N, gemm_t *matrixA, gemm_t *matrixB, gemm_t *matrixC) {
  printf("Calculating matrix check...\n");

  long long int count = 0;
#pragma omp parallel for reduction(+ : final_sum, count)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      count++;
    }
  }

  return 0;
}

static inline void
free_gemm(gemm_t *matrixA, gemm_t *matrixB, gemm_t *matrixC, double *free_time) {
  double free_start = get_milliseconds();
  free(matrixA);
  free(matrixB);
  free(matrixC);
  double free_end = get_milliseconds();
  *free_time = free_end - free_start;
}

static inline void
calc_gemm(int repeats, int N, double dalpha, double dbeta,
          gemm_t *matrixA, gemm_t *matrixB, gemm_t *matrixC,
          double *device_alloc_time, double *host_to_device_time,
          double *gemm_time, double *device_to_host_time, double *device_free_time) {
  printf("Performing multiplication...\n");

  gemm_t alpha = dalpha;
  gemm_t beta = dbeta;
  gemm_t *d_matrixA, *d_matrixB, *d_matrixC;

  // Stage 2a: GPU memory allocation
  double alloc_start = get_milliseconds();
  cudaError_t errorA = cudaMalloc((void **)&d_matrixA, N * N * sizeof(gemm_t));
  cudaError_t errorB = cudaMalloc((void **)&d_matrixB, N * N * sizeof(gemm_t));
  cudaError_t errorC = cudaMalloc((void **)&d_matrixC, N * N * sizeof(gemm_t));
  double alloc_end = get_milliseconds();
  *device_alloc_time = alloc_end - alloc_start;

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

  // Stage 2b: Host to GPU transfer
  double h2d_start = get_milliseconds();
  cublasStatus_t statusA = cublasSetMatrix(N, N, sizeof(gemm_t), matrixA, N, d_matrixA, N);
  cublasStatus_t statusB = cublasSetMatrix(N, N, sizeof(gemm_t), matrixB, N, d_matrixB, N);
  cublasStatus_t statusC = cublasSetMatrix(N, N, sizeof(gemm_t), matrixC, N, d_matrixC, N);
  double h2d_end = get_milliseconds();
  *host_to_device_time = h2d_end - h2d_start;

  if (statusA != CUBLAS_STATUS_SUCCESS || statusB != CUBLAS_STATUS_SUCCESS || statusC != CUBLAS_STATUS_SUCCESS) {
    printf("ERROR: initializing device matrices\n");
    exit(1);
  }

  sleep_cpu_gpu_idle(SLEEP_TIME);
  
  // Stage 3: GEMM operations
  double gemm_start = get_milliseconds();
  for (int r = 0; r < repeats; r++) {
    cublasStatus_t matmulStatus = gemm_f(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                         &alpha, d_matrixA, N, d_matrixB, N, &beta, d_matrixC, N);
    if (matmulStatus != CUBLAS_STATUS_SUCCESS) {
      printf("MatMul Function failed with error %d\n", matmulStatus);
      exit(1);
    }
  }
  cudaDeviceSynchronize();
  double gemm_end = get_milliseconds();
  *gemm_time = gemm_end - gemm_start;

  sleep_cpu_gpu_idle(SLEEP_TIME);

  // Stage 4: GPU to Host transfer
  double d2h_start = get_milliseconds();
  cublasGetMatrix(N, N, sizeof(gemm_t), d_matrixC, N, matrixC, N);
  double d2h_end = get_milliseconds();
  *device_to_host_time = d2h_end - d2h_start;

  // Free allocated memory
  double device_free_start = get_milliseconds();
  cudaFree(d_matrixA);
  cudaFree(d_matrixB);
  cudaFree(d_matrixC);
  cublasDestroy(handle);
  cudaDeviceSynchronize();
  double device_free_end = get_milliseconds();
  *device_free_time = device_free_end - device_free_start;
}

#undef gemm_t
#undef gemm_f
#undef cumode
