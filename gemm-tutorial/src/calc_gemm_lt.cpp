//////////////////////////////////
/// FOR INTEGER
//////////////////////////////////
#if (PRECISION == 'I')

static inline void
alloc_gemm_int(int N,
               int8_t **pmatrixA,
               int8_t **pmatrixB,
               int8_t **pmatrixC,
               int32_t **pmatrixD)
{

  mprintf("Allocating Matrices...\n");
  int8_t *__restrict__ matrixA = (int8_t *)malloc(sizeof(int8_t) * N * N);
  int8_t *__restrict__ matrixB = (int8_t *)malloc(sizeof(int8_t) * N * N);
  int8_t *__restrict__ matrixC = (int8_t *)malloc(sizeof(int8_t) * N * N);
  int32_t *__restrict__ matrixD = (int32_t *)malloc(sizeof(int32_t) * N * N);

  mprintf("Allocation complete, populating with values...\n");
#pragma omp parallel for
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      // Populate with random integers
      matrixA[i * N + j] = rand();
      matrixB[i * N + j] = rand();
      matrixC[i * N + j] = rand();
      matrixD[i * N + j] = rand();
    }
  }

  *pmatrixA = matrixA;
  *pmatrixB = matrixB;
  *pmatrixC = matrixC;
  *pmatrixD = matrixD;
  return;
}

static inline long long int
check_gemm_int(int N,
               int8_t *matrixA,
               int8_t *matrixB,
               int8_t *matrixC,
               int32_t *matrixD)
{
  mprintf("Calculating matrix check...\n");

  // double final_sum = 0;
  long long int count = 0;
#pragma omp parallel for reduction(+ : final_sum, count)
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      // final_sum += matrixC[i*N + j];
      // printf("matrixC[i*N + j] %f \n",__half2float(matrixC[i*N + j]));
      count++;
    }
  }

  return 0;
}

static inline void
free_gemm_int(int8_t *matrixA,
              int8_t *matrixB,
              int8_t *matrixC,
              int32_t *matrixD)
{
  free(matrixA);
  free(matrixB);
  free(matrixC);
  free(matrixD);
  return;
}

static inline double
calc_gemm_int(int repeats, int N, double dalpha, double dbeta,
              int8_t *matrixA, int8_t *matrixB, int8_t *matrixC, int32_t *matrixD)
{
  mprintf("Performing multiplication...\n");

  int32_t alpha = dalpha;
  int32_t beta = dbeta;

  cudaError_t errorA, errorB, errorC, errorD;
  int8_t *d_matrixA, *d_matrixB, *d_matrixC;
  int32_t *d_matrixD;
  errorA = cudaMalloc((void **)&d_matrixA, N * N * sizeof(int8_t));
  errorB = cudaMalloc((void **)&d_matrixB, N * N * sizeof(int8_t));
  errorC = cudaMalloc((void **)&d_matrixC, N * N * sizeof(int8_t));
  errorD = cudaMalloc((void **)&d_matrixD, N * N * sizeof(int32_t));
  if ((errorA != cudaSuccess) || (errorB != cudaSuccess) || (errorC != cudaSuccess) || (errorD != cudaSuccess))
  {
    printf("ERROR: allocating device matrices\n");
    exit(1);
  }

  // Create cuBLASLt handle
  cublasStatus_t status;
  cublasLtHandle_t ltHandle;
  status = cublasLtCreate(&ltHandle);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    printf("ERROR: creating a device handle\n");
    exit(1);
  }

  // Define CUDA data types and compute types based on precision
  cudaDataType_t dataTypeAB = CUDA_R_8I;
  cudaDataType_t dataTypeCD = CUDA_R_32I;
  cudaDataType_t ScaleType = CUDA_R_32I;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;

  // Create matrix layouts
  cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
  cublasLtMatrixLayoutCreate(&Adesc, dataTypeAB, N, N, N);
  cublasLtMatrixLayoutCreate(&Bdesc, dataTypeAB, N, N, N);
  cublasLtMatrixLayoutCreate(&Cdesc, dataTypeCD, N, N, N);
  cublasLtMatrixLayoutCreate(&Ddesc, dataTypeCD, N, N, N);

  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatmulDescCreate(&matmulDesc, computeType, ScaleType);

  cublasStatus_t statusA, statusB, statusC, statusD;
  statusA = cublasSetMatrix(N, N, sizeof(int8_t), matrixA, N, d_matrixA, N);
  statusB = cublasSetMatrix(N, N, sizeof(int8_t), matrixB, N, d_matrixB, N);
  statusC = cublasSetMatrix(N, N, sizeof(int8_t), matrixC, N, d_matrixC, N);
  statusD = cublasSetMatrix(N, N, sizeof(int32_t), matrixD, N, d_matrixD, N);
  if ((statusA != CUBLAS_STATUS_SUCCESS) || (statusB != CUBLAS_STATUS_SUCCESS) || (statusC != CUBLAS_STATUS_SUCCESS) || (statusD != CUBLAS_STATUS_SUCCESS))
  {
    printf("ERROR: intializing device matrices\n");
    exit(1);
  }

  // Repeat multiple times
  const double start = get_seconds();
  for (int r = 0; r < repeats; r++)
  {
    cublasStatus_t matmulStatus;
    matmulStatus = cublasLtMatmul(ltHandle,
                                  matmulDesc,
                                  &alpha,
                                  d_matrixA,
                                  Adesc,
                                  d_matrixB,
                                  Bdesc,
                                  &beta,
                                  d_matrixC,
                                  Cdesc,
                                  d_matrixD,
                                  Ddesc,
                                  nullptr, // algo
                                  nullptr, // workspace
                                  0,       // workspace size
                                  0);      // stream

    if (matmulStatus != CUBLAS_STATUS_SUCCESS)
    {
      printf("cublasLtMatmul failed with error %d\n", matmulStatus);
      exit(1);
    }
  }
  cudaDeviceSynchronize();
  const double end = get_seconds();

  cublasGetMatrix(N, N, sizeof(int32_t), d_matrixD, N, matrixD, N);
  cudaFree(d_matrixA);
  cudaFree(d_matrixB);
  cudaFree(d_matrixC);
  cudaFree(d_matrixD);
  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Cdesc);
  cublasLtMatrixLayoutDestroy(Ddesc);
  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtDestroy(ltHandle);

  return (end - start);
}

#else
///////////////////////
/// FOR FLOAT
///////////////////////

#if (PRECISION == 'S')
#define gemm_t float
#elif (PRECISION == 'D')
#define gemm_t double
#elif (PRECISION == 'H')
#define gemm_t __half
#else
#error "Must define PRECISION before including calc_gemm_lt.cpp"
#endif

static inline void
alloc_gemm(int N,
           gemm_t **pmatrixA,
           gemm_t **pmatrixB,
           gemm_t **pmatrixC,
           gemm_t **pmatrixD)
{

  mprintf("Allocating Matrices...\n");
  gemm_t *__restrict__ matrixA = (gemm_t *)malloc(sizeof(gemm_t) * N * N);
  gemm_t *__restrict__ matrixB = (gemm_t *)malloc(sizeof(gemm_t) * N * N);
  gemm_t *__restrict__ matrixC = (gemm_t *)malloc(sizeof(gemm_t) * N * N);
  gemm_t *__restrict__ matrixD = (gemm_t *)malloc(sizeof(gemm_t) * N * N);

  mprintf("Allocation complete, populating with values...\n");
#pragma omp parallel for
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      matrixA[i * N + j] = drand48() - 0.5;
      matrixB[i * N + j] = drand48() - 0.5;
      matrixC[i * N + j] = drand48() - 0.5;
      matrixD[i * N + j] = drand48() - 0.5;
    }
  }

  *pmatrixA = matrixA;
  *pmatrixB = matrixB;
  *pmatrixC = matrixC;
  *pmatrixD = matrixD;
  return;
}

static inline long long int
check_gemm(int N,
           gemm_t *matrixA,
           gemm_t *matrixB,
           gemm_t *matrixC,
           gemm_t *matrixD)
{
  mprintf("Calculating matrix check...\n");

  // double final_sum = 0;
  long long int count = 0;
#pragma omp parallel for reduction(+ : final_sum, count)
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      // final_sum += matrixC[i*N + j];
      // printf("matrixC[i*N + j] %f \n",__half2float(matrixC[i*N + j]));
      count++;
    }
  }

  return 0;
}

static inline void
free_gemm(gemm_t *matrixA,
          gemm_t *matrixB,
          gemm_t *matrixC,
          gemm_t *matrixD)
{
  free(matrixA);
  free(matrixB);
  free(matrixC);
  free(matrixD);
  return;
}

static inline double
calc_gemm(int repeats, int N, double dalpha, double dbeta,
          gemm_t *matrixA, gemm_t *matrixB, gemm_t *matrixC, gemm_t *matrixD)
{
  mprintf("Performing multiplication...\n");

  gemm_t alpha = dalpha;
  gemm_t beta = dbeta;

  cudaError_t errorA, errorB, errorC, errorD;
  gemm_t *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD;
  errorA = cudaMalloc((void **)&d_matrixA, N * N * sizeof(gemm_t));
  errorB = cudaMalloc((void **)&d_matrixB, N * N * sizeof(gemm_t));
  errorC = cudaMalloc((void **)&d_matrixC, N * N * sizeof(gemm_t));
  errorD = cudaMalloc((void **)&d_matrixD, N * N * sizeof(gemm_t));
  if ((errorA != cudaSuccess) || (errorB != cudaSuccess) || (errorC != cudaSuccess) || (errorD != cudaSuccess))
  {
    printf("ERROR: allocating device matrices\n");
    exit(1);
  }

  // Create cuBLASLt handle
  cublasStatus_t status;
  cublasLtHandle_t ltHandle;
  status = cublasLtCreate(&ltHandle);
  if (status != CUBLAS_STATUS_SUCCESS)
  {
    printf("ERROR: creating a device handle\n");
    exit(1);
  }

// Define CUDA data types and compute types based on precision
#if (PRECISION == 'S')
  cudaDataType_t dataType = CUDA_R_32F;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
#elif (PRECISION == 'D')
  cudaDataType_t dataType = CUDA_R_64F;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_64F;
#elif (PRECISION == 'H')
  cudaDataType_t dataType = CUDA_R_16F;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_16F;
#endif

  // Create matrix layouts
  cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
  cublasLtMatrixLayoutCreate(&Adesc, dataType, N, N, N);
  cublasLtMatrixLayoutCreate(&Bdesc, dataType, N, N, N);
  cublasLtMatrixLayoutCreate(&Cdesc, dataType, N, N, N);
  cublasLtMatrixLayoutCreate(&Ddesc, dataType, N, N, N);

  // Create matmul descriptor with double precision scaling
  cublasLtMatmulDesc_t matmulDesc;
  cublasLtMatmulDescCreate(&matmulDesc, computeType, dataType);

  cublasStatus_t statusA, statusB, statusC, statusD;
  statusA = cublasSetMatrix(N, N, sizeof(gemm_t), matrixA, N, d_matrixA, N);
  statusB = cublasSetMatrix(N, N, sizeof(gemm_t), matrixB, N, d_matrixB, N);
  statusC = cublasSetMatrix(N, N, sizeof(gemm_t), matrixC, N, d_matrixC, N);
  statusD = cublasSetMatrix(N, N, sizeof(gemm_t), matrixD, N, d_matrixD, N);
  if ((statusA != CUBLAS_STATUS_SUCCESS) || (statusB != CUBLAS_STATUS_SUCCESS) || (statusC != CUBLAS_STATUS_SUCCESS) || (statusD != CUBLAS_STATUS_SUCCESS))
  {
    printf("ERROR: intializing device matrices\n");
    exit(1);
  }

  // Repeat multiple times
  const double start = get_seconds();
  for (int r = 0; r < repeats; r++)
  {
    cublasStatus_t matmulStatus;
    matmulStatus = cublasLtMatmul(ltHandle,
                                  matmulDesc,
                                  &alpha,
                                  d_matrixA,
                                  Adesc,
                                  d_matrixB,
                                  Bdesc,
                                  &beta,
                                  d_matrixC,
                                  Cdesc,
                                  d_matrixD,
                                  Ddesc,
                                  nullptr, // algo
                                  nullptr, // workspace
                                  0,       // workspace size
                                  0);      // stream

    if (matmulStatus != CUBLAS_STATUS_SUCCESS)
    {
      printf("cublasLtMatmul failed with error %d\n", matmulStatus);
      exit(1);
    }
  }
  cudaDeviceSynchronize();
  const double end = get_seconds();

  cublasGetMatrix(N, N, sizeof(gemm_t), d_matrixD, N, matrixD, N);
  cudaFree(d_matrixA);
  cudaFree(d_matrixB);
  cudaFree(d_matrixC);
  cudaFree(d_matrixD);
  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Cdesc);
  cublasLtMatrixLayoutDestroy(Ddesc);
  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtDestroy(ltHandle);

  return (end - start);
}
#undef gemm_t

#endif

