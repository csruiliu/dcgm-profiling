//////////////////////////////////
/// FOR INTEGER
//////////////////////////////////
#if (PRECISION == 'I')

#define gemm_lt_a int8_t
#define gemm_lt_b int8_t
#define gemm_lt_c int32_t
#define gemm_lt_alpha int32_t
#define gemm_lt_beta int32_t
#define gemm_lt_mat int32_t

int roundoff(int v, int d) {
  return (v + d - 1) / d * d;
}

static inline void
alloc_gemm_int(int N, gemm_lt_a **pmatrixA, gemm_lt_b **pmatrixB, gemm_lt_c **pmatrixC) {
  // Calculate proper leading dimensions
  // int lda = 32 * N;
  // int ldb = 32 * roundoff(N, 32);
  // int ldc = 32 * N;

  printf("Allocating Matrices...\n");
  // Allocate with proper padded dimensions
  gemm_lt_a *__restrict__ matrixA = (gemm_lt_a *)malloc(sizeof(gemm_lt_a) * N * N);
  gemm_lt_b *__restrict__ matrixB = (gemm_lt_b *)malloc(sizeof(gemm_lt_b) * N * N);
  gemm_lt_c *__restrict__ matrixC = (gemm_lt_c *)malloc(sizeof(gemm_lt_c) * N * N);

  printf("Allocation complete, populating with values...\n");
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      matrixA[i * N + j] = rand();               
      matrixB[i * N + j] = rand();
      matrixC[i * N + j] = rand();
    }
  }

  *pmatrixA = matrixA;
  *pmatrixB = matrixB;
  *pmatrixC = matrixC;
  return;
}

static inline long long int
check_gemm_int(int N, gemm_lt_a *matrixA, gemm_lt_b *matrixB, gemm_lt_c *matrixC) {
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
free_gemm_int(gemm_lt_a *matrixA, gemm_lt_b *matrixB, gemm_lt_c *matrixC) {
  free(matrixA);
  free(matrixB);
  free(matrixC);
  return;
}


static inline double
calc_gemm_int(int repeats, int N, double dalpha, double dbeta,
              gemm_lt_a *matrixA, gemm_lt_b *matrixB, gemm_lt_c *matrixC) {
  printf("Performing multiplication...\n");

  gemm_lt_alpha alpha = dalpha;
  gemm_lt_beta beta = dbeta;
  gemm_lt_alpha transformAlpha = dalpha; 
  gemm_lt_beta transformBeta = dbeta;

  cublasLtMatmulDesc_t matmulDesc = NULL;
  cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
  cublasOperation_t opTranspose = CUBLAS_OP_T;

  // tensor op igemm kernels require specialized memory order of data
  cublasLtMatrixTransformDesc_t transformDesc = NULL;
  gemm_lt_a *Atransform = NULL;
  gemm_lt_b *Btransform = NULL;
  gemm_lt_c *Ctransform = NULL;
  
  cublasLtMatrixLayout_t AtransformDesc = NULL, BtransformDesc = NULL, CtransformDesc = NULL;
  
  cublasLtOrder_t order_COL32 = CUBLASLT_ORDER_COL32;
  cublasLtOrder_t order_B = CUBLASLT_ORDER_COL32_2R_4R4;
  
  int ldatransform = 32 * N;
  int ldbtransform = 32 * roundoff(N, 32);
  int ldctransform = 32 * N;

  // Create cuBLASLt handle
  cublasStatus_t status;
  cublasLtHandle_t ltHandle;
  status = cublasLtCreate(&ltHandle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("ERROR: creating a device handle\n");
    exit(1);
  }

  // Allocate device memory with proper padding
  gemm_lt_a *d_matrixA;
  gemm_lt_b *d_matrixB;
  gemm_lt_c *d_matrixC;

  cudaError_t errorA, errorB, errorC;
  errorA = cudaMalloc(reinterpret_cast<void**>(&d_matrixA), sizeof(gemm_lt_a) * N * N);
  errorB = cudaMalloc(reinterpret_cast<void**>(&d_matrixB), sizeof(gemm_lt_b) * N * N);
  errorC = cudaMalloc(reinterpret_cast<void**>(&d_matrixC), sizeof(gemm_lt_c) * N * N);
  if ((errorA != cudaSuccess) || (errorB != cudaSuccess) || (errorC != cudaSuccess)) {
    printf("ERROR: allocating device matrices\n");
    exit(1);
  }
  
  cublasStatus_t statusA, statusB, statusC;
  statusA = cublasSetMatrix(N, N, sizeof(gemm_lt_a), matrixA, N, d_matrixA, N);
  statusB = cublasSetMatrix(N, N, sizeof(gemm_lt_b), matrixB, N, d_matrixB, N);
  statusC = cublasSetMatrix(N, N, sizeof(gemm_lt_c), matrixC, N, d_matrixC, N);
  if (statusA != CUBLAS_STATUS_SUCCESS || statusB != CUBLAS_STATUS_SUCCESS || statusC != CUBLAS_STATUS_SUCCESS) {
    printf("ERROR: intializing device matrices\n");
    exit(1);
  }

  printf("cudaMalloc...\n");

  cudaMalloc(reinterpret_cast<void**>(&Atransform), sizeof(gemm_lt_a) * ldatransform * N);
  cudaMalloc(reinterpret_cast<void**>(&Btransform), sizeof(gemm_lt_b) * ldbtransform * N);
  cudaMalloc(reinterpret_cast<void**>(&Ctransform), sizeof(gemm_lt_c) * ldctransform * N);
  
  cublasLtMatrixTransformDescCreate(&transformDesc, CUDA_R_32I);

  cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I);
  // tensor op igemm kernels only support NT gemm
  cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opTranspose, sizeof(opTranspose));

  // create descriptors for original matrices
  cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, N, N, N);
  cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, N, N, N);
  cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, N, N, N);

  cublasLtMatrixLayoutCreate(&AtransformDesc, CUDA_R_8I, N, N, ldatransform);
  cublasLtMatrixLayoutSetAttribute(AtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));

  // data memory order is set to CUBLASLT_ORDER_COL4_4R2_8C in order to achieve best performance on Turing devices.
  // for best performance on Ampere, consider setting the memory order to CUBLASLT_ORDER_COL32_2R_4R4.
  cublasLtMatrixLayoutCreate(&BtransformDesc, CUDA_R_8I, N, N, ldbtransform);
  cublasLtMatrixLayoutSetAttribute(BtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_B, sizeof(order_B));

  cublasLtMatrixLayoutCreate(&CtransformDesc, CUDA_R_32I, N, N, ldctransform);
  cublasLtMatrixLayoutSetAttribute(CtransformDesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));

  // ---------------------------------------------------------------------------------------------
  // transforms and computation

  cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, d_matrixA, Adesc, &transformBeta, NULL, NULL, Atransform, AtransformDesc, 0);

  // B matrix is non-transposed, but transposed matrix is needed - add transpose operation in matrix transform.
  cublasLtMatrixTransformDescSetAttribute(transformDesc, CUBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &opTranspose, sizeof(opTranspose));

  cublasLtMatrixTransform(ltHandle, transformDesc, &transformAlpha, d_matrixB, Bdesc, &transformBeta, NULL, NULL, Btransform, BtransformDesc, 0);

  printf("Transformation ...\n");

  // Repeat multiple times
  const double start = get_seconds();
  for (int r = 0; r < repeats; r++) {
    cublasStatus_t matmulStatus;
    matmulStatus = cublasLtMatmul(ltHandle,
                                  matmulDesc,
                                  &alpha,
                                  Atransform,
                                  AtransformDesc,
                                  Btransform,
                                  BtransformDesc,
                                  &beta,
                                  Ctransform,
                                  CtransformDesc,
                                  Ctransform,
                                  CtransformDesc,
                                  nullptr, // algo
                                  nullptr, // workspace
                                  0,       // workspace size
                                  0);      // stream

    if (matmulStatus != CUBLAS_STATUS_SUCCESS) {
      printf("cublasLtMatmul failed with error %d\n", matmulStatus);
      exit(1);
    }
  }
  cudaDeviceSynchronize();
  const double end = get_seconds();

  // cublasGetMatrix(N, N, sizeof(gemm_lt_mat), d_matrixD, N, matrixD, N);
  
  cudaFree(d_matrixA);
  cudaFree(d_matrixB);
  cudaFree(d_matrixC);

  cudaFree(Atransform);
  cudaFree(Btransform);
  cudaFree(Ctransform);
  
  cublasLtMatrixLayoutDestroy(Adesc);
  cublasLtMatrixLayoutDestroy(Bdesc);
  cublasLtMatrixLayoutDestroy(Cdesc);
  
  cublasLtMatrixLayoutDestroy(AtransformDesc);
  cublasLtMatrixLayoutDestroy(BtransformDesc);
  cublasLtMatrixLayoutDestroy(CtransformDesc);

  cublasLtMatmulDescDestroy(matmulDesc);
  cublasLtMatrixTransformDescDestroy(transformDesc);

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
alloc_gemm(int N, gemm_t **pmatrixA, gemm_t **pmatrixB, gemm_t **pmatrixC, gemm_t **pmatrixD) {
  printf("Allocating Matrices...\n");
  gemm_t *__restrict__ matrixA = (gemm_t *)malloc(sizeof(gemm_t) * N * N);
  gemm_t *__restrict__ matrixB = (gemm_t *)malloc(sizeof(gemm_t) * N * N);
  gemm_t *__restrict__ matrixC = (gemm_t *)malloc(sizeof(gemm_t) * N * N);
  gemm_t *__restrict__ matrixD = (gemm_t *)malloc(sizeof(gemm_t) * N * N);

  printf("Allocation complete, populating with values...\n");
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
check_gemm(int N, gemm_t *matrixA, gemm_t *matrixB, gemm_t *matrixC, gemm_t *matrixD) {
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
free_gemm(gemm_t *matrixA, gemm_t *matrixB, gemm_t *matrixC, gemm_t *matrixD) {
  free(matrixA);
  free(matrixB);
  free(matrixC);
  free(matrixD);
  return;
}

static inline double
calc_gemm(int repeats, int N, double dalpha, double dbeta,
          gemm_t *matrixA, gemm_t *matrixB, gemm_t *matrixC, gemm_t *matrixD) {
  printf("Performing multiplication...\n");

  gemm_t alpha = dalpha;
  gemm_t beta = dbeta;

  cudaError_t errorA, errorB, errorC, errorD;
  gemm_t *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD;
  errorA = cudaMalloc((void **)&d_matrixA, N * N * sizeof(gemm_t));
  errorB = cudaMalloc((void **)&d_matrixB, N * N * sizeof(gemm_t));
  errorC = cudaMalloc((void **)&d_matrixC, N * N * sizeof(gemm_t));
  errorD = cudaMalloc((void **)&d_matrixD, N * N * sizeof(gemm_t));
  if (errorA != cudaSuccess || errorB != cudaSuccess || errorC != cudaSuccess || errorD != cudaSuccess) {
    printf("ERROR: allocating device matrices\n");
    exit(1);
  }

  // Create cuBLASLt handle
  cublasStatus_t status;
  cublasLtHandle_t ltHandle;
  status = cublasLtCreate(&ltHandle);
  if (status != CUBLAS_STATUS_SUCCESS) {
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
  if (statusA != CUBLAS_STATUS_SUCCESS || statusB != CUBLAS_STATUS_SUCCESS || statusC != CUBLAS_STATUS_SUCCESS || statusD != CUBLAS_STATUS_SUCCESS) {
    printf("ERROR: intializing device matrices\n");
    exit(1);
  }

  // Repeat multiple times
  const double start = get_seconds();
  for (int r = 0; r < repeats; r++) {
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

    if (matmulStatus != CUBLAS_STATUS_SUCCESS) {
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