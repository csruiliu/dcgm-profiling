# GEMM Performance Profiling

This repository contains some codes for profiling GPU performance across different GEMM (General Matrix Multiplication) operations and scenarios.

**Basic GEMM Profiling**

`gemm.cpp` and `cal_gemm.cpp`: Profile cuBLAS performance for standard GEMM operations across multiple precision formats:

(1) Single precision: cublasSgemm

(2) Double precision: cublasDgemm

(3) Half precision: cublasHgemm

**cuBLASLt GEMM Profiling**

`gemm_lt.cpp` and `cal_gemm_lt.cpp`: Profile GPU performance using the cublasLtMatmul API with support for:

(1) Half precision (FP16)

(2) Single precision (FP32)

(3) Double precision (FP64)

(4) Integer operations

**Multi-GPU GEMM Profiling**

`gemm_mpi.cpp` and `cal_gemm_mpi.cpp`: Profile GPU performance in distributed computing environments with: 

(1) Multiple GPU support

(2) MPI-based inter-GPU communication

(3) standard GEMM operations across mutiple precision formats: single precision (cublasSgemm), double precision (cublasDgemm), and half precision (cublasHgemm).


```bash
# enter src folder
cd src

# build binary code gemm.x or gemm_lt.x 
make 

# copy all *.x to script folder
cp *.x ../scripts

# enter scripts folder
cd scripts

# using slurm to run script and check performance results in `results` folder
sbatch run_gemm.sh
sbatch run_gemm_lt.sh
sbatch run_gemm_mpi.sh
```
