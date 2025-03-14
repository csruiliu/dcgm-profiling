# GEMM Tutorials & Exercises

This is a simple tutorial for GEMM (General Matrix Multiplication) performance evaluation.

`gemm.cpp` and `cal_gemm.cpp`: profiling for `cublasSgemm`, `cublasDgemm`, and `cublasHgemm`

`gemm_lt.cpp` and `cal_gemm_lt.cpp`: profiling for cublasLT

More details about cublasLT via link: https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api

```bash
# enter src folder
cd src

# build binary code gemm.x or gemm_lt.x 
make 

# copy gemm.x and gemm_lt.x to results folder
cp gemm.x gemm_lt.x ../results

# enter results folder
cd results

# using slurm to run script to get performance results
sbatch run_gemm.sh
# sbatch run_gemm_lt.sh
```
