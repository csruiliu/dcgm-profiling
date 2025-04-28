# GEMM Performance Profiling

This is a simple GPU utilization profiling.

`gemm.cpp` and `cal_gemm.cpp`: profiling for `cublasSgemm`, `cublasDgemm`, and `cublasHgemm`

`gemm_lt.cpp` and `cal_gemm_lt.cpp`: profiling for cublasLT

```bash
# enter src folder
cd src

# build binary code gemm.x or gemm_lt.x 
make 

# copy gemm.x and gemm_lt.x to results folder
cp gemm.x gemm_lt.x ../scripts

# enter scripts folder
cd scripts

# using slurm to run script to get performance results
sbatch run_gemm.sh
# sbatch run_gemm_lt.sh

# check results in the results folder
cd ../results
```
