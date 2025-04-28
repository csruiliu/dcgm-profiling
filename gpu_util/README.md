# GPU Utilization Performance Profiling

This is a simple performance evaluation for GPU utilization

`gpu_util_eval_init.cpp`: profiling GPU utlization when initializing GPUs.   

`gpu_util_eval_io.cpp`: profiling GPU utilization when transferring data between host and GPU.

```bash
# enter src folder
cd src

# build binary code *.x 
make 

# copy gpu_util_eval_init.x and gpu_util_eval_io.x to results folder
cp gpu_util_eval_init.x gpu_util_eval_io.x ../scripts

# enter scripts folder
cd scripts

# using slurm to run script to get performance results
sbatch run_gpu_util_eval.sh
# sbatch run_gemm_lt.sh

# check results in the results folder
cd ../results
```
