# GPU Utilization Performance Profiling

This is a simple performance evaluation for GPU utilization

`eval_init.cpp`: profiling GPU utlization when initializing GPUs.   

`eval_io_pcie.cpp`: profiling GPU utilization when transferring data between host and a single GPU.

`eval_io_nvlink.cpp`: profiling GPU utilization when transferring data between host and mutiple GPUs using NVLINK.


```bash
# enter src folder
cd src

# build binary code *.x 
make 

# copy binary code to results folder
cp *.x ../scripts

# enter scripts folder
cd scripts

# using slurm to run script to get performance results
sbatch run_eval_init.sh
sbatch run_eval_io_pcie.sh
sbatch run_eval_io_nvlink.sh

# check results in the results folder
cd ../results
```
