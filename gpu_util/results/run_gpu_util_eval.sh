#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -A nstaff

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#gemm.x args
# 1: matrix size
# 2: repeats
# 3: alpha
# 4: beta
# 5: precision

dcgm_delay=10 \
	srun -n 1 -c 1 --cpu_bind=cores -G 1 --gpu-bind=single:1 \
	./wrap_dcgmi.sh \
	./gpu_util_eval_init.x \
	> gpu_util_eval_init.dcgmi

dcgm_delay=10 \
	srun -n 1 -c 1 --cpu_bind=cores -G 1 --gpu-bind=single:1 \
	./wrap_dcgmi.sh \
	./gpu_util_eval_io.x \
	> gpu_util_eval_io.dcgmi


