#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -A m888

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

sleep 60

srun -n 1 -c 1 --cpu_bind=cores -G 1 --gpu-bind=single:1 \
	./gemm.x 16384 1000 1.0 1.0 $prec

sleep 60



