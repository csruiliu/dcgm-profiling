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

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

#gemm.x args
# 1: matrix size
# 2: repeats
# 3: alpha
# 4: beta
# 5: precision

for prec in D S H I; do
#run the application:
dcgm_delay=100 \
	srun -n 1 -c 1 --cpu_bind=cores -G 1 --gpu-bind=single:1 \
	./wrap_dcgmi.sh \
	./gemm_lt.x 16384 100 1.0 1.0 $prec \
	> ../results/"$prec"gemm_lt-$SLURM_JOBID.dcgmi
done



