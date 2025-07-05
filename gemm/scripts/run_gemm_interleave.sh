#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu&hbm40g
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -t 00:05:00
#SBATCH -A nstaff
#SBATCH -o ../results/GEMM_INTERLEAVE_%j/GEMM_INTERLEAVE_%j.out

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

export RESULTS_DIR=../results/GEMM_INTERLEAVE_${SLURM_JOBID}

export DCGM_SAMPLE_RATE=1000

#run the application:
dcgm_delay=${DCGM_SAMPLE_RATE} \
srun --cpu_bind=cores --gpu-bind=single:1 ./wrap_dcgmi.sh ./gemm_interleave.x \
	> ${RESULTS_DIR}/gemm_interleave-${SLURM_JOBID}.dcgmi

