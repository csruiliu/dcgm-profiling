#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -A nstaff
#SBATCH --exclusive
#SBATCH -o ../results/GPU_UTIL_%j/GPU_UTIL_%j.out

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

export RESULTS_DIR=../results/GPU_UTIL_${SLURM_JOBID}

export DCGM_SAMPLE_RATE=100

dcgm_delay=${DCGM_SAMPLE_RATE} \
	srun -n 1 -c 1 --cpu_bind=cores -G 1 --gpu-bind=single:1 \
	./wrap_dcgmi.sh \
	./eval_io_nvlink.x \
	> $RESULTS_DIR/gpu_util_eval_io-$SLURM_JOBID.dcgmi


