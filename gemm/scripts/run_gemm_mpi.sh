#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -A nstaff
#SBATCH --exclusive
#SBATCH -o ../results/GEMM_MPI_%j/GEMM_MPI_%j.out

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

export RESULTS_DIR=../results/GEMM_MPI_${SLURM_JOBID}

export DCGM_SAMPLE_RATE=100

for prec in D S H I; do
#run the application:
dcgm_delay=${DCGM_SAMPLE_RATE} \
	srun -n 4 -c 1 --cpu_bind=cores -G 4 \
	./wrap_dcgmi_mpi.sh \
	./gemm_mpi.x 16384 100 1.0 1.0 $prec \
	> ${RESULTS_DIR}/"$prec"gemm_mpi-${SLURM_JOBID}.dcgmi
done



