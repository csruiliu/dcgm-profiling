#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -A nstaff
#SBATCH -o ../results/GEMM_%j/GEMM_%j.out

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

export RESULTS_DIR=../results/GEMM_${SLURM_JOBID}

export DCGM_SAMPLE_RATE=100
#gemm.x args
# 1: matrix size
# 2: repeats
# 3: alpha
# 4: beta
# 5: precision

for prec in D S H; do
#run the application:
start=$(date +%s.%N)
dcgm_delay=${DCGM_SAMPLE_RATE} \
	srun -n 1 -c 1 --cpu_bind=cores -G 1 --gpu-bind=single:0 \
	./wrap_dcgmi.sh \
	./gemm.x 16384 100 1.0 1.0 $prec \
	> ${RESULTS_DIR}/"$prec"gemm-${SLURM_JOBID}.dcgmi
end=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end $start | bc -l)
printf "Elapsed Time: %.2f seconds\n" $elapsed > ${RESULTS_DIR}/"$prec"gemm_d${DCGM_SAMPLE_RATE}_runtime.out
done



