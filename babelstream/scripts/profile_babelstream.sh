#!/bin/bash
#SBATCH --qos=debug
#SBATCH -C gpu&hbm40g
#SBATCH -G 1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 00:30:00
#SBATCH -A nstaff
#SBATCH -o ../results/BABELSTREAM_%j/BABELSTREAM_%j.out

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=true

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

export RESULTS_DIR=../results/BABELSTREAM_${SLURM_JOBID}

export DCGM_SAMPLE_RATE=1000

#Array size must be a multiple of 1024
export ARRAYSIZE=100663296
export NUMTIMES=100000 
export BABELSTREAM="/pscratch/sd/r/ruiliu/BabelStream-5.0/build/cuda-stream"

#run the application:
start=$(date +%s.%N)
dcgm_delay=${DCGM_SAMPLE_RATE} srun --cpu_bind=cores ./wrap_dcgmi.sh $BABELSTREAM -s $ARRAYSIZE -n $NUMTIMES \
	> ${RESULTS_DIR}/"$prec"babelstream-${SLURM_JOBID}.dcgmi
end=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end $start | bc -l)
printf "Elapsed Time: %.2f seconds\n" $elapsed > ${RESULTS_DIR}/"$prec"babelstream_d${DCGM_SAMPLE_RATE}_runtime.out



dcgm_delay=1000 ./wrap_dcgmi.sh $BABELSTREAM -s $ARRAYSIZE -n $NUMTIMES \
> /babelstream-${SLURM_JOBID}.dcgmi