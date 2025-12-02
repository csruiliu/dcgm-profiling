#!/bin/bash
#SBATCH -t 00:10:00
#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -A nstaff
#SBATCH -J bgw_eps_Si214
#SBATCH -C gpu&hbm40g
#SBATCH -o /pscratch/sd/r/ruiliu/BGW_EPSILON_SI214_%j/BGW_EPSILON_SI214_%j.out
#SBATCH --exclusive

# #SBATCH -o /dev/shm/BGW_EPSILON_SI214_%j/BGW_EPSILON_SI214_%j.out
# #SBATCH -o ../results/BGW_EPSILON_SI214_%j/BGW_EPSILON_SI214_%j.out
# #SBATCH --reservation=n10scaling

source site_path_config.sh

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

#RESULTS_DIR=../results/BGW_EPSILON_SI214_$SLURM_JOBID
#RESULTS_DIR=/dev/shm/BGW_EPSILON_SI214_$SLURM_JOBID
RESULTS_DIR=/pscratch/sd/r/ruiliu/BGW_EPSILON_SI214_$SLURM_JOBID

mkdir $RESULTS_DIR
stripe_large $RESULTS_DIR
cp ./wrap_dcgmi.sh $RESULTS_DIR
cd    $RESULTS_DIR
ln -s $BGW_DIR/epsilon.cplx.x .
ln -s ${Si214_Benchmark_folder}/epsilon.inp .
ln -sfn  ${Si214_WFN_folder}/WFNq.h5      .
ln -sfn  ${Si214_WFN_folder}/WFN_out.h5   ./WFN.h5

ulimit -s unlimited

export OMP_NUM_THREADS=16
export OMP_PROC_BIND=true
export OMP_PLACES=threads
#export OMP_MAX_ACTIVE_LEVELS=1
#export OMP_WAIT_POLICY=active
#export OMP_DYNAMIC=false

# BerkeleyGW specific
export HDF5_USE_FILE_LOCKING=FALSE
export BGW_HDF5_WRITE_REDIST=1
export BGW_WFN_HDF5_INDEPENDENT=1

export DCGM_SAMPLE_RATE=1000

start=$(date +%s.%N)

dcgm_delay=${DCGM_SAMPLE_RATE} srun --cpu-bind=cores wrap_dcgmi.sh ./epsilon.cplx.x
#srun --cpu-bind=cores ./epsilon.cplx.x
end=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end $start | bc -l)

printf "Elapsed Time: %.2f seconds\n" $elapsed > epsilon_d${DCGM_SAMPLE_RATE}_runtime.out

rm -f eps0mat.h5
rm WFNq.h5
rm WFN.h5
mv $RESULTS_DIR $HOME/dcgm-profiling/berkeleyGW/results/perf_model/
