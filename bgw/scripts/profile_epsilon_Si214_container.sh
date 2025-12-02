#!/bin/bash
#SBATCH -t 00:20:00
#SBATCH --qos=sow
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=28
#SBATCH -A nstaff
#SBATCH -J bgw_eps_Si214
#SBATCH -C gpu&hbm40g
#SBATCH --perf=generic
#SBATCH -o /global/homes/r/ruiliu/dcgm-profiling/berkeleyGW/results/BGW_EPSILON_SI214_CNTR_%j/BGW_EPSILON_SI214_CNTR_%j.out

# #SBATCH -o /pscratch/sd/r/ruiliu/BGW_EPSILON_SI214_CNTR_%j/BGW_EPSILON_SI214_CNTR_%j.out
# #SBATCH -o /dev/shm/BGW_EPSILON_SI214_%j/BGW_EPSILON_SI214_%j.out
# #SBATCH -o ../results/BGW_EPSILON_SI214_%j/BGW_EPSILON_SI214_%j.out
# #SBATCH --reservation=n10scaling

podman-hpc run -d -it --name dcgm-container --rm --gpu --cap-add SYS_ADMIN nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04

source site_path_config.sh

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

export RESULTS_DIR=/global/homes/r/ruiliu/dcgm-profiling/berkeleyGW/results/BGW_EPSILON_SI214_CNTR_$SLURM_JOBID
#export RESULTS_DIR=/dev/shm/BGW_EPSILON_SI214_CNTR_$SLURM_JOBID
#export RESULTS_DIR=/pscratch/sd/r/ruiliu/BGW_EPSILON_SI214_CNTR_$SLURM_JOBID

if [ ! -d "$RESULTS_DIR" ]; then
  mkdir $RESULTS_DIR
fi

#stripe_large $RESULTS_DIR
cp ./wrap_dcgmi_container.sh $RESULTS_DIR
cd    $RESULTS_DIR
ln -s $BGW_DIR/epsilon.cplx.x .
ln -s $Si214_Benchmark_folder/epsilon.inp .
ln -sfn  ${Si214_WFN_folder}/WFNq.h5      .
ln -sfn  ${Si214_WFN_folder}/WFN_out.h5   ./WFN.h5

ulimit -s unlimited

export OMP_NUM_THREADS=14
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

dcgm_delay=${DCGM_SAMPLE_RATE} srun --cpu-bind=cores ./wrap_dcgmi_container.sh ./epsilon.cplx.x
#srun --cpu-bind=cores ./epsilon.cplx.x
end=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end $start | bc -l)

printf "Elapsed Time: %.2f seconds\n" $elapsed > epsilon_d${DCGM_SAMPLE_RATE}_runtime.out

rm -f eps0mat.h5
rm WFNq.h5
rm WFN.h5
#mv $RESULTS_DIR $HOME/dcgm-profiling/berkeleyGW/results/perf_model/
