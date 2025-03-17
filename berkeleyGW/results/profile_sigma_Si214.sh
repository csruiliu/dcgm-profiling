#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH --qos=regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -A nstaff
#SBATCH -J bgw_eps_Si214
#SBATCH -C gpu
#SBATCH -o BGW_EPSILON_%j.out
# #SBATCH --reservation=n10scaling

source ../site_path_config.sh

mkdir BGW_SIGMA_$SLURM_JOBID
stripe_large BGW_SIGMA_$SLURM_JOBID
cp ./wrap_dcgmi.sh BGW_SIGMA_$SLURM_JOBID
cd    BGW_SIGMA_$SLURM_JOBID
ln -s $BGW_DIR/sigma.cplx.x .
ln -s  ../sigma.inp .
ln -sfn  ${Si214_WFN_folder}/WFNq.h5      .
ln -sfn  ${Si214_WFN_folder}/WFN_out.h5   ./WFN.h5


ulimit -s unlimited
export OMP_PROC_BIND=true
export OMP_PLACES=threads
export HDF5_USE_FILE_LOCKING=FALSE
export BGW_HDF5_WRITE_REDIST=1
export BGW_WFN_HDF5_INDEPENDENT=1

export OMP_NUM_THREADS=16
export DCGM_SAMPLE_RATE=100

start=$(date +%s.%N)

dcgm_delay=${DCGM_SAMPLE_RATE} srun -N 1 -n 4 -c 32 --cpu-bind=cores wrap_dcgmi.sh ./sigma.cplx.x

end=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end $start | bc -l)

printf "Elapsed Time: %.2f seconds\n" $elapsed > sigma_d${DCGM_SAMPLE_RATE}_runtime.out
