#!/bin/bash
#SBATCH -t 00:30:00
##SBATCH --qos=regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -A nstaff
#SBATCH -J bgw_sig_Si214
#SBATCH  -C gpu
#SBATCH -o BGW_SIGMA_%j.out
# #SBATCH --reservation=n10scaling

source ../site_path_config.sh

mkdir BGW_SIGMA_$SLURM_JOB_ID
stripe_large BGW_SIGMA_$SLURM_JOB_ID
cp ./wrap_dcgmi.sh BGW_SIGMA_$SLURM_JOB_ID
cd    BGW_SIGMA_$SLURM_JOB_ID

ln -s $BGW_DIR/sigma.cplx.x .
NNPOOL=2
cat ../sigma.inp |\
sed "s/NNPOOL/${NNPOOL}/g" > sigma.inp
ln -sfn  ${Si214_WFN_folder}/WFN_out.h5   ./WFN_inner.h5
ln -sfn  ${Si214_WFN_folder}/RHO          .
ln -sfn  ${Si214_WFN_folder}/VXC          .
ln -sfn  ${Si214_WFN_folder}/eps0mat.h5   .


ulimit -s unlimited
export OMP_PROC_BIND=true
export OMP_PLACES=threads
export HDF5_USE_FILE_LOCKING=FALSE
export BGW_HDF5_WRITE_REDIST=1
export BGW_WFN_HDF5_INDEPENDENT=1

export OMP_NUM_THREADS=16
export DCGM_SAMPLE_RATE=1000

start=$(date +%s.%N)
dcgm_delay=${DCGM_SAMPLE_RATE} srun -n 4 -c 32 --cpu-bind=cores wrap_dcgmi.sh ./sigma.cplx.x

end=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end $start | bc -l)

printf "Elapsed Time: %.2f seconds\n" $elapsed > sigma_d${DCGM_SAMPLE_RATE}_runtime.out