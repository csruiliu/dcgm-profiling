#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH --qos=regular
#SBATCH -N 16
#SBATCH --gpus-per-node=4
#SBATCH -A nstaff
#SBATCH -J bgw_sig_Si510
#SBATCH  -C  gpu
#SBATCH -o BGW_SIGMA_%j.out
# #SBATCH --reservation=n10scaling

source site_path_config.sh

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

RESULTS_DIR=../results/BGW_EPSILON_$SLURM_JOBID

mkdir $RESULTS_DIR
stripe_large $RESULTS_DIR
cp ./wrap_dcgmi.sh $RESULTS_DIR
cd    $RESULTS_DIR

ln -s $BGW_DIR/sigma.cplx.x .
NNPOOL=8
cat ../sigma.inp |\
sed "s/NNPOOL/${NNPOOL}/g" > sigma.inp
ln -sfn  ${Si510_WFN_folder}/WFN_out.h5   ./WFN_inner.h5
ln -sfn  ${Si510_WFN_folder}/RHO          .
ln -sfn  ${Si510_WFN_folder}/VXC          .
ln -sfn  ${Si510_WFN_folder}/eps0mat.h5   .


ulimit -s unlimited
export OMP_PROC_BIND=true
export OMP_PLACES=threads
export HDF5_USE_FILE_LOCKING=FALSE
export BGW_HDF5_WRITE_REDIST=1
export BGW_WFN_HDF5_INDEPENDENT=1

export OMP_NUM_THREADS=16
export DCGM_SAMPLE_RATE=1000

start=$(date +%s.%N)
dcgm_delay=${DCGM_SAMPLE_RATE} srun -n 64 -c 32 --cpu-bind=cores wrap_dcgmi.sh ./sigma.cplx.x

end=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end $start | bc -l)

printf "Elapsed Time: %.2f seconds\n" $elapsed > sigma_d${DCGM_SAMPLE_RATE}_runtime.out