#!/bin/bash
#SBATCH -t 00:20:00
#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -A nstaff
#SBATCH -J bgw_sig_Si214
#SBATCH -C gpu&hbm40g
#SBATCH -o /pscratch/sd/r/ruiliu/BGW_SIGMA_SI214_%j/BGW_SIGMA_SI214_%j.out
#SBATCH --exclusive

# #SBATCH -o /dev/shm/BGW_SIGMA_SI214_%j/BGW_SIGMA_SI214_%j.out
# #SBATCH -o ../results/BGW_SIGMA_SI214_%j/BGW_SIGMA_SI214_%j.out
# #SBATCH --reservation=n10scaling

source site_path_config.sh

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

#RESULTS_DIR=../results/BGW_SIGMA_SI214_$SLURM_JOBID
#RESULTS_DIR=/dev/shm/BGW_SIGMA_SI214_$SLURM_JOBID
RESULTS_DIR=/pscratch/sd/r/ruiliu/BGW_SIGMA_SI214_$SLURM_JOBID

mkdir $RESULTS_DIR
stripe_large $RESULTS_DIR
cp ./wrap_dcgmi.sh $RESULTS_DIR
cd    $RESULTS_DIR

ln -s $BGW_DIR/sigma.cplx.x .
NNPOOL=2
cat ../sigma.inp |\
sed "s/NNPOOL/${NNPOOL}/g" > sigma.inp
ln -sfn  ${Si214_WFN_folder}/WFN_out.h5   ./WFN_inner.h5
ln -sfn  ${Si214_WFN_folder}/RHO          .
ln -sfn  ${Si214_WFN_folder}/VXC          .
ln -sfn  ${Si214_WFN_folder}/eps0mat.h5   .


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
dcgm_delay=${DCGM_SAMPLE_RATE} srun -n 4 -c 14 --cpu-bind=cores wrap_dcgmi.sh ./sigma.cplx.x

end=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end $start | bc -l)

printf "Elapsed Time: %.2f seconds\n" $elapsed > sigma_d${DCGM_SAMPLE_RATE}_runtime.out

rm -f eps0mat.h5
rm WFNq.h5
rm WFN.h5
mv $RESULTS_DIR $HOME/dcgm-profiling/berkeleyGW/results/perf_model/