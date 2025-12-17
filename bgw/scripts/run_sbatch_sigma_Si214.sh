#!/bin/bash
#SBATCH -t 00:20:00
#SBATCH --qos=sow
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --cpu-bind=cores
#SBATCH -A nstaff
#SBATCH -J bgw_sig_Si214
#SBATCH -C gpu&hbm40g
#SBATCH --perf=generic
#SBATCH -o /pscratch/sd/r/ruiliu/BGW_SIGMA_SI214_%j/BGW_SIGMA_SI214_%j.out

podman-hpc run -d -it --name dcgm-container --rm --gpu --cap-add SYS_ADMIN nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04

source ../site_path_config.sh

# create results directory if not exist
if [ ! -d "../results" ]; then
  mkdir ../results
fi

RESULTS_DIR="/pscratch/sd/r/ruiliu/BGW_SIGMA_SI214_${SLURM_JOBID}"

mkdir $RESULTS_DIR
stripe_large $RESULTS_DIR
cp ./wrap_dcgmi_container.sh $RESULTS_DIR
cd    $RESULTS_DIR

ln -s $BGW_DIR/sigma.cplx.x .
NNPOOL=2
cat ${Si214_Benchmark_folder}/sigma.inp |\
sed "s/NNPOOL/${NNPOOL}/g" > sigma.inp
ln -sfn  ${Si214_WFN_folder}/WFN_out.h5   ./WFN_inner.h5
ln -sfn  ${Si214_WFN_folder}/RHO          .
ln -sfn  ${Si214_WFN_folder}/VXC          .
ln -sfn  ${Si214_WFN_folder}/eps0mat.h5   .

ulimit -s unlimited
export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# BerkeleyGW specific
export HDF5_USE_FILE_LOCKING=FALSE
export BGW_HDF5_WRITE_REDIST=1
export BGW_WFN_HDF5_INDEPENDENT=1

export DCGM_SAMPLE_RATE=1000

start=$(date +%s.%N)
dcgm_delay=${DCGM_SAMPLE_RATE} srun --cpu-bind=cores wrap_dcgmi_container.sh ./sigma.cplx.x

end=$(date +%s.%N)
elapsed=$(printf "%s - %s\n" $end $start | bc -l)

printf "Elapsed Time: %.2f seconds\n" $elapsed > sigma_d${DCGM_SAMPLE_RATE}_runtime.out

rm -f eps0mat.h5
rm WFNq.h5
rm WFN.h5
#mv $RESULTS_DIR $HOME/dcgm-profiling/berkeleyGW/results/perf_model/