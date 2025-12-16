#!/bin/bash

source ../site_path_config.sh

export RESULTS_DIR=/global/scratch/users/rliu5/bgw-lrc-h100-fp64/scripts/small_Si214/BGW_EPSILON_${SLURM_JOB_ID}

mkdir $RESULTS_DIR
#stripe_large $RESULTS_DIR
cd    $RESULTS_DIR
ln -s $BGW_DIR/epsilon.cplx.x .
ln -s  ../epsilon.inp .
ln -sfn  ${Si214_WFN_folder}/WFNq.h5      .
ln -sfn  ${Si214_WFN_folder}/WFN_out.h5   ./WFN.h5


ulimit -s unlimited

export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

export HDF5_USE_FILE_LOCKING=FALSE
export BGW_HDF5_WRITE_REDIST=1
export BGW_WFN_HDF5_INDEPENDENT=1

DCGM_PATH=/global/scratch/users/rliu5/bgw-lrc-h100-fp64/scripts/wrap_dcgmi_container.sh

srun -n 1 -c 32 --cpu-bind=cores $DCGM_PATH ./epsilon.cplx.x > ${RESULTS_DIR}/bgw-${SLURM_JOB_ID}.out

rm -f eps0mat.h5
unlink epsilon.cplx.x
unlink epsilon.inp
unlink WFN.h5
unlink WFNq.h5

