#!/bin/bash

# spec.txt provides the input specification
# by defining the variables spec and BENCH_SPEC
source small_spec.txt

mkdir lammps_$spec.$SLURM_JOB_ID
cd    lammps_$spec.$SLURM_JOB_ID
ln -s ../../common .
#cp ${0} .
cp ../small_spec.txt .
#ln -s ../../wrap_dcgmi.sh .

# This is needed if LAMMPS is built using cmake.
#install_dir="../../../install_PM"
#export LD_LIBRARY_PATH=${install_dir}/lib64:$LD_LIBRARY_PATH
EXE="/global/scratch/users/rliu5/lammps-lrc-h100-fp64/install_lammps/bin/lmp"

# Match the build env.
export MPICH_GPU_SUPPORT_ENABLED=1

export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

input="-k on g $gpus_per_node -sf kk -pk kokkos newton on neigh half ${BENCH_SPEC} "

command="srun -n 1 -c 32 --cpu-bind=cores $EXE $input"

echo $command

$command > lammps_small_a100_${SLURM_JOB_ID}.out

#unlink wrap_dcgmi.sh
unlink common

cd ..
mv lammps_$spec.$SLURM_JOB_ID ../../results/

