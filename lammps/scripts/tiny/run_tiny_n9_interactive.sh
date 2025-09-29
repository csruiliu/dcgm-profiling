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
EXE="/global/homes/r/ruiliu/dcgm-profiling/lammps/bin-ptlin/lmp_fp32_n9"

# Match the build env.
export MPICH_GPU_SUPPORT_ENABLED=1

gpus_per_node=4

input="-k on g $gpus_per_node -sf kk -pk kokkos newton on neigh half ${BENCH_SPEC} "

command="srun -n $gpus_per_node $EXE $input"

echo $command

$command > lammps_small_a100_${SLURM_JOB_ID}.out

#unlink wrap_dcgmi.sh
unlink common

cd ..
mv lammps_$spec.$SLURM_JOB_ID ../../results/

