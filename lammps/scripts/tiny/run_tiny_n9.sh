#!/bin/bash
#SBATCH --qos=sow
#SBATCH --account=nstaff
#SBATCH --job-name=lmp_tiny
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --exclusive
#SBATCH -C gpu&hbm40g
#SBATCH -G 4
#SBATCH --gpu-bind=none
#SBATCH -t 00:10:00
# #SBATCH -o ../../results/lammps_small.%j

# spec.txt provides the input specification
# by defining the variables spec and BENCH_SPEC
source small_spec.txt

mkdir lammps_$spec.$SLURM_JOB_ID
cd    lammps_$spec.$SLURM_JOB_ID
ln -s ../../common .
#cp ${0} .
cp ../small_spec.txt .

# This is needed if LAMMPS is built using cmake.
#install_dir="../../../install_PM"
#export LD_LIBRARY_PATH=${install_dir}/lib64:$LD_LIBRARY_PATH
EXE="/global/homes/r/ruiliu/dcgm-profiling/lammps/bin-ptlin/lmp_fp32_n9"

# Match the build env.
export MPICH_GPU_SUPPORT_ENABLED=1

gpus_per_node=4

input="-k on g $gpus_per_node -sf kk -pk kokkos newton on neigh half ${BENCH_SPEC} " 

command="srun -n $SLURM_NTASKS $EXE $input"

echo $command

$command > lammps_tiny_a100_${SLURM_JOB_ID}.out


