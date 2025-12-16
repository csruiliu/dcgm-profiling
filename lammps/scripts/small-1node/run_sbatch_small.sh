#!/bin/bash
#SBATCH --qos=sow
#SBATCH --account=nstaff
#SBATCH --job-name=lmp_small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH -C gpu&hbm40g
#SBATCH -G 1
#SBATCH --gpu-bind=none
#SBATCH --cpu-bind=cores
#SBATCH --perf=generic
#SBATCH -t 00:30:00
#SBATCH -o lammps_small_a100_1gpu_dcgm_%j/lammps_small.%j

podman-hpc run -d -it --name dcgm-container --rm --gpu --cap-add SYS_ADMIN -p 5555:5555 nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04

# spec.txt provides the input specification
# by defining the variables spec and BENCH_SPEC
source small_spec.txt

gpus_per_node=1

BASE_DIR="/pscratch/sd/r/ruiliu/lammps-pm-a100-fp64"

export RESULTS_DIR="${BASE_DIR}/scripts/small-1node/lammps_small_a100_1gpu_dcgm_${SLURM_JOBID}"

mkdir -p lammps_small_a100_1gpu_dcgm_${SLURM_JOBID}
cd    lammps_small_a100_1gpu_dcgm_${SLURM_JOBID}
ln -s ../../common .
#cp ${0} .
cp ../small_spec.txt .
ln -s ../../wrap_dcgmi_container.sh .

# This is needed if LAMMPS is built using cmake.
#install_dir="../../../install_PM"
#export LD_LIBRARY_PATH=${install_dir}/lib64:$LD_LIBRARY_PATH
EXE="${BASE_DIR}/install_lammps/bin/lmp"

# For different cluster, num_threads could be different
export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# Match the build env.
module load PrgEnv-gnu
module load cudatoolkit
module load craype-accel-nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1

input="-k on g $gpus_per_node -sf kk -pk kokkos newton on neigh half ${BENCH_SPEC} " 

command="srun -n $SLURM_NTASKS ./wrap_dcgmi_container.sh $EXE $input"
#command="srun -n $SLURM_NTASKS $EXE $input"

echo $command

$command

unlink common
unlink wrap_dcgmi_container.sh

cd ..
mv lammps_small_a100_1gpu_dcgm_${SLURM_JOBID} ../../results/
