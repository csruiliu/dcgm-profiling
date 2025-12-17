#!/bin/bash
#SBATCH -N 1
#SBATCH -t 7:30:00
#SBATCH --account=pc_perfume
#SBATCH --job-name=milc-tiny
#SBATCH --qos=es_normal
#SBATCH --partition=es2
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --cpu-bind=cores
#SBATCH --perf=generic

export MPICH_GPU_SUPPORT_ENABLED=1

N10_MILC="/global/scratch/users/rliu5/milc-lrc-h100-fp32"
MILC_QCD_DIR=${N10_MILC}/milc_qcd
LATTICE_DIR=${N10_MILC}/lattices

if [ ! -d lattices ]; then
    ln -s $LATTICE_DIR ./lattices
fi

bind=${N10_MILC}/bin/bind4-perlmutter.sh
exe=${MILC_QCD_DIR}/ks_imp_rhmc/su3_rhmd_hisq
input=input_4864

export OMP_NUM_THREADS=16
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

export QUDA_ENABLE_GDR=1
export QUDA_MILC_HISQ_RECONSTRUCT=13
export QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=9

# Tuning results are stored in qudatune_dir.
qudatune_dir="$PWD/qudatune"
export QUDA_RESOURCE_PATH=${qudatune_dir}
if [ ! -d ${qudatune_dir} ]; then
    mkdir ${qudatune_dir}
fi

export RESULTS_DIR="/global/scratch/users/rliu5/milc-lrc-h100/scripts/tiny1x1x1x1/milc-tiny-${SLURM_JOB_ID}" 
mkdir -p $RESULTS_DIR

SLURM_NTASKS=1
SLURM_CPUS_PER_TASK=32

command="srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK --cpu-bind=cores ./wrap_dcgmi_container.sh $exe $input"

echo $command

$command > ${RESULTS_DIR}/milc-tiny-${SLURM_JOB_ID}.out

