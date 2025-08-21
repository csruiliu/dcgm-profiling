#! /usr/bin/bash

#SBATCH -A nstaff
#SBATCH -C gpu&hbm40g       # Specifically request the 40gb GPUs; this is purely for throughput reasons
#SBATCH --qos sow       # Update as appropriate
#SBATCH -t 1:00:00          # One hour runtime
#SBATCH -N 1                # This is the total number of nodes
#SBATCH --ntasks-per-node=4 # Number of tasks per node
# #SBATCH -J job_nae
#SBATCH --gpus-per-task=1   # 1 GPU per MPI task
#SBATCH --gpu-bind=none     # This is necessary to let all 4 ranks on a node access all 4 GPUs.
#SBATCH --perf=generic

export NNODES=${SLURM_JOB_NUM_NODES}
export NGPUS=${SLURM_NTASKS_PER_NODE}
export NRANK=$((${NGPUS} * ${NNODES}))

# Problem size information
export PROBLEM="medium"
export DECOMP="1 1 1 4" # 1 node, 4 gpu, decomposed in the `t` direction

# Packages and compiler things... probably not necessary, but I'm not in the mood to
# try
module purge
module load PrgEnv-gnu
module load cmake
module load cudatoolkit
module load craype-accel-nvidia80
module load craype-x86-milan
module load cray-fftw

export MPICH_GPU_SUPPORT_ENABLED=1
export CRAY_CPU_TARGET=x86-64
export CC=$(which cc)
export CXX=$(which CC)
export MPI_HOME=$MPICH_DIR
export MPI_CXX_COMPILER=$(which CC)
export MPI_CXX_COMPILER_FLAGS=$(CC --cray-print-opts=all)

### MPI runtime flags
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_NEMESIS_ASYNC_PROGRESS=1

### MPI binding flags
export MPICH_VERSION_DISPLAY=1
export MPICH_OFI_NIC_VERBOSE=2

export MPICH_OFI_NIC_POLICY="USER"
export MPICH_OFI_NIC_MAPPING="0:3;1:2;2:1;3:0"
echo "MPICH_OFI_NIC_POLICY=${MPICH_OFI_NIC_POLICY}"
echo "MPICH_OFI_NIC_MAPPING=${MPICH_OFI_NIC_MAPPING}"

### Cray/Slurm runtime flags
export OMP_NUM_THREADS=16
export SLURM_CPU_BIND=cores
export CRAY_ACCEL_TARGET=nvidia80

# Runstring with some parameters baked in
export RUNSTRING="${PROBLEM}-${NNODES}node-${NGPUS}gpu-${DECOMP// /x}"

export RESULTS_DIR="../results/MILC_${SLURM_JOBID}_${RUNSTRING}"

if [ ! -d "$RESULTS_DIR" ]; then
  mkdir -p $RESULTS_DIR
fi


### QUDA specific flags
export QUDA_RESOURCE_PATH=`pwd`/tunecache-${RUNSTRING} # location of QUDA autotune cache file
mkdir -p $QUDA_RESOURCE_PATH
export QUDA_ENABLE_GDR=1
export QUDA_MILC_HISQ_RECONSTRUCT=13               # set QUDA-MILC solver optimization
export QUDA_MILC_HISQ_RECONSTRUCT_SLOPPY=9         # set QUDA-MILC solver optimization

N10_MILC="/pscratch/sd/r/ruiliu/2025-08-07QudaMilcTest"
MILC_QCD_DIR=${N10_MILC}/milc_qcd
exec=${MILC_QCD_DIR}/ks_imp_rhmc/su3_rhmd_hisq

podman-hpc run -d -it --name dcgm-container --rm --gpu --cap-add SYS_ADMIN nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04

srun -v -N $NNODES --cpu-bind none --ntasks-per-node $NGPUS --gpus-per-task 1 --gpu-bind none $exec in_${PROBLEM}_tune.in -qmp-geom 1 1 1 4 -qmp-logic-map 3 2 1 0 -qmp-alloc-map 3 2 1 0 2>&1 | tee ${RESULTS_DIR}/nersc-${RUNSTRING}.out

export DCGM_SAMPLE_RATE=1000

dcgm_delay=${DCGM_SAMPLE_RATE} srun -v -N $NNODES --cpu-bind none --ntasks-per-node $NGPUS --gpus-per-task 1 --gpu-bind none \
  ./wrap_dcgmi_container.sh $exec in_${PROBLEM}.in -qmp-geom 1 1 1 4 -qmp-logic-map 3 2 1 0 -qmp-alloc-map 3 2 1 0 2>&1 | tee ${RESULTS_DIR}/nersc-${RUNSTRING}.out



