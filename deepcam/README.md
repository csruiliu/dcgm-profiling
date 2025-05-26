# DeepCAM AI Benchmark

## Building and Running DeepCAM AI on Perlmutter

**Setup Environment**

```bash
export N10_DEEPCAM=$(pwd)
sed -i s%FIXME%$N10_DEEPCAM% deepcam_env.sh
```

The following modules will be used for running but not necessary for building.

```bash
module load cudatoolkit/12.4 (loaded by default)
module load cray-mpich/8.1.30 (loaded by default)
module load cudnn/8.9.3_cuda12
module load nccl/2.24.3
module load cray-hdf5/1.14.3.1
```

**Obtain DeepCAM source code**

```bash
# Download the baseline implementation
build_scripts/sparse_checkout.sh baseline
# Download the Perlmutter-optimized implementation
build_scripts/sparse_checkout.sh optimized_pm
```

**Install Required Python **

1. Creating python virtual environment using pyenv and virtualenv (make sure using python 3.9+)

2. Activiating the python virtual environment

```bash
# Make sure the python virtual environment has been activiated
pip install -r $N10_DEEPCAM/build_scripts/requirements.txt
pip install mpi4py==3.1.6
# We are using cudatoolkit/12.4, so install the matched pytorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

**Install Nivida Apex and glog**

```bash
mkdir local
mkdir local/build
export PREFIX="${N10_DEEPCAM}/local"
export BUILD="${N10_DEEPCAM}/local/build"
cd local/build

# Install Nvidia Apex
export CC=$(which gcc)
export CXX=$(which g++)
wget https://github.com/NVIDIA/apex/archive/refs/tags/25.08.tar.gz
cd apex-25.08
pip install -v --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# Install glog
wget https://github.com/google/glog/archive/refs/tags/v0.6.0.tar.gz
tar -xvzf v0.6.0.tar.gz
cd glog-0.6.0
cmake -S . -B build -G "Unix Makefiles"
cmake --build build -j16
cmake --install build --prefix $PREFIX
```

**Obtain DeepCAM source code**

Donwload input data using [globus](https://gitlab.com/NERSC/N10-benchmarks/deepcam/-/blob/main/data/globus.md), and make sure the datasets are in `$N10_DEEPCAM/data/deepcam-data-mini`

```bash
cd $N10_DEEPCAM/data
tar -xzf deepcam-data-mini/deepcam-data-n512.tgz (taking deepcam-data-n512.tgz as example)
./install_data.sh deepcam-data-n512 mini 1
```

**Run DeepCAM Script**

The following is the revised `run_training_mini.slurm`.

```bash
#!/bin/bash
#SBATCH --job-name=deepcam-mini
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=4
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=%x-%j.out
#SBATCH --account=nstaff

#load the DeepCAM environment
#N10_DEEPCAM is defined in deepcam_env.sh

module load cray-hdf5
module load cudnn/8.9.3_cuda12
module load nccl
#module load python
source $HOME/venv-python/deepcam/bin/activate
source ../deepcam_env_rui.sh

#load hyperparameters
#BENCH_RCP is defined by bench_rcp.conf
#this reference convergence point (RCP) should not be modified
source bench_rcp.conf

#the local batch size may be adjusted
#under the constraint that the global batch size is fixed to 2048,
#i.e. processes * local_batch_size = 2048.
#for example: local_batch_size=$(( 2048 / ${SLURM_NTASKS} ))
local_batch_size=2

#other options within this script may be adjusted freely
data_dir=$N10_DEEPCAM/data/mini
output_dir=output_dir
run_tag="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"

srun python3 $N10_DEEPCAM/baseline/src_deepCam/train.py \
    ${BENCH_RCP_BASELINE} \
    --wireup_method "nccl-slurm" \
    --run_tag ${run_tag} \
    --data_dir_prefix ${data_dir} \
    --output_dir ${output_dir} \
    --model_prefix "segmentation" \
    --optimizer "LAMB" \
    --max_epochs 64 \
    --max_inter_threads 1 \
    --local_batch_size ${local_batch_size}


#save results for successful run
if [[ $? == 0 ]]; then
   mkdir -p $N10_DEEPCAM/results/jobscripts
   mkdir -p $N10_DEEPCAM/results/logs
   if [[ $SLURM_JOB_QOS != interactive ]] && [[ SLURM_JOB_NAME != interactive ]]; then
       cp ${0} $N10_DEEPCAM/results/jobscripts/${SLURM_JOB_ID}.slurm
       cp -p $output_dir/logs/${run_tag}.log $N10_DEEPCAM/results/logs 
   fi
fi
```

We copy and modify `deepcam_env.sh` and have `deepcam_env_rui.sh`

```bash
export N10_DEEPCAM="/pscratch/sd/r/ruiliu/deepcam"
if [[ -z "${N10_DEEPCAM}" ]]; then
  echo "The N10_DEEPCAM environment variable has not been set!"
  echo "Edit deepcam_env.sh to define N10_DEEPCAM and try again."
  return 1
fi

prepend_env () {
    # This function is needed since trailing colons
    # on some environment variables can cause major
    # problems...
    local envname=$1
    local envval=$2
    if [ "x${!envname}" = "x" ]; then
        export ${envname}="${envval}"
    else
        export ${envname}="${envval}":${!envname}
    fi
}

load_deepcam () {

    # Location of the software stack
    export PREFIX=${N10_DEEPCAM}/local
    export BUILD=${N10_DEEPCAM}/local/build
    mkdir -p $PREFIX 
    mkdir -p $BUILD 

    # Add software stack to the environment
    prepend_env "PATH" "${PREFIX}/bin"
    prepend_env "CPATH" "${PREFIX}/include"
    prepend_env "LD_LIBRARY_PATH" "${PREFIX}/lib"
    prepend_env "PYTHONPATH" "/global/u1/r/ruiliu/venv-python/deepcam/lib/python3.9/site-packages"
}
```

-----
1. DeepCAM AI benchmark, https://gitlab.com/NERSC/N10-benchmarks/deepcam