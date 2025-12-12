# MILC

## Build MILC on LRC-H100 using the commit of the N10 benchmark

### 1. run build-quda.sh

```bash
#! /usr/bin/bash

BASE_DIRECTORY=$(pwd)
SOURCE_DIRECTORY=${BASE_DIRECTORY}/quda
BUILD_DIRECTORY=${BASE_DIRECTORY}/build
INSTALL_DIRECTORY=${BASE_DIRECTORY}/install

pushd .

cd $BASE_DIRECTORY
if [ ! -d ${SOURCE_DIRECTORY} ]
then
  git clone --branch develop https://github.com/lattice/quda ${SOURCE_DIRECTORY}
fi

cd ${SOURCE_DIRECTORY}
git checkout c75b77c731eb9ad16c93b4fc312e80225a84f1ea

mkdir -p ${BUILD_DIRECTORY}
mkdir -p ${INSTALL_DIRECTORY}

cd ${BUILD_DIRECTORY}

cmake ${BASE_DIRECTORY}/quda/ \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIRECTORY} \
  # change this due to GPUs
  -DQUDA_GPU_ARCH=sm_90 \
  -DQUDA_DIRAC_DEFAULT_OFF=ON \
  -DQUDA_DIRAC_STAGGERED=ON \
  -DQUDA_QMP=ON \
  -DQUDA_QIO=ON \
  -DQUDA_DOWNLOAD_USQCD=ON && \
cmake --build ${BUILD_DIRECTORY} --target all -- -j && \
cmake --build ${BUILD_DIRECTORY} --target install -- -j

popd
```

2. run build-milc.sh

```bash
#! /usr/bin/bash

BASE_DIRECTORY=$(pwd)
QUDA_DIRECTORY=${BASE_DIRECTORY}/install
MILC_DIRECTORY=${BASE_DIRECTORY}/milc_qcd

# Hack to find the CUDA directory
CUDA_DIRECTORY=$(which nvcc)
CUDA_DIRECTORY=${CUDA_DIRECTORY/\bin\/nvcc/}

pushd .

cd $BASE_DIRECTORY
if [ ! -d ${MILC_DIRECTORY} ]
then
  git clone --branch develop https://github.com/milc-qcd/milc_qcd ${MILC_DIRECTORY}
fi

cd ${MILC_DIRECTORY}
git checkout f803f4bf

cd ${MILC_DIRECTORY}/ks_imp_rhmc

if [ ! -f "./Makefile" ]
then
  cp ../Makefile .
fi

if [ -f "./su3_rhmd_hisq" ]
then
  rm ./su3_rhmd_hisq
fi

COMPILER="gnu" \
CTIME="-DCGTIME -DFFTIME -DGATIME -DGFTIME -DREMAP -DPRTIME -DIOTIME" \
CGEOM="-DFIX_NODE_GEOM -DFIX_IONODE_GEOM" \
MY_CC=mpicc \
MY_CXX=mpicxx \
CUDA_HOME=${CUDA_DIRECTORY} \
QUDA_HOME=${QUDA_DIRECTORY} \
WANTQUDA=true \
WANT_FN_CG_GPU=true \
WANT_FL_GPU=true \
WANT_GF_GPU=true \
WANT_FF_GPU=true \
WANT_GA_GPU=true \
WANT_MIXED_PRECISION_GPU=0 \
# 1 for single, 2 for double
PRECISION=1 \
MPP=true \
OMP=true \
WANTQIO=true \
WANTQMP=true \
QIOPAR=${QUDA_DIRECTORY} \
QMPPAR=${QUDA_DIRECTORY} \
make -j 1 su3_rhmd_hisq

popd
```



