# MILC

## Build MILC on LRC-H100 using the commit in the N10 benchmark

### 1. Compile and install QUDA 

```
# Create necessary folders
mkdir ~/dcgm-profiling/milc
cd ~/dcgm-profiling/milc
mkdir quda_install

# Checkout out specific commit 
git clone --branch develop https://github.com/lattice/quda.git
cd quda
git checkout 35b57df9f

# Create build folder 
mkdir build

# Compile and install QUDA 
cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_CXX_COMPILER=$(which nvc++) -DCMAKE_C_COMPILER=$(which nvcc) -DCMAKE_CUDA_COMPILER=$(which nvcc) -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" -DCMAKE_SKIP_RPATH=TRUE -DQUDA_GPU_ARCH=sm_90 -DQUDA_DIRAC_DEFAULT_OFF=ON -DQUDA_DIRAC_STAGGERED=ON -DQUDA_FORCE_HISQ=ON -DQUDA_FORCE_GAUGE=ON -DCMAKE_INSTALL_PREFIX="/global/home/users/rliu5/dcgm-profiling/milc/quda_install" ..
cmake --build . -j
cmake --build . -j install
```

### 2. Compile and install MILC_QCD 

```
# Checkout specific commit
cd ~/dcgm-profiling/milc
git clone --branch develop https://github.com/milc-qcd/milc_qcd.git
cd milc_qcd
git checkout 13ffa851
```

The following two modified souce files in the N10 benchmark should be copied into the MILC source code.

```
https://gitlab.com/NERSC/N10-benchmarks/lattice-qcd-workflow/-/blob/main/build/ks_imp_rhmc_control.c?ref_type=heads
https://gitlab.com/NERSC/N10-benchmarks/lattice-qcd-workflow/-/blob/main/build/ks_imp_rhmc_update_rhmc.c?ref_type=heads
```

```
cp ks_imp_rhmc_control.c milc/milc_qcd/ks_imp_rhmc/control.c
cp ks_imp_rhmc_update_rhmc.c milc/milc_qcd/ks_imp_rhmc/update_rhmc.c
```

### 3. Modfiy the Makefile


