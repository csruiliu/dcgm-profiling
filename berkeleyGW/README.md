# Performance Profiling for BerkeleyGW

## BerkeleyGW Compilation on Perlmutter 

Following the instructions on the [website](https://gitlab.com/NERSC/N10-benchmarks/berkeleygw-workflow) to compile BerkeleyGW on [Permutter](https://docs.nersc.gov/systems/perlmutter/architecture), except for the module loading--using the following instead.

```bash
module swap PrgEnv-gnu PrgEnv-nvidia/8.5.0
module load cray-hdf5-parallel/1.14.3.1
module load cray-fftw/3.3.10.8
module load cray-libsci/24.07.0
module load python 
```

BerkeleyGW has mutiple example makefiles, choose `perlmutter.nersc.gov-nvhpc-openacc.mk` for Permutter, or just use [pm-arch-gpu.mk](pm-arch-gpu.mk). 


## BerkeleyGW Compilation on Lawrencium

### CPU-only Compilation

Compiling Berkeley on Lawrencium is not as well documented as on Perlemutter. The following configuration has been valiedated.

```bash
module load gcc
module load openmpi
module load intel-oneapi-mkl
module load hdf5/1.14.3
```

Please use [lrc-arch-cpu.mk](lrc-arch-cpu.mk) for compilation.

### GPU-enabled Compilation 

Compiling BerkeleyGW on Lawrencium with GPU support and running the compiled binaries on Einsteinium can be a little tricky due to: (1) module configurations on Lawrencium and Einsteinium, and (2) the login nodes on Lawrencium typically use Intel CPUs, whereas a number of compute nodes on Einsteinium are equipped with AMD CPUs. As a result, building BerkeleyGW may require self-compiling certain third-party libraries with cross-platform compatibility in mind. Unlike Perlmutter, where these issues are handled automatically by the module management, Lawrencium and Einsteinium require additional manual setup. The following instructions have been tested and validated for this environment.

First, we want to see how `nvhpc` module set the environment variables, so that we could copy the variables except for OpenMPI.

```bash
module show nvhpc
```

Then, copy the environment variables setting to `~/.bashrc` except for OpenMPI.

According to the offical [website](http://manual.berkeleygw.org/2.0/compilation), we may need to build and install some libraries such as HDF5, FFTW, LAPACK, ScaLAPACK, OpenMPI.

Alternatively, we can use some pre-built libs for LAPACK, ScaLAPACK, OpenMPI from `/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/nvhpc-23.11-gh5cygvdqksy6mxuy2xgoibowwxi3w7t/Linux_x86_64/23.11/comm_libs/12.3/openmpi4` and `/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/nvhpc-23.11-gh5cygvdqksy6mxuy2xgoibowwxi3w7t/Linux_x86_64/23.11/compilers/lib`

**OpenMPI**

Downloading OpenMPI source codes from [github](https://www-lb.open-mpi.org/software/ompi), taking 4.1.6 as an example,

```bash
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz

tar -xvf openmpi-4.1.6

mv openmpi-4.1.6 openmpi-4.1.6-src

cd openmpi-4.1.6-src

./configure \
  --prefix="$HOME/local/openmpi-4.1.6" \
  CC=nvc \
  CXX=nvc++ \
  FC=nvfortran \
  CFLAGS="-fast -tp x86-64-v3 -Mvect=simd -fPIC" \
  CXXFLAGS="-fast -tp x86-64-v3 -Mvect=simd -fPIC" \
  FCFLAGS="-fast -tp x86-64-v3 -Mvect=simd -fPIC" \
  --enable-static \
  --disable-debug \
  --enable-heterogeneous \
  --with-verbs
  --with-cude=/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/nvhpc-23.11-gh5cygvdqksy6mxuy2xgoibowwxi3w7t/Linux_x86_64/23.11/cuda

make -j

make install

#copy to $SCRATCH
cp -r $HOME/local/openmpi-4.1.6 $SCRATCH/local

# adding the following environmet variable to ~/.bashrc
export OPENMPI_DIR="$SCRATCH/local/openmpi-4.1.6"
export PATH="$OPENMPI_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$OPENMPI_DIR/lib:$LD_LIBRARY_PATH"
export CPATH="$OPENMPI_DIR/include:$CPATH"
```

**OpenBLAS**

Download OpenBLAS-0.3.29 source codes from [github](https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.29/OpenBLAS-0.3.29.tar.gz), taking 0.3.29 as an example,

```bash
wget https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.29/OpenBLAS-0.3.29.tar.gz

tar -xvf OpenBLAS-0.3.29.tar.gz

mv OpenBLAS-0.3.29 OpenBLAS-0.3.29-src

cd OpenBLAS-0.3.29-src

make -j \
  CC=nvc \
  CXX=nvc++ \
  FC=nvfortran \
  CFLAGS="-fast -tp x86-64-v3 -mp=multicore -Mvect=simd -Mflushz -fPIC" \
  CXXFLAGS="-fast -tp x86-64-v3 -mp=multicore -Mvect=simd -Mflushz -fPIC" \
  FCFLAGS="-fast -tp x86-64-v3 -mp=multicore -Mvect=simd -Mflushz -fPIC" \
  USE_OPENMP=1 \
  INTERFACE64=0 \
  DYNAMIC_ARCH=1 \
  USE_THREAD=1


make install PREFIX=$HOME/local/OpenBLAS-0.3.29

#copy to $SCRATCH
cp -r $HOME/local/openblas-0.3.29 $SCRATCH/local

# adding the following environmet variable to ~/.bashrc
export OPENBLAS_DIR="$SCRATCH/local/openblas-0.3.29"
export PATH="$OPENBLAS_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$OPENBLAS_DIR/lib:$LD_LIBRARY_PATH"
export CPATH="$OPENBLAS_DIR/include:$CPATH"
```

**HDF5**

Downloading HDF5 source codes from [github](https://github.com/HDFGroup/hdf5/releases/download/hdf5-1_14_3/hdf5-1_14_3.tar.gz), taking 1.14.3 as an example,

```bash
wget https://github.com/HDFGroup/hdf5/releases/download/hdf5-1_14_3/hdf5-1_14_3.tar.gz

tar -xvf hdf5-1_14_3.tar.gz

mv hdfsrc hdf5-1.14.3-src

cd hdf5-1.14.3-src

# run ./configure --help for more details
# rename the uncompressed folder if its name is hdf5-1.14.3
./configure \
  --prefix="$HOME/local/hdf5-1.14.3" \
  CC=mpicc \
  CXX=mpicxx \
  FC=mpifort \
  CFLAGS="-tp x86-64-v3 -fast -Mvect=simd -mp=multicore -Mflushz -fPIC" \
  CXXFLAGS="-tp x86-64-v3 -fast -Mvect=simd -mp=multicore -Mflushz -fPIC" \
  FCFLAGS="-tp x86-64-v3 -fast -Mvect=simd -mp=multicore -Mflushz -fPIC" \
  --enable-fortran \
  --enable-shared \
  --enable-parallel 

# run make as as many cores as possible
make -j
# install all files to prefix path, which is $HOME/local/hdf5-1.14.3
make install 

#copy to $SCRATCH
cp -r $HOME/local/hdf5-1.14.3 $SCRATCH/local

# adding the following environmet variable to ~/.bashrc
export HDF5_DIR="$SCRATCH/local/hdf5-1.14.3"
export PATH="$HDF5_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$HDF5_DIR/lib:$LD_LIBRARY_PATH"
export CPATH="$HDF5_DIR/include:$CPATH"
```

**FFTW**

Downloading FFTW source codes from their [website](https://www.fftw.org/download.html), taking 3.3.10 as an example,

```bash
wget https://www.fftw.org/fftw-3.3.10.tar.gz

tar -xvf fftw-3.3.10.tar.gz

mv fftw-3.3.10 fftw-3.3.10-src

cd fftw-3.3.10-src

# rename the uncompressed folder if its name is fftw-3.3.10
./configure \
  --prefix="$HOME/local/fftw-3.3.10" \
  CC=mpicc \
  CXX=mpicxx \
  FC=mpifort \
  CFLAGS="-tp x86-64-v3 -fast -Mvect=simd -mp=multicore -Mflushz -fPIC" \
  CXXFLAGS="-tp x86-64-v3 -fast -Mvect=simd -mp=multicore -Mflushz -fPIC" \
  FCFLAGS="-tp x86-64-v3 -fast -Mvect=simd -mp=multicore -Mflushz -fPIC" \
  --enable-shared \
  --enable-openmp \
  --enable-threads \
  --enable-mpi \
  --enable-sse2 \
  --enable-avx \
  --enable-avx2 \
  --enable-avx-128-fma \
  --enable-generic-simd128 \
  --enable-generic-simd256

make -j

make install

#copy to $SCRATCH
cp -r $HOME/local/fftw-3.3.10 $SCRATCH/local

# adding the following environment variable to ~/.bashrc
export FFTW_DIR="$SCRATCH/local/fftw-3.3.10"
export PATH="$FFTW_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$FFTW_DIR/lib:$LD_LIBRARY_PATH"
export CPATH="$FFTW_DIR/include:$CPATH"
```

**LAPACK**

Downloading LAPACK source codes from their [website](https://www.netlib.org/lapack), taking 3.12.1 as an example,

```bash
wget https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.1.tar.gz

tar -xvf v3.12.1.tar.gz

cd <uncompressed-folder>

mkdir build

cd build

cmake \
  -DCMAKE_Fortran_COMPILER=mpifort \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_INSTALL_PREFIX="$HOME/local/lapack-3.12.1" \
  -DCMAKE_Fortran_FLAGS="-tp x86-64-v3 -fast -Mvect=simd -mp=multicore -Mflushz -fPIC" \
  -DCMAKE_C_FLAGS="-tp x86-64-v3 -fast -Mvect=simd -mp=multicore -Mflushz -fPIC" \
  -DCMAKE_CXX_FLAGS="-tp x86-64-v3 -fast -Mvect=simd -mp=multicore -Mflushz -fPIC" \
  -DBUILD_SHARED_LIBS=ON \
  -DUSE_OPTIMIZED_BLAS=ON \
  -DBLAS_LIBRARIES="$OPENBLAS_DIR/lib/libopenblas.so" \
  -DBUILD_INDEX64=OFF \
  -DLAPACKE=ON \
  -DCBLAS=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-L$NVHPC_ROOT/compilers/lib -lnvf" ..

make -j

make install

#copy to $SCRATCH
cp -r $HOME/local/lapack-3.12.1 $SCRATCH/local

# adding the following environment variable to ~/.bashrc
export LAPACK_DIR="$SCRATCH/local/lapack-3.12.1"
export LD_LIBRARY_PATH="$LAPACK_DIR/lib64:$LD_LIBRARY_PATH"
```

**ScaLAPACK**

Downloading ScaLAPACK source codes from their [website](https://www.netlib.org/scalapack/), taking 2.2.2 as an example,

```bash
wget https://github.com/Reference-ScaLAPACK/scalapack/archive/refs/tags/v2.2.2.tar.gz

tar -xvf v2.2.2.tar.gz

mv scalapack-2.2.2 scalapack-2.2.2-src

mkdir build

cd build

cmake \
  -DCMAKE_Fortran_COMPILER=mpifort \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_INSTALL_PREFIX="$HOME/local/scalapack-2.2.2" \
  -DCMAKE_Fortran_FLAGS="-tp x86-64-v3 -fast -Mvect=simd -mp=multicore -Mflushz -fPIC" \
  -DCMAKE_C_FLAGS="-tp x86-64-v3 -fast -Mvect=simd -mp=multicore -Mflushz -fPIC" \
  -DCMAKE_CXX_FLAGS="-tp x86-64-v3 -fast -Mvect=simd -mp=multicore -Mflushz -fPIC" \
  -DBUILD_SHARED_LIBS=ON \
  -DBLAS_LIBRARIES="$OPENBLAS_DIR/lib/libopenblas.so" \
  -DLAPACK_LIBRARIES="$LAPACK_DIR/lib64/liblapack.so" ..

make -j

make install

#copy to $SCRATCH
cp -r $HOME/local/scalapack-2.2.2 $SCRATCH/local

# adding the following environment variable to ~/.bashrc
export SCALAPACK_DIR="$SCRATCH/local/scalapack-2.2.2"
export LD_LIBRARY_PATH="$SCALAPACK_DIR/lib:$LD_LIBRARY_PATH"
```

Now, we can use [lrc-arch-gpu.mk](lrc-arch-gpu.mk) for compilation. The file is revised from the compile file for Perlmutter.

The comprehensive bashrc is shown as follows for reference.

```bash
export NVHPC_DIR="/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/nvhpc-23.11-gh5cygvdqksy6mxuy2xgoibowwxi3w7t/Linux_x86_64/23.11"

export CC="$NVHPC_DIR/compilers/bin/nvc"
export CXX="$NVHPC_DIR/compilers/bin/nvc++"
export FC="$NVHPC_DIR/compilers/bin/nvfortran"
export F90="$NVHPC_DIR/compilers/bin/nvfortran"
export F77="$NVHPC_DIR/compilers/bin/nvfortran"

export CPP="cpp" 

export CUDA_DIR="$NVHPC_DIR/cuda"
export NVHPC_COMPILER_EXTRA_DIR="$NVHPC_DIR/compilers/extras/qd"
export NCCL_DIR="$NVHPC_DIR/comm_libs/nccl"
export NVSHMEM_DIR="$NVHPC_DIR/comm_libs/nvshmem"
export MATHLIB_DIR="$NVHPC_DIR/math_libs"

#export OPENBLAS_DIR="$SCRATCH/local/openblas-0.3.29-self"
#export OPENMPI_DIR="$SCRATCH/local/openmpi-4.1.6-self"
#export HDF5_DIR="$SCRATCH/local/hdf5-1.14.3-self"
#export LAPACK_DIR="$SCRATCH/local/lapack-3.12.1-self"
#export FFTW_DIR="$SCRATCH/local/fftw-3.3.10-self"
#export SCALAPACK_DIR="$SCRATCH/local/scalapack-2.2.2-self"

export NVHPC_COMPILER_DIR="$NVHPC_DIR/compilers"
export OPENMPI_DIR="$NVHPC_DIR/comm_libs/12.3/openmpi4/openmpi-4.1.5"
export HDF5_DIR="$SCRATCH/local/hdf5-1.14.3-sys"
export SCALAPACK_DIR="$SCRATCH/local/scalapack-2.2.2-sys"
export FFTW_DIR="$SCRATCH/local/fftw-3.3.10-sys"

export PATH=$NVHPC_COMPILER_EXTRA_DIR/bin:$NVHPC_COMPILER_DIR/bin:$CUDA_DIR/bin:$OPENMPI_DIR/bin:$HDF5_DIR/bin:$FFTW_DIR/bin:$OPENBLAS_DIR/bin:$PATH

export LD_LIBRARY_PATH=$NVSHMEM_DIR/lib:$NCCL_DIR/lib:$MATHLIB_DIR/lib64:$NVHPC_COMPILER_DIR/lib:$NVHPC_COMPILER_EXTRA_DIR/lib:$CUDA_DIR/extras/CUPTI/lib64:$CUDA_DIR/lib64:$SCALAPACK_DIR/lib:$HDF5_DIR/lib:$FFTW_DIR/lib:$OPENMPI_DIR/lib:$OPENBLAS_DIR/lib:$LAPACK_DIR/lib64:$LD_LIBRARY_PATH

export CPATH=$NVHPC_DIR/compilers/extras/qd/include/qd:$NVSHMEM_DIR/include:$NVHPC_COMPILER_DIR/include:$NCCL_DIR/include:$MATHLIB_DIR/include:$OPENMPI_DIR/include:$OPENBLAS_DIR/include:$HDF5_DIR/include:$FFTW_DIR/include:$LAPACK_DIR/include:$CPATH

```

## Berkeley Workflow

Check out the BGW workflow repo from [N10 Benchmark](https://gitlab.com/NERSC/N10-benchmarks/berkeleygw-workflow). 

## Environment Paths

Make sure all the BerkeleyGW-related paths are correct in `scripts/site_path_config.sh`. For instance, `BerkeleyGW` folder, which contains `BerkeleyGW-master` and `berkeleygw-workflow`, should be one level above the current scripts directory.

## DCGM

We use `dcgmi dmon` command to profile the performance under various metrics.

`dcgm_delay` is the parameter for `dcgmi dmon -d` command. It is an integer and in milliseconds. It represents how often to query results from DCGM. [default = 1000 millisecond, Minimum value = 1 millisecond.]