# Performance Profiling for BerkeleyGW

## BerkeleyGW Compilation on Perlmutter 

Following the instructions on the [website](https://gitlab.com/NERSC/N10-benchmarks/berkeleygw-workflow) to compile BerkeleyGW on [Permutter](https://docs.nersc.gov/systems/perlmutter/architecture), except for the module loading--using the following instead.

```bash
module swap PrgEnv-gnu PrgEnv-nvidia
module load cray-hdf5-parallel/1.12.2.9
module load cray-fftw/3.3.10.6
module load cray-libsci/23.12.5
module load python 
```

BerkeleyGW has mutiple example makefiles, choose `perlmutter.nersc.gov-nvhpc-openacc.mk` for Permutter

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

Compiling Berkeley on Lawrencium with GPU support can be challenging due to the module configuration on Lawrencium. It may require some third-party compilers and libraries, which are supported by the modules in Perlmutter. The following instructions have been validated.

First, we need to load the `nvhpc` module.

```bash
module load nvhpc
```

According to the offical [website](http://manual.berkeleygw.org/2.0/compilation), we may need to build and install some libraries such as HDF5, FFTW, ScaLAPACK, OpenMPI

**OpenMPI**

Downloading OpenMPI source codes from [github](https://www-lb.open-mpi.org/software/ompi), taking 4.1.6 as an example,

```bash
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz

tar -xvf openmpi-4.1.6

cd 

./configure --prefix=$HOME/local/openmpi

make -j all

make install
```

**HDF5**

Downloading HDF5 source codes from [github](https://github.com/HDFGroup/hdf5/tags), taking 1.14.3 as an example,

```bash
wget https://github.com/HDFGroup/hdf5/releases/download/hdf5-1_14_3/hdf5-1_14_3.tar.gz

tar -xvf hdf5-1_14_3.tar.gz

# rename the uncompressed fold
mv <uncompressed-fold> fftw-3.3.10-src

cd fftw-3.3.10-src

# run ./configure --help for more details
./configure --prefix=$HOME/local/hdf5-1.14.3 CC=mpicc FC=mpifort CFLAGS="-fPIC" FCFLAGS="-fpic" --enable-fortran --enable-shared --enable-parallel
# run make as as many cores as possible
make -j
# install all files to prefix path, which is $HOME/local/hdf5-1.14.3
make install 

# adding the following environmet variable to ~/.bashrc
export HDF5_DIR="$HOME/local/hdf5-1.14.3"
export PATH="$HDF5_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$HDF5_DIR/lib:$LD_LIBRARY_PATH"
export CPATH="$HDF5_DIR/include:$CPATH"
```

**FFTW**

Downloading FFTW source codes from their [website](https://www.fftw.org/download.html), taking 3.3.10 as an example,

```bash
wget https://www.fftw.org/fftw-3.3.10.tar.gz

tar -xvf fftw-3.3.10.tar.gz

# rename the uncompressed fold
mv fftw-3.3.10 fftw-3.3.10-src

./configure --prefix=$HOME/local/fftw-3.3.10 CC=mpicc FC=mpifort --enable-shared --enable-openmp --enable-threads --enable-mpi

make -j

make install

# adding the following environment variable to ~/.bashrc
export FFTW_DIR="$HOME/local/fftw-3.3.10"
export PATH="$FFTW_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$FFTW_DIR/lib:$LD_LIBRARY_PATH"
export CPATH="$FFTW_DIR/include:$CPATH"
```

**LAPACK**

Downloading ScaLAPACK source codes from their [website](https://www.netlib.org/lapack), taking 3.12.1 as an example,

```bash
wget https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.1.tar.gz

tar -xvf v3.12.1.tar.gz

cd <uncompressed-folder>

mkdir build

cd build

cmake -DCMAKE_INSTALL_PREFIX=$HOME/local/lapack-3.12.1 -DBUILD_SHARED_LIBS=ON ..

make -j

make install

# adding the following environment variable to ~/.bashrc
export LAPACK_DIR="$HOME/local/lapack-3.12.1"
export LD_LIBRARY_PATH="$LAPACK_DIR/lib64:$LD_LIBRARY_PATH"
```

**ScaLAPACK**

Downloading ScaLAPACK source codes from their [website](https://www.netlib.org/scalapack/), taking 2.2.2 as an example,

```bash
wget https://github.com/Reference-ScaLAPACK/scalapack/archive/refs/tags/v2.2.2.tar.gz

tar -xvf v2.2.2.tar.gz

cd <uncompressed-folder>

cmake -DCMAKE_INSTALL_PREFIX=$HOME/local/scalapack-2.2.2 -DBUILD_SHARED_LIBS=ON ..

make -j

make install
```



Now, we can use [lrc-arch-gpu.mk](lrc-arch-gpu.mk) for compilation. The file is revised from the compile file for Perlmutter.


## Berkeley Workflow

## Environment Paths

Make sure all the BerkeleyGW-related paths are correct in `scripts/site_path_config.sh`. For instance, `BerkeleyGW` folder, which contains `BerkeleyGW-master` and `berkeleygw-workflow`, should be one level above the current scripts directory.


## DCGM

We use `dcgmi dmon` command to profile the performance under various metrics.

`dcgm_delay` is the parameter for `dcgmi dmon -d` command. It is an integer and in milliseconds. It represents how often to query results from DCGM. [default = 1000 millisecond, Minimum value = 1 millisecond.]