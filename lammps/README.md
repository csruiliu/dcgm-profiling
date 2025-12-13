# LAMMPS

The following script works for Perlmutter and LRC

```bash
#!/bin/bash

# for Perlmutter, you may check cmake versions for different machine
ml cmake

BASE_DIRECTORY=<BASE_PATH_FOR_LAMMPS>
SOURCE_DIRECTORY=${BASE_DIRECTORY}/lammps
BUILD_DIRECTORY=${BASE_DIRECTORY}/build_lammps
INSTALL_DIRECTORY=${BASE_DIRECTORY}/install_lammps

cd $BASE_DIRECTORY
if [ ! -d ${SOURCE_DIRECTORY} ]
then
  git clone https://github.com/lammps/lammps
fi

mkdir -p ${BUILD_DIRECTORY}
mkdir -p ${INSTALL_DIRECTORY}

cd ${BUILD_DIRECTORY}

cmake ${SOURCE_DIRECTORY}/cmake \
    -D BUILD_SHARED_LIBS=OFF \
    -D CMAKE_INSTALL_PREFIX=${INSTALL_DIRECTORY} \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_VERBOSE_MAKEFILE=ON \
    -D CMAKE_CXX_COMPILER=${SOURCE_DIRECTORY}/lib/kokkos/bin/nvcc_wrapper \
    -D MPI_C_COMPILER=`which cc` \
    -D MPI_CXX_COMPILER=`which CC` \
    -D BUILD_MPI=yes \
    -D FFT=KISS \
    -D FFT_KOKKOS=CUFFT \
    -D PKG_ML-SNAP=yes \
    -D PKG_KOKKOS=yes \
    # for single precision
    -D KOKKOS_PREC=single \
    -D Kokkos_ENABLE_CUDA=yes \
    -D Kokkos_ENABLE_SERIAL=yes \
    -D Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC=OFF \
    -D Kokkos_ARCH_AMPERE80=ON && \
cmake --build ${BUILD_DIRECTORY} --target all -- -j4 && \
cmake --build ${BUILD_DIRECTORY} --target install -- -j4

```

For different GPU, we can change the "-D Kokkos_ARCH_AMPERE80=ON" to:

```bash
A40: -D Kokkos_ARCH_AMPERE86=ON
H100: -D Kokkos_ARCH_HOPPER90=ON
```
