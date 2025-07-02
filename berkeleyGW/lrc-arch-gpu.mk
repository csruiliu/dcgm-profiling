# arch.mk for BerkeleyGW codes
#
# Do:
# module swap PrgEnv-gnu PrgEnv-nvidia ; module load cray-hdf5-parallel ; module load cray-fftw ; module load cray-libsci ; module load python
#
#
COMPFLAG  = -DNVHPC -DNVHPC_API -DNVIDIA_GPU
PARAFLAG  = -DMPI  -DOMP
MATHFLAG  = -DUSESCALAPACK -DUNPACKED -DUSEFFTW3 -DHDF5 -DOPENACC -DOMP_TARGET # -DUSEPRIMME -DUSEELPA # -DOMP_TARGET
# DEBUGFLAG = -DDEBUG -DNVTX
#
NVCC=nvcc 
NVCCOPT= -O3 -use_fast_math
# CUDALIB= -lcufft -lcublasLt -lcublas -lcudart -lcuda -lnvToolsExt
FCPP    = /usr/bin/cpp  -C   -nostdinc   #  -C  -P  -E -ansi  -nostdinc  /usr/bin/cpp
F90free = mpifort -Mfree -acc -mp=multicore,gpu -gpu=cc80  -cudalib=cublas,cufft -traceback -Minfo=all,mp,accel -gopt -traceback -tp=x86-64-v3
LINK    = mpifort        -acc -mp=multicore,gpu -gpu=cc80  -cudalib=cublas,cufft -Minfo=mp,accel -tp=x86-64-v3# -lnvToolsExt  
FOPTS   = -fast -Mfree -Mlarge_arrays -tp=x86-64-v3
# F90free = ftn -Mfree -acc=sync,wait -mp=multicore,gpu -gpu=cc80  -cudalib=cublas,cufft -traceback -Minfo=all,mp,acc -gopt -traceback
# LINK    = ftn        -acc=sync,wait -mp=multicore,gpu -gpu=cc80  -cudalib=cublas,cufft -Minfo=mp,acc # -lnvToolsExt
# FOPTS   = -O0 -Mfree # -Mlarge_arrays # FF epsilon hangs
FNOOPTS = $(FOPTS)
MOD_OPT = -module  
INCFLAG = -I #./
C_PARAFLAG  = -DPARA -DMPICH_IGNORE_CXX_SEEK
CC_COMP = mpicxx
C_COMP  = mpicc
C_LINK  = mpicc -lstdc++ # ${CUDALIB} -lstdc++
C_OPTS  = -fast -mp -tp=x86-64-v3
C_DEBUGFLAG =
REMOVE  = /bin/rm -f

FFTW_DIR=/global/scratch/users/rliu5/local/fftw-3.3.10
FFTW_LDIR=$(FFTW_DIR)/lib
FFTWLIB      = $(FFTW_LDIR)/libfftw3.so \
	       $(FFTW_LDIR)/libfftw3_threads.so \
               $(FFTW_LDIR)/libfftw3_omp.so \
               ${CUDALIB}  -lstdc++
FFTWINCLUDE  =$(FFTW_DIR)/include
PERFORMANCE  = 

SCALAPACK_DIR=/global/scratch/users/rliu5/local/scalapack-2.2.2
SCALAPACKLIB = -L${SCALAPACK_DIR}/lib -lscalapack

NVHPC_DIR=/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-8.5.0/nvhpc-23.11-gh5cygvdqksy6mxuy2xgoibowwxi3w7t/Linux_x86_64/23.11
LAPACK_DIR=$(NVHPC_DIR)/compilers/lib
LAPACKLIB = -llapack -lblas 

HDF5_DIR=/global/scratch/users/rliu5/local/hdf5-1.14.3
HDF5_LDIR    =  ${HDF5_DIR}/lib/
HDF5LIB      = -L$(HDF5_LDIR)/ -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -lz -ldl
HDF5INCLUDE  = ${HDF5_DIR}/include/
