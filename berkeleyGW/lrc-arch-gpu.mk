# arch.mk for BerkeleyGW codes
#
# suitable for Perlmutter
#
# MDB
# 2021, Perlmutter@NERSC
# nvidia compiler
# 
# Do:
#
# module swap PrgEnv-gnu PrgEnv-nvhpc ; module load cray-hdf5-parallel ; module load cray-fftw ; module load cray-libsci ; module load python
#
#
COMPFLAG  = -DPGI
PARAFLAG  = -DMPI  -DOMP
MATHFLAG  = -DUSESCALAPACK -DUNPACKED -DUSEFFTW3 -DHDF5  -DOPENACC -DOMP_TARGET # -DUSEELPA
# DEBUGFLAG = -DDEBUG -DNVTX
#

NVCC=nvcc 
NVCCOPT= -O2 -use_fast_math
# CUDALIB=-L$(CUDA_DIR)/lib64/  -lcufft -lcublas  -lcudart  -lcuda  -lnvToolsExt
CUDALIB= -lcufft -lcublasLt -lcublas -lcudart -lcuda

FCPP    = /usr/bin/cpp  -C   -nostdinc   #  -C  -P  -E -ansi  -nostdinc  /usr/bin/cpp
# F90free = ftn -Mfree -acc -mp  -Mcudalib=cublas,cufft -Mcuda=lineinfo -traceback -Minfo=mp
# LINK    = ftn        -acc -mp  -Mcudalib=cublas,cufft -Minfo=mp
# FOPTS   = -fast -Mfree -Mlarge_arrays 
F90free = mpifort -Mfree -acc -mp=multicore,gpu -gpu=cc80  -Mcudalib=cublas,cufft -Mcuda=lineinfo -traceback -Minfo=mp,acc -gopt -traceback -tp=x86-64-v3 -O2
LINK    = mpifort        -acc -mp=multicore,gpu -gpu=cc80  -Mcudalib=cublas,cufft -Mcuda=lineinfo -Minfo=mp,acc -tp=x86-64-v3 -O2
FOPTS   = -O2 -Mfree -Mlarge_arrays -tp=x86-64-v3
FNOOPTS = $(FOPTS)
MOD_OPT = -module  
INCFLAG = -I #./

C_PARAFLAG  = -DPARA -DMPICH_IGNORE_CXX_SEEK
CC_COMP = mpicxx
C_COMP  = mpicc
C_LINK  = mpicxx -lstdc++ # ${CUDALIB} -lstdc++
C_OPTS  = -O2 -mp -tp=x86-64-v3
C_DEBUGFLAG =

REMOVE  = /bin/rm -f

FFTW_DIR=/global/home/users/rliu5/local/fftw-3.3.10/lib
FFTWLIB      = $(FFTW_DIR)/libfftw3.so \
               $(FFTW_DIR)/libfftw3_threads.so \
               $(FFTW_DIR)/libfftw3_omp.so \
               ${CUDALIB}  -lstdc++
FFTWINCLUDE  =/global/home/users/rliu5/local/fftw-3.3.10/include
PERFORMANCE  = 

SCALAPACK_DIR=/global/home/users/rliu5/local/scalapack-2.2.2
SCALAPACKLIB = -L${SCALAPACK_DIR}/lib -lscalapack

LAPACK_DIR=/global/home/users/rliu5/local/lapack-3.12.1
LAPACKLIB = -L${LAPACK_DIR}/lib64 -llapack -lblas
#LAPACKLIB = -llapack -lblas 

# # HDF5_LDIR    =  ${HDF5_DIR}/lib/
# HDF5_LDIR    =  /global/homes/m/mdelben/LIBS_local/hdf5-1.10.5/lib/
# HDF5LIB      =  $(HDF5_LDIR)/libhdf5hl_fortran.a \
#                 $(HDF5_LDIR)/libhdf5_hl.a \
#                 $(HDF5_LDIR)/libhdf5_fortran.a \
#                 $(HDF5_LDIR)/libhdf5.a -lz -ldl
# #HDF5INCLUDE  = ${HDF5_DIR}/include/
# HDF5INCLUDE  = /global/homes/m/mdelben/LIBS_local/hdf5-1.10.5/include/

HDF5_DIR=/global/home/users/rliu5/local/hdf5-1.14.3
HDF5_LDIR    =  ${HDF5_DIR}/lib/
# HDF5_LDIR    =  /pscratch/home/mdelben/LIBS_local/hdf5-1.10.5/lib/
HDF5LIB      =  $(HDF5_LDIR)/libhdf5hl_fortran.a \
                $(HDF5_LDIR)/libhdf5_hl.a \
                $(HDF5_LDIR)/libhdf5_fortran.a \
                $(HDF5_LDIR)/libhdf5.a -lz -ldl
HDF5INCLUDE  = ${HDF5_DIR}/include/
# HDF5INCLUDE  = /pscratch/home/mdelben/LIBS_local/hdf5-1.10.5/include/


# ELPALIB = PATH_TO_ELPA_DIR/lib/libelpa.a -lstdc++
# ELPAINCLUDE = PATH_TO_ELPA_DIR/include/elpa-2021.11.001/modules/
