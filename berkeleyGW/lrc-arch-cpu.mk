# arch.mk for BerkeleyGW codes
#
# MDB
# 2025, LBL: GNU+MKL
# module load gcc ; module load openmpi  ; module load intel-oneapi-mkl ; module load hdf5/1.14.3

COMPFLAG  = -DGNU
PARAFLAG  = -DMPI  -DOMP
MATHFLAG  = -DUSESCALAPACK  -DUSEFFTW3  -DUNPACKED  -DHDF5  # -DUSEELPA

FCPP    = cpp -C -nostdinc
F90free = mpifort
LINK    = mpifort
FOPTS   = -fallow-argument-mismatch -O3 -g  -fbounds-check -fbacktrace -Wall -fopenmp -ffree-form -ftree-vectorize -ffree-line-length-512
FNOOPTS = -fallow-argument-mismatch -O1 -g  -fbounds-check -fbacktrace -Wall -fopenmp -ffree-form -ffree-line-length-512
MOD_OPT = -J 
INCFLAG = -I

C_PARAFLAG = -DPARA
CC_COMP = mpiCC
C_COMP  = mpicc
C_LINK  = mpiCC
C_OPTS  = -O3 -fopenmp 
C_DEBUGFLAG =

REMOVE  = /bin/rm -f

# Math Libraries
#
FFTWLIB      = ${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group \
               ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a \
               ${MKLROOT}/lib/intel64/libmkl_tbb_thread.a \
               ${MKLROOT}/lib/intel64/libmkl_core.a \
               ${MKLROOT}/lib/intel64/libmkl_blacs_openmpi_lp64.a -Wl,--end-group \
              -L${TBBROOT}/lib/intel64/gcc4.8 -ltbb -lstdc++ -lpthread -lm -ldl -lz
FFTWINCLUDE  = ${MKLROOT}/include -I${MKLROOT}/include/fftw

LAPACKLIB    = ${MKLROOT}/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group \
               ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a \
               ${MKLROOT}/lib/intel64/libmkl_tbb_thread.a \
               ${MKLROOT}/lib/intel64/libmkl_core.a \
               ${MKLROOT}/lib/intel64/libmkl_blacs_openmpi_lp64.a -Wl,--end-group \
              -L${TBBROOT}/lib/intel64/gcc4.8 -ltbb -lstdc++ -lpthread -lm -ldl -lz
              
SCALAPACKLIB = 

HDF5DIR      = /global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-11.4.0/hdf5-1.14.3-srbevcutioii7lti5mmu2kwy3ve324mg/
# HDF5LIB      = -L$(HDF5DIR)/lib/ -lhdf5hl_fortran -lhdf5_hl -lhdf5_fortran -lhdf5 -lz
HDF5LIB      = $(HDF5DIR)/lib/libhdf5_hl_fortran.a \
               $(HDF5DIR)/lib/libhdf5_fortran.a \
               $(HDF5DIR)/lib/libhdf5_f90cstub.a \
               $(HDF5DIR)/lib/libhdf5_hl_f90cstub.a \
               $(HDF5DIR)/lib/libhdf5_hl.a \
               $(HDF5DIR)/lib/libhdf5_tools.a \
               $(HDF5DIR)/lib/libhdf5.a  -lm -ldl -lz 
HDF5INCLUDE  = $(HDF5DIR)/include

# ELPALIB = 
# ELPAINCLUDE = 

TESTSCRIPT = sbatch hbar.scr

