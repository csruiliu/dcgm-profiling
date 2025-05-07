# This compilation file is still under test

COMPFLAG  = -DPGI
PARAFLAG  = -DMPI  -DOMP
MATHFLAG  = -DUSESCALAPACK -DUNPACKED -DUSEFFTW3 -DHDF5  -DOPENACC -DOMP_TARGET # -DUSEELPA

NVCC=/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-13.2.0/cuda-12.6.0-pe26b3ktosinq7fckw2453oovahnbtui/bin/nvcc
NVCCOPT= -O3 -use_fast_math
CUDALIB= -lcufft -lcublasLt -lcublas -lcudart -lcuda

FCPP    = /usr/bin/cpp  -C   -nostdinc
F90free = mpifort -Mfree -acc -mp=multicore,gpu -gpu=cc80  -Mcudalib=cublas,cufft -Mcuda=lineinfo -traceback -Minfo=mp,acc -gopt -traceback
LINK    = mpifort        -acc -mp=multicore,gpu -gpu=cc80  -Mcudalib=cublas,cufft -Mcuda=lineinfo -Minfo=mp,acc
FOPTS   = -fast -Mfree -Mlarge_arrays
FNOOPTS = $(FOPTS)
MOD_OPT = -module  
INCFLAG = -I #./

C_PARAFLAG  = -DPARA -DMPICH_IGNORE_CXX_SEEK
CC_COMP = mpiCC
C_COMP  = mpicc
C_LINK  = mpicc -lstdc++ # ${CUDALIB} -lstdc++
C_OPTS  = -O3 -fopenmp 
#C_OPTS  = -fast -mp 

REMOVE  = /bin/rm -f

FFTW_DIR=/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-13.2.0/fftw-3.3.10-nxbn4wosmyvhieaqrxfwflhckiys7qus/lib
FFTW_INC=/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-13.2.0/fftw-3.3.10-nxbn4wosmyvhieaqrxfwflhckiys7qus/include
FFTWLIB      = $(FFTW_DIR)/libfftw3.so \
               $(FFTW_DIR)/libfftw3_threads.so \
               $(FFTW_DIR)/libfftw3_mpi.so \
               ${CUDALIB}  -lstdc++
FFTWINCLUDE  = $(FFTW_INC)

HDF5_DIR=/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-13.2.0/hdf5-1.14.3-hd6ccvy36dh7u4tec27ikeghvbddhunk
HDF5_LDIR    =  ${HDF5_DIR}/lib
HDF5LIB      =  $(HDF5_LDIR)/libhdf5hl_fortran.a \
                $(HDF5_LDIR)/libhdf5_hl.a \
                $(HDF5_LDIR)/libhdf5_fortran.a \
                $(HDF5_LDIR)/libhdf5.a -lz -ldl
HDF5INCLUDE  = ${HDF5_DIR}/include


