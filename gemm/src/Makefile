#CC = nvc++
CC = CC
CFLAGS = --std c++11 -O3 -DUSE_CUBLAS -I$(CUDA_HOME)/include
LIBS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64 -lcublas
CFLAGS += -DUSE_MPI

all: gemm.x gemm_lt.x gemm_mpi.x

gemm.x: gemm.cpp calc_gemm.cpp Makefile
	$(CC) $(CFLAGS) -o gemm.x gemm.cpp $(LIBS)

gemm_lt.x: gemm_lt.cpp calc_gemm_lt.cpp Makefile
	$(CC) $(CFLAGS) -DUSE_CUBLASLT -o gemm_lt.x gemm_lt.cpp $(LIBS) -lcublasLt

gemm_mpi.x: gemm_mpi.cpp calc_gemm_mpi.cpp Makefile
	$(CC) $(CFLAGS) -DUSE_CUBLASLT -o gemm_mpi.x gemm_mpi.cpp $(LIBS) -lcublasLt

#EXE = gemm.x

#$(EXE): gemm.cpp calc_gemm.cpp Makefile
#	$(CC) $(CFLAGS) -o $(EXE) gemm.cpp $(LIBS)

clean:
	rm -f *.x *~

