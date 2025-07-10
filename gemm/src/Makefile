NVCC = nvcc
CC = CC
CFLAGS = --std c++14 -O3 -DUSE_CUBLAS -I$(CUDA_HOME)/include
LIBS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64 -lcublas
CFLAGS += -DUSE_MPI

all: gemm.x gemm_lt.x gemm_mpi.x gemm_bursty.x gemm_bursty_stream.x gemm_interleave_sync.x gemm_interleave_async.x

gemm.x: gemm.cpp calc_gemm.cpp Makefile
	$(CC) $(CFLAGS) -o gemm.x gemm.cpp $(LIBS)

gemm_lt.x: gemm_lt.cpp calc_gemm_lt.cpp Makefile
	$(CC) $(CFLAGS) -DUSE_CUBLASLT -o gemm_lt.x gemm_lt.cpp $(LIBS) -lcublasLt

gemm_mpi.x: gemm_mpi.cpp calc_gemm_mpi.cpp Makefile
	$(CC) $(CFLAGS) -DUSE_CUBLASLT -o gemm_mpi.x gemm_mpi.cpp $(LIBS) -lcublasLt

gemm_bursty.x: gemm_bursty.cpp Makefile
	$(CC) $(CFLAGS) -o gemm_bursty.x gemm_bursty.cpp $(LIBS)

gemm_bursty_stream.x: gemm_bursty_stream.cu Makefile
	$(NVCC) $(CFLAGS) -o gemm_bursty_stream.x gemm_bursty_stream.cu $(LIBS)

gemm_interleave_sync.x: gemm_interleave_sync.cpp Makefile
	$(CC) $(CFLAGS) -o gemm_interleave_sync.x gemm_interleave_sync.cpp $(LIBS)

gemm_interleave_async.x: gemm_interleave_async.cpp Makefile
	$(CC) $(CFLAGS) -o gemm_interleave_async.x gemm_interleave_async.cpp $(LIBS)

clean:
	rm -f *.x *~

