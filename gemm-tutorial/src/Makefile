#CC = nvc++
CC = CC
CFLAGS = --std c++11 -O3 -DUSE_CUBLAS -I$(CUDA_ROOT)/include
LIBS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64 -lcublas
CFLAGS += -DUSE_MPI

all: gemm.x gemm_lt.x gpu_util_eval_init.x gpu_util_eval_io.x

gemm.x: gemm.cpp calc_gemm.cpp Makefile
	$(CC) $(CFLAGS) -o gemm.x gemm.cpp $(LIBS)

gemm_lt.x: gemm_lt.cpp calc_gemm_lt.cpp Makefile
	$(CC) $(CFLAGS) -DUSE_CUBLASLT -o gemm_lt.x gemm_lt.cpp $(LIBS) -lcublasLt

gpu_util_eval_init.x: gpu_util_eval_init.cpp Makefile
	$(CC) $(CFLAGS) -o gpu_util_eval_init.x gpu_util_eval_init.cpp

gpu_util_eval_io.x: gpu_util_eval_io.cpp Makefile
	$(CC) $(CFLAGS) -o gpu_util_eval_io.x gpu_util_eval_io.cpp

#EXE = gemm.x

#$(EXE): gemm.cpp calc_gemm.cpp Makefile
#	$(CC) $(CFLAGS) -o $(EXE) gemm.cpp $(LIBS)

clean:
	rm -f *.x *~

