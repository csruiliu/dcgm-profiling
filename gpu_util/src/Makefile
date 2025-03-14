#CC = nvc++
CC = CC
CFLAGS = --std c++11 -O3 -DUSE_CUBLAS -I$(CUDA_ROOT)/include

all: gpu_util_eval_init.x gpu_util_eval_io.x

gpu_util_eval_init.x: gpu_util_eval_init.cpp Makefile
	$(CC) $(CFLAGS) -o gpu_util_eval_init.x gpu_util_eval_init.cpp

gpu_util_eval_io.x: gpu_util_eval_io.cpp Makefile
	$(CC) $(CFLAGS) -o gpu_util_eval_io.x gpu_util_eval_io.cpp

clean:
	rm -f *.x *~

