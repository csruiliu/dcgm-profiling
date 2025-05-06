#CC = nvc++
CC = CC
CFLAGS = --std c++11 -O3 -DUSE_CUBLAS -I$(CUDA_ROOT)/include

all: eval_init.x eval_io_pcie.x eval_io_nvlink.x

eval_init.x: eval_init.cpp Makefile
	$(CC) $(CFLAGS) -o eval_init.x eval_init.cpp

eval_io_pcie.x: eval_io_pcie.cpp Makefile
	$(CC) $(CFLAGS) -o eval_io_pcie.x eval_io_pcie.cpp

eval_io_nvlink.x: eval_io_nvlink.cpp Makefile
	$(CC) $(CFLAGS) -o eval_io_nvlinke.x eval_io_nvlink.cpp -lnccl

clean:
	rm -f *.x *~

