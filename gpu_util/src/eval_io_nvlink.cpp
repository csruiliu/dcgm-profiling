#include <stdio.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <nccl.h>

#include <cuda_runtime.h>

// Uncomment for pageable host memory allocation
#define USE_PINNED_MEMORY
// Uncomment to use std::chrono for timing:
#define TIMING_CUDA_EVENTS

// Size for large data transfer (32 GB)
const size_t SIZE = 32L * 1024 * 1024 * 1024;
// Size for allreduce operation (1 million doubles)
const size_t ALLREDUCE_SIZE = 1024 * 1024;

void *host_alloc(size_t size) {
#ifdef USE_PINNED_MEMORY
    void *ptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    return (err == cudaSuccess) ? ptr : nullptr;
#else
    return malloc(size);
#endif
}

void host_free(void *ptr) {
#ifdef USE_PINNED_MEMORY
    cudaFreeHost(ptr);
#else
    free(ptr);
#endif
}

struct Timer {
#ifdef TIMING_CUDA_EVENTS
    cudaEvent_t start, stop;

    Timer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~Timer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void record_start() { cudaEventRecord(start); }
    void record_stop() { cudaEventRecord(stop); }
    double elapsed()
    {
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        return ms / 1000.0;
    }
#else
    std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;

    void record_start() { start = std::chrono::high_resolution_clock::now(); }
    void record_stop() { stop = std::chrono::high_resolution_clock::now(); }
    double elapsed()
    {
        return std::chrono::duration<double>(stop - start).count();
    }
#endif
};

int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
}