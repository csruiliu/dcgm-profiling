#include <stdio.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>

// Uncomment for pageable host memory allocation
#define USE_PINNED_MEMORY
// Uncomment to use std::chrono for timing:
#define TIMING_CUDA_EVENTS
// Number of copies for sending
#define NUM_COPIES 10

// 32 GB
const size_t SIZE = 32L * 1024 * 1024 * 1024;

void *host_alloc(size_t size)
{
#ifdef USE_PINNED_MEMORY
    void *ptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    return (err == cudaSuccess) ? ptr : nullptr;
#else
    return malloc(size);
#endif
}

void host_free(void *ptr)
{
#ifdef USE_PINNED_MEMORY
    cudaFreeHost(ptr);
#else
    free(ptr);
#endif
}

struct Timer
{
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

int main()
{

    cudaError_t err_set = cudaSetDevice(0);
    if (err_set != cudaSuccess)
    {
        fprintf(stderr, "CUDA set device failed: %s\n", cudaGetErrorString(err_set));
        return 1;
    }
    // Allocate host memory
    void *x_h = host_alloc(SIZE);
    if (!x_h)
    {
        std::cerr << "Host allocation failed" << std::endl;
        return -1;
    }

    // Allocate device memory
    void *x_d;
    cudaError_t err = cudaMalloc(&x_d, SIZE);
    if (err != cudaSuccess)
    {
        std::cerr << "Device allocation failed: " << cudaGetErrorString(err) << std::endl;
        host_free(x_h);
        return -1;
    }

    // Warm-up run
    cudaMemcpy(x_d, x_h, SIZE, cudaMemcpyHostToDevice);

    Timer timer;
    double total_time = 0.0;

    for (int i = 0; i < NUM_COPIES; ++i)
    {
        timer.record_start();
        err = cudaMemcpy(x_d, x_h, SIZE, cudaMemcpyHostToDevice);
        timer.record_stop();

        if (err != cudaSuccess)
        {
            std::cerr << "Copy failed: " << cudaGetErrorString(err) << std::endl;
            break;
        }

        const double copy_time = timer.elapsed();
        total_time += copy_time;

        const double bandwidth = SIZE / ((1024 * 1024 * 1024) * copy_time);
        std::cout << "Transfer " << i << ": " << bandwidth << " GiB/s\n";
    }

    const double avg_bandwidth = NUM_COPIES * SIZE / ((1024 * 1024 * 1024) * total_time);
    std::cout << "\nAverage PCIe bandwidth: " << avg_bandwidth << " GiB/s\n";

    // Cleanup
    cudaFree(x_d);
    host_free(x_h);
    cudaDeviceReset();
    return 0;
}
