#include <stdio.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <mpi.h>

#include <cuda_runtime.h>

// Uncomment for pageable host memory allocation
#define USE_PINNED_MEMORY
// Uncomment to use std::chrono for timing:
#define TIMING_CUDA_EVENTS
// Number of copies for sending
#define NUM_COPIES 10

// Size for data transfer in total (32 GB)
const size_t SIZE = 32L * 1024 * 1024 * 1024;

 // Size for Allreduce, 1 GB buffer (in floats)
#define ALLREDUCE_SIZE (1L * 1024 * 1024 * 1024 / sizeof(float))

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
    
    // Check number of MPI processes
    if (size != 4) {
        if (rank == 0)
            std::cerr << "This program assumes 4 MPI processes" << std::endl;
        MPI_Finalize();
        return 1;
    } 
    std::cout << "I am rank " << rank << " of " << size << std::endl;

    // Check number of available GPUs
    int num_gpus;
    cudaError_t err = cudaGetDeviceCount(&num_gpus);
    if (err != cudaSuccess || num_gpus == 0) {
        std::cerr << "Rank " << rank << ": No CUDA-capable devices found: " << cudaGetErrorString(err) << std::endl;
        MPI_Finalize();
        return 1;
    }
    std::cout << "Rank " << rank << ": Detected " << num_gpus << " GPUs" << std::endl;
    
    // Set GPU device based on rank
    err = cudaSetDevice(rank);
    if (err != cudaSuccess) {
        std::cerr << "Rank " << rank << ": cudaSetDevice failed: " << cudaGetErrorString(err) << std::endl;
        MPI_Finalize();
        return 1;
    }
    std::cout << "Rank " << rank << ": cudaSetDevice sucessfully" << std::endl;
    
    // Enable peer access between GPUs
    for (int i = 0; i < size; i++) {
        if (i != rank)
        {
            int can_access;
            err = cudaDeviceCanAccessPeer(&can_access, rank, i);
            if (err == cudaSuccess && can_access) {
                err = cudaDeviceEnablePeerAccess(i, 0);
                if (err != cudaSuccess) {
                    std::cerr << "Rank " << rank << ": cudaDeviceEnablePeerAccess to " << i << " failed: " << cudaGetErrorString(err) << std::endl;
                }
            }
            else if (err == cudaSuccess) {
                std::cerr << "Rank " << rank << ": cannot enable peer access to " << i << std::endl;
            }
            std::cout << "Rank " << rank << ": enable peer access to " << i << std::endl;
        }
    }

    // Allocate device memory for cudaMemcpyPeer
    void *x_d;
    err = cudaMalloc(&x_d, SIZE);
    if (err != cudaSuccess) {
        std::cerr << "Rank " << rank << ": cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        MPI_Finalize();
        return 1;
    }
    std::cout << "Rank " << rank << ": cudaMalloc sucessfully." << std::endl;

    // Allocate pinned host memory for AllReduce
    float *h_send_buf, *h_recv_buf;
    cudaError_t send_buf_err = cudaMallocHost(&h_send_buf, ALLREDUCE_SIZE * sizeof(float));
    cudaError_t recv_buf_err = cudaMallocHost(&h_recv_buf, ALLREDUCE_SIZE * sizeof(float));
    if (send_buf_err != cudaSuccess) {
        std::cerr << "Rank " << rank << ": cudaMallocHost for Send Buffer failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(x_d);
        MPI_Finalize();
        return 1;
    } else if (recv_buf_err != cudaSuccess) {
        std::cerr << "Rank " << rank << ": cudaMallocHost for Recv Buffer failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(x_d);
        MPI_Finalize();
        return 1;
    } else {
        std::cout << "Rank " << rank << ": cudaMalloc for Send and Recv Buffer sucessfully." << std::endl;
    }

    // Initialize host send buffer (e.g., with rank-specific values)
    for (size_t i = 0; i < ALLREDUCE_SIZE; i++) {
        h_send_buf[i] = static_cast<float>(rank);
    }

    // Create IPC handle for sharing device memory
    cudaIpcMemHandle_t my_handle;
    err = cudaIpcGetMemHandle(&my_handle, x_d);
    if (err != cudaSuccess) {
        std::cerr << "Rank " << rank << ": cudaIpcGetMemHandle failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(x_d);
        cudaFreeHost(h_send_buf);
        cudaFreeHost(h_recv_buf);
        MPI_Finalize();
        return 1;
    }
    std::cout << "Rank " << rank << ": cudaIpcGetMemHandle sucessfully." << std::endl;

    // Share IPC handles among all processes
    cudaIpcMemHandle_t *handles = new cudaIpcMemHandle_t[size];
    MPI_Allgather(&my_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, handles, sizeof(cudaIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    std::cout << "Rank " << rank << ": Share IPC handles sucessfully." << std::endl;

    // Open remote device memory handles
    void **remote_x_d = new void *[size];
    for (int i = 0; i < size; i++) {
        if (i == rank) {
            remote_x_d[i] = x_d;
        }
        else {
            err = cudaIpcOpenMemHandle(&remote_x_d[i], handles[i], cudaIpcMemLazyEnablePeerAccess);
            if (err != cudaSuccess) {
                std::cerr << "Rank " << rank << ": cudaIpcOpenMemHandle for rank " << i << " failed: " << cudaGetErrorString(err) << std::endl;
            }
        }
        std::cout << "Rank " << rank << ": cudaIpcOpenMemHandle for rank " << i << " successfully." << std::endl;
    }

    // Perform GPU-to-GPU transfers in a ring pattern
    int target_rank = (rank + 1) % size;
    int target_device = target_rank % num_gpus;
    Timer timer;
    double total_time = 0.0;

    for (int i = 0; i < NUM_COPIES; i++) {
        MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes

        timer.record_start();
        err = cudaMemcpyPeer(remote_x_d[target_rank], target_device, x_d, rank, SIZE);
        timer.record_stop();
        if (err != cudaSuccess) {
            std::cerr << "Rank " << rank << ": cudaMemcpyPeer to rank " << target_rank << " failed: " << cudaGetErrorString(err) << std::endl;
            break;
        }

        double copy_time = timer.elapsed();
        total_time += copy_time;
        double bandwidth = SIZE / (1024.0 * 1024 * 1024 * copy_time);
        std::cout << "Rank " << rank << " transfer " << i << " to rank " << target_rank << " (GPU " << target_device << "): " << bandwidth << " GiB/s" << std::endl;
    }


    MPI_Finalize();
    return 0;
}