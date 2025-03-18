#include <stdio.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>

#define NUM_COPIES 10

int main(int argc, char *argv[])
{

  size_t size = 32L * 1024 * 1024 * 1024;
  void *x_h = malloc(size);
  if (x_h == nullptr)
  {
    std::cerr << "Failed to allocate host memory" << std::endl;
    return -1;
  }

  // Initialize the GPU
  cudaError_t err = cudaSetDevice(0);
  if (err != cudaSuccess)
  {
    std::cerr << "Failed to set device: " << cudaGetErrorString(err) << std::endl;
    free(x_h);
    return -1;
  }

  // Allocate 32 GB on GPU
  void *x_d;
  err = cudaMalloc(&x_d, size);
  if (err != cudaSuccess)
  {
    std::cerr << "Failed to allocate device memory: " << cudaGetErrorString(err) << std::endl;
    free(x_h);
    return -1;
  }

  double total_time = 0.0;

  // Copy data from host to device multiple times
  for (int i = 0; i < NUM_COPIES; ++i)
  {
    auto start = std::chrono::high_resolution_clock::now();
    
    err = cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);

    auto end = std::chrono::high_resolution_clock::now();

    if (err != cudaSuccess)
    {
      std::cerr << "Failed to copy memory from host to device: " << cudaGetErrorString(err) << std::endl;
      cudaFree(x_d);
      free(x_h);
      return -1;
    }

    std::chrono::duration<double> elapsed = end - start;
    double copy_time = elapsed.count();
    total_time += copy_time;

    // Optional: Print bandwidth for each individual copy
    double bandwidth = (size / (1024.0 * 1024.0 * 1024.0)) / copy_time; // GiB/s
    std::cout << "Copy " << i << " bandwidth: " << bandwidth << " GiB/s" << std::endl;
  }

  // Calculate average bandwidth
  double total_data = NUM_COPIES * (size / (1024.0 * 1024.0 * 1024.0)); // Total data in GiB
  double avg_bandwidth = total_data / total_time; // GiB/s

  std::cout << "Average PCIe bandwidth: " << avg_bandwidth << " GiB/s" << std::endl;

  // Free device memory
  cudaFree(x_d);

  // Finalize the GPU
  cudaDeviceReset();

  // Free host memory
  free(x_h);

  return 0;
}
