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

  // Copy data from host to device multiple times
  for (int i = 0; i < NUM_COPIES; ++i)
  {
    err = cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
      std::cerr << "Failed to copy memory from host to device: " << cudaGetErrorString(err) << std::endl;
      cudaFree(x_d);
      free(x_h);
      return -1;
    }
  }

  // Free device memory
  cudaFree(x_d);

  // Finalize the GPU
  cudaDeviceReset();

  // Free host memory
  free(x_h);

  return 0;
}
