#pragma once

#include <iostream>
#include <vector>
#include <fstream>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

constexpr int W = 3;       // 3   warps per bloc
constexpr int T = 32 * W;  // 96  threads per bloc (32 threads per warp)

__device__ int warp_reduce_max(int m);

namespace kernel {

__global__ void max_abs(const int *x, int N, int *y);

} // namespace kernel

int max_abs_cpu(const int *x, int N);

int max_abs_gpu(const int *x, int N);