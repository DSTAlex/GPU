#pragma once

#include <iostream>
#include <vector>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

constexpr int W = 16;       // 16   warps per bloc
constexpr int T = 32 * W;   // 512  threads per bloc (32 threads per warp)
constexpr int B = 10;       // 10   blocs
constexpr int N = T * B;    // 5120 elements to process in total

__device__ int warp_reduce(int sum);

namespace kernel {

__global__ void reduce1(const int *x, int *y, int N);
__global__ void reduce2(const int *x, int *y, int N);

} // namespace kernel
