#pragma once

#include <iostream>
#include <vector>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

namespace kernel {

__global__ void broadcast1(int* x, int N);
__global__ void broadcast2(int* x, int N);
__global__ void broadcast3(int* x, int N);

} // namespace kernel