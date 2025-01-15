#include "ex1.h"

__device__ int warp_reduce_max(int m)
{
    int other = 0;
    for (int k = 32; k > 1; k/=2){
        other = __shfl_down_sync(0xFFFFFFFF, m, k/2, 32);
        if (other > m)
            m = other;
        __syncthreads();
    }
    return m;
}

namespace kernel {

__global__ void max_abs(const int *x, int N, int *y)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > N)
        return;
    
    int v = warp_reduce_max(abs(x[i])); 

    if (i % 32 == 0)
        y[i / 32] = v;
}

} // namespace kernel

int max_abs_cpu(const int *x, int N)
{
    int result = 0;
    for(int i = 0; i < N; ++i)
    {
        if(abs(x[i]) > result)
        {
            result = abs(x[i]);
        }
    }
    return result;
}

int max_abs_gpu(const int *x, int N)
{
    int max = 0;

    const int B = (N + T - 1) / T;

    std::vector<int> y(B*W);

    int* dy = nullptr;
    CUDA_CHECK( cudaMalloc(&dy, W*B*sizeof(int)) );

    int* dx = nullptr;
    CUDA_CHECK( cudaMalloc(&dx, N*sizeof(int)) );
    CUDA_CHECK( cudaMemcpy(dx, x.data(), N*sizeof(int), cudaMemcpyHostToDevice) );

    kernel::max_abs<<<B, T>>>(dx, N, dy);

    CUDA_CHECK( cudaMemcpy(y.data(), dy, W*B*sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaFree(dy) );
    
    for ( int v : y){
        if (v > max)
            max = v;
    }
}