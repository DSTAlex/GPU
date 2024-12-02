#include "reduce.h"


__device__ int warp_reduce(int sum) 
{
    int y = sum;
    for (int k = 32; k > 1; k/=2){
        y += __shfl_down_sync(0xFFFFFFFF, y, k/2, 32);
        __syncthreads();
    }
    return y;
}


namespace kernel {

__global__ void reduce1(const int *x, int *y, int N) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > N)
        return;
    
    int v = warp_reduce(x[i]); 

    if (i % 32 == 0)
        y[i / 32] = v;
}

__global__ void reduce2(const int *x, int *y, int N) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > N)
        return;
    
    int v = warp_reduce(x[i]); 


    __shared__ int buffer[32];
    if (threadIdx.x == 0)
    {
        for (int j = 0; j < 32; j++)
            buffer[j] = 0;
    }
    __syncthreads();
    
    if (threadIdx.x % 32 == 0)
        buffer[threadIdx.x / 32] = v;
    __syncthreads();

    int val = 0;
    if (threadIdx.x < 32)
    {
        val = warp_reduce(buffer[threadIdx.x]);  
    }
    
    __syncthreads();
    if (threadIdx.x == 0)
        y[blockIdx.x] = val;
}

} // namespace kernel