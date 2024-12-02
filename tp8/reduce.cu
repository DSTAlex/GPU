#include "reduce.h"


__device__ int warp_reduce(int sum) 
{
    int y = sum;
    for (int k = 32; k >= 1; k/=2){
        y += __shfl_down_sync(0xFFFFFFFF, y, k/2, 32);
        __syncthreads();
    }
    return y;
}


namespace kernel {

__global__ void reduce1(const int *x, int *y, int N) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i>N)
        return;
    
    int v = warp_reduce(x[i]); 

    if (i == 1)
        *y = v;
}

__global__ void reduce2(const int *x, int *y, int N) 
{

}

} // namespace kernel