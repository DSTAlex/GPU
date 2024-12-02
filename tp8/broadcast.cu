#include "broadcast.h"

namespace kernel {
    
__global__ void broadcast1(int* x, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i>N)
        return;
    int y = x[i];
    y = __shfl_sync(0xFFFFFFFF, y, 0, N);
    x[i] = y;
}

__global__ void broadcast2(int* x, int N)
{

}

__global__ void broadcast3(int* x, int N)
{

}

} // namespace kernel