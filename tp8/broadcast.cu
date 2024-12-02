#include "broadcast.h"

namespace kernel {
    
__global__ void broadcast1(int* x, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i>N)
        return;
    int y = x[i];
    y = __shfl_sync(mask=0xFFFFFFFF, y, srcLane=0, width=N);
    x[i] = y
}

__global__ void broadcast2(int* x, int N)
{

}

__global__ void broadcast3(int* x, int N)
{

}

} // namespace kernel