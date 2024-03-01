#include <cstdio>

// step 02

__global__
void hello_worl()
{
    printf("Hello World bloc=%i thread = %i", blockIdx.x, threadIdx.x)
}



int main()
{
    // step 03
    hello_worl<<<1,1>>>()

    cudaDeviceSyncronize()

    return 0;
}
