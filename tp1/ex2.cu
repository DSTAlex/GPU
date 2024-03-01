#include <cstdio>

// step 02

__global__
void hello_world()
{
    printf("Hello World bloc=%i thread = %i\n", blockIdx.x, threadIdx.x);
}



int main()
{
    // step 03
    hello_world<<<1,1>>>();

    cudaDeviceSynchronize();

    return 0;
}
