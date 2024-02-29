#include <iostream>

int main(int argc, char const *argv[])
{
    // step 01
    int device_count;

    cudaError_t err = cudaGetDeviceCount(&device_count);


	   printf("device_count = %i\n", device_count);
    for(auto i = 0; i < device_count; ++i)
    {
        cudaDeviceProp device_prop;
	
	cudaError_t err = cudaGetDeviceProperties(&device_prop, i);
	printf("name %s\n", device_prop.name);	
	printf("totalGlobalMem %zu\n", device_prop.totalGlobalMem / 1000000000);
	printf("sharedMemPerBlock %zu\n", device_prop.sharedMemPerBlock / 1000);
	printf("maxThreadsPerBlock %i\n", device_prop.maxThreadsPerBlock);
	
	printf("maxThreadsDim %i, %i, %i\n", device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
	printf("maxGridSize %i, %i, %i\n", device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
	printf("compute capability %i.%i\n", device_prop.major, device_prop.minor);
	printf("warp %i\n", device_prop.warpSize);
	printf("number of registry %i\n", device_prop.regsPerBlock);
	printf("number of SM %i\n", device_prop.multiProcessorCount);
	
    }

    return 0;
}
