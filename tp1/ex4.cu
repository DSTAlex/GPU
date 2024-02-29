#include <iostream>

//
// example: CUDA_CHECK( cudaMalloc(dx, x, N*sizeof(int) );
//
#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        std::cout << file << ':' << line << ": [CUDA ERROR] " << cudaGetErrorString(code) << std::endl; 
        std::abort();
    }
}


// step 06
__global__
void add_strided(int n, const int *dx, int *dy)
{
     int j = threadIdx.x + blockDim.x * blockIdx.x ;
     for(int i = j; i < n; i += gridDim.x * blockDim.x){
        dy[i] = dx[i] + dy[i];
     }
}




int main()
{
    constexpr int N = 1000;
    int* x = (int*)malloc(N*sizeof(int));
    int* y = (int*)malloc(N*sizeof(int));
    for(int i = 0; i < N; ++i) {
        x[i] = i;
        y[i] = i*i;
    }

    // step 07
    int* dx;
    int* dy;
    // 1. allocate on device
    CUDA_CHECK(cudaMalloc(&dx, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dy, N*sizeof(int)));



    // 2. copy from host to device

    CUDA_CHECK(cudaMemcpy(dx, x, N*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y, N*sizeof(int), cudaMemcpyHostToDevice));


    // 3. launch CUDA kernel
    const int threads_per_bloc = 32;
    const int blocs = 8;

    add_strided<<<blocs, threads_per_bloc>>>(N, dx, dy);


    // 4. copy result from device to host
    CUDA_CHECK(cudaMemcpy(y, dy, N*sizeof(int), cudaMemcpyDeviceToHost));


    // 5. free device memory
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));


    // checking results
    bool ok = true;
    for(int i = 0; i < N; ++i) {
        const int expected_result = i + i*i;
        if(y[i] != expected_result) {
            std::cout << "Failure" << std::endl;
            std::cout << "Result at index i=" 
                << i << ": expected " 
                << i << '+' << i*i << '=' << expected_result << ", got " << y[i] << std::endl;
            ok = false;
            break;
        }
    }
    if(ok) std::cout << "Success" << std::endl;

    free(x);
    free(y);

    return 0;
}
