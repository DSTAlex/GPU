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

constexpr int bloc_count       = 128; // constexpr equivalent to blockDim.x in CUDA kernel
constexpr int threads_per_bloc = 32;  // constexpr equivalent to gridDim.x  in CUDA kernel

constexpr int B = bloc_count;
constexpr int T = threads_per_bloc;

//
// step 04
//
// dx: array of size N
// dy: array of size N
// dz: array of size B
//

__global__
void dot(int N, const int* dx, const int* dy, int* dz)
{
    __shared__ int buffer[T];

    buffer[threadIdx.x] = 0;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = j; i < N; i += gridDim.x * blockDim.x){
        buffer[threadIdx.x] += dx[i] * dy[i];
    }

    int thread = T / 2;
    while (thread > 1)
    {
        if (threadIdx.x < thread)
        {
            buffer[threadIdx.x] += buffer[threadIdx.x + thread];
            if (blockIdx.x ==0 && threadIdx.x == 0)
                printf("id=%i, thread=%i\n", threadIdx.x, thread);
        }
        thread = thread / 2;
        __syncthreads();
    }

}




int main()
{
    constexpr int N = 1e6;

    int* x = (int*)malloc(N * sizeof(int));
    int* y = (int*)malloc(N * sizeof(int));
    int host_expected_result = 0;
    for (int i = 0; i < N; i++) {
        x[i] = i % 10;
        y[i] = i % 3 - 1;
        host_expected_result += x[i] * y[i];
    }

    // step 05
    int result = 0;
    int *dx, *dy, *dz;
    int *z = (int*)malloc(B * sizeof(int));


    CUDA_CHECK(cudaMalloc(&dx, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dy, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dz, B*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dx, x, N*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y, N*sizeof(int), cudaMemcpyHostToDevice));

    dot<<<B,T>>>(N, dx, dy, dz);


    CUDA_CHECK(cudaMemcpy(z, dz, B*sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dz));

    cudaDeviceSynchronize();

    for (int i = 0 ; i < B; i++)
    {
        result += z[i];
        printf("z[%i] = %i\n",i, z[i]);
    }



    // checking results
    if(host_expected_result == result) {
        std::cout << "Success" << std::endl;
    } else {
        std::cout << "Error" << std::endl;
        std::cout << "  expected: " << host_expected_result << std::endl;
        std::cout << "  got: " << result << std::endl;
    }

    free(x);
    free(y);
    free(z);

  return 0;
}