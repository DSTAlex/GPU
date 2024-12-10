#include <iostream>


#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

__global__
void kernel1(int* x, int N)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N)
        x[i] *= 2;
}

__global__
void kernel2(int* y, int N)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N) 
        y[i] += 1;
}

__global__
void kernel3(const int* x, const int* y, int* z, int N)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N) 
        z[i] = x[i] + y[i];
}

__global__
void kernel4(int* z, int N)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N) 
        z[i] -= 1;
}

int main(int argc, char const *argv[])
{
    const int T = argc > 1 ? std::stoi(argv[1]) : 512;
    const int N = 4e5;
    const int B = (N+T-1)/T;
    std::cout << "T = " << T << std::endl;
    std::cout << "B = " << B << std::endl;

    int* x = (int*)malloc(N*sizeof(int));
    int* y = (int*)malloc(N*sizeof(int));
    int* z = (int*)malloc(N*sizeof(int));

    for(int i = 0; i < N; ++i)
        x[i] = i;

    for(int i = 0; i < N; ++i)
        y[i] = -i;

    int* dx = nullptr;
    int* dy = nullptr;
    int* dz = nullptr;

    float time;
    cudaEvent_t start, stop;
    CUDA_CHECK( cudaEventCreate(&start) );
    CUDA_CHECK( cudaEventCreate(&stop) );
    CUDA_CHECK( cudaEventRecord(start, 0) );

    CUDA_CHECK( cudaMalloc(&dx, N*sizeof(int)) );
    CUDA_CHECK( cudaMalloc(&dy, N*sizeof(int)) );
    CUDA_CHECK( cudaMalloc(&dz, N*sizeof(int)) );
    
    CUDA_CHECK( cudaMemcpy(dx, x, N*sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dy, y, N*sizeof(int), cudaMemcpyHostToDevice) );

    kernel1<<<B,T>>>(dx,N);
    CUDA_CHECK( cudaGetLastError() );

    kernel2<<<B,T>>>(dy,N);
    CUDA_CHECK( cudaGetLastError() );

    kernel3<<<B,T>>>(dx,dy,dz,N);
    CUDA_CHECK( cudaGetLastError() );

    kernel4<<<B,T>>>(dz,N);
    CUDA_CHECK( cudaGetLastError() );

    CUDA_CHECK( cudaMemcpy(z, dz, N*sizeof(int), cudaMemcpyDeviceToHost) );

    for(int i = 0; i < N; ++i)
    {
        if(z[i] != i) 
        {
            std::cout << "error at i=" << i << std::endl;
            std::cout << "  expected = " << i << std::endl;
            std::cout << "  got      = " << z[i] << std::endl;
            return 1;
        }
    }

    free(x);
    free(y);
    free(z);
    CUDA_CHECK( cudaFree(dx) );
    CUDA_CHECK( cudaFree(dy) );
    CUDA_CHECK( cudaFree(dz) );

    CUDA_CHECK( cudaEventRecord(stop, 0) );
    CUDA_CHECK( cudaEventSynchronize(stop) );
    CUDA_CHECK( cudaEventElapsedTime(&time, start, stop) );
    std::cout << "time = " << time << std::endl;

    return 0;
}
