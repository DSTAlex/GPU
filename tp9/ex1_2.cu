#include <iostream>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

namespace kernel {

__global__
void compute(int* x, int N, int iter=100)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < N) {
        for(int n = 0; n < iter; ++n)
            x[i] += int(powf(-1,n));
    }
}

} // namespace kernel

//
// simple program that performs dummy computations
//
int main(int argc, char const *argv[])
{
    const int T = argc > 1 ? std::stoi(argv[1]) : 512;
    const int N = 4e8;
    const int B = (N+T-1)/T;
    std::cout << "T = " << T << std::endl;
    std::cout << "B = " << B << std::endl;

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK( cudaEventCreate(&start));
    CUDA_CHECK( cudaEventCreate(&stop));

    cudaEventRecord(start, 0);
    cudaEventSynchronize(start);

    int* x = nullptr;
    CUDA_CHECK( cudaMallocHost(&x, N*sizeof(int)) );
    for(int i = 0; i < N; ++i)
        x[i] = -N/2 + i;

    int* dx = nullptr;
    CUDA_CHECK( cudaMalloc(&dx, N*sizeof(int)) );
    CUDA_CHECK( cudaMemcpy(dx, x, N*sizeof(int), cudaMemcpyHostToDevice) );

    kernel::compute<<<B,T>>>(dx,N);
    CUDA_CHECK( cudaGetLastError() );

    CUDA_CHECK( cudaMemcpy(x, dx, N*sizeof(int), cudaMemcpyDeviceToHost) );


    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    float* ms;
    cudaEventElapsedTime(ms, start, stop);

    printf("duree: %f ms\n", ms);

    return 0;
}
