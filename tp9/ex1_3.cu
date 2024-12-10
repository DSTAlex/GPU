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
    if(i < N / 4) {
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

    cudaStream_t s1, s2, s3, s4;

    CUDA_CHECK ( cudaStreamCreate(&s1) );
    CUDA_CHECK ( cudaStreamCreate(&s2) );
    CUDA_CHECK ( cudaStreamCreate(&s3) );
    CUDA_CHECK ( cudaStreamCreate(&s4) );

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK( cudaEventCreate(&start));
    CUDA_CHECK( cudaEventCreate(&stop));

    cudaEventRecord(start, 0);

    int* x = nullptr;
    CUDA_CHECK( cudaMallocHost(&x, N*sizeof(int)) );
    for(int i = 0; i < N; ++i)
        x[i] = -N/2 + i;

    int* dx = nullptr;
    CUDA_CHECK( cudaMalloc(&dx, N*sizeof(int)) );
    
    // CUDA_CHECK( cudaMemcpy(dx, x, N*sizeof(int), cudaMemcpyHostToDevice) );

    CUDA_CHECK ( cudaMemcpyAsync(dx, x, N*sizeof(int) / 4, cudaMemcpyHostToDevice, s1) );
    CUDA_CHECK ( cudaMemcpyAsync(dx + (N/4), x + (N/4), N*sizeof(int) / 4, cudaMemcpyHostToDevice, s2) );
    CUDA_CHECK ( cudaMemcpyAsync(dx + 2*(N/4), x + 2*(N/4), N*sizeof(int) / 4, cudaMemcpyHostToDevice, s3) );
    CUDA_CHECK ( cudaMemcpyAsync(dx + 3*(N/4), x + 3*(N/4), N*sizeof(int) / 4, cudaMemcpyHostToDevice, s4) );

    kernel::compute<<<(N/4 + T - 1),T, 0, s1>>>(dx,N/4);
    kernel::compute<<<(N/4 + T - 1),T, 0, s2>>>(dx + (N/4),N/4);
    kernel::compute<<<(N/4 + T - 1),T, 0, s3>>>(dx + 2*(N/4),N/4);
    kernel::compute<<<(N/4 + T - 1),T, 0, s4>>>(dx + 3*(N/4),N/4);
    CUDA_CHECK( cudaGetLastError() );

    //CUDA_CHECK( cudaMemcpy(x, dx, N*sizeof(int), cudaMemcpyDeviceToHost) );

    CUDA_CHECK ( cudaMemcpyAsync(x, dx, N*sizeof(int) / 4, cudaMemcpyDeviceToHost, s1) );
    CUDA_CHECK ( cudaMemcpyAsync(x + (N/4), dx + (N/4), N*sizeof(int) / 4, cudaMemcpyDeviceToHost, s2) );
    CUDA_CHECK ( cudaMemcpyAsync(x + 2*(N/4), dx + 2*(N/4), N*sizeof(int) / 4, cudaMemcpyDeviceToHost, s3) );
    CUDA_CHECK ( cudaMemcpyAsync(x + 3*(N/4), dx + 3*(N/4), N*sizeof(int) / 4, cudaMemcpyDeviceToHost, s4) );

    CUDA_CHECK ( cudaStreamSynchronize(s1) );
    CUDA_CHECK ( cudaStreamSynchronize(s2) );
    CUDA_CHECK ( cudaStreamSynchronize(s3) );
    CUDA_CHECK ( cudaStreamSynchronize(s4) );

    CUDA_CHECK (  cudaStreamDestroy(s1));
    CUDA_CHECK (  cudaStreamDestroy(s2));
    CUDA_CHECK (  cudaStreamDestroy(s3));
    CUDA_CHECK (  cudaStreamDestroy(s4));

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("temps d'execution: %f ms\n", ms);

    return 0;
}
