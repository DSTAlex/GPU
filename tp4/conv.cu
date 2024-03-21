#include "conv.h"

constexpr int     threads_per_bloc = 16;
constexpr int T = threads_per_bloc;

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


//
// 1D convolution 
// - x: input array of size N
// - y: kernel of odd size M
//
// CPU
//
std::vector<int> conv1(const std::vector<int>& x, const std::vector<int>& y)
{
    //
    // step 01
    //
    const int N = x.size();
    const int M = y.size();
    const int P = (M-1) / 2;

    std::vector<int> z(N);

    for (int i =0 ; i < N ; i++)
    {
        int cov = 0;
        for (int j = 0; j < M; j++)
        {
            int k = i + j - P;
            if (k >= 0 and k < N)
            {
                cov += x[k] * y[j];
            }
        }
        z[i] = cov;
    }


    return z;
}

namespace kernel {

//
// step 02
//
__global__ 
void conv2(const int* dx, const int* dy, int N, int M, int* dz)
{
    const int P = (M-1) / 2;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        int cov = 0;
        for (int j = 0; j < M; j++)
        {
            int k = i + j - P;
            if (k >= 0 and k < N)
            {
                cov += dx[k] * dy[j];
            }
        }
        dz[i] = cov;
    }
}

} // namespace kernel

//
// 1D convolution 
// - x: input array of size N
// - y: kernel of odd size M
//
// GPU (naive)
//
std::vector<int> conv2(const std::vector<int>& x, const std::vector<int>& y)
{
    //
    // step 03
    //
    int N = x.size();
    int M = y.size();
    int *dz, *dx, *dy;

    CUDA_CHECK(cudaMalloc(&dz, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dx, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dy, M*sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(dx, x, N*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y, M*sizeof(int), cudaMemcpyHostToDevice));
    
    kernel::conv2(dx, dy, N, M, dz);

    std::vector<int> z(N);

    CUDA_CHECK(cudaMemcpy(z.data(), dz, N*sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dz));

    return z  
}

namespace kernel {

//
// step 04
//
__global__ 
void conv3(const int* dx, const int* dy, int N, int M, int* dz)
{




}

} // namespace kernel

//
// 1D convolution 
// - x: input array of size N
// - y: kernel of odd size M
//
// GPU (optimized)
//
std::vector<int> conv3(const std::vector<int>& x, const std::vector<int>& y)
{
    //
    // step 05
    //




}
