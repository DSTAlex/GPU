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
