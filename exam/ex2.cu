#include "ex2.h"

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        std::cout << file << ':' << line << ": [CUDA ERROR] " << cudaGetErrorString(code) << std::endl; 
        std::abort();
    }
}

constexpr int T = 16; // threads per bloc


//
// CPU
//
std::vector<int> matvecmul1(
    const std::vector<int>& A,
    const std::vector<int>& b)
{
    // ...
    std::vector<int> res(A.size()/b.size(), 0);

    for(int i =0; i< A.size()/b.size(); i++)
        {
            for(int j = 0; j < b.size(); j++)
            {
                res[i] += A[i * b.size() + j] * b[j];
            }
        }
    return res;
}

namespace kernel {

__global__
void matvecmul2(const int* A, const int* b, int* c, int N, int M)
{
    // ...
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        c[i] = 0;
        for (int j=0; j < M; j++)
        {
            c[i] += A[i*M + j];
        }
    }
}

} // namespace kernel

//
// GPU
//
std::vector<int> matvecmul2(
    const std::vector<int>& A,
    const std::vector<int>& b)
{
    int * dA, *db, *dc;
    CUDA_CHECK(cudaMalloc(&dA, A.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&db, b.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dc, A.size()/b.size()*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), b.size()*sizeof(int), cudaMemcpyHostToDevice));

    kernel::matvecmul2<<<(T + b.size() -1) / T, T>>>(dA, db, dc, A.size()/b.size(), b.size());

    std::vector<int> c;
    CUDA_CHECK(cudaMemcpy(c.data(), dc, A.size()/b.size()*sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    
    return c;
}

namespace kernel {

__global__
void matvecmul3(const int* A, const int* b, int* c, int N, int M)
{
    // ...
}

} // namespace kernel

//
// GPU by bloc
//
std::vector<int> matvecmul3(
    const std::vector<int>& A,
    const std::vector<int>& b)
{
    // ...
    return {};
}

