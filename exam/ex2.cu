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
            c[i] += A[i*M + j] * b[j];
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

    kernel::matvecmul2<<<(T + A.size()/b.size() -1) / T, T>>>(dA, db, dc, A.size()/b.size(), b.size());

    std::vector<int> c(A.size()/b.size());
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
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N * M)
    {
        return;
    }

    __shared__ float s_A[T][T];
    __shared__ float s_B[T];

    for (int S = 0; S < (M + T -1)/ T; S++)
    {
        s_A[threadIdx.x / 4][threadIdx.x % 4] = A[(threadIdx.x / 4) * M + (threadIdx.x % 4) + S * 4]; 
        if (threadIdx.x < T)
            s_B[threadIdx.x] = b[S * T + threadIdx.x];    
        __syncthreads();

        for (int k = 0; k < T; k++)
        {
           c[i / N] += s_A[i / N][k] * s_B[k];
           //c[k] = s_A[k][1];
        }
        
        __syncthreads();
    }
}
} // namespace kernel

//
// GPU by bloc
//
std::vector<int> matvecmul3(
    const std::vector<int>& A,
    const std::vector<int>& b)
{
    int * dA, *db, *dc;
    CUDA_CHECK(cudaMalloc(&dA, A.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&db, b.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dc, A.size()/b.size()*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), A.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), b.size()*sizeof(int), cudaMemcpyHostToDevice));

    kernel::matvecmul3<<<(T + A.size()/b.size() - 1) / T, T>>>(dA, db, dc, A.size(), b.size());

    std::vector<int> c(A.size()/b.size(), 0);

    CUDA_CHECK(cudaMemcpy(c.data(), dc, A.size()/b.size()*sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));
    
    return c;

}

