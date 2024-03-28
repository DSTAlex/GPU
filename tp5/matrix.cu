#include "matrix.h"

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
// step 01
// return the 1D index of a row-major matrix of size (rows,cols) from 2D indices (i,j)
// 
__host__ __device__ int index1(int i, int j, int rows, int cols)
{
    return i * cols + j;
}

//
// CPU
//
std::vector<int> matmul1(
    const std::vector<int>& A,
    const std::vector<int>& B,
    int N, int M, int P)
{
    //
    // step 02
    //
    std::vector<int> C(N*P);
    for (long i = 0; i < N; i++)
    {
        for (long j = 0; j < P; j++)
        {
            //C[index1(i,j,N,P)] = 0;
            for (long k = 0; k < M; k++)
            {
                C[index1(i, j, N,P)] += A[index1(i, k, N ,M)] * B[index1(k,j, M, P)]; 
            }
        }
    }
    return C;
}

namespace kernel {

//
// step 03
//
__global__
void matmul2(const int* A, const int* B, int* C, int N, int M, int P)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < N && j < P)
    {
        for (int k = 0; k < M; k++)
        {
            C[index1(i, j, N,P)] += A[index1(i, k, N ,M)] * B[index1(k,j, M, P)]; 
        }
    }

}

} // namespace kernel

//
// GPU
//
std::vector<int> matmul2(
    const std::vector<int>& A,
    const std::vector<int>& B,
    int N, int M, int P)
{
    //
    // step 04
    //
    int *da, *db, *dc;
    std::vector<int> C(N*P);

    CUDA_CHECK(cudaMalloc(&da, A.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&db, B.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dc, C.size()*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(da, A.data(), A.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, B.data(), B.size()*sizeof(int), cudaMemcpyHostToDevice));

    dim3 thread_bloc = {T, T, 1};
    dim3 bloc = {(unsigned int)((N + T - 1) / T), (unsigned int)((P + T - 1) / T), 1};
    kernel::matmul2<<<bloc, thread_bloc>>>(da, db, dc, N, M, P);

    CUDA_CHECK(cudaMemcpy(C.data(), dc, C.size()*sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));

    return C;
}

namespace kernel {

//
// step 05
// return the 1D index of a row-major matrix of size (rows,cols) from 2D indices (i,j) inside sub-matrix (bi,bj)
//
__device__ int index2(int i, int j, int bi, int bj, int rows, int cols)
{
    return index1(i * T + bi, j * T + bj, rows, cols);
}

//
// step 06
//
__global__
void matmul3(const int* A, const int* B, int* C, int N, int M, int P)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= N || j >= P)
    {
        return;
    }

    for (int S = 0; S < (M +T -1)/ T; S++)
    {
        __shared__ float s_A[T][T];
        __shared__ float s_B[T][T];
        s_A[threadIdx.x][threadIdx.y] = A[index2(blockIdx.x, S, threadIdx.x, threadIdx.y, N, M)]; 
        s_B[threadIdx.x][threadIdx.y] = B[index2(S, blockIdx.y, threadIdx.x, threadIdx.y, N, M)];    
        __syncthreads();

        for (int k = 0; k < T; k++)
        {
            C[index1(i,j,N,P)] += s_A[threadIdx.x][k] * s_B[k][threadIdx.y];
        }
        __syncthreads()
    }


}

} // namespace kernel

//
// GPU by bloc
//
std::vector<int> matmul3(
    const std::vector<int>& A,
    const std::vector<int>& B,
    int N, int M, int P)
{
    //
    // step 07
    //
    int *da, *db, *dc;
    std::vector<int> C(N*P);

    CUDA_CHECK(cudaMalloc(&da, A.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&db, B.size()*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dc, C.size()*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(da, A.data(), A.size()*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, B.data(), B.size()*sizeof(int), cudaMemcpyHostToDevice));

    dim3 thread_bloc = {T, T, 1};
    dim3 bloc = {(unsigned int)((N + T - 1) / T), (unsigned int)((P + T - 1) / T), 1};
    kernel::matmul3<<<bloc, thread_bloc>>>(da, db, dc, N, M, P);

    CUDA_CHECK(cudaMemcpy(C.data(), dc, C.size()*sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));



    return C;
}

