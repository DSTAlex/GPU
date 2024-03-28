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
    for (int i = 0; i < P; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[index1(i,j,N,P)] = 0;
            for (int k = 0; k < M; k++)
            {
                C[index1(i, j, N,P)] += A[index1(i, k, N ,M)] * B[index1(k,j, M, P)]; 
            }
        }
    }
    print_mat(C, N, P);
    return C;
}

namespace kernel {

//
// step 03
//
__global__
void matmul2(const int* A, const int* B, int* C, int N, int M, int P)
{


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
    std::vector<int> C(N*P);




    return C;
}

namespace kernel {

//
// step 05
// return the 1D index of a row-major matrix of size (rows,cols) from 2D indices (i,j) inside sub-matrix (bi,bj)
//
__device__ int index2(int i, int j, int bi, int bj, int rows, int cols)
{


}

//
// step 06
//
__global__
void matmul3(const int* A, const int* B, int* C, int N, int M, int P)
{



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
    std::vector<int> C(N*P);




    return C;
}

