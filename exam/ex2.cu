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
    return {};
}

namespace kernel {

__global__
void matvecmul2(const int* A, const int* b, int* c, int N, int M)
{
    // ...
}

} // namespace kernel

//
// GPU
//
std::vector<int> matvecmul2(
    const std::vector<int>& A,
    const std::vector<int>& b)
{
    // ...
    return {};
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

