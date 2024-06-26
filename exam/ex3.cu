#include "ex3_data.h"
#include <vector>
#include <iostream>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        std::cout << file << ':' << line << ": [CUDA ERROR] " << cudaGetErrorString(code) << std::endl; 
        std::abort();
    }
}

constexpr int B = 32; // blocs
constexpr int T = 64; // threads per bloc

namespace kernel {
    
__global__
void reduce_max(const int* x, int* y, int N)
{
    // ...
}

} // namespace kernel


int reduce_max1(const std::vector<int>& x)
{
    // ...
    // kernel::reduce_max<<<...,...>>>(...);
    // ...
    return 0;
}


int reduce_max2(const std::vector<int>& x)
{
    // ...
    // kernel::reduce_max<<<...,...>>>(...);
    // kernel::reduce_max<<<...,...>>>(...);
    // ...
    return 0;
}


int main()
{
    {
        std::cout << "Test 1 (reduce_max1)" << std::endl;
        const std::vector<int> x = get_x1();
        const int res = reduce_max1(x);
        const int max = 88;
        if(res != max) {
            std::cout << "Error" << std::endl;
            std::cout << "expected " << max << std::endl;
            std::cout << "got " << res << std::endl;
        } else {
            std::cout << "Success" << std::endl;
        }
    }
    {
        std::cout << "Test 2 (reduce_max1)" << std::endl;
        const std::vector<int> x = get_x2();
        const int res = reduce_max1(x);
        const int max = 99900;
        if(res != max) {
            std::cout << "Error" << std::endl;
            std::cout << "expected " << max << std::endl;
            std::cout << "got " << res << std::endl;
        } else {
            std::cout << "Success" << std::endl;
        }
    }
    {
        std::cout << "Test 3 (reduce_max1)" << std::endl;
        const std::vector<int> x = get_x3();
        const int res = reduce_max1(x);
        const int max = -562;
        if(res != max) {
            std::cout << "Error" << std::endl;
            std::cout << "expected " << max << std::endl;
            std::cout << "got " << res << std::endl;
        } else {
            std::cout << "Success" << std::endl;
        }
    }
    {
        std::cout << "Test 4 (reduce_max1)" << std::endl;
        const std::vector<int> x = get_x4();
        const int res = reduce_max1(x);
        const int max = -1216;
        if(res != max) {
            std::cout << "Error" << std::endl;
            std::cout << "expected " << max << std::endl;
            std::cout << "got " << res << std::endl;
        } else {
            std::cout << "Success" << std::endl;
        }
    }
    std::cout << "=========================================" << std::endl;
    {
        std::cout << "Test 1 (reduce_max2)" << std::endl;
        const std::vector<int> x = get_x1();
        const int res = reduce_max2(x);
        const int max = 88;
        if(res != max) {
            std::cout << "Error" << std::endl;
            std::cout << "expected " << max << std::endl;
            std::cout << "got " << res << std::endl;
        } else {
            std::cout << "Success" << std::endl;
        }
    }
    {
        std::cout << "Test 2 (reduce_max2)" << std::endl;
        const std::vector<int> x = get_x2();
        const int res = reduce_max2(x);
        const int max = 99900;
        if(res != max) {
            std::cout << "Error" << std::endl;
            std::cout << "expected " << max << std::endl;
            std::cout << "got " << res << std::endl;
        } else {
            std::cout << "Success" << std::endl;
        }
    }
    {
        std::cout << "Test 3 (reduce_max2)" << std::endl;
        const std::vector<int> x = get_x3();
        const int res = reduce_max2(x);
        const int max = -562;
        if(res != max) {
            std::cout << "Error" << std::endl;
            std::cout << "expected " << max << std::endl;
            std::cout << "got " << res << std::endl;
        } else {
            std::cout << "Success" << std::endl;
        }
    }
    {
        std::cout << "Test 4 (reduce_max2)" << std::endl;
        const std::vector<int> x = get_x4();
        const int res = reduce_max2(x);
        const int max = -1216;
        if(res != max) {
            std::cout << "Error" << std::endl;
            std::cout << "expected " << max << std::endl;
            std::cout << "got " << res << std::endl;
        } else {
            std::cout << "Success" << std::endl;
        }
    }


    return 0;
}
