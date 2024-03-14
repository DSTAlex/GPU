#include <iostream>

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
// step 04
// return a pointer to the value at row i and column j from base_address 
// with pitch in bytes
//
__device__ inline int* get_ptr(int* base_address, int i, int j, size_t pitch) {
    
}

//
// step 05
// CUDA kernel add 
//




int main()
{
    constexpr int rows = 200;
    constexpr int cols = 80;
    int* x = (int*)malloc(rows*cols*sizeof(int));
    int* y = (int*)malloc(rows*cols*sizeof(int));
    for(int i = 0; i < rows*cols; ++i) {
        x[i] = i;
        y[i] = std::pow(-1,i) * i;
    }

    //
    // step 06
    //
    int* dx;
    int* dy;
    size_t pitch;
    // 1. allocate on device

    // 2. copy from host to device

    // 3. launch CUDA kernel
    // const dim3 threads_per_bloc{32,32,1};

    // 4. copy result from device to host

    // 5. free device memory



    // checking results
    bool ok = true;
    for(int i = 0; i < rows*cols; ++i) {
        const int expected_result = std::pow(-1,i) * i + i;
        if(y[i] != expected_result) {
            std::cout << "Failure" << std::endl;
            std::cout << "Result at index i=" 
                << i << ": expected " 
                << std::pow(-1,i) * i << '+' << i << '=' << expected_result << ", got " << y[i] << std::endl;
            ok = false;
            break;
        }
    }
    if(ok) std::cout << "Success" << std::endl;

    free(x);
    free(y);
    
    return 0;
}
