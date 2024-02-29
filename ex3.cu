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


// step 04





int main()
{
    constexpr int N = 1000;
    int* x = (int*)malloc(N*sizeof(int));
    int* y = (int*)malloc(N*sizeof(int));
    for(int i = 0; i < N; ++i) {
        x[i] = i;
        y[i] = i*i;
    }

    // step 05
    int* dx;
    int* dy;
    // 1. allocate on device



    // 2. copy from host to device



    // 3. launch CUDA kernel
    const int threads_per_bloc = 32;



    // 4. copy result from device to host



    // 5. free device memory



    // checking results
    bool ok = true;
    for(int i = 0; i < N; ++i) {
        const int expected_result = i + i*i;
        if(y[i] != expected_result) {
            std::cout << "Failure" << std::endl;
            std::cout << "Result at index i=" 
                << i << ": expected " 
                << i << '+' << i*i << '=' << expected_result << ", got " << y[i] << std::endl;
            ok = false;
            break;
        }
    }
    if(ok) std::cout << "Success" << std::endl;

    free(x);
    free(y);

    return 0;
}
