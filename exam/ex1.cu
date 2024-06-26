#include <vector>
#include <iostream>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        std::cout << file << ':' << line << ": [CUDA ERROR] " << cudaGetErrorString(code) << std::endl; 
        std::abort();
    }
}

constexpr int T = 64; // threads per bloc


namespace kernel {

// CUDA kernel map
// ...
    
    template<typename F>
    __device__  void map(int* dx, int N, F f)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N)
            x[i] = f(x[i]);

    }
    
} // namespace kernel


// apply f on each elements in x: x[i] = f(x[i])
template<typename F>
void map(std::vector<int>& x, F f)
{
    int * dx;
    CUDA_CHECK(cudaMalloc(&dx, x.size()*sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size()*sizeof(int), cudaMemcpyHostToDevice));

    kernel::map<<<(T + x.size() -1) / T, T>>>(dx, x.size(), f);

    CUDA_CHECK(cudaMemcpy(x.data(), dx, x.size()*sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dx));
}



int main()
{
    constexpr int N = 10000;

    std::vector<int> x(N);
    for(int i = 0; i < N; ++i) {
        x[i] = i - N/2;
    }


    // apply square function: f(u) = u^2
    map( x, []__device__(const int u){ return u * u; } );


    // check result
    for(int i = 0; i < N; ++i) {
        const int expected = (i - N/2) * (i - N/2);
        if(x[i] != expected) {
            std::cout << "Error at i=" << i << std::endl;
            std::cout << "expected " << expected << std::endl;
            std::cout << "got " << x[i] << std::endl;
            return 1;
        }
    }
    std::cout << "Success" << std::endl;

    return 0;
}
