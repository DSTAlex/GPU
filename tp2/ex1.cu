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
// step 01
// return the linear index corresponding to the element at row i and column j
// in a matrix of size rows x cols, using row-major storage
//
__device__ int linear_index(int i, int j, int rows, int cols) {
    return i * cols + j;
}

//
// step 02
// CUDA kernel add 
//
__global__ 
void add(const int* dx, int* dy, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = linear_index(i, j, rows, cols);
    if(i > rows || j > cols)
        dy[k] = dx[k] + dy[k];
}


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
    // step 03
    //
    int* dx;
    int* dy;
    // 1. allocate on device
    CUDA_CHECK(cudaMalloc(&dx, rows*cols*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dy, rows*cols*sizeof(int)));

    // 2. copy from host to device
    CUDA_CHECK(cudaMemcpy(dx, x, rows*cols*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y, rows*cols*sizeof(int), cudaMemcpyHostToDevice));

    // 3. launch CUDA kernel
    const dim3 threads_per_bloc{32,32,1};
    const dim3 number_of_bloc{(cols + threads_per_bloc.x - 1)/threads_per_bloc.x,
                                (rows + threads_per_bloc.y -1)/ threads_per_bloc.y ,1};
    add<<<number_of_bloc, threads_per_bloc>>>(dx, dy, rows, cols);

    // 4. copy result from device to host
    CUDA_CHECK(cudaMemcpy(y, dy, rows*cols*sizeof(int), cudaMemcpyDeviceToHost));

    // 5. free device memory
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));


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
