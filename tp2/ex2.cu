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
__device__ 
inline int* get_ptr(int* base_address, int i, int j, size_t pitch) {
    char* adress_in_mat = i * pitch + j * sizeof(int);
    return (char*)base_address + adress_in_mat;
}

//
// step 05
// CUDA kernel add 
//
__global__ 
void add(const int* dx, int* dy, int rows, int cols, size_t pitch)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(j < cols && i < rows)
        int x = *get_ptr(dx, i, j, pitch);
        int y = *get_ptr(dy, i, j, pitch);
        *get_ptr(dy, i, j, pitch) = x + y;

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
    // step 06
    //
    int* dx;
    int* dy;
    size_t pitchx, pitchy;
    // 1. allocate on device
    CUDA_CHECK(cudaMallocPitch(&dx, &pitchx, cols*sizeof(int), row));
    CUDA_CHECK(cudaMallocPitch(&dy, &pitchy, cols*sizeof(int), row));

    // 2. copy from host to device
    CUDA_CHECK(cudaMemcpy2D(dx, pitchx, x, cols*sizeof(int), cols*sizeof(int), rows, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(dy, pitchy, y, cols*sizeof(int), cols*sizeof(int), rows, cudaMemcpyHostToDevice));

    // 3. launch CUDA kernel
    const dim3 threads_per_bloc{32,32,1};
    const dim3 number_of_bloc{(rows + threads_per_bloc.x - 1)/threads_per_bloc.x,
                                (cols + threads_per_bloc.y -1)/ threads_per_bloc.y ,1};
    printf("%i, %i\n", number_of_bloc.x, number_of_bloc.y);
    add<<<number_of_bloc, threads_per_bloc>>>(dx, dy, rows, cols);

    // 4. copy result from device to host
    CUDA_CHECK(cudaMemcpy2D(y, cols*sizeof(int), dy, pitchy, cols*sizeof(int), rows, cudaMemcpyDeviceToHost));

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
