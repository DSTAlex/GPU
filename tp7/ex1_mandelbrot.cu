#include "image.h"
#include <iostream>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

//
// image convention 
// +-------------+----------+-----------+-----------------+
// |  H  height  |  N rows  |  index i  |  coordinates y  |
// |  W  width   |  M cols  |  index j  |  coordinates x  |
// +-------------+----------+-----------+-----------------+
//

template <typename T>
__device__ inline T* get_ptr(T *img, int i, int j, int C, size_t pitch) 
{
    return img +  i * pitch + j * C;
}

__host__ __device__
void map_coordinates(int i, int j, int N, int M, float* x, float* y)
{
    *y = ((float)i / (float)N ) * 2 - 1;
    *x = ((float)j / (float)M ) * 3 - 2;
}

__device__
float compute_convergence(float x, float y, int n_max=100, float tau=10.0)
{
    int reel = 0;
    int imaginaire = 0;
    int z_module = 0;
    int tmp_reel = 0;

    int reel_c = x;
    int imaginaire_c = y;

    float n_final = (float)n_max;

    for (int n = 0; n< n_max; n++)
    {
        z_module = reel * reel + imaginaire * imaginaire;

        if (z_module > tau)
        {
            n_final = (float)n;
            break;
        }

        // r+i ** = r+i * r+i = r(r+i) + i(r+i) = rr + ri + ir + ii = rr-ii + 2ri
        tmp_reel = reel;
        reel = reel * reel - imaginaire * imaginaire + reel_c;
        imaginaire = 2 * tmp_reel * imaginaire + imaginaire_c;
    }

    return n_final / (float)n_max;
}



namespace kernel {

__global__
void generate(int N, int M, int C, int pitch, float* img)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= N || j >= M)
        return;
    int x, y;
    map_coordinates(i, j, N, M, &x, &y);

    float * pixel = get_ptr<float>(img, i, j, C, pitch);
    float val = compute_convergence(x,y);
    *pixel=val;

}

} // namespace kernel


void test(int N, int M){
    float x;
    float y;
    map_coordinates(0, 0, N, M, &x, &y);
    if (x != -2 || y != -1)
    {
        printf("-2 != %f || -1 != %f\n", x, y);
    }
    map_coordinates(N, M, N, M, &x, &y);
    if (x != 1 || y != 1)
    {
        printf("1 != %f || 1 != %f\n", x, y);
    }
    map_coordinates(N/2, 2*M/3, N, M, &x, &y);
    if (x != 0 || y != 0)
    {
        printf("0 != %f || 0 != %f\n", x, y);
    }
}

int main()
{
    float* d_img;
    size_t pitch = 0;
    int C = 1;
    int N = 2 * 320; // 640
    int M = 3 * 320; // 960
    constexpr int T = 32;

    float* img = (float*)malloc(N*M*C*sizeof(float));

    test(N, M);
    float* dx;

    CUDA_CHECK(cudaMalloc(dx, N*M*sizeof(float)));

    // 2. copy from host to device
    CUDA_CHECK(cudaMemcpy(dx, img, N*M*sizeof(float), cudaMemcpyHostToDevice));

    // 3. launch CUDA kernel
    kernel::generate(N, M, C, pitch, img);

    // 4. copy result from device to host
    CUDA_CHECK(cudaMemcpy(img, dx, N*M*sizeof(float), cudaMemcpyDeviceToHost));

    // 5. free device memory
    CUDA_CHECK(cudaFree(dx));

    image::save("fractal.jpg", M, N, C, img);
    std::cout << "Image saved to fractal.jpg" << std::endl;

    free(img);

    return 0;
}


