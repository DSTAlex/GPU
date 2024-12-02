#include "image.h"
#include <iostream>
#include <vector>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        fprintf(stderr,"%s:%d: [CUDA ERROR] %s\n", file, line, cudaGetErrorString(code));
    }
}

constexpr int H = 32; // Histogram bins count
constexpr int T = 64; // Threads per bloc

std::vector<int> compute_hist_cpu(float* img, int size)
{
    std::vector<int> hist(H, 0);

    for (int i = 0; i < size; i++){
        int place = int(img[i] * H);
        if (img[i] != 1)
            hist[place]++;
    }

    return hist;
}

namespace kernel {

__global__
void compute_hist1(float* d_img, int size, int* d_hist)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float value = d_img[i];
    int place = int(valeu * H);
    atomicAdd(d_hist + place + blockDim.x * T, 1);

}

__global__
void merge_hist(int* d_hist, int B)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int k = 1; k < B; k++){
        d_hist[i] += d_hist[i + k * H]
    }

}

} // namespace kernel

std::vector<int> compute_hist_gpu1(float* img, int size)
{
    std::vector<int> h_hist(H);

    float* d_img;
    int * d_hist;
    int B = (size + T -1) / T;

    CUDA_CHECK(cudaMalloc(&d_img, size*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hist, B*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_img, img, size*sizeof(float), cudaMemcpyHostToDevice));


    kernel::compute_hist1<<<B, T>>>(d_img, size, d_hist);
    cudaDeviceSynchronize();
    kernel::merge_hist<<<1, H>>>(d_hist, B);

    CUDA_CHECK(cudaMemcpy(h_hist.data(), d_hist, H*sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaFree(d_img));

    return h_hist;
}

int main()
{
    int N = 0;
    int M = 0;
    int C = 0;
    float* img = image::load("fractal.jpg", &M, &N, &C, 1);
    const int size = N*M*C;
    // std::cout << "size = " << size << std::endl;

    const std::vector<int> hist_cpu  = compute_hist_cpu (img, size);
    const std::vector<int> hist_gpu1 = compute_hist_gpu1(img, size);
    //const std::vector<int> hist_gpu2 = compute_hist_gpu2(img, size);

    std::cout << "Test" << std::endl;
    {
        // test histogram sizes (must be equal to H)
        if(hist_cpu.size() != H or hist_gpu1.size() != H) 
        {
            std::cout << "Size error, expected " << H << std::endl;
            std::cout << "  got " << hist_cpu.size()  << " (CPU)" << std::endl;
            std::cout << "  got " << hist_gpu1.size() << " (GPU)" << std::endl;
            return 1;
        }

        // compare CPU/GPU histograms (each bins must be equal)
        bool has_error = false;
        for(int i = 0; i < H; ++i)
        {
            if(hist_cpu[i] != hist_gpu1[i]) 
            {
                std::cout << "Error at bin " << i << std::endl;
                std::cout << "  got " << hist_cpu[i]  << " (CPU)" << std::endl;
                std::cout << "  got " << hist_gpu1[i] << " (GPU)" << std::endl;
                has_error = true;
            }
        }
        if(has_error) return 1;
        std::cout << "Success" << std::endl;
    }
    return 0;
}
