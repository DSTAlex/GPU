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
        hist[place]++;
    }

    return hist;
}

namespace kernel {

__global__
void compute_hist1(float* d_img, int size, int* d_hist)
{

}

__global__
void merge_hist(int* d_hist, int B)
{

}

} // namespace kernel

std::vector<int> compute_hist_gpu1(float* img, int size)
{
    std::vector<int> h_hist(H);

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
