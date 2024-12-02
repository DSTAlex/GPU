#include "reduce.h"

int main()
{
    std::vector<int> x(N);
    for(int w = 0; w < W*B; ++w) { // W*B warps in total
        for(int k = 0; k < 16; ++k) {
            // init x such that each warp sums to zero
            x[w*32 + k] = -(k+1)*(w+1);
            x[w*32 + 32-1 - k] = (k+1)*(w+1);
        }
        x[w*32] += 1; // add one such that each warp sums to one
    }

    int sum_true = 0;
    for(int i = 0; i < N; ++i) {
        sum_true += x[i];
    }
    
    int* dx = nullptr;
    CUDA_CHECK( cudaMalloc(&dx, N*sizeof(int)) );
    CUDA_CHECK( cudaMemcpy(dx, x.data(), N*sizeof(int), cudaMemcpyHostToDevice) );

    std::cout << "Test reduce1 " << std::endl;
    {
        int sum = 0;

        // ...

        if(sum != sum_true)
        {
            std::cout << "error" << std::endl;
            std::cout << "  expected = " << sum_true << std::endl;
            std::cout << "  got      = " << sum      << std::endl;
        }
        else
        {
            std::cout << "ok" << std::endl;
        }
    }


    std::cout << "Test reduce2 " << std::endl;
    {
        int sum = 0;

        // ...

        if(sum != sum_true)
        {
            std::cout << "error" << std::endl;
            std::cout << "  expected = " << sum_true << std::endl;
            std::cout << "  got      = " << sum      << std::endl;
        }
        else
        {
            std::cout << "ok" << std::endl;
        }
    }


}