#include "broadcast.h"

// test 1 (only 1 bloc of 1 warp of 32 threads in total)
constexpr int W1 = 1;        // 1    warp per bloc
constexpr int T1 = 32 * W1;  // 32   threads per bloc (32 threads per warp)
constexpr int B1 = 1;        // 1    blocs
constexpr int N1 = T1 * B1;  // 32   elements to process in total

// test 2
constexpr int W2 = 16;       // 16   warps per bloc
constexpr int T2 = 32 * W2;  // 512  threads per bloc (32 threads per warp)
constexpr int B2 = 10;       // 10   blocs
constexpr int N2 = T2 * B2;  // 5120 elements to process in total

void check_broadcast1(const std::vector<int>& x)
{
    const int N = x.size();
    int error_count = 0;
    for(int i = 0; i < N; ++i)
    {
        const int val_true = (i / 32) * 32;
        const int val_test = x[i];
        if(val_test != val_true)
        {
            if(error_count > 80) { // stop at 80 errors
                std::cout << "..." << std::endl;
                return;
            }
            std::cout << "Error at i=" << i 
                    << " expected="  << val_true
                    << " got="       << val_test 
                    << std::endl;
            ++error_count;
        }
    }
    std::cout << "success" << std::endl;
}

void check_broadcast2(const std::vector<int>& x)
{
    const int N = x.size();
    int error_count = 0;
    for(int i = 0; i < N; ++i)
    {
        const int val_true = (i / 8) * 8 + 7;
        const int val_test = x[i];
        if(val_test != val_true)
        {
            if(error_count > 80) { // stop at 80 errors
                std::cout << "..." << std::endl;
                return;
            }
            std::cout << "Error at i=" << i 
                    << " expected="  << val_true
                    << " got="       << val_test 
                    << std::endl;
            ++error_count;
        }
    }
    std::cout << "success" << std::endl;
}

void check_broadcast3(const std::vector<int>& x)
{
    const int N = x.size();
    int error_count = 0;
    for(int i = 0; i < N; ++i)
    {
        const int val_true = (i / 2) * 2;
        const int val_test = x[i];
        if(val_test != val_true)
        {
            if(error_count > 80) { // stop at 80 errors
                std::cout << "..." << std::endl;
                return;
            }
            std::cout << "Error at i=" << i 
                    << " expected="  << val_true
                    << " got="       << val_test 
                    << std::endl;
            ++error_count;
        }
    }
    std::cout << "success" << std::endl;
}

template<typename KernelFunc, typename CheckFunc>
void generic_check_broadcast(int N, KernelFunc&& kernel, CheckFunc&& check)
{
    std::vector<int> x(N);
    for(int i = 0; i < N; ++i)
        x[i] = i;
        
    int* dx = nullptr;
    CUDA_CHECK( cudaMalloc(&dx, N*sizeof(int)) );
    CUDA_CHECK( cudaMemcpy(dx, x.data(), N*sizeof(int), cudaMemcpyHostToDevice) );

    kernel(dx);
    CUDA_CHECK( cudaGetLastError() );

    CUDA_CHECK( cudaMemcpy(x.data(), dx, N*sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaFree(dx) );

    check(x);
}

int main()
{
    std::cout << "Test broadcast 1 =============================" << std::endl;
    std::cout << "test 1" << std::endl;
    generic_check_broadcast(N1, [](int* dx){kernel::broadcast1<<<B1,T1>>>(dx, N1);}, [](const auto& x){check_broadcast1(x);});
    std::cout << "test 2" << std::endl;
    generic_check_broadcast(N2, [](int* dx){kernel::broadcast1<<<B2,T2>>>(dx, N2);}, [](const auto& x){check_broadcast1(x);});

    std::cout << "Test broadcast 2 =============================" << std::endl;
    std::cout << "test 1" << std::endl;
    generic_check_broadcast(N1, [](int* dx){kernel::broadcast2<<<B1,T1>>>(dx, N1);}, [](const auto& x){check_broadcast2(x);});
    std::cout << "test 2" << std::endl;
    generic_check_broadcast(N2, [](int* dx){kernel::broadcast2<<<B2,T2>>>(dx, N2);}, [](const auto& x){check_broadcast2(x);});

    std::cout << "Test broadcast 3 =============================" << std::endl;
    std::cout << "test 1" << std::endl;
    generic_check_broadcast(N1, [](int* dx){kernel::broadcast3<<<B1,T1>>>(dx, N1);}, [](const auto& x){check_broadcast3(x);});
    std::cout << "test 2" << std::endl;
    generic_check_broadcast(N2, [](int* dx){kernel::broadcast3<<<B2,T2>>>(dx, N2);}, [](const auto& x){check_broadcast3(x);});

    return 0;
}
