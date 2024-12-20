#include "common.h"
#include "ex1.h"

// in case of compiling in other folder
std::string folder = "tp10/" ;

int main()
{
    {
        std::cout << "Test 1" << std::endl;
        constexpr int N = 8;
        const thrust::host_vector<int> x = read_from_file<int>(folder + "data/ex1_x.txt", N);
        const thrust::host_vector<int> y = read_from_file<int>(folder + "data/ex1_y.txt", N);
        const thrust::host_vector<int> z_true = read_from_file<int>(folder + "data/ex1_z.txt", N);
        const thrust::host_vector<int> z_test = add(x,y);
        for(int i = 0; i<N; ++i) {
            if(z_test[i] != z_true[i]) {
                std::cerr << "Error at i=" << i << std::endl;
                std::cerr << "  expected : " << z_true[i] << std::endl;
                std::cerr << "  got      : " << z_test[i] << std::endl;
                return 0;
            }
        }
        std::cout << "Success" << std::endl;
    }
    {
        std::cout << "Test 2" << std::endl;
        constexpr int N = 4096;
        const thrust::host_vector<int> x = read_from_file<int>(folder + "data/ex1_x.txt", N);
        const thrust::host_vector<int> y = read_from_file<int>(folder + "data/ex1_y.txt", N);
        const thrust::host_vector<int> z_true = read_from_file<int>(folder + "data/ex1_z.txt", N);
        const thrust::host_vector<int> z_test = add(x,y);
        for(int i = 0; i<N; ++i) {
            if(z_test[i] != z_true[i]) {
                std::cerr << "Error at i=" << i << std::endl;
                std::cerr << "  expected : " << z_true[i] << std::endl;
                std::cerr << "  got      : " << z_test[i] << std::endl;
                return 0;
            }
        }
        std::cout << "Success" << std::endl;
    }
    {
        std::cout << "Test 3" << std::endl;
        constexpr int N = 1000000;
        const thrust::host_vector<int> x = read_from_file<int>(folder + "data/ex1_x.txt", N);
        const thrust::host_vector<int> y = read_from_file<int>(folder + "data/ex1_y.txt", N);
        const thrust::host_vector<int> z_true = read_from_file<int>(folder + "data/ex1_z.txt", N);
        const thrust::host_vector<int> z_test = add(x,y);
        for(int i = 0; i<N; ++i) {
            if(z_test[i] != z_true[i]) {
                std::cerr << "Error at i=" << i << std::endl;
                std::cerr << "  expected : " << z_true[i] << std::endl;
                std::cerr << "  got      : " << z_test[i] << std::endl;
                return 0;
            }
        }
        std::cout << "Success" << std::endl;
    }
    return 1;
}