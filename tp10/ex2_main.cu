#include "common.h"
#include "ex2.h"

int main()
{
    {
        std::cout << "Test 1" << std::endl;
        constexpr int N = 100;
        constexpr int K = 10;
        const thrust::host_vector<int> x = read_from_file<int>("data/ex2_x.txt", N);
        const thrust::host_vector<int> top_true = read_from_file<int>("data/ex2_test1_top.txt", K);
        const thrust::host_vector<int> bottom_true = read_from_file<int>("data/ex2_test1_bottom.txt", K);
        const auto results = bottom_top_k_positives(x, K);
        const thrust::host_vector<int>& bottom_test = results.first;
        const thrust::host_vector<int>& top_test = results.second;
        for(int i = 0; i<K; ++i) {
            if(bottom_test[i] != bottom_true[i]) {
                std::cerr << "Error (bottom) at i=" << i << std::endl;
                std::cerr << "  expected : " << bottom_true[i] << std::endl;
                std::cerr << "  got      : " << bottom_test[i] << std::endl;
                return 0;
            }
            if(top_test[i] != top_true[i]) {
                std::cerr << "Error (top) at i=" << i << std::endl;
                std::cerr << "  expected : " << top_true[i] << std::endl;
                std::cerr << "  got      : " << top_test[i] << std::endl;
                return 0;
            }
        }
        std::cout << "Success" << std::endl;
    }
    {
        std::cout << "Test 2" << std::endl;
        constexpr int N = 1000;
        constexpr int K = 100;
        const thrust::host_vector<int> x = read_from_file<int>("data/ex2_x.txt", N);
        const thrust::host_vector<int> top_true = read_from_file<int>("data/ex2_test2_top.txt", K);
        const thrust::host_vector<int> bottom_true = read_from_file<int>("data/ex2_test2_bottom.txt", K);
        const auto results = bottom_top_k_positives(x, K);
        const thrust::host_vector<int>& bottom_test = results.first;
        const thrust::host_vector<int>& top_test = results.second;
        for(int i = 0; i<K; ++i) {
            if(bottom_test[i] != bottom_true[i]) {
                std::cerr << "Error (bottom) at i=" << i << std::endl;
                std::cerr << "  expected : " << bottom_true[i] << std::endl;
                std::cerr << "  got      : " << bottom_test[i] << std::endl;
                return 0;
            }
            if(top_test[i] != top_true[i]) {
                std::cerr << "Error (top) at i=" << i << std::endl;
                std::cerr << "  expected : " << top_true[i] << std::endl;
                std::cerr << "  got      : " << top_test[i] << std::endl;
                return 0;
            }
        }
        std::cout << "Success" << std::endl;
    }
    {
        std::cout << "Test 3" << std::endl;
        constexpr int N = 1000000;
        constexpr int K = 1000;
        const thrust::host_vector<int> x = read_from_file<int>("data/ex2_x.txt", N);
        const thrust::host_vector<int> top_true = read_from_file<int>("data/ex2_test3_top.txt", K);
        const thrust::host_vector<int> bottom_true = read_from_file<int>("data/ex2_test3_bottom.txt", K);
        const auto results = bottom_top_k_positives(x, K);
        const thrust::host_vector<int>& bottom_test = results.first;
        const thrust::host_vector<int>& top_test = results.second;
        for(int i = 0; i<K; ++i) {
            if(bottom_test[i] != bottom_true[i]) {
                std::cerr << "Error (bottom) at i=" << i << std::endl;
                std::cerr << "  expected : " << bottom_true[i] << std::endl;
                std::cerr << "  got      : " << bottom_test[i] << std::endl;
                return 0;
            }
            if(top_test[i] != top_true[i]) {
                std::cerr << "Error (top) at i=" << i << std::endl;
                std::cerr << "  expected : " << top_true[i] << std::endl;
                std::cerr << "  got      : " << top_test[i] << std::endl;
                return 0;
            }
        }
        std::cout << "Success" << std::endl;
    }
    return 1;
}