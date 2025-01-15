#include "ex2.h"

template<typename T>
thrust::host_vector<T> read_from_file(const std::string& filename, int N);

int main()
{
    {
        std::cout << "Test 1" << std::endl;
        const thrust::host_vector<int> x = {1,2,-3,-4,5,6,-7,-8}; // 8 elements
        const thrust::host_vector<int> z_true = copy_positive_cpu(x);
        const thrust::host_vector<int> z_test = copy_positive_gpu(x);
        if(z_true.size() != z_test.size()) {
            std::cerr << "Error: sizes do not match" << std::endl;
            std::cerr << "  expected : " << z_true.size() << std::endl;
            std::cerr << "  got      : " << z_test.size() << std::endl;
            return 0;
        }
        const int N = z_true.size();
        for(int i = 0; i<N; ++i) {
            if(z_test[i] != z_true[i]) {
                std::cerr << "Error: at i=" << i << std::endl;
                std::cerr << "  expected : " << z_true[i] << std::endl;
                std::cerr << "  got      : " << z_test[i] << std::endl;
                return 0;
            }
        }
        std::cout << "Success" << std::endl;
    }
    {
        std::cout << "Test 2" << std::endl;
        const thrust::host_vector<int> x = read_from_file<int>("data.txt", 1024);
        const thrust::host_vector<int> z_true = copy_positive_cpu(x);
        const thrust::host_vector<int> z_test = copy_positive_gpu(x);
        if(z_true.size() != z_test.size()) {
            std::cerr << "Error: sizes do not match" << std::endl;
            std::cerr << "  expected : " << z_true.size() << std::endl;
            std::cerr << "  got      : " << z_test.size() << std::endl;
            return 0;
        }
        const int N = z_true.size();
        for(int i = 0; i<N; ++i) {
            if(z_test[i] != z_true[i]) {
                std::cerr << "Error: at i=" << i << std::endl;
                std::cerr << "  expected : " << z_true[i] << std::endl;
                std::cerr << "  got      : " << z_test[i] << std::endl;
                return 0;
            }
        }
        std::cout << "Success" << std::endl;
    }
    {
        std::cout << "Test 3" << std::endl;
        const thrust::host_vector<int> x = read_from_file<int>("data.txt", 1000000);
        const thrust::host_vector<int> z_true = copy_positive_cpu(x);
        const thrust::host_vector<int> z_test = copy_positive_gpu(x);
        if(z_true.size() != z_test.size()) {
            std::cerr << "Error: sizes do not match" << std::endl;
            std::cerr << "  expected : " << z_true.size() << std::endl;
            std::cerr << "  got      : " << z_test.size() << std::endl;
            return 0;
        }
        const int N = z_true.size();
        for(int i = 0; i<N; ++i) {
            if(z_test[i] != z_true[i]) {
                std::cerr << "Error: at i=" << i << std::endl;
                std::cerr << "  expected : " << z_true[i] << std::endl;
                std::cerr << "  got      : " << z_test[i] << std::endl;
                return 0;
            }
        }
        std::cout << "Success" << std::endl;
    }
    return 1;
}



template<typename T>
thrust::host_vector<T> read_from_file(const std::string& filename, int N)
{
    thrust::host_vector<T> values(N);
    std::ifstream f;
    f.open(filename);
    if(not f.is_open()) {
        std::cerr << "Error: could not open file '" << filename << "'" << std::endl;
        return {};
    }
    for(int i = 0; i < N; ++i) {
        f >> values[i];
    }
    return values;
}