#include "ex1.h"

template<typename T>
std::vector<T> read_from_file(const std::string& filename, int N);

int main()
{
    {
        std::cout << "Test 1" << std::endl;
        constexpr int N = 96;
        const std::vector<int> x = read_from_file<int>("data.txt", N);
        const int z_true = max_abs_cpu(x.data(), N);
        const int z_test = max_abs_gpu(x.data(), N);
        if(z_test != z_true) {
            std::cerr << "Error" << std::endl;
            std::cerr << "  expected : " << z_true << std::endl;
            std::cerr << "  got      : " << z_test << std::endl;
            return 0;
        }
        std::cout << "Success" << std::endl;
    }
    {
        std::cout << "Test 2" << std::endl;
        constexpr int N = 960;
        const std::vector<int> x = read_from_file<int>("data.txt", N);
        const int z_true = max_abs_cpu(x.data(), N);
        const int z_test = max_abs_gpu(x.data(), N);
        if(z_test != z_true) {
            std::cerr << "Error" << std::endl;
            std::cerr << "  expected : " << z_true << std::endl;
            std::cerr << "  got      : " << z_test << std::endl;
            return 0;
        }
        std::cout << "Success" << std::endl;
    }
    {
        std::cout << "Test 3" << std::endl;
        constexpr int N = 10416;
        const std::vector<int> x = read_from_file<int>("data.txt", N);
        const int z_true = max_abs_cpu(x.data(), N);
        const int z_test = max_abs_gpu(x.data(), N);
        if(z_test != z_true) {
            std::cerr << "Error" << std::endl;
            std::cerr << "  expected : " << z_true << std::endl;
            std::cerr << "  got      : " << z_test << std::endl;
            return 0;
        }
        std::cout << "Success" << std::endl;
    }
    return 1;
}



template<typename T>
std::vector<T> read_from_file(const std::string& filename, int N)
{
    std::vector<T> values(N);
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