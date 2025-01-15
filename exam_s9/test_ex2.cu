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
        // if(z_true.size() != z_test.size()) {
        //     std::cerr << "Error: sizes do not match" << std::endl;
        //     std::cerr << "  expected : " << z_true.size() << std::endl;
        //     std::cerr << "  got      : " << z_test.size() << std::endl;
        //     return 0;
        // }
        // const int N = z_true.size();
        // for(int i = 0; i<N; ++i) {
        //     if(z_test[i] != z_true[i]) {
        //         std::cerr << "Error: at i=" << i << std::endl;
        //         std::cerr << "  expected : " << z_true[i] << std::endl;
        //         std::cerr << "  got      : " << z_test[i] << std::endl;
        //         return 0;
        //     }
        // }
        // std::cout << "Success" << std::endl;
        std::cout << "input "
        for (auto e : x)
        {
            std::cout << e ,, " "
        }
        std::cout << "\n outout "
    }
}