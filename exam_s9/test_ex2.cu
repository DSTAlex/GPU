#include "ex2.h"

template<typename T>
thrust::host_vector<T> read_from_file(const std::string& filename, int N);

int main()
{
    {
        std::cout << "Test 1" << std::endl;
        const thrust::host_vector<int> x = read_from_file<int>("exam_s9/data.txt", 1000000);
        // const thrust::host_vector<int> x = {1,2,0,-4,5,6,-7,-8};
        const thrust::host_vector<int> z_true = copy_positive_cpu(x);
        const thrust::host_vector<int> z_test = copy_positive_gpu(x);
        // if(z_true.size() != z_test.size()) {
        //     std::cerr << "Error: sizes do not match" << std::endl;
        //     std::cerr << "  expected : " << z_true.size() << std::endl;
        //     std::cerr << "  got      : " << z_test.size() << std::endl;
        //     return 0;
        // }
        std::cout << x[999999]<< '\n';
        const int N = z_true.size();
        for(int i = 0; i<N; ++i) {
            if(z_test[i] != z_true[i]) {
                std::cout << "Error: at i=" << i << std::endl;
                std::cout << "  expected : " << z_true[i] << std::endl;
                std::cout << "  got      : " << z_test[i] << std::endl;
                // return 0;
            }
        }
        std::cout << "other "<< z_true[N-1] << " " << z_true[N-2];
        // std::cout << "Success" << std::endl;
    //     std::cout << "input  ";
    //     for (auto e : x)
    //     {
    //         std::cout << e << " ";
    //     }
    //     std::cout << "\noutput ";
    //     for (auto e : z_test)
    //     {
    //         std::cout << e << " ";
    //     }
    //     std::cout << "\n";
    }

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