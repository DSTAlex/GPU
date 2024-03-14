#include "Matrix.h"

int main()
{
    {
        const int rows = 4;
        const int cols = 4;
        // instantiate two matrices of integers on the device
        linalg::Matrix<int> A(rows, cols);
        linalg::Matrix<int> B(rows, cols);
        // fill the two matrices
        A.to_cuda({ 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16});
        B.to_cuda({16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2, 1});

        // compute the sum
        auto C = A + B;

        // transfert the result on the host
        std::vector<int> c_res;
        C.to_cpu(c_res);
        C.free();
       
        // check results
        const std::vector<int> c_expected{17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17};
        if(c_res != c_expected) {
            std::cout << __FILE__ << ":" << __LINE__ << ": Failure (+):" << std::endl;
            std::cout << "  expected: ";
            for(int i : c_expected) std::cout << i << " ";
            std::cout << std::endl;
            std::cout << "  got:      ";
            for(int i : c_res) std::cout << i << " ";
            std::cout << std::endl;
        } else {
            std::cout << "Success" << std::endl;
        }

        // compute the difference
        auto D = A - B;

        // transfert the result on the host
        std::vector<int> d_res;
        D.to_cpu(d_res);
        D.free();

        // check results
        const std::vector<int> d_expected{-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15};
        if(d_res != d_expected) {
            std::cout << __FILE__ << ":" << __LINE__ << ": Failure (-):" << std::endl;
            std::cout << "  expected: ";
            for(int i : d_expected) std::cout << i << " ";
            std::cout << std::endl;
            std::cout << "  got:      ";
            for(int i : d_res) std::cout << i << " ";
            std::cout << std::endl;
        } else {
            std::cout << "Success" << std::endl;
        }
    }
    // ------------------------------------------------------------------------
    {
        const int rows = 89;
        const int cols = 128;
        linalg::Matrix<float> A(rows, cols);
        linalg::Matrix<float> B(rows, cols);
        std::vector<float> a_values(rows*cols);
        std::vector<float> b_values(rows*cols);
        for(int i = 0; i < rows*cols; ++i) {
            a_values[i] = 1 + float(i) / 100;
            b_values[i] = std::pow(-1, i) * float(i)/(rows*cols) * 100;
        }
        A.to_cuda(a_values);
        B.to_cuda(b_values);

        auto C = A + B;
        auto D = A - B;
        auto E = A * B;
        auto F = A / B;

        std::vector<float> c_values;
        C.to_cpu(c_values);
        std::vector<float> d_values;
        D.to_cpu(d_values);
        std::vector<float> e_values;
        E.to_cpu(e_values);
        std::vector<float> f_values;
        F.to_cpu(f_values);

        C.free();
        D.free();
        E.free();
        F.free();

        const float epsilon = 0.001;
        bool ok = true;
        for(int i = 0; i < rows*cols; ++i) {
            const float diff = std::abs( c_values[i] - (a_values[i] + b_values[i]) );
            if(diff > epsilon) {
                std::cout << __FILE__ << ":" << __LINE__ << ": Failure (+):" << std::endl;
                std::cout << "  expected: " << a_values[i] + b_values[i] << std::endl;
                std::cout << "  got: "      << c_values[i] << std::endl;
                ok = false;
                break;
            }
        }
        if(ok) std::cout << "Success" << std::endl;
        
        ok = true;
        for(int i = 0; i < rows*cols; ++i) {
            const float diff = std::abs( d_values[i] - (a_values[i] - b_values[i]) );
            if(diff > epsilon) {
                std::cout << __FILE__ << ":" << __LINE__ << ": Failure (-):" << std::endl;
                std::cout << "  expected: " << a_values[i] - b_values[i] << std::endl;
                std::cout << "  got: "      << d_values[i] << std::endl;
                ok = false;
                break;
            }
        }
        if(ok) std::cout << "Success" << std::endl;

        ok = true;
        for(int i = 0; i < rows*cols; ++i) {
            const float diff = std::abs( e_values[i] - (a_values[i] * b_values[i]) );
            if(diff > epsilon) {
                std::cout << __FILE__ << ":" << __LINE__ << ": Failure (*):" << std::endl;
                std::cout << "  expected: " << a_values[i] * b_values[i] << std::endl;
                std::cout << "  got: "      << e_values[i] << std::endl;
                ok = false;
                break;
            }
        }
        if(ok) std::cout << "Success" << std::endl;

        ok = true;
        for(int i = 0; i < rows*cols; ++i) {
            const float diff = std::abs( f_values[i] - (a_values[i] / b_values[i]) );
            if(diff > epsilon) {
                std::cout << __FILE__ << ":" << __LINE__ << ": Failure (/):" << std::endl;
                std::cout << "  expected: " << a_values[i] / b_values[i] << std::endl;
                std::cout << "  got: "      << f_values[i] << std::endl;
                ok = false;
                break;
            }
        }
        if(ok) std::cout << "Success" << std::endl;
    }

    return 0;
}
