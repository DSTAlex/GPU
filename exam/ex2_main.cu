#include "ex2.h"
#include "ex2_data.h"

void print_vec(const std::vector<int>& vec);

int main()
{
    // ====================================================
    std::cout << "Testing matvecmul1 \n";
    {
        const std::vector<int> A = get_A1();
        const std::vector<int> b = get_b1();
        const std::vector<int> c = get_c1();
        // ------------------------------------------------
        const std::vector<int> c_test = matvecmul1(A, b);
        if(c == c_test) {
            std::cout << "Test 1: Success\n";
        } else {
            std::cout << "Test 1: Error!\n";
            // std::cout << "expected:\n";
            // print_vec(c);
            // std::cout << "got:\n";
            // print_vec(c_test);
        }
    }
    {
        const std::vector<int> A = get_A2();
        const std::vector<int> b = get_b2();
        const std::vector<int> c = get_c2();
        // ------------------------------------------------
        const std::vector<int> c_test = matvecmul1(A, b);
        if(c == c_test) {
            std::cout << "Test 2: Success\n";
        } else {
            std::cout << "Test 2: Error!\n";
            // std::cout << "expected:\n";
            // print_vec(c);
            // std::cout << "got:\n";
            // print_vec(c_test);
        }
    }
    {
        const std::vector<int> A = get_A3();
        const std::vector<int> b = get_b3();
        const std::vector<int> c = get_c3();
        // ------------------------------------------------
        const std::vector<int> c_test = matvecmul1(A, b);
        if(c == c_test) {
            std::cout << "Test 3: Success\n";
        } else {
            std::cout << "Test 3: Error!\n";
            // std::cout << "expected:\n";
            // print_vec(c);
            // std::cout << "got:\n";
            // print_vec(c_test);
        }
    }
    // ====================================================
    std::cout << "Testing matvecmul2 \n";
    {
        const std::vector<int> A = get_A1();
        const std::vector<int> b = get_b1();
        const std::vector<int> c = get_c1();
        // ------------------------------------------------
        const std::vector<int> c_test = matvecmul2(A, b);
        if(c == c_test) {
            std::cout << "Test 1: Success\n";
        } else {
            std::cout << "Test 1: Error!\n";
            // std::cout << "expected:\n";
            // print_vec(c);
            // std::cout << "got:\n";
            // print_vec(c_test);
        }
    }
    {
        const std::vector<int> A = get_A2();
        const std::vector<int> b = get_b2();
        const std::vector<int> c = get_c2();
        // ------------------------------------------------
        const std::vector<int> c_test = matvecmul2(A, b);
        if(c == c_test) {
            std::cout << "Test 2: Success\n";
        } else {
            std::cout << "Test 2: Error!\n";
            // std::cout << "expected:\n";
            // print_vec(c);
            // std::cout << "got:\n";
            // print_vec(c_test);
        }
    }
    {
        const std::vector<int> A = get_A3();
        const std::vector<int> b = get_b3();
        const std::vector<int> c = get_c3();
        // ------------------------------------------------
        const std::vector<int> c_test = matvecmul2(A, b);
        if(c == c_test) {
            std::cout << "Test 3: Success\n";
        } else {
            std::cout << "Test 3: Error!\n";
            // std::cout << "expected:\n";
            // print_vec(c);
            // std::cout << "got:\n";
            // print_vec(c_test);
        }
    }
    // ====================================================
    std::cout << "Testing matvecmul3 \n";
    {
        const std::vector<int> A = get_A2();
        const std::vector<int> b = get_b2();
        const std::vector<int> c = get_c2();
        // ------------------------------------------------
        const std::vector<int> c_test = matvecmul3(A, b);
        if(c == c_test) {
            std::cout << "Test 2: Success\n";
        } else {
            std::cout << "Test 2: Error!\n";
            // std::cout << "expected:\n";
            // print_vec(c);
            // std::cout << "got:\n";
            // print_vec(c_test);
        }
    }
    {
        const std::vector<int> A = get_A3();
        const std::vector<int> b = get_b3();
        const std::vector<int> c = get_c3();
        // ------------------------------------------------
        const std::vector<int> c_test = matvecmul3(A, b);
        if(c == c_test) {
            std::cout << "Test 3: Success\n";
        } else {
            std::cout << "Test 3: Error!\n";
            // std::cout << "expected:\n";
            // print_vec(c);
            // std::cout << "got:\n";
            // print_vec(c_test);
        }
    }
    // ====================================================
    return 0;
}

///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////

void print_vec(const std::vector<int>& vec)
{
    std::cout << "[size=" << vec.size() << "] = ";
    if(vec.size() == 0) {
        std::cout << "empty vector" << std::endl;
    } else {
        for(int i = 0; i < int(vec.size()); ++i) {
                std::cout << vec[i] << " ";
        }
        std::cout << std::endl;
    }
}
