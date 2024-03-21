#include "conv.h"

void print(const std::vector<int>& vec) 
{
    if(vec.empty()) 
    {
        std::cout << "[]" << std::endl;
    }
    else
    {
        std::cout << "[";
        for(size_t i = 0; i < vec.size()-1; ++i)
            std::cout << vec[i] << ", ";
        std::cout << vec.back() << "]" << std::endl;
    }
}

int main()
{
    {
        std::cout << "Test 1" << std::endl;
        const std::vector<int> x = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // N = 10
        const std::vector<int> y = {0, 1, 0}; // M = 3
        const std::vector<int> z_sol = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        const std::vector<int> z = conv1(x, y);
        if(z != z_sol)
        {
            std::cout << "Error, expected:" << std::endl;
            print(z_sol);
            std::cout << "got:" << std::endl;
            print(z);
        }
        else
        {
            std::cout << "Ok" << std::endl;
        }
    }
    {
        std::cout << "Test 2" << std::endl;
        const std::vector<int> x = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // N = 10
        const std::vector<int> y = {1, 2, 4, 2, 1}; // M = 5
        const std::vector<int> z_sol = {4, 11, 20, 30, 40, 50, 60, 70, 70, 59};
        const std::vector<int> z = conv1(x, y);
        if(z != z_sol)
        {
            std::cout << "Error, expected:" << std::endl;
            print(z_sol);
            std::cout << "got:" << std::endl;
            print(z);
        }
        else
        {
            std::cout << "Ok" << std::endl;
        }
    }
    {
        std::cout << "Test 3" << std::endl;
        const std::vector<int> x = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34}; // N = 35
        const std::vector<int> y = {1, -2, 4, -8, 16, -32, 64, -128, 256, -1024, 256, -128, 64, -32, 16, -8, 4, -2, 1}; // M = 19
        const std::vector<int> z_sol = {117, -736, -1333, -2058, -2719, -3412, -4089, -4774, -5455, -6138, -6820, -7502, -8184, -8866, -9548, -10230, -10912, -11594, -12276, -12958, -13640, -14322, -15004, -15686, -16368, -17050, -17767, -18380, -19201, -19606, -20843, -20416, -23317, -19562, -29119};
        const std::vector<int> z = conv1(x, y);
        if(z != z_sol)
        {
            std::cout << "Error, expected:" << std::endl;
            print(z_sol);
            std::cout << "got:" << std::endl;
            print(z);
        }
        else
        {
            std::cout << "Ok" << std::endl;
        }
    }

    return 0;
}
