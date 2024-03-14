#pragma once

#include <vector>
#include <iostream>

#define CUDA_CHECK(code) { cuda_check((code), __FILE__, __LINE__); }
inline void cuda_check(cudaError_t code, const char *file, int line) {
    if(code != cudaSuccess) {
        std::cout << file << ':' << line << ": [CUDA ERROR] " << cudaGetErrorString(code) << std::endl; 
        std::abort();
    }
}

namespace linalg {

//
// Generic matrix of type T (int, float, double...)
//
template<typename T>
class Matrix
{
public:
    // construct matrix, allocate the 2D pitched memory on the device
    __host__ Matrix(int rows, int cols);

    // free allocated device memory
    __host__ void free();

public:
    // copy values from host std::vector to device Matrix
    // values must be a vector of size rows x cols
    // allocation is already done in the constructor
    __host__ void to_cuda(const std::vector<T>& values);

    // copy values from device Matrix to host std::vector
    // values may not ne resized
    __host__ void to_cpu(std::vector<T>& values) const;

public:
    // accessor at row i and column j
    __device__ const T& operator()(int i, int j) const;
    __device__       T& operator()(int i, int j);

public:
    __host__ Matrix operator + (const Matrix<T>& other) const;
    __host__ Matrix operator - (const Matrix<T>& other) const;
    __host__ Matrix operator * (const Matrix<T>& other) const;
    __host__ Matrix operator / (const Matrix<T>& other) const;

private:
    // apply binary functor f on all pairs of elements
    // f must provide the following operator 
    //
    //     T operator()(T a, T b)
    //
    // template<typename BinaryFunctor>
    // __host__ Matrix apply(const Matrix<T>& other, BinaryFunctor&& f) const;

public:
    __host__ __device__ inline int rows() const {return m_rows;}
    __host__ __device__ inline int cols() const {return m_cols;}

private:
    T* m_data_ptr; // device pointer
    int m_rows;
    int m_cols;
    size_t m_pitch;
};

} // namespace linalg

#include "Matrix.hpp"