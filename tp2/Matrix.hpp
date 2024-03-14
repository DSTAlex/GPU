#include "Matrix.h"

namespace linalg {

namespace kernel {

//
// step 10
// CUDA kernel add
//



//
// step 12
// CUDA kernel apply
//




} // namespace kernel


template<typename T>
__host__ Matrix<T>::Matrix(int rows, int cols) :
    m_data_ptr(nullptr),
    m_rows(rows),
    m_cols(cols),
    m_pitch(0)
{
    // step 07

}

template<typename T>
__host__ void Matrix<T>::free()
{
    // step 07

}

template<typename T>
__host__ void Matrix<T>::to_cuda(const std::vector<T>& values)
{
    // step 08

}

template<typename T>
__host__ void Matrix<T>::to_cpu(std::vector<T>& values) const
{
    // step 08

}

template<typename T>
__device__ const T& Matrix<T>::operator()(int i, int j) const
{
    // step 09

}

template<typename T>
__device__ T& Matrix<T>::operator()(int i, int j)
{
    // step 09

}

template<typename T>
__host__ Matrix<T> Matrix<T>::operator + (const Matrix<T>& other) const
{
    // step 11

}

template<typename T>
__host__ Matrix<T> Matrix<T>::operator - (const Matrix<T>& other) const
{
    // step 12

}

template<typename T>
__host__ Matrix<T> Matrix<T>::operator * (const Matrix<T>& other) const
{
    // step 12

}

template<typename T>
__host__ Matrix<T> Matrix<T>::operator / (const Matrix<T>& other) const
{
    // step 12

}

} // namespace linalg
