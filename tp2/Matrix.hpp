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
    CUDA_CHECK(cudaMallocPitch(&this.m_data_ptr, &this.m_pitch, this.m_cols*sizeof(T), this.m_rows));
    return this;
}

template<typename T>
__host__ void Matrix<T>::free()
{
    // step 07
    CUDA_CHECK(cudaFree(this.m_data_ptr));
}

template<typename T>
__host__ void Matrix<T>::to_cuda(const std::vector<T>& values)
{
    // step 08
    CUDA_CHECK(cudaMemcpy2D(this.m_data_ptr, this.m_pitch, values, values.size() * sizeof(T), values.size() * sizeof(T), this.m_rows, cudaMemcpyHostToDevice));
    
}

template<typename T>
__host__ void Matrix<T>::to_cpu(std::vector<T>& values) const
{
    // step 08
    CUDA_CHECK(cudaMemcpy2D(values, values.size() * sizeof(T),this.m_data_ptr, this.m_pitch, values.size() * sizeof(T), this.m_rows, cudaMemcpyHostToDevice));
    
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
