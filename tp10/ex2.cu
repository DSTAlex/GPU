#include "ex2.h"

struct is_positif
{
  __host__ __device__
  bool operator()(const int x)
  {
    return x > 0;
  }
};

thrust::device_vector<int> copy_positives(
    const thrust::device_vector<int>& dx)
{
    // ...
    int length = thrust::count_if(dx.begin(), dx.end(), is_positif());

    thrust::device_vector<int> dz(length);

    thrust::copy_if(dz.begin(), dx.end(), dz.begin(), is_positif());

    return dz;
}

std::pair<
    thrust::device_vector<int>, 
    thrust::device_vector<int>>
bottom_top_k(
    const thrust::device_vector<int>& dx, 
    int K)
{
    // ...

    return {};
}

std::pair<
    thrust::host_vector<int>,
    thrust::host_vector<int>>
bottom_top_k_positives(
    const thrust::host_vector<int>& hx, 
    int K)
{   
    // ...
    
    return {};
}
