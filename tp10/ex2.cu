#include "ex2.h"

struct is_positif
{
  __device__
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

    thrust::copy_if(dx.begin(), dx.end(), dz.begin(), is_positif());

    return dz;
}

std::pair<
    thrust::device_vector<int>, 
    thrust::device_vector<int>>
bottom_top_k(
    const thrust::device_vector<int>& dx, 
    int K)
{
    thrust::device_vector<int> top(K);
    thrust::device_vector<int> bottom(K);

    thrust::device_vector<int> new_dx = dx;

    thrust::sort(new_dx.begin(), new_dx.end());
    // thrust::copy(dx.begin(), dx.begin()+K, bottom.begin());
    // thrust::copy(dx.end()-K, dx.end(), top.begin());


    std::pair<thrust::device_vector<int>, thrust::device_vector<int>> res{top, bottom};
    return res;
}

std::pair<
    thrust::host_vector<int>,
    thrust::host_vector<int>>
bottom_top_k_positives(
    const thrust::host_vector<int>& hx, 
    int K)
{   
    // ...

    thrust::device_vector<int> dx = hx;
    auto a = bottom_top_k(dx, K);
    
    return {};
}
