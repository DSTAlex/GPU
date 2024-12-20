#include "ex1.h"

thrust::host_vector<int> add(
    const thrust::host_vector<int>& hx, 
    const thrust::host_vector<int>& hy)
{
    const thrust::device_vector<int> dx = hx; // host to device
    const thrust::device_vector<int> dy = hy; // host to device

    const int N = hx.size();
    thrust::device_vector<int> dz(N);

    auto begin = thrust::make_zip_iterator(dx.begin(), dy.begin());
    auto end = thrust::make_zip_iterator(dx.end(), dy.end());

    thrust::transform(begin, end, dz.begin(), [](auto& zip)->int
        {
            return thrust::get<0>(zip) + thrust::get<1>(zip);
        });

    return dz;
}