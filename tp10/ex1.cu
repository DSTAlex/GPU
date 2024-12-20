#include "ex1.h"

thrust::host_vector<int> add(
    const thrust::host_vector<int>& hx, 
    const thrust::host_vector<int>& hy)
{
    const thrust::device_vector<int> dx = hx; // host to device
    const thrust::device_vector<int> dy = hy; // host to device

    const int N = hx.size();
    thrust::device_vector<int> dz(N);

    thrust::zip_iterator begin =thrust::make_zip_iterator(dx.first, dy.first);
    thrust::zip_iterator end =thrust::make_zip_iterator(dx.last, dy.last);

    thrust::transform(begin, end, dz, [](int a, int b)->int
        {
            return a + b;
        });

    return dz;
}