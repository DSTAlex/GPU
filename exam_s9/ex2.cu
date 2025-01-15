#include "ex2.h"

thrust::host_vector<int> copy_positive_cpu(const thrust::host_vector<int>& hx)
{
    const int N = hx.size();
    thrust::host_vector<int> results;

    for (int i = 0; i < N; i++)
    {
        if (hx[i] > 0)
        {
            results.push_back(hx[i]);
        }
    }

    return results;
}

thrust::host_vector<int> copy_positive_gpu(const thrust::host_vector<int>& hx)
{
     const thrust::device_vector<int> dx = hx; // host to device

    const int N = hx.size();
    thrust::device_vector<int> is_pos(N);
    thrust::device_vector<int> map(N);

    thrust::transform(dx.begin(), dx.end(), is_pos.begin(), []__device__(auto value)->int
        {
            if (value > 0)
                return 1;
            else
                return 0;
        });



    thrust::inclusive_scan(is_pos.begin(), is_pos.end(), map.begin());
    
    const int M = map[N - 1];

    thrust::device_vector<int> results(M);

    thrust::scatter_if(dx.begin(), dx.end(), map.begin(), is_pos.begin(), results.begin());

    return results;
}