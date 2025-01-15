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

}