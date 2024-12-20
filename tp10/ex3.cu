#include "ex3.h"

// random sample M indices according to N scores
thrust::host_vector<int> random_sample(
    const thrust::host_vector<int>& h_scores, 
    int M)
{
    const int N = h_scores.size();
    thrust::device_vector<int> d_scores = h_scores; // host to device
    
    
    int sum = thrust::reduce(thrust::device, d_scores.begin(), d_scores.end(), 0, thrust::plus<int>());

    if (sum == 0)
    {
        sum = 1;
    }
    printf("%lu\n",d_scores.size());
    thrust::transform(thrust::device, d_scores.begin(), d_scores.end(), d_scores.begin(), [sum]__device__(auto zip)->int
        {
            return zip / sum;
        });

    
    printf("%lu\n",d_scores.size());

    return {};
}
