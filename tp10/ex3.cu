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

    thrust::transform(thrust::device, d_scores.begin(), d_scores.end(), d_scores.begin(), [sum]__device__(auto zip)->float
        {
            return (float)zip / (float)sum;
        });

    thrust::inclusive_scan(d_scores.begin(), d_scores.end(), d_scores.begin());

    thrust::device_vector<int> res(M);
    thrust::device_vector<int> random(M);

    int i = -1;
    thrust::transform(thrust::device, random.begin(), random.end(), random.begin(), [i]__device__(auto proba)->int
        {
            i++;
            return RNG(i);
        });

    thrust::transform(thrust::device, random.begin(), random.end(), res.begin(), [d_scores]__device__(auto proba)->int
        {
            int index = 0;
            for(auto score : d_scores){
                if (score > proba)
                    return index;
                index++;
            }
            return index;
        });


    return res;
}
