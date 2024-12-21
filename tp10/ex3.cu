#include "ex3.h"
#include <thrust/iterator/counting_iterator.h>

// random sample M indices according to N scores
thrust::host_vector<int> random_sample(
    const thrust::host_vector<int>& h_scores, 
    int M)
{
    const int N = h_scores.size();
    thrust::device_vector<int> d_scores = h_scores; // host to device
    thrust::device_vector<float> d_proba(N);
    
    int sum = thrust::reduce(thrust::device, d_scores.begin(), d_scores.end(), 0, thrust::plus<int>());

    if (sum == 0)
    {
        sum = 1;
    }

    thrust::transform(thrust::device, d_scores.begin(), d_scores.end(), d_proba.begin(), [sum]__device__(auto zip)->float
        {
            return (float)zip / (float)sum;
        });

    thrust::inclusive_scan(d_proba.begin(), d_proba.end(), d_proba.begin());

    thrust::device_vector<int> res(M);
    thrust::device_vector<float> random(M);

    auto iter = thrust::make_counting_iterator(0);

    thrust::transform(thrust::device, iter, iter+M, random.begin(), []__device__(auto proba)->float
        {
            return RNG()(proba);
        });

    thrust::transform(thrust::device, random.begin(), random.end(), res.begin(), [d_proba]__device__(auto proba)->int
        {
            int index = 0;
            printf("proba:%f ", proba);
            for(auto score : d_proba){
                if (score > proba)
                {
                    printf("score:%f\n", score);
                    return index;
                }
                index++;
            }
            printf("\n");
            return index;
        });


    return res;
}
