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
    
    printf("proba score");
    for(float proba : d_proba)
    {
        printf("%f ", proba);
    }

    thrust::inclusive_scan(d_proba.begin(), d_proba.end(), d_proba.begin());

    printf("\ninclusive scan");
    for(float proba : d_proba)
    {
        printf("%f ", proba);
    }

    thrust::device_vector<int> res(M);
    thrust::device_vector<float> random(M);

    auto iter = thrust::make_counting_iterator(0);

    thrust::transform(thrust::device, iter, iter+M, random.begin(), []__device__(auto proba)->float
        {
            return RNG()(proba);
        });

    // printf("\nrandom");
    // for(float proba : random)
    // {
    //     printf("%f ", proba);
    // }
    auto truc = d_proba.data();
    thrust::transform(thrust::device, random.begin(), random.end(), res.begin(), [truc, N]__device__(auto proba)->int
        {
            for (int i = 0; i < N; i++)
            {
                printf("%f",*(truc+i));
            }
            return 0;
        });


    return res;
}
