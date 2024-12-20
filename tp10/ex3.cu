#include "ex3.h"

// random sample M indices according to N scores
thrust::host_vector<int> random_sample(
    const thrust::host_vector<int>& h_scores, 
    int M)
{
    const int N = h_scores.size();
    thrust::device_vector<int> d_scores = h_scores; // host to device
    
    // ...

    return {};
}