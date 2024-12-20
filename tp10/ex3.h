#pragma once

#include "common.h"

// random sample M indices according to N scores
thrust::host_vector<int> random_sample(
    const thrust::host_vector<int>& h_scores, 
    int M);

thrust::device_vector<int> random_sample_device(
    const thrust::device_vector<int>& d_scores, 
    int M, int N);
