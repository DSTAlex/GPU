#pragma once

#include "common.h"

// random sample M indices according to N scores
thrust::host_vector<int> random_sample(
    const thrust::host_vector<int>& h_scores, 
    int M);

