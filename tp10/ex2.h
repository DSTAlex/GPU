#pragma once

#include "common.h"

thrust::device_vector<int> copy_positives(
    const thrust::device_vector<int>& hx);

std::pair<
    thrust::device_vector<int>,
    thrust::device_vector<int>>
bottom_top_k(
    const thrust::device_vector<int>& hx, 
    int K);

std::pair<
    thrust::host_vector<int>,
    thrust::host_vector<int>>
bottom_top_k_positives(
    const thrust::host_vector<int>& hx, 
    int K);
