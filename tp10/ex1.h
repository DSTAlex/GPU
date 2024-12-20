#pragma once

#include "common.h"

thrust::host_vector<int> add(
    const thrust::host_vector<int>& hx, 
    const thrust::host_vector<int>& hy);