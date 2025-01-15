#pragma once

#include <iostream>
#include <fstream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>

thrust::host_vector<int> copy_positive_cpu(const thrust::host_vector<int>& hx);

thrust::host_vector<int> copy_positive_gpu(const thrust::host_vector<int>& hx);
