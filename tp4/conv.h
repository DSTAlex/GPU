#pragma once

#include <iostream>
#include <vector>

//
// 1D convolution 
// - x: input array of size N
// - y: kernel of odd size M
//

// CPU
std::vector<int> conv1(const std::vector<int>& x, const std::vector<int>& y);

// GPU (naive)
std::vector<int> conv2(const std::vector<int>& x, const std::vector<int>& y);

// GPU (optimized)
std::vector<int> conv3(const std::vector<int>& x, const std::vector<int>& y);
