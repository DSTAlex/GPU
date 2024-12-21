#pragma once

#include <iostream>
#include <fstream>
#include <algorithm>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/sort.h>


// --------------------------------------------------------

// Random Number Generator
struct RNG
{
    // returns a random number in (0.0, 1.0)
    __host__ __device__
    float operator()(int i) const
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<float> dist(0.0, 1.0);
        rng.discard(i);
        return dist(rng);
    }
};

// --------------------------------------------------------

// print a vector (host or device thrust vector) for debugging
template<typename VecT>
void print_vec(const VecT& vec) {
    for(int i = 0; i < vec.size(); ++i) 
        std::cout << vec[i] << ',';
    std::cout << std::endl;
}
// same but print a string 'name' before
template<typename VecT>
void print_vec(const std::string& name, const VecT& vec) {
    std::cout << name << " = ";
    for(int i = 0; i < vec.size(); ++i) 
        std::cout << vec[i] << ',';
    std::cout << std::endl;
}

// --------------------------------------------------------

// read N values (int or float depending on type T) from a file
template<typename T>
thrust::host_vector<T> read_from_file(const std::string& filename, int N)
{
    thrust::host_vector<T> values(N);
    std::ifstream f;
    f.open(filename);
    if(not f.is_open()) {
        std::cerr << "Error: could not open file '" << filename << "'" << std::endl;
        return {};
    }
    for(int i = 0; i < N; ++i) {
        f >> values[i];
    }
    return values;
}

// --------------------------------------------------------

// compute the histogram of M values in (0,N)
inline thrust::host_vector<float> compute_histogram(const thrust::host_vector<int>& values, int N)
{
    const int M = values.size();
    thrust::host_vector<float> histo(N, 0.f);
    for(int j = 0; j < M; ++j)
        histo[values[j]] += 1.f; // increment corresponsing bin
    for(int i = 0; i < N; ++i)
        histo[i] /= M; // normalize such that histo sums to 1
    return histo;
}

// --------------------------------------------------------

// compute the max of bin-wise differences between two histograms of the same size
inline float max_diff_between_histo(
    const thrust::host_vector<float>& histo0,
    const thrust::host_vector<float>& histo1)
{   
    const int N = histo0.size();
    assert(histo1.size() == N);
    float max = 0;
    for(int i = 0; i < N; ++i)
    {
        //printf("test:%f reel%f\n", histo0[i], histo1[i]);
        max = std::max(max, std::abs(histo0[i] - histo1[i]));
    }
    return max;
}

// --------------------------------------------------------
