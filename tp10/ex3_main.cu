#include "common.h"
#include "ex3.h"

int main()
{
    std::cout << "Test 1" << std::endl;
    {
        const int N = 5;
        const int M = 1000;
        thrust::host_vector<int> scores(N); 
        scores[0] =  1;
        scores[1] =  2;
        scores[2] =  3;
        scores[3] = 10;
        scores[4] =  4;

        const thrust::host_vector<int> results = random_sample(scores, M);
        if(results.size() != M) {
            std::cout << "  failure !" << std::endl;
            std::cout << "  results size = " << results.size() << std::endl;
            std::cout << "  expected     = " << M << std::endl;
        }
        for(int i = 0; i < M; ++i) {
            if(results[i] < 0 or N <= results[i]) {
                std::cout << "  failure !" << std::endl;
                std::cout << "  results[" << i << "] = " << results[i] << " out of range (0," << N-1 << ")"<< std::endl;
            }
        }

        const thrust::host_vector<float> histo_test = compute_histogram(results, N);
        thrust::host_vector<float> histo_true(N); 
        histo_true[0] = float( 1) / 20; // = 0.05
        histo_true[1] = float( 2) / 20; // = 0.10
        histo_true[2] = float( 3) / 20; // = 0.15
        histo_true[3] = float(10) / 20; // = 0.50
        histo_true[4] = float( 4) / 20; // = 0.20

        const float max_diff = max_diff_between_histo(histo_test, histo_true);
        if(max_diff < 0.01) {
            std::cout << "success" << std::endl;
        }
        else {
            std::cout << "  failure !" << std::endl;
            std::cout << "  max diff histogram = " << max_diff << std::endl;
        }
    }
    std::cout << "Test 2" << std::endl;
    {
        const int N = 1000;
        const int M = 10000;
        thrust::host_vector<int> scores = read_from_file<int>("data/ex3_scores.txt", N);

        const thrust::host_vector<int> results = random_sample(scores, M);
        if(results.size() != M) {
            std::cout << "  failure !" << std::endl;
            std::cout << "  results size = " << results.size() << std::endl;
            std::cout << "  expected     = " << M << std::endl;
        }
        for(int i = 0; i < M; ++i) {
            if(results[i] < 0 or N <= results[i]) {
                std::cout << "  failure !" << std::endl;
                std::cout << "  results[" << i << "] = " << results[i] << " out of range (0," << N-1 << ")"<< std::endl;
            }
        }

        const thrust::host_vector<float> histo_test = compute_histogram(results, N);
        thrust::host_vector<float> histo_true = read_from_file<float>("data/ex3_histo1.txt", N);

        const float max_diff = max_diff_between_histo(histo_test, histo_true);
        if(max_diff < 0.01) {
            std::cout << "success" << std::endl;
        }
        else {
            std::cout << "  failure !" << std::endl;
            std::cout << "  max diff histogram = " << max_diff << std::endl;
        }
    }
    std::cout << "Test 3" << std::endl;
    {
        const int N = 20000;
        const int M = 10000;
        thrust::host_vector<int> scores = read_from_file<int>("data/ex3_scores.txt", N);

        const thrust::host_vector<int> results = random_sample(scores, M);
        if(results.size() != M) {
            std::cout << "  failure !" << std::endl;
            std::cout << "  results size = " << results.size() << std::endl;
            std::cout << "  expected     = " << M << std::endl;
        }
        for(int i = 0; i < M; ++i) {
            if(results[i] < 0 or N <= results[i]) {
                std::cout << "  failure !" << std::endl;
                std::cout << "  results[" << i << "] = " << results[i] << " out of range (0," << N-1 << ")"<< std::endl;
            }
        }

        const thrust::host_vector<float> histo_test = compute_histogram(results, N);
        thrust::host_vector<float> histo_true = read_from_file<float>("data/ex3_histo2.txt", N);

        const float max_diff = max_diff_between_histo(histo_test, histo_true);
        if(max_diff < 0.01) {
            std::cout << "success" << std::endl;
        }
        else {
            std::cout << "  failure !" << std::endl;
            std::cout << "  max diff histogram = " << max_diff << std::endl;
        }
    }
}