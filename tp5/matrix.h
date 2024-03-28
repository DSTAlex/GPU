#pragma once

#include <vector>
#include <iostream>

constexpr int     threads_per_bloc = 16;
constexpr int T = threads_per_bloc;

//
// CPU
//
std::vector<int> matmul1(
    const std::vector<int>& A,
    const std::vector<int>& B,
    int N, int M, int P);

//
// GPU
//
std::vector<int> matmul2(
    const std::vector<int>& A,
    const std::vector<int>& B,
    int N, int M, int P);

//
// GPU by bloc
//
std::vector<int> matmul3(
    const std::vector<int>& A,
    const std::vector<int>& B,
    int N, int M, int P);