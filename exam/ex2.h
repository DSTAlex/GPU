#pragma once

#include <vector>
#include <iostream>

//
// CPU
//
std::vector<int> matvecmul1(
    const std::vector<int>& A,
    const std::vector<int>& b);

//
// GPU
//
std::vector<int> matvecmul2(
    const std::vector<int>& A,
    const std::vector<int>& b);

//
// GPU by bloc
//
std::vector<int> matvecmul3(
    const std::vector<int>& A,
    const std::vector<int>& b);