#pragma once
#include <iostream>
#include <sstream> // Required for std::ostringstream
#include <string>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <iomanip> // This is the one you're missing!
#include <ctime>
using namespace std;
std::vector<int64_t>  record_time(vector<string> names, vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timePoints, std::vector<int64_t> records, FILE* fptr);
void record_config(vector<string> names, int M_ROW, int N_COL, int K_LEN, float Sparsity, float Variation, FILE* fptr);
std::vector<float> print_ms_speedup(vector<string> names, int M_ROW, int N_COL, int K_LEN, float Sparsity, float Variation, std::vector<float> speedups, std::vector<int64_t> records, FILE* fptr);
void print_speedup_summary(vector<string> names, std::vector<std::tuple<int, int, int, float, float>> Config_MKNSV, std::vector<float> speedups);
std::string time_string();
