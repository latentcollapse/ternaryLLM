#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <filesystem>
#include "original_rsr/naive.h"
#include "original_rsr/utils.h"
#include "original_rsr/rsr.h"
#include "original_rsr/rsrpp.h"
#include "ssr.h"
#include "ssr_vec.h"
#include "ssr_tree.h"
#include "ssr_unroll.h"

using namespace std;
using namespace std::chrono;

// Helper functions for matrix file I/O
void saveMatrixToFile(const vector<vector<int8_t>>& matrix, const string& filename) {
    ofstream file(filename);  // Text mode, overwrites existing file
    if (!file.is_open()) {
        throw runtime_error("Cannot open file for writing: " + filename);
    }
    
    int rows = matrix.size();
    int cols = matrix[0].size();
    
    // Write dimensions as first line
    file << rows << " " << cols << "\n";
    
    // Write matrix data
    for (const auto& row : matrix) {
        for (int j = 0; j < cols; ++j) {
            if (j > 0) file << " ";
            file << static_cast<int>(row[j]);  // Cast int8_t to int for display
        }
        file << "\n";
    }
    file.close();
}

void saveMatrixToFile(const vector<int8_t>& matrix, const string& filename) {
    ofstream file(filename);  // Text mode, overwrites existing file
    if (!file.is_open()) {
        throw runtime_error("Cannot open file for writing: " + filename);
    }

    int rows = matrix.size();

    // Write dimensions as first line
    file << rows << "\n";

    // Write matrix data
    for (int j = 0; j < rows; j++) {
        if (j > 0) file << " ";
        file << static_cast<int>(matrix[j]);  // Cast int8_t to int for display
    }
    file.close();
}

void saveMatrixToFile(const vector<float>& matrix, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file for writing: " + filename);
    }

    int rows = matrix.size();

    // Write dimensions as first line
    file << rows << "\n";

    // Set precision for float output
    file << fixed << setprecision(6);

    // Write matrix data
    for (int j = 0; j < rows; j++) {
        if (j > 0) file << " ";
        file << matrix[j];  // Direct output, no casting
    }
    file.close();
}

string generateFilename(int n, int m, double p, const string& type) {
    ostringstream oss;
    if (type == "v") {
        // v matrix has no sparsity parameter
        oss << "matrices/" << type << "_" << n << "x" << m << ".txt";
    } else {
        // mat matrix includes sparsity
        oss << "matrices/" << type << "_" << n << "x" << m << "_p" << fixed << setprecision(2) << p << ".txt";
    }
    return oss.str();
}
