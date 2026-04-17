#include"util.h"

using namespace std;

std::vector<int64_t>  record_time(vector<string> names, vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timePoints, std::vector<int64_t> records, FILE* fptr) {
    for (int j = 0; j < names.size(); j++) {
        int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(timePoints[j + 1] - timePoints[j]).count();
        records[j] += ns;
        fprintf(fptr, "%lld \t", ns);
    }
    fprintf(fptr, "\n");
    return records;
}

void record_config(vector<string> names, int M_ROW, int N_COL, int K_LEN, float Sparsity, float Variation, FILE* fptr) {
    fprintf(fptr, "Benchmarking M=%d K=%d N=%d Sparsity=%f Variation=%f \n", M_ROW, K_LEN, N_COL, Sparsity, Variation);
    for (int n = 0; n < names.size(); n++) {
        fprintf(fptr, names[n].c_str());
    }
    fprintf(fptr, "\n");
}

std::vector<float> print_ms_speedup(vector<string> names, int M_ROW, int N_COL, int K_LEN, float Sparsity, float Variation, std::vector<float> speedups, std::vector<int64_t> records, FILE* fptr) {
    std::cout << "M=" << M_ROW << ", K=" << K_LEN << ", N=" << N_COL << ", Sparsity=" << std::fixed << std::setprecision(2) << Sparsity << " +/- " << Variation << std::endl;
    int64_t baseline = records[0];
    for (int n = 0; n < names.size(); n++) {
        float speedup = (double)baseline / (double)records[n];
        speedups.push_back(speedup);
        std::cout << names[n] << records[n] / 1000000 << "\t ms, speedup = " << std::fixed << std::setprecision(2) << speedup << std::endl;
        fprintf(fptr, "%lld\t", records[n] / 1000000);
    }
    std::cout << "\n" << std::endl;
    fprintf(fptr, "ms\n");
    for (int n = 0; n < names.size(); n++) {
        float speedup = (double)baseline / (double)records[n];
        fprintf(fptr, "%.2f\t", speedup);
    }
    fprintf(fptr, "speedup\n");
    return speedups;
}

void print_speedup_summary(vector<string> names, std::vector<std::tuple<int, int, int, float, float>> Config_MKNSV, std::vector<float> speedups) {
    std::cout << "speedup" << std::endl;
    for (int n = 0; n < names.size(); n++) {
        std::cout << names[n];
    }
    std::cout << std::endl;
    for (int i = 0; i < Config_MKNSV.size(); i++) {
        std::cout << "M=" << std::get<0>(Config_MKNSV[i]) << " K=" << std::get<1>(Config_MKNSV[i]) << " N=" << std::get<2>(Config_MKNSV[i]) << " S=" << std::fixed << std::setprecision(2) << std::get<3>(Config_MKNSV[i]) << " +/- " << std::get<4>(Config_MKNSV[i]) << "\t";
        for (int n = 0; n < names.size(); n++)
            std::cout << speedups[i * names.size() + n] << "\t";
        std::cout << std::endl;
    }
}

std::string time_string() {
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm* p_tm = std::localtime(&now_c);
    if (p_tm == nullptr) {
        return "2025-01-01_00-00-00";
    }
    std::ostringstream oss;
    oss << std::put_time(p_tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}
