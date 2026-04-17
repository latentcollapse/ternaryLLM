#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
 
#include "TCSC.hpp"
#include "initData.hpp"
#include "SSR.hpp"
#include <torch/torch.h>
using namespace std;

class LlamaLayer {
public:
    torch::nn::Linear q_proj = nullptr;
    torch::nn::Linear k_proj = nullptr;
    torch::nn::Linear v_proj = nullptr;
    torch::nn::MultiheadAttention MHA = nullptr;
    torch::nn::Linear o_proj = nullptr;
    torch::nn::Linear up_proj = nullptr;
    torch::nn::Linear gate_proj = nullptr;
    torch::nn::Linear down_proj = nullptr;
    int QHEADS;
    int KVHEADS;
    int HEAD_SIZE;
    //int EMB_SIZE;
    //int IMM_SIZE;
    LlamaLayer(int QHEADS, int KVHEADS, int EMBEDDING_SIZE, int INTERMEDIATE_SIZE) {
        this->QHEADS = QHEADS;
        this->KVHEADS = KVHEADS;
        this->HEAD_SIZE = EMBEDDING_SIZE / QHEADS;
        int KV_SIZE = this->HEAD_SIZE * KVHEADS;
        this->q_proj = torch::nn::Linear(EMBEDDING_SIZE, EMBEDDING_SIZE);
        this->k_proj = torch::nn::Linear(EMBEDDING_SIZE, KV_SIZE);
        this->v_proj = torch::nn::Linear(EMBEDDING_SIZE, KV_SIZE);
        this->MHA = torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(EMBEDDING_SIZE, QHEADS));
        this->o_proj = torch::nn::Linear(EMBEDDING_SIZE, EMBEDDING_SIZE);
        this->up_proj = torch::nn::Linear(EMBEDDING_SIZE, INTERMEDIATE_SIZE);
        this->gate_proj = torch::nn::Linear(EMBEDDING_SIZE, INTERMEDIATE_SIZE);
        this->down_proj = torch::nn::Linear(INTERMEDIATE_SIZE, EMBEDDING_SIZE);
    }

    torch::Tensor forward(torch::Tensor X) {
        torch::Tensor X_Residual = X;
        torch::Tensor Query = this->q_proj->forward(X);
        torch::Tensor Key = this->k_proj->forward(X).repeat_interleave(this->QHEADS / this->KVHEADS, 2);
        torch::Tensor Value = this->v_proj->forward(X).repeat_interleave(this->QHEADS / this->KVHEADS, 2);
        // torch::nn::functional::multi_head_attention_forward(Query, Key, Value, torch::nn::MultiheadAttentionOptions(EMBEDDING_SIZE, QHEADS));
        auto Output = this->MHA->forward(Query, Key, Value);
        torch::Tensor Y = this->o_proj->forward(std::get<0>(Output));
        X = Y + X_Residual;
        X_Residual = X;
        torch::Tensor Up = this->up_proj->forward(X);
        torch::Tensor Gate = this->gate_proj->forward(X);
        Gate = torch::nn::functional::silu(Gate);
        Gate = torch::mul(Gate, Up);
        Y = this->down_proj->forward(Gate);
        X = Y + X_Residual;
        return X;
    }
};

class LlamaModel {
public:
    vector<LlamaLayer> layers;

    LlamaModel(int LAYERS, int QHEADS, int KVHEADS, int EMBEDDING_SIZE, int INTERMEDIATE_SIZE) {
        for (int i = 0; i < LAYERS; i++) {
            auto layer = LlamaLayer(QHEADS, KVHEADS, EMBEDDING_SIZE, INTERMEDIATE_SIZE);
            this->layers.push_back(layer);
        }
    }

    torch::Tensor forward(torch::Tensor X) {
        torch::Tensor Y = X;
        for (int i = 0; i < this->layers.size(); i++) {
            Y = this->layers[i].forward(Y);
        }
        return Y;
    }
};


class SSRMLP {
public:
    vector<int> perPosU;
    vector<int> segPosU;
    vector<int> perNegU;
    vector<int> segNegU;
    vector<int> perPosG;
    vector<int> segPosG;
    vector<int> perNegG;
    vector<int> segNegG;
    vector<int> perPosD;
    vector<int> segPosD;
    vector<int> perNegD;
    vector<int> segNegD;
    int K_LEN;
    int N_COL;
    int SSR_K;
    SSRMLP() {}
    SSRMLP(int K_LEN, int N_COL, float Sparsity) {
        this->SSR_K = 6;
        // Initialize the weights
        vector<vector<int8_t>> W_SSRU = generateTernaryRandomMatrix(K_LEN, N_COL, Sparsity);
        vector<vector<int8_t>> W_SSRG = generateTernaryRandomMatrix(K_LEN, N_COL, Sparsity);
        vector<vector<int8_t>> W_SSRD = generateTernaryRandomMatrix(N_COL, K_LEN, Sparsity);
        auto W_SSRU_TRANS2 = ssr_preprocess(W_SSRU, SSR_K, K_LEN, N_COL);
        auto W_SSRG_TRANS2 = ssr_preprocess(W_SSRG, SSR_K, K_LEN, N_COL);
        auto W_SSRD_TRANS2 = ssr_preprocess(W_SSRD, SSR_K, N_COL, K_LEN);
        this->perPosU = W_SSRU_TRANS2.first.first;
        this->segPosU = W_SSRU_TRANS2.first.second;
        this->perNegU = W_SSRU_TRANS2.second.first;
        this->segNegU = W_SSRU_TRANS2.second.second;
        this->perPosG = W_SSRG_TRANS2.first.first;
        this->segPosG = W_SSRG_TRANS2.first.second;
        this->perNegG = W_SSRG_TRANS2.second.first;
        this->segNegG = W_SSRG_TRANS2.second.second;
        this->perPosD = W_SSRD_TRANS2.first.first;
        this->segPosD = W_SSRD_TRANS2.first.second;
        this->perNegD = W_SSRD_TRANS2.second.first;
        this->segNegD = W_SSRD_TRANS2.second.second;
    }

    void forward(vector<float> X, vector<float> Ya, vector<float> Yb, int M_ROW, int N_COL, int K_LEN) {
        Ya = ssr_inference_float_vec_L8(X, this->perPosU, this->segPosU, this->perNegU, this->segNegU, K_LEN, N_COL, M_ROW);
        Yb = ssr_inference_float_vec_L8(X, this->perPosG, this->segPosG, this->perNegG, this->segNegG, K_LEN, N_COL, M_ROW);
        Naive_SiLU_Dot_Unroll_AVX2(M_ROW * N_COL, Yb.data(), Ya.data());
        X = ssr_inference_float_vec_L8(Yb, this->perPosD, this->segPosD, this->perNegD, this->segNegD, N_COL, K_LEN, M_ROW);
    }
};


class SSRLlamaLayer {
public:
    torch::nn::Linear q_proj = nullptr;
    torch::nn::Linear k_proj = nullptr;
    torch::nn::Linear v_proj = nullptr;
    torch::nn::MultiheadAttention MHA = nullptr;
    torch::nn::Linear o_proj = nullptr;
    SSRMLP MLP;
    int QHEADS;
    int KVHEADS;
    int HEAD_SIZE;
    //int EMB_SIZE;
    int IMM_SIZE;
    SSRLlamaLayer(int QHEADS, int KVHEADS, int EMBEDDING_SIZE, int INTERMEDIATE_SIZE, float Sparsity) {
        this->QHEADS = QHEADS;
        this->KVHEADS = KVHEADS;
        this->HEAD_SIZE = EMBEDDING_SIZE / QHEADS;
        this->IMM_SIZE = INTERMEDIATE_SIZE;
        int KV_SIZE = this->HEAD_SIZE * KVHEADS;
        this->q_proj = torch::nn::Linear(EMBEDDING_SIZE, EMBEDDING_SIZE);
        this->k_proj = torch::nn::Linear(EMBEDDING_SIZE, KV_SIZE);
        this->v_proj = torch::nn::Linear(EMBEDDING_SIZE, KV_SIZE);
        this->MHA = torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(EMBEDDING_SIZE, QHEADS));
        this->o_proj = torch::nn::Linear(EMBEDDING_SIZE, EMBEDDING_SIZE);
        this->MLP = SSRMLP(EMBEDDING_SIZE, INTERMEDIATE_SIZE, Sparsity);
    }

    torch::Tensor forward(torch::Tensor X, vector<float> XT, vector<float> Ya, vector<float> Yb) {
        torch::Tensor X_Residual = X.contiguous();
        torch::Tensor Query = this->q_proj->forward(X);
        torch::Tensor Key = this->k_proj->forward(X).repeat_interleave(this->QHEADS / this->KVHEADS, 2);
        torch::Tensor Value = this->v_proj->forward(X).repeat_interleave(this->QHEADS / this->KVHEADS, 2);
        // torch::nn::functional::multi_head_attention_forward(Query, Key, Value, torch::nn::MultiheadAttentionOptions(EMBEDDING_SIZE, QHEADS));
        auto Output = this->MHA->forward(Query, Key, Value);
        torch::Tensor Y = this->o_proj->forward(std::get<0>(Output));
        X = Y + X_Residual;
        X_Residual = X;
        int M_ROW = X.sizes()[0] * X.sizes()[1];
        int K_LEN = X.sizes()[2];
        XT.assign(X.data_ptr<float>(), X.data_ptr<float>() + M_ROW * K_LEN); // Use dimension sizes after permute
        this->MLP.forward(XT, Ya, Yb, M_ROW, this->IMM_SIZE, K_LEN);
        std::memcpy(X.data_ptr<float>(), XT.data(), M_ROW * K_LEN * sizeof(float));
        X = X + X_Residual;
        return X;
    }
};


class SSRLlamaModel {
public:
    vector<SSRLlamaLayer> layers;
    int INTERMEDIATE_SIZE;

    SSRLlamaModel(int LAYERS, int QHEADS, int KVHEADS, int EMBEDDING_SIZE, int INTERMEDIATE_SIZE, float Sparsity) {
        this->INTERMEDIATE_SIZE = INTERMEDIATE_SIZE;
        for (int i = 0; i < LAYERS; i++) {
            auto layer = SSRLlamaLayer(QHEADS, KVHEADS, EMBEDDING_SIZE, INTERMEDIATE_SIZE, Sparsity);
            this->layers.push_back(layer);
        }
    }

    torch::Tensor forward(torch::Tensor X) {
        int vlen = X.sizes()[0] * X.sizes()[1] * this->INTERMEDIATE_SIZE;
        vector<float> XT(X.sizes()[0] * X.sizes()[1] * X.sizes()[2], 0.0);
        vector<float> YA(vlen, 0.0);
        vector<float> YB(vlen, 0.0);
        for (int i = 0; i < this->layers.size(); i++) {
            this->layers[i].forward(X, XT, YA, YB);
        }
        return X;
    }
};



int benchmark_FP32_Llama_Torch_Dense(const int NUM_RUNS = 10, const int NUM_FUNCTIONS = 30) {
    std::vector<std::tuple<float, float>> Config_SV = { // Sparsity and data Variation
        {0.50,0.05},
        /*{0.60,0.05},
        {0.70,0.05},
        {0.80,0.05},*/
        {0.90,0.05},
    };
    std::vector<std::tuple<int, int, int, int, int>> Config_LKNHH = {
        {  16, 2048,  8192, 32, 8},
        // {  28, 3072,  8192, 24, 8},
        // {  32, 4096, 14336, 32, 8},
    };
    std::vector<std::tuple<int, int>> Config_BS = {
        //{  1,    1},
        //{  1,   64},
        //{  1,  128},
        {  1,  256},
        //{  1,  512},
        //{  1, 1024},
    };

    std::cout << "Running performance" << std::endl;
    string file_name = "GEMM_CPU_FP32_Layer_Level_LlamaMLP_" + time_string() + ".txt";
    FILE* fptr = fopen(file_name.c_str(), "w");
    vector<string> names = { "TernaryLlama\t", };
    std::vector<float> speedups;
    for (const auto& [LAYERS, K_LEN, N_COL, QHEADS, KVHEADS] : Config_LKNHH) {
        for (const auto& [BS, M_ROW] : Config_BS) {
            LlamaModel model = LlamaModel(LAYERS, QHEADS, KVHEADS, K_LEN, N_COL);
            torch::Tensor X = torch::rand({ BS, M_ROW, K_LEN }).contiguous();

            std::vector<int64_t> records(NUM_FUNCTIONS, 0);
            fprintf(fptr, "Benchmarking L=%d M=%d K=%d N=%d \n", LAYERS, M_ROW, K_LEN, N_COL);
            for (int n = 0; n < names.size(); n++) {
                fprintf(fptr, names[n].c_str());
            }
            fprintf(fptr, "\n");
            for (int i = 0; i < NUM_RUNS; i++) {
                vector<std::chrono::time_point<std::chrono::high_resolution_clock>> timePoints(NUM_FUNCTIONS);
                int j = 0;
                timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
                model.forward(X);
                timePoints[j] = std::chrono::high_resolution_clock::now(); j++;
                for (int j = 0; j < names.size(); j++) {
                    int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(timePoints[j + 1] - timePoints[j]).count();
                    records[j] += ns;
                    fprintf(fptr, "%lld \t", ns);
                }
                fprintf(fptr, "\n");
            }
            std::cout << "L=" << LAYERS << ", M=" << M_ROW << ", K=" << K_LEN << ", N=" << N_COL << std::endl;
            int64_t baseline = records[0];
            for (int n = 0; n < names.size(); n++) {
                float speedup = (double)baseline / (double)records[n];
                speedups.push_back(speedup);
                std::cout << names[n] << records[n] / 1000000 << "\t ms, speedup = " << std::fixed << std::setprecision(2) << speedup << std::endl;
                fprintf(fptr, "%lld\t", records[n] / 1000000);
            }
            std::cout << "\n" << std::endl;
        }
    }

    fclose(fptr);
    return 0;
}
