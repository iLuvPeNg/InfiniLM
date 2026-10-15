#pragma once

#include "deepseek_v4_attention.hpp"
#include "deepseek_v4_moe.hpp"

#include "infinicore/nn/rmsnorm.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4MarkovHead : public infinicore::nn::Module {
public:
    DeepseekV4MarkovHead(size_t vocab_size,
                         const infinicore::DataType &dtype,
                         const infinicore::Device &device);

private:
    INFINICORE_NN_MODULE(infinilm::nn::Linear, markov_w1);
    INFINICORE_NN_MODULE(infinilm::nn::Linear, markov_w2);
};

class DeepseekV4ConfidenceHead : public infinicore::nn::Module {
public:
    DeepseekV4ConfidenceHead(size_t in_features,
                             const infinicore::DataType &dtype,
                             const infinicore::Device &device);

private:
    INFINICORE_NN_MODULE(infinilm::nn::Linear, proj);
};

class DeepseekV4DecoderLayer : public infinicore::nn::Module {
public:
    DeepseekV4DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device,
                           int compress_ratio,
                           bool has_tid2eid,
                           bool has_gate_bias,
                           size_t layer_idx,
                           MtpKind mtp_kind);

    void ingest_expert(size_t expert_id,
                       const std::string &proj,
                       bool is_scale,
                       const infinicore::Tensor &tensor);
    infinicore::Tensor forward(infinicore::Tensor residual,
                               const std::vector<int64_t> &positions,
                               const std::vector<int64_t> &token_ids) const;

private:
    int hc_mult_{4};
    double rms_eps_{1e-6};
    double hc_eps_{1e-6};
    int sinkhorn_iters_{20};
    INFINICORE_NN_MODULE(DeepseekV4Attention, attn);
    INFINICORE_NN_MODULE(DeepseekV4MoE, ffn);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, attn_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, ffn_norm);
    INFINICORE_NN_PARAMETER(hc_attn_fn);
    INFINICORE_NN_PARAMETER(hc_attn_base);
    INFINICORE_NN_PARAMETER(hc_attn_scale);
    INFINICORE_NN_PARAMETER(hc_ffn_fn);
    INFINICORE_NN_PARAMETER(hc_ffn_base);
    INFINICORE_NN_PARAMETER(hc_ffn_scale);

    INFINICORE_NN_MODULE(infinilm::nn::Linear, main_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, main_norm);
    INFINICORE_NN_MODULE(DeepseekV4MarkovHead, markov_head);
    INFINICORE_NN_MODULE(DeepseekV4ConfidenceHead, confidence_head);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    INFINICORE_NN_PARAMETER(hc_head_fn);
    INFINICORE_NN_PARAMETER(hc_head_base);
    INFINICORE_NN_PARAMETER(hc_head_scale);
};

} // namespace infinilm::models::deepseek_v4
