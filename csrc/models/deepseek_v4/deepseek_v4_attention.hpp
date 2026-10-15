#pragma once

#include "deepseek_v4_common.hpp"

#include "../../config/model_config.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace infinilm::models::deepseek_v4 {

// The compressor keeps a BF16 state. Overlap (`coff` = 2) is used when `compress_ratio` is 4.
class DeepseekV4Compressor : public infinicore::nn::Module {
public:
    DeepseekV4Compressor(size_t hidden_size,
                         size_t head_dim,
                         int compress_ratio,
                         double eps,
                         const infinicore::DataType &dtype,
                         const infinicore::Device &device);

    std::pair<infinicore::Tensor, infinicore::Tensor> project(infinicore::Tensor hidden) const;
    infinicore::Tensor ape() const { return ape_; }
    infinicore::Tensor norm_weight() const { return norm_->weight(); }
    int compress_ratio() const { return compress_ratio_; }
    size_t head_dim() const { return head_dim_; }
    size_t state_width() const { return state_width_; }

private:
    int compress_ratio_{0};
    size_t head_dim_{0};
    size_t state_width_{0};
    INFINICORE_NN_PARAMETER(ape);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    INFINICORE_NN_MODULE(infinilm::nn::Linear, wkv);
    INFINICORE_NN_MODULE(infinilm::nn::Linear, wgate);
};

class DeepseekV4Indexer : public infinicore::nn::Module {
public:
    DeepseekV4Indexer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      const infinicore::Device &device,
                      int compress_ratio);

private:
    INFINICORE_NN_MODULE(infinilm::nn::Linear, wq_b);
    INFINICORE_NN_MODULE(infinilm::nn::Linear, weights_proj);
    INFINICORE_NN_MODULE(DeepseekV4Compressor, compressor);
};

class DeepseekV4Attention : public infinicore::nn::Module {
public:
    DeepseekV4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                        const infinicore::Device &device,
                        int compress_ratio,
                        size_t layer_idx);

    infinicore::Tensor forward(infinicore::Tensor hidden, const std::vector<int64_t> &positions) const;

private:
    size_t layer_idx_{0};
    size_t local_heads_{0};
    size_t head_dim_{0};
    size_t rope_dim_{0};
    size_t local_groups_{0};
    size_t o_lora_{0};
    int compress_ratio_{0};
    int sliding_window_{128};
    int index_topk_{512};
    double eps_{1e-6};
    double rope_theta_{10000.0};
    double compress_rope_theta_{160000.0};
    float scale_{1.0f};
    mutable infinicore::Tensor rope_cache_;
    mutable infinicore::Tensor compress_rope_cache_;
    mutable std::vector<uint8_t> compressed_valid_;
    mutable std::vector<float> compress_norm_weight_;
    INFINICORE_NN_PARAMETER(attn_sink);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, kv_norm);
    INFINICORE_NN_MODULE(infinilm::nn::Linear, wq_a);
    INFINICORE_NN_MODULE(infinilm::nn::ColumnParallelLinear, wq_b);
    INFINICORE_NN_MODULE(infinilm::nn::Linear, wkv);
    INFINICORE_NN_MODULE(infinilm::nn::ColumnParallelLinear, wo_a);
    INFINICORE_NN_MODULE(infinilm::nn::RowParallelLinear, wo_b);
    std::shared_ptr<DeepseekV4Compressor> compressor_;
    std::shared_ptr<DeepseekV4Indexer> indexer_;
};

} // namespace infinilm::models::deepseek_v4
