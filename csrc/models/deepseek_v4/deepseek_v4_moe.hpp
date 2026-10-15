#pragma once

#include "deepseek_v4_common.hpp"

#include "../../config/model_config.hpp"
#include "infinicore/nn/module.hpp"

#include <cstdint>
#include <vector>

#include <infiniccl.h>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4Gate : public infinicore::nn::Module {
public:
    DeepseekV4Gate(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                   const infinicore::Device &device,
                   bool has_tid2eid,
                   bool has_bias);

    infinicore::Tensor route_weight() const { return weight_; }
    infinicore::Tensor tid2eid() const { return tid2eid_; }
    infinicore::Tensor bias() const { return bias_; }

private:
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(tid2eid);
    INFINICORE_NN_PARAMETER(bias);
};

class DeepseekV4SharedExperts : public infinicore::nn::Module {
public:
    DeepseekV4SharedExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            const infinicore::Device &device);

    infinicore::Tensor forward(infinicore::Tensor hidden) const;

private:
    float swiglu_limit_{10.0f};
    INFINICORE_NN_MODULE(infinilm::nn::ColumnParallelLinear, w1);
    INFINICORE_NN_MODULE(infinilm::nn::RowParallelLinear, w2);
    INFINICORE_NN_MODULE(infinilm::nn::ColumnParallelLinear, w3);
};

// Routed experts are int4-packed. Each rank stores only its expert slice.
class DeepseekV4MoE : public infinicore::nn::Module {
public:
    DeepseekV4MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  const infinicore::Device &device,
                  bool has_tid2eid,
                  bool has_bias);

    void ingest_expert(size_t expert_id,
                       const std::string &proj,
                       bool is_scale,
                       const infinicore::Tensor &tensor);
    infinicore::Tensor forward(infinicore::Tensor hidden, const std::vector<int64_t> &token_ids) const;

private:
    void copy_slot_(infinicore::Tensor packed,
                    size_t local_index,
                    const infinicore::Tensor &src,
                    const std::string &what) const;

    int tp_rank_{0};
    int tp_size_{1};
    size_t n_experts_{0};
    size_t hidden_{0};
    size_t intermediate_{0};
    size_t topk_{0};
    float routed_scale_{1.5f};
    float swiglu_limit_{10.0f};
    bool renormalize_{true};
    bool hash_{false};
    bool has_bias_{false};
    infinicclComm_t comm_{nullptr};
    mutable infinicore::Tensor gate_t_;
    mutable std::vector<float> bias_cpu_;
    mutable std::vector<int64_t> tid_cpu_;

    INFINICORE_NN_MODULE(DeepseekV4Gate, gate);
    INFINICORE_NN_MODULE(DeepseekV4SharedExperts, shared_experts);
    INFINICORE_NN_BUFFER(w1_packed);
    INFINICORE_NN_BUFFER(w1_scale_packed);
    INFINICORE_NN_BUFFER(w2_packed);
    INFINICORE_NN_BUFFER(w2_scale_packed);
    INFINICORE_NN_BUFFER(w3_packed);
    INFINICORE_NN_BUFFER(w3_scale_packed);
};

} // namespace infinilm::models::deepseek_v4
