#include "deepseek_v4_moe.hpp"

#include "../../global_state/global_state.hpp"
#include "deepseek_v4_runtime.hpp"

#include "infinicore/ops/add.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/mul_scalar.hpp"
#include "infinicore/ops/per_channel_quant_i8.hpp"
#include "infinicore/ops/w4a8_group_gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::deepseek_v4 {

DeepseekV4Gate::DeepseekV4Gate(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               const infinicore::Device &device,
                               bool has_tid2eid,
                               bool has_bias) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden = model_config->get<size_t>("hidden_size");
    const size_t n_experts = model_config->get<size_t>("n_routed_experts");
    const size_t topk = model_config->get<size_t>("num_experts_per_tok");
    const size_t vocab = model_config->get<size_t>("vocab_size");
    INFINICORE_NN_PARAMETER_INIT(weight, ({n_experts, hidden}, dtype, device));
    if (has_tid2eid) {
        INFINICORE_NN_PARAMETER_INIT(tid2eid, ({vocab, topk}, infinicore::DataType::I64, device));
    }
    if (has_bias) {
        INFINICORE_NN_PARAMETER_INIT(bias, ({n_experts}, infinicore::DataType::F32, device));
    }
}

DeepseekV4SharedExperts::DeepseekV4SharedExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                 const infinicore::Device &device) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden = model_config->get<size_t>("hidden_size");
    const size_t intermediate = model_config->get<size_t>("moe_intermediate_size");
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    auto quant = int8_quantization();
    swiglu_limit_ = static_cast<float>(model_config->get<double>("swiglu_limit"));
    INFINICORE_NN_MODULE_INIT(w1, hidden, intermediate, quant, false, dtype, device, rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(w3, hidden, intermediate, quant, false, dtype, device, rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(w2, intermediate, hidden, quant, false, dtype, device, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
}

DeepseekV4MoE::DeepseekV4MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device,
                             bool has_tid2eid,
                             bool has_bias) {
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_rank_ = rank_info.tp_rank;
    tp_size_ = rank_info.tp_size;
    comm_ = rank_info.comm;
    n_experts_ = model_config->get<size_t>("n_routed_experts");
    hidden_ = model_config->get<size_t>("hidden_size");
    intermediate_ = model_config->get<size_t>("moe_intermediate_size");
    topk_ = model_config->get<size_t>("num_experts_per_tok");
    routed_scale_ = static_cast<float>(model_config->get<double>("routed_scaling_factor"));
    swiglu_limit_ = static_cast<float>(model_config->get<double>("swiglu_limit"));
    renormalize_ = model_config->get<bool>("norm_topk_prob");
    hash_ = has_tid2eid;
    has_bias_ = has_bias;
    const size_t hidden = hidden_;
    const size_t intermediate = intermediate_;
    if (tp_size_ <= 0 || n_experts_ % static_cast<size_t>(tp_size_) != 0) {
        throw std::runtime_error("deepseek_v4: n_routed_experts must be divisible by tp size");
    }
    if (hidden % 2 != 0 || intermediate % 2 != 0) {
        throw std::runtime_error("deepseek_v4: hidden and moe intermediate sizes must be even for packed int4 experts");
    }
    const size_t local = n_experts_ / static_cast<size_t>(tp_size_);
    const size_t w1_k = hidden / 2;
    const size_t w2_k = intermediate / 2;

    INFINICORE_NN_MODULE_INIT(gate, model_config, device, has_tid2eid, has_bias);
    INFINICORE_NN_MODULE_INIT(shared_experts, model_config, device);
    // Checkpoint scales are [1, N] (TN). Packed expert weights are [E, N, K/2].
    INFINICORE_NN_BUFFER_INIT(w1_packed, ({local, intermediate, w1_k}, infinicore::DataType::I8, device));
    INFINICORE_NN_BUFFER_INIT(w1_scale_packed, ({local, intermediate, size_t{1}}, infinicore::DataType::F32, device));
    INFINICORE_NN_BUFFER_INIT(w2_packed, ({local, hidden, w2_k}, infinicore::DataType::I8, device));
    INFINICORE_NN_BUFFER_INIT(w2_scale_packed, ({local, hidden, size_t{1}}, infinicore::DataType::F32, device));
    INFINICORE_NN_BUFFER_INIT(w3_packed, ({local, intermediate, w1_k}, infinicore::DataType::I8, device));
    INFINICORE_NN_BUFFER_INIT(w3_scale_packed, ({local, intermediate, size_t{1}}, infinicore::DataType::F32, device));
}

void DeepseekV4MoE::copy_slot_(infinicore::Tensor packed,
                               size_t local_index,
                               const infinicore::Tensor &src,
                               const std::string &what) const {
    try {
        packed->narrow({{0, local_index, 1}})->view(src->shape())->copy_from(src);
    } catch (const std::exception &err) {
        throw std::runtime_error("deepseek_v4 expert copy failed for " + what + ": " + err.what());
    }
}

void DeepseekV4MoE::ingest_expert(size_t expert_id,
                                  const std::string &proj,
                                  bool is_scale,
                                  const infinicore::Tensor &tensor) {
    if (expert_id >= n_experts_) {
        throw std::runtime_error("deepseek_v4 expert id out of range");
    }
    if (expert_id % static_cast<size_t>(tp_size_) != static_cast<size_t>(tp_rank_)) {
        return;
    }
    const size_t local_index = expert_id / static_cast<size_t>(tp_size_);
    infinicore::Tensor packed;
    if (proj == "w1") {
        packed = is_scale ? w1_scale_packed_ : w1_packed_;
    } else if (proj == "w2") {
        packed = is_scale ? w2_scale_packed_ : w2_packed_;
    } else if (proj == "w3") {
        packed = is_scale ? w3_scale_packed_ : w3_packed_;
    } else {
        throw std::runtime_error("deepseek_v4 unknown expert projection " + proj);
    }
    infinicore::Tensor src = tensor;
    if (is_scale) {
        if (src->ndim() != 2) {
            throw std::runtime_error("deepseek_v4 expert scale rank is not 2");
        }
        src = src->permute({1, 0});
        if (!src->is_contiguous()) {
            src = src->contiguous();
        }
    }
    copy_slot_(packed, local_index, src, proj + (is_scale ? ".weight_scale" : ".weight"));
}

infinicore::Tensor DeepseekV4SharedExperts::forward(infinicore::Tensor hidden) const {
    auto gate = int8_gemm(hidden, w1_->weight(), w1_->weight_scale());
    auto up = int8_gemm(hidden, w3_->weight(), w3_->weight_scale());
    auto mid = swiglu_clamp(gate, up, swiglu_limit_);
    auto out = int8_gemm(mid, w2_->weight(), w2_->weight_scale());
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (rank_info.tp_size > 1 && rank_info.comm != nullptr) {
        infinicore::op::distributed::allreduce_(out, out, INFINICCL_SUM, rank_info.comm);
    }
    return out;
}

infinicore::Tensor DeepseekV4MoE::forward(infinicore::Tensor hidden, const std::vector<int64_t> &token_ids) const {
    auto x = hidden->contiguous();
    const size_t tokens = x->size(0);
    const auto device = x->device();
    const auto dtype = x->dtype();
    if (!gate_t_) {
        gate_t_ = cast_dtype(gate_->route_weight()->permute({1, 0})->contiguous(), infinicore::DataType::F32);
    }
    auto logits = download_f32(infinicore::op::matmul(cast_dtype(x, infinicore::DataType::F32), gate_t_));
    if (logits.size() != tokens * n_experts_) {
        throw std::runtime_error("deepseek_v4 moe: router score size mismatch");
    }
    for (float &value : logits) {
        const float sp = value > 20.0f ? value : (value < -20.0f ? std::exp(value) : std::log1p(std::exp(value)));
        value = std::sqrt(std::max(sp, 0.0f));
    }
    if (has_bias_ && bias_cpu_.empty()) {
        bias_cpu_ = download_f32(gate_->bias());
    }
    if (hash_ && tid_cpu_.empty()) {
        tid_cpu_ = download_i64(gate_->tid2eid());
    }

    struct Route {
        size_t token;
        size_t local;
        float weight;
    };
    std::vector<Route> routes;
    for (size_t t = 0; t < tokens; ++t) {
        std::vector<std::pair<int, float>> chosen;
        chosen.reserve(topk_);
        if (hash_) {
            if (t >= token_ids.size()) {
                throw std::runtime_error("deepseek_v4 moe: hash routing needs one token id per row");
            }
            const int64_t token_id = token_ids[t];
            if (token_id < 0 || static_cast<size_t>(token_id) * topk_ + topk_ > tid_cpu_.size()) {
                throw std::runtime_error("deepseek_v4 moe: token id is outside tid2eid");
            }
            for (size_t k = 0; k < topk_; ++k) {
                const int id = static_cast<int>(tid_cpu_[static_cast<size_t>(token_id) * topk_ + k]);
                chosen.push_back({id, logits[t * n_experts_ + static_cast<size_t>(id)]});
            }
        } else {
            std::vector<std::pair<float, int>> ranked(n_experts_);
            for (size_t e = 0; e < n_experts_; ++e) {
                const float bias = has_bias_ ? bias_cpu_[e] : 0.0f;
                ranked[e] = {logits[t * n_experts_ + e] + bias, static_cast<int>(e)};
            }
            std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(topk_), ranked.end(),
                              [](const auto &a, const auto &b) { return a.first > b.first; });
            for (size_t k = 0; k < topk_; ++k) {
                const int id = ranked[k].second;
                chosen.push_back({id, logits[t * n_experts_ + static_cast<size_t>(id)]});
            }
        }
        float denom = 0.0f;
        for (const auto &item : chosen) {
            denom += item.second;
        }
        for (auto &item : chosen) {
            if (renormalize_ && denom != 0.0f) {
                item.second /= denom;
            }
            item.second *= routed_scale_;
            if (item.first < 0 || static_cast<size_t>(item.first) >= n_experts_) {
                throw std::runtime_error("deepseek_v4 moe: expert id out of range");
            }
            if (item.first % tp_size_ != tp_rank_) {
                continue;
            }
            const size_t local_index = static_cast<size_t>(item.first) / static_cast<size_t>(tp_size_);
            routes.push_back(Route{t, local_index, item.second});
        }
    }
    std::stable_sort(routes.begin(), routes.end(), [](const Route &a, const Route &b) { return a.local < b.local; });

    auto acc = infinicore::Tensor::zeros({tokens, hidden_}, dtype, device);
    const size_t rows = routes.size();
    if (rows > 0) {
        const size_t local_experts = w1_packed_->size(0);
        std::vector<int32_t> counts(local_experts, 0);
        auto packed_act = infinicore::Tensor::empty({rows, hidden_}, dtype, device);
        for (size_t i = 0; i < rows; ++i) {
            if (routes[i].local >= local_experts) {
                throw std::runtime_error("deepseek_v4 moe: local expert index out of range");
            }
            counts[routes[i].local] += 1;
            packed_act->narrow({{0, i, 1}})->view({hidden_})->copy_from(x->narrow({{0, routes[i].token, 1}})->contiguous()->view({hidden_}));
        }
        auto tokens_per_experts = upload_i32(counts, {local_experts}, infinicore::Device::cpu());
        auto act_i8 = infinicore::Tensor::empty({rows, hidden_}, infinicore::DataType::I8, device);
        auto act_scale = infinicore::Tensor::empty({rows, size_t{1}}, infinicore::DataType::F32, device);
        infinicore::op::per_channel_quant_i8_(packed_act, act_i8, act_scale);
        auto gate = infinicore::Tensor::empty({rows, intermediate_}, dtype, device);
        auto up = infinicore::Tensor::empty({rows, intermediate_}, dtype, device);
        infinicore::op::w4a8_group_gemm_(gate, act_i8, w1_packed_->contiguous(), act_scale, w1_scale_packed_->contiguous(), tokens_per_experts);
        infinicore::op::w4a8_group_gemm_(up, act_i8, w3_packed_->contiguous(), act_scale, w3_scale_packed_->contiguous(), tokens_per_experts);
        auto mid = swiglu_clamp(gate, up, swiglu_limit_);
        auto mid_i8 = infinicore::Tensor::empty({rows, intermediate_}, infinicore::DataType::I8, device);
        auto mid_scale = infinicore::Tensor::empty({rows, size_t{1}}, infinicore::DataType::F32, device);
        infinicore::op::per_channel_quant_i8_(mid, mid_i8, mid_scale);
        auto down = infinicore::Tensor::empty({rows, hidden_}, dtype, device);
        infinicore::op::w4a8_group_gemm_(down, mid_i8, w2_packed_->contiguous(), mid_scale, w2_scale_packed_->contiguous(), tokens_per_experts);
        for (size_t i = 0; i < rows; ++i) {
            auto row = down->narrow({{0, i, 1}})->contiguous()->view({hidden_});
            auto scaled = infinicore::op::mul_scalar(row, static_cast<double>(routes[i].weight));
            const size_t token = routes[i].token;
            auto dst = acc->narrow({{0, token, 1}})->contiguous()->view({hidden_});
            acc->narrow({{0, token, 1}})->view({hidden_})->copy_from(infinicore::op::add(dst, scaled));
        }
    }
    if (tp_size_ > 1 && comm_ != nullptr) {
        infinicore::op::distributed::allreduce_(acc, acc, INFINICCL_SUM, comm_);
    }
    auto shared = shared_experts_->forward(x);
    return infinicore::op::add(acc, shared);
}

} // namespace infinilm::models::deepseek_v4
