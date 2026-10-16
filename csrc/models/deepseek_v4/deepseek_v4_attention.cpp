#include "deepseek_v4_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "deepseek_v4_runtime.hpp"

#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/flash_mla.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/mul_scalar.hpp"
#include "infinicore/ops/rms_norm.hpp"
#include "infinicore/ops/softmax.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v4 {
namespace {

infinicore::Tensor attend_heads(const infinicore::Tensor &query_heads,
                                const infinicore::Tensor &keys,
                                float scale,
                                const infinicore::Tensor &sink) {
    const size_t heads = query_heads->size(0);
    const size_t rows = keys->size(0);
    auto scores = infinicore::op::matmul(query_heads, keys->permute({1, 0})->contiguous());
    auto scores_f = infinicore::op::mul_scalar(cast_dtype(scores, infinicore::DataType::F32), static_cast<double>(scale));
    auto sink_col = cast_dtype(sink, infinicore::DataType::F32)->view({heads, size_t{1}});
    auto logits = infinicore::op::cat({scores_f, sink_col}, 1);
    auto probs = infinicore::Tensor::empty(logits->shape(), infinicore::DataType::F32, logits->device());
    infinicore::op::softmax_(probs, logits, 1);
    auto kept = cast_dtype(probs->narrow({{1, 0, rows}})->contiguous(), query_heads->dtype());
    return infinicore::op::matmul(kept, keys);
}

} // namespace

DeepseekV4Compressor::DeepseekV4Compressor(size_t hidden_size,
                                           size_t head_dim,
                                           int compress_ratio,
                                           double eps,
                                           const infinicore::DataType &dtype,
                                           const infinicore::Device &device) {
    if (compress_ratio != 4 && compress_ratio != 128) {
        throw std::runtime_error("deepseek_v4 compressor: compress_ratio must be 4 or 128");
    }
    compress_ratio_ = compress_ratio;
    head_dim_ = head_dim;
    const size_t coff = compress_ratio == 4 ? 2 : 1;
    state_width_ = coff * head_dim;
    const size_t state = state_width_;
    INFINICORE_NN_PARAMETER_INIT(ape, ({static_cast<size_t>(compress_ratio), state}, infinicore::DataType::F32, device));
    INFINICORE_NN_MODULE_INIT(norm, head_dim, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(wkv, hidden_size, state, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(wgate, hidden_size, state, false, dtype, device);
}

DeepseekV4Indexer::DeepseekV4Indexer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     const infinicore::Device &device,
                                     int compress_ratio) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden = model_config->get<size_t>("hidden_size");
    const size_t q_lora = model_config->get<size_t>("q_lora_rank");
    const size_t n_heads = model_config->get<size_t>("index_n_heads");
    const size_t head_dim = model_config->get<size_t>("index_head_dim");
    const double eps = model_config->get<double>("rms_norm_eps");
    auto quant = int8_quantization();
    INFINICORE_NN_MODULE_INIT(wq_b, q_lora, n_heads * head_dim, quant, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(weights_proj, hidden, n_heads, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(compressor, hidden, head_dim, compress_ratio, eps, dtype, device);
}

DeepseekV4Attention::DeepseekV4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         const infinicore::Device &device,
                                         int compress_ratio,
                                         size_t layer_idx) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden = model_config->get<size_t>("hidden_size");
    const size_t q_lora = model_config->get<size_t>("q_lora_rank");
    const size_t n_heads = model_config->get<size_t>("num_attention_heads");
    const size_t head_dim = model_config->get<size_t>("head_dim");
    const size_t o_groups = model_config->get<size_t>("o_groups");
    const size_t o_lora = model_config->get<size_t>("o_lora_rank");
    const double eps = model_config->get<double>("rms_norm_eps");
    layer_idx_ = layer_idx;
    local_heads_ = n_heads / static_cast<size_t>(infinilm::global_state::get_tensor_model_parallel_rank_info().tp_size);
    head_dim_ = head_dim;
    rope_dim_ = model_config->get<size_t>("qk_rope_head_dim");
    local_groups_ = o_groups / static_cast<size_t>(infinilm::global_state::get_tensor_model_parallel_rank_info().tp_size);
    o_lora_ = o_lora;
    compress_ratio_ = compress_ratio;
    sliding_window_ = model_config->get<int>("sliding_window");
    index_topk_ = model_config->get<int>("index_topk");
    eps_ = eps;
    rope_theta_ = model_config->get<double>("rope_theta");
    compress_rope_theta_ = model_config->get<double>("compress_rope_theta");
    scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (n_heads % static_cast<size_t>(rank_info.tp_size) != 0 || o_groups % static_cast<size_t>(rank_info.tp_size) != 0) {
        throw std::runtime_error("deepseek_v4 attention: num_attention_heads and o_groups must be divisible by tp size");
    }
    auto quant = int8_quantization();
    const size_t o_width = o_groups * o_lora;

    INFINICORE_NN_PARAMETER_INIT(attn_sink, ({n_heads}, infinicore::DataType::F32, device, 0, rank_info.tp_rank, rank_info.tp_size));
    INFINICORE_NN_MODULE_INIT(q_norm, q_lora, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(kv_norm, head_dim, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(wq_a, hidden, q_lora, quant, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(wq_b, q_lora, n_heads * head_dim, quant, false, dtype, device, rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(wkv, hidden, head_dim, quant, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(wo_a, hidden, o_width, quant, false, dtype, device, rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(wo_b, o_width, hidden, quant, false, dtype, device, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);

    if (compress_ratio > 1) {
        compressor_ = this->register_module<DeepseekV4Compressor>(
            "compressor", hidden, head_dim, compress_ratio, eps, dtype, device);
    }
    if (compress_ratio == 4) {
        indexer_ = this->register_module<DeepseekV4Indexer>(
            "indexer", model_config, device, compress_ratio);
    }
}

namespace {

void copy_last_row(infinicore::Tensor dst, size_t dst_row, const infinicore::Tensor &src, size_t src_row) {
    const size_t n = src->size(src->ndim() - 1);
    dst->narrow({{0, dst_row, 1}})->view({n})->copy_from(src->narrow({{0, src_row, 1}})->contiguous()->view({n}));
}

float rope_freq(size_t pair, size_t rope_dim, double theta) {
    return static_cast<float>(1.0 / std::pow(theta, (2.0 * static_cast<double>(pair)) / static_cast<double>(rope_dim)));
}

infinicore::Tensor cache_slot(size_t layer_idx, size_t slot) {
    auto &kv_cache_vec = infinilm::global_state::get_forward_context().kv_cache_vec;
    const size_t index = layer_idx * kKvCachesPerLayer + slot;
    if (index >= kv_cache_vec.size() || !kv_cache_vec[index]) {
        throw std::runtime_error("deepseek_v4 attention: KV cache slot is missing; call `reset_cache` first");
    }
    return kv_cache_vec[index];
}

} // namespace

std::pair<infinicore::Tensor, infinicore::Tensor>
DeepseekV4Compressor::project(infinicore::Tensor hidden) const {
    auto kv = wkv_->forward(hidden);
    auto score = wgate_->forward(hidden);
    return {kv, score};
}

infinicore::Tensor DeepseekV4Attention::forward(infinicore::Tensor hidden,
                                                const std::vector<int64_t> &positions) const {
    auto x = hidden->contiguous();
    const size_t tokens = x->size(0);
    if (tokens != positions.size()) {
        throw std::runtime_error("deepseek_v4 attention: position count does not match tokens");
    }
    auto swa_cache = cache_slot(layer_idx_, kSwaCacheSlot);
    const size_t cache_len = swa_cache->size(0);
    for (int64_t pos : positions) {
        if (pos < 0 || static_cast<size_t>(pos) >= cache_len) {
            throw std::runtime_error("deepseek_v4 attention: `position` is outside the KV cache");
        }
    }
    const auto device = x->device();
    const auto dtype = x->dtype();

    auto qa = q_norm_->forward(int8_gemm(x, wq_a_->weight(), wq_a_->weight_scale()));
    auto q = int8_gemm(qa, wq_b_->weight(), wq_b_->weight_scale())->contiguous()->view({tokens, local_heads_, head_dim_});
    auto ones = infinicore::Tensor::ones({head_dim_}, q->dtype(), device);
    q = infinicore::op::rms_norm(q, ones, static_cast<float>(eps_));
    auto kv = kv_norm_->forward(int8_gemm(x, wkv_->weight(), wkv_->weight_scale()))->contiguous();

    if (!rope_cache_ || rope_cache_->size(0) != cache_len) {
        rope_cache_ = make_rope_cache(cache_len, rope_dim_, rope_theta_, device);
    }
    q = apply_rope(q, positions, rope_cache_, head_dim_ - rope_dim_, rope_dim_, false);
    kv = apply_rope(kv, positions, rope_cache_, head_dim_ - rope_dim_, rope_dim_, false);

    if (compressed_valid_.size() != cache_len) {
        compressed_valid_.assign(cache_len, 0);
    }
    const bool new_sequence = std::find(positions.begin(), positions.end(), 0) != positions.end();
    if (new_sequence) {
        std::fill(compressed_valid_.begin(), compressed_valid_.end(), 0);
    }
    for (size_t t = 0; t < tokens; ++t) {
        copy_last_row(swa_cache, static_cast<size_t>(positions[t]), kv, t);
    }

    if (compressor_) {
        const int ratio = compressor_->compress_ratio();
        const size_t width = compressor_->state_width();
        const size_t head = compressor_->head_dim();
        auto state_cache = cache_slot(layer_idx_, kStateCacheSlot);
        auto compressed_cache = cache_slot(layer_idx_, kCompressedCacheSlot);
        if (!compress_rope_cache_ || compress_rope_cache_->size(0) != cache_len) {
            compress_rope_cache_ = make_rope_cache(cache_len, rope_dim_, compress_rope_theta_, device);
            compress_norm_weight_ = download_f32(compressor_->norm_weight());
        }
        auto projected = compressor_->project(x);
        auto kv_state = cast_dtype(projected.first, infinicore::DataType::F32);
        auto score_state = cast_dtype(projected.second, infinicore::DataType::F32);
        for (size_t t = 0; t < tokens; ++t) {
            const auto pos = static_cast<size_t>(positions[t]);
            auto ape = compressor_->ape()->narrow({{0, pos % static_cast<size_t>(ratio), 1}})->contiguous()->view({width});
            auto score = infinicore::op::add(score_state->narrow({{0, t, 1}})->contiguous()->view({width}), ape);
            state_cache->narrow({{0, pos, 1}, {1, 0, width}})->view({width})->copy_from(kv_state->narrow({{0, t, 1}})->contiguous()->view({width}));
            state_cache->narrow({{0, pos, 1}, {1, width, width}})->view({width})->copy_from(score);
        }
        const int overlap = ratio == 4 ? 1 : 0;
        const int n_ctx = (1 + overlap) * ratio;
        const size_t nope = head - rope_dim_;
        const size_t half = rope_dim_ / 2;
        for (size_t t = 0; t < tokens; ++t) {
            const int64_t pos = positions[t];
            if ((pos + 1) % ratio != 0) {
                continue;
            }
            const int64_t start = pos - n_ctx + 1;
            std::vector<float> running_max(head, -std::numeric_limits<float>::infinity());
            std::vector<float> running_sum(head, 0.0f);
            std::vector<float> acc(head, 0.0f);
            for (int step = 0; step < n_ctx; ++step) {
                const int64_t pos_t = start + step;
                if (pos_t < 0) {
                    continue;
                }
                const size_t head_off = step >= ratio ? head : 0;
                auto kv_part = download_f32(state_cache->narrow({{0, static_cast<size_t>(pos_t), 1}, {1, head_off, head}}));
                auto score_part = download_f32(state_cache->narrow({{0, static_cast<size_t>(pos_t), 1}, {1, width + head_off, head}}));
                for (size_t d = 0; d < head; ++d) {
                    const float new_max = std::max(running_max[d], score_part[d]);
                    const float old_scale = std::exp(running_max[d] - new_max);
                    const float score_scale = std::exp(score_part[d] - new_max);
                    acc[d] = acc[d] * old_scale + kv_part[d] * score_scale;
                    running_sum[d] = running_sum[d] * old_scale + score_scale;
                    running_max[d] = new_max;
                }
            }
            float variance = 0.0f;
            for (size_t d = 0; d < head; ++d) {
                acc[d] = running_sum[d] == 0.0f ? 0.0f : acc[d] / running_sum[d];
                variance += acc[d] * acc[d];
            }
            const float rrms = 1.0f / std::sqrt(variance / static_cast<float>(head) + static_cast<float>(eps_));
            for (size_t d = 0; d < head; ++d) {
                acc[d] *= rrms * compress_norm_weight_[d];
            }
            const int64_t compressed_pos = (pos / ratio) * ratio;
            for (size_t pair = 0; pair < half; ++pair) {
                const float ang = static_cast<float>(compressed_pos) * rope_freq(pair, rope_dim_, compress_rope_theta_);
                const float cos_v = std::cos(ang);
                const float sin_v = std::sin(ang);
                const size_t even = nope + pair * 2;
                const float x0 = acc[even];
                const float x1 = acc[even + 1];
                acc[even] = x0 * cos_v - x1 * sin_v;
                acc[even + 1] = x0 * sin_v + x1 * cos_v;
            }
            auto row = cast_dtype(upload_f32(acc, {head}, device), dtype);
            compressed_cache->narrow({{0, static_cast<size_t>(pos), 1}})->view({head})->copy_from(row);
            compressed_valid_[static_cast<size_t>(pos)] = 1;
        }
    }

    std::vector<std::vector<std::pair<int, size_t>>> keys(tokens);
    size_t packed_rows = 0;
    size_t max_k = 0;
    for (size_t t = 0; t < tokens; ++t) {
        const int64_t pos = positions[t];
        const int64_t begin = std::max<int64_t>(0, pos - sliding_window_ + 1);
        for (int64_t p = begin; p <= pos; ++p) {
            keys[t].push_back({0, static_cast<size_t>(p)});
        }
        if (compressor_) {
            std::vector<size_t> compressed;
            for (int64_t p = 0; p <= pos; ++p) {
                if (compressed_valid_[static_cast<size_t>(p)] != 0) {
                    compressed.push_back(static_cast<size_t>(p));
                }
            }
            if (index_topk_ > 0 && static_cast<int>(compressed.size()) > index_topk_) {
                compressed.erase(compressed.begin(), compressed.end() - index_topk_);
            }
            for (size_t p : compressed) {
                keys[t].push_back({1, p});
            }
        }
        packed_rows += keys[t].size();
        max_k = std::max(max_k, keys[t].size());
    }
    if (packed_rows == 0 || max_k == 0) {
        throw std::runtime_error("deepseek_v4 attention: no keys for FlashMLA");
    }
    auto packed = infinicore::Tensor::empty({packed_rows, head_dim_}, dtype, device);
    std::vector<size_t> starts(tokens, 0);
    std::vector<size_t> lens(tokens, 0);
    size_t cursor = 0;
    for (size_t t = 0; t < tokens; ++t) {
        starts[t] = cursor;
        lens[t] = keys[t].size();
        for (size_t k = 0; k < keys[t].size(); ++k) {
            const auto &key = keys[t][k];
            const auto cache = key.first == 0 ? swa_cache : cache_slot(layer_idx_, kCompressedCacheSlot);
            copy_last_row(packed, cursor, cache, key.second);
            ++cursor;
        }
    }
    auto query = q->contiguous();
    auto sink = attn_sink_->contiguous();
    auto attn = infinicore::Tensor::empty({tokens, local_heads_, head_dim_}, dtype, device);
    if (device.getType() == infinicore::Device::Type::ILUVATAR) {
        std::vector<int32_t> row_starts(tokens);
        std::vector<int32_t> row_lens(tokens);
        for (size_t t = 0; t < tokens; ++t) {
            row_starts[t] = static_cast<int32_t>(starts[t]);
            row_lens[t] = static_cast<int32_t>(lens[t]);
        }
        infinicore::op::flash_mla_(attn, query, packed, sink, scale_, row_starts, row_lens);
    } else {
        for (size_t t = 0; t < tokens; ++t) {
            auto q_t = query->narrow({{0, t, 1}})->contiguous()->view({local_heads_, head_dim_});
            auto k_t = packed->narrow({{0, starts[t], lens[t]}})->contiguous()->view({lens[t], head_dim_});
            auto out_t = attend_heads(q_t, k_t, scale_, sink);
            attn->narrow({{0, t, 1}})->view({local_heads_, head_dim_})->copy_from(out_t);
        }
    }

    auto mixed = apply_rope(attn, positions, rope_cache_, head_dim_ - rope_dim_, rope_dim_, true);
    const size_t heads_per_group = local_heads_ / local_groups_;
    const size_t group_in = heads_per_group * head_dim_;
    std::vector<infinicore::Tensor> parts;
    parts.reserve(local_groups_);
    for (size_t group = 0; group < local_groups_; ++group) {
        auto slice = mixed->narrow({{1, group * heads_per_group, heads_per_group}})->contiguous()->view({tokens, group_in});
        auto weight = wo_a_->weight()->narrow({{0, group * o_lora_, o_lora_}})->contiguous();
        auto weight_scale = wo_a_->weight_scale()->narrow({{0, group * o_lora_, o_lora_}})->contiguous();
        parts.push_back(int8_gemm(slice, weight, weight_scale));
    }
    auto latent = parts.size() == 1 ? parts[0] : infinicore::op::cat(parts, 1);
    auto out = int8_gemm(latent, wo_b_->weight(), wo_b_->weight_scale());
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (rank_info.tp_size > 1 && rank_info.comm != nullptr) {
        infinicore::op::distributed::allreduce_(out, out, INFINICCL_SUM, rank_info.comm);
    }
    return out;
}

void DeepseekV4Attention::cache_context_kv(infinicore::Tensor hidden,
                                           const std::vector<int64_t> &positions) const {
    auto x = hidden->contiguous();
    const size_t tokens = x->size(0);
    if (tokens != positions.size()) {
        throw std::runtime_error("deepseek_v4 attention: position count does not match tokens");
    }
    auto swa_cache = cache_slot(layer_idx_, kSwaCacheSlot);
    const size_t cache_len = swa_cache->size(0);
    for (int64_t pos : positions) {
        if (pos < 0 || static_cast<size_t>(pos) >= cache_len) {
            throw std::runtime_error("deepseek_v4 attention: `position` is outside the KV cache");
        }
    }
    const auto device = x->device();
    auto kv = kv_norm_->forward(int8_gemm(x, wkv_->weight(), wkv_->weight_scale()))->contiguous();
    if (!rope_cache_ || rope_cache_->size(0) != cache_len) {
        rope_cache_ = make_rope_cache(cache_len, rope_dim_, rope_theta_, device);
    }
    kv = apply_rope(kv, positions, rope_cache_, head_dim_ - rope_dim_, rope_dim_, false);
    for (size_t t = 0; t < tokens; ++t) {
        copy_last_row(swa_cache, static_cast<size_t>(positions[t]), kv, t);
    }
}

} // namespace infinilm::models::deepseek_v4
