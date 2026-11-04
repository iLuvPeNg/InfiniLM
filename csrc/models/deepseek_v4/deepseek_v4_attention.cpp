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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

// Timer for metric 1 in generation. The Iluvatar FlashMLA launch already
// synchronizes its stream before returning, so the wall time of the call is
// the kernel time and this adds no extra sync. Buckets: prefill (many
// tokens), decode (one token), spec (DSpark draft/verify blocks).
struct FlashMlaTimer {
    bool enabled = std::getenv("INFINI_FLASHMLA_TIMER") != nullptr;
    double ms[3] = {0.0, 0.0, 0.0};
    double flops[3] = {0.0, 0.0, 0.0};
    size_t count[3] = {0, 0, 0};
    void add(size_t tokens, double call_ms, double call_flops) {
        const size_t bucket = tokens >= 32 ? 0 : (tokens == 1 ? 1 : 2);
        ms[bucket] += call_ms;
        flops[bucket] += call_flops;
        count[bucket] += 1;
    }
    ~FlashMlaTimer() {
        if (!enabled) {
            return;
        }
        const char *names[3] = {"prefill", "decode", "spec"};
        for (size_t bucket = 0; bucket < 3; ++bucket) {
            if (count[bucket] == 0) {
                continue;
            }
            std::fprintf(stderr,
                         "METRIC_FLASHMLA_GEN %s calls %zu ms %.3f tflops %.3f\n",
                         names[bucket],
                         count[bucket],
                         ms[bucket],
                         ms[bucket] > 0.0 ? flops[bucket] / ms[bucket] / 1.0e9 : 0.0);
        }
    }
};

FlashMlaTimer &flash_mla_timer() {
    thread_local FlashMlaTimer timer;
    return timer;
}

float rope_freq(size_t pair, size_t rope_dim, double theta) {
    return static_cast<float>(1.0 / std::pow(theta, (2.0 * static_cast<double>(pair)) / static_cast<double>(rope_dim)));
}

// Compressor layers use the checkpoint YaRN table. Sliding-window layers keep
// plain RoPE. mscale is disabled, so cos and sin are not rescaled.
std::vector<float> yarn_inv_freq(size_t rope_dim,
                                 double theta,
                                 double factor,
                                 int beta_fast,
                                 int beta_slow,
                                 double original_max) {
    const size_t half = rope_dim / 2;
    auto correction = [&](double rotations) {
        return (static_cast<double>(rope_dim) * std::log(original_max / (rotations * 2.0 * 3.14159265358979323846)))
               / (2.0 * std::log(theta));
    };
    double low = std::floor(correction(static_cast<double>(beta_fast)));
    double high = std::ceil(correction(static_cast<double>(beta_slow)));
    low = std::max(low, 0.0);
    high = std::min(high, static_cast<double>(rope_dim) - 1.0);
    if (low == high) {
        high += 0.001;
    }
    std::vector<float> inv(half);
    for (size_t pair = 0; pair < half; ++pair) {
        const double extra = static_cast<double>(rope_freq(pair, rope_dim, theta));
        const double interpolated = extra / factor;
        double ramp = (static_cast<double>(pair) - low) / (high - low);
        ramp = std::min(1.0, std::max(0.0, ramp));
        inv[pair] = static_cast<float>(interpolated * ramp + extra * (1.0 - ramp));
    }
    return inv;
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
    if (compress_ratio > 1) {
        double factor = 16.0;
        int beta_fast = 32;
        int beta_slow = 1;
        double original_max = 65536.0;
        const auto &cfg = model_config->get_config_json();
        if (cfg.contains("rope_scaling") && cfg["rope_scaling"].is_object()) {
            const auto &scaling = cfg["rope_scaling"];
            factor = scaling.value("factor", factor);
            beta_fast = scaling.value("beta_fast", beta_fast);
            beta_slow = scaling.value("beta_slow", beta_slow);
            original_max = scaling.value("original_max_position_embeddings", original_max);
        }
        rope_inv_freq_ = yarn_inv_freq(rope_dim_, compress_rope_theta_, factor, beta_fast, beta_slow, original_max);
    } else {
        rope_inv_freq_.resize(rope_dim_ / 2);
        for (size_t pair = 0; pair < rope_inv_freq_.size(); ++pair) {
            rope_inv_freq_[pair] = rope_freq(pair, rope_dim_, rope_theta_);
        }
    }
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

void copy_rows(infinicore::Tensor dst, size_t dst_row, const infinicore::Tensor &src, size_t src_row, size_t nrows) {
    if (nrows == 0) {
        return;
    }
    const size_t dim = src->size(src->ndim() - 1);
    dst->narrow({{0, dst_row, nrows}})->view({nrows, dim})->copy_from(
        src->narrow({{0, src_row, nrows}})->contiguous()->view({nrows, dim}));
}

bool contiguous_positions(const std::vector<int64_t> &positions, int64_t *origin) {
    if (positions.empty()) {
        return false;
    }
    for (size_t i = 0; i < positions.size(); ++i) {
        if (positions[i] != positions[0] + static_cast<int64_t>(i)) {
            return false;
        }
    }
    *origin = positions[0];
    return true;
}

infinicore::Tensor rope_table(size_t max_pos,
                              size_t rope_dim,
                              const std::vector<float> &inv_freq,
                              const infinicore::Device &device) {
    const size_t half = rope_dim / 2;
    std::vector<float> table(max_pos * rope_dim);
    for (size_t pos = 0; pos < max_pos; ++pos) {
        for (size_t pair = 0; pair < half; ++pair) {
            const double ang = static_cast<double>(pos) * static_cast<double>(inv_freq[pair]);
            table[pos * rope_dim + pair] = static_cast<float>(std::cos(ang));
            table[pos * rope_dim + half + pair] = static_cast<float>(std::sin(ang));
        }
    }
    return upload_f32(table, {max_pos, rope_dim}, device);
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

    ensure_rope_cache(cache_len, device);
    q = apply_rope(q, positions, rope_cache_, head_dim_ - rope_dim_, rope_dim_, false);
    kv = apply_rope(kv, positions, rope_cache_, head_dim_ - rope_dim_, rope_dim_, false);

    int64_t origin = 0;
    if (contiguous_positions(positions, &origin)) {
        copy_rows(swa_cache, static_cast<size_t>(origin), kv, 0, tokens);
    } else {
        for (size_t t = 0; t < tokens; ++t) {
            copy_last_row(swa_cache, static_cast<size_t>(positions[t]), kv, t);
        }
    }

    if (compressor_) {
        const int ratio = compressor_->compress_ratio();
        const size_t width = compressor_->state_width();
        const size_t head = compressor_->head_dim();
        auto state_cache = cache_slot(layer_idx_, kStateCacheSlot);
        auto compressed_cache = cache_slot(layer_idx_, kCompressedCacheSlot);
        if (compress_ape_.empty()) {
            compress_norm_weight_ = download_f32(compressor_->norm_weight());
            compress_ape_ = download_f32(compressor_->ape());
        }
        auto projected = compressor_->project(x);
        auto kv_state = cast_dtype(projected.first, infinicore::DataType::F32)->contiguous();
        auto score_state = cast_dtype(projected.second, infinicore::DataType::F32)->contiguous();
        if (contiguous_positions(positions, &origin)) {
            std::vector<float> ape_host(tokens * width);
            for (size_t t = 0; t < tokens; ++t) {
                const size_t src = static_cast<size_t>(positions[t] % ratio) * width;
                std::copy(compress_ape_.begin() + static_cast<std::ptrdiff_t>(src),
                          compress_ape_.begin() + static_cast<std::ptrdiff_t>(src + width),
                          ape_host.begin() + static_cast<std::ptrdiff_t>(t * width));
            }
            auto ape_rows = upload_f32(ape_host, {tokens, width}, device);
            auto score = infinicore::op::add(score_state, ape_rows)->contiguous();
            auto block = infinicore::op::cat({kv_state, score}, 1)->contiguous();
            const size_t stride = width * 2;
            state_cache->narrow({{0, static_cast<size_t>(origin), tokens}})->view({tokens, stride})->copy_from(block->view({tokens, stride}));
        } else {
            for (size_t t = 0; t < tokens; ++t) {
                const auto pos = static_cast<size_t>(positions[t]);
                std::vector<float> ape_host(compress_ape_.begin() + static_cast<std::ptrdiff_t>((pos % static_cast<size_t>(ratio)) * width),
                                            compress_ape_.begin() + static_cast<std::ptrdiff_t>((pos % static_cast<size_t>(ratio)) * width + width));
                auto ape = upload_f32(ape_host, {width}, device);
                auto score = infinicore::op::add(score_state->narrow({{0, t, 1}})->contiguous()->view({width}), ape);
                state_cache->narrow({{0, pos, 1}, {1, 0, width}})->view({width})->copy_from(kv_state->narrow({{0, t, 1}})->contiguous()->view({width}));
                state_cache->narrow({{0, pos, 1}, {1, width, width}})->view({width})->copy_from(score);
            }
        }
        const int overlap = ratio == 4 ? 1 : 0;
        const int n_ctx = (1 + overlap) * ratio;
        const size_t nope = head - rope_dim_;
        const size_t half = rope_dim_ / 2;
        std::vector<int64_t> emit;
        for (int64_t pos : positions) {
            if ((pos + 1) % ratio == 0) {
                emit.push_back(pos);
            }
        }
        if (!emit.empty()) {
            int64_t slab_lo = emit.front();
            int64_t slab_hi = emit.front();
            for (int64_t pos : emit) {
                slab_lo = std::min(slab_lo, std::max<int64_t>(0, pos - n_ctx + 1));
                slab_hi = std::max(slab_hi, pos);
            }
            const int64_t slab_n = slab_hi - slab_lo + 1;
            auto slab = download_f32(state_cache->narrow({{0, static_cast<size_t>(slab_lo), static_cast<size_t>(slab_n)}}));
            const size_t stride = width * 2;
            std::vector<float> rows(emit.size() * head, 0.0f);
            for (size_t e = 0; e < emit.size(); ++e) {
                const int64_t pos = emit[e];
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
                    const float *base = slab.data() + static_cast<size_t>(pos_t - slab_lo) * stride;
                    const float *kv_part = base + head_off;
                    const float *score_part = base + width + head_off;
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
                    const float ang = static_cast<float>(compressed_pos) * rope_inv_freq_[pair];
                    const float cos_v = std::cos(ang);
                    const float sin_v = std::sin(ang);
                    const size_t even = nope + pair * 2;
                    const float x0 = acc[even];
                    const float x1 = acc[even + 1];
                    acc[even] = x0 * cos_v - x1 * sin_v;
                    acc[even + 1] = x0 * sin_v + x1 * cos_v;
                }
                std::copy(acc.begin(), acc.end(), rows.begin() + static_cast<std::ptrdiff_t>(e * head));
            }
            auto gpu_rows = cast_dtype(upload_f32(rows, {emit.size(), head}, device), dtype);
            const int64_t first_slot = emit.front() / ratio;
            bool dense_slots = true;
            for (size_t e = 0; e < emit.size(); ++e) {
                if (emit[e] / ratio != first_slot + static_cast<int64_t>(e)) {
                    dense_slots = false;
                    break;
                }
            }
            if (dense_slots) {
                copy_rows(compressed_cache, static_cast<size_t>(first_slot), gpu_rows, 0, emit.size());
            } else {
                for (size_t e = 0; e < emit.size(); ++e) {
                    copy_last_row(compressed_cache, static_cast<size_t>(emit[e] / ratio), gpu_rows, e);
                }
            }
        }
    }

    const int ratio = compressor_ ? compressor_->compress_ratio() : 1;
    size_t packed_rows = 0;
    std::vector<size_t> swa_at(tokens, 0);
    std::vector<size_t> swa_n(tokens, 0);
    std::vector<size_t> cmp_at(tokens, 0);
    std::vector<size_t> cmp_n(tokens, 0);
    for (size_t t = 0; t < tokens; ++t) {
        const int64_t pos = positions[t];
        const int64_t begin = std::max<int64_t>(0, pos - sliding_window_ + 1);
        swa_at[t] = static_cast<size_t>(begin);
        swa_n[t] = static_cast<size_t>(pos - begin + 1);
        packed_rows += swa_n[t];
        if (compressor_) {
            int64_t count = (pos + 1) / ratio;
            int64_t keep = count;
            if (index_topk_ > 0 && keep > index_topk_) {
                keep = index_topk_;
            }
            cmp_at[t] = static_cast<size_t>(count - keep);
            cmp_n[t] = static_cast<size_t>(keep);
            packed_rows += cmp_n[t];
        }
    }
    if (packed_rows == 0) {
        throw std::runtime_error("deepseek_v4 attention: no keys for FlashMLA");
    }
    auto packed = infinicore::Tensor::empty({packed_rows, head_dim_}, dtype, device);
    std::vector<size_t> starts(tokens, 0);
    std::vector<size_t> lens(tokens, 0);
    size_t cursor = 0;
    infinicore::Tensor compressed_cache;
    if (compressor_) {
        compressed_cache = cache_slot(layer_idx_, kCompressedCacheSlot);
    }
    for (size_t t = 0; t < tokens; ++t) {
        starts[t] = cursor;
        copy_rows(packed, cursor, swa_cache, swa_at[t], swa_n[t]);
        cursor += swa_n[t];
        if (cmp_n[t] > 0) {
            copy_rows(packed, cursor, compressed_cache, cmp_at[t], cmp_n[t]);
            cursor += cmp_n[t];
        }
        lens[t] = cursor - starts[t];
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
        auto &timer = flash_mla_timer();
        if (timer.enabled) {
            double rows = 0.0;
            for (const auto len : row_lens) {
                rows += static_cast<double>(len);
            }
            const auto begin = std::chrono::steady_clock::now();
            infinicore::op::flash_mla_(attn, query, packed, sink, scale_, row_starts, row_lens);
            const auto end = std::chrono::steady_clock::now();
            timer.add(tokens,
                      std::chrono::duration<double, std::milli>(end - begin).count(),
                      2.0 * static_cast<double>(local_heads_) * static_cast<double>(head_dim_) * rows);
        } else {
            infinicore::op::flash_mla_(attn, query, packed, sink, scale_, row_starts, row_lens);
        }
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
    ensure_rope_cache(cache_len, device);
    kv = apply_rope(kv, positions, rope_cache_, head_dim_ - rope_dim_, rope_dim_, false);
    int64_t origin = 0;
    if (contiguous_positions(positions, &origin)) {
        copy_rows(swa_cache, static_cast<size_t>(origin), kv, 0, tokens);
    } else {
        for (size_t t = 0; t < tokens; ++t) {
            copy_last_row(swa_cache, static_cast<size_t>(positions[t]), kv, t);
        }
    }
}

void DeepseekV4Attention::ensure_rope_cache(size_t cache_len, const infinicore::Device &device) const {
    if (rope_cache_ && rope_cache_->size(0) == cache_len) {
        return;
    }
    rope_cache_ = rope_table(cache_len, rope_dim_, rope_inv_freq_, device);
}

} // namespace infinilm::models::deepseek_v4
