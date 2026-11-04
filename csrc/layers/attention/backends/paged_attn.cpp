#include "paged_attn.hpp"

#include "../../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include <cstdio>
#include <cstdlib>
#include <infinirt.h>
#include <vector>

namespace infinilm::layers::attention::backends {
namespace {

struct PagedAttnGenTimer {
    bool enabled = std::getenv("INFINI_MODEL_TIMER") != nullptr;
    bool dev_ok = false;
    infiniDevice_t dev_kind{};
    int dev_id = 0;
    void capture_device() {
        if (!dev_ok) {
            dev_ok = infinirtGetDevice(&dev_kind, &dev_id) == INFINI_STATUS_SUCCESS;
        }
    }
    struct Entry {
        int bucket = 0; // 0: prefill, 1: decode
        size_t q_len = 0;
        size_t heads = 0;
        size_t dim = 0;
        infinirtEvent_t a = nullptr, b = nullptr;
    };
    std::vector<Entry> entries;
    int seen[2] = {0, 0};
    int sampled[2] = {0, 0};
    bool want_sample(int bucket) {
        const int n = seen[bucket]++;
        if (sampled[bucket] >= 4) {
            return false;
        }
        if (n == 0 || n == 40 || n == 160 || n == 320) {
            sampled[bucket] += 1;
            return true;
        }
        return false;
    }
    ~PagedAttnGenTimer() {
        if (!enabled || entries.empty()) {
            return;
        }
        if (dev_ok) {
            infinirtSetDevice(dev_kind, dev_id);
        }
        const char *names[2] = {"prefill", "decode"};
        double ms[2] = {0, 0};
        double flops[2] = {0, 0};
        size_t calls[2] = {0, 0};
        size_t last_q[2] = {0, 0};
        for (const auto &e : entries) {
            float m = 0.0f;
            infinirtEventSynchronize(e.b);
            infinirtEventElapsedTime(&m, e.a, e.b);
            infinirtEventDestroy(e.a);
            infinirtEventDestroy(e.b);
            const int b = e.bucket;
            ms[b] += m;
            calls[b] += 1;
            last_q[b] = e.q_len;
            const double kv_rows = b == 0 ? static_cast<double>(e.q_len) * static_cast<double>(e.q_len + 1) * 0.5 : 32.0;
            flops[b] += 4.0 * static_cast<double>(e.heads) * static_cast<double>(e.dim) * kv_rows;
        }
        for (int b = 0; b < 2; ++b) {
            if (calls[b] == 0) {
                continue;
            }
            std::fprintf(stderr,
                         "METRIC_BF16_ATTN_GEN %s q %zu samples %zu total_ms %.3f avg_ms %.4f tflops %.4f\n",
                         names[b],
                         last_q[b],
                         calls[b],
                         ms[b],
                         ms[b] / static_cast<double>(calls[b]),
                         ms[b] > 0.0 ? flops[b] / ms[b] / 1.0e9 : 0.0);
        }
    }
};

PagedAttnGenTimer &paged_attn_gen_timer() {
    thread_local PagedAttnGenTimer timer;
    return timer;
}

} // namespace

PagedAttentionImpl::PagedAttentionImpl(size_t num_heads,
                                       size_t head_size,
                                       float scale,
                                       size_t num_kv_heads,
                                       size_t layer_idx)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      layer_idx_(layer_idx),
      head_dim_(head_size) {}

infinicore::Tensor PagedAttentionImpl::forward(const AttentionLayer &layer,
                                               const infinicore::Tensor &query,
                                               const infinicore::Tensor &key,
                                               const infinicore::Tensor &value,
                                               infinicore::Tensor &kv_cache,
                                               const infinilm::global_state::AttentionMetadata &attn_metadata) const {
    auto total_sequence_lengths = attn_metadata.total_sequence_lengths;
    auto input_offsets = attn_metadata.input_offsets;
    auto block_tables = attn_metadata.block_tables;
    auto slot_mapping = attn_metadata.slot_mapping;
    ASSERT(block_tables.has_value());
    ASSERT(slot_mapping.has_value());

    // 1. update paged kv cache
    auto [k_total, v_total] = do_kv_cache_update(layer, key, value, kv_cache, slot_mapping.value());

    size_t seq_len = query->shape()[0];
    bool is_prefill = (seq_len != total_sequence_lengths.value()->shape()[0]);

    // 2. Compute attention
    const size_t value_head_dim = value->size(value->ndim() - 1);
    infinicore::Tensor attn_output = infinicore::Tensor::empty({seq_len, num_heads_, value_head_dim}, query->dtype(), query->device());
    auto &pt = paged_attn_gen_timer();
    PagedAttnGenTimer::Entry pentry{};
    infinirtStream_t pstream = pt.enabled ? infinicore::context::getStream() : nullptr;
    if (pt.enabled) {
        pt.capture_device();
        pentry.bucket = is_prefill ? 0 : 1;
        pentry.q_len = seq_len;
        pentry.heads = num_heads_;
        pentry.dim = value_head_dim;
        if (pt.want_sample(pentry.bucket)) {
            infinirtEventCreate(&pentry.a);
            infinirtEventCreate(&pentry.b);
            infinirtEventRecord(pentry.a, pstream);
        } else {
            pstream = nullptr;
        }
    }
    if (is_prefill) {
        infinicore::op::paged_attention_prefill_(
            attn_output,
            query,
            k_total,
            v_total,
            block_tables.value(),
            total_sequence_lengths.value(),
            input_offsets.value(),
            std::nullopt,
            scale_);
    } else {
        infinicore::op::paged_attention_(
            attn_output,
            query,
            k_total,
            v_total,
            block_tables.value(),
            total_sequence_lengths.value(),
            std::nullopt,
            scale_);
    }
    if (pt.enabled && pstream != nullptr) {
        infinirtEventRecord(pentry.b, pstream);
        pt.entries.push_back(pentry);
    }
    attn_output = attn_output->view({1, seq_len, num_heads_ * value_head_dim});
    return attn_output;
}

std::tuple<infinicore::Tensor, infinicore::Tensor> PagedAttentionImpl::do_kv_cache_update(const AttentionLayer &layer,
                                                                                          const infinicore::Tensor key,
                                                                                          const infinicore::Tensor value,
                                                                                          infinicore::Tensor &kv_cache,
                                                                                          const infinicore::Tensor slot_mapping) const {
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    v_cache_layer = v_cache_layer->narrow({{3, 0, value->size(value->ndim() - 1)}});
    infinicore::op::paged_caching_(
        k_cache_layer,
        v_cache_layer,
        key,
        value,
        slot_mapping);

    return {k_cache_layer, v_cache_layer};
}
} // namespace infinilm::layers::attention::backends
