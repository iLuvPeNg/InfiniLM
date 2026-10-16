#include "deepseek_v4_allocate_kv_cache_tensors.hpp"

#include "../../global_state/global_state.hpp"
#include "deepseek_v4_common.hpp"
#include "infinicore/context/context.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

size_t cache_capacity(const cache::CacheConfig *cache_config) {
    if (const auto *paged = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config)) {
        return paged->num_blocks() * paged->block_size();
    }
    if (const auto *static_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config)) {
        const size_t length = static_config->max_cache_len();
        if (length == 0 || length > (size_t{1} << 20)) {
            throw std::runtime_error("deepseek_v4_allocate_kv_cache_tensors: `max_cache_len` is not a finite cache length");
        }
        return length;
    }
    throw std::runtime_error("deepseek_v4_allocate_kv_cache_tensors: unsupported cache config");
}

size_t state_width(int compress_ratio, size_t head_dim) {
    const size_t coff = compress_ratio == 4 ? 2 : 1;
    return coff * head_dim;
}

} // namespace

std::vector<infinicore::Tensor> deepseek_v4_allocate_kv_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
    const backends::AttentionBackend &attention_backend) {
    if (cache_config == nullptr) {
        return {};
    }
    if (text_config == nullptr) {
        throw std::runtime_error("deepseek_v4_allocate_kv_cache_tensors: `text_config` is null");
    }
    switch (attention_backend) {
    case backends::AttentionBackend::STATIC_ATTN:
    case backends::AttentionBackend::PAGED_ATTN:
    case backends::AttentionBackend::FLASH_ATTN:
    case backends::AttentionBackend::FLASHINFER:
        break;
    default:
        throw std::runtime_error(
            "deepseek_v4_allocate_kv_cache_tensors: unsupported attention backend "
            + std::to_string(static_cast<int>(attention_backend)));
    }

    const size_t num_layers = text_config->get<size_t>("num_hidden_layers");
    const size_t mtp_layers = static_cast<size_t>(std::max(text_config->get_or<int>("num_hash_layers", 0), 0));
    const size_t total_layers = num_layers + mtp_layers;
    const size_t head_dim = text_config->get<size_t>("head_dim");
    const size_t capacity = cache_capacity(cache_config);
    const auto ratios = text_config->get_config_json().at("compress_ratios").get<std::vector<int>>();
    const auto &device = global_state::get_tensor_model_parallel_rank_info().device;
    const auto dtype = text_config->get_dtype();

    std::vector<infinicore::Tensor> kv_cache_vec(total_layers * kKvCachesPerLayer);
    for (size_t layer = 0; layer < total_layers; ++layer) {
        const size_t base = layer * kKvCachesPerLayer;
        kv_cache_vec[base + kSwaCacheSlot] = infinicore::Tensor::zeros({capacity, head_dim}, dtype, device);
        const int ratio = layer < ratios.size() ? ratios[layer] : 0;
        if (ratio > 1) {
            const size_t width = state_width(ratio, head_dim);
            kv_cache_vec[base + kCompressedCacheSlot] = infinicore::Tensor::zeros({capacity, head_dim}, dtype, device);
            kv_cache_vec[base + kStateCacheSlot] = infinicore::Tensor::zeros(
                {capacity, width * 2}, infinicore::DataType::F32, device);
        }
    }
    infinicore::context::syncStream();
    return kv_cache_vec;
}

} // namespace infinilm::models::deepseek_v4
