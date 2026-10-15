#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../cache/kv_cache.hpp"
#include "../../config/model_config.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <vector>

namespace infinilm::models::deepseek_v4 {

/** Implemented in `deepseek_v4_allocate_kv_cache_tensors.cpp`. */
std::vector<infinicore::Tensor> deepseek_v4_allocate_kv_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
    const backends::AttentionBackend &attention_backend);

} // namespace infinilm::models::deepseek_v4
