#pragma once

#include "../../layers/linear/linear.hpp"
#include "../../layers/quantization/compressed_tensors.hpp"

#include <memory>

namespace infinilm::models::deepseek_v4 {

inline std::shared_ptr<infinilm::quantization::BaseQuantization> int8_quantization() {
    return std::make_shared<infinilm::quantization::CompressedTensors>(nlohmann::json::object());
}

// `mtp.0` keeps a BF16 main projection. `mtp.2` is the DSpark Markov and confidence head.
enum class MtpKind {
    None,
    MainProj,
    Markov,
};

// Each decoder layer owns three dense cache tensors in `kv_cache_vec`.
inline constexpr size_t kKvCachesPerLayer = 3;
inline constexpr size_t kSwaCacheSlot = 0;
inline constexpr size_t kCompressedCacheSlot = 1;
inline constexpr size_t kStateCacheSlot = 2;

} // namespace infinilm::models::deepseek_v4
