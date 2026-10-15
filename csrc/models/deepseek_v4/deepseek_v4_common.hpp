#pragma once

#include "../../layers/linear/linear.hpp"
#include "../../layers/quantization/compressed_tensors.hpp"

#include <memory>

namespace infinilm::models::deepseek_v4 {

inline std::shared_ptr<infinilm::quantization::BaseQuantization> int8_quantization() {
    return std::make_shared<infinilm::quantization::CompressedTensors>(nlohmann::json::object());
}

// mtp.0 keeps a BF16 main projection. mtp.2 is the DSpark markov/confidence head.
enum class MtpKind {
    None,
    MainProj,
    Markov,
};

} // namespace infinilm::models::deepseek_v4
