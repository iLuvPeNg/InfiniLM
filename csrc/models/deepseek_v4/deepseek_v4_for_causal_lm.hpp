#pragma once

#include "deepseek_v4_decoderLayer.hpp"

#include "../infinilm_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"

#include <cstdint>
#include <vector>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4ForCausalLM : public infinilm::InfinilmModel {
public:
    DeepseekV4ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                          const infinicore::Device &device);

    Output forward(const Input &input) const override;
    void reset_cache(const cache::CacheConfig *cache_config) override;
    bool try_load_parameter(const std::string &name, const infinicore::Tensor &param) override;

private:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    INFINICORE_NN_MODULE(infinilm::nn::Linear, head);
    INFINICORE_NN_PARAMETER(hc_head_fn);
    INFINICORE_NN_PARAMETER(hc_head_base);
    INFINICORE_NN_PARAMETER(hc_head_scale);
    std::vector<int> dspark_target_layers_;
    int dspark_block_size_{0};
    int64_t dspark_noise_token_{0};
    std::vector<std::shared_ptr<DeepseekV4DecoderLayer>> layers_;
    std::vector<std::shared_ptr<DeepseekV4DecoderLayer>> mtp_;
};

std::shared_ptr<infinilm::config::ModelConfig>
create_deepseek_v4_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::deepseek_v4
