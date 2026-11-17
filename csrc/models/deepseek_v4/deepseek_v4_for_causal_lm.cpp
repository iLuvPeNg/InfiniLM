#include "deepseek_v4_for_causal_lm.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include "deepseek_v4_allocate_kv_cache_tensors.hpp"
#include "deepseek_v4_runtime.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/mul_scalar.hpp"
#include "infinicore/ops/random_sample.hpp"
#include "infinicore/ops/sum.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v4 {
namespace {

bool parse_routed_expert(const std::string &name,
                         bool &is_mtp,
                         size_t &layer,
                         size_t &expert,
                         std::string &proj,
                         bool &is_scale) {
    const char *cursor = name.c_str();
    if (name.rfind("layers.", 0) == 0) {
        is_mtp = false;
        cursor += 7;
    } else if (name.rfind("mtp.", 0) == 0) {
        is_mtp = true;
        cursor += 4;
    } else {
        return false;
    }

    char *end = nullptr;
    const unsigned long layer_id = std::strtoul(cursor, &end, 10);
    if (end == cursor || std::strncmp(end, ".ffn.experts.", 13) != 0) {
        return false;
    }
    cursor = end + 13;
    const unsigned long expert_id = std::strtoul(cursor, &end, 10);
    if (end == cursor || *end != '.') {
        return false;
    }
    cursor = end + 1;
    if (!(cursor[0] == 'w' && (cursor[1] == '1' || cursor[1] == '2' || cursor[1] == '3') && cursor[2] == '.')) {
        return false;
    }
    proj.assign(cursor, 2);
    cursor += 3;
    if (std::strcmp(cursor, "weight") == 0) {
        is_scale = false;
    } else if (std::strcmp(cursor, "weight_scale") == 0) {
        is_scale = true;
    } else {
        return false;
    }
    layer = static_cast<size_t>(layer_id);
    expert = static_cast<size_t>(expert_id);
    return true;
}

} // namespace

DeepseekV4ForCausalLM::DeepseekV4ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                             const infinicore::Device &device) {
    model_config_ = model_config;
    const auto &dtype = model_config->get_dtype();
    const size_t hidden = model_config->get<size_t>("hidden_size");
    const size_t vocab = model_config->get<size_t>("vocab_size");
    const size_t n_layers = model_config->get<size_t>("num_hidden_layers");
    const int hc_mult = model_config->get<int>("hc_mult");
    const double eps = model_config->get<double>("rms_norm_eps");
    const size_t hc_dim = static_cast<size_t>(hc_mult) * hidden;
    const auto ratios = model_config->get_config_json().at("compress_ratios").get<std::vector<int>>();

    INFINICORE_NN_MODULE_INIT(embed, vocab, hidden, std::nullopt, dtype, device);
    layers_.reserve(n_layers);
    for (size_t i = 0; i < n_layers; ++i) {
        const int ratio = i < ratios.size() ? ratios[i] : 0;
        const bool has_tid = i < 3;
        layers_.push_back(this->register_module<DeepseekV4DecoderLayer>(
            "layers." + std::to_string(i),
            model_config,
            device,
            ratio,
            has_tid,
            !has_tid,
            i,
            MtpKind::None));
    }
    INFINICORE_NN_MODULE_INIT(norm, hidden, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(head, hidden, vocab, false, dtype, device);
    INFINICORE_NN_PARAMETER_INIT(hc_head_fn, ({static_cast<size_t>(hc_mult), hc_dim}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_head_base, ({static_cast<size_t>(hc_mult)}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_head_scale, ({size_t{1}}, infinicore::DataType::F32, device));

    const int n_mtp = model_config->get_or<int>("num_hash_layers", 3);
    mtp_.reserve(static_cast<size_t>(n_mtp));
    for (int i = 0; i < n_mtp; ++i) {
        MtpKind kind = MtpKind::None;
        if (i == 0) {
            kind = MtpKind::MainProj;
        } else if (i == n_mtp - 1) {
            kind = MtpKind::Markov;
        }
        mtp_.push_back(this->register_module<DeepseekV4DecoderLayer>(
            "mtp." + std::to_string(i),
            model_config,
            device,
            0,
            false,
            true,
            n_layers + static_cast<size_t>(i),
            kind));
    }
    dspark_block_size_ = model_config->get_or<int>("dspark_block_size", 0);
    dspark_noise_token_ = model_config->get_or<int>("dspark_noise_token_id", 0);
    if (model_config->get_config_json().contains("dspark_target_layer_ids")) {
        dspark_target_layers_ = model_config->get_config_json().at("dspark_target_layer_ids").get<std::vector<int>>();
    }
}

infinilm::InfinilmModel::Output DeepseekV4ForCausalLM::forward(const Input &input) const {
    if (!input.input_ids.has_value() || !input.position_ids.has_value()) {
        throw std::runtime_error("deepseek_v4: `input_ids` and `position_ids` are required");
    }
    const auto token_ids = download_i64(input.input_ids.value());
    const auto positions = download_i64(input.position_ids.value());
    if (token_ids.empty() || token_ids.size() != positions.size()) {
        throw std::runtime_error("deepseek_v4: `input_ids` and `position_ids` must have the same length");
    }
    auto ids = input.input_ids.value()->contiguous();
    if (ids->ndim() == 2) {
        if (ids->size(0) != 1) {
            throw std::runtime_error("deepseek_v4: only batch size 1 is implemented");
        }
        ids = ids->view({ids->numel()});
    }
    auto hidden = embed_->forward(ids)->contiguous();
    if (hidden->ndim() == 3) {
        hidden = hidden->view({hidden->numel() / hidden->size(hidden->ndim() - 1), hidden->size(hidden->ndim() - 1)});
    }
    infinicore::Tensor residual = hidden;
    std::vector<infinicore::Tensor> target_hidden;
    size_t next_target = 0;
    const bool time_decode = std::getenv("INFINI_METRIC_TIMER") != nullptr && token_ids.size() == 1;
    if (time_decode) {
        infinicore::context::syncStream();
    }
    const auto metric_stamp = std::chrono::steady_clock::now();
    auto metric_lap = [&]() {
        if (!time_decode) {
            return 0.0;
        }
        infinicore::context::syncStream();
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(now - metric_stamp).count();
    };
    for (size_t i = 0; i < layers_.size(); ++i) {
        try {
            residual = layers_[i]->forward(residual, positions, token_ids);
        } catch (const std::exception &err) {
            throw std::runtime_error("deepseek_v4 layer " + std::to_string(i) + ": " + err.what());
        }
        if (next_target < dspark_target_layers_.size() && static_cast<int>(i) == dspark_target_layers_[next_target]) {
            if (residual->ndim() == 2) {
                target_hidden.push_back(residual);
            } else {
                auto summed = infinicore::op::sum(residual, {1}, false);
                target_hidden.push_back(infinicore::op::mul_scalar(summed, 1.0 / static_cast<double>(residual->size(1))));
            }
            ++next_target;
        }
    }
    const double metric_layers = metric_lap();
    const double rms_eps = model_config_->get<double>("rms_norm_eps");
    const double hc_eps = model_config_->get<double>("hc_eps");
    auto collapsed = hc_head(residual, hc_head_fn_, hc_head_scale_, hc_head_base_, rms_eps, hc_eps);
    auto normed = norm_->forward(collapsed);
    auto logits = head_->forward(normed)->contiguous();
    auto logits_view = logits->view({size_t{1}, token_ids.size(), logits->size(logits->ndim() - 1)});
    const double metric_head = metric_lap();
    infinicore::Tensor draft_ids = infinicore::Tensor::empty({size_t{0}}, infinicore::DataType::I64, infinicore::Device::cpu());
    const bool run_dspark = !input.sample_all_positions && dspark_block_size_ > 0 && mtp_.size() >= 3 && target_hidden.size() == dspark_target_layers_.size();
    if (run_dspark) {
        auto main_hidden = target_hidden.size() == 1 ? target_hidden[0] : infinicore::op::cat(target_hidden, 1);
        auto main_x = mtp_.front()->combine_main(main_hidden);
        for (const auto &layer : mtp_) {
            layer->cache_draft_context(main_x, positions);
        }
        const double metric_cache = metric_lap();
        if (token_ids.size() == 1) {
            const size_t vocab = logits->size(logits->ndim() - 1);
            auto row = logits->narrow({{0, token_ids.size() - 1, 1}})->contiguous()->view({vocab});
            auto index = infinicore::Tensor::empty({size_t{1}}, infinicore::DataType::I64, row->device())->view({});
            infinicore::op::random_sample_(index, row, 0.0f, 1.0f, 1, 1.0f);
            const int64_t anchor = download_i64(index).at(0);
            std::vector<int64_t> block_ids(static_cast<size_t>(dspark_block_size_), dspark_noise_token_);
            std::vector<int64_t> block_positions(static_cast<size_t>(dspark_block_size_));
            block_ids[0] = anchor;
            for (int i = 0; i < dspark_block_size_; ++i) {
                block_positions[static_cast<size_t>(i)] = positions.back() + 1 + i;
            }
            auto block_tokens = infinicore::Tensor::empty({block_ids.size()}, infinicore::DataType::I64, infinicore::Device::cpu());
            std::memcpy(block_tokens->data(), block_ids.data(), block_ids.size() * sizeof(int64_t));
            block_tokens = block_tokens->to(hidden->device());
            auto block_hidden = embed_->forward(block_tokens)->contiguous();
            if (block_hidden->ndim() == 3) {
                block_hidden = block_hidden->view({block_hidden->numel() / block_hidden->size(block_hidden->ndim() - 1), block_hidden->size(block_hidden->ndim() - 1)});
            }
            for (const auto &layer : mtp_) {
                block_hidden = layer->forward(block_hidden, block_positions, block_ids);
            }
            const double metric_draft_layers = metric_lap();
            const auto sampled = mtp_.back()->sample_dspark_block(block_hidden, anchor, dspark_block_size_, *head_);
            draft_ids = infinicore::Tensor::empty({sampled.size()}, infinicore::DataType::I64, infinicore::Device::cpu());
            std::memcpy(draft_ids->data(), sampled.data(), sampled.size() * sizeof(int64_t));
            const double metric_sample = metric_lap();
            if (time_decode) {
                std::fprintf(stderr,
                             "METRIC_MTP target_layers_ms %.3f head_ms %.3f cache_ms %.3f draft_layers_ms %.3f sample_ms %.3f\n",
                             metric_layers,
                             metric_head - metric_layers,
                             metric_cache - metric_head,
                             metric_draft_layers - metric_cache,
                             metric_sample - metric_draft_layers);
            }
        }
    }
    gpu_timer_flush_thread();
    return {logits_view, draft_ids};
}

void DeepseekV4ForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        InfinilmModel::reset_cache(nullptr);
        return;
    }
    cache_config_ = cache_config->unique_copy();
    auto &kv_cache_vec = infinilm::global_state::get_forward_context().kv_cache_vec;
    kv_cache_vec.clear();
    const backends::AttentionBackend attention_backend = infinilm::global_state::get_infinilm_config().attention_backend;
    kv_cache_vec = deepseek_v4_allocate_kv_cache_tensors(cache_config, model_config_, attention_backend);
}

bool DeepseekV4ForCausalLM::try_load_parameter(const std::string &name, const infinicore::Tensor &param) {
    bool is_mtp = false;
    size_t layer = 0;
    size_t expert = 0;
    std::string proj;
    bool is_scale = false;
    if (!parse_routed_expert(name, is_mtp, layer, expert, proj, is_scale)) {
        return false;
    }
    auto &blocks = is_mtp ? mtp_ : layers_;
    if (layer >= blocks.size()) {
        throw std::runtime_error("deepseek_v4: routed expert layer is out of range for `" + name + "`");
    }
    blocks[layer]->ingest_expert(expert, proj, is_scale, param);
    return true;
}

std::shared_ptr<infinilm::config::ModelConfig>
create_deepseek_v4_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    if (model_config->get<std::string>("model_type") != "deepseek_v4") {
        throw std::runtime_error("create_deepseek_v4_model_config: `model_type` is not `deepseek_v4`");
    }
    return model_config;
}

} // namespace infinilm::models::deepseek_v4

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    deepseek_v4,
    infinilm::models::deepseek_v4::DeepseekV4ForCausalLM,
    infinilm::models::deepseek_v4::create_deepseek_v4_model_config);
} // namespace
