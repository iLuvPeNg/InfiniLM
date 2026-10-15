#include "deepseek_v4_model.hpp"

#include "deepseek_v4_runtime.hpp"
#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

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

DeepseekV4MarkovHead::DeepseekV4MarkovHead(size_t vocab_size,
                                           const infinicore::DataType &dtype,
                                           const infinicore::Device &device) {
    INFINICORE_NN_MODULE_INIT(markov_w1, 256, vocab_size, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(markov_w2, 256, vocab_size, false, dtype, device);
}

DeepseekV4ConfidenceHead::DeepseekV4ConfidenceHead(size_t in_features,
                                                   const infinicore::DataType &dtype,
                                                   const infinicore::Device &device) {
    INFINICORE_NN_MODULE_INIT(proj, in_features, 1, false, dtype, device);
}

DeepseekV4DecoderLayer::DeepseekV4DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               const infinicore::Device &device,
                                               int compress_ratio,
                                               bool has_tid2eid,
                                               bool has_gate_bias,
                                               MtpKind mtp_kind) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden = model_config->get<size_t>("hidden_size");
    const size_t vocab = model_config->get<size_t>("vocab_size");
    const int hc_mult = model_config->get<int>("hc_mult");
    const double eps = model_config->get<double>("rms_norm_eps");
    hc_mult_ = hc_mult;
    rms_eps_ = eps;
    hc_eps_ = model_config->get<double>("hc_eps");
    sinkhorn_iters_ = model_config->get<int>("hc_sinkhorn_iters");
    const size_t mix_hc = static_cast<size_t>((2 + hc_mult) * hc_mult);
    const size_t hc_dim = static_cast<size_t>(hc_mult) * hidden;

    INFINICORE_NN_MODULE_INIT(attn, model_config, device, compress_ratio);
    INFINICORE_NN_MODULE_INIT(ffn, model_config, device, has_tid2eid, has_gate_bias);
    INFINICORE_NN_MODULE_INIT(attn_norm, hidden, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(ffn_norm, hidden, eps, dtype, device);
    INFINICORE_NN_PARAMETER_INIT(hc_attn_fn, ({mix_hc, hc_dim}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_attn_base, ({mix_hc}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_attn_scale, ({size_t{3}}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_ffn_fn, ({mix_hc, hc_dim}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_ffn_base, ({mix_hc}, infinicore::DataType::F32, device));
    INFINICORE_NN_PARAMETER_INIT(hc_ffn_scale, ({size_t{3}}, infinicore::DataType::F32, device));

    if (mtp_kind == MtpKind::MainProj) {
        INFINICORE_NN_MODULE_INIT(main_proj, hidden * 3, hidden, false, dtype, device);
        INFINICORE_NN_MODULE_INIT(main_norm, hidden, eps, dtype, device);
    } else if (mtp_kind == MtpKind::Markov) {
        INFINICORE_NN_MODULE_INIT(markov_head, vocab, dtype, device);
        INFINICORE_NN_MODULE_INIT(confidence_head, hidden + 256, dtype, device);
        INFINICORE_NN_MODULE_INIT(norm, hidden, eps, dtype, device);
        INFINICORE_NN_PARAMETER_INIT(hc_head_fn, ({static_cast<size_t>(hc_mult), hc_dim}, infinicore::DataType::F32, device));
        INFINICORE_NN_PARAMETER_INIT(hc_head_base, ({static_cast<size_t>(hc_mult)}, infinicore::DataType::F32, device));
        INFINICORE_NN_PARAMETER_INIT(hc_head_scale, ({size_t{1}}, infinicore::DataType::F32, device));
    }
}

void DeepseekV4DecoderLayer::ingest_expert(size_t expert_id,
                                           const std::string &proj,
                                           bool is_scale,
                                           const infinicore::Tensor &tensor) {
    ffn_->ingest_expert(expert_id, proj, is_scale, tensor);
}

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
            kind));
    }
}

infinicore::Tensor DeepseekV4DecoderLayer::forward(infinicore::Tensor residual,
                                                       const std::vector<int64_t> &positions,
                                                       const std::vector<int64_t> &token_ids) const {
    auto attn_mix = mhc_pre(residual, hc_attn_fn_, hc_attn_scale_, hc_attn_base_, hc_mult_, rms_eps_, hc_eps_, sinkhorn_iters_);
    auto attn_normed = attn_norm_->forward(attn_mix.layer_input);
    auto attn_out = attn_->forward(attn_normed, positions);
    auto mid = mhc_post(attn_out, attn_mix.residual, attn_mix.post_mix, attn_mix.comb);

    auto ffn_mix = mhc_pre(mid, hc_ffn_fn_, hc_ffn_scale_, hc_ffn_base_, hc_mult_, rms_eps_, hc_eps_, sinkhorn_iters_);
    auto ffn_normed = ffn_norm_->forward(ffn_mix.layer_input);
    auto ffn_out = ffn_->forward(ffn_normed, token_ids);
    return mhc_post(ffn_out, ffn_mix.residual, ffn_mix.post_mix, ffn_mix.comb);
}

infinilm::InfinilmModel::Output DeepseekV4ForCausalLM::forward(const Input &input) const {
    if (!input.input_ids.has_value() || !input.position_ids.has_value()) {
        throw std::runtime_error("deepseek_v4: input_ids and position_ids are required");
    }
    const auto token_ids = download_i64(input.input_ids.value());
    const auto positions = download_i64(input.position_ids.value());
    if (token_ids.empty() || token_ids.size() != positions.size()) {
        throw std::runtime_error("deepseek_v4: input_ids and position_ids must have the same length");
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
    for (size_t i = 0; i < layers_.size(); ++i) {
        try {
            residual = layers_[i]->forward(residual, positions, token_ids);
        } catch (const std::exception &err) {
            throw std::runtime_error("deepseek_v4 layer " + std::to_string(i) + ": " + err.what());
        }
    }
    const double rms_eps = model_config_->get<double>("rms_norm_eps");
    const double hc_eps = model_config_->get<double>("hc_eps");
    auto collapsed = hc_head(residual, hc_head_fn_, hc_head_scale_, hc_head_base_, rms_eps, hc_eps);
    auto normed = norm_->forward(collapsed);
    auto logits = head_->forward(normed)->contiguous();
    return {logits->view({size_t{1}, token_ids.size(), logits->size(logits->ndim() - 1)})};
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
        throw std::runtime_error("deepseek_v4 routed expert layer out of range: " + name);
    }
    blocks[layer]->ingest_expert(expert, proj, is_scale, param);
    return true;
}

std::shared_ptr<infinilm::config::ModelConfig>
create_deepseek_v4_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    if (model_config->get<std::string>("model_type") != "deepseek_v4") {
        throw std::runtime_error("create_deepseek_v4_model_config: model_type is not deepseek_v4");
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
