#include "deepseek_v4_decoderLayer.hpp"

#include "deepseek_v4_runtime.hpp"

namespace infinilm::models::deepseek_v4 {

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
                                               size_t layer_idx,
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

    INFINICORE_NN_MODULE_INIT(attn, model_config, device, compress_ratio, layer_idx);
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

} // namespace infinilm::models::deepseek_v4
