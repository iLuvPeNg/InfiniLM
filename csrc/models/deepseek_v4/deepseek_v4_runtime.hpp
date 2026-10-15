#pragma once

#include "infinicore/tensor.hpp"

#include <cstdint>
#include <vector>

namespace infinilm::models::deepseek_v4 {

infinicore::Tensor cast_dtype(const infinicore::Tensor &x, infinicore::DataType dtype);

std::vector<float> download_f32(const infinicore::Tensor &x);
std::vector<int64_t> download_i64(const infinicore::Tensor &x);

infinicore::Tensor upload_f32(const std::vector<float> &data,
                              const infinicore::Shape &shape,
                              const infinicore::Device &device);
infinicore::Tensor upload_i32(const std::vector<int32_t> &data,
                              const infinicore::Shape &shape,
                              const infinicore::Device &device);

infinicore::Tensor make_rope_cache(size_t max_pos,
                                   size_t rope_dim,
                                   double theta,
                                   const infinicore::Device &device);

// x is [tokens, heads, dim] or [tokens, dim]. RoPE covers the last rope_dim lanes.
infinicore::Tensor apply_rope(const infinicore::Tensor &x,
                              const std::vector<int64_t> &positions,
                              const infinicore::Tensor &cos_sin,
                              size_t nope_dim,
                              size_t rope_dim,
                              bool inverse);

struct MhcOut {
    infinicore::Tensor residual;
    infinicore::Tensor post_mix;
    infinicore::Tensor comb;
    infinicore::Tensor layer_input;
};

// residual is [tokens, hidden] on the first layer and [tokens, hc, hidden] after that.
MhcOut mhc_pre(const infinicore::Tensor &residual,
               const infinicore::Tensor &fn,
               const infinicore::Tensor &scale,
               const infinicore::Tensor &base,
               int hc_mult,
               double rms_eps,
               double hc_eps,
               int sinkhorn_iters);

infinicore::Tensor mhc_post(const infinicore::Tensor &x,
                            const infinicore::Tensor &residual,
                            const infinicore::Tensor &post_mix,
                            const infinicore::Tensor &comb);

infinicore::Tensor hc_head(const infinicore::Tensor &residual,
                           const infinicore::Tensor &fn,
                           const infinicore::Tensor &scale,
                           const infinicore::Tensor &base,
                           double rms_eps,
                           double hc_eps);

infinicore::Tensor swiglu_clamp(const infinicore::Tensor &gate,
                                const infinicore::Tensor &up,
                                float limit);

// Native infiniop int8 GEMM. weight is [N, K], weight_scale is [N, 1], input is [M, K].
infinicore::Tensor int8_gemm(const infinicore::Tensor &input,
                             const infinicore::Tensor &weight,
                             const infinicore::Tensor &weight_scale);

} // namespace infinilm::models::deepseek_v4
