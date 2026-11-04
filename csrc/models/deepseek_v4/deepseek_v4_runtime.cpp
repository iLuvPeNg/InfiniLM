#include "deepseek_v4_runtime.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/cast.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/embedding.hpp"
#include "infinicore/ops/fmin.hpp"
#include "infinicore/ops/linear_w8a8i8.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/mul.hpp"
#include "infinicore/ops/mul_scalar.hpp"
#include "infinicore/ops/sigmoid.hpp"
#include "infinicore/ops/silu.hpp"
#include "infinicore/ops/sum.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v4 {
namespace {

constexpr float kHcPostAlpha = 2.0f;
constexpr size_t kHostRetainLimit = 32;

void sync_device() {
    infinicore::context::syncDevice();
}

// Async H2D reads the CPU tensor after this function returns. Keep those
// tensors until a later sync, and flush a whole batch instead of stalling
// the device on every upload.
void retain_host(infinicore::Tensor cpu) {
    struct Retainer {
        std::vector<infinicore::Tensor> kept;
        ~Retainer() {
            if (!kept.empty()) {
                infinicore::context::syncDevice();
            }
        }
    };
    thread_local Retainer retainer;
    retainer.kept.push_back(std::move(cpu));
    if (retainer.kept.size() >= kHostRetainLimit) {
        sync_device();
        retainer.kept.clear();
    }
}

float sigmoid_cpu(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

void sinkhorn_cpu(std::vector<float> &comb, int hc, int iters, float eps) {
    auto at = [&](int row, int col) -> float & {
        return comb[static_cast<size_t>(row * hc + col)];
    };
    for (int row = 0; row < hc; ++row) {
        float max_v = at(row, 0);
        for (int col = 1; col < hc; ++col) {
            max_v = std::max(max_v, at(row, col));
        }
        float denom = 0.0f;
        for (int col = 0; col < hc; ++col) {
            const float e = std::exp(at(row, col) - max_v);
            at(row, col) = e;
            denom += e;
        }
        for (int col = 0; col < hc; ++col) {
            at(row, col) = at(row, col) / denom + eps;
        }
    }
    auto col_norm = [&]() {
        for (int col = 0; col < hc; ++col) {
            float denom = 0.0f;
            for (int row = 0; row < hc; ++row) {
                denom += at(row, col);
            }
            denom += eps;
            for (int row = 0; row < hc; ++row) {
                at(row, col) /= denom;
            }
        }
    };
    auto row_norm = [&]() {
        for (int row = 0; row < hc; ++row) {
            float denom = eps;
            for (int col = 0; col < hc; ++col) {
                denom += at(row, col);
            }
            for (int col = 0; col < hc; ++col) {
                at(row, col) /= denom;
            }
        }
    };
    col_norm();
    for (int iter = 1; iter < iters; ++iter) {
        row_norm();
        col_norm();
    }
}

infinicore::Tensor f32_row(const std::vector<float> &data, size_t offset, size_t n, const infinicore::Device &device) {
    return upload_f32(std::vector<float>(data.begin() + static_cast<std::ptrdiff_t>(offset),
                                         data.begin() + static_cast<std::ptrdiff_t>(offset + n)),
                      {1, n},
                      device);
}

} // namespace

infinicore::Tensor cast_dtype(const infinicore::Tensor &x, infinicore::DataType dtype) {
    if (x->dtype() == dtype) {
        return x->contiguous();
    }
    auto out = infinicore::Tensor::empty(x->shape(), dtype, x->device());
    infinicore::op::cast_(out, x->contiguous());
    return out;
}

std::vector<float> download_f32(const infinicore::Tensor &x) {
    sync_device();
    auto cpu = cast_dtype(x, infinicore::DataType::F32)->to(infinicore::Device::cpu())->contiguous();
    const auto *ptr = reinterpret_cast<const float *>(cpu->data());
    return std::vector<float>(ptr, ptr + cpu->numel());
}

std::vector<int64_t> download_i64(const infinicore::Tensor &x) {
    sync_device();
    auto cpu = x->to(infinicore::Device::cpu())->contiguous();
    sync_device();
    std::vector<int64_t> out(cpu->numel());
    if (cpu->dtype() == infinicore::DataType::I64) {
        const auto *ptr = reinterpret_cast<const int64_t *>(cpu->data());
        std::copy(ptr, ptr + cpu->numel(), out.begin());
    } else if (cpu->dtype() == infinicore::DataType::I32) {
        const auto *ptr = reinterpret_cast<const int32_t *>(cpu->data());
        for (size_t i = 0; i < cpu->numel(); ++i) {
            out[i] = ptr[i];
        }
    } else {
        throw std::runtime_error("deepseek_v4: expected an int32 or int64 tensor");
    }
    return out;
}

infinicore::Tensor upload_f32(const std::vector<float> &data,
                              const infinicore::Shape &shape,
                              const infinicore::Device &device) {
    size_t n = 1;
    for (size_t dim : shape) {
        n *= dim;
    }
    if (n != data.size()) {
        throw std::runtime_error("deepseek_v4: upload_f32 size mismatch");
    }
    auto cpu = infinicore::Tensor::empty(shape, infinicore::DataType::F32, infinicore::Device::cpu());
    std::memcpy(cpu->data(), data.data(), n * sizeof(float));
    if (device.getType() == infinicore::Device::Type::CPU) {
        return cpu;
    }
    auto gpu = cpu->to(device);
    retain_host(cpu);
    return gpu;
}

infinicore::Tensor upload_i32(const std::vector<int32_t> &data,
                              const infinicore::Shape &shape,
                              const infinicore::Device &device) {
    size_t n = 1;
    for (size_t dim : shape) {
        n *= dim;
    }
    if (n != data.size()) {
        throw std::runtime_error("deepseek_v4: upload_i32 size mismatch");
    }
    auto cpu = infinicore::Tensor::empty(shape, infinicore::DataType::I32, infinicore::Device::cpu());
    std::memcpy(cpu->data(), data.data(), n * sizeof(int32_t));
    if (device.getType() == infinicore::Device::Type::CPU) {
        return cpu;
    }
    auto gpu = cpu->to(device);
    retain_host(cpu);
    return gpu;
}

infinicore::Tensor make_rope_cache(size_t max_pos,
                                   size_t rope_dim,
                                   double theta,
                                   const infinicore::Device &device) {
    if (rope_dim % 2 != 0) {
        throw std::runtime_error("deepseek_v4: rope dim must be even");
    }
    const size_t half = rope_dim / 2;
    std::vector<float> table(max_pos * rope_dim);
    for (size_t pos = 0; pos < max_pos; ++pos) {
        for (size_t i = 0; i < half; ++i) {
            const double freq = 1.0 / std::pow(theta, (2.0 * static_cast<double>(i)) / static_cast<double>(rope_dim));
            const double ang = static_cast<double>(pos) * freq;
            table[pos * rope_dim + i] = static_cast<float>(std::cos(ang));
            table[pos * rope_dim + half + i] = static_cast<float>(std::sin(ang));
        }
    }
    return upload_f32(table, {max_pos, rope_dim}, device);
}

infinicore::Tensor apply_rope(const infinicore::Tensor &x,
                              const std::vector<int64_t> &positions,
                              const infinicore::Tensor &cos_sin,
                              size_t nope_dim,
                              size_t rope_dim,
                              bool inverse) {
    const bool three_d = x->ndim() == 3;
    if (!three_d && x->ndim() != 2) {
        throw std::runtime_error("deepseek_v4 rope: expected rank 2 or 3");
    }
    const size_t tokens = three_d ? x->size(0) : x->size(0);
    const size_t heads = three_d ? x->size(1) : 1;
    const size_t dim = x->size(x->ndim() - 1);
    if (positions.size() != tokens || nope_dim + rope_dim != dim || rope_dim % 2 != 0) {
        throw std::runtime_error("deepseek_v4 rope: shape mismatch");
    }
    auto flat = three_d ? x->contiguous()->view({tokens * heads, dim}) : x->contiguous();
    auto xf = cast_dtype(flat, infinicore::DataType::F32);
    const size_t half = rope_dim / 2;
    const size_t rows = tokens * heads;
    std::vector<int64_t> repeated;
    repeated.reserve(rows);
    for (int64_t pos : positions) {
        for (size_t h = 0; h < heads; ++h) {
            repeated.push_back(pos);
        }
    }
    auto pos = infinicore::Tensor::empty({rows}, infinicore::DataType::I64, infinicore::Device::cpu());
    std::memcpy(pos->data(), repeated.data(), rows * sizeof(int64_t));
    auto pos_dev = pos->to(x->device());
    retain_host(pos);
    auto cs = infinicore::op::embedding(pos_dev, cos_sin);
    auto cos = cs->narrow({{1, 0, half}})->contiguous();
    auto sin = cs->narrow({{1, half, half}})->contiguous();
    auto nope = xf->narrow({{1, 0, nope_dim}})->contiguous();
    auto rope = xf->narrow({{1, nope_dim, rope_dim}})->contiguous()->view({rows, half, 2});
    auto x0 = rope->narrow({{2, 0, 1}})->contiguous()->view({rows, half});
    auto x1 = rope->narrow({{2, 1, 1}})->contiguous()->view({rows, half});
    infinicore::Tensor y0;
    infinicore::Tensor y1;
    if (!inverse) {
        y0 = infinicore::op::add(infinicore::op::mul(x0, cos), infinicore::op::mul_scalar(infinicore::op::mul(x1, sin), -1.0));
        y1 = infinicore::op::add(infinicore::op::mul(x0, sin), infinicore::op::mul(x1, cos));
    } else {
        y0 = infinicore::op::add(infinicore::op::mul(x0, cos), infinicore::op::mul(x1, sin));
        y1 = infinicore::op::add(infinicore::op::mul(x1, cos), infinicore::op::mul_scalar(infinicore::op::mul(x0, sin), -1.0));
    }
    auto paired = infinicore::Tensor::empty({rows, half, 2}, infinicore::DataType::F32, x->device());
    paired->narrow({{2, 0, 1}})->view({rows, half})->copy_from(y0->contiguous());
    paired->narrow({{2, 1, 1}})->view({rows, half})->copy_from(y1->contiguous());
    auto rotated = infinicore::op::cat({nope, paired->view({rows, rope_dim})->contiguous()}, 1);
    auto bf = cast_dtype(rotated, x->dtype());
    if (!three_d) {
        return bf;
    }
    return bf->view({tokens, heads, dim});
}

MhcOut mhc_pre(const infinicore::Tensor &residual,
               const infinicore::Tensor &fn,
               const infinicore::Tensor &scale,
               const infinicore::Tensor &base,
               int hc_mult,
               double rms_eps,
               double hc_eps,
               int sinkhorn_iters) {
    if (hc_mult <= 0) {
        throw std::runtime_error("deepseek_v4 mhc: hc_mult must be positive");
    }
    const bool broadcast = residual->ndim() == 2;
    const size_t tokens = residual->size(0);
    const size_t hidden = broadcast ? residual->size(1) : residual->size(2);
    const size_t hc = static_cast<size_t>(hc_mult);
    const size_t mix = hc * (2 + hc);
    const auto device = residual->device();
    const auto dtype = residual->dtype();

    infinicore::Tensor flat32;
    infinicore::Tensor fn_used;
    size_t rms_mult;
    if (broadcast) {
        flat32 = cast_dtype(residual->contiguous(), infinicore::DataType::F32);
        auto viewed = fn->contiguous()->view({mix, hc, hidden});
        fn_used = infinicore::op::sum(viewed, {1}, false);
        rms_mult = 1;
    } else {
        flat32 = cast_dtype(residual->contiguous()->view({tokens, hc * hidden}), infinicore::DataType::F32);
        fn_used = fn;
        rms_mult = hc;
    }
    auto fn_t = cast_dtype(fn_used, infinicore::DataType::F32)->permute({1, 0})->contiguous();
    auto gemm = infinicore::op::matmul(flat32, fn_t);
    auto sumsq = infinicore::op::sum(infinicore::op::mul(flat32, flat32), {1}, false);
    const auto gemm_cpu = download_f32(gemm);
    const auto sumsq_cpu = download_f32(sumsq);
    const auto scale_cpu = download_f32(scale);
    const auto base_cpu = download_f32(base);
    if (gemm_cpu.size() != tokens * mix || sumsq_cpu.size() != tokens || scale_cpu.size() < 3 || base_cpu.size() < mix) {
        throw std::runtime_error("deepseek_v4 mhc: mix tensor size mismatch");
    }

    std::vector<float> pre_cpu(tokens * hc);
    std::vector<float> post_cpu(tokens * hc);
    std::vector<float> comb_cpu(tokens * hc * hc);
    const float eps = static_cast<float>(hc_eps);
    const float rms_eps_f = static_cast<float>(rms_eps);
    const float denom_count = static_cast<float>(rms_mult * hidden);
    for (size_t t = 0; t < tokens; ++t) {
        const float rms = 1.0f / std::sqrt(sumsq_cpu[t] / denom_count + rms_eps_f);
        const float *g = gemm_cpu.data() + t * mix;
        for (size_t c = 0; c < hc; ++c) {
            pre_cpu[t * hc + c] = sigmoid_cpu(scale_cpu[0] * g[c] * rms + base_cpu[c]) + eps;
            post_cpu[t * hc + c] = sigmoid_cpu(scale_cpu[1] * g[hc + c] * rms + base_cpu[hc + c]) * kHcPostAlpha;
        }
        std::vector<float> comb(hc * hc);
        for (size_t i = 0; i < hc * hc; ++i) {
            comb[i] = scale_cpu[2] * g[2 * hc + i] * rms + base_cpu[2 * hc + i];
        }
        sinkhorn_cpu(comb, static_cast<int>(hc), sinkhorn_iters, eps);
        std::copy(comb.begin(), comb.end(), comb_cpu.begin() + static_cast<std::ptrdiff_t>(t * hc * hc));
    }

    auto residual_out = broadcast ? infinicore::Tensor::empty({tokens, hc, hidden}, dtype, device) : residual->contiguous();
    if (broadcast) {
        auto src = residual->contiguous();
        for (size_t c = 0; c < hc; ++c) {
            residual_out->narrow({{1, c, 1}})->view({tokens, hidden})->copy_from(src);
        }
    }
    auto pre_all = upload_f32(pre_cpu, {tokens, hc}, device);
    auto layer_input = infinicore::Tensor::empty({tokens, hidden}, dtype, device);
    for (size_t t = 0; t < tokens; ++t) {
        auto pre = pre_all->narrow({{0, t, 1}})->contiguous()->view({1, hc});
        auto res_t = cast_dtype(residual_out->narrow({{0, t, 1}})->contiguous()->view({hc, hidden}), infinicore::DataType::F32);
        auto mixed = infinicore::op::matmul(pre, res_t);
        layer_input->narrow({{0, t, 1}})->view({hidden})->copy_from(cast_dtype(mixed->view({hidden}), dtype));
    }

    MhcOut out;
    out.residual = residual_out;
    out.post_mix = upload_f32(post_cpu, {tokens, hc}, device);
    out.comb = upload_f32(comb_cpu, {tokens, hc, hc}, device);
    out.layer_input = layer_input;
    return out;
}

infinicore::Tensor mhc_post(const infinicore::Tensor &x,
                            const infinicore::Tensor &residual,
                            const infinicore::Tensor &post_mix,
                            const infinicore::Tensor &comb) {
    const size_t tokens = residual->size(0);
    const size_t hc = residual->size(1);
    const size_t hidden = residual->size(2);
    const auto device = residual->device();
    const auto dtype = residual->dtype();
    const auto post_cpu = download_f32(post_mix);
    auto out = infinicore::Tensor::empty({tokens, hc, hidden}, dtype, device);
    auto x_flat = x->contiguous()->view({tokens, hidden});
    for (size_t t = 0; t < tokens; ++t) {
        auto comb_t = comb->narrow({{0, t, 1}})->contiguous()->view({hc, hc})->permute({1, 0})->contiguous();
        auto res_t = cast_dtype(residual->narrow({{0, t, 1}})->contiguous()->view({hc, hidden}), infinicore::DataType::F32);
        auto mixed = infinicore::op::matmul(cast_dtype(comb_t, infinicore::DataType::F32), res_t);
        auto x_t = cast_dtype(x_flat->narrow({{0, t, 1}})->contiguous()->view({hidden}), infinicore::DataType::F32);
        for (size_t c = 0; c < hc; ++c) {
            auto row = mixed->narrow({{0, c, 1}})->contiguous()->view({hidden});
            auto added = infinicore::op::add(row, infinicore::op::mul_scalar(x_t, post_cpu[t * hc + c]));
            mixed->narrow({{0, c, 1}})->view({hidden})->copy_from(added);
        }
        out->narrow({{0, t, 1}})->view({hc, hidden})->copy_from(cast_dtype(mixed, dtype));
    }
    return out;
}

infinicore::Tensor hc_head(const infinicore::Tensor &residual,
                           const infinicore::Tensor &fn,
                           const infinicore::Tensor &scale,
                           const infinicore::Tensor &base,
                           double rms_eps,
                           double hc_eps) {
    const size_t tokens = residual->size(0);
    const size_t hc = residual->size(1);
    const size_t hidden = residual->size(2);
    const auto device = residual->device();
    const auto dtype = residual->dtype();
    auto flat32 = cast_dtype(residual->contiguous()->view({tokens, hc * hidden}), infinicore::DataType::F32);
    auto fn_t = cast_dtype(fn, infinicore::DataType::F32)->permute({1, 0})->contiguous();
    auto mixes = infinicore::op::matmul(flat32, fn_t);
    auto sumsq = infinicore::op::sum(infinicore::op::mul(flat32, flat32), {1}, false);
    const auto mix_cpu = download_f32(mixes);
    const auto sumsq_cpu = download_f32(sumsq);
    const auto scale_cpu = download_f32(scale);
    const auto base_cpu = download_f32(base);
    const float eps = static_cast<float>(hc_eps);
    const float rms_eps_f = static_cast<float>(rms_eps);
    const float denom_count = static_cast<float>(hc * hidden);
    const float scale0 = scale_cpu.empty() ? 1.0f : scale_cpu[0];
    std::vector<float> pre_cpu(tokens * hc);
    for (size_t t = 0; t < tokens; ++t) {
        const float rms = 1.0f / std::sqrt(sumsq_cpu[t] / denom_count + rms_eps_f);
        for (size_t c = 0; c < hc; ++c) {
            pre_cpu[t * hc + c] = sigmoid_cpu(scale0 * mix_cpu[t * hc + c] * rms + base_cpu[c]) + eps;
        }
    }
    auto pre_all = upload_f32(pre_cpu, {tokens, hc}, device);
    auto out = infinicore::Tensor::empty({tokens, hidden}, dtype, device);
    for (size_t t = 0; t < tokens; ++t) {
        auto pre = pre_all->narrow({{0, t, 1}})->contiguous()->view({1, hc});
        auto res_t = cast_dtype(residual->narrow({{0, t, 1}})->contiguous()->view({hc, hidden}), infinicore::DataType::F32);
        auto mixed = infinicore::op::matmul(pre, res_t);
        out->narrow({{0, t, 1}})->view({hidden})->copy_from(cast_dtype(mixed->view({hidden}), dtype));
    }
    return out;
}

infinicore::Tensor int8_gemm(const infinicore::Tensor &input,
                             const infinicore::Tensor &weight,
                             const infinicore::Tensor &weight_scale) {
    auto x = input->contiguous();
    if (x->ndim() != 2) {
        throw std::runtime_error("deepseek_v4 int8 gemm expects a 2D activation");
    }
    return infinicore::op::linear_w8a8i8(x, weight->contiguous(), weight_scale->contiguous(), std::nullopt);
}

infinicore::Tensor swiglu_clamp(const infinicore::Tensor &gate,
                                const infinicore::Tensor &up,
                                float limit) {
    auto limit_hi = infinicore::op::mul_scalar(infinicore::Tensor::ones(gate->shape(), gate->dtype(), gate->device()), static_cast<double>(limit));
    auto gate_c = infinicore::op::fmin(gate, limit_hi);
    auto up_neg = infinicore::op::mul_scalar(up, -1.0);
    auto up_neg_c = infinicore::op::fmin(up_neg, limit_hi);
    auto up_lo = infinicore::op::mul_scalar(up_neg_c, -1.0);
    auto up_c = infinicore::op::fmin(up_lo, limit_hi);
    return infinicore::op::mul(infinicore::op::silu(gate_c), up_c);
}

} // namespace infinilm::models::deepseek_v4
