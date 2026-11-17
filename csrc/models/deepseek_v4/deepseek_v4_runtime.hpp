#pragma once

#include "infinicore/tensor.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

namespace infinilm::models::deepseek_v4 {

enum class HostBucket : int {
    Sync = 0,
    Download,
    Upload,
    MhcPre,
    MhcPost,
    HcHead,
    MhcHostMath,
    MhcTokenLoop,
    Layer,
    Count,
};

// Env-gated (INFINI_HOST_TIMER) accounting of host-side stalls and scalar work.
// Everything is a no-op when the variable is unset; totals are printed to
// stderr at process exit as METRIC_HOST lines, summed over all threads.
class HostScope {
public:
    explicit HostScope(HostBucket bucket);
    ~HostScope();
    HostScope(const HostScope &) = delete;
    HostScope &operator=(const HostScope &) = delete;

private:
    HostBucket bucket_;
    bool active_;
    std::chrono::steady_clock::time_point start_;
};

enum class GpuBucket : int {
    MhcPreAttn = 0,
    AttnNorm,
    Attn,
    MhcPostAttn,
    MhcPreFfn,
    FfnNorm,
    Ffn,
    MhcPostFfn,
    FfnRouter,
    FfnShared,
    FfnQuant,
    FfnGemm,
    FfnScatter,
    FfnAllreduce,
    AttnQkv,
    AttnCompressor,
    AttnPack,
    AttnFlash,
    AttnProj,
    AttnAllreduce,
    Layer,
    Count,
};

// Env-gated (INFINI_GPU_TIMER) wall-clock attribution of one decoder layer.
// Each scope records a start/end CUDA event pair on the current device; the
// elapsed times are read back when the slot pool wraps and at process exit,
// then printed to stderr as METRIC_GPU lines (summed over all ranks).
class GpuScope {
public:
    explicit GpuScope(GpuBucket bucket);
    ~GpuScope();
    GpuScope(const GpuScope &) = delete;
    GpuScope &operator=(const GpuScope &) = delete;

private:
    GpuBucket bucket_;
    size_t slot_;
    bool active_;
};

// Reads back the current thread's recorded GPU sections. Cheap no-op unless
// INFINI_GPU_TIMER is set. Call once per forward step, on the rank thread.
void gpu_timer_flush_thread();

// With INFINI_GPU_DECODE_ONLY set, sections are only recorded while the
// current forward pass is a single-token decode, so the numbers are not
// diluted by the wider verify/prefill batches.
void gpu_timer_set_decode(bool decode);

infinicore::Tensor cast_dtype(const infinicore::Tensor &x, infinicore::DataType dtype);

std::vector<float> download_f32(const infinicore::Tensor &x);
std::vector<int64_t> download_i64(const infinicore::Tensor &x);

infinicore::Tensor upload_f32(const std::vector<float> &data,
                              const infinicore::Shape &shape,
                              const infinicore::Device &device);
infinicore::Tensor upload_i32(const std::vector<int32_t> &data,
                              const infinicore::Shape &shape,
                              const infinicore::Device &device);

// Device-side MoE routing (libdsv4_kernels.so). There used to be one host round
// trip per decoder layer here: download the [tokens, n_experts] router logits,
// rank the experts on the CPU, rebuild and upload the route table. Measured on
// this box it costs ~1.25 ms of wall per layer - 40% of decode - because all
// eight TP ranks then wait to line up at the next collective.
struct DeviceRoute {
    infinicore::Tensor counts;     // [local_experts] i32, rows per local expert
    infinicore::Tensor row_token;  // [rows_cap] i32, token of each route row
    infinicore::Tensor row_weight; // [rows_cap] f32
    infinicore::Tensor mix;        // [tokens, rows_cap] f32, weights in row order
};

// Returns false when the kernel library is missing or the shape is outside what
// it handles; the caller then keeps the host implementation. rows_cap must be
// tokens * topk - the kernels never need to know the live route count.
bool device_moe_route(const infinicore::Tensor &logits,
                      const infinicore::Tensor *bias,
                      const infinicore::Tensor *tid2eid,
                      const std::vector<int64_t> &token_ids,
                      size_t topk,
                      size_t tp_size,
                      size_t tp_rank,
                      bool hash,
                      bool renormalize,
                      float routed_scale,
                      size_t rows_cap,
                      DeviceRoute &out);

// Writes out's rows in expert-major order using per-expert counts that live on
// the device, so no host read is needed. Rows past sum(counts) are untouched.
bool device_group_gemm(const infinicore::Tensor &out,
                       const infinicore::Tensor &input_i8,
                       const infinicore::Tensor &weight,
                       const infinicore::Tensor &input_scale,
                       const infinicore::Tensor &weight_scale,
                       const infinicore::Tensor &counts,
                       size_t rows_cap,
                       size_t experts);

// Runs the compressor emit (online softmax over n_ctx state rows, RMS scale,
// partial RoPE) on the device for every position in emit_positions, so the
// state slab never has to be drained. `out` is F32 [emit_positions.size(), head].
// Returns false when the kernel library is missing or the shapes do not fit.
bool device_attn_pack(const infinicore::Tensor &packed,
                      const infinicore::Tensor &swa_cache,
                      const infinicore::Tensor *compressed_cache,
                      const std::vector<size_t> &starts,
                      const std::vector<size_t> &swa_at,
                      const std::vector<size_t> &swa_n,
                      const std::vector<size_t> &cmp_at,
                      const std::vector<size_t> &cmp_n);

bool device_compress_emit(const infinicore::Tensor &state_cache,
                          size_t slab_lo,
                          const std::vector<int64_t> &emit_positions,
                          size_t ratio,
                          size_t n_ctx,
                          size_t head,
                          size_t width,
                          size_t rope_dim,
                          const infinicore::Tensor &norm_weight,
                          const infinicore::Tensor &inv_freq,
                          float eps,
                          infinicore::Tensor &out);

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
