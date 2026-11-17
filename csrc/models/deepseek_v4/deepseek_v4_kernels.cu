// Fused device-side kernels for DeepSeek-V4-Flash.
//
// Two stages that used to be chains of small framework ops (and, for mHC, host
// math between full device drains) run here as single launches:
//
//  * mhc_pre: rms scaling, the sigmoid pre/post mixes and the sinkhorn
//    combination, straight off the [tokens, mix] GEMM output. mhc_pre runs
//    twice per decoder layer and the host version drained the device twice per
//    call, so this was the largest host stall in decode.
//  * rope: one launch per call instead of the cast/narrow/mul/cat chain that
//    apply_rope used to build, three or four times per layer.
//
// Built into libdsv4_kernels.so, which InfiniLM loads lazily from its own
// directory; when the library is missing the model falls back to the host
// implementations in deepseek_v4_runtime.cpp.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

// hc_mult is 4 for the released checkpoints; keep some headroom so a larger
// value still works instead of silently falling back to the host path.
constexpr int kMaxHc = 16;

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Byte-for-byte the same sequence as sinkhorn_cpu() in deepseek_v4_runtime.cpp:
// row softmax with max subtraction, add eps, column normalise, then alternate
// row/column normalisation for the remaining iterations.
__device__ void sinkhorn_f(float *m, int hc, int iters, float eps) {
    for (int row = 0; row < hc; ++row) {
        float *r = m + row * hc;
        float max_v = r[0];
        for (int c = 1; c < hc; ++c) {
            max_v = fmaxf(max_v, r[c]);
        }
        float denom = 0.0f;
        for (int c = 0; c < hc; ++c) {
            r[c] = expf(r[c] - max_v);
            denom += r[c];
        }
        for (int c = 0; c < hc; ++c) {
            r[c] = r[c] / denom + eps;
        }
    }
    for (int c = 0; c < hc; ++c) {
        float denom = 0.0f;
        for (int row = 0; row < hc; ++row) {
            denom += m[row * hc + c];
        }
        denom += eps;
        for (int row = 0; row < hc; ++row) {
            m[row * hc + c] /= denom;
        }
    }
    for (int iter = 1; iter < iters; ++iter) {
        for (int row = 0; row < hc; ++row) {
            float *r = m + row * hc;
            float denom = eps;
            for (int c = 0; c < hc; ++c) {
                denom += r[c];
            }
            for (int c = 0; c < hc; ++c) {
                r[c] /= denom;
            }
        }
        for (int c = 0; c < hc; ++c) {
            float denom = 0.0f;
            for (int row = 0; row < hc; ++row) {
                denom += m[row * hc + c];
            }
            denom += eps;
            for (int row = 0; row < hc; ++row) {
                m[row * hc + c] /= denom;
            }
        }
    }
}

__global__ void mhc_pre_kernel(const float *__restrict__ gemm,
                               const float *__restrict__ sumsq,
                               const float *__restrict__ scale,
                               const float *__restrict__ base,
                               float *__restrict__ pre,
                               float *__restrict__ post,
                               float *__restrict__ comb,
                               int tokens,
                               int hc,
                               int mix,
                               float inv_count,
                               float rms_eps,
                               float hc_eps,
                               int iters,
                               float post_alpha) {
    const int t = blockIdx.x;
    if (t >= tokens) {
        return;
    }
    __shared__ float sh_rms;
    __shared__ float sh_scale[3];
    __shared__ float sh_comb[kMaxHc * kMaxHc];
    if (threadIdx.x < 3) {
        sh_scale[threadIdx.x] = scale[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        sh_rms = 1.0f / sqrtf(sumsq[t] * inv_count + rms_eps);
    }
    __syncthreads();
    const float rms = sh_rms;
    const float *row = gemm + static_cast<long long>(t) * mix;
    float *pre_row = pre + static_cast<long long>(t) * hc;
    float *post_row = post + static_cast<long long>(t) * hc;
    for (int c = threadIdx.x; c < hc; c += blockDim.x) {
        pre_row[c] = sigmoid_f(sh_scale[0] * row[c] * rms + base[c]) + hc_eps;
        post_row[c] = sigmoid_f(sh_scale[1] * row[hc + c] * rms + base[hc + c]) * post_alpha;
    }
    if (threadIdx.x == 0) {
        for (int i = 0; i < hc * hc; ++i) {
            sh_comb[i] = sh_scale[2] * row[2 * hc + i] * rms + base[2 * hc + i];
        }
        sinkhorn_f(sh_comb, hc, iters, hc_eps);
        float *dst = comb + static_cast<long long>(t) * hc * hc;
        for (int i = 0; i < hc * hc; ++i) {
            dst[i] = sh_comb[i];
        }
    }
}

template <typename T>
__device__ __forceinline__ float to_f32(T v);
template <>
__device__ __forceinline__ float to_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <>
__device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <>
__device__ __forceinline__ __half from_f32<__half>(float v) { return __float2half(v); }

// The paired (interleaved) half-rotation used by the model: the last `rope`
// lanes of each row are rotated in place, the leading `nope` lanes are copied
// by the caller. `pos` holds one position per token, shared by all `heads` rows
// of that token.
template <typename T>
__global__ void rope_kernel(T *__restrict__ x,
                            const float *__restrict__ cos_sin,
                            const long long *__restrict__ pos,
                            long long rows,
                            int heads,
                            int dim,
                            int nope,
                            int rope,
                            int inverse) {
    // The elementwise path this replaces evaluates every multiply and every add
    // as its own kernel, so each step rounds on its own. Contracting them into an
    // FMA changes the last bit of a handful of lanes per call, and on the bf16
    // ties that happen about twice per call that is enough to flip a greedy
    // token, so scripts/build_kernels.sh passes -ffp-contract=off. The rounded
    // intrinsics do not help here: clang defines __fadd_rn/__fmul_rn as plain
    // `+`/`*`, and this pragma is not honoured in device code by this toolchain.
#pragma clang fp contract(off)
    const int half = rope >> 1;
    const long long total = rows * half;
    for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long row = idx / half;
        const int i = static_cast<int>(idx - row * half);
        const float *cs = cos_sin + pos[row / heads] * rope;
        const float c = cs[i];
        const float s = cs[i + half];
        T *lane = x + row * dim + nope + 2 * i;
        const float x0 = to_f32<T>(lane[0]);
        const float x1 = to_f32<T>(lane[1]);
        float y0;
        float y1;
        if (!inverse) {
            y0 = x0 * c - x1 * s;
            y1 = x0 * s + x1 * c;
        } else {
            y0 = x0 * c + x1 * s;
            y1 = x1 * c - x0 * s;
        }
        lane[0] = from_f32<T>(y0);
        lane[1] = from_f32<T>(y1);
    }
}

// --- MoE routing -------------------------------------------------------------
// The router used to be the model's worst host stall: download [tokens, 256]
// logits, softplus/sqrt + bias + top-k on the CPU, rebuild the route table,
// upload it. Measured on this box that is ~1.25 ms of wall per decoder layer,
// 40% of decode time, because the eight TP ranks have to line up again at the
// next collective after it. The selection below is the host code transcribed,
// so the layer never round-trips.

// Same expression the host uses: softplus, clamped at +-20, then sqrt.
__device__ __forceinline__ float softplus_sqrt(float v) {
    const float sp = v > 20.0f ? v : (v < -20.0f ? expf(v) : log1pf(expf(v)));
    return sqrtf(fmaxf(sp, 0.0f));
}

// Far more experts per token than the released checkpoints use (topk is 6-8);
// only the block's index arithmetic depends on it, and it is checked on entry.
constexpr int kRouterMaxTopK = 16;
constexpr int kRouterThreads = 256;

// One block per token. The k-th chosen expert is the argmax of the biased
// score, with the lower expert index winning a tie - families of checkpoints
// have exact ties in the bias vector and the host's partial_sort then keeps the
// lower index too.
__global__ void router_topk_kernel(const float *__restrict__ logits,
                                   const float *__restrict__ bias,
                                   const long long *__restrict__ tid2eid,
                                   const long long *__restrict__ token_ids,
                                   int tokens,
                                   int n_experts,
                                   int topk,
                                   int hash_mode,
                                   int tp_size,
                                   int tp_rank,
                                   int *__restrict__ chosen_id,
                                   float *__restrict__ chosen_w,
                                   int *__restrict__ counts) {
    extern __shared__ float sh_scores[];
    float *raw = sh_scores + n_experts;
    const int t = static_cast<int>(blockIdx.x);
    if (t >= tokens) {
        return;
    }
    const float *row = logits + static_cast<long long>(t) * n_experts;
    for (int e = static_cast<int>(threadIdx.x); e < n_experts; e += static_cast<int>(blockDim.x)) {
        const float r = softplus_sqrt(row[e]);
        raw[e] = r;
        sh_scores[e] = r + (bias != nullptr ? bias[e] : 0.0f);
    }
    __syncthreads();

    if (hash_mode) {
        // Hash routing takes the expert set from the token id and the weights
        // from the same scores, so there is nothing to rank.
        if (static_cast<int>(threadIdx.x) < topk) {
            const long long tid = token_ids[t];
            const int id = static_cast<int>(tid2eid[tid * topk + threadIdx.x]);
            chosen_id[t * topk + threadIdx.x] = id;
            chosen_w[t * topk + threadIdx.x] = raw[id];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int k = 0; k < topk; ++k) {
                const int id = chosen_id[t * topk + k];
                if (id % tp_size == tp_rank) {
                    atomicAdd(counts + id / tp_size, 1);
                }
            }
        }
        return;
    }

    __shared__ float red_val[kRouterThreads];
    __shared__ int red_idx[kRouterThreads];
    for (int k = 0; k < topk; ++k) {
        float best = -CUDART_INF_F;
        int best_i = n_experts;
        for (int e = static_cast<int>(threadIdx.x); e < n_experts; e += static_cast<int>(blockDim.x)) {
            if (sh_scores[e] > best) {
                best = sh_scores[e];
                best_i = e;
            }
        }
        red_val[threadIdx.x] = best;
        red_idx[threadIdx.x] = best_i;
        __syncthreads();
        for (int stride = kRouterThreads / 2; stride > 0; stride >>= 1) {
            if (static_cast<int>(threadIdx.x) < stride) {
                const float other = red_val[threadIdx.x + stride];
                const int other_i = red_idx[threadIdx.x + stride];
                if (other > red_val[threadIdx.x]
                    || (other == red_val[threadIdx.x] && other_i < red_idx[threadIdx.x])) {
                    red_val[threadIdx.x] = other;
                    red_idx[threadIdx.x] = other_i;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const int id = red_idx[0];
            chosen_id[t * topk + k] = id;
            chosen_w[t * topk + k] = raw[id];
            sh_scores[id] = -CUDART_INF_F;
            if (id < n_experts && id % tp_size == tp_rank) {
                atomicAdd(counts + id / tp_size, 1);
            }
        }
        __syncthreads();
    }
}

// Turns the per-token choices into the layout the expert GEMM expects: rows
// grouped by local expert, and within an expert in (token, k) order, which is
// exactly what the host's stable_sort over its push order produced. Single
// thread: at most tokens*topk <= 96 routes and <= 32 local experts.
__global__ void router_emit_kernel(const int *__restrict__ chosen_id,
                                   const float *__restrict__ chosen_w,
                                   int tokens,
                                   int topk,
                                   int tp_size,
                                   int tp_rank,
                                   int renormalize,
                                   float routed_scale,
                                   int local_experts,
                                   int rows_cap,
                                   int *__restrict__ counts,
                                   int *__restrict__ row_token,
                                   float *__restrict__ row_weight,
                                   float *__restrict__ mix) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    // Exclusive prefix sum of the counts; each expert's cursor advances as its
    // routes are placed, which is what makes the order stable.
    int cursor[64];
    int off = 0;
    for (int e = 0; e < local_experts; ++e) {
        cursor[e] = off;
        off += counts[e];
    }
    const int total = off;
    for (int t = 0; t < tokens; ++t) {
        float denom = 0.0f;
        for (int k = 0; k < topk; ++k) {
            denom += chosen_w[t * topk + k];
        }
        for (int k = 0; k < topk; ++k) {
            const int e = chosen_id[t * topk + k];
            if (e % tp_size != tp_rank) {
                continue;
            }
            float w = chosen_w[t * topk + k];
            if (renormalize && denom != 0.0f) {
                w /= denom;
            }
            w *= routed_scale;
            const int slot = cursor[e / tp_size]++;
            row_token[slot] = t;
            row_weight[slot] = w;
            mix[t * rows_cap + slot] = w;
        }
    }
    // Rows past the live route count are never written by the GEMM; the gather
    // still reads their index, so point them at token 0 and zero their weight.
    for (int r = total; r < rows_cap; ++r) {
        row_token[r] = 0;
        row_weight[r] = 0.0f;
    }
}

// --- expert GEMM with device-side token counts -------------------------------
// Same kernel as w4a8_tn_warp_kernel in InfiniCore's Iluvatar w4a8_group_gemm,
// except that the per-expert row ranges come from a device tensor, which is what
// lets the router stay on the device. blockIdx.y selects the expert and the row
// offset is the prefix sum of the counts.
constexpr int kGroupWarps = 8;
constexpr int kGroupRowsPerPass = 8;

__device__ __forceinline__ int group_nibble(int byte, bool high) {
    const int nibble = high ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
    return (nibble ^ 8) - 8;
}

__device__ __forceinline__ int group_dot_byte(int packed_byte, int act_lo, int act_hi) {
    const int alo = static_cast<int>(static_cast<int8_t>(act_lo));
    const int ahi = static_cast<int>(static_cast<int8_t>(act_hi));
    return group_nibble(packed_byte, false) * alo + group_nibble(packed_byte, true) * ahi;
}

template <typename T>
__global__ void group_gemm_kernel(T *__restrict__ out,
                                  const int8_t *__restrict__ input,
                                  const int8_t *__restrict__ weight,
                                  const float *__restrict__ input_scale,
                                  const float *__restrict__ weight_scale,
                                  const int *__restrict__ counts,
                                  int experts,
                                  int n,
                                  int k,
                                  int k_packed) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int col = static_cast<int>(blockIdx.x) * kGroupWarps + warp;
    if (col >= n) {
        return;
    }
    const int words = k_packed / 4;
    // One block owns a fixed set of output columns and walks the experts, rather
    // than one block per (expert, column set): in decode only a couple of the 32
    // local experts have rows, so a per-expert grid launches thousands of blocks
    // that immediately return.
    int row0 = 0;
    for (int expert = 0; expert < experts; ++expert) {
        const int rows = counts[expert];
        if (rows <= 0) {
            continue;
        }
        const int8_t *expert_weight = weight + static_cast<long long>(expert) * n * k_packed;
        const uint32_t *packed = reinterpret_cast<const uint32_t *>(
            expert_weight + static_cast<long long>(col) * k_packed);
        const float w_scale = weight_scale[static_cast<long long>(expert) * n + col];

        for (int m0 = 0; m0 < rows; m0 += kGroupRowsPerPass) {
            const int mrows = min(kGroupRowsPerPass, rows - m0);
            int acc[kGroupRowsPerPass];
#pragma unroll
            for (int m = 0; m < kGroupRowsPerPass; ++m) {
                acc[m] = 0;
            }
            for (int wi = lane * 4; wi < words; wi += 128) {
                const uint32_t w[4] = {packed[wi], packed[wi + 1], packed[wi + 2], packed[wi + 3]};
                const int base_k = wi * 8;
                for (int m = 0; m < mrows; ++m) {
                    const int8_t *arow = input + static_cast<long long>(row0 + m0 + m) * k + base_k;
                    const uint4 alo = *reinterpret_cast<const uint4 *>(arow);
                    const uint4 ahi = *reinterpret_cast<const uint4 *>(arow + 16);
                    const uint32_t a[8] = {alo.x, alo.y, alo.z, alo.w, ahi.x, ahi.y, ahi.z, ahi.w};
                    int sum = acc[m];
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
#pragma unroll
                        for (int b = 0; b < 4; ++b) {
                            const int byte_index = q * 4 + b;
                            const int packed_byte = static_cast<int>((w[q] >> (b * 8)) & 0xFFu);
                            const int act_lo = static_cast<int>((a[byte_index >> 2] >> ((byte_index & 3) * 8)) & 0xFFu);
                            const int act_hi = static_cast<int>((a[4 + (byte_index >> 2)] >> ((byte_index & 3) * 8)) & 0xFFu);
                            sum += group_dot_byte(packed_byte, act_lo, act_hi);
                        }
                    }
                    acc[m] = sum;
                }
            }
#pragma unroll
            for (int m = 0; m < kGroupRowsPerPass; ++m) {
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    acc[m] += __shfl_down_sync(0xFFFFFFFFu, acc[m], off);
                }
                if (lane == 0 && m < mrows) {
                    const float scale = 18.0f * input_scale[row0 + m0 + m] * w_scale;
                    out[static_cast<long long>(row0 + m0 + m) * n + col] =
                        from_f32<T>(static_cast<float>(acc[m]) * scale);
                }
            }
        }
        row0 += rows;
    }
}

// One block per compressed row: the online softmax over the n_ctx state rows,
// the RMS scale and the partial RoPE. This is the loop in
// DeepseekV4Attention::forward that used to drain the whole slab of state rows
// to the host first.
constexpr int kEmitThreads = 256;

__device__ __forceinline__ void compress_emit_row(const float *__restrict__ state,
                                                 int pos,
                                                 int slab_lo,
                                                 int ratio,
                                                 int n_ctx,
                                                 int head,
                                                 int width,
                                                 int nope,
                                                 int rope_dim,
                                                 const float *__restrict__ norm_weight,
                                                 const float *__restrict__ inv_freq,
                                                 float eps,
                                                 float *__restrict__ row,
                                                 float *__restrict__ sh) {
    const int stride = width * 2;
    const int start = pos - n_ctx + 1;
    for (int d = threadIdx.x; d < head; d += kEmitThreads) {
        float running_max = -CUDART_INF_F;
        float running_sum = 0.0f;
        float acc = 0.0f;
        for (int step = 0; step < n_ctx; ++step) {
            const int pos_t = start + step;
            if (pos_t < 0) {
                continue;
            }
            const int head_off = step >= ratio ? head : 0;
            // `state` already points at the slab's first row.
            const float *base = state + static_cast<long long>(pos_t - slab_lo) * stride;
            const float kv_part = base[head_off + d];
            const float score_part = base[width + head_off + d];
            const float new_max = fmaxf(running_max, score_part);
            const float old_scale = expf(running_max - new_max);
            const float score_scale = expf(score_part - new_max);
            acc = acc * old_scale + kv_part * score_scale;
            running_sum = running_sum * old_scale + score_scale;
            running_max = new_max;
        }
        sh[d] = running_sum == 0.0f ? 0.0f : acc / running_sum;
    }
    __syncthreads();
    float variance = 0.0f;
    for (int d = threadIdx.x; d < head; d += kEmitThreads) {
        variance += sh[d] * sh[d];
    }
    __shared__ float red[kEmitThreads];
    red[threadIdx.x] = variance;
    __syncthreads();
    for (int off = kEmitThreads / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            red[threadIdx.x] += red[threadIdx.x + off];
        }
        __syncthreads();
    }
    const float rrms = 1.0f / sqrtf(red[0] / static_cast<float>(head) + eps);
    for (int d = threadIdx.x; d < head; d += kEmitThreads) {
        sh[d] *= rrms * norm_weight[d];
    }
    __syncthreads();
    const int compressed_pos = (pos / ratio) * ratio;
    const int half = rope_dim / 2;
    for (int d = threadIdx.x; d < nope; d += kEmitThreads) {
        row[d] = sh[d];
    }
    for (int pair = threadIdx.x; pair < half; pair += kEmitThreads) {
        const float ang = static_cast<float>(compressed_pos) * inv_freq[pair];
        const float cos_v = cosf(ang);
        const float sin_v = sinf(ang);
        const int even = nope + pair * 2;
        const float x0 = sh[even];
        const float x1 = sh[even + 1];
        row[even] = x0 * cos_v - x1 * sin_v;
        row[even + 1] = x0 * sin_v + x1 * cos_v;
    }
}

__global__ void compress_emit_kernel(const float *__restrict__ state,
                                     const int *__restrict__ emit_pos,
                                     int n_emit,
                                     int slab_lo,
                                     int ratio,
                                     int n_ctx,
                                     int head,
                                     int width,
                                     int nope,
                                     int rope_dim,
                                     const float *__restrict__ norm_weight,
                                     const float *__restrict__ inv_freq,
                                     float eps,
                                     float *__restrict__ out) {
    extern __shared__ float sh[];
    const int e = blockIdx.x;
    if (e >= n_emit) {
        return;
    }
    compress_emit_row(state,
                      emit_pos[e],
                      slab_lo,
                      ratio,
                      n_ctx,
                      head,
                      width,
                      nope,
                      rope_dim,
                      norm_weight,
                      inv_freq,
                      eps,
                      out + static_cast<long long>(e) * head,
                      sh);
}

// The last thing a forward does: mix the hc streams with the sigmoid weights
// the head GEMM produced. The host version drained mixes+sumsq, applied the
// sigmoid there and uploaded the weights again; here the only host work left is
// the (already device-side) GEMM that produces `mixes`.
constexpr int kHcHeadThreads = 256;

__global__ void hc_head_kernel(const float *__restrict__ mixes,
                               const float *__restrict__ sumsq,
                               const float *__restrict__ scale,
                               const float *__restrict__ base,
                               const float *__restrict__ residual,
                               float *__restrict__ out,
                               int tokens,
                               int hc,
                               int hidden,
                               float denom_count,
                               float rms_eps,
                               float hc_eps) {
    const int t = blockIdx.x;
    if (t >= tokens) {
        return;
    }
    const float scale0 = scale != nullptr ? scale[0] : 1.0f;
    const float rms = 1.0f / sqrtf(sumsq[t] / denom_count + rms_eps);
    float pre[kMaxHc];
    for (int c = 0; c < hc; ++c) {
        pre[c] = sigmoid_f(scale0 * mixes[t * hc + c] * rms + base[c]) + hc_eps;
    }
    const float *res = residual + static_cast<long long>(t) * hc * hidden;
    float *row = out + static_cast<long long>(t) * hidden;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        float acc = 0.0f;
        for (int c = 0; c < hc; ++c) {
            acc += pre[c] * res[static_cast<long long>(c) * hidden + d];
        }
        row[d] = acc;
    }
}

} // namespace

// Returns 0 on success and -1 for shapes this kernel does not handle, so the
// caller can fall back to the host implementation.
extern "C" int dsv4_mhc_pre_f32(const void *gemm,
                                const void *sumsq,
                                const void *scale,
                                const void *base,
                                void *pre,
                                void *post,
                                void *comb,
                                int tokens,
                                int hc,
                                int mix,
                                float inv_count,
                                float rms_eps,
                                float hc_eps,
                                int iters,
                                float post_alpha,
                                void *stream) {
    if (tokens <= 0 || hc <= 0 || hc > kMaxHc || mix != hc * (2 + hc)) {
        return -1;
    }
    mhc_pre_kernel<<<tokens, 32, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(gemm),
        static_cast<const float *>(sumsq),
        static_cast<const float *>(scale),
        static_cast<const float *>(base),
        static_cast<float *>(pre),
        static_cast<float *>(post),
        static_cast<float *>(comb),
        tokens,
        hc,
        mix,
        inv_count,
        rms_eps,
        hc_eps,
        iters,
        post_alpha);
    return 0;
}

// elem_width is sizeof(T): 4 for f32, 2 for bf16. Returns -1 for shapes this
// kernel does not handle.
extern "C" int dsv4_rope_(void *x,
                          const void *cos_sin,
                          const void *pos,
                          long long rows,
                          int heads,
                          int dim,
                          int nope,
                          int rope,
                          int inverse,
                          int elem_width,
                          void *stream) {
    if (rows <= 0 || heads <= 0 || rope <= 0 || (rope & 1) != 0 || nope + rope != dim) {
        return -1;
    }
    const long long total = rows * (rope / 2);
    const int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    if (blocks > 65535) {
        blocks = 65535;
    }
    auto s = reinterpret_cast<cudaStream_t>(stream);
    if (elem_width == 4) {
        rope_kernel<<<static_cast<int>(blocks), threads, 0, s>>>(
            static_cast<float *>(x), static_cast<const float *>(cos_sin),
            static_cast<const long long *>(pos), rows, heads, dim, nope, rope, inverse);
    } else if (elem_width == 2) {
        rope_kernel<<<static_cast<int>(blocks), threads, 0, s>>>(
            reinterpret_cast<__nv_bfloat16 *>(x), static_cast<const float *>(cos_sin),
            static_cast<const long long *>(pos), rows, heads, dim, nope, rope, inverse);
    } else {
        return -1;
    }
    return 0;
}

// Runs both router passes. `counts` must be zero-filled on the device and has
// one entry per local expert; `chosen_id`/`chosen_w` are scratch of size
// tokens*topk. Returns -1 for shapes this pair of kernels does not handle.
extern "C" int dsv4_moe_route_f32(const void *logits,
                                  const void *bias,
                                  const void *tid2eid,
                                  const void *token_ids,
                                  int tokens,
                                  int n_experts,
                                  int topk,
                                  int tp_size,
                                  int tp_rank,
                                  int hash_mode,
                                  int renormalize,
                                  float routed_scale,
                                  int rows_cap,
                                  int *chosen_id,
                                  float *chosen_w,
                                  int *counts,
                                  int *row_token,
                                  float *row_weight,
                                  float *mix,
                                  void *stream) {
    const int local_experts = tp_size > 0 ? n_experts / tp_size : 0;
    if (tokens <= 0 || n_experts <= 0 || topk <= 0 || topk > kRouterMaxTopK
        || tp_size <= 0 || n_experts % tp_size != 0 || local_experts > 64
        || rows_cap < tokens * topk || n_experts > 4096) {
        return -1;
    }
    if (hash_mode && (tid2eid == nullptr || token_ids == nullptr)) {
        return -1;
    }
    auto s = reinterpret_cast<cudaStream_t>(stream);
    const size_t shared = static_cast<size_t>(2 * n_experts) * sizeof(float);
    router_topk_kernel<<<tokens, kRouterThreads, shared, s>>>(
        static_cast<const float *>(logits),
        static_cast<const float *>(bias),
        static_cast<const long long *>(tid2eid),
        static_cast<const long long *>(token_ids),
        tokens,
        n_experts,
        topk,
        hash_mode,
        tp_size,
        tp_rank,
        chosen_id,
        chosen_w,
        counts);
    router_emit_kernel<<<1, 1, 0, s>>>(chosen_id,
                                       chosen_w,
                                       tokens,
                                       topk,
                                       tp_size,
                                       tp_rank,
                                       renormalize,
                                       routed_scale,
                                       local_experts,
                                       rows_cap,
                                       counts,
                                       row_token,
                                       row_weight,
                                       mix);
    return 0;
}

// out_bf16 selects the store type (1 = bf16, 0 = fp16). Rows past the sum of
// counts in `out` are left untouched, so the caller must zero them.
extern "C" int dsv4_group_gemm_counts(void *out,
                                      const void *input,
                                      const void *weight,
                                      const void *input_scale,
                                      const void *weight_scale,
                                      const void *counts,
                                      int experts,
                                      int rows_cap,
                                      int n,
                                      int k,
                                      int k_packed,
                                      int out_bf16,
                                      void *stream) {
    if (experts <= 0 || n <= 0 || k <= 0 || k_packed * 4 < k || rows_cap < 0) {
        return -1;
    }
    const int threads = kGroupWarps * 32;
    const dim3 grid(static_cast<unsigned>((n + kGroupWarps - 1) / kGroupWarps));
    auto s = reinterpret_cast<cudaStream_t>(stream);
    if (out_bf16 != 0) {
        group_gemm_kernel<__nv_bfloat16><<<grid, threads, 0, s>>>(
            static_cast<__nv_bfloat16 *>(out),
            static_cast<const int8_t *>(input),
            static_cast<const int8_t *>(weight),
            static_cast<const float *>(input_scale),
            static_cast<const float *>(weight_scale),
            static_cast<const int *>(counts),
            experts, n, k, k_packed);
    } else {
        group_gemm_kernel<__half><<<grid, threads, 0, s>>>(
            static_cast<__half *>(out),
            static_cast<const int8_t *>(input),
            static_cast<const int8_t *>(weight),
            static_cast<const float *>(input_scale),
            static_cast<const float *>(weight_scale),
            static_cast<const int *>(counts),
            experts, n, k, k_packed);
    }
    return 0;
}

// state points at the first row of the slab the emit positions live in; row
// indices inside the kernel are therefore (pos - slab_lo).
extern "C" int dsv4_compress_emit_f32(const void *state,
                                      const void *emit_pos,
                                      int n_emit,
                                      int slab_lo,
                                      int ratio,
                                      int n_ctx,
                                      int head,
                                      int width,
                                      int rope_dim,
                                      const void *norm_weight,
                                      const void *inv_freq,
                                      float eps,
                                      void *out,
                                      void *stream) {
    if (n_emit <= 0 || ratio <= 0 || n_ctx <= 0 || head <= 0 || head > 8192
        || width <= 0 || rope_dim <= 0 || rope_dim > head || (rope_dim & 1) != 0) {
        return -1;
    }
    const size_t shared = static_cast<size_t>(head) * sizeof(float);
    if (shared > 48 * 1024) {
        return -1;
    }
    compress_emit_kernel<<<n_emit, kEmitThreads, shared, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(state),
        static_cast<const int *>(emit_pos),
        n_emit,
        slab_lo,
        ratio,
        n_ctx,
        head,
        width,
        head - rope_dim,
        rope_dim,
        static_cast<const float *>(norm_weight),
        static_cast<const float *>(inv_freq),
        eps,
        static_cast<float *>(out));
    return 0;
}

extern "C" int dsv4_hc_head_f32(const void *mixes,
                                const void *sumsq,
                                const void *scale,
                                const void *base,
                                const void *residual,
                                void *out,
                                int tokens,
                                int hc,
                                int hidden,
                                float denom_count,
                                float rms_eps,
                                float hc_eps,
                                void *stream) {
    if (tokens <= 0 || hc <= 0 || hc > kMaxHc || hidden <= 0
        || base == nullptr || residual == nullptr || out == nullptr) {
        return -1;
    }
    hc_head_kernel<<<tokens, kHcHeadThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<const float *>(mixes),
        static_cast<const float *>(sumsq),
        static_cast<const float *>(scale),
        static_cast<const float *>(base),
        static_cast<const float *>(residual),
        static_cast<float *>(out),
        tokens,
        hc,
        hidden,
        denom_count,
        rms_eps,
        hc_eps);
    return 0;
}


// out[t, c, :] = sum_j comb[t, j, c] * residual[t, j, :] + post[t, c] * x[t, :]
// One block per token, hidden-indexed lanes. The batched matmul this replaces
// built three f32 copies of the residual per call (cast, cat, permute ->
// contiguous), which at prefill width moved hundreds of MB for a 5x5 product.
constexpr int kMhcPostThreads = 256;

template <typename T>
__global__ void mhc_post_kernel(const T *__restrict__ residual,
                                const T *__restrict__ x,
                                const float *__restrict__ post,
                                const float *__restrict__ comb,
                                T *__restrict__ out,
                                int hc,
                                int hidden) {
    const int t = static_cast<int>(blockIdx.x);
    const T *res = residual + static_cast<long long>(t) * hc * hidden;
    const T *xt = x + static_cast<long long>(t) * hidden;
    T *o = out + static_cast<long long>(t) * hc * hidden;
    const float *post_t = post + static_cast<long long>(t) * hc;
    const float *comb_t = comb + static_cast<long long>(t) * hc * hc;

    __shared__ float sh_post[kMaxHc];
    __shared__ float sh_comb[kMaxHc * kMaxHc];
    for (int i = static_cast<int>(threadIdx.x); i < hc; i += kMhcPostThreads) {
        sh_post[i] = post_t[i];
    }
    for (int i = static_cast<int>(threadIdx.x); i < hc * hc; i += kMhcPostThreads) {
        sh_comb[i] = comb_t[i];
    }
    __syncthreads();

    for (int d = static_cast<int>(threadIdx.x); d < hidden; d += kMhcPostThreads) {
        float r[kMaxHc];
        for (int j = 0; j < hc; ++j) {
            r[j] = to_f32<T>(res[static_cast<long long>(j) * hidden + d]);
        }
        const float xv = to_f32<T>(xt[d]);
        for (int c = 0; c < hc; ++c) {
            float acc = sh_post[c] * xv;
            for (int j = 0; j < hc; ++j) {
                acc += sh_comb[j * hc + c] * r[j];
            }
            o[static_cast<long long>(c) * hidden + d] = from_f32<T>(acc);
        }
    }
}

extern "C" int dsv4_mhc_post(const void *residual,
                             const void *x,
                             const void *post,
                             const void *comb,
                             void *out,
                             int tokens,
                             int hc,
                             int hidden,
                             int elem_width,
                             void *stream) {
    if (tokens <= 0 || hc <= 0 || hc > kMaxHc || hidden <= 0) {
        return -1;
    }
    auto s = reinterpret_cast<cudaStream_t>(stream);
    if (elem_width == 2) {
        mhc_post_kernel<__nv_bfloat16><<<tokens, kMhcPostThreads, 0, s>>>(
            static_cast<const __nv_bfloat16 *>(residual),
            static_cast<const __nv_bfloat16 *>(x),
            static_cast<const float *>(post),
            static_cast<const float *>(comb),
            static_cast<__nv_bfloat16 *>(out),
            hc,
            hidden);
    } else if (elem_width == 4) {
        mhc_post_kernel<float><<<tokens, kMhcPostThreads, 0, s>>>(
            static_cast<const float *>(residual),
            static_cast<const float *>(x),
            static_cast<const float *>(post),
            static_cast<const float *>(comb),
            static_cast<float *>(out),
            hc,
            hidden);
    } else {
        return -1;
    }
    return 0;
}



// Gathers each token's [sliding window | compressed prefix] rows into `packed`
// in one launch. Copies are byte moves, so this is dtype-agnostic: `words` is
// the row width in 4-byte units. meta holds five int arrays of `tokens` ints:
// starts, swa_at, swa_n, cmp_at, cmp_n.
constexpr int kPackThreads = 256;

__global__ void attn_pack_kernel(uint32_t *__restrict__ packed,
                                 const uint32_t *__restrict__ swa,
                                 const uint32_t *__restrict__ compressed,
                                 const int *__restrict__ meta,
                                 int tokens,
                                 int words,
                                 int has_compressed) {
    const int t = static_cast<int>(blockIdx.x);
    const int *starts = meta;
    const int *swa_at = meta + tokens;
    const int *swa_n = meta + 2 * tokens;
    const int *cmp_at = meta + 3 * tokens;
    const int *cmp_n = meta + 4 * tokens;
    int dst_row = starts[t];
    const int n_swa = swa_n[t];
    const int a_swa = swa_at[t];
    for (int r = 0; r < n_swa; ++r, ++dst_row) {
        const uint32_t *src = swa + static_cast<long long>(a_swa + r) * words;
        uint32_t *dst = packed + static_cast<long long>(dst_row) * words;
        for (int w = static_cast<int>(threadIdx.x); w < words; w += kPackThreads) {
            dst[w] = src[w];
        }
    }
    if (has_compressed == 0) {
        return;
    }
    const int n_cmp = cmp_n[t];
    const int a_cmp = cmp_at[t];
    for (int r = 0; r < n_cmp; ++r, ++dst_row) {
        const uint32_t *src = compressed + static_cast<long long>(a_cmp + r) * words;
        uint32_t *dst = packed + static_cast<long long>(dst_row) * words;
        for (int w = static_cast<int>(threadIdx.x); w < words; w += kPackThreads) {
            dst[w] = src[w];
        }
    }
}

extern "C" int dsv4_attn_pack(void *packed,
                               const void *swa,
                               const void *compressed,
                               const void *meta,
                               int tokens,
                               int words,
                               int has_compressed,
                               void *stream) {
    if (tokens <= 0 || words <= 0 || meta == nullptr) {
        return -1;
    }
    attn_pack_kernel<<<tokens, kPackThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<uint32_t *>(packed),
        static_cast<const uint32_t *>(swa),
        static_cast<const uint32_t *>(compressed),
        static_cast<const int *>(meta),
        tokens,
        words,
        has_compressed ? 1 : 0);
    return 0;
}

extern "C" const char *dsv4_kernels_abi_tag(void) {
    return "dsv4-kernels-5";
}
