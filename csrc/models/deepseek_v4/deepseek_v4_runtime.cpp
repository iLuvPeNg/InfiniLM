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

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace infinilm::models::deepseek_v4 {
namespace {

constexpr float kHcPostAlpha = 2.0f;
constexpr size_t kHostRetainLimit = 32;

struct HostBucketState {
    std::atomic<size_t> calls{0};
    std::atomic<long long> us{0};
};

HostBucketState g_host_buckets[static_cast<int>(HostBucket::Count)];
const char *const kHostBucketNames[static_cast<int>(HostBucket::Count)] = {
    "sync_device", "download_f32", "upload_f32", "mhc_pre",
    "mhc_post", "hc_head", "mhc_host_math", "mhc_token_loop", "layer",
};

bool host_timer_enabled() {
    static const bool enabled = std::getenv("INFINI_HOST_TIMER") != nullptr;
    return enabled;
}

void host_timer_dump() {
    if (!host_timer_enabled()) {
        return;
    }
    std::fprintf(stderr, "METRIC_HOST begin\n");
    for (int i = 0; i < static_cast<int>(HostBucket::Count); ++i) {
        const size_t n = g_host_buckets[i].calls.load();
        if (n == 0) {
            continue;
        }
        const long long us = g_host_buckets[i].us.load();
        std::fprintf(stderr,
                     "METRIC_HOST %-14s calls %zu total_ms %.1f avg_us %.1f\n",
                     kHostBucketNames[i],
                     n,
                     static_cast<double>(us) / 1000.0,
                     static_cast<double>(us) / static_cast<double>(n));
    }
}

struct HostTimerRegistrar {
    HostTimerRegistrar() {
        std::atexit(host_timer_dump);
    }
};
HostTimerRegistrar g_host_timer_registrar;

// --- GPU section timer -------------------------------------------------------
// Wall-clock attribution of a decoder layer without a profiler. Each section
// records a CUDA event pair on the current device; the elapsed times are read
// back at the end of every forward step, while the rank thread and its device
// are still alive, so the readback can never race model teardown.
constexpr size_t kGpuPoolCap = 64;

struct GpuPair {
    infinicore::Device device;
    infinirtEvent_t start;
    infinirtEvent_t end;
    // False between recordEvent(start) and recordEvent(end). A flush that runs
    // while another bucket's scope is still open must not read this pair back.
    bool done = true;
};

struct GpuPool {
    std::vector<GpuPair> pairs;
    size_t next = 0;
};

struct GpuThreadPools {
    GpuPool pools[static_cast<int>(GpuBucket::Count)];
};

thread_local GpuThreadPools t_gpu_pools;

struct GpuAccum {
    std::atomic<size_t> calls{0};
    std::atomic<long long> us{0};
};

GpuAccum g_gpu_accum[static_cast<int>(GpuBucket::Count)];

// Same totals, split by device, so a collective that only stalls because one
// rank arrived late shows up as an asymmetric distribution instead of an
// average that hides it.
constexpr int kMaxTimerDevice = 128;
GpuAccum g_gpu_dev_accum[kMaxTimerDevice][static_cast<int>(GpuBucket::Count)];
const char *const kGpuBucketNames[static_cast<int>(GpuBucket::Count)] = {
    "mhc_pre_attn", "attn_norm", "attention", "mhc_post_attn",
    "mhc_pre_ffn", "ffn_norm", "ffn", "mhc_post_ffn",
    "ffn_router", "ffn_shared", "ffn_quant", "ffn_gemm", "ffn_scatter",
    "ffn_allreduce", "attn_qkv", "attn_compress", "attn_pack", "attn_flash",
    "attn_proj", "attn_allreduce", "layer",
};

bool gpu_timer_enabled() {
    static const bool enabled = std::getenv("INFINI_GPU_TIMER") != nullptr;
    return enabled;
}

bool gpu_decode_only() {
    static const bool enabled = std::getenv("INFINI_GPU_DECODE_ONLY") != nullptr;
    return enabled;
}

thread_local bool t_gpu_decode = true;

void gpu_timer_dump() {
    if (!gpu_timer_enabled()) {
        return;
    }
    std::fprintf(stderr, "METRIC_GPU begin\n");
    for (int i = 0; i < static_cast<int>(GpuBucket::Count); ++i) {
        const size_t n = g_gpu_accum[i].calls.load();
        if (n == 0) {
            continue;
        }
        const long long us = g_gpu_accum[i].us.load();
        std::fprintf(stderr,
                     "METRIC_GPU %-14s calls %zu total_ms %.1f avg_us %.1f\n",
                     kGpuBucketNames[i],
                     n,
                     static_cast<double>(us) / 1000.0,
                     static_cast<double>(us) / static_cast<double>(n));
    }
    for (int i = 0; i < static_cast<int>(GpuBucket::Count); ++i) {
        bool any = false;
        char line[512] = {0};
        size_t off = 0;
        for (int d = 0; d < kMaxTimerDevice; ++d) {
            const size_t n = g_gpu_dev_accum[d][i].calls.load();
            if (n == 0) {
                continue;
            }
            any = true;
            const long long us = g_gpu_dev_accum[d][i].us.load();
            off += std::snprintf(line + off, sizeof(line) - off, " d%d=%.0f",
                                 d, static_cast<double>(us) / static_cast<double>(n));
            if (off >= sizeof(line) - 32) {
                break;
            }
        }
        if (any) {
            std::fprintf(stderr, "METRIC_GPU_BYDEV %-14s%s\n", kGpuBucketNames[i], line);
        }
    }
}

struct GpuTimerRegistrar {
    GpuTimerRegistrar() {
        std::atexit(gpu_timer_dump);
    }
};
GpuTimerRegistrar g_gpu_timer_registrar;

void sync_device() {
    HostScope scope(HostBucket::Sync);
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
    HostScope scope(HostBucket::Download);
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
    HostScope scope(HostBucket::Upload);
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

// infiniop's sum() takes its parallel path only when the reduction covers the
// whole input, so a [tokens, n] row reduction falls back to one thread per row
// running a serial n-iteration loop: ~22 ms for n = 16384 as soon as tokens > 1.
// The same row sums of squares come out of a GEMM against a column of ones in
// ~0.05 ms at that shape, which is what the hyper-connection RMS scale needs.
infinicore::Tensor row_sumsq(const infinicore::Tensor &x) {
    auto ones = infinicore::Tensor::ones({x->size(1), size_t{1}},
                                         infinicore::DataType::F32,
                                         x->device());
    return infinicore::op::matmul(infinicore::op::mul(x, x), ones);
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

namespace {

// libdsv4_kernels.so carries the fused device-side kernels for this model (see
// deepseek_v4_kernels.cu). It is optional: if it cannot be loaded, the host
// implementations are used instead. Set INFINI_MHC_HOST / INFINI_ROPE_HOST to
// force a host path for A/B comparison.
struct KernelLib {
    using PreFn = int (*)(const void *, const void *, const void *, const void *,
                          void *, void *, void *,
                          int, int, int, float, float, float, int, float, void *);
    using RopeFn = int (*)(void *, const void *, const void *,
                           long long, int, int, int, int, int, int, void *);
    using RouteFn = int (*)(const void *, const void *, const void *, const void *,
                            int, int, int, int, int, int, int, float, int,
                            int *, float *, int *, int *, float *, float *, void *);
    using GroupFn = int (*)(void *, const void *, const void *, const void *, const void *,
                            const void *, int, int, int, int, int, int, void *);
    using EmitFn = int (*)(const void *, const void *, int, int, int, int, int, int, int,
                           const void *, const void *, float, void *, void *);
    PreFn mhc_pre = nullptr;
    RopeFn rope = nullptr;
    RouteFn route = nullptr;
    GroupFn group_gemm = nullptr;
    EmitFn compress_emit = nullptr;
    using HcHeadFn = int (*)(const void *, const void *, const void *, const void *, const void *,
                             void *, int, int, int, float, float, float, void *);
    HcHeadFn hc_head = nullptr;
    using PostFn = int (*)(const void *, const void *, const void *, const void *, void *,
                           int, int, int, int, void *);
    PostFn mhc_post = nullptr;
    using PackFn = int (*)(void *, const void *, const void *, const void *, int, int, int, void *);
    PackFn attn_pack = nullptr;

    static const KernelLib &instance() {
        static const KernelLib lib;
        return lib;
    }

    KernelLib() {
        if (const char *env = std::getenv("INFINILM_KERNEL_LIB")) {
            load(env);
            return;
        }
        // Prefer the directory this shared object was loaded from, so deployment
        // only needs the files side by side.
        Dl_info info{};
        if (dladdr(reinterpret_cast<const void *>(&KernelLib::instance), &info) != 0 && info.dli_fname != nullptr) {
            std::string path(info.dli_fname);
            const auto slash = path.find_last_of('/');
            if (slash != std::string::npos && load(path.substr(0, slash + 1) + "libdsv4_kernels.so")) {
                return;
            }
        }
        load("libdsv4_kernels.so");
    }

    bool load(const std::string &path) {
        void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            return false;
        }
        mhc_pre = reinterpret_cast<PreFn>(dlsym(handle, "dsv4_mhc_pre_f32"));
        rope = reinterpret_cast<RopeFn>(dlsym(handle, "dsv4_rope_"));
        route = reinterpret_cast<RouteFn>(dlsym(handle, "dsv4_moe_route_f32"));
        group_gemm = reinterpret_cast<GroupFn>(dlsym(handle, "dsv4_group_gemm_counts"));
        compress_emit = reinterpret_cast<EmitFn>(dlsym(handle, "dsv4_compress_emit_f32"));
        hc_head = reinterpret_cast<HcHeadFn>(dlsym(handle, "dsv4_hc_head_f32"));
        mhc_post = reinterpret_cast<PostFn>(dlsym(handle, "dsv4_mhc_post"));
        attn_pack = reinterpret_cast<PackFn>(dlsym(handle, "dsv4_attn_pack"));
        return mhc_pre != nullptr || rope != nullptr;
    }
};


} // namespace

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
    const auto device = x->device();
    const size_t half = rope_dim / 2;
    const size_t rows = tokens * heads;
    auto pos = infinicore::Tensor::empty({tokens}, infinicore::DataType::I64, infinicore::Device::cpu());
    std::memcpy(pos->data(), positions.data(), tokens * sizeof(int64_t));
    auto pos_dev = pos->to(device);
    retain_host(pos);

    // Preferred path: copy the input and rotate the rope lanes in place with one
    // launch. The generic path below builds a dozen elementwise ops per call.
    // The kernel rounds each multiply and each add on its own, like the chain it
    // replaces, so the results agree everywhere except the odd lane that lands
    // within one f32 ULP of a bf16 rounding boundary; those show up as a 1-ULP
    // difference in bf16. INFINI_ROPE_HOST=1 forces the elementwise path.
    const auto &kernels = KernelLib::instance();
    const size_t elem_width = x->dtype() == infinicore::DataType::F32 ? 4
                             : x->dtype() == infinicore::DataType::BF16 ? 2
                                                                        : 0;
    static const bool rope_host_forced = std::getenv("INFINI_ROPE_HOST") != nullptr;
    if (kernels.rope != nullptr && !rope_host_forced && elem_width != 0) {
        auto out = infinicore::Tensor::empty(x->shape(), x->dtype(), device);
        out->copy_from(x->contiguous());
        const int rc = kernels.rope(out->data(), cos_sin->contiguous()->data(), pos_dev->data(),
                                    static_cast<long long>(rows), static_cast<int>(heads),
                                    static_cast<int>(dim), static_cast<int>(nope_dim),
                                    static_cast<int>(rope_dim), inverse ? 1 : 0,
                                    static_cast<int>(elem_width),
                                    infinicore::context::getStream());
        if (rc == 0) {
            return out;
        }
    }

    auto flat = three_d ? x->contiguous()->view({rows, dim}) : x->contiguous();
    auto xf = cast_dtype(flat, infinicore::DataType::F32);
    std::vector<int64_t> repeated;
    repeated.reserve(rows);
    for (int64_t pos : positions) {
        for (size_t h = 0; h < heads; ++h) {
            repeated.push_back(pos);
        }
    }
    auto pos_rows = infinicore::Tensor::empty({rows}, infinicore::DataType::I64, infinicore::Device::cpu());
    std::memcpy(pos_rows->data(), repeated.data(), rows * sizeof(int64_t));
    auto pos_rows_dev = pos_rows->to(x->device());
    retain_host(pos_rows);
    auto cs = infinicore::op::embedding(pos_rows_dev, cos_sin);
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

namespace {

// Per-layer mhc scale/base are parameters: they never change during a run, and
// mhc_pre only uses them in host math, so read them once instead of paying two
// device drains per call. Keyed by data pointer, which is stable for the model's
// lifetime.
const std::vector<float> &cached_param_cpu(const infinicore::Tensor &t) {
    static std::mutex mutex;
    static std::unordered_map<const void *, std::vector<float>> cache;
    const void *key = t->data();
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it == cache.end()) {
        it = cache.emplace(key, download_f32(t)).first;
    }
    return it->second;
}

// The permuted float copy of mhc `fn` is also a constant: building it costs a
// cast plus a contiguous copy over hc*(2+hc)*hidden elements on every call.
const infinicore::Tensor &cached_fn_t(const infinicore::Tensor &fn,
                                      bool broadcast,
                                      size_t mix,
                                      size_t hc,
                                      size_t hidden) {
    static std::mutex mutex;
    static std::unordered_map<const void *, infinicore::Tensor> cache;
    const void *key = fn->data();
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it == cache.end()) {
        infinicore::Tensor used;
        if (broadcast) {
            used = infinicore::op::sum(fn->contiguous()->view({mix, hc, hidden}), {1}, false);
        } else {
            used = fn;
        }
        it = cache.emplace(key,
                           cast_dtype(used, infinicore::DataType::F32)->permute({1, 0})->contiguous())
                 .first;
    }
    return it->second;
}

} // namespace

MhcOut mhc_pre(const infinicore::Tensor &residual,
               const infinicore::Tensor &fn,
               const infinicore::Tensor &scale,
               const infinicore::Tensor &base,
               int hc_mult,
               double rms_eps,
               double hc_eps,
               int sinkhorn_iters) {
    HostScope scope(HostBucket::MhcPre);
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
    size_t rms_mult;
    if (broadcast) {
        flat32 = cast_dtype(residual->contiguous(), infinicore::DataType::F32);
        rms_mult = 1;
    } else {
        flat32 = cast_dtype(residual->contiguous()->view({tokens, hc * hidden}), infinicore::DataType::F32);
        rms_mult = hc;
    }
    auto gemm = infinicore::op::matmul(flat32, cached_fn_t(fn, broadcast, mix, hc, hidden));

    auto residual_out = broadcast ? infinicore::Tensor::empty({tokens, hc, hidden}, dtype, device) : residual->contiguous();
    if (broadcast) {
        auto src = residual->contiguous();
        for (size_t c = 0; c < hc; ++c) {
            residual_out->narrow({{1, c, 1}})->view({tokens, hidden})->copy_from(src);
        }
    }

    const float eps = static_cast<float>(hc_eps);
    const float rms_eps_f = static_cast<float>(rms_eps);
    const float denom_count = static_cast<float>(rms_mult * hidden);

    infinicore::Tensor pre_all;
    infinicore::Tensor post_all;
    infinicore::Tensor comb_all;

    // Preferred path: everything (rms, sigmoid, sinkhorn) runs on the device in
    // one launch off the GEMM output. The host path below drains the device
    // three times per call, twice per layer.
    bool device_math = false;
    const auto &kernels = KernelLib::instance();
    static const bool host_mhc_forced = std::getenv("INFINI_MHC_HOST") != nullptr;
    if (kernels.mhc_pre != nullptr && !host_mhc_forced && hc <= 16 &&
        scale->dtype() == infinicore::DataType::F32 && base->dtype() == infinicore::DataType::F32) {
        auto sumsq = row_sumsq(flat32)->contiguous();
        auto gemm_c = gemm->contiguous();
        auto scale_c = scale->contiguous();
        auto base_c = base->contiguous();
        pre_all = infinicore::Tensor::empty({tokens, hc}, infinicore::DataType::F32, device);
        post_all = infinicore::Tensor::empty({tokens, hc}, infinicore::DataType::F32, device);
        comb_all = infinicore::Tensor::empty({tokens, hc, hc}, infinicore::DataType::F32, device);
        const int rc = kernels.mhc_pre(gemm_c->data(), sumsq->data(), scale_c->data(), base_c->data(),
                                   pre_all->data(), post_all->data(), comb_all->data(),
                                   static_cast<int>(tokens), static_cast<int>(hc), static_cast<int>(mix),
                                   1.0f / denom_count, rms_eps_f, eps,
                                   sinkhorn_iters, kHcPostAlpha,
                                   infinicore::context::getStream());
        device_math = rc == 0;
    }

    if (!device_math) {
        auto sumsq = row_sumsq(flat32);
        // One download for both reductions instead of two: every download is a full
        // device drain, and this runs twice per layer.
        auto gemm_and_sumsq_cpu = download_f32(infinicore::op::cat({gemm, sumsq}, 1));
        const auto &scale_cpu = cached_param_cpu(scale);
        const auto &base_cpu = cached_param_cpu(base);
        if (gemm_and_sumsq_cpu.size() != tokens * (mix + 1) || scale_cpu.size() < 3 || base_cpu.size() < mix) {
            throw std::runtime_error("deepseek_v4 mhc: mix tensor size mismatch");
        }
        const size_t gemm_stride = mix + 1;
        const auto gemm_at = [&](size_t t, size_t c) { return gemm_and_sumsq_cpu[t * gemm_stride + c]; };

        std::vector<float> pre_cpu(tokens * hc);
        std::vector<float> post_cpu(tokens * hc);
        std::vector<float> comb_cpu(tokens * hc * hc);
        {
            HostScope host_math(HostBucket::MhcHostMath);
            for (size_t t = 0; t < tokens; ++t) {
                const float rms = 1.0f / std::sqrt(gemm_at(t, mix) / denom_count + rms_eps_f);
                for (size_t c = 0; c < hc; ++c) {
                    pre_cpu[t * hc + c] = sigmoid_cpu(scale_cpu[0] * gemm_at(t, c) * rms + base_cpu[c]) + eps;
                    post_cpu[t * hc + c] = sigmoid_cpu(scale_cpu[1] * gemm_at(t, hc + c) * rms + base_cpu[hc + c]) * kHcPostAlpha;
                }
                std::vector<float> comb(hc * hc);
                for (size_t i = 0; i < hc * hc; ++i) {
                    comb[i] = scale_cpu[2] * gemm_at(t, 2 * hc + i) * rms + base_cpu[2 * hc + i];
                }
                sinkhorn_cpu(comb, static_cast<int>(hc), sinkhorn_iters, eps);
                std::copy(comb.begin(), comb.end(), comb_cpu.begin() + static_cast<std::ptrdiff_t>(t * hc * hc));
            }
        }
        pre_all = upload_f32(pre_cpu, {tokens, hc}, device);
        post_all = upload_f32(post_cpu, {tokens, hc}, device);
        comb_all = upload_f32(comb_cpu, {tokens, hc, hc}, device);
    }

    // One batched matmul instead of a per-token loop of narrow/copy launches:
    // layer_input[t] = sum_c pre[t, c] * residual_out[t, c, :].
    infinicore::Tensor layer_input;
    {
        HostScope token_loop(HostBucket::MhcTokenLoop);
        auto mixed = infinicore::op::matmul(pre_all->view({tokens, size_t{1}, hc}),
                                            cast_dtype(residual_out->contiguous(), infinicore::DataType::F32));
        layer_input = cast_dtype(mixed->view({tokens, hidden}), dtype);
    }

    MhcOut out;
    out.residual = residual_out;
    out.post_mix = post_all;
    out.comb = comb_all;
    out.layer_input = layer_input;
    return out;
}


bool device_attn_pack(const infinicore::Tensor &packed,
                      const infinicore::Tensor &swa_cache,
                      const infinicore::Tensor *compressed_cache,
                      const std::vector<size_t> &starts,
                      const std::vector<size_t> &swa_at,
                      const std::vector<size_t> &swa_n,
                      const std::vector<size_t> &cmp_at,
                      const std::vector<size_t> &cmp_n) {
    const auto &kernels = KernelLib::instance();
    if (kernels.attn_pack == nullptr || packed->ndim() != 2 || swa_cache->ndim() != 2) {
        return false;
    }
    const size_t tokens = starts.size();
    if (tokens == 0 || tokens > 65535 || swa_at.size() != tokens || swa_n.size() != tokens
        || cmp_at.size() != tokens || cmp_n.size() != tokens) {
        return false;
    }
    const size_t head_dim = packed->size(1);
    if (swa_cache->size(1) != head_dim || packed->dtype() != swa_cache->dtype()) {
        return false;
    }
    size_t elem_bytes = 0;
    if (packed->dtype() == infinicore::DataType::BF16 || packed->dtype() == infinicore::DataType::F16) {
        elem_bytes = 2;
    } else if (packed->dtype() == infinicore::DataType::F32) {
        elem_bytes = 4;
    } else {
        return false;
    }
    const size_t row_bytes = head_dim * elem_bytes;
    if (row_bytes % 4 != 0) {
        return false;
    }
    bool has_compressed = false;
    if (compressed_cache != nullptr && (*compressed_cache)->numel() > 0) {
        if ((*compressed_cache)->ndim() != 2 || (*compressed_cache)->size(1) != head_dim) {
            return false;
        }
        has_compressed = true;
    }
    auto packed_c = packed->contiguous();
    auto swa_c = swa_cache->contiguous();
    infinicore::Tensor cmp_c;
    if (has_compressed) {
        cmp_c = (*compressed_cache)->contiguous();
    }
    std::vector<int32_t> meta(5 * tokens, 0);
    for (size_t t = 0; t < tokens; ++t) {
        meta[t] = static_cast<int32_t>(starts[t]);
        meta[tokens + t] = static_cast<int32_t>(swa_at[t]);
        meta[2 * tokens + t] = static_cast<int32_t>(swa_n[t]);
        meta[3 * tokens + t] = static_cast<int32_t>(cmp_at[t]);
        meta[4 * tokens + t] = static_cast<int32_t>(cmp_n[t]);
    }
    auto meta_dev = upload_i32(meta, {5 * tokens}, packed->device());
    const int rc = kernels.attn_pack(packed_c->data(),
                                     swa_c->data(),
                                     has_compressed ? cmp_c->data() : nullptr,
                                     meta_dev->data(),
                                     static_cast<int>(tokens),
                                     static_cast<int>(row_bytes / 4),
                                     has_compressed ? 1 : 0,
                                     infinicore::context::getStream());
    return rc == 0;
}

bool device_mhc_post(const infinicore::Tensor &x,
                     const infinicore::Tensor &residual,
                     const infinicore::Tensor &post_mix,
                     const infinicore::Tensor &comb,
                     infinicore::Tensor &out) {
    const auto &kernels = KernelLib::instance();
    if (kernels.mhc_post == nullptr || residual->ndim() != 3 || x->ndim() != 2) {
        return false;
    }
    const size_t tokens = residual->size(0);
    const size_t hc = residual->size(1);
    const size_t hidden = residual->size(2);
    if (tokens == 0 || hc == 0 || hc > 16 || hidden == 0 || x->size(0) != tokens || x->size(1) != hidden
        || post_mix->numel() != tokens * hc || comb->numel() != tokens * hc * hc
        || post_mix->dtype() != infinicore::DataType::F32 || comb->dtype() != infinicore::DataType::F32
        || x->dtype() != residual->dtype()) {
        return false;
    }
    int elem_width = 0;
    if (residual->dtype() == infinicore::DataType::BF16) {
        elem_width = 2;
    } else if (residual->dtype() == infinicore::DataType::F32) {
        elem_width = 4;
    } else {
        return false;
    }
    auto res_c = residual->contiguous();
    auto x_c = x->contiguous();
    auto post_c = post_mix->contiguous();
    auto comb_c = comb->contiguous();
    auto mixed = infinicore::Tensor::empty({tokens, hc, hidden}, residual->dtype(), residual->device());
    const int rc = kernels.mhc_post(res_c->data(), x_c->data(), post_c->data(), comb_c->data(), mixed->data(),
                                    static_cast<int>(tokens), static_cast<int>(hc), static_cast<int>(hidden),
                                    elem_width, infinicore::context::getStream());
    if (rc != 0) {
        return false;
    }
    out = mixed;
    return true;
}

infinicore::Tensor mhc_post(const infinicore::Tensor &x,
                            const infinicore::Tensor &residual,
                            const infinicore::Tensor &post_mix,
                            const infinicore::Tensor &comb) {
    HostScope scope(HostBucket::MhcPost);
    static const bool post_host_forced = std::getenv("INFINI_MHC_POST_HOST") != nullptr;
    if (!post_host_forced) {
        infinicore::Tensor mixed;
        if (device_mhc_post(x, residual, post_mix, comb, mixed)) {
            return mixed;
        }
    }
    const size_t tokens = residual->size(0);
    const size_t hc = residual->size(1);
    const size_t hidden = residual->size(2);
    const auto dtype = residual->dtype();
    // out[t, c, :] = sum_j comb[t, j, c] * residual[t, j, :] + post_mix[t, c] * x[t, :]
    // The x term is folded into the same batched matmul by appending x as one
    // extra residual row and post_mix as the matching row of the comb matrix.
    // Everything stays on the device: this used to be a host round trip plus a
    // per-token, per-slot chain of narrow/add/copy launches.
    auto res_aug = infinicore::op::cat({cast_dtype(residual->contiguous(), infinicore::DataType::F32),
                                        cast_dtype(x->contiguous()->view({tokens, size_t{1}, hidden}), infinicore::DataType::F32)},
                                       1);
    auto comb_aug = infinicore::op::cat({cast_dtype(comb->contiguous(), infinicore::DataType::F32),
                                         cast_dtype(post_mix->contiguous(), infinicore::DataType::F32)->view({tokens, size_t{1}, hc})},
                                        1);
    auto mixed = infinicore::op::matmul(comb_aug->permute({0, 2, 1})->contiguous(), res_aug);
    return cast_dtype(mixed->view({tokens, hc, hidden}), dtype);
}

bool device_hc_head(const infinicore::Tensor &mixes,
                    const infinicore::Tensor &sumsq,
                    const infinicore::Tensor *scale,
                    const infinicore::Tensor &base,
                    const infinicore::Tensor &residual_f32,
                    size_t hc,
                    size_t hidden,
                    float denom_count,
                    float rms_eps,
                    float hc_eps,
                    infinicore::Tensor &out);

infinicore::Tensor hc_head(const infinicore::Tensor &residual,
                           const infinicore::Tensor &fn,
                           const infinicore::Tensor &scale,
                           const infinicore::Tensor &base,
                           double rms_eps,
                           double hc_eps) {
    HostScope scope(HostBucket::HcHead);
    const size_t tokens = residual->size(0);
    const size_t hc = residual->size(1);
    const size_t hidden = residual->size(2);
    const auto device = residual->device();
    const auto dtype = residual->dtype();
    auto flat32 = cast_dtype(residual->contiguous()->view({tokens, hc * hidden}), infinicore::DataType::F32);
    auto fn_t = cast_dtype(fn, infinicore::DataType::F32)->permute({1, 0})->contiguous();
    auto mixes = infinicore::op::matmul(flat32, fn_t);
    auto sumsq = row_sumsq(flat32);
    const float denom_count = static_cast<float>(hc * hidden);
    // mixes and sumsq are already on the device, so the mixing itself needs no
    // round trip either. Off by default: this drain sits at the end of the step,
    // where the queue is already empty, so the host path measured no slower.
    static const bool hc_head_device = std::getenv("INFINI_HC_HEAD_DEVICE") != nullptr;
    if (hc_head_device) {
        infinicore::Tensor mixed;
        if (device_hc_head(mixes,
                           sumsq,
                           &scale,
                           base,
                           flat32,
                           hc,
                           hidden,
                           denom_count,
                           static_cast<float>(rms_eps),
                           static_cast<float>(hc_eps),
                           mixed)) {
            return cast_dtype(mixed, dtype);
        }
    }
    // One drain instead of two: sumsq rides along as an extra column.
    const size_t packed_stride = hc + 1;
    const auto packed = download_f32(infinicore::op::cat({mixes, sumsq}, 1));
    // scale/base are parameters like the mhc ones: one download for the whole
    // run instead of two per call.
    const auto &scale_cpu = cached_param_cpu(scale);
    const auto &base_cpu = cached_param_cpu(base);
    const float eps = static_cast<float>(hc_eps);
    const float rms_eps_f = static_cast<float>(rms_eps);
    const float scale0 = scale_cpu.empty() ? 1.0f : scale_cpu[0];
    std::vector<float> pre_cpu(tokens * hc);
    for (size_t t = 0; t < tokens; ++t) {
        const float rms = 1.0f / std::sqrt(packed[t * packed_stride + hc] / denom_count + rms_eps_f);
        for (size_t c = 0; c < hc; ++c) {
            pre_cpu[t * hc + c] = sigmoid_cpu(scale0 * packed[t * packed_stride + c] * rms + base_cpu[c]) + eps;
        }
    }
    auto pre_all = upload_f32(pre_cpu, {tokens, hc}, device);
    // out[t] = sum_c pre[t, c] * residual[t, c, :], one batched matmul rather than
    // a per-token loop of narrow/copy launches.
    auto mixed = infinicore::op::matmul(pre_all->view({tokens, size_t{1}, hc}),
                                        cast_dtype(residual->contiguous(), infinicore::DataType::F32));
    return cast_dtype(mixed->view({tokens, hidden}), dtype);
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
                      DeviceRoute &out) {
    const auto &kernels = KernelLib::instance();
    if (kernels.route == nullptr) {
        return false;
    }
    const auto logits_c = logits->contiguous();
    if (logits_c->ndim() != 2 || logits_c->dtype() != infinicore::DataType::F32) {
        return false;
    }
    const size_t tokens = logits_c->size(0);
    const size_t n_experts = logits_c->size(1);
    if (tokens == 0 || tp_size == 0 || n_experts % tp_size != 0 || rows_cap < tokens * topk) {
        return false;
    }
    const auto device = logits_c->device();
    const auto i32 = infinicore::DataType::I32;
    const auto f32 = infinicore::DataType::F32;

    infinicore::Tensor bias_c;
    if (bias != nullptr) {
        bias_c = (*bias)->contiguous();
        if (bias_c->dtype() != f32 || bias_c->numel() != n_experts) {
            return false;
        }
    }
    // Hash routing reads the expert set from tid2eid[token_id], so the token ids
    // have to travel to the device with the rest of the routing inputs.
    infinicore::Tensor tid_c;
    infinicore::Tensor token_dev;
    if (hash) {
        if (tid2eid == nullptr || (*tid2eid)->dtype() != infinicore::DataType::I64
            || token_ids.size() < tokens) {
            return false;
        }
        tid_c = (*tid2eid)->contiguous();
        auto cpu_ids = infinicore::Tensor::empty({tokens}, infinicore::DataType::I64,
                                                 infinicore::Device::cpu());
        std::memcpy(cpu_ids->data(), token_ids.data(), tokens * sizeof(int64_t));
        token_dev = cpu_ids->to(device);
    }

    const size_t local_experts = n_experts / tp_size;
    out.counts = infinicore::Tensor::zeros({local_experts}, i32, device);
    out.row_token = infinicore::Tensor::empty({rows_cap}, i32, device);
    out.row_weight = infinicore::Tensor::empty({rows_cap}, f32, device);
    out.mix = infinicore::Tensor::zeros({tokens, rows_cap}, f32, device);
    auto chosen_id = infinicore::Tensor::empty({tokens * topk}, i32, device);
    auto chosen_w = infinicore::Tensor::empty({tokens * topk}, f32, device);

    const int rc = kernels.route(
        logits_c->data(),
        bias_c ? bias_c->data() : nullptr,
        tid_c ? tid_c->data() : nullptr,
        token_dev ? token_dev->data() : nullptr,
        static_cast<int>(tokens),
        static_cast<int>(n_experts),
        static_cast<int>(topk),
        static_cast<int>(tp_size),
        static_cast<int>(tp_rank),
        hash ? 1 : 0,
        renormalize ? 1 : 0,
        routed_scale,
        static_cast<int>(rows_cap),
        reinterpret_cast<int *>(chosen_id->data()),
        reinterpret_cast<float *>(chosen_w->data()),
        reinterpret_cast<int *>(out.counts->data()),
        reinterpret_cast<int *>(out.row_token->data()),
        reinterpret_cast<float *>(out.row_weight->data()),
        reinterpret_cast<float *>(out.mix->data()),
        infinicore::context::getStream());
    return rc == 0;
}

bool device_group_gemm(const infinicore::Tensor &out,
                       const infinicore::Tensor &input_i8,
                       const infinicore::Tensor &weight,
                       const infinicore::Tensor &input_scale,
                       const infinicore::Tensor &weight_scale,
                       const infinicore::Tensor &counts,
                       size_t rows_cap,
                       size_t experts) {
    const auto &kernels = KernelLib::instance();
    if (kernels.group_gemm == nullptr) {
        return false;
    }
    auto o = out->contiguous();
    auto in = input_i8->contiguous();
    auto w = weight->contiguous();
    auto in_s = input_scale->contiguous();
    auto w_s = weight_scale->contiguous();
    auto c = counts->contiguous();
    const bool bf16 = o->dtype() == infinicore::DataType::BF16;
    if ((!bf16 && o->dtype() != infinicore::DataType::F16) || o->ndim() != 2 || w->ndim() != 3
        || in->dtype() != infinicore::DataType::I8 || c->dtype() != infinicore::DataType::I32
        || in->size(0) != rows_cap || o->size(0) != rows_cap || w->size(0) != experts) {
        return false;
    }
    const int rc = kernels.group_gemm(
        o->data(),
        in->data(),
        w->data(),
        in_s->data(),
        w_s->data(),
        c->data(),
        static_cast<int>(experts),
        static_cast<int>(rows_cap),
        static_cast<int>(o->size(1)),
        static_cast<int>(in->size(1)),
        static_cast<int>(w->size(2)),
        bf16 ? 1 : 0,
        infinicore::context::getStream());
    return rc == 0;
}

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
                          infinicore::Tensor &out) {
    const auto &kernels = KernelLib::instance();
    if (kernels.compress_emit == nullptr) {
        return false;
    }
    auto state = state_cache->contiguous();
    auto norm = norm_weight->contiguous();
    auto freq = inv_freq->contiguous();
    if (state->dtype() != infinicore::DataType::F32 || state->ndim() != 2
        || state->size(1) != width * 2 || norm->numel() != head
        || freq->numel() != rope_dim / 2 || head <= rope_dim || emit_positions.empty()) {
        return false;
    }
    const size_t rows = state->size(0);
    for (int64_t pos : emit_positions) {
        if (pos < static_cast<int64_t>(slab_lo) || static_cast<size_t>(pos) >= rows) {
            return false;
        }
    }
    const auto device = state->device();
    auto cpu_pos = infinicore::Tensor::empty({emit_positions.size()}, infinicore::DataType::I32,
                                             infinicore::Device::cpu());
    auto *pos_out = reinterpret_cast<int32_t *>(cpu_pos->data());
    for (size_t i = 0; i < emit_positions.size(); ++i) {
        pos_out[i] = static_cast<int32_t>(emit_positions[i]);
    }
    auto pos_dev = cpu_pos->to(device);
    out = infinicore::Tensor::empty({emit_positions.size(), head}, infinicore::DataType::F32, device);

    auto *state_base = reinterpret_cast<const float *>(state->data()) + slab_lo * width * 2;
    const int rc = kernels.compress_emit(state_base,
                                         pos_dev->data(),
                                         static_cast<int>(emit_positions.size()),
                                         static_cast<int>(slab_lo),
                                         static_cast<int>(ratio),
                                         static_cast<int>(n_ctx),
                                         static_cast<int>(head),
                                         static_cast<int>(width),
                                         static_cast<int>(rope_dim),
                                         norm->data(),
                                         freq->data(),
                                         eps,
                                         out->data(),
                                         infinicore::context::getStream());
    return rc == 0;
}

bool device_hc_head(const infinicore::Tensor &mixes,
                    const infinicore::Tensor &sumsq,
                    const infinicore::Tensor *scale,
                    const infinicore::Tensor &base,
                    const infinicore::Tensor &residual_f32,
                    size_t hc,
                    size_t hidden,
                    float denom_count,
                    float rms_eps,
                    float hc_eps,
                    infinicore::Tensor &out) {
    const auto &kernels = KernelLib::instance();
    if (kernels.hc_head == nullptr) {
        return false;
    }
    const auto f32 = infinicore::DataType::F32;
    auto m = mixes->contiguous();
    auto s = sumsq->contiguous();
    auto b = base->contiguous();
    auto r = residual_f32->contiguous();
    if (m->dtype() != f32 || m->ndim() != 2 || m->size(1) != hc || b->dtype() != f32
        || b->numel() != hc || s->dtype() != f32 || s->numel() != m->size(0)
        || r->dtype() != f32 || r->numel() != m->size(0) * hc * hidden) {
        return false;
    }
    infinicore::Tensor scale_c;
    if (scale != nullptr) {
        scale_c = (*scale)->contiguous();
        if (scale_c->dtype() != f32 || scale_c->numel() > 1) {
            return false;
        }
        if (scale_c->numel() == 0) {
            scale_c = infinicore::Tensor{};
        }
    }
    const size_t tokens = m->size(0);
    out = infinicore::Tensor::empty({tokens, hidden}, f32, m->device());
    const int rc = kernels.hc_head(m->data(),
                                   s->data(),
                                   scale_c ? scale_c->data() : nullptr,
                                   b->data(),
                                   r->data(),
                                   out->data(),
                                   static_cast<int>(tokens),
                                   static_cast<int>(hc),
                                   static_cast<int>(hidden),
                                   denom_count,
                                   rms_eps,
                                   hc_eps,
                                   infinicore::context::getStream());
    return rc == 0;
}

HostScope::HostScope(HostBucket bucket)
    : bucket_(bucket),
      active_(host_timer_enabled() && !(gpu_decode_only() && !t_gpu_decode)),
      start_(std::chrono::steady_clock::now()) {}

HostScope::~HostScope() {
    if (!active_) {
        return;
    }
    auto &bucket = g_host_buckets[static_cast<int>(bucket_)];
    bucket.calls.fetch_add(1, std::memory_order_relaxed);
    bucket.us.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - start_)
                            .count(),
                        std::memory_order_relaxed);
}

GpuScope::GpuScope(GpuBucket bucket)
    : bucket_(bucket), slot_(0), active_(false) {
    if (!gpu_timer_enabled()) {
        return;
    }
    if (gpu_decode_only() && !t_gpu_decode) {
        return;
    }
    const auto device = infinicore::context::getDevice();
    if (device.getType() == infinicore::Device::Type::CPU) {
        return;
    }
    auto &pool = t_gpu_pools.pools[static_cast<int>(bucket_)];
    if (pool.next == pool.pairs.size()) {
        if (pool.pairs.size() < kGpuPoolCap) {
            pool.pairs.push_back({device,
                                  infinicore::context::createEvent(),
                                  infinicore::context::createEvent()});
        } else {
            gpu_timer_flush_thread();
        }
    }
    slot_ = pool.next++;
    pool.pairs[slot_].done = false;
    infinicore::context::recordEvent(pool.pairs[slot_].start);
    active_ = true;
}

GpuScope::~GpuScope() {
    if (!active_) {
        return;
    }
    auto &pool = t_gpu_pools.pools[static_cast<int>(bucket_)];
    infinicore::context::recordEvent(pool.pairs[slot_].end);
    pool.pairs[slot_].done = true;
}

void gpu_timer_set_decode(bool decode) {
    t_gpu_decode = decode;
}

void gpu_timer_flush_thread() {
    if (!gpu_timer_enabled()) {
        return;
    }
    const auto saved = infinicore::context::getDevice();
    for (int b = 0; b < static_cast<int>(GpuBucket::Count); ++b) {
        auto &pool = t_gpu_pools.pools[b];
        if (pool.next == 0) {
            continue;
        }
        // A flush can be triggered by a different bucket (an outer section that
        // is still running); leave this pool alone until its own section closes.
        bool any_open = false;
        for (size_t i = 0; i < pool.next; ++i) {
            if (!pool.pairs[i].done) {
                any_open = true;
                break;
            }
        }
        if (any_open) {
            continue;
        }
        // One rank thread drives one device, so a single device sync covers
        // every event in this pool.
        infinicore::context::setDevice(pool.pairs[0].device);
        infinicore::context::syncDevice();
        long long us = 0;
        for (size_t i = 0; i < pool.next; ++i) {
            us += static_cast<long long>(
                infinicore::context::elapsedTime(pool.pairs[i].start, pool.pairs[i].end) * 1000.0f);
        }
        g_gpu_accum[b].calls.fetch_add(pool.next, std::memory_order_relaxed);
        g_gpu_accum[b].us.fetch_add(us, std::memory_order_relaxed);
        const auto dev_index = pool.pairs[0].device.getIndex();
        if (dev_index < static_cast<size_t>(kMaxTimerDevice)) {
            g_gpu_dev_accum[dev_index][b].calls.fetch_add(pool.next, std::memory_order_relaxed);
            g_gpu_dev_accum[dev_index][b].us.fetch_add(us, std::memory_order_relaxed);
        }
        pool.next = 0;
    }
    infinicore::context::setDevice(saved);
}

} // namespace infinilm::models::deepseek_v4
