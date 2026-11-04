#include "none_quantization.hpp"
#include "../../global_state/global_state.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/linear_allreduce.hpp"
#include <cstdio>
#include <cstdlib>
#include <infinirt.h>
#include <map>
#include <optional>
#include <tuple>
#include <vector>

namespace infinilm::quantization {
namespace {

struct Bf16GemmGenTimer {
    bool enabled = std::getenv("INFINI_MODEL_TIMER") != nullptr;
    bool dev_ok = false;
    infiniDevice_t dev_kind{};
    int dev_id = 0;
    struct Entry {
        int bucket = 0; // 0: prefill, 1: decode
        int fused = 0;  // 1: linear+allreduce kept as the model launches it
        size_t M = 0, N = 0, K = 0;
        infinirtEvent_t a = nullptr, b = nullptr;
    };
    std::vector<Entry> entries;
    std::map<std::tuple<int, int, size_t, size_t, size_t>, int> seen;
    std::map<std::tuple<int, int, size_t, size_t, size_t>, int> sampled;
    bool want_sample(int fused, int bucket, size_t M, size_t N, size_t K) {
        const auto key = std::tuple<int, int, size_t, size_t, size_t>{fused, bucket, M, N, K};
        const int n = seen[key]++;
        int &got = sampled[key];
        if (got >= 4) {
            return false;
        }
        if (n == 0 || n == 40 || n == 160 || n == 320) {
            got += 1;
            return true;
        }
        return false;
    }
    void capture_device() {
        if (!dev_ok) {
            dev_ok = infinirtGetDevice(&dev_kind, &dev_id) == INFINI_STATUS_SUCCESS;
        }
    }
    ~Bf16GemmGenTimer() {
        if (!enabled || entries.empty()) {
            return;
        }
        if (dev_ok) {
            infinirtSetDevice(dev_kind, dev_id);
        }
        struct Stat {
            double ms = 0.0;
            size_t calls = 0;
        };
        std::map<std::tuple<int, int, size_t, size_t, size_t>, Stat> agg;
        for (const auto &e : entries) {
            float m = 0.0f;
            if (infinirtEventSynchronize(e.b) == INFINI_STATUS_SUCCESS) {
                infinirtEventElapsedTime(&m, e.a, e.b);
            }
            infinirtEventDestroy(e.a);
            infinirtEventDestroy(e.b);
            auto &st = agg[{e.fused, e.bucket, e.M, e.N, e.K}];
            st.ms += m;
            st.calls += 1;
        }
        const char *bnames[2] = {"prefill", "decode"};
        const char *tags[2] = {"METRIC_BF16_GEMM_GEN", "METRIC_BF16_FUSED_GEN"};
        for (const auto &[key, st] : agg) {
            const auto [fused, b, M, N, K] = key;
            const double avg = st.ms / static_cast<double>(st.calls);
            const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) * static_cast<double>(st.calls);
            std::fprintf(stderr,
                         "%s %s M %zu N %zu K %zu samples %zu total_ms %.3f avg_ms %.4f tflops %.3f\n",
                         tags[fused ? 1 : 0],
                         bnames[b],
                         M, N, K,
                         st.calls,
                         st.ms,
                         avg,
                         st.ms > 0.0 ? flops / st.ms / 1.0e9 : 0.0);
        }
    }
};

Bf16GemmGenTimer &bf16_gemm_gen_timer() {
    thread_local Bf16GemmGenTimer timer;
    return timer;
}

} // namespace

NoneQuantization::NoneQuantization() : NoneQuantization(nlohmann::json()) {}

std::vector<ParamDescriptor> NoneQuantization::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {

    std::vector<ParamDescriptor> descs;
    descs.push_back({"weight", {out_features, in_features}, dtype, split_dim, tp_rank, tp_size});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, split_dim >= 0 ? 0 : -1, split_dim >= 0 ? tp_rank : 0, split_dim >= 0 ? tp_size : 1});
    }
    return descs;
}

infinicore::Tensor NoneQuantization::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float alpha) const {

    auto input_contiguous = input->is_contiguous() ? input : input->contiguous();
    auto weight = params.at("weight");

    std::optional<infinicore::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }

    // Ascend path: weight was pre-packed to [IC, OC] in process_weights_after_loading.
    // Use linear_packed to skip the runtime permute({1,0}).
    auto &gt = bf16_gemm_gen_timer();
    Bf16GemmGenTimer::Entry gentry{};
    infinirtStream_t gstream = gt.enabled ? infinicore::context::getStream() : nullptr;
    if (gt.enabled) {
        gt.capture_device();
        size_t M = 1;
        for (size_t i = 0; i + 1 < input_contiguous->ndim(); ++i) {
            M *= input_contiguous->size(i);
        }
        const size_t N = weight_prepacked_ ? weight->size(1) : weight->size(0);
        const size_t K = weight_prepacked_ ? weight->size(0) : weight->size(1);
        gentry.bucket = M > 1 ? 0 : 1;
        gentry.M = M;
        gentry.N = N;
        gentry.K = K;
        if (gt.want_sample(0, gentry.bucket, M, N, K)) {
            infinirtEventCreate(&gentry.a);
            infinirtEventCreate(&gentry.b);
            infinirtEventRecord(gentry.a, gstream);
        } else {
            gstream = nullptr;
        }
    }
    infinicore::Tensor out = weight_prepacked_
        ? infinicore::op::linear_packed(input_contiguous, weight, bias_opt, alpha)
        : infinicore::op::linear(input_contiguous->contiguous(), weight->contiguous(), bias_opt, alpha);
    if (gt.enabled && gstream != nullptr) {
        infinirtEventRecord(gentry.b, gstream);
        gt.entries.push_back(gentry);
    }
    return out;
}

infinicore::Tensor NoneQuantization::forward_allreduce(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    infinicclComm_t communicator,
    float alpha) const {
    if (alpha != 1.0f) {
        return BaseQuantization::forward_allreduce(
            params, input, has_bias, communicator, alpha);
    }

    auto input_contiguous = input->is_contiguous()
                              ? input
                              : input->contiguous();
    auto weight = params.at("weight");
    std::optional<infinicore::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }

    auto &gt = bf16_gemm_gen_timer();
    Bf16GemmGenTimer::Entry gentry{};
    infinirtStream_t gstream = nullptr;
    if (gt.enabled) {
        gt.capture_device();
        gstream = infinicore::context::getStream();
        size_t M = 1;
        for (size_t i = 0; i + 1 < input_contiguous->ndim(); ++i) {
            M *= input_contiguous->size(i);
        }
        const size_t N = weight_prepacked_ ? weight->size(1) : weight->size(0);
        const size_t K = weight_prepacked_ ? weight->size(0) : weight->size(1);
        gentry.bucket = M > 1 ? 0 : 1;
        gentry.fused = 1;
        gentry.M = M;
        gentry.N = N;
        gentry.K = K;
        if (gt.want_sample(1, gentry.bucket, M, N, K)) {
            infinirtEventCreate(&gentry.a);
            infinirtEventCreate(&gentry.b);
            infinirtEventRecord(gentry.a, gstream);
        } else {
            gstream = nullptr;
        }
    }
    infinicore::Tensor out = weight_prepacked_
        ? infinicore::op::linear_allreduce_packed(input_contiguous, weight, bias_opt, communicator)
        : infinicore::op::linear_allreduce(input_contiguous, weight->contiguous(), bias_opt, communicator);
    if (gt.enabled && gstream != nullptr) {
        infinirtEventRecord(gentry.b, gstream);
        gt.entries.push_back(gentry);
    }
    return out;
}

std::vector<SplitParam> NoneQuantization::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int narrow_dim,
    int tp_rank, int tp_size, int /*tp_num_heads*/) const {

    std::vector<SplitParam> result;
    auto weight_it = params.find("weight");
    auto bias_it = params.find("bias");

    for (const auto &s : splits) {
        result.push_back({s.prefix + ".weight",
                          infinicore::nn::Parameter(
                              weight_it->second->narrow({{static_cast<size_t>(narrow_dim), s.start, s.size}}),
                              narrow_dim, tp_rank, tp_size, s.num_shards)});
        if (bias_it != params.end()) {
            result.push_back({s.prefix + ".bias",
                              infinicore::nn::Parameter(
                                  bias_it->second->narrow({{0, s.start, s.size}}),
                                  0, tp_rank, tp_size, s.num_shards)});
        }
    }
    return result;
}

std::shared_ptr<BaseQuantization> NoneQuantization::process_weights_after_loading(
    ParamsMap &params,
    const infinicore::Device &device,
    int /*split_dim*/) const {

    // Controlled by --pre-transpose CLI flag, default off.
    if (!global_state::get_infinilm_config().pre_transpose) {
        return nullptr;
    }

    auto weight_it = params.find("weight");
    if (weight_it != params.end()) {
        // Transpose weight from [OC, IC] to [IC, OC] once.
        // contiguous() materializes the transposed layout so that
        // subsequent forwards can feed it directly to GEMM.
        params["weight"] = weight_it->second->permute({1, 0})->contiguous();

        // Mark as pre-packed so forward() uses linear_packed.
        weight_prepacked_ = true;
    }

    // Must return non-null so that BaseLinear::process_weights_after_loading
    // writes the modified params back into parameters_.
    // Returning shared_from_this() triggers the "quantization changed" path
    // which calls parameters_.clear() + re-insert from params.
    return std::const_pointer_cast<BaseQuantization>(shared_from_this());
}

} // namespace infinilm::quantization
