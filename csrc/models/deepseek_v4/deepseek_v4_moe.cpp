#include "deepseek_v4_moe.hpp"

#include "../../global_state/global_state.hpp"
#include "deepseek_v4_runtime.hpp"

#include "infinicore/ops/add.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/embedding.hpp"
#include "infinicore/ops/per_channel_quant_i8.hpp"
#include "infinicore/ops/w4a8_group_gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <infinirt.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

struct RouteNvtx {
    using PushFn = int (*)(const char *);
    using PopFn = int (*)();
    PushFn push_fn = nullptr;
    PopFn pop_fn = nullptr;
    bool enabled = false;
    RouteNvtx() {
        if (std::getenv("INFINI_NVTX") == nullptr) {
            return;
        }
        void *lib = dlopen("/usr/local/corex/lib64/libnvToolsExt.so", RTLD_LAZY);
        if (lib == nullptr) {
            return;
        }
        push_fn = reinterpret_cast<PushFn>(dlsym(lib, "nvtxRangePushA"));
        pop_fn = reinterpret_cast<PopFn>(dlsym(lib, "nvtxRangePop"));
        enabled = push_fn != nullptr && pop_fn != nullptr;
    }
    void on(const char *name) const {
        if (enabled) {
            push_fn(name);
        }
    }
    void off() const {
        if (enabled) {
            pop_fn();
        }
    }
};

const RouteNvtx &route_nvtx() {
    static RouteNvtx nvtx;
    return nvtx;
}

struct RouteRange {
    const RouteNvtx &nv;
    explicit RouteRange(const char *name) : nv(route_nvtx()) { nv.on(name); }
    ~RouteRange() { nv.off(); }
};

struct MoeGenTimer {
    bool enabled = std::getenv("INFINI_MODEL_TIMER") != nullptr;
    bool dev_ok = false;
    infiniDevice_t dev_kind{};
    int dev_id = 0;
    void capture_device() {
        if (!dev_ok) {
            dev_ok = infinirtGetDevice(&dev_kind, &dev_id) == INFINI_STATUS_SUCCESS;
        }
    }
    struct Pair {
        infinirtEvent_t a = nullptr;
        infinirtEvent_t b = nullptr;
    };
    struct Entry {
        int bucket = 0; // 0: prefill, 1: decode, 2: spec
        size_t rows = 0;
        size_t max_expert = 0;
        Pair gather;
        Pair q1;
        Pair g13;
        Pair q2;
        Pair g2;
        Pair scatter;
    };
    std::vector<Entry> entries;
    static Pair make_pair() {
        Pair p;
        infinirtEventCreate(&p.a);
        infinirtEventCreate(&p.b);
        return p;
    }
    static float elapsed(const Pair &p) {
        float ms = 0.0f;
        infinirtEventSynchronize(p.b);
        infinirtEventElapsedTime(&ms, p.a, p.b);
        infinirtEventDestroy(p.a);
        infinirtEventDestroy(p.b);
        return ms;
    }
    ~MoeGenTimer() {
        if (!enabled || entries.empty()) {
            return;
        }
        if (dev_ok) {
            infinirtSetDevice(dev_kind, dev_id);
        }
        const char *bnames[3] = {"prefill", "decode", "spec"};
        double route_ms[3] = {0, 0, 0};
        double quant_ms[3] = {0, 0, 0};
        double gemm_ms[3] = {0, 0, 0};
        double gemm_flops[3] = {0, 0, 0};
        size_t calls[3] = {0, 0, 0};
        size_t total_rows[3] = {0, 0, 0};
        double dist_ms[3] = {0, 0, 0};
        size_t dist_n[3] = {0, 0, 0};
        for (const auto &e : entries) {
            const float ga = elapsed(e.gather);
            const float sc = elapsed(e.scatter);
            const float qa = elapsed(e.q1) + elapsed(e.q2);
            const float gm = elapsed(e.g13) + elapsed(e.g2);
            const int b = e.bucket;
            route_ms[b] += ga + sc;
            quant_ms[b] += qa;
            gemm_ms[b] += gm;
            // 3 GEMMs per MoE layer: w1, w3 (rows x 2048 x 4096) and w2 (rows x 4096 x 2048)
            gemm_flops[b] += 3.0 * 2.0 * static_cast<double>(e.rows) * 2048.0 * 4096.0;
            calls[b] += 1;
            total_rows[b] += e.rows;
            if (b == 0 && e.rows > 0) {
                const double ratio = static_cast<double>(e.max_expert) / static_cast<double>(e.rows);
                const int d = ratio >= 0.5 ? 2 : (ratio >= 0.12 ? 1 : 0);
                dist_ms[d] += ga + sc;
                dist_n[d] += 1;
            }
        }
        for (int b = 0; b < 3; ++b) {
            if (calls[b] == 0) {
                continue;
            }
            const double tot = quant_ms[b] + gemm_ms[b];
            std::fprintf(stderr,
                         "METRIC_MOE_GEN %s calls %zu rows %zu route_ms %.3f route_avg_ms %.3f quant_ms %.3f gemm_ms %.3f gemm_tflops %.4f quant_share %.4f\n",
                         bnames[b],
                         calls[b],
                         total_rows[b],
                         route_ms[b],
                         route_ms[b] / static_cast<double>(calls[b]),
                         quant_ms[b],
                         gemm_ms[b],
                         gemm_ms[b] > 0.0 ? gemm_flops[b] / gemm_ms[b] / 1.0e9 : 0.0,
                         tot > 0.0 ? quant_ms[b] / tot : 0.0);
        }
        const char *dnames[3] = {"uniform", "skewed", "extreme_sparse"};
        for (int d = 0; d < 3; ++d) {
            if (dist_n[d] > 0) {
                std::fprintf(stderr,
                             "METRIC_MOE_DIST %s layers %zu avg_ms %.3f\n",
                             dnames[d],
                             dist_n[d],
                             dist_ms[d] / static_cast<double>(dist_n[d]));
            }
        }
    }
};

MoeGenTimer &moe_gen_timer() {
    thread_local MoeGenTimer timer;
    return timer;
}

} // namespace

DeepseekV4Gate::DeepseekV4Gate(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               const infinicore::Device &device,
                               bool has_tid2eid,
                               bool has_bias) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden = model_config->get<size_t>("hidden_size");
    const size_t n_experts = model_config->get<size_t>("n_routed_experts");
    const size_t topk = model_config->get<size_t>("num_experts_per_tok");
    const size_t vocab = model_config->get<size_t>("vocab_size");
    INFINICORE_NN_PARAMETER_INIT(weight, ({n_experts, hidden}, dtype, device));
    if (has_tid2eid) {
        INFINICORE_NN_PARAMETER_INIT(tid2eid, ({vocab, topk}, infinicore::DataType::I64, device));
    }
    if (has_bias) {
        INFINICORE_NN_PARAMETER_INIT(bias, ({n_experts}, infinicore::DataType::F32, device));
    }
}

DeepseekV4SharedExperts::DeepseekV4SharedExperts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                 const infinicore::Device &device) {
    const auto &dtype = model_config->get_dtype();
    const size_t hidden = model_config->get<size_t>("hidden_size");
    const size_t intermediate = model_config->get<size_t>("moe_intermediate_size");
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    auto quant = int8_quantization();
    swiglu_limit_ = static_cast<float>(model_config->get<double>("swiglu_limit"));
    INFINICORE_NN_MODULE_INIT(w1, hidden, intermediate, quant, false, dtype, device, rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(w3, hidden, intermediate, quant, false, dtype, device, rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(w2, intermediate, hidden, quant, false, dtype, device, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
}

DeepseekV4MoE::DeepseekV4MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device,
                             bool has_tid2eid,
                             bool has_bias) {
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    tp_rank_ = rank_info.tp_rank;
    tp_size_ = rank_info.tp_size;
    comm_ = rank_info.comm;
    n_experts_ = model_config->get<size_t>("n_routed_experts");
    hidden_ = model_config->get<size_t>("hidden_size");
    intermediate_ = model_config->get<size_t>("moe_intermediate_size");
    topk_ = model_config->get<size_t>("num_experts_per_tok");
    routed_scale_ = static_cast<float>(model_config->get<double>("routed_scaling_factor"));
    swiglu_limit_ = static_cast<float>(model_config->get<double>("swiglu_limit"));
    renormalize_ = model_config->get<bool>("norm_topk_prob");
    hash_ = has_tid2eid;
    has_bias_ = has_bias;
    const size_t hidden = hidden_;
    const size_t intermediate = intermediate_;
    if (tp_size_ <= 0 || n_experts_ % static_cast<size_t>(tp_size_) != 0) {
        throw std::runtime_error("deepseek_v4: n_routed_experts must be divisible by tp size");
    }
    if (hidden % 2 != 0 || intermediate % 2 != 0) {
        throw std::runtime_error("deepseek_v4: hidden and moe intermediate sizes must be even for packed int4 experts");
    }
    const size_t local = n_experts_ / static_cast<size_t>(tp_size_);
    const size_t w1_k = hidden / 2;
    const size_t w2_k = intermediate / 2;

    INFINICORE_NN_MODULE_INIT(gate, model_config, device, has_tid2eid, has_bias);
    INFINICORE_NN_MODULE_INIT(shared_experts, model_config, device);
    // Checkpoint scales are [1, N] (TN). Packed expert weights are [E, N, K/2].
    INFINICORE_NN_BUFFER_INIT(w1_packed, ({local, intermediate, w1_k}, infinicore::DataType::I8, device));
    INFINICORE_NN_BUFFER_INIT(w1_scale_packed, ({local, intermediate, size_t{1}}, infinicore::DataType::F32, device));
    INFINICORE_NN_BUFFER_INIT(w2_packed, ({local, hidden, w2_k}, infinicore::DataType::I8, device));
    INFINICORE_NN_BUFFER_INIT(w2_scale_packed, ({local, hidden, size_t{1}}, infinicore::DataType::F32, device));
    INFINICORE_NN_BUFFER_INIT(w3_packed, ({local, intermediate, w1_k}, infinicore::DataType::I8, device));
    INFINICORE_NN_BUFFER_INIT(w3_scale_packed, ({local, intermediate, size_t{1}}, infinicore::DataType::F32, device));
}

void DeepseekV4MoE::copy_slot_(infinicore::Tensor packed,
                               size_t local_index,
                               const infinicore::Tensor &src,
                               const std::string &what) const {
    try {
        packed->narrow({{0, local_index, 1}})->view(src->shape())->copy_from(src);
    } catch (const std::exception &err) {
        throw std::runtime_error("deepseek_v4 expert copy failed for " + what + ": " + err.what());
    }
}

void DeepseekV4MoE::ingest_expert(size_t expert_id,
                                  const std::string &proj,
                                  bool is_scale,
                                  const infinicore::Tensor &tensor) {
    if (expert_id >= n_experts_) {
        throw std::runtime_error("deepseek_v4 expert id out of range");
    }
    if (expert_id % static_cast<size_t>(tp_size_) != static_cast<size_t>(tp_rank_)) {
        return;
    }
    const size_t local_index = expert_id / static_cast<size_t>(tp_size_);
    infinicore::Tensor packed;
    if (proj == "w1") {
        packed = is_scale ? w1_scale_packed_ : w1_packed_;
    } else if (proj == "w2") {
        packed = is_scale ? w2_scale_packed_ : w2_packed_;
    } else if (proj == "w3") {
        packed = is_scale ? w3_scale_packed_ : w3_packed_;
    } else {
        throw std::runtime_error("deepseek_v4 unknown expert projection " + proj);
    }
    infinicore::Tensor src = tensor;
    if (is_scale) {
        if (src->ndim() != 2) {
            throw std::runtime_error("deepseek_v4 expert scale rank is not 2");
        }
        src = src->permute({1, 0});
        if (!src->is_contiguous()) {
            src = src->contiguous();
        }
    }
    copy_slot_(packed, local_index, src, proj + (is_scale ? ".weight_scale" : ".weight"));
}

infinicore::Tensor DeepseekV4SharedExperts::forward(infinicore::Tensor hidden) const {
    auto gate = int8_gemm(hidden, w1_->weight(), w1_->weight_scale());
    auto up = int8_gemm(hidden, w3_->weight(), w3_->weight_scale());
    auto mid = swiglu_clamp(gate, up, swiglu_limit_);
    auto out = int8_gemm(mid, w2_->weight(), w2_->weight_scale());
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (rank_info.tp_size > 1 && rank_info.comm != nullptr) {
        infinicore::op::distributed::allreduce_(out, out, INFINICCL_SUM, rank_info.comm);
    }
    return out;
}

infinicore::Tensor DeepseekV4MoE::forward(infinicore::Tensor hidden, const std::vector<int64_t> &token_ids) const {
    auto x = hidden->contiguous();
    const size_t tokens = x->size(0);
    const auto device = x->device();
    const auto dtype = x->dtype();
    if (!gate_t_) {
        gate_t_ = cast_dtype(gate_->route_weight()->permute({1, 0})->contiguous(), infinicore::DataType::F32);
    }
    auto logits = download_f32(infinicore::op::matmul(cast_dtype(x, infinicore::DataType::F32), gate_t_));
    if (logits.size() != tokens * n_experts_) {
        throw std::runtime_error("deepseek_v4 moe: router score size mismatch");
    }
    for (float &value : logits) {
        const float sp = value > 20.0f ? value : (value < -20.0f ? std::exp(value) : std::log1p(std::exp(value)));
        value = std::sqrt(std::max(sp, 0.0f));
    }
    if (has_bias_ && bias_cpu_.empty()) {
        bias_cpu_ = download_f32(gate_->bias());
    }
    if (hash_ && tid_cpu_.empty()) {
        tid_cpu_ = download_i64(gate_->tid2eid());
    }

    struct Route {
        size_t token;
        size_t local;
        float weight;
    };
    std::vector<Route> routes;
    for (size_t t = 0; t < tokens; ++t) {
        std::vector<std::pair<int, float>> chosen;
        chosen.reserve(topk_);
        if (hash_) {
            if (t >= token_ids.size()) {
                throw std::runtime_error("deepseek_v4 moe: hash routing needs one token id per row");
            }
            const int64_t token_id = token_ids[t];
            if (token_id < 0 || static_cast<size_t>(token_id) * topk_ + topk_ > tid_cpu_.size()) {
                throw std::runtime_error("deepseek_v4 moe: token id is outside tid2eid");
            }
            for (size_t k = 0; k < topk_; ++k) {
                const int id = static_cast<int>(tid_cpu_[static_cast<size_t>(token_id) * topk_ + k]);
                chosen.push_back({id, logits[t * n_experts_ + static_cast<size_t>(id)]});
            }
        } else {
            std::vector<std::pair<float, int>> ranked(n_experts_);
            for (size_t e = 0; e < n_experts_; ++e) {
                const float bias = has_bias_ ? bias_cpu_[e] : 0.0f;
                ranked[e] = {logits[t * n_experts_ + e] + bias, static_cast<int>(e)};
            }
            std::partial_sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(topk_), ranked.end(),
                              [](const auto &a, const auto &b) { return a.first > b.first; });
            for (size_t k = 0; k < topk_; ++k) {
                const int id = ranked[k].second;
                chosen.push_back({id, logits[t * n_experts_ + static_cast<size_t>(id)]});
            }
        }
        float denom = 0.0f;
        for (const auto &item : chosen) {
            denom += item.second;
        }
        for (auto &item : chosen) {
            if (renormalize_ && denom != 0.0f) {
                item.second /= denom;
            }
            item.second *= routed_scale_;
            if (item.first < 0 || static_cast<size_t>(item.first) >= n_experts_) {
                throw std::runtime_error("deepseek_v4 moe: expert id out of range");
            }
            if (item.first % tp_size_ != tp_rank_) {
                continue;
            }
            const size_t local_index = static_cast<size_t>(item.first) / static_cast<size_t>(tp_size_);
            routes.push_back(Route{t, local_index, item.second});
        }
    }
    std::stable_sort(routes.begin(), routes.end(), [](const Route &a, const Route &b) { return a.local < b.local; });

    auto acc = infinicore::Tensor::zeros({tokens, hidden_}, dtype, device);
    const size_t rows = routes.size();
    if (rows > 0) {
        auto &mt = moe_gen_timer();
        infinirtStream_t stream = mt.enabled ? infinicore::context::getStream() : nullptr;
        MoeGenTimer::Entry entry{};
        if (mt.enabled) {
            mt.capture_device();
            entry.bucket = tokens >= 32 ? 0 : (tokens == 1 ? 1 : 2);
            entry.rows = rows;
            entry.gather = MoeGenTimer::make_pair();
            entry.q1 = MoeGenTimer::make_pair();
            entry.g13 = MoeGenTimer::make_pair();
            entry.q2 = MoeGenTimer::make_pair();
            entry.g2 = MoeGenTimer::make_pair();
            entry.scatter = MoeGenTimer::make_pair();
            infinirtEventRecord(entry.gather.a, stream);
        }
        const size_t local_experts = w1_packed_->size(0);
        std::vector<int32_t> counts(local_experts, 0);
        std::vector<int32_t> token_indices(rows);
        std::vector<float> route_weights(rows);
        for (size_t i = 0; i < rows; ++i) {
            if (routes[i].local >= local_experts) {
                throw std::runtime_error("deepseek_v4 moe: local expert index out of range");
            }
            counts[routes[i].local] += 1;
            token_indices[i] = static_cast<int32_t>(routes[i].token);
            route_weights[i] = routes[i].weight;
        }
        auto route_index = upload_i32(token_indices, {rows}, device);
        infinicore::Tensor packed_act;
        {
            RouteRange gather_range("moe_route_gather");
            packed_act = infinicore::op::embedding(route_index, x);
        }
        if (mt.enabled) {
            infinirtEventRecord(entry.gather.b, stream);
            for (int32_t c : counts) {
                entry.max_expert = std::max(entry.max_expert, static_cast<size_t>(c));
            }
        }
        auto tokens_per_experts = upload_i32(counts, {local_experts}, infinicore::Device::cpu());
        auto act_i8 = infinicore::Tensor::empty({rows, hidden_}, infinicore::DataType::I8, device);
        auto act_scale = infinicore::Tensor::empty({rows, size_t{1}}, infinicore::DataType::F32, device);
        if (mt.enabled) {
            infinirtEventRecord(entry.q1.a, stream);
        }
        infinicore::op::per_channel_quant_i8_(packed_act, act_i8, act_scale);
        if (mt.enabled) {
            infinirtEventRecord(entry.q1.b, stream);
        }
        auto gate = infinicore::Tensor::empty({rows, intermediate_}, dtype, device);
        auto up = infinicore::Tensor::empty({rows, intermediate_}, dtype, device);
        if (mt.enabled) {
            infinirtEventRecord(entry.g13.a, stream);
        }
        infinicore::op::w4a8_group_gemm_(gate, act_i8, w1_packed_->contiguous(), act_scale, w1_scale_packed_->contiguous(), tokens_per_experts);
        infinicore::op::w4a8_group_gemm_(up, act_i8, w3_packed_->contiguous(), act_scale, w3_scale_packed_->contiguous(), tokens_per_experts);
        if (mt.enabled) {
            infinirtEventRecord(entry.g13.b, stream);
        }
        auto mid = swiglu_clamp(gate, up, swiglu_limit_);
        auto mid_i8 = infinicore::Tensor::empty({rows, intermediate_}, infinicore::DataType::I8, device);
        auto mid_scale = infinicore::Tensor::empty({rows, size_t{1}}, infinicore::DataType::F32, device);
        if (mt.enabled) {
            infinirtEventRecord(entry.q2.a, stream);
        }
        infinicore::op::per_channel_quant_i8_(mid, mid_i8, mid_scale);
        if (mt.enabled) {
            infinirtEventRecord(entry.q2.b, stream);
        }
        auto down = infinicore::Tensor::empty({rows, hidden_}, dtype, device);
        if (mt.enabled) {
            infinirtEventRecord(entry.g2.a, stream);
        }
        infinicore::op::w4a8_group_gemm_(down, mid_i8, w2_packed_->contiguous(), mid_scale, w2_scale_packed_->contiguous(), tokens_per_experts);
        if (mt.enabled) {
            infinirtEventRecord(entry.g2.b, stream);
            infinirtEventRecord(entry.scatter.a, stream);
        }
        std::vector<float> route_mix(tokens * rows, 0.0f);
        for (size_t i = 0; i < rows; ++i) {
            route_mix[routes[i].token * rows + i] = route_weights[i];
        }
        auto route_matrix = cast_dtype(upload_f32(route_mix, {tokens, rows}, device), dtype);
        {
            RouteRange scatter_range("moe_route_scatter");
            acc = infinicore::op::matmul(route_matrix, down);
        }
        if (mt.enabled) {
            infinirtEventRecord(entry.scatter.b, stream);
            mt.entries.push_back(entry);
        }
    }
    if (tp_size_ > 1 && comm_ != nullptr) {
        infinicore::op::distributed::allreduce_(acc, acc, INFINICCL_SUM, comm_);
    }
    auto shared = shared_experts_->forward(x);
    return infinicore::op::add(acc, shared);
}

} // namespace infinilm::models::deepseek_v4
