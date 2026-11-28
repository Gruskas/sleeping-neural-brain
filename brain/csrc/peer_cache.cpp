#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/Parallel.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <tuple>

void launch_peer_quantize_evict_cuda(
    const void* cache_u,
    const void* cache_v,
    bool cache_bf16,
    const void* slots,
    bool slots_i64,
    int8_t* u_q_out,
    void* u_s_out,
    int8_t* v_q_out,
    void* v_s_out,
    int64_t n_slots,
    int64_t dim,
    uint64_t seed,
    cudaStream_t stream
);

void launch_peer_dequant_scatter_cuda(
    const int8_t* u_q,
    const void* u_s,
    const int8_t* v_q,
    const void* v_s,
    const void* victim_slots,
    bool slots_i64,
    void* cache_u,
    void* cache_v,
    bool cache_bf16,
    int64_t n_miss,
    int64_t dim,
    cudaStream_t stream
);

namespace {

constexpr int64_t kPinStep = (int64_t(1) << 62);

std::mutex g_stage_evt_mu;
cudaEvent_t g_stage_event = nullptr;
bool g_stage_event_pending = false;

void sync_stage_event() {
    std::lock_guard<std::mutex> lock(g_stage_evt_mu);
    if (g_stage_event != nullptr && g_stage_event_pending) {
        C10_CUDA_CHECK(cudaEventSynchronize(g_stage_event));
        g_stage_event_pending = false;
    }
}

void record_stage_event(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_stage_evt_mu);
    if (g_stage_event == nullptr) {
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&g_stage_event, cudaEventDisableTiming));
    }
    C10_CUDA_CHECK(cudaEventRecord(g_stage_event, stream));
    g_stage_event_pending = true;
}

void copy_rows_from_ram(
    const int64_t* exp_ids,
    int64_t n_rows,
    int64_t dim,
    const int8_t* ram_u_q,
    const uint16_t* ram_u_s,
    const int8_t* ram_v_q,
    const uint16_t* ram_v_s,
    int8_t* stg_u_q,
    uint16_t* stg_u_s,
    int8_t* stg_v_q,
    uint16_t* stg_v_s
) {
    at::parallel_for(0, n_rows, 32, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            int64_t e = exp_ids[i];
            std::memcpy(stg_u_q + i * dim, ram_u_q + e * dim, static_cast<size_t>(dim));
            std::memcpy(stg_v_q + i * dim, ram_v_q + e * dim, static_cast<size_t>(dim));
            stg_u_s[i] = ram_u_s[e];
            stg_v_s[i] = ram_v_s[e];
        }
    });
}

void copy_rows_to_ram(
    const int64_t* exp_ids,
    int64_t n_rows,
    int64_t dim,
    const int8_t* stg_u_q,
    const uint16_t* stg_u_s,
    const int8_t* stg_v_q,
    const uint16_t* stg_v_s,
    int8_t* ram_u_q,
    uint16_t* ram_u_s,
    int8_t* ram_v_q,
    uint16_t* ram_v_s
) {
    at::parallel_for(0, n_rows, 32, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            int64_t e = exp_ids[i];
            std::memcpy(ram_u_q + e * dim, stg_u_q + i * dim, static_cast<size_t>(dim));
            std::memcpy(ram_v_q + e * dim, stg_v_q + i * dim, static_cast<size_t>(dim));
            ram_u_s[e] = stg_u_s[i];
            ram_v_s[e] = stg_v_s[i];
        }
    });
}

void flush_slots_to_ram_impl(
    const torch::Tensor& cache_u,
    const torch::Tensor& cache_v,
    const torch::Tensor& dirty_slots,
    const torch::Tensor& dirty_experts,
    torch::Tensor& ram_u_q,
    torch::Tensor& ram_u_scale,
    torch::Tensor& ram_v_q,
    torch::Tensor& ram_v_scale,
    torch::Tensor& stage_u_q,
    torch::Tensor& stage_u_scale,
    torch::Tensor& stage_v_q,
    torch::Tensor& stage_v_scale,
    int64_t seed
) {
    auto slots_c = dirty_slots.contiguous();
    int64_t n_dirty = slots_c.size(0);
    if (n_dirty == 0) return;

    int64_t dim = cache_u.size(1);
    auto dev_opts = cache_u.options();
    auto u_q_dev = torch::empty({n_dirty, dim}, dev_opts.dtype(torch::kInt8));
    auto u_s_dev = torch::empty({n_dirty}, dev_opts.dtype(torch::kFloat16));
    auto v_q_dev = torch::empty({n_dirty, dim}, dev_opts.dtype(torch::kInt8));
    auto v_s_dev = torch::empty({n_dirty}, dev_opts.dtype(torch::kFloat16));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(cache_u.device().index()).stream();
    launch_peer_quantize_evict_cuda(
        cache_u.data_ptr(),
        cache_v.data_ptr(),
        cache_u.scalar_type() == torch::kBFloat16,
        slots_c.data_ptr(),
        slots_c.scalar_type() == torch::kInt64,
        u_q_dev.data_ptr<int8_t>(),
        u_s_dev.data_ptr(),
        v_q_dev.data_ptr<int8_t>(),
        v_s_dev.data_ptr(),
        n_dirty,
        dim,
        static_cast<uint64_t>(seed),
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto dirty_exp_cpu = dirty_experts.to(torch::kInt64).cpu().contiguous();

    sync_stage_event();

    torch::Tensor stg_uq = (n_dirty <= stage_u_q.size(0)) ? stage_u_q.slice(0, 0, n_dirty) : torch::empty({n_dirty, dim}, torch::kInt8);
    torch::Tensor stg_us = (n_dirty <= stage_u_scale.size(0)) ? stage_u_scale.slice(0, 0, n_dirty) : torch::empty({n_dirty}, torch::kFloat16);
    torch::Tensor stg_vq = (n_dirty <= stage_v_q.size(0)) ? stage_v_q.slice(0, 0, n_dirty) : torch::empty({n_dirty, dim}, torch::kInt8);
    torch::Tensor stg_vs = (n_dirty <= stage_v_scale.size(0)) ? stage_v_scale.slice(0, 0, n_dirty) : torch::empty({n_dirty}, torch::kFloat16);

    C10_CUDA_CHECK(cudaMemcpyAsync(stg_uq.data_ptr(), u_q_dev.data_ptr(), static_cast<size_t>(n_dirty * dim) * sizeof(int8_t), cudaMemcpyDeviceToHost, stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(stg_us.data_ptr(), u_s_dev.data_ptr(), static_cast<size_t>(n_dirty) * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(stg_vq.data_ptr(), v_q_dev.data_ptr(), static_cast<size_t>(n_dirty * dim) * sizeof(int8_t), cudaMemcpyDeviceToHost, stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(stg_vs.data_ptr(), v_s_dev.data_ptr(), static_cast<size_t>(n_dirty) * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));

    const int64_t* exp_ptr = dirty_exp_cpu.data_ptr<int64_t>();
    const int8_t* sq_u = stg_uq.data_ptr<int8_t>();
    const uint16_t* ss_u = reinterpret_cast<const uint16_t*>(stg_us.data_ptr<at::Half>());
    const int8_t* sq_v = stg_vq.data_ptr<int8_t>();
    const uint16_t* ss_v = reinterpret_cast<const uint16_t*>(stg_vs.data_ptr<at::Half>());

    int8_t* rq_u = ram_u_q.data_ptr<int8_t>();
    uint16_t* rs_u = reinterpret_cast<uint16_t*>(ram_u_scale.data_ptr<at::Half>());
    int8_t* rq_v = ram_v_q.data_ptr<int8_t>();
    uint16_t* rs_v = reinterpret_cast<uint16_t*>(ram_v_scale.data_ptr<at::Half>());

    {
        py::gil_scoped_release release;
        copy_rows_to_ram(exp_ptr, n_dirty, dim, sq_u, ss_u, sq_v, ss_v, rq_u, rs_u, rq_v, rs_v);
    }
}

}  // namespace

std::tuple<torch::Tensor, int64_t, int64_t, int64_t> peer_cache_ensure_cpp(
    const torch::Tensor& expert_ids,
    const torch::Tensor& uniq_ids,
    torch::Tensor cache_u,
    torch::Tensor cache_v,
    torch::Tensor expert_to_slot,
    torch::Tensor slot_to_expert,
    torch::Tensor slot_last_step,
    torch::Tensor slot_dirty,
    torch::Tensor ram_u_q,
    torch::Tensor ram_u_scale,
    torch::Tensor ram_v_q,
    torch::Tensor ram_v_scale,
    torch::Tensor stage_u_q,
    torch::Tensor stage_u_scale,
    torch::Tensor stage_v_q,
    torch::Tensor stage_v_scale,
    int64_t step_counter,
    int64_t seed
) {
    torch::NoGradGuard no_grad;
    TORCH_CHECK(cache_u.is_cuda() && uniq_ids.is_cuda(), "peer_cache_ensure_cpp expects CUDA cache tensors");

    int64_t u_num = uniq_ids.size(0);
    int64_t cache_slots = cache_u.size(0);
    TORCH_CHECK(u_num <= cache_slots, "Active unique PEER experts exceed cache_slots");

    auto slots = expert_to_slot.index_select(0, uniq_ids);
    auto miss_mask = slots.lt(0);
    auto hit_mask = ~miss_mask;

    int64_t n_miss = miss_mask.sum().item<int64_t>();
    int64_t n_hits = u_num - n_miss;

    if (n_hits > 0) {
        auto hit_slots = slots.index({hit_mask}).to(torch::kInt64);
        slot_last_step.index_fill_(0, hit_slots, kPinStep);
    }

    if (n_miss > 0) {
        int64_t dim = cache_u.size(1);
        auto miss_experts = uniq_ids.index({miss_mask}).contiguous();
        auto victim_slots = std::get<1>(torch::topk(slot_last_step, n_miss, /*dim=*/0, /*largest=*/false, /*sorted=*/false)).contiguous();

        auto old_experts = slot_to_expert.index_select(0, victim_slots);
        auto evict_mask = old_experts.ge(0);
        auto dirty_mask = evict_mask & slot_dirty.index_select(0, victim_slots);

        if (dirty_mask.any().item<bool>()) {
            auto dirty_slots = victim_slots.index({dirty_mask});
            auto dirty_experts = old_experts.index({dirty_mask});
            flush_slots_to_ram_impl(
                cache_u, cache_v, dirty_slots, dirty_experts,
                ram_u_q, ram_u_scale, ram_v_q, ram_v_scale,
                stage_u_q, stage_u_scale, stage_v_q, stage_v_scale,
                seed
            );
        }

        if (evict_mask.any().item<bool>()) {
            expert_to_slot.index_fill_(0, old_experts.index({evict_mask}).to(torch::kInt64), -1);
        }

        auto miss_exp_cpu = miss_experts.to(torch::kInt64).cpu().contiguous();
        sync_stage_event();
        torch::Tensor stg_uq = (n_miss <= stage_u_q.size(0)) ? stage_u_q.slice(0, 0, n_miss) : torch::empty({n_miss, dim}, torch::kInt8);
        torch::Tensor stg_us = (n_miss <= stage_u_scale.size(0)) ? stage_u_scale.slice(0, 0, n_miss) : torch::empty({n_miss}, torch::kFloat16);
        torch::Tensor stg_vq = (n_miss <= stage_v_q.size(0)) ? stage_v_q.slice(0, 0, n_miss) : torch::empty({n_miss, dim}, torch::kInt8);
        torch::Tensor stg_vs = (n_miss <= stage_v_scale.size(0)) ? stage_v_scale.slice(0, 0, n_miss) : torch::empty({n_miss}, torch::kFloat16);

        const int64_t* exp_ptr = miss_exp_cpu.data_ptr<int64_t>();
        const int8_t* rq_u = ram_u_q.data_ptr<int8_t>();
        const uint16_t* rs_u = reinterpret_cast<const uint16_t*>(ram_u_scale.data_ptr<at::Half>());
        const int8_t* rq_v = ram_v_q.data_ptr<int8_t>();
        const uint16_t* rs_v = reinterpret_cast<const uint16_t*>(ram_v_scale.data_ptr<at::Half>());

        int8_t* sq_u = stg_uq.data_ptr<int8_t>();
        uint16_t* ss_u = reinterpret_cast<uint16_t*>(stg_us.data_ptr<at::Half>());
        int8_t* sq_v = stg_vq.data_ptr<int8_t>();
        uint16_t* ss_v = reinterpret_cast<uint16_t*>(stg_vs.data_ptr<at::Half>());

        {
            py::gil_scoped_release release;
            copy_rows_from_ram(exp_ptr, n_miss, dim, rq_u, rs_u, rq_v, rs_v, sq_u, ss_u, sq_v, ss_v);
        }

        auto dev_opts = cache_u.options();
        auto u_q_gpu = torch::empty({n_miss, dim}, dev_opts.dtype(torch::kInt8));
        auto u_s_gpu = torch::empty({n_miss}, dev_opts.dtype(torch::kFloat16));
        auto v_q_gpu = torch::empty({n_miss, dim}, dev_opts.dtype(torch::kInt8));
        auto v_s_gpu = torch::empty({n_miss}, dev_opts.dtype(torch::kFloat16));

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(cache_u.device().index()).stream();
        C10_CUDA_CHECK(cudaMemcpyAsync(u_q_gpu.data_ptr(), sq_u, static_cast<size_t>(n_miss * dim) * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(u_s_gpu.data_ptr(), ss_u, static_cast<size_t>(n_miss) * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(v_q_gpu.data_ptr(), sq_v, static_cast<size_t>(n_miss * dim) * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
        C10_CUDA_CHECK(cudaMemcpyAsync(v_s_gpu.data_ptr(), ss_v, static_cast<size_t>(n_miss) * sizeof(uint16_t), cudaMemcpyHostToDevice, stream));
        record_stage_event(stream);

        launch_peer_dequant_scatter_cuda(
            u_q_gpu.data_ptr<int8_t>(),
            u_s_gpu.data_ptr(),
            v_q_gpu.data_ptr<int8_t>(),
            v_s_gpu.data_ptr(),
            victim_slots.data_ptr(),
            victim_slots.scalar_type() == torch::kInt64,
            cache_u.data_ptr(),
            cache_v.data_ptr(),
            cache_u.scalar_type() == torch::kBFloat16,
            n_miss,
            dim,
            stream
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        slot_dirty.index_fill_(0, victim_slots, false);
        slot_to_expert.index_copy_(0, victim_slots, miss_experts.to(torch::kInt32));
        expert_to_slot.index_copy_(0, miss_experts, victim_slots.to(torch::kInt32));
    }

    int64_t new_step = step_counter + 1;
    auto active_slots = expert_to_slot.index_select(0, uniq_ids).to(torch::kInt64);
    slot_last_step.index_fill_(0, active_slots, new_step);

    auto slot_ids = expert_to_slot.index_select(0, expert_ids.reshape({-1})).view(expert_ids.sizes()).to(torch::kInt64);
    return {slot_ids, n_hits, n_miss, new_step};
}

int64_t peer_flush_dirty_cpp(
    const torch::Tensor& cache_u,
    const torch::Tensor& cache_v,
    const torch::Tensor& slot_to_expert,
    torch::Tensor slot_dirty,
    torch::Tensor ram_u_q,
    torch::Tensor ram_u_scale,
    torch::Tensor ram_v_q,
    torch::Tensor ram_v_scale,
    torch::Tensor stage_u_q,
    torch::Tensor stage_u_scale,
    torch::Tensor stage_v_q,
    torch::Tensor stage_v_scale,
    int64_t seed
) {
    torch::NoGradGuard no_grad;
    auto dirty_mask = slot_dirty & slot_to_expert.ge(0);
    auto dirty_slots = torch::nonzero(dirty_mask).squeeze(-1);
    int64_t n_flushed = dirty_slots.size(0);
    if (n_flushed == 0) return 0;

    auto dirty_experts = slot_to_expert.index_select(0, dirty_slots).to(torch::kInt64);
    flush_slots_to_ram_impl(
        cache_u, cache_v, dirty_slots, dirty_experts,
        ram_u_q, ram_u_scale, ram_v_q, ram_v_scale,
        stage_u_q, stage_u_scale, stage_v_q, stage_v_scale,
        seed
    );
    slot_dirty.index_fill_(0, dirty_slots, false);
    return n_flushed;
}
