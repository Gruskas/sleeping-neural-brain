#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>

void launch_quantize_rows_cuda(
    const void* src,
    int src_dtype_id,
    void* q_out,
    float* row_scale,
    int64_t rows,
    int64_t cols,
    int mode_id,
    bool stochastic,
    uint64_t seed,
    cudaStream_t stream
);

void launch_shy_prune_rows_cuda(
    void* q_data,
    float* row_scale,
    int64_t rows,
    int64_t cols,
    int mode_id,
    float downscale,
    float eps,
    bool zero_diag,
    unsigned long long* pruned_counter,
    cudaStream_t stream
);

void launch_peer_forward_cuda(
    const void* x_flat,
    bool x_bf16,
    const void* slot_ids,
    bool slots_i64,
    const float* weights,
    const void* cache_u,
    const void* cache_v,
    bool cache_bf16,
    void* out,
    float* pre_act,
    int64_t n_tokens,
    int64_t k_tot,
    int64_t dim,
    cudaStream_t stream
);

void launch_peer_backward_update_cuda(
    const void* grad_out,
    const void* x_flat,
    bool x_bf16,
    const void* slot_ids,
    bool slots_i64,
    const float* weights,
    const float* pre_act,
    void* cache_u,
    void* cache_v,
    bool cache_bf16,
    bool* slot_dirty,
    void* grad_x,
    float* grad_weights,
    float* g_u,
    float* g_v,
    int32_t* slot_touched,
    int64_t n_tokens,
    int64_t k_tot,
    int64_t num_slots,
    int64_t dim,
    float lr,
    float weight_decay,
    bool do_update,
    uint64_t seed,
    cudaStream_t stream
);

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
);

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
);

namespace {

std::tuple<torch::Tensor, torch::Tensor> quantize_rows_cuda(
    const torch::Tensor& src,
    int64_t mode_id,
    bool stochastic,
    int64_t seed
) {
    TORCH_CHECK(src.is_cuda(), "quantize_rows_cuda expects a CUDA tensor");
    TORCH_CHECK(src.dim() == 2, "quantize_rows_cuda expects a 2D tensor");
    TORCH_CHECK(mode_id == 0 || mode_id == 1, "quantize_rows_cuda mode_id must be 0 (fp8) or 1 (int8)");

    auto src_c = src.contiguous();
    int64_t rows = src_c.size(0);
    int64_t cols = src_c.size(1);

    auto q_dtype = (mode_id == 0) ? torch::kFloat8_e4m3fn : torch::kInt8;
    auto q_data = torch::empty({rows, cols}, src_c.options().dtype(q_dtype));
    auto row_scale = torch::empty({rows}, src_c.options().dtype(torch::kFloat32));

    if (rows == 0 || cols == 0) {
        row_scale.fill_(1.0f);
        return {q_data, row_scale};
    }

    int src_dtype_id = 0;
    torch::Tensor src_work = src_c;
    if (src_c.scalar_type() == torch::kFloat32) {
        src_dtype_id = 0;
    } else if (src_c.scalar_type() == torch::kBFloat16) {
        src_dtype_id = 1;
    } else if (src_c.scalar_type() == torch::kFloat16) {
        src_dtype_id = 2;
    } else {
        src_work = src_c.to(torch::kFloat32);
        src_dtype_id = 0;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(src_work.device().index()).stream();
    launch_quantize_rows_cuda(
        src_work.data_ptr(),
        src_dtype_id,
        q_data.data_ptr(),
        row_scale.data_ptr<float>(),
        rows,
        cols,
        static_cast<int>(mode_id),
        stochastic,
        static_cast<uint64_t>(seed),
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q_data, row_scale};
}

int64_t shy_prune_rows_cuda(
    torch::Tensor q_data,
    torch::Tensor row_scale,
    double downscale,
    double eps,
    bool zero_diag
) {
    TORCH_CHECK(q_data.is_cuda() && row_scale.is_cuda(), "shy_prune_rows_cuda expects CUDA tensors");
    TORCH_CHECK(q_data.dim() == 2 && row_scale.dim() == 1, "Invalid dimensions for shy_prune_rows_cuda");
    TORCH_CHECK(q_data.is_contiguous() && row_scale.is_contiguous(), "shy_prune_rows_cuda expects contiguous tensors");

    int64_t rows = q_data.size(0);
    int64_t cols = q_data.size(1);
    if (rows == 0 || cols == 0) {
        if (rows > 0) {
            row_scale.mul_(static_cast<float>(downscale)).clamp_min_(1e-8f);
        }
        return 0;
    }

    int mode_id = (q_data.scalar_type() == torch::kFloat8_e4m3fn) ? 0 : 1;
    TORCH_CHECK(
        q_data.scalar_type() == torch::kFloat8_e4m3fn || q_data.scalar_type() == torch::kInt8,
        "shy_prune_rows_cuda expects float8_e4m3fn or int8 q_data"
    );

    auto pruned_counter = torch::zeros({1}, q_data.options().dtype(torch::kInt64));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q_data.device().index()).stream();
    launch_shy_prune_rows_cuda(
        q_data.data_ptr(),
        row_scale.data_ptr<float>(),
        rows,
        cols,
        mode_id,
        static_cast<float>(downscale),
        static_cast<float>(eps),
        zero_diag,
        reinterpret_cast<unsigned long long*>(pruned_counter.data_ptr<int64_t>()),
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return pruned_counter.item<int64_t>();
}

std::tuple<torch::Tensor, torch::Tensor> peer_forward_cuda(
    const torch::Tensor& x_flat,
    const torch::Tensor& slot_ids,
    const torch::Tensor& weights,
    const torch::Tensor& cache_u,
    const torch::Tensor& cache_v
) {
    TORCH_CHECK(x_flat.is_cuda() && slot_ids.is_cuda() && weights.is_cuda(), "peer_forward_cuda expects CUDA inputs");
    TORCH_CHECK(cache_u.is_cuda() && cache_v.is_cuda(), "peer_forward_cuda expects CUDA cache");

    auto x_c = x_flat.contiguous();
    auto slots_c = slot_ids.contiguous();
    auto w_c = weights.to(torch::kFloat32).contiguous();
    auto cu_c = cache_u.contiguous();
    auto cv_c = cache_v.contiguous();

    int64_t n_tokens = x_c.size(0);
    int64_t dim = x_c.size(1);
    int64_t k_tot = slots_c.size(1);

    auto out = torch::empty({n_tokens, dim}, x_c.options());
    auto pre_act = torch::empty({n_tokens, k_tot}, x_c.options().dtype(torch::kFloat32));
    if (n_tokens == 0 || k_tot == 0) {
        out.zero_();
        return {out, pre_act};
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x_c.device().index()).stream();
    launch_peer_forward_cuda(
        x_c.data_ptr(),
        x_c.scalar_type() == torch::kBFloat16,
        slots_c.data_ptr(),
        slots_c.scalar_type() == torch::kInt64,
        w_c.data_ptr<float>(),
        cu_c.data_ptr(),
        cv_c.data_ptr(),
        cu_c.scalar_type() == torch::kBFloat16,
        out.data_ptr(),
        pre_act.data_ptr<float>(),
        n_tokens,
        k_tot,
        dim,
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, pre_act};
}

std::tuple<torch::Tensor, torch::Tensor> peer_backward_update_cuda(
    const torch::Tensor& grad_out,
    const torch::Tensor& x_flat,
    const torch::Tensor& slot_ids,
    const torch::Tensor& weights,
    const torch::Tensor& pre_act,
    torch::Tensor cache_u,
    torch::Tensor cache_v,
    torch::Tensor slot_dirty,
    double lr,
    double weight_decay,
    bool do_update,
    int64_t seed
) {
    TORCH_CHECK(grad_out.is_cuda() && x_flat.is_cuda() && slot_ids.is_cuda(), "peer_backward_update_cuda expects CUDA tensors");
    TORCH_CHECK(cache_u.is_contiguous() && cache_v.is_contiguous() && slot_dirty.is_contiguous(), "Cache tensors must be contiguous");

    auto gy_c = grad_out.to(x_flat.scalar_type()).contiguous();
    auto x_c = x_flat.contiguous();
    auto slots_c = slot_ids.contiguous();
    auto w_c = weights.to(torch::kFloat32).contiguous();
    auto pre_c = pre_act.to(torch::kFloat32).contiguous();

    int64_t n_tokens = x_c.size(0);
    int64_t dim = x_c.size(1);
    int64_t k_tot = slots_c.size(1);
    int64_t num_slots = cache_u.size(0);

    auto grad_x = torch::empty({n_tokens, dim}, x_c.options());
    auto grad_weights = torch::empty({n_tokens, k_tot}, x_c.options().dtype(torch::kFloat32));
    if (n_tokens == 0 || k_tot == 0) {
        grad_x.zero_();
        grad_weights.zero_();
        return {grad_x, grad_weights};
    }

    bool active_update = do_update && (lr > 0.0) && (num_slots > 0);
    torch::Tensor g_u, g_v, slot_touched;
    float* gu_ptr = nullptr;
    float* gv_ptr = nullptr;
    int32_t* touched_ptr = nullptr;
    if (active_update) {
        g_u = torch::zeros({num_slots, dim}, x_c.options().dtype(torch::kFloat32));
        g_v = torch::zeros({num_slots, dim}, x_c.options().dtype(torch::kFloat32));
        slot_touched = torch::zeros({num_slots}, x_c.options().dtype(torch::kInt32));
        gu_ptr = g_u.data_ptr<float>();
        gv_ptr = g_v.data_ptr<float>();
        touched_ptr = slot_touched.data_ptr<int32_t>();
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x_c.device().index()).stream();
    launch_peer_backward_update_cuda(
        gy_c.data_ptr(),
        x_c.data_ptr(),
        x_c.scalar_type() == torch::kBFloat16,
        slots_c.data_ptr(),
        slots_c.scalar_type() == torch::kInt64,
        w_c.data_ptr<float>(),
        pre_c.data_ptr<float>(),
        cache_u.data_ptr(),
        cache_v.data_ptr(),
        cache_u.scalar_type() == torch::kBFloat16,
        slot_dirty.data_ptr<bool>(),
        grad_x.data_ptr(),
        grad_weights.data_ptr<float>(),
        gu_ptr,
        gv_ptr,
        touched_ptr,
        n_tokens,
        k_tot,
        num_slots,
        dim,
        static_cast<float>(lr),
        static_cast<float>(weight_decay),
        active_update,
        static_cast<uint64_t>(seed),
        stream
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_x, grad_weights};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantize_rows_cuda", &quantize_rows_cuda, "Fused FP8/INT8 row quantization CUDA kernel");
    m.def("shy_prune_rows_cuda", &shy_prune_rows_cuda, "Fused SHY scaling & pruning CUDA kernel");
    m.def("peer_forward_cuda", &peer_forward_cuda, "Fused PEER rank-1 micro-expert forward CUDA kernel");
    m.def("peer_backward_update_cuda", &peer_backward_update_cuda, "Fused PEER backward + stateless RMS-SGD CUDA kernel");
    m.def("peer_cache_ensure_cpp", &peer_cache_ensure_cpp, "C++ LRU cache manager with cudaMemcpyAsync");
    m.def("peer_flush_dirty_cpp", &peer_flush_dirty_cpp, "C++ dirty slot flush to Pinned RAM");
}
