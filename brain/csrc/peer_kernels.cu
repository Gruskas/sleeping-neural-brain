#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;

template <typename T>
struct IsBf16 {
    static constexpr bool value = false;
};

template <>
struct IsBf16<__nv_bfloat16> {
    static constexpr bool value = true;
};

template <typename T>
__device__ __forceinline__ float to_f32(T v);

template <>
__device__ __forceinline__ float to_f32<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_f32(float v);

template <>
__device__ __forceinline__ float from_f32<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ __nv_bfloat16 stochastic_round_bf16_dev(float x, uint64_t seed, uint64_t idx) {
    float clamped = fminf(3.3e38f, fmaxf(-3.3e38f, x));
    uint32_t bits = __float_as_uint(clamped);
    uint64_t z = seed + idx * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= (z >> 31);
    uint32_t rand16 = static_cast<uint32_t>(z & 0xFFFFu);
    uint32_t rounded = (bits + rand16) & 0xFFFF0000u;
    return __float2bfloat16(__uint_as_float(rounded));
}

__device__ __forceinline__ float rand_uniform_01(uint64_t seed, uint64_t idx) {
    uint64_t z = seed + idx * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= (z >> 31);
    return static_cast<float>(z >> 40) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float s_sum[kWarps];
    int tid = threadIdx.x;
    int lane = tid & (kWarpSize - 1);
    int wid = tid >> 5;

    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    if (lane == 0) {
        s_sum[wid] = val;
    }
    __syncthreads();

    float out = (tid < kWarps) ? s_sum[tid] : 0.0f;
    if (wid == 0) {
        for (int offset = kWarps >> 1; offset > 0; offset >>= 1) {
            out += __shfl_down_sync(0xffffffffu, out, offset);
        }
        if (tid == 0) {
            s_sum[0] = out;
        }
    }
    __syncthreads();
    float total = s_sum[0];
    __syncthreads();
    return total;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float s_max[kWarps];
    int tid = threadIdx.x;
    int lane = tid & (kWarpSize - 1);
    int wid = tid >> 5;

    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffffu, val, offset));
    }
    if (lane == 0) {
        s_max[wid] = val;
    }
    __syncthreads();

    float out = (tid < kWarps) ? s_max[tid] : 0.0f;
    if (wid == 0) {
        for (int offset = kWarps >> 1; offset > 0; offset >>= 1) {
            out = fmaxf(out, __shfl_down_sync(0xffffffffu, out, offset));
        }
        if (tid == 0) {
            s_max[0] = out;
        }
    }
    __syncthreads();
    float total = s_max[0];
    __syncthreads();
    return total;
}

template <typename XT, typename CT, typename SlotT>
__global__ void peer_forward_kernel(
    const XT* __restrict__ x_flat,
    const SlotT* __restrict__ slot_ids,
    const float* __restrict__ weights,
    const CT* __restrict__ cache_u,
    const CT* __restrict__ cache_v,
    XT* __restrict__ out,
    float* __restrict__ pre_act,
    int64_t n_tokens,
    int64_t k_tot,
    int64_t dim,
    bool use_smem
) {
    extern __shared__ float smem[];
    int64_t n = static_cast<int64_t>(blockIdx.x);
    if (n >= n_tokens) return;
    int tid = threadIdx.x;

    const XT* x_row = x_flat + n * dim;
    XT* out_row = out + n * dim;

    float* x_sh = use_smem ? smem : nullptr;
    float* out_sh = use_smem ? (smem + dim) : nullptr;

    if (use_smem) {
        for (int64_t d = tid; d < dim; d += kThreads) {
            x_sh[d] = to_f32<XT>(x_row[d]);
            out_sh[d] = 0.0f;
        }
        __syncthreads();
    } else {
        for (int64_t d = tid; d < dim; d += kThreads) {
            out_row[d] = from_f32<XT>(0.0f);
        }
        __syncthreads();
    }

    for (int64_t k = 0; k < k_tot; ++k) {
        int64_t slot = static_cast<int64_t>(slot_ids[n * k_tot + k]);
        const CT* u_row = cache_u + slot * dim;
        const CT* v_row = cache_v + slot * dim;

        float dot = 0.0f;
        for (int64_t d = tid; d < dim; d += kThreads) {
            float xv = use_smem ? x_sh[d] : to_f32<XT>(x_row[d]);
            dot += xv * to_f32<CT>(u_row[d]);
        }
        float pre = block_reduce_sum(dot);
        if (tid == 0) {
            pre_act[n * k_tot + k] = pre;
        }

        float sig = 1.0f / (1.0f + expf(-pre));
        float act = pre * sig;
        float aw = weights[n * k_tot + k] * act;

        if (use_smem) {
            for (int64_t d = tid; d < dim; d += kThreads) {
                out_sh[d] += aw * to_f32<CT>(v_row[d]);
            }
        } else {
            for (int64_t d = tid; d < dim; d += kThreads) {
                float cur = to_f32<XT>(out_row[d]) + aw * to_f32<CT>(v_row[d]);
                out_row[d] = from_f32<XT>(cur);
            }
        }
    }

    if (use_smem) {
        __syncthreads();
        for (int64_t d = tid; d < dim; d += kThreads) {
            out_row[d] = from_f32<XT>(out_sh[d]);
        }
    }
}

template <typename XT, typename CT, typename SlotT>
__global__ void peer_backward_grad_kernel(
    const XT* __restrict__ grad_out,
    const XT* __restrict__ x_flat,
    const SlotT* __restrict__ slot_ids,
    const float* __restrict__ weights,
    const float* __restrict__ pre_act,
    const CT* __restrict__ cache_u,
    const CT* __restrict__ cache_v,
    XT* __restrict__ grad_x,
    float* __restrict__ grad_weights,
    float* __restrict__ g_u,
    float* __restrict__ g_v,
    int32_t* __restrict__ slot_touched,
    int64_t n_tokens,
    int64_t k_tot,
    int64_t dim,
    bool do_update,
    bool use_smem
) {
    extern __shared__ float smem[];
    int64_t n = static_cast<int64_t>(blockIdx.x);
    if (n >= n_tokens) return;
    int tid = threadIdx.x;

    const XT* x_row = x_flat + n * dim;
    const XT* gy_row = grad_out + n * dim;
    XT* gx_row = grad_x + n * dim;

    float* x_sh = use_smem ? smem : nullptr;
    float* gy_sh = use_smem ? (smem + dim) : nullptr;
    float* gx_sh = use_smem ? (smem + 2 * dim) : nullptr;

    if (use_smem) {
        for (int64_t d = tid; d < dim; d += kThreads) {
            x_sh[d] = to_f32<XT>(x_row[d]);
            gy_sh[d] = to_f32<XT>(gy_row[d]);
            gx_sh[d] = 0.0f;
        }
        __syncthreads();
    } else {
        for (int64_t d = tid; d < dim; d += kThreads) {
            gx_row[d] = from_f32<XT>(0.0f);
        }
        __syncthreads();
    }

    for (int64_t k = 0; k < k_tot; ++k) {
        int64_t nk = n * k_tot + k;
        int64_t slot = static_cast<int64_t>(slot_ids[nk]);
        const CT* u_row = cache_u + slot * dim;
        const CT* v_row = cache_v + slot * dim;

        float pre = pre_act[nk];
        float w_k = weights[nk];
        float sig = 1.0f / (1.0f + expf(-pre));
        float act = pre * sig;
        float silu_g = sig * (1.0f + pre * (1.0f - sig));

        float dot = 0.0f;
        for (int64_t d = tid; d < dim; d += kThreads) {
            float gyv = use_smem ? gy_sh[d] : to_f32<XT>(gy_row[d]);
            dot += gyv * to_f32<CT>(v_row[d]);
        }
        float d_c = block_reduce_sum(dot);

        if (tid == 0) {
            grad_weights[nk] = d_c * act;
            if (do_update) {
                slot_touched[slot] = 1;
            }
        }

        float dp = d_c * w_k * silu_g;
        float wa = w_k * act;

        for (int64_t d = tid; d < dim; d += kThreads) {
            float u_val = to_f32<CT>(u_row[d]);
            if (use_smem) {
                gx_sh[d] += dp * u_val;
            } else {
                float cur = to_f32<XT>(gx_row[d]) + dp * u_val;
                gx_row[d] = from_f32<XT>(cur);
            }
            if (do_update) {
                float xv = use_smem ? x_sh[d] : to_f32<XT>(x_row[d]);
                float gyv = use_smem ? gy_sh[d] : to_f32<XT>(gy_row[d]);
                atomicAdd(&g_u[slot * dim + d], dp * xv);
                atomicAdd(&g_v[slot * dim + d], wa * gyv);
            }
        }
    }

    if (use_smem) {
        __syncthreads();
        for (int64_t d = tid; d < dim; d += kThreads) {
            gx_row[d] = from_f32<XT>(gx_sh[d]);
        }
    }
}

template <typename CT>
__global__ void peer_stateless_rms_sgd_kernel(
    const float* __restrict__ g_u,
    const float* __restrict__ g_v,
    const int32_t* __restrict__ slot_touched,
    CT* __restrict__ cache_u,
    CT* __restrict__ cache_v,
    bool* __restrict__ slot_dirty,
    int64_t num_slots,
    int64_t dim,
    float lr,
    float weight_decay,
    uint64_t seed
) {
    int64_t s = static_cast<int64_t>(blockIdx.x);
    if (s >= num_slots) return;
    if (slot_touched[s] == 0) return;

    int tid = threadIdx.x;
    int64_t base = s * dim;

    float sq_u = 0.0f;
    float sq_v = 0.0f;
    for (int64_t d = tid; d < dim; d += kThreads) {
        float gu = g_u[base + d];
        float gv = g_v[base + d];
        sq_u += gu * gu;
        sq_v += gv * gv;
    }
    float sum_sq_u = block_reduce_sum(sq_u);
    float sum_sq_v = block_reduce_sum(sq_v);

    float inv_d = 1.0f / static_cast<float>(dim);
    float rms_u = fmaxf(sqrtf(sum_sq_u * inv_d), 1e-5f);
    float rms_v = fmaxf(sqrtf(sum_sq_v * inv_d), 1e-5f);
    float decay = 1.0f - lr * weight_decay;

    for (int64_t d = tid; d < dim; d += kThreads) {
        int64_t idx = base + d;
        float step_u = fminf(2.0f, fmaxf(-2.0f, g_u[idx] / rms_u));
        float step_v = fminf(2.0f, fmaxf(-2.0f, g_v[idx] / rms_v));
        float u_new = to_f32<CT>(cache_u[idx]) * decay - lr * step_u;
        float v_new = to_f32<CT>(cache_v[idx]) * decay - lr * step_v;

        if constexpr (IsBf16<CT>::value) {
            cache_u[idx] = stochastic_round_bf16_dev(u_new, seed, static_cast<uint64_t>(idx) * 2ULL);
            cache_v[idx] = stochastic_round_bf16_dev(v_new, seed, static_cast<uint64_t>(idx) * 2ULL + 1ULL);
        } else {
            cache_u[idx] = from_f32<CT>(u_new);
            cache_v[idx] = from_f32<CT>(v_new);
        }
    }
    if (tid == 0) {
        slot_dirty[s] = true;
    }
}

template <typename CT, typename SlotT>
__global__ void peer_quantize_evict_kernel(
    const CT* __restrict__ cache_u,
    const CT* __restrict__ cache_v,
    const SlotT* __restrict__ slots,
    int8_t* __restrict__ u_q_out,
    __half* __restrict__ u_s_out,
    int8_t* __restrict__ v_q_out,
    __half* __restrict__ v_s_out,
    int64_t n_slots,
    int64_t dim,
    uint64_t seed
) {
    int64_t r = static_cast<int64_t>(blockIdx.x);
    if (r >= n_slots) return;
    int tid = threadIdx.x;
    int64_t s = static_cast<int64_t>(slots[r]);
    int64_t src_base = s * dim;
    int64_t dst_base = r * dim;

    float max_u = 0.0f;
    float max_v = 0.0f;
    for (int64_t d = tid; d < dim; d += kThreads) {
        max_u = fmaxf(max_u, fabsf(to_f32<CT>(cache_u[src_base + d])));
        max_v = fmaxf(max_v, fabsf(to_f32<CT>(cache_v[src_base + d])));
    }
    float amax_u = block_reduce_max(max_u);
    float amax_v = block_reduce_max(max_v);

    __half hs_u = __float2half(fmaxf(amax_u / 127.0f, 1e-6f));
    __half hs_v = __float2half(fmaxf(amax_v / 127.0f, 1e-6f));
    float su = fmaxf(__half2float(hs_u), 1e-6f);
    float sv = fmaxf(__half2float(hs_v), 1e-6f);
    if (tid == 0) {
        u_s_out[r] = hs_u;
        v_s_out[r] = hs_v;
    }

    for (int64_t d = tid; d < dim; d += kThreads) {
        float fu = fminf(127.0f, fmaxf(-127.0f, to_f32<CT>(cache_u[src_base + d]) / su));
        float fv = fminf(127.0f, fmaxf(-127.0f, to_f32<CT>(cache_v[src_base + d]) / sv));

        float fl_u = floorf(fu);
        float p_u = fu - fl_u;
        float rnd_u = rand_uniform_01(seed, static_cast<uint64_t>(dst_base + d) * 2ULL);
        int qi_u = static_cast<int>(fl_u + (rnd_u < p_u ? 1.0f : 0.0f));
        if (qi_u > 127) qi_u = 127;
        if (qi_u < -127) qi_u = -127;
        u_q_out[dst_base + d] = static_cast<int8_t>(qi_u);

        float fl_v = floorf(fv);
        float p_v = fv - fl_v;
        float rnd_v = rand_uniform_01(seed, static_cast<uint64_t>(dst_base + d) * 2ULL + 1ULL);
        int qi_v = static_cast<int>(fl_v + (rnd_v < p_v ? 1.0f : 0.0f));
        if (qi_v > 127) qi_v = 127;
        if (qi_v < -127) qi_v = -127;
        v_q_out[dst_base + d] = static_cast<int8_t>(qi_v);
    }
}

template <typename CT, typename SlotT>
__global__ void peer_dequant_scatter_kernel(
    const int8_t* __restrict__ u_q,
    const __half* __restrict__ u_s,
    const int8_t* __restrict__ v_q,
    const __half* __restrict__ v_s,
    const SlotT* __restrict__ victim_slots,
    CT* __restrict__ cache_u,
    CT* __restrict__ cache_v,
    int64_t n_miss,
    int64_t dim
) {
    int64_t r = static_cast<int64_t>(blockIdx.x);
    if (r >= n_miss) return;
    int tid = threadIdx.x;
    int64_t slot = static_cast<int64_t>(victim_slots[r]);
    float su = __half2float(u_s[r]);
    float sv = __half2float(v_s[r]);
    int64_t src_base = r * dim;
    int64_t dst_base = slot * dim;

    for (int64_t d = tid; d < dim; d += kThreads) {
        cache_u[dst_base + d] = from_f32<CT>(static_cast<float>(u_q[src_base + d]) * su);
        cache_v[dst_base + d] = from_f32<CT>(static_cast<float>(v_q[src_base + d]) * sv);
    }
}

}  // namespace

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
) {
    if (n_tokens <= 0 || k_tot <= 0) return;
    bool use_smem = (dim <= 4096);
    size_t smem_bytes = use_smem ? static_cast<size_t>(2 * dim) * sizeof(float) : 0;
    dim3 grid(static_cast<unsigned int>(n_tokens));
    dim3 block(kThreads);

    auto launch = [&](auto x_tag, auto c_tag, auto s_tag) {
        using XT = decltype(x_tag);
        using CT = decltype(c_tag);
        using ST = decltype(s_tag);
        peer_forward_kernel<XT, CT, ST><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const XT*>(x_flat),
            reinterpret_cast<const ST*>(slot_ids),
            weights,
            reinterpret_cast<const CT*>(cache_u),
            reinterpret_cast<const CT*>(cache_v),
            reinterpret_cast<XT*>(out),
            pre_act,
            n_tokens,
            k_tot,
            dim,
            use_smem
        );
    };

    if (x_bf16 && cache_bf16 && slots_i64) {
        launch(__nv_bfloat16{}, __nv_bfloat16{}, int64_t{});
    } else if (!x_bf16 && !cache_bf16 && slots_i64) {
        launch(float{}, float{}, int64_t{});
    } else if (!x_bf16 && cache_bf16 && slots_i64) {
        launch(float{}, __nv_bfloat16{}, int64_t{});
    } else if (x_bf16 && !cache_bf16 && slots_i64) {
        launch(__nv_bfloat16{}, float{}, int64_t{});
    } else if (x_bf16 && cache_bf16 && !slots_i64) {
        launch(__nv_bfloat16{}, __nv_bfloat16{}, int32_t{});
    } else if (!x_bf16 && !cache_bf16 && !slots_i64) {
        launch(float{}, float{}, int32_t{});
    } else if (!x_bf16 && cache_bf16 && !slots_i64) {
        launch(float{}, __nv_bfloat16{}, int32_t{});
    } else {
        launch(__nv_bfloat16{}, float{}, int32_t{});
    }
}

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
) {
    if (n_tokens <= 0 || k_tot <= 0) return;
    bool use_smem = (dim <= 3072);
    size_t smem_bytes = use_smem ? static_cast<size_t>(3 * dim) * sizeof(float) : 0;
    dim3 grid(static_cast<unsigned int>(n_tokens));
    dim3 block(kThreads);

    auto launch_grad = [&](auto x_tag, auto c_tag, auto s_tag) {
        using XT = decltype(x_tag);
        using CT = decltype(c_tag);
        using ST = decltype(s_tag);
        peer_backward_grad_kernel<XT, CT, ST><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const XT*>(grad_out),
            reinterpret_cast<const XT*>(x_flat),
            reinterpret_cast<const ST*>(slot_ids),
            weights,
            pre_act,
            reinterpret_cast<const CT*>(cache_u),
            reinterpret_cast<const CT*>(cache_v),
            reinterpret_cast<XT*>(grad_x),
            grad_weights,
            g_u,
            g_v,
            slot_touched,
            n_tokens,
            k_tot,
            dim,
            do_update,
            use_smem
        );
    };

    if (x_bf16 && cache_bf16 && slots_i64) {
        launch_grad(__nv_bfloat16{}, __nv_bfloat16{}, int64_t{});
    } else if (!x_bf16 && !cache_bf16 && slots_i64) {
        launch_grad(float{}, float{}, int64_t{});
    } else if (!x_bf16 && cache_bf16 && slots_i64) {
        launch_grad(float{}, __nv_bfloat16{}, int64_t{});
    } else if (x_bf16 && !cache_bf16 && slots_i64) {
        launch_grad(__nv_bfloat16{}, float{}, int64_t{});
    } else if (x_bf16 && cache_bf16 && !slots_i64) {
        launch_grad(__nv_bfloat16{}, __nv_bfloat16{}, int32_t{});
    } else if (!x_bf16 && !cache_bf16 && !slots_i64) {
        launch_grad(float{}, float{}, int32_t{});
    } else if (!x_bf16 && cache_bf16 && !slots_i64) {
        launch_grad(float{}, __nv_bfloat16{}, int32_t{});
    } else {
        launch_grad(__nv_bfloat16{}, float{}, int32_t{});
    }

    if (do_update && num_slots > 0) {
        dim3 s_grid(static_cast<unsigned int>(num_slots));
        if (cache_bf16) {
            peer_stateless_rms_sgd_kernel<__nv_bfloat16><<<s_grid, block, 0, stream>>>(
                g_u,
                g_v,
                slot_touched,
                reinterpret_cast<__nv_bfloat16*>(cache_u),
                reinterpret_cast<__nv_bfloat16*>(cache_v),
                slot_dirty,
                num_slots,
                dim,
                lr,
                weight_decay,
                seed
            );
        } else {
            peer_stateless_rms_sgd_kernel<float><<<s_grid, block, 0, stream>>>(
                g_u,
                g_v,
                slot_touched,
                reinterpret_cast<float*>(cache_u),
                reinterpret_cast<float*>(cache_v),
                slot_dirty,
                num_slots,
                dim,
                lr,
                weight_decay,
                seed
            );
        }
    }
}

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
) {
    if (n_slots <= 0) return;
    dim3 grid(static_cast<unsigned int>(n_slots));
    dim3 block(kThreads);

    auto launch = [&](auto c_tag, auto s_tag) {
        using CT = decltype(c_tag);
        using ST = decltype(s_tag);
        peer_quantize_evict_kernel<CT, ST><<<grid, block, 0, stream>>>(
            reinterpret_cast<const CT*>(cache_u),
            reinterpret_cast<const CT*>(cache_v),
            reinterpret_cast<const ST*>(slots),
            u_q_out,
            reinterpret_cast<__half*>(u_s_out),
            v_q_out,
            reinterpret_cast<__half*>(v_s_out),
            n_slots,
            dim,
            seed
        );
    };

    if (cache_bf16 && slots_i64) {
        launch(__nv_bfloat16{}, int64_t{});
    } else if (!cache_bf16 && slots_i64) {
        launch(float{}, int64_t{});
    } else if (cache_bf16 && !slots_i64) {
        launch(__nv_bfloat16{}, int32_t{});
    } else {
        launch(float{}, int32_t{});
    }
}

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
) {
    if (n_miss <= 0) return;
    dim3 grid(static_cast<unsigned int>(n_miss));
    dim3 block(kThreads);

    auto launch = [&](auto c_tag, auto s_tag) {
        using CT = decltype(c_tag);
        using ST = decltype(s_tag);
        peer_dequant_scatter_kernel<CT, ST><<<grid, block, 0, stream>>>(
            u_q,
            reinterpret_cast<const __half*>(u_s),
            v_q,
            reinterpret_cast<const __half*>(v_s),
            reinterpret_cast<const ST*>(victim_slots),
            reinterpret_cast<CT*>(cache_u),
            reinterpret_cast<CT*>(cache_v),
            n_miss,
            dim
        );
    };

    if (cache_bf16 && slots_i64) {
        launch(__nv_bfloat16{}, int64_t{});
    } else if (!cache_bf16 && slots_i64) {
        launch(float{}, int64_t{});
    } else if (cache_bf16 && !slots_i64) {
        launch(__nv_bfloat16{}, int32_t{});
    } else {
        launch(float{}, int32_t{});
    }
}
