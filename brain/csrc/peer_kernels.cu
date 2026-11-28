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

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffffu, val, offset);
    }
    return __shfl_sync(0xffffffffu, val, 0);
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

template <typename T>
__device__ __forceinline__ bool is_vec16_aligned(const T* ptr, int64_t dim) {
    constexpr int64_t kElems = 16 / sizeof(T);
    return (dim % kElems == 0) && ((reinterpret_cast<uintptr_t>(ptr) & 0xFu) == 0);
}

__device__ __forceinline__ void load_bf16x8(const __nv_bfloat16* ptr, float out[8]) {
    uint4 raw = *reinterpret_cast<const uint4*>(ptr);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f2 = __bfloat1622float2(b2[i]);
        out[2 * i] = f2.x;
        out[2 * i + 1] = f2.y;
    }
}

__device__ __forceinline__ void load_f32x4(const float* ptr, float out[4]) {
    float4 raw = *reinterpret_cast<const float4*>(ptr);
    out[0] = raw.x;
    out[1] = raw.y;
    out[2] = raw.z;
    out[3] = raw.w;
}

__device__ __forceinline__ void store_bf16x8(__nv_bfloat16* ptr, const float val[8]) {
    uint4 raw;
    __nv_bfloat162* b2 = reinterpret_cast<__nv_bfloat162*>(&raw);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        b2[i] = __floats2bfloat162_rn(val[2 * i], val[2 * i + 1]);
    }
    *reinterpret_cast<uint4*>(ptr) = raw;
}

__device__ __forceinline__ void store_f32x4(float* ptr, const float val[4]) {
    float4 raw = make_float4(val[0], val[1], val[2], val[3]);
    *reinterpret_cast<float4*>(ptr) = raw;
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
    int smem_mode  // 2: warp-private smem (dim<=1024), 1: shared atomic smem (dim<=4096), 0: global fallback
) {
    extern __shared__ float smem[];
    int64_t n = static_cast<int64_t>(blockIdx.x);
    if (n >= n_tokens) return;
    int tid = threadIdx.x;
    int lane = tid & (kWarpSize - 1);
    int wid = tid >> 5;

    const XT* x_row = x_flat + n * dim;
    XT* out_row = out + n * dim;

    float* x_sh = (smem_mode >= 1) ? smem : nullptr;
    float* warp_out = (smem_mode == 2) ? (smem + dim) : nullptr;
    float* out_sh = (smem_mode == 1) ? (smem + dim) : nullptr;

    if (smem_mode == 2) {
        for (int64_t d = tid; d < dim; d += kThreads) {
            x_sh[d] = to_f32<XT>(x_row[d]);
        }
        int64_t total_w = static_cast<int64_t>(kWarps) * dim;
        for (int64_t idx = tid; idx < total_w; idx += kThreads) {
            warp_out[idx] = 0.0f;
        }
        __syncthreads();
    } else if (smem_mode == 1) {
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

    const bool vec_c_bf16 = IsBf16<CT>::value && is_vec16_aligned(cache_u, dim) && is_vec16_aligned(cache_v, dim);
    const bool vec_c_f32 = (!IsBf16<CT>::value) && is_vec16_aligned(cache_u, dim) && is_vec16_aligned(cache_v, dim);
    float* my_wout = (smem_mode == 2) ? (warp_out + static_cast<int64_t>(wid) * dim) : nullptr;

    // Warp-parallel expert loop: ZERO __syncthreads() inside loop!
    for (int64_t k = wid; k < k_tot; k += kWarps) {
        int64_t slot = static_cast<int64_t>(slot_ids[n * k_tot + k]);
        const CT* u_row = cache_u + slot * dim;
        const CT* v_row = cache_v + slot * dim;

        float dot = 0.0f;
        if (vec_c_bf16 && (smem_mode >= 1)) {
            int64_t d_vec = dim >> 3;
            for (int64_t dv = lane; dv < d_vec; dv += kWarpSize) {
                int64_t d = dv << 3;
                float u_vals[8];
                load_bf16x8(reinterpret_cast<const __nv_bfloat16*>(u_row + d), u_vals);
                float4 x0 = *reinterpret_cast<const float4*>(x_sh + d);
                float4 x1 = *reinterpret_cast<const float4*>(x_sh + d + 4);
                dot += x0.x * u_vals[0] + x0.y * u_vals[1] + x0.z * u_vals[2] + x0.w * u_vals[3]
                     + x1.x * u_vals[4] + x1.y * u_vals[5] + x1.z * u_vals[6] + x1.w * u_vals[7];
            }
        } else if (vec_c_f32 && (smem_mode >= 1)) {
            int64_t d_vec = dim >> 2;
            for (int64_t dv = lane; dv < d_vec; dv += kWarpSize) {
                int64_t d = dv << 2;
                float u_vals[4];
                load_f32x4(reinterpret_cast<const float*>(u_row + d), u_vals);
                float4 x0 = *reinterpret_cast<const float4*>(x_sh + d);
                dot += x0.x * u_vals[0] + x0.y * u_vals[1] + x0.z * u_vals[2] + x0.w * u_vals[3];
            }
        } else {
            for (int64_t d = lane; d < dim; d += kWarpSize) {
                float xv = (smem_mode >= 1) ? x_sh[d] : to_f32<XT>(x_row[d]);
                dot += xv * to_f32<CT>(u_row[d]);
            }
        }

        float pre = warp_reduce_sum(dot);
        if (lane == 0) {
            pre_act[n * k_tot + k] = pre;
        }

        float sig = 1.0f / (1.0f + expf(-pre));
        float act = pre * sig;
        float aw = weights[n * k_tot + k] * act;

        if (smem_mode == 2) {
            if (vec_c_bf16) {
                int64_t d_vec = dim >> 3;
                for (int64_t dv = lane; dv < d_vec; dv += kWarpSize) {
                    int64_t d = dv << 3;
                    float v_vals[8];
                    load_bf16x8(reinterpret_cast<const __nv_bfloat16*>(v_row + d), v_vals);
                    float4 o0 = *reinterpret_cast<const float4*>(my_wout + d);
                    float4 o1 = *reinterpret_cast<const float4*>(my_wout + d + 4);
                    o0.x += aw * v_vals[0];
                    o0.y += aw * v_vals[1];
                    o0.z += aw * v_vals[2];
                    o0.w += aw * v_vals[3];
                    o1.x += aw * v_vals[4];
                    o1.y += aw * v_vals[5];
                    o1.z += aw * v_vals[6];
                    o1.w += aw * v_vals[7];
                    *reinterpret_cast<float4*>(my_wout + d) = o0;
                    *reinterpret_cast<float4*>(my_wout + d + 4) = o1;
                }
            } else if (vec_c_f32) {
                int64_t d_vec = dim >> 2;
                for (int64_t dv = lane; dv < d_vec; dv += kWarpSize) {
                    int64_t d = dv << 2;
                    float v_vals[4];
                    load_f32x4(reinterpret_cast<const float*>(v_row + d), v_vals);
                    float4 o0 = *reinterpret_cast<const float4*>(my_wout + d);
                    o0.x += aw * v_vals[0];
                    o0.y += aw * v_vals[1];
                    o0.z += aw * v_vals[2];
                    o0.w += aw * v_vals[3];
                    *reinterpret_cast<float4*>(my_wout + d) = o0;
                }
            } else {
                for (int64_t d = lane; d < dim; d += kWarpSize) {
                    my_wout[d] += aw * to_f32<CT>(v_row[d]);
                }
            }
        } else if (smem_mode == 1) {
            for (int64_t d = lane; d < dim; d += kWarpSize) {
                atomicAdd(&out_sh[d], aw * to_f32<CT>(v_row[d]));
            }
        } else {
            // Global fallback: serialize writes across warps via atomicAdd if float, else loop safely
            if constexpr (!IsBf16<XT>::value) {
                for (int64_t d = lane; d < dim; d += kWarpSize) {
                    atomicAdd(reinterpret_cast<float*>(out_row + d), aw * to_f32<CT>(v_row[d]));
                }
            }
        }
    }

    if (smem_mode == 2) {
        __syncthreads();
        if (IsBf16<XT>::value && is_vec16_aligned(out_row, dim)) {
            int64_t d_vec = dim >> 3;
            for (int64_t dv = tid; dv < d_vec; dv += kThreads) {
                int64_t d = dv << 3;
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
                #pragma unroll
                for (int w = 0; w < kWarps; ++w) {
                    const float* wp = warp_out + static_cast<int64_t>(w) * dim + d;
                    float4 a0 = *reinterpret_cast<const float4*>(wp);
                    float4 a1 = *reinterpret_cast<const float4*>(wp + 4);
                    acc0 += a0.x; acc1 += a0.y; acc2 += a0.z; acc3 += a0.w;
                    acc4 += a1.x; acc5 += a1.y; acc6 += a1.z; acc7 += a1.w;
                }
                float vals[8] = {acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7};
                store_bf16x8(reinterpret_cast<__nv_bfloat16*>(out_row + d), vals);
            }
        } else if ((!IsBf16<XT>::value) && is_vec16_aligned(out_row, dim)) {
            int64_t d_vec = dim >> 2;
            for (int64_t dv = tid; dv < d_vec; dv += kThreads) {
                int64_t d = dv << 2;
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                #pragma unroll
                for (int w = 0; w < kWarps; ++w) {
                    float4 a0 = *reinterpret_cast<const float4*>(warp_out + static_cast<int64_t>(w) * dim + d);
                    acc0 += a0.x; acc1 += a0.y; acc2 += a0.z; acc3 += a0.w;
                }
                float vals[4] = {acc0, acc1, acc2, acc3};
                store_f32x4(reinterpret_cast<float*>(out_row + d), vals);
            }
        } else {
            for (int64_t d = tid; d < dim; d += kThreads) {
                float acc = 0.0f;
                #pragma unroll
                for (int w = 0; w < kWarps; ++w) {
                    acc += warp_out[static_cast<int64_t>(w) * dim + d];
                }
                out_row[d] = from_f32<XT>(acc);
            }
        }
    } else if (smem_mode == 1) {
        __syncthreads();
        for (int64_t d = tid; d < dim; d += kThreads) {
            out_row[d] = from_f32<XT>(out_sh[d]);
        }
    } else if constexpr (IsBf16<XT>::value) {
        // Rare dim > 4096 bf16 global fallback: warp 0 sequentializes
        __syncthreads();
        if (wid == 0) {
            for (int64_t k = 0; k < k_tot; ++k) {
                int64_t slot = static_cast<int64_t>(slot_ids[n * k_tot + k]);
                const CT* v_row = cache_v + slot * dim;
                float pre = pre_act[n * k_tot + k];
                float sig = 1.0f / (1.0f + expf(-pre));
                float aw = weights[n * k_tot + k] * (pre * sig);
                for (int64_t d = lane; d < dim; d += kWarpSize) {
                    out_row[d] = from_f32<XT>(to_f32<XT>(out_row[d]) + aw * to_f32<CT>(v_row[d]));
                }
            }
        }
    }
}

template <typename XT, typename CT, typename SlotT>
__global__ void peer_backward_grad_kernel(
    const XT* __restrict__ grad_out,
    const XT* __restrict__ x_flat,
    const SlotT* __restrict__ slot_ids,
    const int64_t* __restrict__ inv_slot_ids,
    const float* __restrict__ weights,
    const float* __restrict__ pre_act,
    const CT* __restrict__ cache_u,
    const CT* __restrict__ cache_v,
    XT* __restrict__ grad_x,
    float* __restrict__ grad_weights,
    float* __restrict__ g_u,
    float* __restrict__ g_v,
    int64_t n_tokens,
    int64_t k_tot,
    int64_t dim,
    bool do_update,
    int smem_mode  // 2: warp-private gx (dim<=1024), 1: shared atomic gx (dim<=3072), 0: global fallback
) {
    extern __shared__ float smem[];
    int64_t n = static_cast<int64_t>(blockIdx.x);
    if (n >= n_tokens) return;
    int tid = threadIdx.x;
    int lane = tid & (kWarpSize - 1);
    int wid = tid >> 5;

    const XT* x_row = x_flat + n * dim;
    const XT* gy_row = grad_out + n * dim;
    XT* gx_row = grad_x + n * dim;

    float* x_sh = (smem_mode >= 1) ? smem : nullptr;
    float* gy_sh = (smem_mode >= 1) ? (smem + dim) : nullptr;
    float* warp_gx = (smem_mode == 2) ? (smem + 2 * dim) : nullptr;
    float* gx_sh = (smem_mode == 1) ? (smem + 2 * dim) : nullptr;

    if (smem_mode == 2) {
        for (int64_t d = tid; d < dim; d += kThreads) {
            x_sh[d] = to_f32<XT>(x_row[d]);
            gy_sh[d] = to_f32<XT>(gy_row[d]);
        }
        int64_t total_w = static_cast<int64_t>(kWarps) * dim;
        for (int64_t idx = tid; idx < total_w; idx += kThreads) {
            warp_gx[idx] = 0.0f;
        }
        __syncthreads();
    } else if (smem_mode == 1) {
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

    const bool vec_c_bf16 = IsBf16<CT>::value && is_vec16_aligned(cache_u, dim) && is_vec16_aligned(cache_v, dim);
    const bool vec_c_f32 = (!IsBf16<CT>::value) && is_vec16_aligned(cache_u, dim) && is_vec16_aligned(cache_v, dim);
    float* my_wgx = (smem_mode == 2) ? (warp_gx + static_cast<int64_t>(wid) * dim) : nullptr;

    // Warp-parallel expert loop: ZERO __syncthreads() inside loop!
    for (int64_t k = wid; k < k_tot; k += kWarps) {
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
        if (vec_c_bf16 && (smem_mode >= 1)) {
            int64_t d_vec = dim >> 3;
            for (int64_t dv = lane; dv < d_vec; dv += kWarpSize) {
                int64_t d = dv << 3;
                float v_vals[8];
                load_bf16x8(reinterpret_cast<const __nv_bfloat16*>(v_row + d), v_vals);
                float4 g0 = *reinterpret_cast<const float4*>(gy_sh + d);
                float4 g1 = *reinterpret_cast<const float4*>(gy_sh + d + 4);
                dot += g0.x * v_vals[0] + g0.y * v_vals[1] + g0.z * v_vals[2] + g0.w * v_vals[3]
                     + g1.x * v_vals[4] + g1.y * v_vals[5] + g1.z * v_vals[6] + g1.w * v_vals[7];
            }
        } else if (vec_c_f32 && (smem_mode >= 1)) {
            int64_t d_vec = dim >> 2;
            for (int64_t dv = lane; dv < d_vec; dv += kWarpSize) {
                int64_t d = dv << 2;
                float v_vals[4];
                load_f32x4(reinterpret_cast<const float*>(v_row + d), v_vals);
                float4 g0 = *reinterpret_cast<const float4*>(gy_sh + d);
                dot += g0.x * v_vals[0] + g0.y * v_vals[1] + g0.z * v_vals[2] + g0.w * v_vals[3];
            }
        } else {
            for (int64_t d = lane; d < dim; d += kWarpSize) {
                float gyv = (smem_mode >= 1) ? gy_sh[d] : to_f32<XT>(gy_row[d]);
                dot += gyv * to_f32<CT>(v_row[d]);
            }
        }

        float d_c = warp_reduce_sum(dot);
        if (lane == 0) {
            grad_weights[nk] = d_c * act;
        }

        float dp = d_c * w_k * silu_g;
        float wa = w_k * act;
        int64_t u_off = do_update ? (inv_slot_ids[nk] * dim) : 0;

        if (smem_mode == 2) {
            if (vec_c_bf16) {
                int64_t d_vec = dim >> 3;
                for (int64_t dv = lane; dv < d_vec; dv += kWarpSize) {
                    int64_t d = dv << 3;
                    float u_vals[8];
                    load_bf16x8(reinterpret_cast<const __nv_bfloat16*>(u_row + d), u_vals);
                    float4 gx0 = *reinterpret_cast<const float4*>(my_wgx + d);
                    float4 gx1 = *reinterpret_cast<const float4*>(my_wgx + d + 4);
                    gx0.x += dp * u_vals[0]; gx0.y += dp * u_vals[1]; gx0.z += dp * u_vals[2]; gx0.w += dp * u_vals[3];
                    gx1.x += dp * u_vals[4]; gx1.y += dp * u_vals[5]; gx1.z += dp * u_vals[6]; gx1.w += dp * u_vals[7];
                    *reinterpret_cast<float4*>(my_wgx + d) = gx0;
                    *reinterpret_cast<float4*>(my_wgx + d + 4) = gx1;

                    if (do_update) {
                        float4 x0 = *reinterpret_cast<const float4*>(x_sh + d);
                        float4 x1 = *reinterpret_cast<const float4*>(x_sh + d + 4);
                        float4 g0 = *reinterpret_cast<const float4*>(gy_sh + d);
                        float4 g1 = *reinterpret_cast<const float4*>(gy_sh + d + 4);
                        atomicAdd(&g_u[u_off + d + 0], dp * x0.x);
                        atomicAdd(&g_u[u_off + d + 1], dp * x0.y);
                        atomicAdd(&g_u[u_off + d + 2], dp * x0.z);
                        atomicAdd(&g_u[u_off + d + 3], dp * x0.w);
                        atomicAdd(&g_u[u_off + d + 4], dp * x1.x);
                        atomicAdd(&g_u[u_off + d + 5], dp * x1.y);
                        atomicAdd(&g_u[u_off + d + 6], dp * x1.z);
                        atomicAdd(&g_u[u_off + d + 7], dp * x1.w);

                        atomicAdd(&g_v[u_off + d + 0], wa * g0.x);
                        atomicAdd(&g_v[u_off + d + 1], wa * g0.y);
                        atomicAdd(&g_v[u_off + d + 2], wa * g0.z);
                        atomicAdd(&g_v[u_off + d + 3], wa * g0.w);
                        atomicAdd(&g_v[u_off + d + 4], wa * g1.x);
                        atomicAdd(&g_v[u_off + d + 5], wa * g1.y);
                        atomicAdd(&g_v[u_off + d + 6], wa * g1.z);
                        atomicAdd(&g_v[u_off + d + 7], wa * g1.w);
                    }
                }
            } else if (vec_c_f32) {
                int64_t d_vec = dim >> 2;
                for (int64_t dv = lane; dv < d_vec; dv += kWarpSize) {
                    int64_t d = dv << 2;
                    float u_vals[4];
                    load_f32x4(reinterpret_cast<const float*>(u_row + d), u_vals);
                    float4 gx0 = *reinterpret_cast<const float4*>(my_wgx + d);
                    gx0.x += dp * u_vals[0]; gx0.y += dp * u_vals[1]; gx0.z += dp * u_vals[2]; gx0.w += dp * u_vals[3];
                    *reinterpret_cast<float4*>(my_wgx + d) = gx0;

                    if (do_update) {
                        float4 x0 = *reinterpret_cast<const float4*>(x_sh + d);
                        float4 g0 = *reinterpret_cast<const float4*>(gy_sh + d);
                        atomicAdd(&g_u[u_off + d + 0], dp * x0.x);
                        atomicAdd(&g_u[u_off + d + 1], dp * x0.y);
                        atomicAdd(&g_u[u_off + d + 2], dp * x0.z);
                        atomicAdd(&g_u[u_off + d + 3], dp * x0.w);
                        atomicAdd(&g_v[u_off + d + 0], wa * g0.x);
                        atomicAdd(&g_v[u_off + d + 1], wa * g0.y);
                        atomicAdd(&g_v[u_off + d + 2], wa * g0.z);
                        atomicAdd(&g_v[u_off + d + 3], wa * g0.w);
                    }
                }
            } else {
                for (int64_t d = lane; d < dim; d += kWarpSize) {
                    my_wgx[d] += dp * to_f32<CT>(u_row[d]);
                    if (do_update) {
                        atomicAdd(&g_u[u_off + d], dp * x_sh[d]);
                        atomicAdd(&g_v[u_off + d], wa * gy_sh[d]);
                    }
                }
            }
        } else if (smem_mode == 1) {
            for (int64_t d = lane; d < dim; d += kWarpSize) {
                atomicAdd(&gx_sh[d], dp * to_f32<CT>(u_row[d]));
                if (do_update) {
                    atomicAdd(&g_u[u_off + d], dp * x_sh[d]);
                    atomicAdd(&g_v[u_off + d], wa * gy_sh[d]);
                }
            }
        } else {
            for (int64_t d = lane; d < dim; d += kWarpSize) {
                if constexpr (!IsBf16<XT>::value) {
                    atomicAdd(reinterpret_cast<float*>(gx_row + d), dp * to_f32<CT>(u_row[d]));
                }
                if (do_update) {
                    atomicAdd(&g_u[u_off + d], dp * to_f32<XT>(x_row[d]));
                    atomicAdd(&g_v[u_off + d], wa * to_f32<XT>(gy_row[d]));
                }
            }
        }
    }

    if (smem_mode == 2) {
        __syncthreads();
        if (IsBf16<XT>::value && is_vec16_aligned(gx_row, dim)) {
            int64_t d_vec = dim >> 3;
            for (int64_t dv = tid; dv < d_vec; dv += kThreads) {
                int64_t d = dv << 3;
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
                #pragma unroll
                for (int w = 0; w < kWarps; ++w) {
                    const float* wp = warp_gx + static_cast<int64_t>(w) * dim + d;
                    float4 a0 = *reinterpret_cast<const float4*>(wp);
                    float4 a1 = *reinterpret_cast<const float4*>(wp + 4);
                    acc0 += a0.x; acc1 += a0.y; acc2 += a0.z; acc3 += a0.w;
                    acc4 += a1.x; acc5 += a1.y; acc6 += a1.z; acc7 += a1.w;
                }
                float vals[8] = {acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7};
                store_bf16x8(reinterpret_cast<__nv_bfloat16*>(gx_row + d), vals);
            }
        } else if ((!IsBf16<XT>::value) && is_vec16_aligned(gx_row, dim)) {
            int64_t d_vec = dim >> 2;
            for (int64_t dv = tid; dv < d_vec; dv += kThreads) {
                int64_t d = dv << 2;
                float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
                #pragma unroll
                for (int w = 0; w < kWarps; ++w) {
                    float4 a0 = *reinterpret_cast<const float4*>(warp_gx + static_cast<int64_t>(w) * dim + d);
                    acc0 += a0.x; acc1 += a0.y; acc2 += a0.z; acc3 += a0.w;
                }
                float vals[4] = {acc0, acc1, acc2, acc3};
                store_f32x4(reinterpret_cast<float*>(gx_row + d), vals);
            }
        } else {
            for (int64_t d = tid; d < dim; d += kThreads) {
                float acc = 0.0f;
                #pragma unroll
                for (int w = 0; w < kWarps; ++w) {
                    acc += warp_gx[static_cast<int64_t>(w) * dim + d];
                }
                gx_row[d] = from_f32<XT>(acc);
            }
        }
    } else if (smem_mode == 1) {
        __syncthreads();
        for (int64_t d = tid; d < dim; d += kThreads) {
            gx_row[d] = from_f32<XT>(gx_sh[d]);
        }
    } else if constexpr (IsBf16<XT>::value) {
        __syncthreads();
        if (wid == 0) {
            for (int64_t k = 0; k < k_tot; ++k) {
                int64_t nk = n * k_tot + k;
                int64_t slot = static_cast<int64_t>(slot_ids[nk]);
                const CT* u_row = cache_u + slot * dim;
                float pre = pre_act[nk];
                float sig = 1.0f / (1.0f + expf(-pre));
                float silu_g = sig * (1.0f + pre * (1.0f - sig));
                float dp = (grad_weights[nk] / fmaxf(fabsf(pre * sig), 1e-8f)) * weights[nk] * silu_g;
                // Recompute d_c cleanly so sign/small act stays exact
                float dot = 0.0f;
                const CT* v_row = cache_v + slot * dim;
                for (int64_t d = lane; d < dim; d += kWarpSize) {
                    dot += to_f32<XT>(gy_row[d]) * to_f32<CT>(v_row[d]);
                }
                float d_c = warp_reduce_sum(dot);
                dp = d_c * weights[nk] * silu_g;
                for (int64_t d = lane; d < dim; d += kWarpSize) {
                    gx_row[d] = from_f32<XT>(to_f32<XT>(gx_row[d]) + dp * to_f32<CT>(u_row[d]));
                }
            }
        }
    }
}

template <typename CT, typename SlotT>
__global__ void peer_stateless_rms_sgd_kernel(
    const float* __restrict__ g_u,
    const float* __restrict__ g_v,
    const SlotT* __restrict__ uniq_slots,
    CT* __restrict__ cache_u,
    CT* __restrict__ cache_v,
    bool* __restrict__ slot_dirty,
    int64_t u_slots,
    int64_t dim,
    float lr,
    float weight_decay,
    uint64_t seed
) {
    int64_t u = static_cast<int64_t>(blockIdx.x);
    if (u >= u_slots) return;
    int64_t s = static_cast<int64_t>(uniq_slots[u]);

    int tid = threadIdx.x;
    int64_t g_base = u * dim;
    int64_t c_base = s * dim;

    float sq_u = 0.0f;
    float sq_v = 0.0f;
    for (int64_t d = tid; d < dim; d += kThreads) {
        float gu = g_u[g_base + d];
        float gv = g_v[g_base + d];
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
        int64_t g_idx = g_base + d;
        int64_t c_idx = c_base + d;
        float step_u = fminf(2.0f, fmaxf(-2.0f, g_u[g_idx] / rms_u));
        float step_v = fminf(2.0f, fmaxf(-2.0f, g_v[g_idx] / rms_v));
        float u_new = to_f32<CT>(cache_u[c_idx]) * decay - lr * step_u;
        float v_new = to_f32<CT>(cache_v[c_idx]) * decay - lr * step_v;

        if constexpr (IsBf16<CT>::value) {
            cache_u[c_idx] = stochastic_round_bf16_dev(u_new, seed, static_cast<uint64_t>(c_idx) * 2ULL);
            cache_v[c_idx] = stochastic_round_bf16_dev(v_new, seed, static_cast<uint64_t>(c_idx) * 2ULL + 1ULL);
        } else {
            cache_u[c_idx] = from_f32<CT>(u_new);
            cache_v[c_idx] = from_f32<CT>(v_new);
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
    int smem_mode = (dim <= 1024) ? 2 : ((dim <= 4096) ? 1 : 0);
    size_t smem_bytes = 0;
    if (smem_mode == 2) {
        smem_bytes = static_cast<size_t>((1 + kWarps) * dim) * sizeof(float);
    } else if (smem_mode == 1) {
        smem_bytes = static_cast<size_t>(2 * dim) * sizeof(float);
    }
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
            smem_mode
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
    const void* uniq_slots,
    const int64_t* inv_slot_ids,
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
    int64_t n_tokens,
    int64_t k_tot,
    int64_t u_slots,
    int64_t dim,
    float lr,
    float weight_decay,
    bool do_update,
    uint64_t seed,
    cudaStream_t stream
) {
    if (n_tokens <= 0 || k_tot <= 0) return;
    int smem_mode = (dim <= 1024) ? 2 : ((dim <= 3072) ? 1 : 0);
    size_t smem_bytes = 0;
    if (smem_mode == 2) {
        smem_bytes = static_cast<size_t>((2 + kWarps) * dim) * sizeof(float);
    } else if (smem_mode == 1) {
        smem_bytes = static_cast<size_t>(3 * dim) * sizeof(float);
    }
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
            inv_slot_ids,
            weights,
            pre_act,
            reinterpret_cast<const CT*>(cache_u),
            reinterpret_cast<const CT*>(cache_v),
            reinterpret_cast<XT*>(grad_x),
            grad_weights,
            g_u,
            g_v,
            n_tokens,
            k_tot,
            dim,
            do_update,
            smem_mode
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

    if (do_update && u_slots > 0) {
        dim3 s_grid(static_cast<unsigned int>(u_slots));
        auto launch_sgd = [&](auto c_tag, auto s_tag) {
            using CT = decltype(c_tag);
            using ST = decltype(s_tag);
            peer_stateless_rms_sgd_kernel<CT, ST><<<s_grid, block, 0, stream>>>(
                g_u,
                g_v,
                reinterpret_cast<const ST*>(uniq_slots),
                reinterpret_cast<CT*>(cache_u),
                reinterpret_cast<CT*>(cache_v),
                slot_dirty,
                u_slots,
                dim,
                lr,
                weight_decay,
                seed
            );
        };
        if (cache_bf16 && slots_i64) {
            launch_sgd(__nv_bfloat16{}, int64_t{});
        } else if (!cache_bf16 && slots_i64) {
            launch_sgd(float{}, int64_t{});
        } else if (cache_bf16 && !slots_i64) {
            launch_sgd(__nv_bfloat16{}, int32_t{});
        } else {
            launch_sgd(float{}, int32_t{});
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
