#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarps = kThreads / kWarpSize;

__device__ __forceinline__ float rand_uniform_01(uint64_t seed, uint64_t idx) {
    uint64_t z = seed + idx * 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= (z >> 31);
    return static_cast<float>(z >> 40) * (1.0f / 16777216.0f);
}

template <typename T>
__device__ __forceinline__ float load_as_f32(const T* ptr, int64_t idx);

template <>
__device__ __forceinline__ float load_as_f32<float>(const float* ptr, int64_t idx) {
    return ptr[idx];
}

template <>
__device__ __forceinline__ float load_as_f32<__nv_bfloat16>(const __nv_bfloat16* ptr, int64_t idx) {
    return __bfloat162float(ptr[idx]);
}

template <>
__device__ __forceinline__ float load_as_f32<__half>(const __half* ptr, int64_t idx) {
    return __half2float(ptr[idx]);
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

__device__ __forceinline__ int64_t block_reduce_sum_i64(int64_t val) {
    __shared__ int64_t s_sum[kWarps];
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

    int64_t out = (tid < kWarps) ? s_sum[tid] : 0;
    if (wid == 0) {
        for (int offset = kWarps >> 1; offset > 0; offset >>= 1) {
            out += __shfl_down_sync(0xffffffffu, out, offset);
        }
        if (tid == 0) {
            s_sum[0] = out;
        }
    }
    __syncthreads();
    int64_t total = s_sum[0];
    __syncthreads();
    return total;
}

template <typename SrcT, int ModeId>
__global__ void quantize_rows_kernel(
    const SrcT* __restrict__ src,
    uint8_t* __restrict__ q_out,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols,
    bool stochastic,
    uint64_t seed
) {
    int64_t r = static_cast<int64_t>(blockIdx.x);
    if (r >= rows) return;
    int tid = threadIdx.x;
    int64_t base = r * cols;

    float local_max = 0.0f;
    for (int64_t c = tid; c < cols; c += kThreads) {
        float v = load_as_f32<SrcT>(src, base + c);
        if (!isfinite(v)) v = 0.0f;
        local_max = fmaxf(local_max, fabsf(v));
    }
    float row_amax = block_reduce_max(local_max);

    constexpr float kMaxVal = (ModeId == 0) ? 448.0f : 127.0f;
    float sc = fmaxf(row_amax / kMaxVal, 1e-8f);
    if (tid == 0) {
        row_scale[r] = sc;
    }

    for (int64_t c = tid; c < cols; c += kThreads) {
        int64_t idx = base + c;
        float v = load_as_f32<SrcT>(src, idx);
        if (!isfinite(v)) v = 0.0f;
        if (v == 0.0f) {
            q_out[idx] = 0;
            continue;
        }
        float s = fminf(kMaxVal, fmaxf(-kMaxVal, v / sc));
        if constexpr (ModeId == 0) {
            if (stochastic) {
                float abs_s = fmaxf(fabsf(s), 0.015625f);  // 2^-6
                float exp_f = floorf(log2f(abs_s));
                float ulp = fmaxf(ldexpf(1.0f, static_cast<int>(exp_f) - 3), 0.001953125f);  // 2^-9
                float u = rand_uniform_01(seed, static_cast<uint64_t>(idx));
                s = fminf(448.0f, fmaxf(-448.0f, s + (u - 0.5f) * ulp));
            }
            __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(s, __NV_SATFINITE, __NV_E4M3);
            q_out[idx] = static_cast<uint8_t>(q);
        } else {
            if (stochastic) {
                float u = rand_uniform_01(seed, static_cast<uint64_t>(idx));
                s = fminf(127.0f, fmaxf(-127.0f, s + (u - 0.5f)));
            }
            int q_i = static_cast<int>(nearbyintf(s));
            if (q_i > 127) q_i = 127;
            if (q_i < -127) q_i = -127;
            q_out[idx] = static_cast<uint8_t>(static_cast<int8_t>(q_i));
        }
    }
}

template <int ModeId>
__global__ void shy_prune_rows_kernel(
    uint8_t* __restrict__ q_data,
    float* __restrict__ row_scale,
    int64_t rows,
    int64_t cols,
    float downscale,
    float eps,
    bool zero_diag,
    unsigned long long* __restrict__ pruned_counter
) {
    int64_t r = static_cast<int64_t>(blockIdx.x);
    if (r >= rows) return;
    int tid = threadIdx.x;

    __shared__ float s_scale;
    if (tid == 0) {
        float sc = fmaxf(row_scale[r] * downscale, 1e-8f);
        row_scale[r] = sc;
        s_scale = sc;
    }
    __syncthreads();
    float sc = s_scale;

    int64_t base = r * cols;
    int64_t local_pruned = 0;

    for (int64_t c = tid; c < cols; c += kThreads) {
        int64_t idx = base + c;
        uint8_t raw = q_data[idx];
        if constexpr (ModeId == 0) {
            if ((raw & 0x7Fu) == 0) continue;
            __half_raw hr = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(raw), __NV_E4M3);
            float val = __half2float(*reinterpret_cast<__half*>(&hr));
            bool is_diag = zero_diag && (r == c);
            if (is_diag || fabsf(val * sc) < eps) {
                q_data[idx] = 0;
                local_pruned += 1;
            }
        } else {
            int8_t sraw = static_cast<int8_t>(raw);
            if (sraw == 0) continue;
            float val = static_cast<float>(sraw);
            bool is_diag = zero_diag && (r == c);
            if (is_diag || fabsf(val * sc) < eps) {
                q_data[idx] = 0;
                local_pruned += 1;
            }
        }
    }

    int64_t blk_pruned = block_reduce_sum_i64(local_pruned);
    if (tid == 0 && blk_pruned > 0) {
        atomicAdd(pruned_counter, static_cast<unsigned long long>(blk_pruned));
    }
}

}  // namespace

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
) {
    if (rows <= 0 || cols <= 0) return;
    dim3 grid(static_cast<unsigned int>(rows));
    dim3 block(kThreads);
    uint8_t* q_ptr = reinterpret_cast<uint8_t*>(q_out);

    if (src_dtype_id == 0) {
        const float* s_ptr = reinterpret_cast<const float*>(src);
        if (mode_id == 0) {
            quantize_rows_kernel<float, 0><<<grid, block, 0, stream>>>(s_ptr, q_ptr, row_scale, rows, cols, stochastic, seed);
        } else {
            quantize_rows_kernel<float, 1><<<grid, block, 0, stream>>>(s_ptr, q_ptr, row_scale, rows, cols, stochastic, seed);
        }
    } else if (src_dtype_id == 1) {
        const __nv_bfloat16* s_ptr = reinterpret_cast<const __nv_bfloat16*>(src);
        if (mode_id == 0) {
            quantize_rows_kernel<__nv_bfloat16, 0><<<grid, block, 0, stream>>>(s_ptr, q_ptr, row_scale, rows, cols, stochastic, seed);
        } else {
            quantize_rows_kernel<__nv_bfloat16, 1><<<grid, block, 0, stream>>>(s_ptr, q_ptr, row_scale, rows, cols, stochastic, seed);
        }
    } else {
        const __half* s_ptr = reinterpret_cast<const __half*>(src);
        if (mode_id == 0) {
            quantize_rows_kernel<__half, 0><<<grid, block, 0, stream>>>(s_ptr, q_ptr, row_scale, rows, cols, stochastic, seed);
        } else {
            quantize_rows_kernel<__half, 1><<<grid, block, 0, stream>>>(s_ptr, q_ptr, row_scale, rows, cols, stochastic, seed);
        }
    }
}

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
) {
    if (rows <= 0 || cols <= 0) return;
    dim3 grid(static_cast<unsigned int>(rows));
    dim3 block(kThreads);
    uint8_t* q_ptr = reinterpret_cast<uint8_t*>(q_data);
    if (mode_id == 0) {
        shy_prune_rows_kernel<0><<<grid, block, 0, stream>>>(
            q_ptr, row_scale, rows, cols, downscale, eps, zero_diag, pruned_counter
        );
    } else {
        shy_prune_rows_kernel<1><<<grid, block, 0, stream>>>(
            q_ptr, row_scale, rows, cols, downscale, eps, zero_diag, pruned_counter
        );
    }
}
