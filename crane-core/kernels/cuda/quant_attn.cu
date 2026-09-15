// SPDX-License-Identifier: MIT
/**
 * Fused dequantize-in-attention kernels for QuantKvCache's GPU decode path.
 *
 * Targets: sm_80+ (Ampere & newer, bf16 support)
 *
 * Reads int8/int4 quantized K/V codes + per-token f32 scales directly,
 * dequantizing on the fly, instead of the caller materializing a full-size
 * dequantized K/V tensor first. See `quant_kv_cache.rs`'s `quantize_per_token`/
 * `dequantize_per_token` for the reference (scalar) math these kernels mirror:
 *   int8:  value = (code        - 128) * scale
 *   int4:  value = (nibble(code) -  8) * scale, nibble 0 = low bits (even d),
 *          nibble 1 = high bits (odd d) — matches `pack_nibbles`'s layout.
 *
 * Each thread computes one output element and loops serially over the
 * reduction axis (head_dim for `quant_qk_dot`, seq_len for `quant_sv_dot`).
 * This trades some memory-traffic redundancy (K/V codes and weights are
 * re-read across threads that share them) for a kernel simple enough to
 * verify by inspection; a warp-cooperative / shared-memory tiled version is
 * a follow-up optimization once this is verified against real hardware.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ---- dtype conversion helpers ---------------------------------------------

template <typename T>
__device__ __forceinline__ float qattn_to_f32(T x);

template <>
__device__ __forceinline__ float qattn_to_f32<float>(float x) {
    return x;
}
template <>
__device__ __forceinline__ float qattn_to_f32<__half>(__half x) {
    return __half2float(x);
}
template <>
__device__ __forceinline__ float qattn_to_f32<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T qattn_from_f32(float x);

template <>
__device__ __forceinline__ float qattn_from_f32<float>(float x) {
    return x;
}
template <>
__device__ __forceinline__ __half qattn_from_f32<__half>(float x) {
    return __float2half(x);
}
template <>
__device__ __forceinline__ __nv_bfloat16 qattn_from_f32<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

// ---- shared dequantization helpers -----------------------------------------

__device__ __forceinline__ float qattn_dequant_int8(uint8_t code, float scale) {
    return (static_cast<float>(code) - 128.0f) * scale;
}

// `nibble_idx == 0` selects the low 4 bits (even d), `1` the high 4 bits (odd
// d) — matches `pack_nibbles`'s `byte = lo + hi * 16` layout.
__device__ __forceinline__ float qattn_dequant_int4(uint8_t packed, int nibble_idx, float scale) {
    float code = (nibble_idx == 0) ? static_cast<float>(packed & 0x0F) : static_cast<float>(packed >> 4);
    return (code - 8.0f) * scale;
}

// ---- quant_qk_dot: Q_scaled @ dequant(K)^T -> scores -----------------------
//
// q:       [B, KV, R, D]         (pre-scaled by the caller)
// k_codes: [B, KV, S, D]  (int8) or [B, KV, S, D/2] (int4, nibble-packed)
// k_scale: [B, KV, S, 1]
// scores:  [B, KV, R, S]
//
// One thread per (b, kv, r, s) output score; each thread reduces over D.

template <typename T, int BITS>
__device__ __forceinline__ void quant_qk_dot_body(
    const T *__restrict__ q,
    const uint8_t *__restrict__ k_codes,
    const float *__restrict__ k_scale,
    T *__restrict__ scores,
    int B, int KV, int R, int S, int D, int S_alloc
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(B) * KV * R * S;
    if (idx >= total) {
        return;
    }

    int s = static_cast<int>(idx % S);
    int64_t tmp = idx / S;
    int r = static_cast<int>(tmp % R);
    tmp /= R;
    int kv = static_cast<int>(tmp % KV);
    int b = static_cast<int>(tmp / KV);

    int64_t bkv = static_cast<int64_t>(b) * KV + kv;
    const T *q_ptr = q + (bkv * R + r) * D;
    // k_codes/k_scale are a narrowed view into a buffer with headroom past S,
    // so their token stride is S_alloc, not S — see quant_kv_cache.rs's
    // grow_append (narrow(2, 0, S) over a [.., S_alloc, ..] allocation).
    float scale = k_scale[bkv * S_alloc + s];

    float sum = 0.0f;
    if (BITS == 8) {
        const uint8_t *k_ptr = k_codes + (bkv * S_alloc + s) * D;
        for (int d = 0; d < D; d++) {
            sum += qattn_to_f32<T>(q_ptr[d]) * qattn_dequant_int8(k_ptr[d], scale);
        }
    } else {
        int d2 = D / 2;
        const uint8_t *k_ptr = k_codes + (bkv * S_alloc + s) * d2;
        for (int d = 0; d < D; d++) {
            sum += qattn_to_f32<T>(q_ptr[d]) * qattn_dequant_int4(k_ptr[d / 2], d % 2, scale);
        }
    }
    scores[idx] = qattn_from_f32<T>(sum);
}

// ---- quant_sv_dot: softmax(scores) @ dequant(V) -> output ------------------
//
// weights: [B, KV, R, S]  (post-softmax)
// v_codes: [B, KV, S, D]  (int8) or [B, KV, S, D/2] (int4, nibble-packed)
// v_scale: [B, KV, S, 1]
// output:  [B, KV, R, D]
//
// One thread per (b, kv, r, d) output element; each thread reduces over S.

template <typename T, int BITS>
__device__ __forceinline__ void quant_sv_dot_body(
    const T *__restrict__ weights,
    const uint8_t *__restrict__ v_codes,
    const float *__restrict__ v_scale,
    T *__restrict__ output,
    int B, int KV, int R, int S, int D, int S_alloc
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = static_cast<int64_t>(B) * KV * R * D;
    if (idx >= total) {
        return;
    }

    int d = static_cast<int>(idx % D);
    int64_t tmp = idx / D;
    int r = static_cast<int>(tmp % R);
    tmp /= R;
    int kv = static_cast<int>(tmp % KV);
    int b = static_cast<int>(tmp / KV);

    int64_t bkv = static_cast<int64_t>(b) * KV + kv;
    const T *w_ptr = weights + (bkv * R + r) * S;

    // v_codes/v_scale are a narrowed view into a buffer with headroom past S,
    // so their token stride is S_alloc, not S — see quant_kv_cache.rs's
    // grow_append (narrow(2, 0, S) over a [.., S_alloc, ..] allocation).
    float acc = 0.0f;
    if (BITS == 8) {
        const uint8_t *v_ptr = v_codes + bkv * S_alloc * D;
        for (int s = 0; s < S; s++) {
            float w = qattn_to_f32<T>(w_ptr[s]);
            acc += w * qattn_dequant_int8(v_ptr[static_cast<int64_t>(s) * D + d], v_scale[bkv * S_alloc + s]);
        }
    } else {
        int d2 = D / 2;
        const uint8_t *v_ptr = v_codes + bkv * S_alloc * d2;
        int nibble_idx = d % 2;
        int byte_col = d / 2;
        for (int s = 0; s < S; s++) {
            float w = qattn_to_f32<T>(w_ptr[s]);
            uint8_t packed = v_ptr[static_cast<int64_t>(s) * d2 + byte_col];
            acc += w * qattn_dequant_int4(packed, nibble_idx, v_scale[bkv * S_alloc + s]);
        }
    }
    output[idx] = qattn_from_f32<T>(acc);
}

// ---- extern "C" entry points (6 per kernel: {int8,int4} x {f32,f16,bf16}) --

#define QATTN_QK_KERNEL(NAME, T, BITS)                                      \
    extern "C" __global__ void NAME(                                        \
        const T *__restrict__ q, const uint8_t *__restrict__ k_codes,       \
        const float *__restrict__ k_scale, T *__restrict__ scores, int B,   \
        int KV, int R, int S, int D, int S_alloc                            \
    ) {                                                                     \
        quant_qk_dot_body<T, BITS>(q, k_codes, k_scale, scores, B, KV, R, S, D, S_alloc); \
    }

#define QATTN_SV_KERNEL(NAME, T, BITS)                                      \
    extern "C" __global__ void NAME(                                        \
        const T *__restrict__ weights, const uint8_t *__restrict__ v_codes, \
        const float *__restrict__ v_scale, T *__restrict__ output, int B,   \
        int KV, int R, int S, int D, int S_alloc                            \
    ) {                                                                     \
        quant_sv_dot_body<T, BITS>(weights, v_codes, v_scale, output, B, KV, R, S, D, S_alloc); \
    }

QATTN_QK_KERNEL(quant_qk_dot_int8_f32, float, 8)
QATTN_QK_KERNEL(quant_qk_dot_int8_f16, __half, 8)
QATTN_QK_KERNEL(quant_qk_dot_int8_bf16, __nv_bfloat16, 8)
QATTN_QK_KERNEL(quant_qk_dot_int4_f32, float, 4)
QATTN_QK_KERNEL(quant_qk_dot_int4_f16, __half, 4)
QATTN_QK_KERNEL(quant_qk_dot_int4_bf16, __nv_bfloat16, 4)

QATTN_SV_KERNEL(quant_sv_dot_int8_f32, float, 8)
QATTN_SV_KERNEL(quant_sv_dot_int8_f16, __half, 8)
QATTN_SV_KERNEL(quant_sv_dot_int8_bf16, __nv_bfloat16, 8)
QATTN_SV_KERNEL(quant_sv_dot_int4_f32, float, 4)
QATTN_SV_KERNEL(quant_sv_dot_int4_f16, __half, 4)
QATTN_SV_KERNEL(quant_sv_dot_int4_bf16, __nv_bfloat16, 4)

#undef QATTN_QK_KERNEL
#undef QATTN_SV_KERNEL
