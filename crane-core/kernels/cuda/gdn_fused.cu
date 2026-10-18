// Fused multi-token Gated Delta Net stages (bf16 activations, head dim 128).
//
// The op-by-op path between the input projection and the output projection of
// a GDN layer costs ~45 small launches for a short pass (a speculative-decoding
// verification block, a prefill chunk), which makes such passes dispatch-bound.
// These three kernels replace it, reproducing the existing path's rounding
// points exactly (they were aligned to Torch's bf16 autocast behaviour):
//
//   gdn_conv_silu_bf16     causal conv1d (kernel 4) + silu, with the new conv
//                          state and an optional mid-sequence checkpoint
//   gdn_prep_bf16          split Q|K|V, per-head L2 norm, 1/sqrt(K) scale,
//                          beta/g gates, and the [heads, S, dim] f32 layout the
//                          recurrence kernel reads
//   gdn_post_norm_bf16     recurrence output -> bf16 -> gated RMSNorm -> bf16
//
// Every f32 mul/add that the Candle chain performs as a separate op is spelled
// with __fmul_rn/__fadd_rn so nvcc cannot contract it into an FMA.

#include <cuda_bf16.h>
#include <math.h>

#define GDN_HEAD_DIM 128
#define GDN_CONV_K 4

__device__ __forceinline__ float bf16_round(float v) {
    return __bfloat162float(__float2bfloat16_rn(v));
}

// Candle's f32 silu: x / (1 + exp(-x)).
__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

// One thread per channel; the window of the last K inputs slides along time.
// x: [S, C] (row stride xs), state: [C, K] bf16, w: [C, K] f32 (already rounded
// through bf16 like the Candle path), y: [S, C] bf16, new_state/ckpt: [C, K].
// `hidden[c, i]` is state[c, i] for i < K and x[i - K, c] after that; output t
// reads hidden[t + 1 .. t + K].
extern "C" __global__ void gdn_conv_silu_bf16(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ state,
    const float* __restrict__ w, __nv_bfloat16* __restrict__ y,
    __nv_bfloat16* __restrict__ new_state, __nv_bfloat16* __restrict__ ckpt,
    int S, int C, int xs, int ckpt_at) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float wk[GDN_CONV_K];
#pragma unroll
    for (int j = 0; j < GDN_CONV_K; ++j) wk[j] = w[c * GDN_CONV_K + j];

    auto hidden = [&](int i) -> float {
        return i < GDN_CONV_K ? __bfloat162float(state[c * GDN_CONV_K + i])
                              : __bfloat162float(x[(size_t)(i - GDN_CONV_K) * xs + c]);
    };

    float win[GDN_CONV_K];  // hidden[t + 1 .. t + K]
#pragma unroll
    for (int j = 0; j < GDN_CONV_K; ++j) win[j] = hidden(1 + j);
    for (int t = 0; t < S; ++t) {
        float acc = __fmul_rn(win[0], wk[0]);
#pragma unroll
        for (int j = 1; j < GDN_CONV_K; ++j) acc = __fadd_rn(acc, __fmul_rn(win[j], wk[j]));
        float r = bf16_round(acc);  // .to_dtype(bf16) before the activation
        y[(size_t)t * C + c] = __float2bfloat16_rn(silu_f32(r));
        // Slide the window to hidden[t + 2 .. t + K + 1]; the last step has no
        // next output, so don't read past the end of x.
#pragma unroll
        for (int j = 0; j + 1 < GDN_CONV_K; ++j) win[j] = win[j + 1];
        if (t + 1 < S) win[GDN_CONV_K - 1] = hidden(t + 1 + GDN_CONV_K);
    }
    // New state = hidden[S .. S + K).
#pragma unroll
    for (int k = 0; k < GDN_CONV_K; ++k)
        new_state[c * GDN_CONV_K + k] = __float2bfloat16_rn(hidden(S + k));
    if (ckpt != nullptr) {
#pragma unroll
        for (int k = 0; k < GDN_CONV_K; ++k)
            ckpt[c * GDN_CONV_K + k] = __float2bfloat16_rn(hidden(ckpt_at + k));
    }
}

// Candle's L2 norm for bf16 inputs, per 128-wide row, one warp per row: squares
// rounded to bf16 individually, four lane-local accumulators (elements
// lane + 32 j), warp tree reduce, eps added to the *sum*.
__device__ __forceinline__ void l2_row(const float (&v)[4], float (&out)[4]) {
    float p[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) p[j] = bf16_round(v[j] * v[j]);
    float sum = __fadd_rn(__fadd_rn(__fadd_rn(p[0], p[1]), p[2]), p[3]);
    for (int off = 16; off > 0; off >>= 1) sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, off));
    float inv = rsqrtf(__fadd_rn(__shfl_sync(0xffffffff, sum, 0), 1e-6f));
#pragma unroll
    for (int j = 0; j < 4; ++j) out[j] = v[j] * inv;
}

// Grid: (len, ceil(H / warps), 1); each warp owns one (t, value head).
//   conv:  [S, C] bf16, C = 2*key_dim + value_dim, layout Q | K | V
//   a, b:  [S, H] bf16 (row strides as, bs)
//   q3,k3,v3: [H, len, 128] f32, g2/beta2: [H, len] f32 for tokens t0 .. t0+len
// Key head of value head hv: order 0 (interleaved) hv / vpg, order 1 (chunked)
// hv % num_k.
extern "C" __global__ void gdn_prep_bf16(
    const __nv_bfloat16* __restrict__ conv, const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b, const float* __restrict__ neg_exp_a_log,
    const float* __restrict__ dt_bias, float* __restrict__ q3, float* __restrict__ k3,
    float* __restrict__ v3, float* __restrict__ g2, float* __restrict__ beta2,
    int t0, int len, int H, int num_k, int vpg, int order, int key_dim, int C, int as_,
    int bs_, float scale) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    int tt = blockIdx.x;
    int hv = blockIdx.y * warps + warp;
    if (hv >= H) return;
    int t = t0 + tt;
    int kh = order == 0 ? hv / vpg : hv % num_k;

    const __nv_bfloat16* qrow = conv + (size_t)t * C + kh * GDN_HEAD_DIM;
    const __nv_bfloat16* krow = conv + (size_t)t * C + key_dim + kh * GDN_HEAD_DIM;
    const __nv_bfloat16* vrow = conv + (size_t)t * C + 2 * key_dim + hv * GDN_HEAD_DIM;
    size_t out_row = ((size_t)hv * len + tt) * GDN_HEAD_DIM;

    float qv[4], kv[4], qn[4], kn[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        qv[j] = __bfloat162float(qrow[lane + 32 * j]);
        kv[j] = __bfloat162float(krow[lane + 32 * j]);
    }
    l2_row(qv, qn);
    l2_row(kv, kn);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        q3[out_row + lane + 32 * j] = __fmul_rn(qn[j], scale);
        k3[out_row + lane + 32 * j] = kn[j];
        v3[out_row + lane + 32 * j] = __bfloat162float(vrow[lane + 32 * j]);
    }
    if (lane == 0) {
        float bv = __bfloat162float(b[(size_t)t * bs_ + hv]);
        float av = __bfloat162float(a[(size_t)t * as_ + hv]);
        float beta = 1.0f / (1.0f + expf(-bv));
        float sp = __fadd_rn(av, dt_bias[hv]);
        sp = sp > 20.f ? sp : log1pf(expf(sp));
        g2[(size_t)hv * len + tt] = __fmul_rn(neg_exp_a_log[hv], sp);
        beta2[(size_t)hv * len + tt] = bf16_round(beta);  // beta is rounded to bf16
    }
}

// Recurrence output y [H, len, 128] f32 (tokens t0 .. t0+len) -> out [S, H*128] bf16:
//   x = bf16(y); n = bf16(rms_norm(x) in the ordered-f32 layout); w' = bf16(n * w);
//   out = bf16(w' * silu(z))
// rms_norm uses the `rms_norm_f32_ordered` lane layout (lane*4 + j).
extern "C" __global__ void gdn_post_norm_bf16(
    const float* __restrict__ y, const __nv_bfloat16* __restrict__ z,
    const float* __restrict__ w, __nv_bfloat16* __restrict__ out, int t0, int len, int H,
    int zs, float eps) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    int tt = blockIdx.x;
    int hv = blockIdx.y * warps + warp;
    if (hv >= H) return;
    int t = t0 + tt;
    const float* yrow = y + ((size_t)hv * len + tt) * GDN_HEAD_DIM;
    const __nv_bfloat16* zrow = z + (size_t)t * zs + hv * GDN_HEAD_DIM;

    float xv[4], sums[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        xv[j] = bf16_round(yrow[lane * 4 + j]);
        sums[j] = __fadd_rn(0.f, __fmul_rn(xv[j], xv[j]));
    }
    float sum = __fadd_rn(__fadd_rn(__fadd_rn(sums[0], sums[1]), sums[2]), sums[3]);
    for (int off = 16; off > 0; off >>= 1) sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, off));
    sum = __shfl_sync(0xffffffff, sum, 0);
    float inv = rsqrtf(__fadd_rn(sum / (float)GDN_HEAD_DIM, eps));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int col = lane * 4 + j;
        float normalized = bf16_round(__fmul_rn(xv[j], inv));
        float weighted = bf16_round(__fmul_rn(normalized, w[col]));
        float gate = silu_f32(__bfloat162float(zrow[col]));
        out[(size_t)t * H * GDN_HEAD_DIM + hv * GDN_HEAD_DIM + col] =
            __float2bfloat16_rn(__fmul_rn(weighted, gate));
    }
}
