// SPDX-License-Identifier: MIT
//
// Grouped-by-expert `MoE` matvec for small token batches (DSpark verify
// blocks, short prefill chunks) over packed llama.cpp k-quant experts
// (`[experts, rows, cols]`, Q4_K / Q5_K / Q6_K).
//
// candle's `indexed_moe_forward` launches one thread block per (output row,
// token-expert pair), so an expert routed by several tokens of the batch is
// streamed from memory once per pair. Here the pairs are first bucketed by
// expert (`moe_group_pairs`, on the device: no host round-trip), then one warp
// per (output row, used expert) decodes each weight block once and dots it
// against every routed activation row with `dp4a`.
//
// Activations come pre-quantized to int8 with one f32 scale per 32 values
// (`iq4_quantize_q8_*` in `quant_iq4.cu`). The output is f32, laid out like
// `indexed_moe_forward`'s: `out[pair * n_rows + row]`.

#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define QK_K 256
#define MG_WARPS 4        // output rows per thread block, one warp each
#define MG_PAIRS 8        // activation rows accumulated per pass over a weight row
#define MG_GROUP_THREADS 256

enum MgType { MG_Q4_K = 0, MG_Q5_K = 1, MG_Q6_K = 2 };

template <int TY> __device__ __forceinline__ constexpr int mg_block_bytes() {
    return TY == MG_Q4_K ? 144 : TY == MG_Q5_K ? 176 : 210;
}

__device__ __forceinline__ float mg_half(const uint8_t * p) {
    const uint16_t bits = uint16_t(p[0]) | (uint16_t(p[1]) << 8);
    return __half2float(*reinterpret_cast<const half *>(&bits));
}

// Two little-endian words from a 2-byte aligned address (Q6_K blocks are 210
// bytes, so nothing wider is safe).
__device__ __forceinline__ uint2 mg_load8_a2(const uint8_t * p) {
    const uint16_t * h = reinterpret_cast<const uint16_t *>(p);
    return make_uint2(uint32_t(h[0]) | (uint32_t(h[1]) << 16), uint32_t(h[2]) | (uint32_t(h[3]) << 16));
}

// ---------------------------------------------------------------------------
// Pair grouping. `group` layout (u32): [0] = used experts U, then
// `experts[P]` (slot -> expert id), `offsets[P + 1]` (slot -> first entry of
// `pairs`), `pairs[P]` (pair indices bucketed by slot). Only the first U slots
// are meaningful. Pair order inside a bucket is arbitrary; it does not affect
// any result since every pair's dot product is computed on its own.
// ---------------------------------------------------------------------------

extern "C" __global__ void moe_group_pairs(
    const uint32_t * __restrict__ ids, uint32_t * __restrict__ group, int n_pairs, int n_experts) {
    extern __shared__ uint32_t smem[];
    uint32_t * counts = smem;                 // [n_experts]
    uint32_t * slot_of = smem + n_experts;    // [n_experts]
    __shared__ uint32_t scan_used[MG_GROUP_THREADS];
    __shared__ uint32_t scan_pairs[MG_GROUP_THREADS];

    const int tid = threadIdx.x;
    for (int e = tid; e < n_experts; e += MG_GROUP_THREADS) counts[e] = 0;
    __syncthreads();
    for (int p = tid; p < n_pairs; p += MG_GROUP_THREADS) atomicAdd(&counts[ids[p]], 1u);
    __syncthreads();

    // Each thread owns a contiguous run of experts; an inclusive scan over the
    // per-thread (used, pairs) totals yields every slot and offset.
    const int per = (n_experts + MG_GROUP_THREADS - 1) / MG_GROUP_THREADS;
    const int e0 = tid * per;
    const int e1 = min(e0 + per, n_experts);
    uint32_t used = 0, pairs = 0;
    for (int e = e0; e < e1; ++e) {
        used += counts[e] > 0;
        pairs += counts[e];
    }
    scan_used[tid] = used;
    scan_pairs[tid] = pairs;
    __syncthreads();
    for (int step = 1; step < MG_GROUP_THREADS; step <<= 1) {
        const uint32_t u = tid >= step ? scan_used[tid - step] : 0;
        const uint32_t q = tid >= step ? scan_pairs[tid - step] : 0;
        __syncthreads();
        scan_used[tid] += u;
        scan_pairs[tid] += q;
        __syncthreads();
    }

    uint32_t * experts = group + 1;
    uint32_t * offsets = experts + n_pairs;
    uint32_t * bucketed = offsets + n_pairs + 1;
    uint32_t slot = scan_used[tid] - used;
    uint32_t offset = scan_pairs[tid] - pairs;
    for (int e = e0; e < e1; ++e) {
        if (counts[e] > 0) {
            experts[slot] = e;
            offsets[slot] = offset;
            slot_of[e] = slot;
            offset += counts[e];
            ++slot;
        }
        counts[e] = 0;  // reused as the scatter cursor below
    }
    // Slots past the used count get empty ranges, so the matvec needs no
    // separate check of `group[0]`.
    const uint32_t n_used = scan_used[MG_GROUP_THREADS - 1];
    for (int s = n_used + tid; s <= n_pairs; s += MG_GROUP_THREADS) offsets[s] = uint32_t(n_pairs);
    if (tid == 0) group[0] = n_used;
    __syncthreads();
    for (int p = tid; p < n_pairs; p += MG_GROUP_THREADS) {
        const uint32_t e = ids[p];
        const uint32_t s = slot_of[e];
        bucketed[offsets[s] + atomicAdd(&counts[e], 1u)] = p;
    }
}

// ---------------------------------------------------------------------------
// Grouped matvec. Lane l covers values 8*(l%4)..+7 of 32-value sub-block l/4
// of every 256-value block, so its weight scale is constant per block and its
// activation loads are 8-byte aligned.
// ---------------------------------------------------------------------------

// One lane's raw bytes of one block, loaded before any are decoded so that a
// warp keeps several blocks' loads in flight.
struct MgRaw {
    uint4 head;   // Q4_K/Q5_K: d, dmin, 12 scale bytes. Q6_K: .x = d, .y = scale byte
    uint2 q;      // 8 quant bytes (Q6_K: `ql`)
    uint2 h;      // Q5_K/Q6_K high bits
};

template <int TY>
__device__ __forceinline__ MgRaw mg_load(const uint8_t * blk, int ib, int j0) {
    MgRaw raw;
    if (TY == MG_Q4_K || TY == MG_Q5_K) {
        raw.head = *reinterpret_cast<const uint4 *>(blk);
        raw.q = *reinterpret_cast<const uint2 *>(blk + (TY == MG_Q5_K ? 48 : 16) + 32 * (ib / 2) + j0);
        raw.h = TY == MG_Q5_K ? *reinterpret_cast<const uint2 *>(blk + 16 + j0) : make_uint2(0, 0);
    } else {
        const int half_idx = ib >> 2;
        const int k = ib & 3;
        raw.q = mg_load8_a2(blk + 64 * half_idx + 32 * (k & 1) + j0);
        raw.h = mg_load8_a2(blk + 128 + 32 * half_idx + j0);
        raw.head.x = *reinterpret_cast<const uint16_t *>(blk + 208);
        raw.head.y = blk[192 + 8 * half_idx + 2 * k + (j0 >> 4)];
    }
    return raw;
}

__device__ __forceinline__ int mg_byte(uint4 v, int i) {
    const uint32_t w = i < 4 ? v.x : i < 8 ? v.y : i < 12 ? v.z : v.w;
    return int((w >> (8 * (i & 3))) & 0xff);
}

__device__ __forceinline__ float mg_half_bits(uint32_t bits) {
    const uint16_t b = uint16_t(bits);
    return __half2float(*reinterpret_cast<const half *>(&b));
}

// One lane's decoded share of one block: 8 unsigned quants as two words, the
// scale multiplying `dot(q, x)` and the one multiplying `sum(x)`.
struct MgLane {
    int q0, q1;
    float dq;   // d * scale
    float dm;   // Q4_K/Q5_K: dmin * min; Q6_K: 32 * d * scale (the -32 offset)
};

template <int TY>
__device__ __forceinline__ MgLane mg_decode(const MgRaw & raw, int ib) {
    MgLane out;
    if (TY == MG_Q4_K || TY == MG_Q5_K) {
        // ggml `get_scale_min_k4` over header bytes 4..15.
        int s, m;
        if (ib < 4) {
            s = mg_byte(raw.head, 4 + ib) & 63;
            m = mg_byte(raw.head, 8 + ib) & 63;
        } else {
            s = (mg_byte(raw.head, 8 + ib) & 0xf) | ((mg_byte(raw.head, ib) >> 6) << 4);
            m = (mg_byte(raw.head, 8 + ib) >> 4) | ((mg_byte(raw.head, 4 + ib) >> 6) << 4);
        }
        out.dq = mg_half_bits(raw.head.x) * float(s);
        out.dm = mg_half_bits(raw.head.x >> 16) * float(m);
        const int shift = 4 * (ib & 1);
        uint32_t q0 = (raw.q.x >> shift) & 0x0f0f0f0fu;
        uint32_t q1 = (raw.q.y >> shift) & 0x0f0f0f0fu;
        if (TY == MG_Q5_K) {
            q0 |= ((raw.h.x >> ib) & 0x01010101u) << 4;
            q1 |= ((raw.h.y >> ib) & 0x01010101u) << 4;
        }
        out.q0 = int(q0);
        out.q1 = int(q1);
    } else {
        // Sub-block ib = 4*half + k takes the low (k < 2) or high nibbles of 32
        // `ql` bytes and bits 2k..2k+1 of 32 `qh` bytes; one i8 scale per 16.
        const int k = ib & 3;
        const int nshift = 4 * (k >> 1);
        const int hshift = 2 * k;
        out.q0 = int(((raw.q.x >> nshift) & 0x0f0f0f0fu) | (((raw.h.x >> hshift) & 0x03030303u) << 4));
        out.q1 = int(((raw.q.y >> nshift) & 0x0f0f0f0fu) | (((raw.h.y >> hshift) & 0x03030303u) << 4));
        out.dq = mg_half_bits(raw.head.x) * float(int8_t(raw.head.y));
        out.dm = 32.f * out.dq;
    }
    return out;
}

// Thread layout. Lane l of a warp covers values 8*(l%4)..+7 of sub-block l/4
// of each 256-value block. WPR warps share a row (normally one: the whole row
// in one warp makes the per-pair reduction a plain shuffle tree, which beat
// splitting rows over 2 or 4 warps at every batch size measured), and each
// thread takes ITERS blocks `warp_in_row + i * WPR` of its row. A thread block
// holds MG_WARPS / WPR rows of one expert; per-row partials of a split row
// meet in shared memory.
//
// The kernel is latency-bound, so two rules shape it: no load sits behind a
// branch (nvcc would serialize them; loop bounds are compile-time and
// out-of-range work is clamped and discarded instead), and the dependent
// chain is short (the pair ids are fetched alongside the weights, not after).
#define MG_GROUP 4        // most pairs whose activation loads are issued together

// Warp-reduce G values at once (G a power of two <= 32): each butterfly step
// halves the values a lane keeps and swaps the other half with its partner,
// so it takes G - 1 + log2(32 / G) shuffles instead of 5 G. Afterwards every
// lane l holds the full sum of value l / (32 / G).
template <int G>
__device__ __forceinline__ float mg_reduce_multi(float (&v)[G]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int width = G; width > 1; width >>= 1) {
        // The first (widest) step decides the top bit of the kept value index,
        // so it pairs lanes across the top bit of l / (32 / G).
        const int bit = (width / 2) * (32 / G);
        const bool upper = lane & bit;
#pragma unroll
        for (int k = 0; k < width / 2; ++k) {
            const float send = upper ? v[k] : v[k + width / 2];
            const float keep = upper ? v[k + width / 2] : v[k];
            v[k] = keep + __shfl_xor_sync(0xffffffffu, send, bit);
        }
    }
    float r = v[0];
#pragma unroll
    for (int offset = 32 / G / 2; offset > 0; offset >>= 1) {
        r += __shfl_xor_sync(0xffffffffu, r, offset);
    }
    return r;
}

// G pairs starting at `pairs`: dot every activation row with this thread's
// weight units, reduce over the warp and leave each pair's warp sum in
// `partial[warp][slot0 + g]`. Loads past `count` are clamped to the last pair
// and their results dropped, so no load sits behind a branch.
template <int TY, int ITERS, int G>
__device__ __forceinline__ void mg_pairs(
    const MgLane (&w)[ITERS], const int (&wcol)[ITERS],
    const int8_t * __restrict__ xq, const float * __restrict__ xd,
    const uint32_t * __restrict__ pairs, int count, int slot0,
    float (&partial)[MG_WARPS][MG_PAIRS], int cols, int x_div) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    int2 x[G][ITERS];
    float d[G][ITERS];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const uint32_t xrow = (pairs[min(g, count - 1)] / x_div) * uint32_t(cols);
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            x[g][i] = *reinterpret_cast<const int2 *>(xq + xrow + wcol[i]);
            d[g][i] = xd[(xrow + wcol[i]) >> 5];
        }
    }
    float v[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        v[g] = 0.f;
#pragma unroll
        for (int i = 0; i < ITERS; ++i) {
            const int dot = __dp4a(w[i].q1, x[g][i].y, __dp4a(w[i].q0, x[g][i].x, 0));
            const int sum = __dp4a(0x01010101, x[g][i].y, __dp4a(0x01010101, x[g][i].x, 0));
            v[g] += d[g][i] * (w[i].dq * float(dot) - w[i].dm * float(sum));
        }
    }
    const float r = mg_reduce_multi<G>(v);
    // After the reduction lane l holds value l / (32 / G).
    if (lane % (32 / G) == 0) {
        const int g = lane / (32 / G);
        if (g < count) partial[warp][slot0 + g] = r;
    }
}

template <int TY, int ITERS, int WPR>
__device__ __forceinline__ void moe_grouped_matvec(
    const uint8_t * __restrict__ weights, unsigned long long expert_stride,
    const int8_t * __restrict__ xq, const float * __restrict__ xd,
    const uint32_t * __restrict__ group, float * __restrict__ out,
    int n_rows, int cols, int x_div, int n_pairs) {
    __shared__ float partial[MG_WARPS][MG_PAIRS];
    __shared__ uint32_t pair_ids[MG_PAIRS];
    const uint32_t * experts = group + 1;
    const uint32_t * offsets = experts + n_pairs;
    const uint32_t * bucketed = offsets + n_pairs + 1;
    const int slot = blockIdx.y;
    const uint32_t begin = offsets[slot];
    const uint32_t end = offsets[slot + 1];
    const uint32_t expert = experts[slot];  // garbage for an unused slot; not used then
    if (begin == end) return;  // unused slot (uniform across the block)

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int blocks = cols / QK_K;
    constexpr int wpr = WPR;
    constexpr int rows_per_tb = MG_WARPS / WPR;
    const int row = min(int(blockIdx.x) * rows_per_tb + warp / wpr, n_rows - 1);
    const int warp_in_row = warp % wpr;
    const int ib = lane >> 2;
    const int j0 = (lane & 3) * 8;
    const uint8_t * wrow = weights + expert * expert_stride + size_t(row) * blocks * mg_block_bytes<TY>();

    // This thread's weight units, loaded once for all pairs. A unit past the
    // row end (blocks not a multiple of wpr) re-reads the last block and is
    // weighted to zero.
    MgRaw raw[ITERS];
    int wcol[ITERS];
#pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        const int b = min(warp_in_row + i * wpr, blocks - 1);
        raw[i] = mg_load<TY>(wrow + size_t(b) * mg_block_bytes<TY>(), ib, j0);
        wcol[i] = b * QK_K + 32 * ib + j0;
    }
    // Pair ids go to shared memory, fetched while the weights are in flight.
    const int first_n = min(int(end - begin), MG_PAIRS);
    if (threadIdx.x < first_n) pair_ids[threadIdx.x] = bucketed[begin + threadIdx.x];

    MgLane w[ITERS];
#pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        w[i] = mg_decode<TY>(raw[i], ib);
        if (warp_in_row + i * wpr >= blocks) w[i].dq = w[i].dm = 0.f;
    }

    for (uint32_t c0 = begin; c0 < end; c0 += MG_PAIRS) {
        const int n = min(int(end - c0), MG_PAIRS);
        if (c0 != begin && threadIdx.x < n) pair_ids[threadIdx.x] = bucketed[c0 + threadIdx.x];
        __syncthreads();
        // n is uniform across the block, so every thread takes the same path.
        for (int g0 = 0; g0 < n; g0 += MG_GROUP) {
            const int count = min(n - g0, MG_GROUP);
            if (count == 1) {
                mg_pairs<TY, ITERS, 1>(w, wcol, xq, xd, pair_ids + g0, count, g0, partial, cols, x_div);
            } else if (count == 2) {
                mg_pairs<TY, ITERS, 2>(w, wcol, xq, xd, pair_ids + g0, count, g0, partial, cols, x_div);
            } else {
                mg_pairs<TY, ITERS, MG_GROUP>(w, wcol, xq, xd, pair_ids + g0, count, g0, partial, cols, x_div);
            }
        }
        __syncthreads();
        // One thread per (row of this block, pair) adds its row's warps.
        if (threadIdx.x < rows_per_tb * MG_PAIRS) {
            const int r = threadIdx.x % MG_PAIRS;
            const int local_row = threadIdx.x / MG_PAIRS;
            const int out_row = int(blockIdx.x) * rows_per_tb + local_row;
            if (r < n && out_row < n_rows) {
                float v = 0.f;
                for (int k = 0; k < wpr; ++k) v += partial[local_row * wpr + k][r];
                out[size_t(pair_ids[r]) * n_rows + out_row] = v;
            }
        }
        __syncthreads();
    }
}

#define MG_MATVEC(name, TY, ITERS, WPR)                                                       \
    extern "C" __global__ void name(                                                          \
        const uint8_t * weights, unsigned long long expert_stride, const int8_t * xq,         \
        const float * xd, const uint32_t * group, float * out, int n_rows, int cols,          \
        int x_div, int n_pairs) {                                                             \
        moe_grouped_matvec<TY, ITERS, WPR>(weights, expert_stride, xq, xd, group, out,        \
                                           n_rows, cols, x_div, n_pairs);                     \
    }

// `_iN_wW`: W warps share a row, each thread takes N of its 256-value blocks
// (N * W >= blocks; the host picks the variant from the row width).
#define MG_VARIANTS(ty, TY)                                                                   \
    MG_MATVEC(moe_grouped_matvec_##ty##_i1_w1, TY, 1, 1)                                      \
    MG_MATVEC(moe_grouped_matvec_##ty##_i2_w1, TY, 2, 1)                                      \
    MG_MATVEC(moe_grouped_matvec_##ty##_i4_w1, TY, 4, 1)                                      \
    MG_MATVEC(moe_grouped_matvec_##ty##_i8_w1, TY, 8, 1)                                      \
    MG_MATVEC(moe_grouped_matvec_##ty##_i8_w2, TY, 8, 2)

MG_VARIANTS(q4_k, MG_Q4_K)
MG_VARIANTS(q5_k, MG_Q5_K)
MG_VARIANTS(q6_k, MG_Q6_K)

// ---------------------------------------------------------------------------
// The glue around the two grouped projections, one launch each.
// ---------------------------------------------------------------------------

// SwiGLU between the projections, straight into the int8 form the down
// projection reads. `gate_up` is [pairs, 2 * inter] f32 (gate half first);
// writes xq [pairs, inter] and one scale per 32 values. Bit-identical to
// `swiglu_f32` (same expression) followed by `iq4_quantize_q8_f32` (same
// scale and rounding), minus the two copies that made the halves contiguous.
// One warp per 32 outputs.
extern "C" __global__ void moe_swiglu_quantize_f32(
    const float * __restrict__ gate_up, int8_t * __restrict__ xq, float * __restrict__ xd,
    int pairs, int inter) {
    const int blk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int per_row = inter / 32;
    if (blk >= pairs * per_row) return;
    const int p = blk / per_row;
    const int col = (blk % per_row) * 32 + lane;
    const float g = gate_up[size_t(p) * 2 * inter + col];
    const float u = gate_up[size_t(p) * 2 * inter + inter + col];
    const float v = g / (1.0f + expf(-g)) * u;
    float amax = fabsf(v);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
    }
    const float d = amax / 127.f;
    xq[size_t(p) * inter + col] = int8_t(amax == 0.f ? 0 : __float2int_rn(v / d));
    if (lane == 0) xd[blk] = d;
}

template <typename T> __device__ __forceinline__ T mg_store(float v);
template <> __device__ __forceinline__ float mg_store<float>(float v) { return v; }
template <> __device__ __forceinline__ half mg_store<half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 mg_store<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// out[t, h] = sum_k weights[t, k] * y[t * top_k + k, h], summed over k in
// order in f32 and stored in the output dtype. One thread per output value.
template <typename T>
__device__ __forceinline__ void moe_combine(
    const float * __restrict__ y, const float * __restrict__ weights, T * __restrict__ out,
    int tokens, int top_k, int hidden) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(tokens) * hidden) return;
    const int t = int(i / hidden);
    const int h = int(i % hidden);
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) {
        acc += weights[t * top_k + k] * y[(size_t(t) * top_k + k) * hidden + h];
    }
    out[i] = mg_store<T>(acc);
}

#define MG_COMBINE(name, T)                                                                   \
    extern "C" __global__ void name(const float * y, const float * weights, T * out,          \
                                    int tokens, int top_k, int hidden) {                      \
        moe_combine<T>(y, weights, out, tokens, top_k, hidden);                               \
    }

MG_COMBINE(moe_combine_f32, float)
MG_COMBINE(moe_combine_f16, half)
MG_COMBINE(moe_combine_bf16, __nv_bfloat16)
