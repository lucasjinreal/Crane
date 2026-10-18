// SPDX-License-Identifier: MIT
//
// DSpark drafter: the sequential greedy Markov chain in one launch.
//
// For each block position i, with `prev` the previous token (the anchor for
// i = 0): score every one of the k candidates as
//     logits[i, j] + dot(W2[i, j, :], W1[prev, :])
// and pick the best; its token id becomes `prev` for position i + 1. The op
// chain this replaces issued ~10 launches per position (gather, matmul,
// argmax, ...), so a 15-token block cost ~150 launches of near-zero GPU work.
// One thread block does all positions here.
//
// The dot product is rounded to the weight dtype before it is added, like the
// matmul it replaces (whose output is in that dtype). Ties go to the lowest
// candidate index, i.e. the higher base logit (candidates are sorted).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define DS_MAX_RANK 1024

template <typename T> __device__ __forceinline__ float ds_load(const T * p);
template <> __device__ __forceinline__ float ds_load<float>(const float * p) { return *p; }
template <> __device__ __forceinline__ float ds_load<half>(const half * p) { return __half2float(*p); }
template <> __device__ __forceinline__ float ds_load<__nv_bfloat16>(const __nv_bfloat16 * p) {
    return __bfloat162float(*p);
}

template <typename T> __device__ __forceinline__ float ds_round(float v);
template <> __device__ __forceinline__ float ds_round<float>(float v) { return v; }
template <> __device__ __forceinline__ float ds_round<half>(float v) { return __half2float(__float2half(v)); }
template <> __device__ __forceinline__ float ds_round<__nv_bfloat16>(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

// Eight consecutive weights as floats (one 16-byte load for half types).
template <typename T> __device__ __forceinline__ void ds_load8(const T * p, float (&out)[8]);
template <> __device__ __forceinline__ void ds_load8<float>(const float * p, float (&out)[8]) {
    const float4 a = reinterpret_cast<const float4 *>(p)[0];
    const float4 b = reinterpret_cast<const float4 *>(p)[1];
    out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w;
    out[4] = b.x; out[5] = b.y; out[6] = b.z; out[7] = b.w;
}
template <> __device__ __forceinline__ void ds_load8<half>(const half * p, float (&out)[8]) {
    const uint4 v = *reinterpret_cast<const uint4 *>(p);
    const half2 * h = reinterpret_cast<const half2 *>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __half22float2(h[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}
template <> __device__ __forceinline__ void ds_load8<__nv_bfloat16>(const __nv_bfloat16 * p, float (&out)[8]) {
    const uint4 v = *reinterpret_cast<const uint4 *>(p);
    const __nv_bfloat162 * h = reinterpret_cast<const __nv_bfloat162 *>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(h[i]);
        out[2 * i] = f.x;
        out[2 * i + 1] = f.y;
    }
}

#define DS_THREADS 256
#define DS_MAX_K 1024

// Launch with one block of DS_THREADS threads. k <= DS_MAX_K, r a multiple
// of 8 and <= DS_MAX_RANK, and `w2` / `w1` 16-byte aligned rows.
// ids: [blk, k] u32; logits: [blk, k] f32; w2: [blk, k, r]; w1: [vocab, r].
// Writes the drafted tokens and, per position, the token it was conditioned on.
//
// Each warp scores a strided subset of the candidates; for one candidate the
// lanes read consecutive 8-weight slices of its W2 row (coalesced), and a
// shuffle tree sums them.
template <typename T>
__device__ __forceinline__ void ds_markov_chain(
    const uint32_t * __restrict__ ids, const float * __restrict__ logits,
    const T * __restrict__ w2, const T * __restrict__ w1, uint32_t anchor,
    uint32_t * __restrict__ tokens, uint32_t * __restrict__ prevs, int blk, int k, int r) {
    __shared__ float e[DS_MAX_RANK];
    __shared__ float score[DS_MAX_K];
    __shared__ float best_v[DS_THREADS / 32];
    __shared__ int best_j[DS_THREADS / 32];
    __shared__ uint32_t prev_s;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int warps = DS_THREADS / 32;
    if (tid == 0) prev_s = anchor;
    __syncthreads();

    for (int i = 0; i < blk; ++i) {
        const uint32_t prev = prev_s;
        for (int t = tid; t < r; t += DS_THREADS) e[t] = ds_load<T>(w1 + size_t(prev) * r + t);
        __syncthreads();

        for (int j = warp; j < k; j += warps) {
            const T * row = w2 + (size_t(i) * k + j) * r;
            float acc = 0.f;
            for (int t0 = lane * 8; t0 < r; t0 += 32 * 8) {
                float w[8];
                ds_load8<T>(row + t0, w);
#pragma unroll
                for (int q = 0; q < 8; ++q) acc += w[q] * e[t0 + q];
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc += __shfl_xor_sync(0xffffffffu, acc, offset);
            }
            if (lane == 0) score[j] = logits[size_t(i) * k + j] + ds_round<T>(acc);
        }
        __syncthreads();

        // Argmax, ties to the lower index (the higher base logit).
        float v = -INFINITY;
        int bj = k;
        for (int j = tid; j < k; j += DS_THREADS) {
            if (score[j] > v) {
                v = score[j];
                bj = j;
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, v, offset);
            const int oj = __shfl_down_sync(0xffffffffu, bj, offset);
            if (ov > v || (ov == v && oj < bj)) {
                v = ov;
                bj = oj;
            }
        }
        if (lane == 0) {
            best_v[warp] = v;
            best_j[warp] = bj;
        }
        __syncthreads();
        if (tid == 0) {
            float bv = best_v[0];
            int bb = best_j[0];
            for (int w = 1; w < warps; ++w) {
                if (best_v[w] > bv || (best_v[w] == bv && best_j[w] < bb)) {
                    bv = best_v[w];
                    bb = best_j[w];
                }
            }
            const uint32_t tok = ids[size_t(i) * k + bb];
            prevs[i] = prev;
            tokens[i] = tok;
            prev_s = tok;
        }
        __syncthreads();
    }
}

#define DS_MARKOV_CHAIN(name, T)                                                               \
    extern "C" __global__ void name(                                                           \
        const uint32_t * ids, const float * logits, const T * w2, const T * w1, uint32_t anchor, \
        uint32_t * tokens, uint32_t * prevs, int blk, int k, int r) {                          \
        ds_markov_chain<T>(ids, logits, w2, w1, anchor, tokens, prevs, blk, k, r);             \
    }

DS_MARKOV_CHAIN(dspark_markov_chain_f32, float)
DS_MARKOV_CHAIN(dspark_markov_chain_f16, half)
DS_MARKOV_CHAIN(dspark_markov_chain_bf16, __nv_bfloat16)
