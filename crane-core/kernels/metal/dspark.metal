#include <metal_stdlib>
using namespace metal;

// DSpark drafter: the sequential greedy Markov chain in one dispatch, the
// Metal port of `kernels/cuda/dspark.cu`. Launcher: `src/models/dspark/metal.rs`.
//
// For each block position i, with `prev` the previous token (the anchor for
// i = 0), score every one of the k candidates as
//     logits[i, j] + T(dot(W2[i, j, :], W1[prev, :]))
// and pick the best; its token id becomes `prev` for position i + 1. The dot
// product is rounded to the weight dtype T like the matmul it replaces. Ties
// go to the lowest candidate index (the higher base logit: candidates are
// sorted).
//
// One threadgroup of DS_THREADS threads does all positions. Each SIMD-group
// scores a strided subset of the candidates; its lanes read consecutive
// weights of a candidate's W2 row and `simd_sum` combines them.

#define DS_THREADS 256
#define DS_GROUPS (DS_THREADS / 32)
#define DS_MAX_RANK 1024
#define DS_MAX_K 1024

struct ds_args {
    uint anchor;
    uint blk;
    uint k;
    uint r;
};

template <typename T>
kernel void dspark_markov_chain(
        device const uint  * ids    [[buffer(0)]],  // [blk, k]
        device const float * logits [[buffer(1)]],  // [blk, k]
        device const T     * w2     [[buffer(2)]],  // [blk, k, r]
        device const T     * w1     [[buffer(3)]],  // [vocab, r]
        device uint        * tokens [[buffer(4)]],  // [blk]
        device uint        * prevs  [[buffer(5)]],  // [blk]
        constant ds_args & args     [[buffer(6)]],
        uint tid  [[thread_index_in_threadgroup]],
        uint sg   [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float e[DS_MAX_RANK];
    threadgroup float score[DS_MAX_K];
    threadgroup float best_v[DS_GROUPS];
    threadgroup uint best_j[DS_GROUPS];
    threadgroup uint prev_s;

    const uint k = args.k;
    const uint r = args.r;
    if (tid == 0) {
        prev_s = args.anchor;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < args.blk; ++i) {
        const uint prev = prev_s;
        for (uint t = tid; t < r; t += DS_THREADS) {
            e[t] = float(w1[(ulong)prev * r + t]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = sg; j < k; j += DS_GROUPS) {
            device const T * row = w2 + ((ulong)i * k + j) * r;
            float acc = 0.0f;
            for (uint t = lane; t < r; t += 32) {
                acc += float(row[t]) * e[t];
            }
            acc = simd_sum(acc);
            if (lane == 0) {
                score[j] = logits[(ulong)i * k + j] + float(T(acc));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Argmax, ties to the lower index.
        float v = -INFINITY;
        uint bj = k;
        for (uint j = tid; j < k; j += DS_THREADS) {
            if (score[j] > v) {
                v = score[j];
                bj = j;
            }
        }
        for (ushort offset = 16; offset > 0; offset >>= 1) {
            const float ov = simd_shuffle_down(v, offset);
            const uint oj = simd_shuffle_down(bj, offset);
            if (ov > v || (ov == v && oj < bj)) {
                v = ov;
                bj = oj;
            }
        }
        if (lane == 0) {
            best_v[sg] = v;
            best_j[sg] = bj;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float bv = best_v[0];
            uint bb = best_j[0];
            for (uint w = 1; w < DS_GROUPS; ++w) {
                if (best_v[w] > bv || (best_v[w] == bv && best_j[w] < bb)) {
                    bv = best_v[w];
                    bb = best_j[w];
                }
            }
            const uint tok = ids[(ulong)i * k + bb];
            prevs[i] = prev;
            tokens[i] = tok;
            prev_s = tok;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define DS_MARKOV_CHAIN(T, TNAME)                                                              \
    template [[host_name("dspark_markov_chain_" TNAME)]] kernel void dspark_markov_chain<T>(   \
        device const uint *, device const float *, device const T *, device const T *,         \
        device uint *, device uint *, constant ds_args &, uint, uint, uint);

DS_MARKOV_CHAIN(float, "f32")
DS_MARKOV_CHAIN(half, "f16")
#if defined(__HAVE_BFLOAT__)
DS_MARKOV_CHAIN(bfloat, "bf16")
#endif
