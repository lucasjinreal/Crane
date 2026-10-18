#include <metal_stdlib>
using namespace metal;

// Exact top-k over the rows of an f32 matrix: the Metal port of
// `kernels/cuda/topk.cu` (see there for the design). Launcher:
// `src/ops/fused_ops/metal_topk.rs`.
//
// Indices come out ordered by value descending and, for equal values, index
// ascending: every element folds into one 64-bit key
//     key = order_preserving_u32(value) << 32 | ~index
// so no two elements compare equal and the whole kernel is integer compares.
//
//   stage 1: threadgroups (span, row), each reducing a contiguous span of its
//            row to its own top-k, emitted as keys.
//   stage 2: one threadgroup per row reducing those keys to indices.
//
// The block routine is a threshold scan: a thread touches threadgroup memory
// only for an element that beats the running k-th best; a flush bitonic-sorts
// the used prefix of the buffer and republishes the threshold.

// Keys a threadgroup buffers (16 KB). Power of two for the bitonic network.
#define TOPK_CAP 2048u
// Threads per threadgroup; both stages use exactly this.
#define TOPK_THREADS 256u
// Elements each thread loads per iteration.
#define TOPK_UNROLL 4u

// -0.0 is normalised to +0.0 (a float compare calls them equal). A key is
// never 0 for a non-NaN value, so 0 is the empty-slot sentinel.
inline ulong topk_key(float v, uint i) {
    uint u = (v == 0.0f) ? 0u : as_type<uint>(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (ulong(u) << 32) | ulong(~i);
}

inline uint topk_pow2_ceil(uint v) {
    uint p = 1u;
    while (p < v) {
        p <<= 1;
    }
    return p;
}

// Bitonic sort of s[0 .. p) into descending order; p a power of two.
inline void topk_bitonic_desc(threadgroup ulong * s, uint p, uint tid) {
    for (uint size = 2u; size <= p; size <<= 1) {
        for (uint stride = size >> 1; stride > 0u; stride >>= 1) {
            for (uint i = tid; i < p; i += TOPK_THREADS) {
                const uint j = i ^ stride;
                if (j > i) {
                    const bool desc = (i & size) == 0u;
                    const ulong a = s[i];
                    const ulong b = s[j];
                    if ((a < b) == desc) {
                        s[i] = b;
                        s[j] = a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// Merge the staging area into the best k and republish the threshold. Every
// thread reads the count before the network, whose barriers order that read
// before thread 0's reset (p == 1 implies count == 0, a no-op reset).
inline void topk_flush(threadgroup ulong * s_buf, uint k, threadgroup atomic_uint * s_count,
                       threadgroup ulong * s_thresh, uint tid) {
    const uint count = atomic_load_explicit(s_count, memory_order_relaxed);
    const uint p = topk_pow2_ceil(k + count);
    topk_bitonic_desc(s_buf, p, tid);
    for (uint i = k + tid; i < p; i += TOPK_THREADS) {
        s_buf[i] = 0ul;
    }
    if (tid == 0) {
        atomic_store_explicit(s_count, 0u, memory_order_relaxed);
        *s_thresh = s_buf[k - 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Reduce elements start .. end (f32 values of `x` when !KEYS, keys of `keys`
// when KEYS) to the threadgroup's top k, left sorted in s_buf[0 .. k).
template <bool KEYS>
inline void topk_block_scan(device const float * x, device const ulong * keys, uint start,
                            uint end, uint k, threadgroup ulong * s_buf,
                            threadgroup atomic_uint * s_count, threadgroup ulong * s_thresh,
                            uint tid) {
    const uint step = TOPK_THREADS * TOPK_UNROLL;
    const uint stage = TOPK_CAP - k;

    for (uint i = tid; i < TOPK_CAP; i += TOPK_THREADS) {
        s_buf[i] = 0ul;
    }
    if (tid == 0) {
        atomic_store_explicit(s_count, 0u, memory_order_relaxed);
        *s_thresh = 0ul;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint span = end > start ? end - start : 0u;
    const uint iters = (span + step - 1u) / step;
    for (uint it = 0u; it < iters; ++it) {
        const ulong thresh = *s_thresh;
        const uint base = start + it * step;
        ulong key[TOPK_UNROLL];
        for (uint u = 0u; u < TOPK_UNROLL; ++u) {
            const uint i = base + u * TOPK_THREADS + tid;
            key[u] = i < end ? (KEYS ? keys[i] : topk_key(x[i], i)) : 0ul;
        }
        for (uint u = 0u; u < TOPK_UNROLL; ++u) {
            if (key[u] > thresh) {
                const uint slot = atomic_fetch_add_explicit(s_count, 1u, memory_order_relaxed);
                s_buf[k + slot] = key[u];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Flush before the next iteration could overflow; uniform across the
        // threadgroup, so every thread takes the same branch.
        if (atomic_load_explicit(s_count, memory_order_relaxed) + step > stage) {
            topk_flush(s_buf, k, s_count, s_thresh, tid);
        }
    }
    topk_flush(s_buf, k, s_count, s_thresh, tid);
}

struct topk_args {
    uint n;               // row length (stage 1) / keys per row (stage 2)
    uint k;
    uint items_per_group; // stage 1 only
};

// Per-span top-k of row `tg.y`, as keys at out_keys[row][tg.x].
kernel void topk_stage1_f32(
        device const float * x        [[buffer(0)]],
        device ulong       * out_keys [[buffer(1)]],
        constant topk_args & args     [[buffer(2)]],
        uint2 tg  [[threadgroup_position_in_grid]],
        uint2 ntg [[threadgroups_per_grid]],
        uint  tid [[thread_index_in_threadgroup]]) {
    threadgroup ulong s_buf[TOPK_CAP];
    threadgroup atomic_uint s_count;
    threadgroup ulong s_thresh;

    const uint row = tg.y;
    const uint start = tg.x * args.items_per_group;
    const uint end = min(start + args.items_per_group, args.n);
    topk_block_scan<false>(x + (ulong)row * args.n, nullptr, start, end, args.k, s_buf,
                           &s_count, &s_thresh, tid);
    // An empty span still writes its k sentinels; stage 2 drops them because a
    // sentinel never beats the initial threshold.
    device ulong * dst = out_keys + ((ulong)row * ntg.x + tg.x) * args.k;
    for (uint j = tid; j < args.k; j += TOPK_THREADS) {
        dst[j] = s_buf[j];
    }
}

// Final reduction of the stage-1 keys of row `tg.y` to k indices.
kernel void topk_stage2_u64(
        device const ulong * in_keys [[buffer(0)]],
        device uint        * out_idx [[buffer(1)]],
        constant topk_args & args    [[buffer(2)]],
        uint2 tg  [[threadgroup_position_in_grid]],
        uint  tid [[thread_index_in_threadgroup]]) {
    threadgroup ulong s_buf[TOPK_CAP];
    threadgroup atomic_uint s_count;
    threadgroup ulong s_thresh;

    const uint row = tg.y;
    topk_block_scan<true>(nullptr, in_keys + (ulong)row * args.n, 0u, args.n, args.k, s_buf,
                          &s_count, &s_thresh, tid);
    device uint * dst = out_idx + (ulong)row * args.k;
    for (uint j = tid; j < args.k; j += TOPK_THREADS) {
        dst[j] = ~uint(s_buf[j]);
    }
}
