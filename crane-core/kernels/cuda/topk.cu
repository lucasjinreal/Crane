/**
 * Exact GPU top-k over an f32 row.
 *
 * Returns the indices of the `k` largest values of `x[0 .. n)`, ordered by
 * value descending and, where two values are equal, by index ascending. That
 * total order is the whole trick: every element folds into one 64-bit key
 *
 *     key = order_preserving_u32(value) << 32 | ~index
 *
 * so no two elements ever compare equal, a tie can neither duplicate nor drop
 * an index, and the entire kernel is unsigned integer comparison.
 *
 * Two launches, both running the same block routine:
 *   stage 1 — `gridDim.x` blocks, each reducing a contiguous span of the row
 *             to its own top-k, emitted as raw keys.
 *   stage 2 — one block reducing those `gridDim.x * k` keys to the final
 *             indices.
 *
 * The block routine is a threshold scan, not a per-thread candidate list. The
 * block holds its running best-k plus a staging area in LDS; a thread touches
 * LDS only for an element that beats the current k-th best. After the first
 * flush the threshold already sits near the row's true k-th value, so the rest
 * of the scan is one compare per element and the kernel runs at memory
 * bandwidth. A flush is a bitonic sort of the buffer's *used prefix* driven by
 * the whole block — there is no serial merge anywhere, which is what the
 * previous single-lane insertion merge cost.
 *
 * Portability: only `atomicAdd` on shared memory, `__syncthreads` and
 * `__float_as_uint` are used — identical on CUDA and HIP. No shuffles, no
 * wave-size assumption, no gfx-specific intrinsic. The two tuning constants
 * below are shaped for RDNA3 (16 KB of a workgroup's 64 KB LDS) but are
 * correctness-neutral on any target that can hold them.
 *
 * gridDim.y is the row axis and is currently always launched as 1; a batched
 * `[nrows, n]` variant is a launch change, not a kernel change.
 *
 * Where the time goes, measured at n = 248320, k = 40 on gfx1101: stage 1 is
 * ~1 µs (the row is only ~1 MB, so the scan is not the cost), stage 2 ~50 µs.
 * Stage 2 is one block merging `grid * k` keys, and its threshold starts at
 * the sentinel, so its first iteration stages everything and pays a full
 * 2048-element bitonic. The lever, if it ever matters, is a merge tree: the
 * stage-1 runs are already sorted, so pairs of them can be combined with a
 * bitonic *merge* (log2(2k) stages) instead of a bitonic *sort*.
 */

// Under HIP the runtime header is force-included by candle's shim, which does
// not shadow <cuda_runtime.h>; nvcc includes it implicitly but name it anyway.
#ifndef __HIP_PLATFORM_AMD__
#include <cuda_runtime.h>
#endif
#include <stdint.h>

// Keys a block buffers in LDS. Power of two (the bitonic network needs it) and
// 16 KB at 8 bytes per key. Bounds the supported k: see TOPK_MAX_K.
#define TOPK_CAP 2048u
// Threads per block. Both launches use exactly this.
#define TOPK_THREADS 256u
// Elements each thread loads per iteration. The loads are issued together so
// they pipeline; without it the per-iteration __syncthreads would leave one
// outstanding load per thread and the scan would run at latency, not
// bandwidth.
#define TOPK_UNROLL 4u
// Largest k a launcher may pass. Correctness needs only
// `k <= TOPK_CAP - TOPK_THREADS * TOPK_UNROLL`, so that the staging area can
// absorb one iteration's worth of pushes; a quarter of the buffer is well
// inside that and leaves the flush trigger some slack, which keeps large k
// from flushing on every single iteration.
#define TOPK_MAX_K (TOPK_CAP / 4)

// Total order over (value, index): value descending, index ascending.
//
// The float->uint map is the standard monotonic one (flip the sign bit for
// positives, invert everything for negatives). -0.0 is normalised to +0.0 so
// the order matches a host sort that compares floats, which calls them equal.
// A key is never 0 for a non-NaN value, so 0 serves as the "empty slot"
// sentinel and sorts below every real entry.
static __device__ __forceinline__ uint64_t topk_key(float v, uint32_t i) {
    uint32_t u = (v == 0.0f) ? 0u : __float_as_uint(v);
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return ((uint64_t)u << 32) | (uint64_t)(~i);
}

static __device__ __forceinline__ uint32_t topk_index_of(uint64_t key) {
    return ~(uint32_t)key;
}

static __device__ __forceinline__ uint32_t topk_pow2_ceil(uint32_t v) {
    uint32_t p = 1u;
    while (p < v) p <<= 1;
    return p;
}

// Bitonic sort of `s[0 .. p)` into descending order. `p` must be a power of
// two; the rest of the buffer is untouched.
static __device__ void topk_bitonic_desc(uint64_t *s, uint32_t p) {
    for (uint32_t size = 2u; size <= p; size <<= 1) {
        for (uint32_t stride = size >> 1; stride > 0u; stride >>= 1) {
            for (uint32_t i = threadIdx.x; i < p; i += blockDim.x) {
                const uint32_t j = i ^ stride;
                if (j > i) {
                    const bool desc = (i & size) == 0u;
                    const uint64_t a = s[i];
                    const uint64_t b = s[j];
                    if ((a < b) == desc) {
                        s[i] = b;
                        s[j] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Merge the staging area into the best-k and republish the threshold.
//
// Only the prefix the block actually filled is sorted: after the first flush
// the staging area holds a handful of keys, so this is a ~128-element network
// rather than a 2048-element one. Everything above the prefix is already the
// sentinel, because a flush clears what it demotes.
static __device__ void topk_flush(
    uint64_t *s_buf,
    uint32_t k,
    uint32_t *s_count,
    uint64_t *s_thresh
) {
    // Every thread reads the count before entering the network, and the network
    // barriers before thread 0 resets it — except when p == 1, which needs
    // k + count <= 1 and so can only mean count == 0, making the reset a no-op.
    const uint32_t p = topk_pow2_ceil(k + *s_count);
    topk_bitonic_desc(s_buf, p);
    for (uint32_t i = k + threadIdx.x; i < p; i += blockDim.x) {
        s_buf[i] = 0ull;
    }
    if (threadIdx.x == 0) {
        *s_count = 0u;
        *s_thresh = s_buf[k - 1];
    }
    __syncthreads();
}

// Reduce `fetch(start .. end)` to the block's top-k, left in `s_buf[0 .. k)`
// sorted descending. `Fetch` is a functor rather than a lambda so the source
// needs no nvcc extended-lambda flag.
template <typename Fetch>
static __device__ void topk_block_scan(
    Fetch fetch,
    uint32_t start,
    uint32_t end,
    uint32_t k,
    uint64_t *s_buf,
    uint32_t *s_count,
    uint64_t *s_thresh
) {
    const uint32_t t = blockDim.x;
    const uint32_t step = t * TOPK_UNROLL;
    const uint32_t stage = TOPK_CAP - k;

    for (uint32_t i = threadIdx.x; i < TOPK_CAP; i += t) {
        s_buf[i] = 0ull;
    }
    if (threadIdx.x == 0) {
        *s_count = 0u;
        *s_thresh = 0ull;
    }
    __syncthreads();

    const uint32_t span = (end > start) ? (end - start) : 0u;
    const uint32_t iters = (span + step - 1u) / step;

    for (uint32_t it = 0u; it < iters; ++it) {
        // Stable for the whole iteration: only a flush writes it, and a flush
        // only runs at the sync point below.
        const uint64_t thresh = *s_thresh;
        const uint32_t base = start + it * step;

        uint64_t key[TOPK_UNROLL];
#pragma unroll
        for (uint32_t u = 0u; u < TOPK_UNROLL; ++u) {
            const uint32_t i = base + u * t + threadIdx.x;
            key[u] = (i < end) ? fetch(i) : 0ull;
        }
#pragma unroll
        for (uint32_t u = 0u; u < TOPK_UNROLL; ++u) {
            if (key[u] > thresh) {
                const uint32_t slot = atomicAdd(s_count, 1u);
                s_buf[k + slot] = key[u];
            }
        }
        __syncthreads();

        // Flush before the next iteration *could* overflow, never after the
        // fact: that keeps every push in bounds without a retry path. The
        // condition is uniform across the block, so the call is too.
        if (*s_count + step > stage) {
            topk_flush(s_buf, k, s_count, s_thresh);
        }
    }
    topk_flush(s_buf, k, s_count, s_thresh);
}

struct TopkFetchF32 {
    const float *x;
    __device__ uint64_t operator()(uint32_t i) const { return topk_key(x[i], i); }
};

struct TopkFetchKeys {
    const uint64_t *keys;
    __device__ uint64_t operator()(uint32_t i) const { return keys[i]; }
};

// Per-block top-k of `x[row]`, emitted as keys at `out_keys[row][blockIdx.x]`.
extern "C" __global__ __launch_bounds__(TOPK_THREADS) void topk_stage1_f32(
    const float *__restrict__ x,
    const uint32_t n,
    const uint32_t k,
    const uint32_t items_per_block,
    uint64_t *__restrict__ out_keys
) {
    __shared__ uint64_t s_buf[TOPK_CAP];
    __shared__ uint32_t s_count;
    __shared__ uint64_t s_thresh;

    const uint32_t row = blockIdx.y;
    const uint32_t start = blockIdx.x * items_per_block;
    const uint32_t end = (start + items_per_block < n) ? (start + items_per_block) : n;

    TopkFetchF32 fetch = {x + (size_t)row * n};
    topk_block_scan(fetch, start, end, k, s_buf, &s_count, &s_thresh);

    // A block whose span is empty still writes its k sentinels; stage 2 filters
    // them out because a sentinel never beats the initial threshold.
    uint64_t *dst = out_keys + ((size_t)row * gridDim.x + blockIdx.x) * k;
    for (uint32_t j = threadIdx.x; j < k; j += blockDim.x) {
        dst[j] = s_buf[j];
    }
}

// Final reduction of the `m` stage-1 keys of one row to `k` indices.
extern "C" __global__ __launch_bounds__(TOPK_THREADS) void topk_stage2_u64(
    const uint64_t *__restrict__ in_keys,
    const uint32_t m,
    const uint32_t k,
    uint32_t *__restrict__ out_idx
) {
    __shared__ uint64_t s_buf[TOPK_CAP];
    __shared__ uint32_t s_count;
    __shared__ uint64_t s_thresh;

    const uint32_t row = blockIdx.y;

    TopkFetchKeys fetch = {in_keys + (size_t)row * m};
    topk_block_scan(fetch, 0u, m, k, s_buf, &s_count, &s_thresh);

    uint32_t *dst = out_idx + (size_t)row * k;
    for (uint32_t j = threadIdx.x; j < k; j += blockDim.x) {
        dst[j] = topk_index_of(s_buf[j]);
    }
}
