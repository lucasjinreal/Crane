// Exact top-k over the rows of an f32 matrix: the SYCL port of
// `kernels/metal/topk.metal` / `kernels/cuda/topk.cu` (see there for the
// design). Built into `libcrane_gdn_sycl.so` by `crane-core/build.rs`;
// launcher: `src/ops/fused_ops/sycl_topk.rs`.
//
// Indices come out ordered by value descending and, for equal values, index
// ascending: every element folds into one 64-bit key
//     key = order_preserving_u32(value) << 32 | ~index
// so no two elements compare equal and the whole kernel is integer compares.
//
//   stage 1: work-groups (row, span), each reducing a contiguous span of its
//            row to its own top-k, emitted as keys.
//   stage 2: one work-group per row reducing those keys to indices.
//
// The block routine is a threshold scan: a work-item touches local memory
// only for an element that beats the running k-th best; a flush bitonic-sorts
// the used prefix of the buffer and republishes the threshold.
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>

namespace {

// Keys a work-group buffers (16 KB). Power of two for the bitonic network.
constexpr uint32_t TOPK_CAP = 2048;
// Work-items per work-group; both stages use exactly this.
constexpr uint32_t TOPK_THREADS = 256;
// Elements each work-item loads per iteration.
constexpr uint32_t TOPK_UNROLL = 4;

using local_count =
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space>;

// -0.0 is normalised to +0.0 (a float compare calls them equal). A key is
// never 0 for a non-NaN value, so 0 is the empty-slot sentinel.
inline uint64_t topk_key(float v, uint32_t i) {
  uint32_t u = (v == 0.0f) ? 0u : sycl::bit_cast<uint32_t>(v);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return (uint64_t(u) << 32) | uint64_t(~i);
}

inline uint32_t topk_pow2_ceil(uint32_t v) {
  uint32_t p = 1u;
  while (p < v) p <<= 1;
  return p;
}

// Local state of one work-group's scan.
struct Scratch {
  uint64_t *buf;  // TOPK_CAP keys: best k in [0, k), staging after
  uint32_t *count;
  uint64_t *thresh;
};

// Bitonic sort of s[0 .. p) into descending order; p a power of two.
inline void topk_bitonic_desc(const sycl::nd_item<2> &it, uint64_t *s, uint32_t p,
                              uint32_t tid) {
  for (uint32_t size = 2u; size <= p; size <<= 1) {
    for (uint32_t stride = size >> 1; stride > 0u; stride >>= 1) {
      for (uint32_t i = tid; i < p; i += TOPK_THREADS) {
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
      sycl::group_barrier(it.get_group());
    }
  }
}

// Merge the staging area into the best k and republish the threshold. Every
// work-item reads the count before the network, whose barriers order that
// read before work-item 0's reset (p == 1 implies count == 0, a no-op reset).
inline void topk_flush(const sycl::nd_item<2> &it, const Scratch &sc, uint32_t k,
                       uint32_t tid) {
  const uint32_t count = local_count(*sc.count).load();
  const uint32_t p = topk_pow2_ceil(k + count);
  topk_bitonic_desc(it, sc.buf, p, tid);
  for (uint32_t i = k + tid; i < p; i += TOPK_THREADS) sc.buf[i] = 0ull;
  if (tid == 0) {
    local_count(*sc.count).store(0u);
    *sc.thresh = sc.buf[k - 1];
  }
  sycl::group_barrier(it.get_group());
}

// Reduce elements start .. end (f32 values of `x` when !KEYS, keys of `keys`
// when KEYS) to the work-group's top k, left sorted in sc.buf[0 .. k).
template <bool KEYS>
inline void topk_block_scan(const sycl::nd_item<2> &it, const float *x, const uint64_t *keys,
                            uint32_t start, uint32_t end, uint32_t k, const Scratch &sc) {
  const uint32_t tid = it.get_local_id(1);
  const uint32_t step = TOPK_THREADS * TOPK_UNROLL;
  const uint32_t stage = TOPK_CAP - k;

  for (uint32_t i = tid; i < TOPK_CAP; i += TOPK_THREADS) sc.buf[i] = 0ull;
  if (tid == 0) {
    local_count(*sc.count).store(0u);
    *sc.thresh = 0ull;
  }
  sycl::group_barrier(it.get_group());

  const uint32_t span = end > start ? end - start : 0u;
  const uint32_t iters = (span + step - 1u) / step;
  for (uint32_t iter = 0u; iter < iters; ++iter) {
    const uint64_t thresh = *sc.thresh;
    const uint32_t base = start + iter * step;
    uint64_t key[TOPK_UNROLL];
#pragma unroll
    for (uint32_t u = 0u; u < TOPK_UNROLL; ++u) {
      const uint32_t i = base + u * TOPK_THREADS + tid;
      if constexpr (KEYS) {
        key[u] = i < end ? keys[i] : 0ull;
      } else {
        key[u] = i < end ? topk_key(x[i], i) : 0ull;
      }
    }
#pragma unroll
    for (uint32_t u = 0u; u < TOPK_UNROLL; ++u) {
      if (key[u] > thresh) {
        const uint32_t slot = local_count(*sc.count).fetch_add(1u);
        sc.buf[k + slot] = key[u];
      }
    }
    sycl::group_barrier(it.get_group());
    // Flush before the next iteration could overflow; uniform across the
    // work-group, so every work-item takes the same branch.
    if (local_count(*sc.count).load() + step > stage) topk_flush(it, sc, k, tid);
  }
  topk_flush(it, sc, k, tid);
}

struct Local {
  sycl::local_accessor<uint64_t, 1> buf;
  sycl::local_accessor<uint32_t, 1> count;
  sycl::local_accessor<uint64_t, 1> thresh;
  explicit Local(sycl::handler &h) : buf(TOPK_CAP, h), count(1, h), thresh(1, h) {}
  Scratch get() const {
    return {buf.get_multi_ptr<sycl::access::decorated::no>().get(),
            count.get_multi_ptr<sycl::access::decorated::no>().get(),
            thresh.get_multi_ptr<sycl::access::decorated::no>().get()};
  }
};

}  // namespace

// Top-`k` indices of every row of the contiguous f32 matrix `x` `[rows, n]`
// into `out` `[rows, k]` (u32). `keys` is scratch for `rows * groups * k`
// u64s; each of the `groups` stage-1 work-groups of a row scans
// `items_per_group` elements. Requires 0 < k <= TOPK_CAP / 4. Returns 0 on
// success, 1 on a SYCL error, 2 on bad arguments.
extern "C" int crane_topk_rows_sycl(void *queue, const float *x, uint64_t *keys, uint32_t *out,
                                    uint32_t rows, uint32_t n, uint32_t k, uint32_t groups,
                                    uint32_t items_per_group) {
  if (k == 0 || k > TOPK_CAP / 4 || k > n || rows == 0 || groups == 0) return 2;
  try {
    auto &q = *static_cast<sycl::queue *>(queue);
    q.submit([&](sycl::handler &h) {
      Local loc(h);
      h.parallel_for(
          sycl::nd_range<2>({rows, size_t(groups) * TOPK_THREADS}, {1, TOPK_THREADS}),
          [=](sycl::nd_item<2> it) {
            const Scratch sc = loc.get();
            const uint32_t row = it.get_group(0);
            const uint32_t g = it.get_group(1);
            const uint32_t start = g * items_per_group;
            const uint32_t end = sycl::min(start + items_per_group, n);
            topk_block_scan<false>(it, x + uint64_t(row) * n, nullptr, start, end, k, sc);
            // An empty span still writes its k sentinels; stage 2 drops them
            // because a sentinel never beats the initial threshold.
            uint64_t *dst = keys + (uint64_t(row) * groups + g) * k;
            for (uint32_t j = it.get_local_id(1); j < k; j += TOPK_THREADS) dst[j] = sc.buf[j];
          });
    });
    const uint32_t per_row = groups * k;
    q.submit([&](sycl::handler &h) {
      Local loc(h);
      h.parallel_for(sycl::nd_range<2>({rows, TOPK_THREADS}, {1, TOPK_THREADS}),
                     [=](sycl::nd_item<2> it) {
                       const Scratch sc = loc.get();
                       const uint32_t row = it.get_group(0);
                       topk_block_scan<true>(it, nullptr, keys + uint64_t(row) * per_row, 0u,
                                             per_row, k, sc);
                       uint32_t *dst = out + uint64_t(row) * k;
                       for (uint32_t j = it.get_local_id(1); j < k; j += TOPK_THREADS)
                         dst[j] = ~uint32_t(sc.buf[j]);
                     });
    });
    return 0;
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "crane_topk_rows_sycl: %s\n", e.what());
    return 1;
  }
}
