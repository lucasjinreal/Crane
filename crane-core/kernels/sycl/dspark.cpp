// DSpark drafter: the sequential greedy Markov chain in one submission, the
// SYCL port of `kernels/metal/dspark.metal` / `kernels/cuda/dspark.cu`. Built
// into `libcrane_gdn_sycl.so` by `crane-core/build.rs`; launcher:
// `src/models/dspark/sycl.rs`.
//
// For each block position i, with `prev` the previous token (the anchor for
// i = 0), score every one of the k candidates as
//     logits[i, j] + T(dot(W2[i, j, :], W1[prev, :]))
// and pick the best; its token id becomes `prev` for position i + 1. The dot
// product is rounded to the weight dtype T like the matmul it replaces. Ties
// go to the lowest candidate index (the higher base logit: candidates are
// sorted).
//
// One work-group of DS_THREADS work-items does all positions. Each sub-group
// scores a strided subset of the candidates, its lanes reading consecutive
// weights of a candidate's W2 row. The argmax folds (score, candidate) into
// one order-preserving u64 key (see `kernels/sycl/topk.cpp`), so a single
// group max picks the best score and, among equal ones, the lowest index.
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>

namespace {

using bf16 = sycl::ext::oneapi::bfloat16;

constexpr uint32_t DS_THREADS = 256;
constexpr uint32_t DS_MAX_RANK = 1024;
constexpr uint32_t DS_MAX_K = 1024;

// dtype tags match `src/models/dspark/sycl.rs`.
enum { DS_F32 = 0, DS_F16 = 1, DS_BF16 = 2 };

// Larger key = larger score, then lower index; -0.0 counts as +0.0.
inline uint64_t ds_key(float v, uint32_t j) {
  uint32_t u = (v == 0.0f) ? 0u : sycl::bit_cast<uint32_t>(v);
  u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return (uint64_t(u) << 32) | uint64_t(~j);
}

template <typename T>
void markov_chain_launch(sycl::queue &q, const uint32_t *ids, const float *logits, const T *w2,
                         const T *w1, uint32_t *tokens, uint32_t *prevs, uint32_t anchor,
                         uint32_t blk, uint32_t k, uint32_t r) {
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> e(DS_MAX_RANK, h);
    sycl::local_accessor<float, 1> score(DS_MAX_K, h);
    sycl::local_accessor<uint32_t, 1> prev_s(1, h);
    h.parallel_for(
        sycl::nd_range<1>(DS_THREADS, DS_THREADS), [=](sycl::nd_item<1> it) {
          const auto grp = it.get_group();
          const auto sg = it.get_sub_group();
          const uint32_t tid = it.get_local_id(0);
          const uint32_t lane = sg.get_local_linear_id();
          const uint32_t sg_size = sg.get_local_linear_range();
          const uint32_t n_sg = sg.get_group_linear_range();
          const uint32_t sg_id = sg.get_group_linear_id();
          if (tid == 0) prev_s[0] = anchor;
          sycl::group_barrier(grp);

          for (uint32_t i = 0; i < blk; ++i) {
            const uint32_t prev = prev_s[0];
            for (uint32_t t = tid; t < r; t += DS_THREADS)
              e[t] = static_cast<float>(w1[uint64_t(prev) * r + t]);
            sycl::group_barrier(grp);

            // Uniform trip count per sub-group, so every lane reaches the
            // sub-group reduction.
            for (uint32_t j = sg_id; j < k; j += n_sg) {
              const T *row = w2 + (uint64_t(i) * k + j) * r;
              float acc = 0.0f;
              for (uint32_t t = lane; t < r; t += sg_size)
                acc += static_cast<float>(row[t]) * e[t];
              acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
              if (lane == 0)
                score[j] = logits[uint64_t(i) * k + j] +
                           static_cast<float>(static_cast<T>(acc));
            }
            sycl::group_barrier(grp);

            uint64_t best = 0;
            for (uint32_t j = tid; j < k; j += DS_THREADS)
              best = sycl::max(best, ds_key(score[j], j));
            best = sycl::reduce_over_group(grp, best, sycl::maximum<uint64_t>());
            if (tid == 0) {
              const uint32_t bj = ~uint32_t(best);
              const uint32_t tok = ids[uint64_t(i) * k + bj];
              prevs[i] = prev;
              tokens[i] = tok;
              prev_s[0] = tok;
            }
            sycl::group_barrier(grp);
          }
        });
  });
}

}  // namespace

// `ids` `[blk, k]` u32, `logits` `[blk, k]` f32, `w2` `[blk, k, r]` and `w1`
// `[vocab, r]` in `dtype`; writes `tokens` / `prevs` `[blk]` u32. Requires
// k <= DS_MAX_K and r <= DS_MAX_RANK. Returns 0 on success, 1 on a SYCL error,
// 2 on bad arguments.
extern "C" int crane_dspark_markov_chain_sycl(void *queue, int dtype, const uint32_t *ids,
                                              const float *logits, const void *w2,
                                              const void *w1, uint32_t *tokens, uint32_t *prevs,
                                              uint32_t anchor, uint32_t blk, uint32_t k,
                                              uint32_t r) {
  if (k == 0 || k > DS_MAX_K || r > DS_MAX_RANK) return 2;
  try {
    auto &q = *static_cast<sycl::queue *>(queue);
    switch (dtype) {
    case DS_F32:
      markov_chain_launch<float>(q, ids, logits, static_cast<const float *>(w2),
                                 static_cast<const float *>(w1), tokens, prevs, anchor, blk, k,
                                 r);
      return 0;
    case DS_F16:
      markov_chain_launch<sycl::half>(q, ids, logits, static_cast<const sycl::half *>(w2),
                                      static_cast<const sycl::half *>(w1), tokens, prevs,
                                      anchor, blk, k, r);
      return 0;
    case DS_BF16:
      markov_chain_launch<bf16>(q, ids, logits, static_cast<const bf16 *>(w2),
                                static_cast<const bf16 *>(w1), tokens, prevs, anchor, blk, k,
                                r);
      return 0;
    default:
      return 2;
    }
  } catch (const sycl::exception &e) {
    std::fprintf(stderr, "crane_dspark_markov_chain_sycl: %s\n", e.what());
    return 1;
  }
}
