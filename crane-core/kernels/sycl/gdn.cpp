// Fused Gated Delta Net recurrence for the Intel SYCL backend — the counterpart
// of `kernels/cuda/gdn.cu`. Same math, same contiguous-f32 `[BH, S, *]` layouts.
// Built into `libcrane_gdn_sycl.so` by `crane-core/build.rs` (icpx, `--features
// sycl` only) and driven by `ops/gdn/sycl_backend.rs`.
//
// One work-item owns its state column `S[:, vcol]` (K elements, private) and
// steps through the whole sequence — each value column of each (batch*head) is
// an independent sequential recurrence. This collapses the per-timestep candle
// op graph into one submission per GDN layer per forward pass.
//
// v0: no local-memory staging — every work-item reads `k_t` / `q_t` straight
// from global. Redundant bandwidth, but no barrier and easy to trust. The
// shared-memory tiling is the next optimisation once numerics are verified.
//
// Layouts (all contiguous f32):
//   q, k     : [BH, S, K]   (q already pre-scaled by 1/sqrt(K) by the caller)
//   v, y     : [BH, S, V]
//   g, beta  : [BH, S]      (g is the log-decay; decay = exp(g))
//   state    : [BH, K, V]
//
// Recurrence per timestep t (matches the CPU reference exactly):
//   S      *= exp(g_t)
//   kv_mem  = sum_k S[k,:] * k_t[k]
//   delta   = (v_t - kv_mem) * beta_t
//   S[k,:] += k_t[k] * delta
//   y_t     = sum_k S[k,:] * q_t[k]
#include <sycl/sycl.hpp>

#define GDN_MAX_K 256

template <int KT>
static void gdn_launch(sycl::queue &q, const float *qp, const float *kp,
                       const float *vp, const float *gp, const float *bp,
                       const float *st_in, float *st_out, float *yp, int BH,
                       int S, int Kr, int V, int V_TILE) {
  const int K = (KT > 0) ? KT : Kr;
  const int tiles = (V + V_TILE - 1) / V_TILE;
  const std::size_t local = static_cast<std::size_t>(V_TILE);
  const std::size_t groups = static_cast<std::size_t>(BH) * tiles;

  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * local),
                          sycl::range<1>(local)),
        [=](sycl::nd_item<1> it) {
          const int gid = static_cast<int>(it.get_global_id(0));
          const int bh = gid / V_TILE / tiles;
          const int rem = gid / V_TILE % tiles;
          const int lid = gid % V_TILE;
          const int vcol = rem * V_TILE + lid;
          if (bh >= BH || vcol >= V)
            return;

          float Scol[(KT > 0) ? KT : GDN_MAX_K];
          const float *sti = st_in + static_cast<long long>(bh) * K * V;
          for (int kk = 0; kk < K; ++kk)
            Scol[kk] = sti[kk * V + vcol];

          const float *qb = qp + static_cast<long long>(bh) * S * K;
          const float *kb = kp + static_cast<long long>(bh) * S * K;
          const float *vb = vp + static_cast<long long>(bh) * S * V;
          const float *gb = gp + static_cast<long long>(bh) * S;
          const float *bb = bp + static_cast<long long>(bh) * S;
          float *yb = yp + static_cast<long long>(bh) * S * V;

          for (int t = 0; t < S; ++t) {
            const float decay = sycl::exp(gb[t]);
            const float beta_t = bb[t];
            const float v_t = vb[t * V + vcol];
            const float *kt = kb + static_cast<long long>(t) * K;
            const float *qt = qb + static_cast<long long>(t) * K;

            float kv = 0.f;
            for (int kk = 0; kk < K; ++kk) {
              Scol[kk] *= decay;
              kv += Scol[kk] * kt[kk];
            }
            const float delta = (v_t - kv) * beta_t;

            float y = 0.f;
            for (int kk = 0; kk < K; ++kk) {
              Scol[kk] += kt[kk] * delta;
              y += Scol[kk] * qt[kk];
            }
            yb[t * V + vcol] = y;
          }

          float *sto = st_out + static_cast<long long>(bh) * K * V;
          for (int kk = 0; kk < K; ++kk)
            sto[kk * V + vcol] = Scol[kk];
        });
  });
}

extern "C" int crane_gdn_recurrence_sycl(void *queue, const float *q,
                                        const float *k, const float *v,
                                        const float *g, const float *beta,
                                        const float *state_in, float *state_out,
                                        float *y, int BH, int S, int K, int V,
                                        int V_TILE) {
  try {
    auto &sq = *static_cast<sycl::queue *>(queue);
    if (V_TILE <= 0 || V_TILE > V)
      V_TILE = V;
    if (K == 128)
      gdn_launch<128>(sq, q, k, v, g, beta, state_in, state_out, y, BH, S, 128, V,
                      V_TILE);
    else if (K > 0 && K <= GDN_MAX_K)
      gdn_launch<0>(sq, q, k, v, g, beta, state_in, state_out, y, BH, S, K, V,
                    V_TILE);
    else
      return 2; // unsupported head_k_dim
    // No wait: candle's SYCL queue is in-order, so the result buffers are
    // correctly ordered against every later op the launcher queues on it.
    return 0;
  } catch (const sycl::exception &) {
    return 1;
  } catch (...) {
    return 1;
  }
}
