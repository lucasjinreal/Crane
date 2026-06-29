// Fused Gated Delta Net recurrence (CUDA), f32.
//
// One thread block per (batch * value-head); one thread per value column.
// Each thread owns state column S[:, vcol] (K elements) in a per-thread array,
// and steps through the whole sequence — collapsing the per-timestep Candle op
// graph (thousands of tiny launches) into a single kernel launch.
//
// Layouts (all contiguous, f32):
//   q, k     : [BH, S, K]   (q already pre-scaled by 1/sqrt(K) by the caller)
//   v, y     : [BH, S, V]
//   g, beta  : [BH, S]      (g is the log-decay; decay = exp(g))
//   state    : [BH, K, V]   (state_in read once, state_out written once)
//
// Recurrence per timestep t (matches the CPU reference exactly):
//   S      *= exp(g_t)
//   kv_mem  = sum_k S[k,:] * k_t[k]
//   delta   = (v_t - kv_mem) * beta_t
//   S[k,:] += k_t[k] * delta
//   y_t     = sum_k S[k,:] * q_t[k]

#include <cuda_runtime.h>

#define GDN_MAX_K 256

extern "C" __global__ void gdn_recurrence_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    const float* __restrict__ state_in,
    float* __restrict__ state_out,
    float* __restrict__ y,
    int BH, int S, int K, int V)
{
    const int bh = blockIdx.x;
    const int vcol = threadIdx.x;
    if (bh >= BH || vcol >= V) return;

    // Per-thread state column S[:, vcol].
    float Scol[GDN_MAX_K];
    const float* st_in = state_in + (long long)bh * K * V;
    for (int kk = 0; kk < K; ++kk) Scol[kk] = st_in[kk * V + vcol];

    // Shared k_t / q_t for the current timestep (2*K floats).
    extern __shared__ float sh[];
    float* k_sh = sh;
    float* q_sh = sh + K;

    const float* qb = q + (long long)bh * S * K;
    const float* kb = k + (long long)bh * S * K;
    const float* vb = v + (long long)bh * S * V;
    const float* gb = g + (long long)bh * S;
    const float* bb = beta + (long long)bh * S;
    float* yb = y + (long long)bh * S * V;

    for (int t = 0; t < S; ++t) {
        for (int kk = threadIdx.x; kk < K; kk += blockDim.x) {
            k_sh[kk] = kb[t * K + kk];
            q_sh[kk] = qb[t * K + kk];
        }
        __syncthreads();

        const float decay = expf(gb[t]);
        const float beta_t = bb[t];
        const float v_t = vb[t * V + vcol];

        float kv_mem = 0.f;
        #pragma unroll 4
        for (int kk = 0; kk < K; ++kk) {
            Scol[kk] *= decay;
            kv_mem += Scol[kk] * k_sh[kk];
        }
        const float delta = (v_t - kv_mem) * beta_t;

        float y_t = 0.f;
        #pragma unroll 4
        for (int kk = 0; kk < K; ++kk) {
            Scol[kk] += k_sh[kk] * delta;
            y_t += Scol[kk] * q_sh[kk];
        }
        yb[t * V + vcol] = y_t;
        __syncthreads();
    }

    float* st_out = state_out + (long long)bh * K * V;
    for (int kk = 0; kk < K; ++kk) st_out[kk * V + vcol] = Scol[kk];
}
