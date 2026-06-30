// Fused Gated Delta Net recurrence (CUDA), f32.
//
// Each value column of each (batch*head) is an INDEPENDENT sequential
// recurrence (no coupling across V). One thread owns its state column
// S[:, vcol] (K elements) and steps through the whole sequence, staging
// k_t/q_t in shared memory per step. This collapses the per-timestep Candle op
// graph (thousands of tiny launches) into a single kernel launch.
//
// Two variants:
//   * gdn_recurrence_f32_k<K>  — K is a compile-time constant, so the state
//     column lives in REGISTERS (fully unrolled). Used for the common head
//     dims (128). This is the fast path: the inner loop is the serial
//     bottleneck and registers remove the local-memory traffic.
//   * gdn_recurrence_f32       — runtime K fallback (state column in local
//     memory) for head dims without a specialization.
//
// V columns may be tiled across blocks (V_TILE) but for these shapes one block
// per head (V_TILE == V, ~4 warps) hides latency best, so the launcher defaults
// there; the tiling knob is kept for other shapes.
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

// Shared logic; K is either a compile-time constant (registers) or runtime.
template <int KT>
__device__ __forceinline__ void gdn_run(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    const float* __restrict__ state_in,
    float* __restrict__ state_out,
    float* __restrict__ y,
    int BH, int S, int Kr, int V, int V_TILE)
{
    const int K = (KT > 0) ? KT : Kr;
    const int tiles = (V + V_TILE - 1) / V_TILE;
    const int bh = blockIdx.x / tiles;
    const int tile = blockIdx.x % tiles;
    const int vcol = tile * V_TILE + threadIdx.x;
    if (bh >= BH) return;
    const bool active = vcol < V;

    float Scol[(KT > 0) ? KT : GDN_MAX_K];
    const float* st_in = state_in + (long long)bh * K * V;
    if (active) {
        #pragma unroll
        for (int kk = 0; kk < K; ++kk) Scol[kk] = st_in[kk * V + vcol];
    }

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

        if (active) {
            const float decay = expf(gb[t]);
            const float beta_t = bb[t];
            const float v_t = vb[t * V + vcol];

            // 4 independent accumulators break the 128-long reduction
            // dependency chain (more ILP to hide FMA latency). Element-wise
            // Scol updates are independent and stay in registers.
            float kv0 = 0.f, kv1 = 0.f, kv2 = 0.f, kv3 = 0.f;
            int kk = 0;
            #pragma unroll
            for (; kk + 3 < K; kk += 4) {
                Scol[kk] *= decay;     kv0 += Scol[kk] * k_sh[kk];
                Scol[kk + 1] *= decay; kv1 += Scol[kk + 1] * k_sh[kk + 1];
                Scol[kk + 2] *= decay; kv2 += Scol[kk + 2] * k_sh[kk + 2];
                Scol[kk + 3] *= decay; kv3 += Scol[kk + 3] * k_sh[kk + 3];
            }
            for (; kk < K; ++kk) { Scol[kk] *= decay; kv0 += Scol[kk] * k_sh[kk]; }
            const float kv_mem = (kv0 + kv1) + (kv2 + kv3);
            const float delta = (v_t - kv_mem) * beta_t;

            float y0 = 0.f, y1 = 0.f, y2 = 0.f, y3 = 0.f;
            kk = 0;
            #pragma unroll
            for (; kk + 3 < K; kk += 4) {
                Scol[kk] += k_sh[kk] * delta;         y0 += Scol[kk] * q_sh[kk];
                Scol[kk + 1] += k_sh[kk + 1] * delta; y1 += Scol[kk + 1] * q_sh[kk + 1];
                Scol[kk + 2] += k_sh[kk + 2] * delta; y2 += Scol[kk + 2] * q_sh[kk + 2];
                Scol[kk + 3] += k_sh[kk + 3] * delta; y3 += Scol[kk + 3] * q_sh[kk + 3];
            }
            for (; kk < K; ++kk) { Scol[kk] += k_sh[kk] * delta; y0 += Scol[kk] * q_sh[kk]; }
            yb[t * V + vcol] = (y0 + y1) + (y2 + y3);
        }
        __syncthreads();
    }

    if (active) {
        float* st_out = state_out + (long long)bh * K * V;
        #pragma unroll
        for (int kk = 0; kk < K; ++kk) st_out[kk * V + vcol] = Scol[kk];
    }
}

// Compile-time-K fast path: state column in registers.
extern "C" __global__ void gdn_recurrence_f32_k128(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, const float* __restrict__ state_in,
    float* __restrict__ state_out, float* __restrict__ y,
    int BH, int S, int V, int V_TILE)
{
    gdn_run<128>(q, k, v, g, beta, state_in, state_out, y, BH, S, 128, V, V_TILE);
}

// Runtime-K fallback: state column in local memory.
extern "C" __global__ void gdn_recurrence_f32(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, const float* __restrict__ state_in,
    float* __restrict__ state_out, float* __restrict__ y,
    int BH, int S, int K, int V, int V_TILE)
{
    gdn_run<0>(q, k, v, g, beta, state_in, state_out, y, BH, S, K, V, V_TILE);
}
