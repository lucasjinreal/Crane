#include <metal_stdlib>
using namespace metal;

// Gated Delta Net on Metal: the recurrence and the stages around it, so a GDN
// layer pass (a decode step, a speculative-decoding verification block, a
// prefill chunk) costs five dispatches between its input and output
// projections instead of ~45 (op-by-op) or ~15 per token (the per-timestep
// portable recurrence). Launchers: `src/ops/gdn/metal_backend.rs`.
//
//   gdn_recurrence_f32_k*   the gated delta rule over a whole pass (CUDA's
//                           `gdn_recurrence_f32_k128`, `kernels/cuda/gdn.cu`)
//   gdn_tri_inverse_f32     (I + kb)^-1 per chunk, for the chunked prefill path
//   gdn_conv_silu_*         causal conv1d (kernel 4) + silu, new conv state and
//                           an optional mid-pass checkpoint
//   gdn_prep_*              split Q|K|V, per-head L2 norm, 1/sqrt(K) scale,
//                           beta/g gates, and the [heads, S, dim] f32 layout
//   gdn_post_norm_*         recurrence output -> gated RMSNorm
//
// The stage kernels follow the op chain's rounding points in the activation
// dtype T (half or bfloat): a value the chain stores as T is rounded to T
// here too, everything in between is f32. Each token's arithmetic does not
// depend on how many tokens the pass holds, so a token decoded on its own and
// the same token inside a verification block see identical GDN math.

#define GDN_HEAD_DIM 128
#define GDN_CONV_K 4
// Threads cooperating on one state column in the recurrence.
#define GDN_LANES 4
#define GDN_MAX_CHUNK 64

// ─────────────────────────────────────────────────────────────────────
//  Recurrence
// ─────────────────────────────────────────────────────────────────────
//
// Layouts (contiguous f32): q, k [BH, S, K] (q pre-scaled by 1/sqrt(K)),
// v, y [BH, S, V], g, beta [BH, S], state [BH, K, V] (read once, written once
// to a fresh buffer: callers may hold the input state as a checkpoint).
//
// Per step: S *= exp(g); kv = S^T k; delta = (v - kv) * beta;
// S += k delta^T; y = S^T q.
//
// Value columns are independent; each is owned by GDN_LANES adjacent SIMD
// lanes holding K / GDN_LANES state entries each in registers, and the two
// K-long reductions finish with two lane shuffles (addition commutes exactly,
// so all lanes of a column agree on kv). A thread reads its own K-slice of
// k_t / q_t straight from device memory: every column of a head reads the
// same GDN_LANES slices, so these are cache hits, and with no threadgroup
// staging there is no barrier in the time loop. That lets a head's columns
// spread over several small threadgroups (GDN_REC_COLS each) instead of one
// large one, which keeps more of the GPU busy on the serial time loop.

#define GDN_REC_COLS 32

struct gdn_rec_args {
    uint s;
    uint v;
    uint tiles; // threadgroups per (batch, head)
};

template <uint K>
kernel void gdn_recurrence_f32(
        device const float * q         [[buffer(0)]],
        device const float * k         [[buffer(1)]],
        device const float * v         [[buffer(2)]],
        device const float * g         [[buffer(3)]],
        device const float * beta      [[buffer(4)]],
        device const float * state_in  [[buffer(5)]],
        device float       * state_out [[buffer(6)]],
        device float       * y         [[buffer(7)]],
        constant gdn_rec_args & args   [[buffer(8)]],
        uint tg  [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint KPT = K / GDN_LANES;
    constexpr uint KV4 = KPT / 4;

    const uint S = args.s;
    const uint V = args.v;
    const uint bh = tg / args.tiles;
    const uint col = (tg % args.tiles) * GDN_REC_COLS + tid / GDN_LANES;
    const uint k0 = (tid % GDN_LANES) * KPT;
    const bool active = col < V;

    device const float * st_in = state_in + (ulong)bh * K * V;
    float4 st[KV4];
    for (uint i = 0; i < KV4; ++i) {
        for (uint c = 0; c < 4; ++c) {
            st[i][c] = active ? st_in[(k0 + 4 * i + c) * V + col] : 0.0f;
        }
    }

    device const float * qb = q + (ulong)bh * S * K + k0;
    device const float * kb = k + (ulong)bh * S * K + k0;
    device const float * vb = v + (ulong)bh * S * V;
    device const float * gb = g + (ulong)bh * S;
    device const float * bb = beta + (ulong)bh * S;
    device float * yb = y + (ulong)bh * S * V;

    for (uint t = 0; t < S; ++t) {
        device const float4 * kt = (device const float4 *)(kb + (ulong)t * K);
        device const float4 * qt = (device const float4 *)(qb + (ulong)t * K);
        const float decay = exp(gb[t]);
        float kv = 0.0f;
        for (uint i = 0; i < KV4; ++i) {
            st[i] *= decay;
            kv += dot(st[i], kt[i]);
        }
        kv += simd_shuffle_xor(kv, 1);
        kv += simd_shuffle_xor(kv, 2);
        const float vt = active ? vb[t * V + col] : 0.0f;
        const float delta = (vt - kv) * bb[t];

        float acc = 0.0f;
        for (uint i = 0; i < KV4; ++i) {
            st[i] += kt[i] * delta;
            acc += dot(st[i], qt[i]);
        }
        acc += simd_shuffle_xor(acc, 1);
        acc += simd_shuffle_xor(acc, 2);
        if (active && k0 == 0) {
            yb[t * V + col] = acc;
        }
    }

    if (active) {
        device float * st_o = state_out + (ulong)bh * K * V;
        for (uint i = 0; i < KV4; ++i) {
            for (uint c = 0; c < 4; ++c) {
                st_o[(k0 + 4 * i + c) * V + col] = st[i][c];
            }
        }
    }
}

template [[host_name("gdn_recurrence_f32_k64")]] kernel void gdn_recurrence_f32<64>(
    device const float *, device const float *, device const float *, device const float *,
    device const float *, device const float *, device float *, device float *,
    constant gdn_rec_args &, uint, uint);
template [[host_name("gdn_recurrence_f32_k128")]] kernel void gdn_recurrence_f32<128>(
    device const float *, device const float *, device const float *, device const float *,
    device const float *, device const float *, device float *, device float *,
    constant gdn_rec_args &, uint, uint);

// Inverse of the unit lower-triangular (I + kb), kb strictly lower triangular
// ([B, cs, cs], cs <= GDN_MAX_CHUNK), by forward substitution row by row. One
// threadgroup per matrix, one thread per output column (columns are
// independent: a thread only ever reads back its own column, so no barrier);
// the partial inverse lives in threadgroup memory. Dispatched with exactly
// `cs` threads.
kernel void gdn_tri_inverse_f32(
        device const float * kb  [[buffer(0)]],
        device float       * out [[buffer(1)]],
        constant uint & cs       [[buffer(2)]],
        uint b [[threadgroup_position_in_grid]],
        uint t [[thread_position_in_threadgroup]]) {
    threadgroup float inv[GDN_MAX_CHUNK * GDN_MAX_CHUNK];
    device const float * m = kb + (ulong)b * cs * cs;
    for (uint i = 0; i < cs; ++i) {
        float acc = (t == i) ? 1.0f : 0.0f;
        for (uint j = 0; j < i; ++j) {
            acc -= m[i * cs + j] * inv[j * cs + t];
        }
        inv[i * cs + t] = acc;
    }
    device float * o = out + (ulong)b * cs * cs;
    for (uint i = 0; i < cs; ++i) {
        o[i * cs + t] = inv[i * cs + t];
    }
}

// ─────────────────────────────────────────────────────────────────────
//  Stages around the recurrence
// ─────────────────────────────────────────────────────────────────────

// Candle's f32 silu.
inline float silu_f32(float x) {
    return x / (1.0f + exp(-x));
}

// Causal conv1d + silu over x [S, C] (row stride xs) continuing from
// state [C, K]. `hidden[c, i]` is state[c, i] for i < K and x[i - K, c] after
// that; output t reads hidden[t + 1 .. t + K]. w: [C, K] f32, already rounded
// through T. One thread per (channel, row): rows 0..S are outputs, the next K
// the new state hidden[S .. S + K), the K after that (if `ckpt_at` > 0) the
// checkpoint hidden[ckpt_at .. ckpt_at + K).
struct gdn_conv_args {
    uint s;
    uint c;
    uint xs;
    uint ckpt_at;
};

template <typename T>
inline float conv_hidden(device const T * x, device const T * state, uint xs, uint c, uint i) {
    return i < GDN_CONV_K ? float(state[c * GDN_CONV_K + i])
                          : float(x[(ulong)(i - GDN_CONV_K) * xs + c]);
}

template <typename T>
kernel void gdn_conv_silu(
        device const T     * x         [[buffer(0)]],
        device const T     * state     [[buffer(1)]],
        device const float * w         [[buffer(2)]],
        device T           * y         [[buffer(3)]],
        device T           * new_state [[buffer(4)]],
        device T           * ckpt      [[buffer(5)]],
        constant gdn_conv_args & args  [[buffer(6)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const uint r = gid.y;
    const uint S = args.s;
    const uint C = args.c;
    if (c >= C) {
        return;
    }
    if (r < S) {
        // Same order as the op chain's shifted multiply-accumulates.
        float acc = conv_hidden(x, state, args.xs, c, r + 1) * w[c * GDN_CONV_K];
        for (uint j = 1; j < GDN_CONV_K; ++j) {
            acc += conv_hidden(x, state, args.xs, c, r + 1 + j) * w[c * GDN_CONV_K + j];
        }
        const float rounded = float(T(acc));
        y[(ulong)r * C + c] = T(silu_f32(rounded));
    } else if (r < S + GDN_CONV_K) {
        const uint j = r - S;
        new_state[c * GDN_CONV_K + j] = T(conv_hidden(x, state, args.xs, c, S + j));
    } else if (args.ckpt_at > 0 && r < S + 2 * GDN_CONV_K) {
        const uint j = r - S - GDN_CONV_K;
        ckpt[c * GDN_CONV_K + j] = T(conv_hidden(x, state, args.xs, c, args.ckpt_at + j));
    }
}

// L2 norm of one 128-wide row held 4-per-lane across a SIMD-group: squares
// rounded to T (the chain squares in T), f32 sum, eps added to the sum.
template <typename T>
inline float l2_inv(thread const float (&v)[4]) {
    float sum = 0.0f;
    for (uint j = 0; j < 4; ++j) {
        sum += float(T(v[j] * v[j]));
    }
    sum = simd_sum(sum);
    return 1.0f / sqrt(sum + 1e-6f);
}

// One SIMD-group per (token, value head); threadgroups hold `heads` groups.
//   conv: [S, C] T, C = 2*key_dim + value_dim, layout Q | K | V
//   a, b: [S, H] T (token strides as_, bs_)
//   q3, k3, v3: [H, len, 128] f32; g2, beta2: [H, len] f32, for tokens
//   t0 .. t0 + len.
// Key head of value head hv: order 0 (interleaved) hv / vpg, 1 (chunked)
// hv % num_k.
struct gdn_prep_args {
    uint t0;
    uint len;
    uint h;
    uint num_k;
    uint vpg;
    uint order;
    uint key_dim;
    uint c;
    uint as_;
    uint bs_;
    float scale;
};

template <typename T>
kernel void gdn_prep(
        device const T     * conv          [[buffer(0)]],
        device const T     * a             [[buffer(1)]],
        device const T     * b             [[buffer(2)]],
        device const float * neg_exp_a_log [[buffer(3)]],
        device const float * dt_bias       [[buffer(4)]],
        device float       * q3            [[buffer(5)]],
        device float       * k3            [[buffer(6)]],
        device float       * v3            [[buffer(7)]],
        device float       * g2            [[buffer(8)]],
        device float       * beta2         [[buffer(9)]],
        constant gdn_prep_args & args      [[buffer(10)]],
        uint2 tg   [[threadgroup_position_in_grid]],
        uint  sg   [[simdgroup_index_in_threadgroup]],
        uint  nsg  [[simdgroups_per_threadgroup]],
        uint  lane [[thread_index_in_simdgroup]]) {
    const uint tt = tg.x;
    const uint hv = tg.y * nsg + sg;
    if (hv >= args.h) {
        return;
    }
    const uint t = args.t0 + tt;
    const uint kh = args.order == 0 ? hv / args.vpg : hv % args.num_k;

    device const T * qrow = conv + (ulong)t * args.c + kh * GDN_HEAD_DIM;
    device const T * krow = conv + (ulong)t * args.c + args.key_dim + kh * GDN_HEAD_DIM;
    device const T * vrow = conv + (ulong)t * args.c + 2 * args.key_dim + hv * GDN_HEAD_DIM;
    const ulong out_row = ((ulong)hv * args.len + tt) * GDN_HEAD_DIM;

    float qv[4], kv[4];
    for (uint j = 0; j < 4; ++j) {
        qv[j] = float(qrow[lane + 32 * j]);
        kv[j] = float(krow[lane + 32 * j]);
    }
    const float qi = l2_inv<T>(qv);
    const float ki = l2_inv<T>(kv);
    for (uint j = 0; j < 4; ++j) {
        const uint d = lane + 32 * j;
        q3[out_row + d] = (qv[j] * qi) * args.scale;
        k3[out_row + d] = kv[j] * ki;
        v3[out_row + d] = float(vrow[d]);
    }
    if (lane == 0) {
        const float bv = float(b[(ulong)t * args.bs_ + hv]);
        const float av = float(a[(ulong)t * args.as_ + hv]);
        const float sp_in = av + dt_bias[hv];
        const float sp = sp_in > 20.0f ? sp_in : log(1.0f + exp(sp_in));
        g2[(ulong)hv * args.len + tt] = neg_exp_a_log[hv] * sp;
        beta2[(ulong)hv * args.len + tt] = float(T(1.0f / (1.0f + exp(-bv))));
    }
}

// Recurrence output y [H, len, 128] f32 (tokens t0 .. t0 + len) -> rows of
// out [S, H*128] T:
//   x = T(y); n = T(rms_norm_f32(x)); n = T(n * w); out = T(n * silu(z))
// One SIMD-group per (token, value head), as in gdn_prep.
struct gdn_post_args {
    uint t0;
    uint len;
    uint h;
    uint zs;
    float eps;
};

template <typename T>
kernel void gdn_post_norm(
        device const float * y   [[buffer(0)]],
        device const T     * z   [[buffer(1)]],
        device const float * w   [[buffer(2)]],
        device T           * out [[buffer(3)]],
        constant gdn_post_args & args [[buffer(4)]],
        uint2 tg   [[threadgroup_position_in_grid]],
        uint  sg   [[simdgroup_index_in_threadgroup]],
        uint  nsg  [[simdgroups_per_threadgroup]],
        uint  lane [[thread_index_in_simdgroup]]) {
    const uint tt = tg.x;
    const uint hv = tg.y * nsg + sg;
    if (hv >= args.h) {
        return;
    }
    const uint t = args.t0 + tt;
    device const float * yrow = y + ((ulong)hv * args.len + tt) * GDN_HEAD_DIM;
    device const T * zrow = z + (ulong)t * args.zs + hv * GDN_HEAD_DIM;
    device T * orow = out + (ulong)t * args.h * GDN_HEAD_DIM + hv * GDN_HEAD_DIM;

    float xv[4];
    float sum = 0.0f;
    for (uint j = 0; j < 4; ++j) {
        xv[j] = float(T(yrow[lane + 32 * j]));
        sum += xv[j] * xv[j];
    }
    sum = simd_sum(sum);
    const float inv = 1.0f / sqrt(sum / float(GDN_HEAD_DIM) + args.eps);
    for (uint j = 0; j < 4; ++j) {
        const uint d = lane + 32 * j;
        const float normalized = float(T(xv[j] * inv));
        const float weighted = float(T(normalized * w[d]));
        orow[d] = T(weighted * silu_f32(float(zrow[d])));
    }
}

#define GDN_STAGES(T, TNAME)                                                                    \
    template [[host_name("gdn_conv_silu_" TNAME)]] kernel void gdn_conv_silu<T>(               \
        device const T *, device const T *, device const float *, device T *, device T *,      \
        device T *, constant gdn_conv_args &, uint2);                                          \
    template [[host_name("gdn_prep_" TNAME)]] kernel void gdn_prep<T>(                         \
        device const T *, device const T *, device const T *, device const float *,            \
        device const float *, device float *, device float *, device float *, device float *,  \
        device float *, constant gdn_prep_args &, uint2, uint, uint, uint);                    \
    template [[host_name("gdn_post_norm_" TNAME)]] kernel void gdn_post_norm<T>(               \
        device const float *, device const T *, device const float *, device T *,              \
        constant gdn_post_args &, uint2, uint, uint, uint);

GDN_STAGES(half, "f16")
GDN_STAGES(float, "f32")
#if defined(__HAVE_BFLOAT__)
GDN_STAGES(bfloat, "bf16")
#endif
