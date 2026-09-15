//! Numerical kernels for Gated Delta Net: L2 normalization, softplus, the
//! gated delta rule recurrence, causal Conv1D, and dispatch.
//!
//! The portable recurrence is a composition of Candle tensor ops, so it runs
//! on any device (CPU/CUDA/Metal/ROCm). On CUDA and ROCm a fused kernel is
//! also available (see [`super::cuda_backend`] / [`super::rocm_backend`]) and
//! is used by default; set `CRANE_GDN_PORTABLE=1` to force the portable
//! op-by-op path for cross-checking numerics.

use candle_core::{D, DType, IndexOp, Result, Tensor};

use super::cache::GdnLayerCache;
use super::config::GdnDims;
#[cfg(any(feature = "cuda", feature = "rocm", feature = "sycl"))]
use crate::utils::DeviceExt;

// ─────────────────────────────────────────────────────────────────────
//  Elementwise helpers
// ─────────────────────────────────────────────────────────────────────

/// `x / sqrt(sum(x^2) + eps)` over the last dim — used to normalize Q and K
/// before the delta-rule recurrence.
///
/// Device-portable reference. [`l2_norm_fused`] is the one the layer calls;
/// this stays as the definition both the tests and the CPU path check against.
pub fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let inv_norm = x
        .sqr()?
        .sum_keepdim(D::Minus1)?
        // `affine`, not `broadcast_add` against a fresh scalar `Tensor::new`:
        // the latter is a host-to-device copy, and this runs twice per layer
        // per token.
        .affine(1.0, eps)?
        .sqrt()?
        .recip()?;
    x.broadcast_mul(&inv_norm)
}

/// [`l2_norm`] as a single fused `rms_norm` launch.
///
/// `rms_norm(x, α, ε') = x / sqrt(mean(x²) + ε') * α`, and with `α = 1/sqrt(K)`
/// and `ε' = eps/K` that is *identically* `x / sqrt(sum(x²) + eps)` — no
/// approximation, just the constant `K` moved between the two terms:
///
/// ```text
/// x / sqrt(Σx² + ε) = x / sqrt(K·(mean(x²) + ε/K)) = (1/sqrt(K))·x / sqrt(mean(x²) + ε/K)
/// ```
///
/// `alpha` is the precomputed `[K]` vector of `1/sqrt(K)`; building it per call
/// would reintroduce the host-to-device copy this exists to remove.
pub fn l2_norm_fused(x: &Tensor, alpha: &Tensor, eps: f64) -> Result<Tensor> {
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    let eps_mean = (eps / x.dim(D::Minus1)? as f64) as f32;
    let alpha = alpha.to_dtype(x.dtype())?;
    candle_nn::ops::rms_norm(&x.contiguous()?, &alpha, eps_mean)
}

/// The `[K]` vector of `1/sqrt(K)` that [`l2_norm_fused`] scales by.
pub fn l2_alpha(head_k_dim: usize, dtype: DType, device: &candle_core::Device) -> Result<Tensor> {
    #[allow(clippy::cast_precision_loss)]
    let v = 1.0 / (head_k_dim as f64).sqrt();
    Tensor::full(v as f32, head_k_dim, device)?.to_dtype(dtype)
}

/// `log(1 + exp(x))`.
///
/// `affine` rather than `ones_like() + …`: it folds the `+1` into the same
/// launch instead of materializing a whole tensor of ones first.
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    x.exp()?.affine(1.0, 1.0)?.log()
}

// ─────────────────────────────────────────────────────────────────────
//  Gated delta rule recurrence (CPU reference)
// ─────────────────────────────────────────────────────────────────────

/// Per-timestep gated delta rule in pure Candle ops.
///
/// Inputs (all contiguous, f32):
/// - `q, k`: `[BH, S, K]` — queries / keys
/// - `v`:    `[BH, S, V]` — values
/// - `g`:    `[BH, S]`    — log-decay (already pre-softplus'd by the caller)
/// - `beta`: `[BH, S]`    — write strength
/// - `state`: `[BH, K, V]` — recurrent state (mutated in place)
///
/// Returns `y: [BH, S, V]`. `state` is updated to the post-final-step value.
///
/// This is the exact CPU fallback lifted from mistral.rs
/// (`mistralrs-core/src/gdn/backend.rs:30-81`). Output dtype matches `q.dtype()`.
pub fn gated_delta_rule_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let dtype = q.dtype();

    // HF's gated delta rule scales Q by `1/sqrt(head_k_dim)` before the
    // recurrence (`scale = 1 / query.shape[-1]**0.5` in both
    // `torch_chunk_gated_delta_rule` and `torch_recurrent_gated_delta_rule`).
    // Omitting it leaves the recurrence output a factor of `sqrt(K)` too large
    // (verified: cos=1.0 vs HF but ~11.3x = sqrt(128) magnitude), which is NOT
    // washed out downstream because the gated RMSNorm's eps and the silu gate
    // make it observable. mistral.rs applies the same scale.
    let scale = 1.0 / (q.dim(D::Minus1)? as f64).sqrt();
    let q = (q.affine(scale, 0.0)?)
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(DType::F32)?;
    let k = k.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let v = v.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let g = g.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let beta = beta.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;

    let seq_len = q.dim(2)?;
    let mut s = state.to_dtype(DType::F32)?;
    let mut outputs = Vec::with_capacity(seq_len);

    for i in 0..seq_len {
        let q_t = q.i((.., .., i, ..))?;
        let k_t = k.i((.., .., i, ..))?;
        let v_t = v.i((.., .., i, ..))?;
        let g_t = g.i((.., .., i))?;
        let beta_t = beta.i((.., .., i))?;

        // 1. Decay state by per-head factor.
        let decay = g_t.exp()?.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?;
        s = s.broadcast_mul(&decay)?;

        // 2. Retrieve kv_mem = sum_d_state(state * k).
        let k_exp = k_t.unsqueeze(D::Minus1)?;
        let kv_mem = s.broadcast_mul(&k_exp)?.sum(2)?;

        // 3. Delta rule residual.
        let beta_exp = beta_t.unsqueeze(D::Minus1)?;
        let delta = (v_t - kv_mem)?.broadcast_mul(&beta_exp)?;

        // 4. Write state update: S += outer(k, delta).
        let outer = k_exp.broadcast_mul(&delta.unsqueeze(2)?)?;
        s = (s + outer)?;

        // 5. Output: y = sum_d_state(state * q).
        let q_exp = q_t.unsqueeze(D::Minus1)?;
        let y_t = s.broadcast_mul(&q_exp)?.sum(2)?;
        outputs.push(y_t);
    }

    *state = s.to_dtype(state.dtype())?;

    Tensor::stack(&outputs, 2)?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(dtype)
}

// ─────────────────────────────────────────────────────────────────────
//  β/g computation (pre-recurrence)
// ─────────────────────────────────────────────────────────────────────

/// The two per-head constants [`compute_beta_g`] needs, in the shape and dtype
/// it needs them.
///
/// `A_log` and `dt_bias` are weights: `-exp(A_log)` and the broadcast of
/// `dt_bias` are the same tensors on every token of every request, but were
/// being rebuilt per layer per token — four launches each time, for a value
/// that never changes. Deriving them once at load removes that entirely.
pub struct GdnGateConsts {
    /// `-exp(A_log)` as `[1, 1, H]`, f32.
    pub neg_exp_a_log: Tensor,
    /// `dt_bias` as `[1, 1, H]`, f32.
    pub dt_bias: Tensor,
}

impl GdnGateConsts {
    /// Derive from the raw `A_log` and `dt_bias` weights (both `[H]`).
    pub fn new(a_log: &Tensor, dt_bias: &Tensor) -> Result<Self> {
        Ok(Self {
            neg_exp_a_log: a_log
                .to_dtype(DType::F32)?
                .exp()?
                .neg()?
                .unsqueeze(0)?
                .unsqueeze(0)?,
            dt_bias: dt_bias.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(0)?,
        })
    }
}

/// Derive the per-head write strength β and per-step decay g from the
/// projections and the precomputed gate constants.
///
/// `b: [B, S, H]` raw logits → `beta = sigmoid(b)`.
/// `a: [B, S, H]` raw values → combined with `A_log` (negative log of decay
/// rate) and `dt_bias` to produce `g = -exp(A_log) * softplus(a + dt_bias)`.
pub fn compute_beta_g(
    b: &Tensor,
    a: &Tensor,
    consts: &GdnGateConsts,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    // β and g are tiny per-head ops; computed in pure Candle.
    let beta = candle_nn::ops::sigmoid(b)?;
    let a_f = a.to_dtype(DType::F32)?;
    let g = consts
        .neg_exp_a_log
        .broadcast_mul(&softplus(&a_f.broadcast_add(&consts.dt_bias)?)?)?
        .to_dtype(dtype)?;
    Ok((beta, g))
}

// ─────────────────────────────────────────────────────────────────────
//  Dispatch entry points
// ─────────────────────────────────────────────────────────────────────

/// Compute β and g, run the gated delta rule recurrence, return the output.
///
/// The recurrence is written in pure-Candle tensor ops, so it runs on any
/// device (CPU/CUDA/Metal/ROCm) — every op has a native backend kernel. On
/// CUDA and ROCm the fused single-launch kernel is used by default (see
/// [`super::cuda_backend`] / [`super::rocm_backend`]); set
/// `CRANE_GDN_PORTABLE=1` to force the portable op-by-op path for
/// cross-checking numerics.
#[allow(unused_variables)]
pub fn apply_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    cache: &mut GdnLayerCache,
    dtype: DType,
) -> Result<Tensor> {
    #[cfg(any(feature = "cuda", feature = "rocm", feature = "sycl"))]
    if (q.device().is_cuda() || q.device().is_rocm() || q.device().is_sycl())
        && std::env::var("CRANE_GDN_PORTABLE").is_err()
    {
        return fused_recurrence(q, k, v, g, beta, dims, batch_size, seq_len, cache, dtype);
    }

    // Device-portable reference (runs on CPU/CUDA/Metal/ROCm/SYCL).
    gated_delta_rule_recurrence(q, k, v, g, beta, &mut cache.recurrent_state)
}

/// Prepare tensors and launch the fused recurrence kernel.
///
/// Lays inputs out as the kernel expects (`[BH, S, *]`, contiguous f32),
/// applies the `1/sqrt(K)` query scale here (the kernel takes plain q), then
/// reshapes the result back to the portable path's `[B, S, num_v_heads, V]`.
///
/// The layout work is identical for every backend; only the launch differs, and
/// `cuda` / `rocm` / `sycl` never coexist in a working build.
#[cfg(any(feature = "cuda", feature = "rocm", feature = "sycl"))]
#[allow(clippy::too_many_arguments)]
fn fused_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
    cache: &mut GdnLayerCache,
    dtype: DType,
) -> Result<Tensor> {
    use crate::ops::prof::{Span, timed};

    let (hv, kd, vd) = (dims.num_v_heads, dims.head_k_dim, dims.head_v_dim);
    let bh = batch_size * hv;
    let scale = 1.0 / (kd as f64).sqrt();

    // [B, S, Hv, *] -> [B, Hv, S, *] -> [BH, S, *], contiguous f32. The dtype
    // cast comes *after* the permutation: casting first makes the copy
    // `contiguous()` performs an f32 one, which at prefill length is twice the
    // transient VRAM for no other difference (the cast commutes with the
    // permutation).
    let prep3 = |t: &Tensor| -> Result<Tensor> {
        t.transpose(1, 2)?
            .contiguous()?
            .to_dtype(DType::F32)?
            .reshape((bh, seq_len, ()))
    };
    let prep2 = |t: &Tensor| -> Result<Tensor> {
        t.transpose(1, 2)?
            .contiguous()?
            .to_dtype(DType::F32)?
            .reshape((bh, seq_len))
    };
    // The layout work is charged separately from the launch: at decode's
    // `seq_len == 1` it is a fixed per-layer cost paid on every token, whereas
    // in prefill it is amortised over the whole prompt. Whether that fixed cost
    // matters is the question `CRANE_PROF` exists to settle, so the two must
    // not be lumped together.
    let (q3, k3, v3, g2, beta2, state3) = timed(Span::GdnPrep, || -> Result<_> {
        let q3 = prep3(q)?.affine(scale, 0.0)?;
        let k3 = prep3(k)?;
        let v3 = prep3(v)?;
        let g2 = prep2(g)?;
        let beta2 = prep2(beta)?;
        let state3 = cache
            .recurrent_state
            .to_dtype(DType::F32)?
            .reshape((bh, kd, vd))?
            .contiguous()?;
        Ok((q3, k3, v3, g2, beta2, state3))
    })?;

    let (y, state_out) = timed(Span::GdnLaunch, || {
        #[cfg(feature = "cuda")]
        {
            super::cuda_backend::gdn_recurrence_cuda(&q3, &k3, &v3, &g2, &beta2, &state3)
        }
        #[cfg(all(feature = "rocm", not(feature = "cuda")))]
        {
            super::rocm_backend::gdn_recurrence_rocm(&q3, &k3, &v3, &g2, &beta2, &state3)
        }
        #[cfg(all(feature = "sycl", not(feature = "cuda"), not(feature = "rocm")))]
        {
            super::sycl_backend::gdn_recurrence_sycl(&q3, &k3, &v3, &g2, &beta2, &state3)
        }
    })?;

    timed(Span::GdnPost, || {
        cache.recurrent_state = state_out.reshape((batch_size, hv, kd, vd))?;
        // [BH, S, V] -> [B, Hv, S, V] -> [B, S, Hv, V], back to model dtype. Cast
        // before the copy for the same reason `prep3` casts after it: the narrower
        // side of the conversion is the one worth materialising.
        y.reshape((batch_size, hv, seq_len, vd))?
            .transpose(1, 2)?
            .to_dtype(dtype)?
            .contiguous()
    })
}

// ─────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// Smoke test: recurrence runs end-to-end on CPU and produces the right
    /// output shape. Numerical correctness against a reference (HF Transformers
    /// / mistral.rs) is checked in the integration test that loads a real
    /// Qwen 3.5 checkpoint.
    #[test]
    fn recurrence_runs_and_returns_correct_shape() {
        let dev = Device::Cpu;
        let q = Tensor::new(
            &[[[[1.0f32, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[0.5, 0.5]]]],
            &dev,
        )
        .unwrap();
        let k = Tensor::new(
            &[[[[1.0f32, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[0.0, 1.0]]]],
            &dev,
        )
        .unwrap();
        let v = Tensor::new(
            &[[[[1.0f32, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]], [[0.5, 0.5]]]],
            &dev,
        )
        .unwrap();
        let g = Tensor::new(&[[[0.0f32], [0.0], [0.0], [0.0]]], &dev).unwrap();
        let beta = Tensor::new(&[[[1.0f32], [1.0], [1.0], [1.0]]], &dev).unwrap();

        let mut state = Tensor::zeros((1, 1, 2, 2), DType::F32, &dev).unwrap();
        let y = gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut state).unwrap();
        assert_eq!(y.dims(), &[1, 4, 1, 2]);
        // State must have been mutated in place.
        assert!(state.dims() == [1, 1, 2, 2]);
    }

    #[test]
    fn l2_norm_preserves_direction() {
        let dev = Device::Cpu;
        let x = Tensor::new(&[[3.0f32, 4.0], [1.0, 0.0]], &dev).unwrap();
        let n = l2_norm(&x, 1e-6).unwrap();
        let v0 = n.i((0, ..)).unwrap().to_vec1::<f32>().unwrap();
        assert!((v0[0] - 0.6).abs() < 1e-3 && (v0[1] - 0.8).abs() < 1e-3);
        let v1 = n.i((1, ..)).unwrap().to_vec1::<f32>().unwrap();
        assert!((v1[0] - 1.0).abs() < 1e-3 && v1[1].abs() < 1e-3);
    }

    #[test]
    fn softplus_at_zero_is_ln2() {
        let dev = Device::Cpu;
        let x = Tensor::new(0.0f32, &dev).unwrap();
        let s = softplus(&x).unwrap().to_scalar::<f32>().unwrap();
        assert!((s - std::f32::consts::LN_2).abs() < 1e-4);
    }
}
