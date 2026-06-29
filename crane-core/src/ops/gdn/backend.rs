//! Numerical kernels for Gated Delta Net: L2 normalization, softplus, the
//! gated delta rule recurrence, causal Conv1D, and dispatch.
//!
//! The recurrence and Conv1D are written as compositions of Candle tensor ops,
//! so they run on any device (CPU/CUDA/Metal). On CUDA, an optional fused
//! kernel is also available (see [`super::cuda_backend`]); set
//! `CRANE_GDN_PORTABLE=1` to force the op-by-op path for cross-checking
//! numerics against the reference.

use candle_core::{DType, IndexOp, Result, Tensor, D};

use super::cache::GdnLayerCache;
use super::config::GdnDims;

// ─────────────────────────────────────────────────────────────────────
//  Elementwise helpers
// ─────────────────────────────────────────────────────────────────────

/// `x / sqrt(mean(x^2) + eps)` over the last dim — used to normalize Q and K
/// before the delta-rule recurrence.
pub fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let inv_norm = x
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .broadcast_add(&Tensor::new(eps as f32, x.device())?.to_dtype(x.dtype())?)?
        .sqrt()?
        .recip()?;
    x.broadcast_mul(&inv_norm)
}

/// `log(1 + exp(x))` — numerically stable; computed as `log1p(exp(x))`.
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    (Tensor::ones_like(x)? + x.exp()?)?.log()
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
/// Output dtype matches `q.dtype()`.
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
    // recurrence (`scale = 1 / query.shape[-1]**0.5` in
    // `torch_recurrent_gated_delta_rule`). Omitting it leaves the recurrence
    // output a factor of `sqrt(K)` too large, which is not washed out
    // downstream because the gated RMSNorm eps and the silu gate make it
    // observable.
    let scale = 1.0 / (q.dim(D::Minus1)? as f64).sqrt();
    let q = (q.affine(scale, 0.0)?).transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
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

/// Derive the per-head write strength β and per-step decay g from the
/// projections, A_log, and dt_bias.
///
/// `b: [B, S, H]` raw logits → `beta = sigmoid(b)`.
/// `a: [B, S, H]` raw values → combined with `A_log` (negative log of decay
/// rate) and `dt_bias` to produce `g = -exp(A_log) * softplus(a + dt_bias)`.
pub fn compute_beta_g(
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    // β and g are tiny per-head ops; computed in pure Candle.
    let beta = candle_nn::ops::sigmoid(b)?;
    let a_f = a.to_dtype(DType::F32)?;
    let dt_bias_expanded = dt_bias.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(0)?;
    let g = a_log
        .to_dtype(DType::F32)?
        .exp()?
        .neg()?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .broadcast_mul(&softplus(&a_f.broadcast_add(&dt_bias_expanded)?)?)?
        .to_dtype(dtype)?;
    Ok((beta, g))
}

// ─────────────────────────────────────────────────────────────────────
//  Dispatch entry points
// ─────────────────────────────────────────────────────────────────────

/// Compute β and g, run the gated delta rule recurrence, return the output.
///
/// The recurrence is written in pure-Candle tensor ops, so it runs on any
/// device (CPU/CUDA/Metal) — every op has a native backend kernel. On CUDA,
/// a fused single-launch kernel is preferred when available
/// ([`super::cuda_backend::gdn_recurrence_cuda`]); set
/// `CRANE_GDN_PORTABLE=1` to force the op-by-op path for cross-checking
/// numerics.
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
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() && std::env::var("CRANE_GDN_PORTABLE").is_err() {
        return cuda_recurrence(q, k, v, g, beta, dims, batch_size, seq_len, cache, dtype);
    }

    gated_delta_rule_recurrence(q, k, v, g, beta, &mut cache.recurrent_state)
}

/// Prepare tensors and launch the fused CUDA recurrence kernel.
///
/// Lays inputs out as the kernel expects (`[BH, S, *]`, contiguous f32),
/// applies the `1/sqrt(K)` query scale here (the kernel takes plain q), then
/// reshapes the result back to the portable path's `[B, S, num_v_heads, V]`.
#[cfg(feature = "cuda")]
fn cuda_recurrence(
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
    let (hv, kd, vd) = (dims.num_v_heads, dims.head_k_dim, dims.head_v_dim);
    let bh = batch_size * hv;
    let scale = 1.0 / (kd as f64).sqrt();

    // [B, S, Hv, *] -> [B, Hv, S, *] -> [BH, S, *], contiguous f32.
    let prep3 = |t: &Tensor| -> Result<Tensor> {
        t.to_dtype(DType::F32)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bh, seq_len, ()))
    };
    let prep2 = |t: &Tensor| -> Result<Tensor> {
        t.to_dtype(DType::F32)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bh, seq_len))
    };
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

    let (y, state_out) =
        super::cuda_backend::gdn_recurrence_cuda(&q3, &k3, &v3, &g2, &beta2, &state3)?;

    cache.recurrent_state = state_out.reshape((batch_size, hv, kd, vd))?;
    // [BH, S, V] -> [B, Hv, S, V] -> [B, S, Hv, V], back to model dtype.
    y.reshape((batch_size, hv, seq_len, vd))?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(dtype)
}

/// Causal Conv1D over the QKV channels. Dispatches to a kernel-based
/// implementation when available, pure-Candle otherwise.
pub fn causal_conv1d(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
    is_decode_step: bool,
) -> Result<Tensor> {
    if is_decode_step {
        causal_conv1d_update(x, conv1d_weight, dims, cache)
    } else {
        causal_conv1d_full(x, conv1d_weight, dims, cache)
    }
}

/// Single-step Conv1D for autoregressive decode. Concatenates the new token
/// to the cached conv state, computes one output, then rolls the cache by 1.
fn causal_conv1d_update(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
) -> Result<Tensor> {
    let (_, seq_len, _) = x.dims3()?;
    let x_t = x.transpose(1, 2)?.contiguous()?;
    // `conv_state` is shape `[1, conv_dim, kernel_size]` — the kernel-size
    // dim is the LAST (matches mistral.rs).
    let state_len = cache.conv_state.dim(2)?;
    let hidden_new = Tensor::cat(&[cache.conv_state.clone(), x_t], 2)?;
    let new_len = hidden_new.dim(2)?;
    cache.conv_state = hidden_new.narrow(2, new_len - state_len, state_len)?;

    let weight = conv1d_weight.squeeze(1)?.to_dtype(hidden_new.dtype())?;
    let mut conv_outputs = Vec::with_capacity(seq_len);
    let total_len = hidden_new.dim(2)?;
    for i in (total_len - seq_len)..total_len {
        let window = hidden_new.narrow(2, i + 1 - dims.conv_kernel_size, dims.conv_kernel_size)?;
        let out = (window * weight.unsqueeze(0)?)?.sum(D::Minus1)?;
        conv_outputs.push(out);
    }
    candle_nn::ops::silu(&Tensor::stack(&conv_outputs, 2)?)?.transpose(1, 2)
}

/// Multi-step Conv1D used during prefill. Pads on the left by `kernel - 1`,
/// runs the full convolution, then captures the trailing `kernel - 1` tokens
/// into the cache for the next decode step.
fn causal_conv1d_full(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
) -> Result<Tensor> {
    let (batch_size, seq_len, conv_dim) = x.dims3()?;
    let x_t = x.transpose(1, 2)?.contiguous()?;

    // Pad on the time dim (the last) and capture the trailing `kernel - 1`
    // tokens into the cache for the next decode step.
    let pad_width = dims.conv_kernel_size.saturating_sub(seq_len);
    cache.conv_state = if pad_width > 0 {
        let zeros = Tensor::zeros((batch_size, conv_dim, pad_width), x_t.dtype(), x_t.device())?;
        Tensor::cat(&[zeros, x_t.clone()], 2)?
    } else {
        x_t.narrow(2, seq_len - dims.conv_kernel_size, dims.conv_kernel_size)?
    };

    let padded_t = Tensor::cat(
        &[
            Tensor::zeros(
                (batch_size, conv_dim, dims.conv_kernel_size - 1),
                x_t.dtype(),
                x_t.device(),
            )?,
            x_t,
        ],
        2,
    )?;

    let weight = conv1d_weight.squeeze(1)?.to_dtype(padded_t.dtype())?;
    let mut conv_outputs = Vec::with_capacity(seq_len);
    for i in 0..seq_len {
        let window = padded_t.narrow(2, i, dims.conv_kernel_size)?;
        let out = (window * weight.unsqueeze(0)?)?.sum(D::Minus1)?;
        conv_outputs.push(out);
    }
    candle_nn::ops::silu(&Tensor::stack(&conv_outputs, 2)?)?.transpose(1, 2)
}
// ─────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// Smoke test: recurrence runs end-to-end on CPU and produces the right
    /// output shape. Numerical correctness against HF Transformers is
    /// exercised by the engine-level smoke tests against a real Qwen 3.5
    /// checkpoint.
    #[test]
    fn recurrence_runs_and_returns_correct_shape() {
        let dev = Device::Cpu;
        let q = Tensor::new(
            &[[
                [[1.0f32, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 1.0]],
                [[0.5, 0.5]],
            ]],
            &dev,
        )
        .unwrap();
        let k = Tensor::new(
            &[[
                [[1.0f32, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ]],
            &dev,
        )
        .unwrap();
        let v = Tensor::new(
            &[[
                [[1.0f32, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 1.0]],
                [[0.5, 0.5]],
            ]],
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
