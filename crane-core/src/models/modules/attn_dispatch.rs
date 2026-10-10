// SPDX-License-Identifier: MIT
//! Backend-agnostic matmul SDPA. Pure `candle_core`/`candle_nn` ops, no
//! feature gates. Compiles and runs on every backend (CPU, CUDA, `ROCm`,
//! Metal, SYCL).
//!
//! Public entry points (models call these, not the `pub(super)` helpers):
//! - [`causal`], [`full`], [`windowed`] try `gpu_flash_attn`'s fused kernel
//!   first and fall back internally on CPU or unsupported `head_dim`.
//! - [`decode`] handles the `seq_len == 1` case with a GQA-grouped reshape
//!   that avoids expanding K/V.
//! - [`prefill_masked`] applies an arbitrary additive mask (e.g. continuous-
//!   batching padding) with no fused-kernel attempt.
//! - [`attention_scale`] computes `1 / sqrt(head_dim)`.
//!
//! [`causal_with_mask`], [`causal_without_mask`], [`full_matmul`], and
//! [`windowed_matmul`] are `pub(super)` fallback helpers for `gpu_flash_attn`.
//!
//! Softmax runs in native dtype, not F32, to avoid regressing GPU throughput.

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::attention::AttnMask;
use candle_nn::ops::softmax_last_dim;

use super::flash_attn::dispatch_flash_attn;
use crate::models::utils::{CausalMask, build_additive_causal_mask, repeat_kv};

/// Attention scale factor `1 / sqrt(head_dim)`, shared so callers don't each
/// repeat the `f64`-intermediate cast and its `#[allow]`s.
#[must_use]
pub fn attention_scale(head_dim: usize) -> f32 {
    // head_dim is a small positive integer (attention head sizes, typically
    // 64..256), exactly representable in f32.
    #[allow(clippy::cast_precision_loss)]
    {
        1.0 / (head_dim as f32).sqrt()
    }
}

/// Builds a `[1, 1, q_len, kv_len]` additive mask where position `(i, j)` is
/// `0` when `i + kv_offset - window_left <= j <= i + kv_offset + window_right`
/// and `f32::NEG_INFINITY` otherwise. Shares `kv_offset`'s meaning, and the
/// per-call allocation caveat, with [`build_additive_causal_mask`].
fn build_windowed_mask(
    q_len: usize,
    kv_len: usize,
    kv_offset: usize,
    window_left: usize,
    window_right: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut data = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        for j in 0..kv_len {
            // q_len/kv_len/window bounds are sequence lengths (far below
            // i64::MAX), so these never wrap; i64 is needed since `rel` can
            // be negative.
            #[allow(clippy::cast_possible_wrap)]
            let (center, j_i64, window_left_i64, window_right_i64) = (
                (i + kv_offset) as i64,
                j as i64,
                window_left as i64,
                window_right as i64,
            );
            let rel = j_i64 - center;
            let valid = rel >= -window_left_i64 && rel <= window_right_i64;
            if !valid {
                data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, q_len, kv_len), device)
}

/// Shared prefill SDPA core for [`causal_without_mask`], [`causal_with_mask`],
/// [`full`], and [`windowed`]: `repeat_kv` GQA expansion,
/// `Q @ K^T * scale [+ mask]`, native-dtype softmax, `@ V`. `q`/`k`/`v` are
/// BHSD; the mask (if any) is additive, built in F32 by this module's
/// mask-building functions, and broadcastable to `[B, H_q, q_len, kv_len]`.
/// Returns BHSD.
///
/// The mask is cast to `attn_weights`' native dtype rather than upcasting
/// `attn_weights` to F32: `NEG_INFINITY`/`0.0` are exactly representable in
/// every float format, so this is lossless, and it avoids upcasting the
/// `[B, H_q, q_len, kv_len]` score tensor — `H_q` times larger than the mask
/// — which measurably regressed GPU prefill throughput (confirmed via
/// `crane-serve`) for the long-context case this tier exists to serve.
fn prefill_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let n_rep = q.dim(1)? / k.dim(1)?;
    let k = repeat_kv(k.clone(), n_rep)?.contiguous()?;
    let v = repeat_kv(v.clone(), n_rep)?.contiguous()?;
    // Q may be non-contiguous after the caller's transpose(1, 2) into BHSD.
    let q = q.contiguous()?;

    let attn_weights = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? * f64::from(scale))?;
    let attn_weights = match mask {
        Some(mask) => attn_weights.broadcast_add(&mask.to_dtype(attn_weights.dtype())?)?,
        None => attn_weights,
    };
    let attn_weights = softmax_last_dim(&attn_weights)?;

    attn_weights.matmul(&v)
}

/// Causal matmul SDPA, building its own mask internally on every call:
/// `j <= i + kv_offset`, where `kv_offset = k`'s sequence length minus
/// `q`'s. Mirrors
/// [`gpu_flash_attn::try_causal`](super::gpu_flash_attn::try_causal)'s mask
/// semantics and BHSD/GQA shape conventions, but always succeeds (no
/// `Option`) and never requires a specific `head_dim` or backend.
///
/// Named `_without_mask` (rather than a plain `causal`) deliberately, paired
/// with [`causal_with_mask`]: a per-layer caller reaching for the shorter
/// name by habit is exactly the mistake that regressed Qwen3's prefill
/// throughput once already (see this module's doc comment). Only reach for
/// this function when no shared mask exists for the call (e.g.
/// `gpu_flash_attn`'s fallback tier) — a decoder stack calling this once per
/// layer per forward pass should build the mask once itself and call
/// [`causal_with_mask`] instead.
///
/// # Errors
///
/// Returns a candle error if `q`/`k`/`v`'s shapes are incompatible (GQA
/// head-count divisibility, matching `head_dim`) or if any tensor op fails.
pub(super) fn causal_without_mask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let q_len = q.dim(2)?;
    let kv_len = k.dim(2)?;
    let kv_offset = kv_len.saturating_sub(q_len);
    let mask = build_additive_causal_mask(q_len, kv_len, kv_offset, DType::F32, q.device())?;
    prefill_sdpa(q, k, v, scale, Some(&mask))
}

/// Causal matmul SDPA using a caller-provided, already-built additive mask
/// (see [`build_additive_causal_mask`]) instead of building one internally. Exists
/// for callers that share one mask across multiple layers in a single
/// forward pass — every layer in a decoder stack sees the same
/// `q_len`/`kv_len`/`kv_offset`, so building the mask once per forward pass
/// and reusing it here is correct and avoids [`causal_without_mask`]'s
/// per-call `O(q_len * kv_len)` allocation (and, on GPU, a host->device
/// re-upload) repeated once per layer. Otherwise identical to
/// [`causal_without_mask`]: same math, same BHSD/GQA shape conventions.
///
/// # Errors
///
/// Returns a candle error if `q`/`k`/`v`'s shapes are incompatible (GQA
/// head-count divisibility, matching `head_dim`), if `mask` isn't
/// broadcastable to `[B, H_q, q_len, kv_len]`, or if any tensor op fails.
pub(super) fn causal_with_mask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: &CausalMask,
) -> Result<Tensor> {
    prefill_sdpa(q, k, v, scale, Some(mask.as_tensor()))
}

/// Causal prefill SDPA with the best available backend: tries the fused GPU
/// flash-attn kernel first ([`super::gpu_flash_attn::try_causal`]/
/// [`try_causal_with_mask`](super::gpu_flash_attn::try_causal_with_mask) —
/// tensor-core MMA, then scalar kernel), falling back to this module's
/// matmul SDPA on CPU or when the fused kernel can't serve the call
/// (unsupported `head_dim`, no `cuda`/`rocm` feature compiled).
///
/// `mask` is an optional pre-built additive causal mask (see
/// [`build_additive_causal_mask`]), and **must encode exactly the plain
/// shifted-diagonal causal pattern** (`j <= i + kv_offset`, nothing else) —
/// the same pattern [`build_additive_causal_mask`] itself builds. When
/// `Some`, both the fused-kernel path (threaded straight into the kernel via
/// [`try_causal_with_mask`](super::gpu_flash_attn::try_causal_with_mask),
/// instead of rebuilding an equivalent mask from `kv_offset` on every call —
/// this kernel family's mask build is `O(q_len * kv_len)`, so sharing one
/// built once per forward pass across every layer, as described below, is
/// the whole reason this path exists) and the matmul-SDPA fallback (via
/// [`causal_with_mask`]) read `mask` directly. Despite the fused path now
/// genuinely reading `mask`'s contents, it still must be exactly the plain
/// causal pattern: `try_causal`/`try_causal_with_mask`'s analytic `KV_max`
/// bound (the per-tile KV-loop truncation that makes the fused kernel fast)
/// is derived purely from `kv_offset`, assuming this exact pattern,
/// independent of what `mask` itself contains — a mask that folds in
/// anything beyond plain causality (e.g. continuous-batching padding, a
/// sliding window) would still apply correctly on the matmul-SDPA fallback
/// (which has no `KV_max` concept) but could have its extra masking
/// silently defeated by a `KV_max` bound computed as if it weren't there, a
/// backend-dependent correctness divergence only visible on `cuda`/`rocm`
/// hardware. Use [`prefill_masked`] for an arbitrary mask instead. Per-layer
/// callers in a decoder stack (every layer shares the same
/// `q_len`/`kv_len`/`kv_offset`) should build the mask once per forward pass
/// and pass `Some` here — see [`causal_with_mask`]'s doc for why a per-layer
/// rebuild regressed throughput. Pass `None` only for a caller with no
/// shared mask to reuse.
///
/// # Errors
///
/// Returns a candle error under the same conditions as [`causal_without_mask`],
/// plus any error the fused-kernel dispatch itself returns on the
/// `cuda`/`rocm` success path (see
/// [`super::gpu_flash_attn::try_causal`]).
pub fn causal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&CausalMask>,
) -> Result<Tensor> {
    let gpu_result = match mask {
        Some(m) => super::gpu_flash_attn::try_causal_with_mask(q, k, v, scale, m),
        None => super::gpu_flash_attn::try_causal(q, k, v, scale),
    };
    match gpu_result {
        Some(result) => result,
        None => {
            if mask.is_none() && q.dim(0)? == 1 {
                return causal_cpu_flash(q, k, v, scale);
            }
            match mask {
                Some(m) => causal_with_mask(q, k, v, scale, m),
                None => causal_without_mask(q, k, v, scale),
            }
        },
    }
}

/// CPU single-sequence causal prefill: BHSD->BSHD, `dispatch_flash_attn`
/// with `AttnMask::Causal`, cast the kernel's F32 output back to `q`'s
/// dtype, return BHSD. Avoids materializing the full
/// `[B, H_q, q_len, kv_len]` score matrix that [`causal_without_mask`]'s
/// matmul path would, mirroring [`decode_cpu_flash`]'s CPU fast path for
/// the prefill case.
fn causal_cpu_flash(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let q_bshd = q.transpose(1, 2)?;
    let k_bshd = k.transpose(1, 2)?;
    let v_bshd = v.transpose(1, 2)?;
    let kv_offset = k.dim(2)?.saturating_sub(q.dim(2)?);

    let out = dispatch_flash_attn(
        &q_bshd,
        &k_bshd,
        &v_bshd,
        scale,
        AttnMask::Causal { kv_offset },
    )?;
    // dispatch_flash_attn always accumulates and returns F32 regardless of
    // input dtype.
    out.to_dtype(q.dtype())
}

/// Non-causal (full, bidirectional) attention dispatcher: tries the fused
/// GPU flash-attn kernel first ([`super::gpu_flash_attn::try_full`]),
/// falling back to [`full_matmul`] on CPU or when the fused kernel can't
/// serve the call. See [`causal`] for the general dispatcher pattern.
///
/// # Errors
///
/// See [`full_matmul`], plus any error the fused-kernel dispatch itself
/// returns on the `cuda`/`rocm` success path.
pub fn full(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    match super::gpu_flash_attn::try_full(q, k, v, scale) {
        Some(result) => result,
        None => full_matmul(q, k, v, scale),
    }
}

/// Matmul SDPA with a caller-provided arbitrary additive mask (e.g.
/// continuous-batching padding) — no kernel-attempt tier, since the fused
/// GPU kernels only support causal/full/windowed patterns, not arbitrary
/// masks. `mask` must be broadcastable to `[B, H_q, q_len, kv_len]`; unlike
/// [`causal`]'s `mask` parameter, this one is applied as-is, with no
/// assumption about its pattern. See [`prefill_sdpa`]'s doc comment for why
/// the mask is cast to `attn_weights`' native dtype rather than upcasting
/// `attn_weights` to F32.
///
/// # Errors
///
/// Returns a candle error if `q`/`k`/`v`'s shapes are incompatible (GQA
/// head-count divisibility, matching `head_dim`), if `mask` isn't
/// broadcastable to `[B, H_q, q_len, kv_len]`, or if any tensor op fails.
pub fn prefill_masked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: &Tensor,
) -> Result<Tensor> {
    prefill_sdpa(q, k, v, scale, Some(mask))
}

/// Sliding-window attention dispatcher: tries the fused GPU flash-attn
/// kernel first ([`super::gpu_flash_attn::try_windowed`]), falling back to
/// [`windowed_matmul`] on CPU or when the fused kernel can't serve the
/// call. See [`causal`] for the general dispatcher pattern.
///
/// # Errors
///
/// See [`windowed_matmul`], plus any error the fused-kernel dispatch itself
/// returns on the `cuda`/`rocm` success path.
pub fn windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    window_left: usize,
    window_right: usize,
) -> Result<Tensor> {
    match super::gpu_flash_attn::try_windowed(q, k, v, scale, window_left, window_right) {
        Some(result) => result,
        None => windowed_matmul(q, k, v, scale, window_left, window_right),
    }
}

/// Non-causal (full, bidirectional) matmul-only SDPA: every query attends to
/// every key. See [`causal_without_mask`] for the error contract and shape
/// conventions. Internal to `modules/` — [`gpu_flash_attn`](super::gpu_flash_attn)'s
/// `try_full` fallback is the only caller; model code should call [`full`]
/// (the dispatcher) instead.
///
/// # Errors
///
/// See [`causal_without_mask`].
pub(super) fn full_matmul(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    prefill_sdpa(q, k, v, scale, None)
}

/// Sliding-window matmul-only SDPA: a query at `i` sees keys `j` with
/// `i + kv_offset - window_left <= j <= i + kv_offset + window_right`
/// (`kv_offset` computed the same way as [`causal_without_mask`]). See
/// [`causal_without_mask`] for the error contract and shape conventions.
/// Internal to `modules/` — [`gpu_flash_attn`](super::gpu_flash_attn)'s
/// `try_windowed` fallback is the only caller; model code should call
/// `windowed` (the dispatcher) instead.
///
/// # Errors
///
/// See [`causal_without_mask`].
pub(super) fn windowed_matmul(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    window_left: usize,
    window_right: usize,
) -> Result<Tensor> {
    let q_len = q.dim(2)?;
    let kv_len = k.dim(2)?;
    let kv_offset = kv_len.saturating_sub(q_len);
    let mask = build_windowed_mask(
        q_len,
        kv_len,
        kv_offset,
        window_left,
        window_right,
        q.device(),
    )?;
    prefill_sdpa(q, k, v, scale, Some(&mask))
}

/// CPU single-sequence decode: BHSD->BSHD, `dispatch_flash_attn`, cast the
/// kernel's F32 output back to `q`'s dtype, return BHSD. Mirrors the
/// CPU-flash branch `GqaAttention`/Qwen3's decode paths used to hand-roll.
fn decode_cpu_flash(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let q_bshd = q.transpose(1, 2)?;
    let k_bshd = k.transpose(1, 2)?;
    let v_bshd = v.transpose(1, 2)?;

    let attn_mask = match mask {
        Some(mask) => {
            debug_assert!(
                mask.dim(1).is_ok_and(|d| d == 1),
                "CPU flash-attn decode broadcasts one mask row across all heads; \
                 a mask with head dim != 1 would be silently misapplied"
            );
            // AttnMask::Mask takes ownership. Tensor is Arc-backed, so this
            // is a refcount bump, not a data copy.
            AttnMask::Mask(mask.clone())
        },
        None => AttnMask::None,
    };

    let out = dispatch_flash_attn(&q_bshd, &k_bshd, &v_bshd, scale, attn_mask)?;
    // dispatch_flash_attn always accumulates and returns F32 regardless of
    // input dtype.
    out.to_dtype(q.dtype())
}

/// GQA-grouped decode matmul: reshapes `q` to `[B, H_kv, n_rep, D]` and dots
/// against `k` without expanding K/V, avoiding the `repeat_kv` cost for a
/// single query position. No F32 softmax upcast — see this module's doc
/// comment for which prior GPU decode paths this does and doesn't match.
/// Returns BHSD `[B, H_q, 1, D]`.
fn decode_grouped_matmul(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b_sz, h_q, _one, head_dim) = q.dims4()?;
    let h_kv = k.dim(1)?;
    let n_rep = h_q / h_kv;

    let q_g = (q.reshape((b_sz, h_kv, n_rep, head_dim))? * f64::from(scale))?;
    let k_t = k.transpose(2, 3)?;
    let attn_weights = q_g.matmul(&k_t)?;
    let attn_weights = match mask {
        Some(mask) => attn_weights.broadcast_add(&mask.to_dtype(attn_weights.dtype())?)?,
        None => attn_weights,
    };
    let attn_weights = softmax_last_dim(&attn_weights)?;
    let attn_output = attn_weights.matmul(v)?;

    attn_output.reshape((b_sz, h_q, 1, head_dim))
}

/// Single-token decode SDPA. `q` is `[B, H_q, 1, D]`, `k`/`v` are
/// `[B, H_kv, kv_len, D]` (BHSD), `H_q` a multiple of `H_kv`. `attention_mask`
/// is an optional additive mask broadcastable to `[B, 1, 1, kv_len]` (e.g.
/// continuous-batching padding).
///
/// On a single-sequence CPU call (`q.dim(0) == 1 && q.device().is_cpu()`),
/// delegates to the existing CPU flash-attn dispatch. Every other case
/// (GPU, or CPU with more than one sequence) uses a GQA-grouped reshape that
/// dots `q` against `k` without expanding K/V. Returns BHSD `[B, H_q, 1, D]`.
///
/// # Errors
///
/// Returns a candle error if `q`/`k`/`v`'s shapes are incompatible or if any
/// tensor op fails.
pub fn decode(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    attention_mask: Option<&Tensor>,
) -> Result<Tensor> {
    if q.dim(0)? == 1 && q.device().is_cpu() {
        return decode_cpu_flash(q, k, v, scale, attention_mask);
    }
    decode_grouped_matmul(q, k, v, scale, attention_mask)
}

/// Reshapes BHSD attention output `[B, H, S, D]` to `[B, S, H*D]` for the
/// output projection, the common step every [`causal`]/[`full`]/[`windowed`]/
/// [`decode`] caller performs afterward. Handles both prefill (`S > 1`,
/// needs a transpose) and decode (`S == 1`, a direct reshape since `H` and
/// `D` are already contiguous).
///
/// # Errors
///
/// Returns a candle error if `attn_output` isn't 4-dimensional or if the
/// reshape/transpose/contiguous ops fail.
pub fn merge_heads(attn_output: &Tensor) -> Result<Tensor> {
    let (b_sz, _h, s, _d) = attn_output.dims4()?;
    if s == 1 {
        attn_output.reshape((b_sz, 1, ()))
    } else {
        attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, s, ()))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    /// `softmax(q @ k^T * scale [+ mask]) @ v` in F32 on the CPU, BHSD
    /// layout, with explicit GQA expansion. This is the ground truth every
    /// test in this module checks against.
    fn naive_attention_bhsd(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        causal: bool,
        kv_offset: usize,
        window: Option<(usize, usize)>,
    ) -> Vec<f32> {
        let (b, hq, sq, d) = q.dims4().unwrap();
        let (_, hkv, skv, _) = k.dims4().unwrap();
        let n_rep = hq / hkv;

        let expand_kv = |t: &Tensor| {
            t.unsqueeze(2)
                .unwrap()
                .expand((b, hkv, n_rep, skv, d))
                .unwrap()
                .reshape((b, hq, skv, d))
                .unwrap()
                .contiguous()
                .unwrap()
        };
        let k = expand_kv(k);
        let v = expand_kv(v);

        let scores = (q.matmul(&k.transpose(2, 3).unwrap()).unwrap() * f64::from(scale)).unwrap();
        let scores = if !causal && window.is_none() {
            scores
        } else {
            let mut mask_vals = vec![0f32; sq * skv];
            for i in 0..sq {
                for j in 0..skv {
                    let rel = j as i64 - (i as i64 + kv_offset as i64);
                    let valid = match window {
                        Some((left, right)) => rel >= -(left as i64) && rel <= right as i64,
                        None => rel <= 0,
                    };
                    mask_vals[i * skv + j] = if valid { 0.0 } else { f32::NEG_INFINITY };
                }
            }
            let mask = Tensor::from_vec(mask_vals, (1, 1, sq, skv), q.device()).unwrap();
            scores.broadcast_add(&mask).unwrap()
        };
        let probs = softmax_last_dim(&scores).unwrap();
        probs
            .matmul(&v)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn assert_close(got: &Tensor, want: &[f32], tol: f32) {
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol * w.abs().max(1.0),
                "[{i}] got {g}, want {w}"
            );
        }
    }

    fn randn(shape: (usize, usize, usize, usize), device: &Device) -> Tensor {
        Tensor::randn(0f32, 1f32, shape, device).unwrap()
    }

    // attention_scale() must compute 1/sqrt(head_dim).
    #[test]
    fn attention_scale_matches_formula() {
        assert!((attention_scale(64) - 0.125).abs() < 1e-6);
        assert!((attention_scale(128) - (1.0 / 128f32.sqrt())).abs() < 1e-6);
    }

    // causal_without_mask() with q_len == kv_len must match a naive causal reference.
    #[test]
    fn causal_matches_naive() {
        let device = Device::Cpu;
        let (b, hq, hkv, s, d) = (1, 4, 2, 6, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = randn((b, hq, s, d), &device);
        let k = randn((b, hkv, s, d), &device);
        let v = randn((b, hkv, s, d), &device);

        let want = naive_attention_bhsd(&q, &k, &v, scale, true, 0, None);
        let got = causal_without_mask(&q, &k, &v, scale).unwrap();
        assert_eq!(got.dims(), &[b, hq, s, d]);
        assert_close(&got, &want, 1e-4);
    }

    // causal_without_mask() with kv_len > q_len must apply the shifted diagonal, not a
    // naive j <= i mask, matching continuation prefill against a KV cache.
    #[test]
    fn causal_with_kv_offset_matches_naive() {
        let device = Device::Cpu;
        let (b, hq, hkv, sq, skv, d) = (1, 4, 2, 3, 9, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;
        let q = randn((b, hq, sq, d), &device);
        let k = randn((b, hkv, skv, d), &device);
        let v = randn((b, hkv, skv, d), &device);

        let want = naive_attention_bhsd(&q, &k, &v, scale, true, kv_offset, None);
        let got = causal_without_mask(&q, &k, &v, scale).unwrap();
        assert_close(&got, &want, 1e-4);
    }

    // causal_with_mask() given the exact mask build_additive_causal_mask() builds
    // must match causal_without_mask()'s own (self-built) output byte-for-byte — this is
    // the shared-mask path Qwen3's decode() relies on to avoid rebuilding
    // the mask once per layer, so it must compute exactly the same thing as
    // the per-call path it replaces there.
    #[test]
    fn causal_with_mask_matches_causal() {
        let device = Device::Cpu;
        let (b, hq, hkv, sq, skv, d) = (1, 4, 2, 3, 9, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;
        let q = randn((b, hq, sq, d), &device);
        let k = randn((b, hkv, skv, d), &device);
        let v = randn((b, hkv, skv, d), &device);

        let want = causal_without_mask(&q, &k, &v, scale).unwrap();
        let mask = CausalMask::new(sq, skv, kv_offset, DType::F32, &device).unwrap();
        let got = causal_with_mask(&q, &k, &v, scale, &mask).unwrap();
        assert_close(
            &got,
            &want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            0.0,
        );
    }

    // causal() on CPU (no GPU feature compiled) with b_sz == 1 and no mask
    // must take the causal_cpu_flash path and match a naive reference
    // (tolerance, not exact: dispatch_flash_attn accumulates in F32
    // internally, a different numerical path than causal_without_mask's
    // matmul SDPA). With an explicit mask it must still fall through to
    // causal_with_mask exactly, matching gpu_flash_attn's
    // try_causal/try_causal_with_mask contract for CPU.
    #[test]
    fn causal_cpu_matches_direct_calls() {
        let device = Device::Cpu;
        let (b, hq, hkv, sq, skv, d) = (1, 4, 2, 3, 9, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;
        let q = randn((b, hq, sq, d), &device);
        let k = randn((b, hkv, skv, d), &device);
        let v = randn((b, hkv, skv, d), &device);

        let want_no_mask = naive_attention_bhsd(&q, &k, &v, scale, true, kv_offset, None);
        let got_no_mask = causal(&q, &k, &v, scale, None).unwrap();
        assert_close(&got_no_mask, &want_no_mask, 1e-3);

        let mask = CausalMask::new(sq, skv, kv_offset, DType::F32, &device).unwrap();
        let want_with_mask = causal_with_mask(&q, &k, &v, scale, &mask).unwrap();
        let got_with_mask = causal(&q, &k, &v, scale, Some(&mask)).unwrap();
        assert_close(
            &got_with_mask,
            &want_with_mask
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            0.0,
        );
    }

    // full_matmul() applies no mask at all.
    #[test]
    fn full_matches_naive() {
        let device = Device::Cpu;
        let (b, hq, hkv, s, d) = (1, 2, 2, 5, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = randn((b, hq, s, d), &device);
        let k = randn((b, hkv, s, d), &device);
        let v = randn((b, hkv, s, d), &device);

        let want = naive_attention_bhsd(&q, &k, &v, scale, false, 0, None);
        let got = full_matmul(&q, &k, &v, scale).unwrap();
        assert_close(&got, &want, 1e-4);
    }

    // windowed_matmul() restricts each query to a band around its shifted position.
    #[test]
    fn windowed_matches_naive() {
        let device = Device::Cpu;
        let (b, hq, hkv, sq, skv, d) = (1, 2, 2, 4, 12, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;
        let q = randn((b, hq, sq, d), &device);
        let k = randn((b, hkv, skv, d), &device);
        let v = randn((b, hkv, skv, d), &device);

        let want = naive_attention_bhsd(&q, &k, &v, scale, false, kv_offset, Some((3, 0)));
        let got = windowed_matmul(&q, &k, &v, scale, 3, 0).unwrap();
        assert_close(&got, &want, 1e-4);
    }

    // full() on CPU (no GPU feature compiled) must fall through to
    // full_matmul() exactly, matching gpu_flash_attn's try_full contract
    // for CPU.
    #[test]
    fn full_dispatcher_cpu_matches_full_matmul() {
        let device = Device::Cpu;
        let (b, hq, hkv, s, d) = (1, 2, 2, 5, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = randn((b, hq, s, d), &device);
        let k = randn((b, hkv, s, d), &device);
        let v = randn((b, hkv, s, d), &device);

        let want = full_matmul(&q, &k, &v, scale).unwrap();
        let got = full(&q, &k, &v, scale).unwrap();
        assert_close(
            &got,
            &want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            0.0,
        );
    }

    // windowed() on CPU (no GPU feature compiled) must fall through to
    // windowed_matmul() exactly, matching gpu_flash_attn's try_windowed
    // contract for CPU.
    #[test]
    fn windowed_dispatcher_cpu_matches_windowed_matmul() {
        let device = Device::Cpu;
        let (b, hq, hkv, sq, skv, d) = (1, 2, 2, 4, 12, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = randn((b, hq, sq, d), &device);
        let k = randn((b, hkv, skv, d), &device);
        let v = randn((b, hkv, skv, d), &device);

        let want = windowed_matmul(&q, &k, &v, scale, 3, 0).unwrap();
        let got = windowed(&q, &k, &v, scale, 3, 0).unwrap();
        assert_close(
            &got,
            &want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            0.0,
        );
    }

    // GQA ratio > 1 must be handled correctly by causal_without_mask()'s repeat_kv expansion.
    #[test]
    fn causal_gqa_expansion_correct() {
        let device = Device::Cpu;
        let (b, hq, hkv, s, d) = (1, 8, 2, 5, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = randn((b, hq, s, d), &device);
        let k = randn((b, hkv, s, d), &device);
        let v = randn((b, hkv, s, d), &device);

        let want = naive_attention_bhsd(&q, &k, &v, scale, true, 0, None);
        let got = causal_without_mask(&q, &k, &v, scale).unwrap();
        assert_close(&got, &want, 1e-4);
    }

    /// Reference matching `GqaAttention`'s old hand-rolled masked-prefill
    /// SDPA: same `repeat_kv` GQA expansion and additive mask as
    /// `prefill_sdpa`, but upcasts `attn_weights` to F32 around softmax
    /// instead of casting only the mask. Returns F32 for comparison via
    /// `assert_close`.
    fn f32_upcast_softmax_reference(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        mask: &Tensor,
    ) -> Vec<f32> {
        let n_rep = q.dim(1).unwrap() / k.dim(1).unwrap();
        let k = repeat_kv(k.clone(), n_rep).unwrap().contiguous().unwrap();
        let v = repeat_kv(v.clone(), n_rep).unwrap().contiguous().unwrap();
        let q = q.contiguous().unwrap();

        let attn_weights = (q
            .matmul(&k.transpose(D::Minus1, D::Minus2).unwrap())
            .unwrap()
            * f64::from(scale))
        .unwrap();
        let attn_weights = attn_weights.broadcast_add(mask).unwrap();
        let input_dtype = attn_weights.dtype();
        let attn_weights = softmax_last_dim(&attn_weights.to_dtype(DType::F32).unwrap())
            .unwrap()
            .to_dtype(input_dtype)
            .unwrap();
        attn_weights
            .matmul(&v)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    // prefill_masked()'s native-dtype softmax must match a reference that
    // upcasts attn_weights to F32 around softmax (GqaAttention's old
    // hand-rolled behavior), for a padding mask of 0.0/NEG_INFINITY in F16
    // — both values are exactly representable in every float format, so
    // the upcast is provably unnecessary and dropping it is safe. BF16 is
    // not tested here: candle's CPU backend doesn't implement `matmul` for
    // BF16 at all (`unsupported dtype BF16 for op matmul`), so a CPU-only
    // test can't exercise it; `gpu_flash_attn.rs`'s BF16 coverage is
    // kernel-based, not matmul-based SDPA. This is the test Round 5 of the
    // attn_dispatch rebase plan requires before replacing GqaAttention's
    // hand-rolled block.
    #[test]
    fn prefill_masked_matches_f32_upcast_softmax() {
        let device = Device::Cpu;
        let (b, hq, hkv, sq, skv, d) = (2, 8, 2, 4, 6, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q_f32 = randn((b, hq, sq, d), &device);
        let k_f32 = randn((b, hkv, skv, d), &device);
        let v_f32 = randn((b, hkv, skv, d), &device);

        // Padding mask: mask out the last 2 of skv kv positions for every query.
        let mut mask_data = vec![0f32; sq * skv];
        for i in 0..sq {
            for j in (skv - 2)..skv {
                mask_data[i * skv + j] = f32::NEG_INFINITY;
            }
        }
        let mask_f32 = Tensor::from_vec(mask_data, (1, 1, sq, skv), &device).unwrap();

        let q = q_f32.to_dtype(DType::F16).unwrap();
        let k = k_f32.to_dtype(DType::F16).unwrap();
        let v = v_f32.to_dtype(DType::F16).unwrap();
        let mask = mask_f32.to_dtype(DType::F16).unwrap();

        let got = prefill_masked(&q, &k, &v, scale, &mask)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let want = f32_upcast_softmax_reference(&q, &k, &v, scale, &mask);
        assert_close(&got, &want, 1e-2);
    }

    // decode() on a single CPU sequence must match dispatch_flash_attn directly.
    #[test]
    fn decode_cpu_single_sequence_matches_flash_attn() {
        let device = Device::Cpu;
        let (b, hq, hkv, skv, d) = (1, 4, 2, 7, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = randn((b, hq, 1, d), &device);
        let k = randn((b, hkv, skv, d), &device);
        let v = randn((b, hkv, skv, d), &device);

        let want = naive_attention_bhsd(&q, &k, &v, scale, false, 0, None);
        let got = decode(&q, &k, &v, scale, None).unwrap();
        assert_eq!(got.dims(), &[b, hq, 1, d]);
        assert_close(&got, &want, 1e-4);
    }

    // decode() with b_sz > 1 on CPU must take the grouped-matmul path (not
    // the single-sequence flash path, which only supports b_sz == 1) and
    // still produce correct output.
    #[test]
    fn decode_cpu_multi_sequence_matches_naive() {
        let device = Device::Cpu;
        let (b, hq, hkv, skv, d) = (2, 8, 2, 6, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = randn((b, hq, 1, d), &device);
        let k = randn((b, hkv, skv, d), &device);
        let v = randn((b, hkv, skv, d), &device);

        let want = naive_attention_bhsd(&q, &k, &v, scale, false, 0, None);
        let got = decode(&q, &k, &v, scale, None).unwrap();
        assert_eq!(got.dims(), &[b, hq, 1, d]);
        assert_close(&got, &want, 1e-4);
    }

    // decode() with n_rep == 1 (no GQA) must still work through the
    // grouped-matmul path's reshape, which is a no-op fold in this case.
    #[test]
    fn decode_n_rep_one() {
        let device = Device::Cpu;
        let (b, h, skv, d) = (2, 4, 5, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = randn((b, h, 1, d), &device);
        let k = randn((b, h, skv, d), &device);
        let v = randn((b, h, skv, d), &device);

        let want = naive_attention_bhsd(&q, &k, &v, scale, false, 0, None);
        let got = decode(&q, &k, &v, scale, None).unwrap();
        assert_close(&got, &want, 1e-4);
    }

    // An explicit mask passed to decode() must change the output, on both
    // the CPU single-sequence path and the grouped-matmul path.
    #[test]
    fn decode_explicit_mask_changes_output() {
        let device = Device::Cpu;
        let (b, hq, hkv, skv, d) = (1, 4, 2, 6, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = randn((b, hq, 1, d), &device);
        let k = randn((b, hkv, skv, d), &device);
        let v = randn((b, hkv, skv, d), &device);

        let y_no_mask = decode(&q, &k, &v, scale, None).unwrap();

        let mut mask_data = vec![-1e9f32; skv];
        mask_data[0] = 0.0;
        let mask = Tensor::from_vec(mask_data, (1, 1, 1, skv), &device).unwrap();
        let y_masked = decode(&q, &k, &v, scale, Some(&mask)).unwrap();

        let diff: f32 = (y_no_mask - y_masked)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff > 1e-4, "explicit mask must change decode output");
    }

    // Cross-check: for n_rep == 1 (no GQA complication), causal_without_mask()'s shifted
    // mask must agree with the CPU flash-attn path's AttnMask::Causal on the
    // exact same kv_offset semantics. This guards against the causal formula
    // drifting out of sync between the two independent implementations.
    #[test]
    fn causal_matches_cpu_flash_attn_causal_mask() {
        let device = Device::Cpu;
        let (b, h, sq, skv, d) = (1, 2, 3, 9, 8usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;
        let q = randn((b, h, sq, d), &device);
        let k = randn((b, h, skv, d), &device);
        let v = randn((b, h, skv, d), &device);

        let q_bshd = q.transpose(1, 2).unwrap();
        let k_bshd = k.transpose(1, 2).unwrap();
        let v_bshd = v.transpose(1, 2).unwrap();
        // dispatch_flash_attn takes BSHD and returns BHSD directly.
        let flash_out = dispatch_flash_attn(
            &q_bshd,
            &k_bshd,
            &v_bshd,
            scale,
            AttnMask::Causal { kv_offset },
        )
        .unwrap();
        let want = flash_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let got = causal_without_mask(&q, &k, &v, scale).unwrap();
        assert_close(&got, &want, 1e-3);
    }

    #[test]
    // Verifies merge_heads's decode path (S == 1): a direct reshape with no
    // transpose, since H and D are already contiguous.
    fn merge_heads_decode_single_token() {
        let device = Device::Cpu;
        let (b, h, d) = (2, 4, 8usize);
        let attn_output = randn((b, h, 1, d), &device);

        let got = merge_heads(&attn_output).unwrap();
        assert_eq!(got.dims(), &[b, 1, h * d]);

        let want = attn_output
            .reshape((b, 1, h * d))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let got_flat = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got_flat, want);
    }

    #[test]
    // Verifies merge_heads's prefill path (S > 1): BHSD -> BSHD transpose
    // before flattening heads, matching the hand-rolled boilerplate it
    // replaces.
    fn merge_heads_prefill_multi_token() {
        let device = Device::Cpu;
        let (b, h, s, d) = (2, 4, 3, 8usize);
        let attn_output = randn((b, h, s, d), &device);

        let got = merge_heads(&attn_output).unwrap();
        assert_eq!(got.dims(), &[b, s, h * d]);

        let want = attn_output
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((b, s, h * d))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let got_flat = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(got_flat, want);
    }
}

/// End-to-end coverage of [`causal`], [`full`], and [`windowed`] on a real
/// GPU (this machine's ROCm GPU, or a CUDA GPU when built with
/// `--features cuda` elsewhere), exercising the fused-kernel dispatch tier
/// that this file's CPU-only `tests` module can't reach — `gpu_flash_attn`'s
/// own `gpu_tests` already cover `try_causal`/`try_causal_with_mask`/
/// `try_full`/`try_windowed` directly, this module checks each dispatcher's
/// wrapper on top of them.
#[cfg(all(test, any(feature = "cuda", feature = "rocm")))]
mod gpu_tests {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::ops::softmax_last_dim;

    use super::{causal, full, windowed};
    use crate::ops::fused_ops::fattn::test_support::test_gpu_device;

    /// `softmax(q @ k^T * scale [+ mask]) @ v` in F32 on the CPU, BHSD
    /// layout, with explicit GQA expansion — the same reference this
    /// module's CPU-only tests use. `causal`/`window` select the masking
    /// pattern the same way as the CPU `tests` module's helper of the same
    /// name: no mask when `causal` is false and `window` is `None`.
    fn naive_attention_bhsd(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        causal: bool,
        kv_offset: usize,
        window: Option<(usize, usize)>,
    ) -> Vec<f32> {
        let (b, hq, sq, d) = q.dims4().unwrap();
        let (_, hkv, skv, _) = k.dims4().unwrap();
        let n_rep = hq / hkv;

        let expand_kv = |t: &Tensor| {
            t.unsqueeze(2)
                .unwrap()
                .expand((b, hkv, n_rep, skv, d))
                .unwrap()
                .reshape((b, hq, skv, d))
                .unwrap()
                .contiguous()
                .unwrap()
        };
        let k = expand_kv(k);
        let v = expand_kv(v);

        let scores = (q.matmul(&k.transpose(2, 3).unwrap()).unwrap() * f64::from(scale)).unwrap();
        let scores = if !causal && window.is_none() {
            scores
        } else {
            let mut mask_vals = vec![0f32; sq * skv];
            for i in 0..sq {
                for j in 0..skv {
                    let rel = j as i64 - (i as i64 + kv_offset as i64);
                    let valid = match window {
                        Some((left, right)) => rel >= -(left as i64) && rel <= right as i64,
                        None => rel <= 0,
                    };
                    mask_vals[i * skv + j] = if valid { 0.0 } else { f32::NEG_INFINITY };
                }
            }
            let mask = Tensor::from_vec(mask_vals, (1, 1, sq, skv), q.device()).unwrap();
            scores.broadcast_add(&mask).unwrap()
        };
        let probs = softmax_last_dim(&scores).unwrap();
        probs
            .matmul(&v)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn got_vec(t: Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol * w.abs().max(1.0),
                "[{i}] got {g}, want {w}"
            );
        }
    }

    // causal() with no mask on a real ROCm GPU must take the fused-kernel
    // tier (not the matmul-SDPA fallback) and match the naive reference.
    #[test]
    fn causal_gpu_no_mask_matches_reference() {
        let rocm = test_gpu_device();
        let (b, hq, hkv, sq, skv, d) = (1, 4, 2, 20, 150, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;

        let q_f32 = Tensor::randn(0f32, 1f32, (b, hq, sq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();
        let want = naive_attention_bhsd(&q_f32, &k_f32, &v_f32, scale, true, kv_offset, None);

        let q = q_f32
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&rocm)
            .unwrap();
        let k = k_f32
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&rocm)
            .unwrap();
        let v = v_f32
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&rocm)
            .unwrap();

        let got = causal(&q, &k, &v, scale, None).unwrap();
        assert_eq!(got.dims(), &[b, hq, sq, d]);
        assert_close(&got_vec(got), &want, 3e-2);
    }

    // causal() with a shared pre-built mask on a real ROCm GPU must take the
    // Some(mask) branch (try_causal_with_mask, not try_causal) and still
    // match the naive reference — the fused kernel now reads the mask
    // tensor directly (see try_causal_with_mask's doc comment), so this also
    // guards that the Some(mask) dispatch arm is wired to the right
    // gpu_flash_attn function and that the mask it threads through is
    // correct, not just present.
    #[test]
    fn causal_gpu_with_mask_matches_reference() {
        let rocm = test_gpu_device();
        let (b, hq, hkv, sq, skv, d) = (1, 4, 2, 20, 150, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;

        let q_f32 = Tensor::randn(0f32, 1f32, (b, hq, sq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();
        let want = naive_attention_bhsd(&q_f32, &k_f32, &v_f32, scale, true, kv_offset, None);

        let q = q_f32
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&rocm)
            .unwrap();
        let k = k_f32
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&rocm)
            .unwrap();
        let v = v_f32
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&rocm)
            .unwrap();
        let mask =
            crate::models::utils::CausalMask::new(sq, skv, kv_offset, DType::F16, &rocm).unwrap();

        let got = causal(&q, &k, &v, scale, Some(&mask)).unwrap();
        assert_eq!(got.dims(), &[b, hq, sq, d]);
        assert_close(&got_vec(got), &want, 3e-2);
    }

    // full() on a real ROCm GPU must take the fused-kernel tier (not the
    // matmul-SDPA fallback) and match the naive bidirectional reference.
    #[test]
    fn full_gpu_matches_reference() {
        let rocm = test_gpu_device();
        let (b, hq, hkv, s, d) = (1, 4, 2, 20, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();

        let q_f32 = Tensor::randn(0f32, 1f32, (b, hq, s, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, hkv, s, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, hkv, s, d), &Device::Cpu).unwrap();
        let want = naive_attention_bhsd(&q_f32, &k_f32, &v_f32, scale, false, 0, None);

        let q = q_f32
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&rocm)
            .unwrap();
        let k = k_f32
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&rocm)
            .unwrap();
        let v = v_f32
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&rocm)
            .unwrap();

        let got = full(&q, &k, &v, scale).unwrap();
        assert_eq!(got.dims(), &[b, hq, s, d]);
        assert_close(&got_vec(got), &want, 3e-2);
    }

    // windowed() on a real ROCm GPU must take the fused-kernel tier (not the
    // matmul-SDPA fallback) and match the naive sliding-window reference.
    #[test]
    fn windowed_gpu_matches_reference() {
        let rocm = test_gpu_device();
        let (b, hq, hkv, sq, skv, d) = (1, 4, 2, 20, 150, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;
        let (window_left, window_right) = (20, 0);

        let q_f32 = Tensor::randn(0f32, 1f32, (b, hq, sq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();
        let want = naive_attention_bhsd(
            &q_f32,
            &k_f32,
            &v_f32,
            scale,
            false,
            kv_offset,
            Some((window_left, window_right)),
        );

        let q = q_f32
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&rocm)
            .unwrap();
        let k = k_f32
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&rocm)
            .unwrap();
        let v = v_f32
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&rocm)
            .unwrap();

        let got = windowed(&q, &k, &v, scale, window_left, window_right).unwrap();
        assert_eq!(got.dims(), &[b, hq, sq, d]);
        assert_close(&got_vec(got), &want, 3e-2);
    }
}
