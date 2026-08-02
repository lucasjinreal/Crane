//! Cross-checks for the normalizations that were collapsed from candle op
//! chains into single fused `rms_norm` launches.
//!
//! Each of these is an algebraic rewrite, not an approximation, so the bar is
//! near-exactness against the formula it replaced — not "close enough". A
//! silently wrong norm does not crash or fail a shape assertion; it degrades
//! generation quality, which is exactly the failure mode a benchmark-driven
//! change can hide.
//!
//! Runs on CPU always, and additionally on the GPU when built with `cuda` or
//! `rocm` — the fused op has a separate kernel per backend, so CPU agreement
//! does not imply device agreement.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use crane_core::models::qwen3_5::Qwen35RmsNorm;
use crane_core::ops::gdn::{
    compute_beta_g, l2_alpha, l2_norm, l2_norm_fused, GdnGateConsts, RmsNormGated,
};

/// Whether this build has a GPU backend compiled in at all.
const HAS_GPU_FEATURE: bool = cfg!(any(feature = "cuda", feature = "rocm"));

/// Devices to check: CPU, plus the GPU when this build has one.
fn devices() -> Vec<Device> {
    #[allow(unused_mut)]
    let mut out = vec![Device::Cpu];
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        out.push(d);
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    if let Ok(d) = Device::new_rocm(0) {
        out.push(d);
    }
    out
}

/// A GPU build whose device never opens would run every check below on the CPU
/// and still report six passes — device coverage that silently isn't. Fail
/// instead, so "tests green" cannot mean "kernel untested".
#[test]
fn gpu_build_actually_exercises_the_gpu() {
    if !HAS_GPU_FEATURE {
        return;
    }
    let devs = devices();
    assert!(
        devs.iter().any(|d| !d.is_cpu()),
        "built with a GPU feature but no GPU device opened: {devs:?}"
    );
}

/// Largest absolute difference between two tensors.
fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()
}

/// Deterministic pseudo-random values — a fixed pattern beats `randn` here
/// because a failure has to be reproducible from the test name alone.
fn values(n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as f32).mul_add(0.7325, 1.0);
            (x.sin() + 0.3 * (x * 2.7).cos()) * scale
        })
        .collect()
}

/// HF's `Qwen3_5RMSNorm`: `x / sqrt(mean(x^2) + eps) * (1 + weight)`, in f32.
fn rms_norm_reference(x: &Tensor, weight: &Tensor, eps: f64, unit_offset: bool) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    let x_normed = x.broadcast_div(&(var + eps)?.sqrt()?)?;
    let scale = weight.to_dtype(DType::F32)?;
    let scale = if unit_offset { scale.affine(1.0, 1.0)? } else { scale };
    x_normed.broadcast_mul(&scale)?.to_dtype(dtype)
}

/// The block/QK norm must match HF's formula including the unit offset. The
/// offset is now folded at load, so this is what catches a fold applied twice
/// or not at all — a mistake worth ~5x on every activation.
#[test]
fn qwen35_rms_norm_matches_hf_formula() -> Result<()> {
    const SIZE: usize = 128;
    let eps = 1e-6;
    for dev in devices() {
        let x = Tensor::from_vec(values(8 * SIZE, 1.7), (8, SIZE), &dev)?;
        let w = Tensor::from_vec(values(SIZE, 0.24), SIZE, &dev)?;

        // `from_folded` takes the scale with the offset already applied, which
        // is what both real load paths hand it.
        let folded = Qwen35RmsNorm::from_folded(w.affine(1.0, 1.0)?, eps);
        let got = folded.forward(&x)?;
        let want = rms_norm_reference(&x, &w, eps, true)?;
        let diff = max_abs_diff(&got, &want)?;
        assert!(diff < 1e-5, "{dev:?}: unit-offset norm differs by {diff}");

        // And without the offset (GGUF weights arrive pre-folded).
        let plain = Qwen35RmsNorm::from_folded(w.clone(), eps);
        let got = plain.forward(&x)?;
        let want = rms_norm_reference(&x, &w, eps, false)?;
        let diff = max_abs_diff(&got, &want)?;
        assert!(diff < 1e-5, "{dev:?}: plain norm differs by {diff}");
    }
    Ok(())
}

/// Guards the fold itself: `load` applies `1 + weight`, `from_folded` does not,
/// so feeding the same raw weight to both must differ. If the fold were
/// dropped, every other assertion here would still pass.
#[test]
fn unit_offset_actually_changes_the_scale() -> Result<()> {
    const SIZE: usize = 32;
    let dev = Device::Cpu;
    let x = Tensor::from_vec(values(4 * SIZE, 1.0), (4, SIZE), &dev)?;
    let w = Tensor::from_vec(values(SIZE, 0.24), SIZE, &dev)?;

    let folded = Qwen35RmsNorm::from_folded(w.affine(1.0, 1.0)?, 1e-6).forward(&x)?;
    let unfolded = Qwen35RmsNorm::from_folded(w, 1e-6).forward(&x)?;
    let diff = max_abs_diff(&folded, &unfolded)?;
    assert!(diff > 1e-2, "unit offset had no effect (diff {diff})");
    Ok(())
}

/// `l2_norm_fused` is `l2_norm` with the `1/sqrt(K)` constant moved between the
/// mean and the scale. Not an approximation — the tolerance is f32 rounding.
#[test]
fn l2_norm_fused_matches_the_op_chain() -> Result<()> {
    let eps = 1e-6;
    for dev in devices() {
        for k in [8usize, 64, 128] {
            let x = Tensor::from_vec(values(6 * k, 2.3), (6, k), &dev)?;
            let alpha = l2_alpha(k, DType::F32, &dev)?;
            let got = l2_norm_fused(&x, &alpha, eps)?;
            let want = l2_norm(&x, eps)?;
            let diff = max_abs_diff(&got, &want)?;
            assert!(diff < 1e-5, "{dev:?} k={k}: l2 norm differs by {diff}");
        }
    }
    Ok(())
}

/// The fused path must not quietly become a plain RMS norm: L2 divides by
/// `sqrt(sum)`, RMS by `sqrt(mean)`, so the two differ by exactly `sqrt(K)`.
#[test]
fn l2_norm_is_not_rms_norm() -> Result<()> {
    const K: usize = 64;
    let dev = Device::Cpu;
    let x = Tensor::from_vec(values(3 * K, 1.0), (3, K), &dev)?;
    let alpha = l2_alpha(K, DType::F32, &dev)?;
    let l2 = l2_norm_fused(&x, &alpha, 1e-6)?;

    let ones = Tensor::ones(K, DType::F32, &dev)?;
    let rms = candle_nn::ops::rms_norm(&x, &ones, 1e-6f32 / K as f32)?;
    // rms = l2 * sqrt(K), so scaling l2 back up must reproduce it.
    let diff = max_abs_diff(&(l2 * (K as f64).sqrt())?, &rms)?;
    assert!(diff < 1e-4, "l2/rms relationship broken: {diff}");
    Ok(())
}

/// The gated GDN norm: `rms_norm(x) * weight * silu(gate)`, HF order.
#[test]
fn gated_rms_norm_matches_the_op_chain() -> Result<()> {
    const SIZE: usize = 128;
    let eps = 1e-6;
    for dev in devices() {
        let x = Tensor::from_vec(values(10 * SIZE, 1.1), (10, SIZE), &dev)?;
        let gate = Tensor::from_vec(values(10 * SIZE, 0.9), (10, SIZE), &dev)?;
        let w = Tensor::from_vec(values(SIZE, 1.0), SIZE, &dev)?;

        let got = RmsNormGated::from_weight(w.clone(), eps).forward(&x, &gate)?;

        // The op chain this replaced, in f32 throughout.
        let xf = x.to_dtype(DType::F32)?;
        let var = xf.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = xf
            .broadcast_div(&(var + eps)?.sqrt()?)?
            .broadcast_mul(&w.to_dtype(DType::F32)?)?;
        let want = normed.broadcast_mul(&candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?)?;

        let diff = max_abs_diff(&got, &want)?;
        assert!(diff < 1e-4, "{dev:?}: gated norm differs by {diff}");
    }
    Ok(())
}

/// Hoisting `-exp(A_log)` and the `dt_bias` broadcast to load time must not
/// change `g`. Checked against the inline formula they were lifted out of.
#[test]
fn precomputed_gate_consts_match_the_inline_formula() -> Result<()> {
    const HEADS: usize = 16;
    const S: usize = 5;
    for dev in devices() {
        let b = Tensor::from_vec(values(S * HEADS, 1.3), (1, S, HEADS), &dev)?;
        let a = Tensor::from_vec(values(S * HEADS, 0.8), (1, S, HEADS), &dev)?;
        let a_log = Tensor::from_vec(values(HEADS, 0.5), HEADS, &dev)?;
        let dt_bias = Tensor::from_vec(values(HEADS, 0.2), HEADS, &dev)?;

        let consts = GdnGateConsts::new(&a_log, &dt_bias)?;
        let (beta, g) = compute_beta_g(&b, &a, &consts, DType::F32)?;

        // Inline original: g = -exp(A_log) * softplus(a + dt_bias).
        let want_beta = candle_nn::ops::sigmoid(&b)?;
        let dt = dt_bias.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(0)?;
        let inner = a.to_dtype(DType::F32)?.broadcast_add(&dt)?;
        let softplus = (Tensor::ones_like(&inner)? + inner.exp()?)?.log()?;
        let want_g = a_log
            .to_dtype(DType::F32)?
            .exp()?
            .neg()?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_mul(&softplus)?;

        assert!(max_abs_diff(&beta, &want_beta)? < 1e-6, "{dev:?}: beta drifted");
        let diff = max_abs_diff(&g, &want_g)?;
        assert!(diff < 1e-5, "{dev:?}: g differs by {diff}");
    }
    Ok(())
}
