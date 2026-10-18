// SPDX-License-Identifier: MIT

//! Chunked (blocked) formulation of the gated delta rule recurrence, used for
//! prefill (`seq_len > 1`) on CUDA, `ROCm`, and SYCL — the backends whose
//! serial per-timestep kernel pays a device kernel-launch/sync cost per
//! token. The CPU never reaches this module: it keeps using the original
//! per-timestep [`super::gated_delta_rule_recurrence`], whose per-op cost is
//! a plain function call rather than a device launch. On Metal it is opt-in
//! (`CRANE_GDN_CHUNKED=1`): the Metal serial kernel is faster at every length.
//!
//! [`super::cuda_backend`] / [`super::rocm_backend`] launch a kernel that
//! scans the whole sequence one timestep at a time — for a 2048-token prefill
//! chunk that is 2048 serial steps with the GPU mostly idle. This module
//! instead expresses the recurrence the way llama.cpp's
//! `build_delta_net_chunking` and HF's `torch_chunk_gated_delta_rule` do: the
//! sequence is split into chunks of `CS` tokens, the intra-chunk dependencies
//! are solved as a small unit-lower-triangular linear system, and only the
//! inter-chunk state carry remains sequential (`S / CS` steps instead of
//! `S`). Every step besides the triangular solve is a batched matmul, so it
//! runs on cuBLAS/rocBLAS instead of per-timestep register updates.
//!
//! The triangular solve itself (building `T_inv = (I + kb)^-1`) is a fused
//! CUDA/HIP kernel (`gdn_chunk_tri_inverse_f32` in `kernels/cuda/gdn.cu`) on
//! CUDA/ROCm: a first version built it with a host-orchestrated loop of 63
//! sequential Candle ops per chunk, which measured 2.4x *slower* than the
//! serial kernel it was meant to replace (launch-count and
//! full-tensor-copy-per-step overhead dominated). SYCL still uses that
//! portable (slow but correct) loop — it doesn't get the fused kernel in
//! this change.
//!
//! Only the non-KDA (scalar gate) path is implemented, matching Qwen 3.5.

// This module's entire public surface has no non-test caller unless a GPU
// feature is enabled: `super::backend::fused_recurrence`, its only
// production caller, is itself `#[cfg(any(feature = "cuda", feature =
// "rocm", feature = "sycl"))]`. Kept unconditionally compiled anyway (rather
// than `#[cfg]`-gating the module) so its unit tests — pure Candle-op math
// with no GPU dependency — keep running in CI's CPU `test` job, the only CI
// job that actually executes tests (GPU-feature builds only compile-check
// there; see AGENTS.md's Build Commands section).
#![cfg_attr(
    not(any(
        feature = "cuda",
        feature = "rocm",
        feature = "sycl",
        feature = "metal"
    )),
    allow(dead_code)
)]

use candle_core::{DType, Result, Tensor};

/// Maximum chunk size the triangular-inverse kernel supports without opting
/// into larger-than-default dynamic shared memory: `2*cs*cs*4 <= 49152`
/// (the 48KB default limit on every CUDA/ROCm GPU without an opt-in call).
#[cfg_attr(not(any(feature = "cuda", feature = "rocm")), allow(dead_code))]
pub(super) const MAX_CHUNK_SIZE: usize = 78;

/// Chunk size for the blocked recurrence. Matches llama.cpp's non-KDA
/// default; override via `CRANE_GDN_CHUNK_SIZE` for tuning. CUDA/ROCm cap the
/// usable chunk size at [`MAX_CHUNK_SIZE`] (78) — larger values work on
/// SYCL's portable fallback but error on CUDA/ROCm.
fn chunk_size() -> usize {
    std::env::var("CRANE_GDN_CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|cs| *cs > 0)
        .unwrap_or(64)
}

/// Chunked gated delta rule recurrence for prefill.
///
/// Drop-in alternative to [`super::cuda_backend::gdn_recurrence_cuda`] /
/// [`super::rocm_backend::gdn_recurrence_rocm`] for `seq_len > 1`: same
/// shapes, same contiguous-f32 contract, pure Candle ops so it runs on any
/// backend. `q` must already be pre-scaled by `1/sqrt(K)` (the caller does
/// this, matching the serial-kernel contract).
///
/// Shapes: `q,k = [BH,S,K]`, `v = [BH,S,V]`, `g,beta = [BH,S]`,
/// `state = [BH,K,V]`. Returns `(y = [BH,S,V], state_out = [BH,K,V])`.
///
/// # Errors
///
/// Returns an error if `seq_len == 0`, if the inputs' shapes are
/// inconsistent, or if any underlying tensor op fails.
pub fn chunked_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    chunked_recurrence_with_cs(q, k, v, g, beta, state, chunk_size())
}

/// Same as [`chunked_recurrence`], with the chunk size passed explicitly
/// rather than read from `CRANE_GDN_CHUNK_SIZE` — lets tests exercise
/// specific chunk sizes without mutating process-global environment state.
///
/// # Errors
///
/// Returns an error if `seq_len == 0`, if the inputs' shapes are
/// inconsistent, or if any underlying tensor op fails.
#[allow(clippy::many_single_char_names)]
/// Match Transformers' portable recurrence under CUDA BF16 autocast.
pub fn chunked_recurrence_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    chunked_recurrence_with_dtype(q, k, v, g, beta, state, 64, Some(DType::BF16))
}

fn chunked_recurrence_with_cs(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    cs: usize,
) -> Result<(Tensor, Tensor)> {
    chunked_recurrence_with_dtype(q, k, v, g, beta, state, cs, None)
}

fn chunked_recurrence_with_dtype(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
    cs: usize,
    mm_dtype: Option<DType>,
) -> Result<(Tensor, Tensor)> {
    let mm = |a: &Tensor, b: &Tensor| -> Result<Tensor> {
        match mm_dtype {
            Some(dtype) => a.to_dtype(dtype)?.matmul(&b.to_dtype(dtype)?),
            None => a.matmul(b),
        }
    };
    // Zero-pad the tail chunk. Padded rows have beta=0 (so k_beta=v_beta=0)
    // and g=0 (so they add no extra decay to g_cum) — they neither
    // contribute to nor disturb the real rows' state updates, and their
    let (bh, s, kdim) = q.dims3()?;
    let vdim = v.dim(2)?;
    let device = q.device();
    if s == 0 {
        candle_core::bail!("gdn chunked recurrence: seq_len must be >= 1, got 0");
    }
    let cs = if mm_dtype.is_some() { cs } else { cs.min(s) };
    let n_chunks = s.div_ceil(cs);
    let s_pad = n_chunks * cs;
    let pad = s_pad - s;
    // output is discarded below.
    let (q, k, v, g, beta) = if pad > 0 {
        (
            q.pad_with_zeros(1, 0, pad)?,
            k.pad_with_zeros(1, 0, pad)?,
            v.pad_with_zeros(1, 0, pad)?,
            g.pad_with_zeros(1, 0, pad)?,
            beta.pad_with_zeros(1, 0, pad)?,
        )
    } else {
        (q.clone(), k.clone(), v.clone(), g.clone(), beta.clone())
    };

    let b_all = bh * n_chunks;
    let q = q.reshape((b_all, cs, kdim))?;
    let k = k.reshape((b_all, cs, kdim))?;
    let v = v.reshape((b_all, cs, vdim))?;
    let g = g.reshape((b_all, cs))?;
    let beta = beta.reshape((b_all, cs))?;

    // Lower-triangular (incl. diagonal) and strictly-lower masks, shared
    // across the whole batch.
    let mask_lower_diag = Tensor::tril2(cs, DType::F32, device)?.reshape((1, cs, cs))?;
    let mask_strict_lower = mask_lower_diag
        .broadcast_sub(&Tensor::eye(cs, DType::F32, device)?.reshape((1, cs, cs))?)?;

    let g_cum = crate::ops::float_math::cumsum_last_dim(&g)?;
    let g_cum_exp = g_cum.exp()?; // [B_all, CS]

    let beta3 = beta.unsqueeze(2)?; // [B_all, CS, 1]
    let k_beta = k.broadcast_mul(&beta3)?; // [B_all, CS, K]
    let v_beta = v.broadcast_mul(&beta3)?; // [B_all, CS, V]

    // decay[b,i,j] = exp(g_cum[b,i] - g_cum[b,j]), masked to i>=j. Masking
    // the exponent (not just the result) before `exp` keeps i<j entries at
    // exp(0)=1 instead of risking overflow from a large positive exponent.
    let diff = g_cum.unsqueeze(2)?.broadcast_sub(&g_cum.unsqueeze(1)?)?; // [B_all, CS, CS]
    let diff_safe = diff.broadcast_mul(&mask_lower_diag)?;
    let decay = diff_safe.exp()?;
    let decay_lower_diag = decay.broadcast_mul(&mask_lower_diag)?;
    let decay_strict_lower = decay.broadcast_mul(&mask_strict_lower)?;

    let k_t = k.transpose(1, 2)?.contiguous()?; // [B_all, K, CS]
    let kb = mm(&k_beta, &k_t)?
        .to_dtype(DType::F32)?
        .mul(&decay_strict_lower)?;
    let kq = mm(&q, &k_t)?.to_dtype(DType::F32)?.mul(&decay_lower_diag)?;

    let g_cum_exp3 = g_cum_exp.unsqueeze(2)?; // [B_all, CS, 1]
    let k_beta_g = k_beta.broadcast_mul(&g_cum_exp3)?; // [B_all, CS, K]
    let q_g_exp = q.broadcast_mul(&g_cum_exp3)?; // [B_all, CS, K]

    let g_last = g_cum.narrow(1, cs - 1, 1)?; // [B_all, 1]
    let g_diff = g_last.broadcast_sub(&g_cum)?; // [B_all, CS]
    let k_gdiff = k.broadcast_mul(&g_diff.exp()?.unsqueeze(2)?)?; // [B_all, CS, K]
    let g_last_exp = g_last.exp()?; // [B_all, 1]

    // T_inv = (I + kb)^-1. See `build_tri_inverse`: fused kernel on CUDA/ROCm,
    // portable row-by-row fallback elsewhere.
    #[cfg(feature = "cuda")]
    let t_inv = if mm_dtype.is_some() && kb.device().is_cuda() {
        super::cuda_backend::gdn_chunk_tri_inverse_torch_cuda(&kb.contiguous()?)?
    } else {
        build_tri_inverse(&kb, cs)?
    };
    #[cfg(not(feature = "cuda"))]
    let t_inv = build_tri_inverse(&kb, cs)?;

    let v_corrected = mm(&t_inv, &v_beta)?;
    let k_cd = mm(&t_inv, &k_beta_g)?;

    // Unmerge chunks from the batch dim for the sequential inter-chunk loop.
    let v_corrected = v_corrected.reshape((bh, n_chunks, cs, vdim))?;
    let k_cd = k_cd.reshape((bh, n_chunks, cs, kdim))?;
    let kq = kq.reshape((bh, n_chunks, cs, cs))?;
    let q_g_exp = q_g_exp.reshape((bh, n_chunks, cs, kdim))?;
    let k_gdiff = k_gdiff.reshape((bh, n_chunks, cs, kdim))?;
    let g_last_exp = g_last_exp.reshape((bh, n_chunks, 1, 1))?;

    let mut state = state.contiguous()?;
    let mut y_chunks = Vec::with_capacity(n_chunks);
    for c in 0..n_chunks {
        let k_cd_c = k_cd.narrow(1, c, 1)?.squeeze(1)?; // [BH, CS, K]
        let v_corrected_c = v_corrected.narrow(1, c, 1)?.squeeze(1)?; // [BH, CS, V]
        let kq_c = kq.narrow(1, c, 1)?.squeeze(1)?; // [BH, CS, CS]
        let q_g_exp_c = q_g_exp.narrow(1, c, 1)?.squeeze(1)?; // [BH, CS, K]
        let k_gdiff_c = k_gdiff.narrow(1, c, 1)?.squeeze(1)?; // [BH, CS, K]
        let g_last_exp_c = g_last_exp.narrow(1, c, 1)?.squeeze(1)?; // [BH, 1, 1]

        let v_prime = mm(&k_cd_c, &state)?;
        let v_new = (v_corrected_c - v_prime)?; // [BH, CS, V]
        let y_intra = mm(&kq_c, &v_new)?;
        let y_inter = mm(&q_g_exp_c, &state)?;
        y_chunks.push((y_inter + y_intra)?);

        let kgv = mm(&k_gdiff_c.transpose(1, 2)?.contiguous()?, &v_new)?;
        state = (state.broadcast_mul(&g_last_exp_c)? + kgv.to_dtype(DType::F32)?)?;
    }

    let y = Tensor::cat(&y_chunks, 1)?.narrow(1, 0, s)?.contiguous()?;
    Ok((y, state))
}

/// Inverse of the unit lower-triangular `[B_all,cs,cs]` matrix `I + kb`.
/// Fused single-launch kernel on CUDA/ROCm; portable (but slow — see
/// [`build_tri_inverse_portable`]) row-by-row Candle ops elsewhere.
fn build_tri_inverse(kb: &Tensor, cs: usize) -> Result<Tensor> {
    let kb = &kb.contiguous()?;
    #[cfg(feature = "cuda")]
    if kb.device().is_cuda() {
        return super::cuda_backend::gdn_chunk_tri_inverse_cuda(kb);
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    if kb.device().is_rocm() {
        return super::rocm_backend::gdn_chunk_tri_inverse_rocm(kb);
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    if kb.device().is_metal()
        && let Some(t_inv) = super::metal_backend::gdn_chunk_tri_inverse_metal(kb)?
    {
        return Ok(t_inv);
    }
    build_tri_inverse_portable(kb, cs)
}

/// Device-portable reference for [`build_tri_inverse`]: builds `T_inv` row by
/// row by forward substitution. `kb` is strictly lower triangular, so row `i`
/// only depends on rows `< i`; since `kb[i,k]=0` for `k>=i`, multiplying the
/// *whole* (still partly zero) `T_inv` against row `i` of `-kb` is safe even
/// before later rows are filled in.
///
/// Measured 2.4x *slower* than the serial kernel it was meant to replace when
/// this was the only implementation: `slice_assign` isn't in-place, so each
/// of the `cs-1` steps copies the entire `[B_all,cs,cs]` tensor, and each
/// step is a separate host-orchestrated launch. Kept only as the SYCL
/// fallback (and exercised directly, on the CPU device, by this module's own
/// unit tests); CUDA/ROCm use the fused kernel instead.
fn build_tri_inverse_portable(kb: &Tensor, cs: usize) -> Result<Tensor> {
    let b_all = kb.dim(0)?;
    let device = kb.device();
    let neg_kb = kb.neg()?;
    let identity = Tensor::eye(cs, DType::F32, device)?
        .reshape((1, cs, cs))?
        .broadcast_as((b_all, cs, cs))?
        .contiguous()?;
    let mut t_inv = Tensor::zeros((b_all, cs, cs), DType::F32, device)?;
    for i in 0..cs {
        let contrib = neg_kb.narrow(1, i, 1)?.matmul(&t_inv)?; // [B_all,1,CS]
        let e_i = identity.narrow(1, i, 1)?; // [B_all,1,CS]
        let row_i = (e_i + contrib)?;
        t_inv = t_inv.slice_assign(&[0..b_all, i..i + 1, 0..cs], &row_i)?;
    }
    Ok(t_inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::gdn::gated_delta_rule_recurrence;
    use candle_core::Device;

    /// Pins the row-recurrence formula against a hand-checked 3x3 example,
    /// independent of the GPU kernel (which must match this same formula).
    #[test]
    fn tri_inverse_matches_hand_checked_example() -> Result<()> {
        let dev = Device::Cpu;
        #[rustfmt::skip]
        let kb = Tensor::new(
            &[[[0f32, 0., 0.],
               [1.,   0., 0.],
               [2.,   3., 0.]]],
            &dev,
        )?;
        let t_inv = build_tri_inverse_portable(&kb, 3)?;
        #[rustfmt::skip]
        let expected = Tensor::new(
            &[[[1f32, 0., 0.],
               [-1.,  1., 0.],
               [1.,  -3., 1.]]],
            &dev,
        )?;
        let diff = (t_inv - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-5, "tri inverse diff too large: {diff}");
        Ok(())
    }

    /// The fused ROCm kernel against the portable reference it replaces, on
    /// the same device — a divergence is the kernel's, not a layout mismatch.
    /// Returns early when the machine has no ROCm device, matching
    /// `tests/rocm_kernels.rs`'s convention.
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    #[test]
    fn tri_inverse_rocm_matches_portable_reference() -> Result<()> {
        let Ok(dev) = Device::new_rocm(0) else {
            return Ok(());
        };
        let (b_all, cs) = (37usize, 64usize);

        // A random strictly-lower-triangular kb: full random, masked to
        // strict-lower so (I + kb) stays well-conditioned for the solve.
        let raw = Tensor::randn(0f32, 1.0, (b_all, cs, cs), &dev)?;
        let mask = Tensor::tril2(cs, DType::F32, &dev)?
            .broadcast_sub(&Tensor::eye(cs, DType::F32, &dev)?)?
            .reshape((1, cs, cs))?;
        let kb = raw.broadcast_mul(&mask)?.contiguous()?;

        let t_inv_kernel = super::super::rocm_backend::gdn_chunk_tri_inverse_rocm(&kb)?;
        let kb_cpu = kb.to_device(&Device::Cpu)?;
        let t_inv_ref = build_tri_inverse_portable(&kb_cpu, cs)?.to_device(&dev)?;

        let diff = (t_inv_kernel - t_inv_ref)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-3, "rocm tri inverse diff too large: {diff}");
        Ok(())
    }

    /// The fused CUDA kernel against the portable reference it replaces, on
    /// the same device — a divergence is the kernel's, not a layout mismatch.
    /// Returns early when the machine has no CUDA device, matching
    /// `tri_inverse_rocm_matches_portable_reference`'s convention.
    #[cfg(feature = "cuda")]
    #[test]
    fn tri_inverse_cuda_matches_portable_reference() -> Result<()> {
        let Ok(dev) = Device::new_cuda(0) else {
            return Ok(());
        };
        let (b_all, cs) = (37usize, 64usize);

        // A random strictly-lower-triangular kb: full random, masked to
        // strict-lower so (I + kb) stays well-conditioned for the solve.
        let raw = Tensor::randn(0f32, 1.0, (b_all, cs, cs), &dev)?;
        let mask = Tensor::tril2(cs, DType::F32, &dev)?
            .broadcast_sub(&Tensor::eye(cs, DType::F32, &dev)?)?
            .reshape((1, cs, cs))?;
        let kb = raw.broadcast_mul(&mask)?.contiguous()?;

        let t_inv_kernel = super::super::cuda_backend::gdn_chunk_tri_inverse_cuda(&kb)?;
        let kb_cpu = kb.to_device(&Device::Cpu)?;
        let t_inv_ref = build_tri_inverse_portable(&kb_cpu, cs)?.to_device(&dev)?;

        let diff = (t_inv_kernel - t_inv_ref)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-3, "cuda tri inverse diff too large: {diff}");
        Ok(())
    }

    /// Chunked recurrence must match the per-timestep reference for a
    /// sequence shorter than one chunk.
    #[test]
    fn matches_reference_within_one_chunk() -> Result<()> {
        let dev = Device::Cpu;
        let (bh, s, k, v) = (2usize, 5usize, 3usize, 4usize);
        let q = Tensor::rand(0f32, 1f32, (bh, s, k), &dev)?;
        let kk = Tensor::rand(0f32, 1f32, (bh, s, k), &dev)?;
        let vv = Tensor::rand(0f32, 1f32, (bh, s, v), &dev)?;
        let g = (Tensor::rand(0f32, 1f32, (bh, s), &dev)? * -0.1)?;
        let beta = candle_nn::ops::sigmoid(&Tensor::rand(-1f32, 1f32, (bh, s), &dev)?)?;
        let state0 = Tensor::zeros((bh, k, v), DType::F32, &dev)?;

        // `chunked_recurrence` expects q pre-scaled by 1/sqrt(K) (same
        // contract as `gdn_recurrence_cuda`); the reference applies that
        // scale itself, so only the chunked call needs it done here.
        let q_scaled = q.affine(1.0 / (k as f64).sqrt(), 0.0)?;
        let (y_chunked, state_chunked) =
            chunked_recurrence(&q_scaled, &kk, &vv, &g, &beta, &state0)?;

        // Reference expects [B,S,H,*] / [B,S,H]; treat bh as B with H=1.
        let q4 = q.reshape((bh, s, 1, k))?;
        let k4 = kk.reshape((bh, s, 1, k))?;
        let v4 = vv.reshape((bh, s, 1, v))?;
        let g3 = g.reshape((bh, s, 1))?;
        let beta3 = beta.reshape((bh, s, 1))?;
        let mut state_ref = state0.reshape((bh, 1, k, v))?;
        let y_ref = gated_delta_rule_recurrence(&q4, &k4, &v4, &g3, &beta3, &mut state_ref)?;
        let y_ref = y_ref.reshape((bh, s, v))?;

        let diff = (y_chunked - &y_ref)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-3, "y diff too large: {diff}");

        let state_diff = (state_chunked - state_ref.reshape((bh, k, v))?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(state_diff < 1e-3, "state diff too large: {state_diff}");
        Ok(())
    }

    /// A sequence spanning multiple chunks, including a padded final chunk,
    /// must still match the reference (exercises the inter-chunk loop and
    /// zero-padding).
    #[test]
    fn matches_reference_across_padded_chunks() -> Result<()> {
        let dev = Device::Cpu;
        let (bh, s, k, v) = (1usize, 20usize, 2usize, 2usize);
        let q = Tensor::rand(0f32, 1f32, (bh, s, k), &dev)?;
        let kk = Tensor::rand(0f32, 1f32, (bh, s, k), &dev)?;
        let vv = Tensor::rand(0f32, 1f32, (bh, s, v), &dev)?;
        let g = (Tensor::rand(0f32, 1f32, (bh, s), &dev)? * -0.1)?;
        let beta = candle_nn::ops::sigmoid(&Tensor::rand(-1f32, 1f32, (bh, s), &dev)?)?;
        let state0 = Tensor::zeros((bh, k, v), DType::F32, &dev)?;

        // `chunked_recurrence` expects q pre-scaled by 1/sqrt(K) (same
        // contract as `gdn_recurrence_cuda`); the reference applies that
        // scale itself, so only the chunked call needs it done here.
        let q_scaled = q.affine(1.0 / (k as f64).sqrt(), 0.0)?;
        // A chunk size smaller than `s` and not a divisor of it, so padding
        // is exercised.
        let (y_chunked, state_chunked) =
            chunked_recurrence_with_cs(&q_scaled, &kk, &vv, &g, &beta, &state0, 8)?;

        let q4 = q.reshape((bh, s, 1, k))?;
        let k4 = kk.reshape((bh, s, 1, k))?;
        let v4 = vv.reshape((bh, s, 1, v))?;
        let g3 = g.reshape((bh, s, 1))?;
        let beta3 = beta.reshape((bh, s, 1))?;
        let mut state_ref = state0.reshape((bh, 1, k, v))?;
        let y_ref = gated_delta_rule_recurrence(&q4, &k4, &v4, &g3, &beta3, &mut state_ref)?;
        let y_ref = y_ref.reshape((bh, s, v))?;

        let diff = (y_chunked - &y_ref)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-3, "y diff too large: {diff}");

        let state_diff = (state_chunked - state_ref.reshape((bh, k, v))?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(state_diff < 1e-3, "state diff too large: {state_diff}");
        Ok(())
    }
}
