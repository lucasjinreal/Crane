// SPDX-License-Identifier: MIT
//! High-level dispatch for the vendored flash-attention prefill kernels
//! (`ops/fused_ops/fattn_mma.rs` tensor-core, `ops/fused_ops/fattn_tile.rs`
//! scalar fallback). `pub(super)`, an implementation detail of
//! [`attn_dispatch`](super::attn_dispatch). Models call
//! [`attn_dispatch::causal`](super::attn_dispatch::causal)/[`full`](super::attn_dispatch::full)/[`windowed`](super::attn_dispatch::windowed),
//! not the functions here.
//!
//! [`try_causal`], [`try_causal_with_mask`], [`try_full`], [`try_windowed`]
//! return `None` on CPU (caller uses its own path), `Some(Ok(..))` on
//! kernel success, `Some(Err(..))` on kernel failure (propagated, never
//! swallowed). Unsupported `head_dim` or missing `cuda`/`rocm` feature
//! falls back to `attn_dispatch`'s matmul SDPA via `Some`.
//!
//! `CRANE_DISABLE_MMA_FLASH_ATTN=1` forces the tile kernel, skipping
//! tensor-core dispatch entirely (see [`mma_disabled`]).
//!
//! **Known gap:** the tile kernel produces numerically wrong output at
//! `head_dim=64` on real `ROCm` hardware, for reasons not yet root-caused
//! (see `ops::fused_ops::fattn_tile`'s `#[ignore]`d `causal_hd64`/
//! `full_no_mask_hd64` tests). A `head_dim=64` call that can't reach the MMA
//! kernel (AMD plain-MHA shapes, or an unpadded `seq_kv`, see
//! `ops::fused_ops::fattn::select_mma_ncols`'s doc comment) never reaches
//! this known-bad tile path: [`run`] routes it straight to the caller's
//! verified `attn_dispatch` matmul-SDPA fallback instead.

use candle_core::{Result, Tensor};

#[cfg(any(feature = "cuda", feature = "rocm"))]
use crate::ops::fused_ops::fattn_mma::{
    fattn_mma_bf16_causal, fattn_mma_bf16_causal_with_mask, fattn_mma_bf16_full,
    fattn_mma_bf16_windowed, fattn_mma_causal, fattn_mma_causal_with_mask, fattn_mma_full,
    fattn_mma_windowed,
};
#[cfg(any(feature = "cuda", feature = "rocm"))]
use crate::ops::fused_ops::fattn_tile::{
    fattn_tile_causal, fattn_tile_causal_with_mask, fattn_tile_full, fattn_tile_windowed,
};
#[cfg(any(feature = "cuda", feature = "rocm"))]
use candle_core::DType;
#[cfg(any(feature = "cuda", feature = "rocm"))]
use std::sync::Once;

/// Ensures the dtype-narrowing/widening note in [`run`] logs once per
/// process rather than once per forward pass. Every dtype needs *some*
/// cast for this kernel family's mixed-precision contract (`q` must be F32,
/// `k`/`v` must be F16, see `ops::fused_ops::fattn_mma`'s module doc): F16
/// input needs `q` widened to F32, F32 input needs `k`/`v` narrowed to F16,
/// and BF16 input needs both.
#[cfg(any(feature = "cuda", feature = "rocm"))]
static CAST_WARNED: Once = Once::new();

/// Ensures the MMA-kernel-fell-back-to-tile note in [`run`] logs once per
/// process rather than once per forward pass.
#[cfg(any(feature = "cuda", feature = "rocm"))]
static MMA_FALLBACK_WARNED: Once = Once::new();

/// Ensures the head_dim=64-skipped-known-bad-tile-kernel note in [`run`]
/// logs once per process rather than once per forward pass.
#[cfg(any(feature = "cuda", feature = "rocm"))]
static TILE_HD64_FALLBACK_WARNED: Once = Once::new();

/// Whether `CRANE_DISABLE_MMA_FLASH_ATTN` is set, forcing [`run`] to skip
/// the tensor-core kernel and always use the tile kernel. Escape hatch for
/// the RDNA3/RDNA3.5/NVIDIA Ampere+ MMA paths this module's doc comment
/// calls out as unverified on real hardware: a kernel that launches but
/// computes wrong values isn't caught by the launch-error fallback below, so
/// an operator on one of those architectures (or anyone who hits a
/// correctness issue later) needs a way to force the tile kernel without
/// rebuilding. Read once per process, the same `OnceLock`-cached pattern as
/// `qwen3_5::modeling::legacy_attn_expand`.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn mma_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("CRANE_DISABLE_MMA_FLASH_ATTN").is_ok_and(|v| !matches!(v.trim(), "" | "0"))
    })
}

/// Whether `CRANE_DISABLE_BF16_FLASH_ATTN` is set, forcing [`run`] to skip
/// the native-bf16 MMA kernel and fall through to the existing BF16->F16
/// cast path. Escape hatch for `fattn_mma_bf16.cuh`, which (unlike the f16
/// kernel) has no production usage history yet: a kernel that launches but
/// computes wrong values isn't caught by the launch-error fallback below, so
/// this gives a way to force the proven cast-to-F16 path without rebuilding.
/// Same `OnceLock`-cached pattern as [`mma_disabled`].
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn bf16_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("CRANE_DISABLE_BF16_FLASH_ATTN").is_ok_and(|v| !matches!(v.trim(), "" | "0"))
    })
}

/// `kv_offset` for continuation prefill against an append-only KV cache:
/// `k`'s sequence length minus `q`'s, as `AttnMask::Causal`'s CPU-path
/// counterpart already does. `q`/`k` are BHSD, so the sequence axis is
/// index 2.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn kv_offset(q: &Tensor, k: &Tensor) -> Result<usize> {
    Ok(k.dim(2)?.saturating_sub(q.dim(2)?))
}

/// Whether `q`'s `head_dim` is one the vendored kernels instantiate
/// (`ops::fused_ops::fattn`'s module doc: {64, 128, 256, 512}). `q` is
/// BHSD, so `head_dim` is axis 3. Checked here so `try_causal`/`try_full`/
/// `try_windowed` can return `None` for this case, matching their
/// documented "incapable of serving this call" contract, instead of
/// propagating a `validate_fattn_bshd` `Err` for a case that isn't actually
/// a failure, just an unsupported model shape.
///
/// On AMD, `head_dim > 256` is excluded even though it structurally fits
/// the `{64, 128, 256, 512}` set: `select_mma_ncols`'s doc comment confirms
/// (via a real hardware exception) that both `AMD_WMMA_AVAILABLE` and
/// `AMD_MFMA_AVAILABLE` reject `DKQ > 256` unconditionally, so the MMA
/// kernel never launches there, every call falls through to the scalar
/// tile kernel. Measured on real ROCm hardware (Gemma4's `head_dim=512`
/// full-attention layers, `gqa_ratio=8`), that tile-kernel path is slower
/// than this module's own matmul-SDPA fallback, not just slower than MMA:
/// a 29904-token prefill dropped from 904.7 tok/s to 131.7 tok/s. Returning
/// `false` here skips the kernel attempt entirely on AMD for this shape, so
/// `try_causal`/`try_full`/`try_windowed` go straight to matmul SDPA
/// instead. NVIDIA CUDA has no such restriction (`is_amd_backend` is a
/// compile-time, `cuda`-takes-priority check, so this exclusion never
/// applies to a `cuda`-feature build) and may still reach a real MMA kernel
/// at `head_dim=512`.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn head_dim_supported(q: &Tensor) -> bool {
    use crate::ops::fused_ops::fattn::is_amd_backend;

    match q.dim(3) {
        Ok(d @ (64 | 128 | 256 | 512)) => !(is_amd_backend() && d > 256),
        _ => false,
    }
}

/// Ensures the bf16-kernel-fell-back-to-cast note in [`run`] logs once per
/// process rather than once per forward pass.
#[cfg(any(feature = "cuda", feature = "rocm"))]
static BF16_FALLBACK_WARNED: Once = Once::new();

/// Transposes BHSD to BSHD (zero-copy). When `q`'s original dtype is BF16
/// and native bf16 dispatch isn't disabled (see [`bf16_disabled`]), tries
/// `mma_bf16` first (`k`/`v` passed through as BF16, no downcast) before
/// falling through to the cast-based path below; `mma_bf16`'s failure is
/// silent and falls through only when it is tagged "unsupported config"
/// (`ops::fused_ops::fattn::is_unsupported_config`: no instantiated kernel
/// for this shape, a GQA/mask combination the kernel can't serve) — a
/// genuine runtime failure (a launch rejection, an invalid argument, OOM)
/// propagates instead, since the cast path exists to serve shapes the
/// native kernel doesn't cover, not to paper over a real failure.
///
/// The cast path then casts to this kernel family's mixed-precision
/// contract (`q` F32, `k`/`v` F16; see this module's doc comment), tries
/// `mma` then falls back to `tile`, then transposes the result back to BHSD
/// (also zero-copy, since both kernels are stride-aware) and casts back to
/// the caller's original dtype.
///
/// `mma`'s failure falls through to `tile` under the same "unsupported
/// config" rule as `mma_bf16` above (unsupported GPU architecture, no
/// instantiated kernel for this `(head_dim, GQA ratio, seq_q, seq_kv)`
/// shape, an unservable mask/GQA combination); any other error (a launch
/// rejection, ...) propagates rather than falling through, matching
/// `try_causal` et al.'s `Option`-style contract one layer up. A kernel
/// that launches but computes wrong values returns `Ok` here and is not
/// caught, except the known `head_dim=64` tile bug this module's doc
/// comment calls out, which routes to `fallback` instead of calling `tile`.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn run(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mma_bf16: impl FnOnce(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    mma: impl FnOnce(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    tile: impl FnOnce(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    fallback: impl FnOnce() -> Result<Tensor>,
) -> Result<Tensor> {
    use ribo::utils::log;

    let orig_dtype = q.dtype();
    let q_bshd = q.transpose(1, 2)?;
    let k_bshd = k.transpose(1, 2)?;
    let v_bshd = v.transpose(1, 2)?;

    let finish_bf16 = |out_bshd: Tensor| -> Result<Tensor> { out_bshd.transpose(1, 2) };

    if orig_dtype == DType::BF16 && !bf16_disabled() {
        // See run's doc comment: ops::rocm::device_ptr requires full
        // contiguity, which the BHSD->BSHD transpose above doesn't provide
        // on its own (it only swaps axes 1/2 without moving data).
        let try_bf16 = (|| {
            let q_f32 = q_bshd.to_dtype(DType::F32)?;
            let k_bf16 = k_bshd.contiguous()?;
            let v_bf16 = v_bshd.contiguous()?;
            mma_bf16(&q_f32, &k_bf16, &v_bf16)
        })();
        match try_bf16 {
            Ok(out_bshd) => return finish_bf16(out_bshd),
            Err(e) if crate::ops::fused_ops::fattn::is_unsupported_config(&e) => {
                BF16_FALLBACK_WARNED.call_once(|| {
                    log::debug!(
                        "gpu_flash_attn: native-bf16 MMA kernel unavailable, falling back to the BF16->F16 cast path (logged once): {e}"
                    );
                });
            },
            Err(e) => return Err(e),
        }
    }

    // ops::rocm::device_ptr requires full contiguity, not just a contiguous
    // last axis (unlike validate_fattn_bshd's own check): the BHSD->BSHD
    // transpose above only swaps axes 1/2 without moving data, so a branch
    // that skips the dtype cast must contiguous() explicitly instead. The
    // cast branches get this for free, since to_dtype's elementwise copy
    // always produces a contiguous result.
    let q_f32 = if orig_dtype == DType::F32 {
        q_bshd.contiguous()?
    } else {
        CAST_WARNED.call_once(|| {
            log::debug!(
                "gpu_flash_attn: casting {orig_dtype:?} q to F32 and k/v to F16 for the vendored kernels (logged once)"
            );
        });
        q_bshd.to_dtype(DType::F32)?
    };
    let (k_f16, v_f16) = if orig_dtype == DType::F16 {
        (k_bshd.contiguous()?, v_bshd.contiguous()?)
    } else {
        CAST_WARNED.call_once(|| {
            log::debug!(
                "gpu_flash_attn: casting {orig_dtype:?} q to F32 and k/v to F16 for the vendored kernels (logged once)"
            );
        });
        (k_bshd.to_dtype(DType::F16)?, v_bshd.to_dtype(DType::F16)?)
    };

    // Casts the fused kernel's BSHD output back to BHSD and to the caller's
    // original dtype. Both fattn_mma_prefill and fattn_tile_prefill always
    // return F16 (their internal accumulation is F32, cast down once at the
    // end, independent of q's dtype), so this is a no-op only when the
    // caller's original dtype was already F16.
    let finish = |out_bshd: Tensor| -> Result<Tensor> {
        let out_bhsd = out_bshd.transpose(1, 2)?;
        if orig_dtype == DType::F16 {
            Ok(out_bhsd)
        } else {
            out_bhsd.to_dtype(orig_dtype)
        }
    };

    if !mma_disabled() {
        match mma(&q_f32, &k_f16, &v_f16) {
            Ok(out_bshd) => return finish(out_bshd),
            Err(e) if crate::ops::fused_ops::fattn::is_unsupported_config(&e) => {
                MMA_FALLBACK_WARNED.call_once(|| {
                    log::debug!(
                        "gpu_flash_attn: MMA kernel unavailable, falling back to the tile kernel (logged once): {e}"
                    );
                });
            },
            Err(e) => return Err(e),
        }
    }

    // See this module's doc comment: head_dim=64 on the tile kernel is a
    // known-wrong-output path on real ROCm hardware, not yet root-caused.
    // Skip the known-bad kernel entirely and route to the caller's verified
    // matmul-SDPA fallback instead of silently returning a wrong result.
    if matches!(q_f32.dim(3), Ok(64)) {
        TILE_HD64_FALLBACK_WARNED.call_once(|| {
            log::debug!(
                "gpu_flash_attn: head_dim=64 tile kernel is known to produce wrong output on \
                 real ROCm hardware (not yet root-caused); using the matmul-SDPA fallback \
                 instead (logged once)"
            );
        });
        return fallback();
    }

    let out_bshd = tile(&q_f32, &k_f16, &v_f16)?;
    finish(out_bshd)
}

/// Tries causal flash-attention: `j <= i + kv_offset`, where
/// `kv_offset = k`'s sequence length minus `q`'s. This is the shifted
/// diagonal that continuation prefill against an append-only KV cache
/// needs, and matches `AttnMask::Causal`'s CPU-path semantics. Returns
/// `None` only on a CPU device. CPU callers use their own specialized
/// flash-attn path instead. Every other case returns `Some`: either the
/// fused kernel's result, or (unsupported `head_dim`, no GPU feature
/// compiled, or a kernel-chain failure)
/// [`attn_dispatch::causal_without_mask`](super::attn_dispatch::causal_without_mask)'s
/// matmul SDPA.
///
/// `q` is `[B, H_q, S, D]`, `k`/`v` are `[B, H_kv, kv_len, D]` (BHSD), and
/// `H_q` must be a multiple of `H_kv` (GQA, handled natively by both the
/// fused kernel and the `attn_dispatch` fallback, so callers skip any K/V
/// head expansion either way).
///
/// **Per-layer callers: use [`try_causal_with_mask`] instead if you already
/// have a shared mask.** This function's fallback rebuilds a fresh mask (and,
/// on GPU, re-uploads it) on every call, fine for an occasional caller, but
/// wrong for a decoder stack calling it once per layer per forward pass
/// (every layer shares the same mask): that's the exact regression Qwen3's
/// prefill wiring hit and fixed by switching to a shared, once-per-forward-
/// pass mask (see `attn_dispatch`'s module doc). [`try_causal_with_mask`]
/// reuses a caller-provided mask in the fallback case instead.
///
/// # Errors
///
/// The inner `Result` errors under the same conditions as
/// [`ops::fused_ops::fattn_tile::fattn_tile_causal`](crate::ops::fused_ops::fattn_tile::fattn_tile_causal)
/// or [`attn_dispatch::causal_without_mask`](super::attn_dispatch::causal_without_mask).
#[must_use]
pub fn try_causal(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Option<Result<Tensor>> {
    if q.device().is_cpu() {
        return None;
    }
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    if head_dim_supported(q) {
        return Some((|| {
            let offset = kv_offset(q, k)?;
            run(
                q,
                k,
                v,
                |qf, kf, vf| fattn_mma_bf16_causal(qf, kf, vf, scale, offset),
                |qf, kf, vf| fattn_mma_causal(qf, kf, vf, scale, offset),
                |qf, kf, vf| fattn_tile_causal(qf, kf, vf, scale, offset),
                || super::attn_dispatch::causal_without_mask(q, k, v, scale),
            )
        })());
    }
    Some(super::attn_dispatch::causal_without_mask(q, k, v, scale))
}

/// Same as [`try_causal`], but for callers that already have a mask shared
/// across multiple layers in one forward pass (e.g. a decoder stack whose
/// `forward()` builds the mask once and passes it to every layer). Unlike
/// `try_causal`, the fused-kernel success path here threads `mask` straight
/// into [`fattn_mma_causal_with_mask`]/[`fattn_tile_causal_with_mask`]
/// instead of letting them rebuild an equivalent mask from `kv_offset` on
/// every call — this kernel family's mask build is `O(seq_q * seq_kv)`, so
/// doing it once per forward pass and reusing it here (rather than once per
/// layer, as [`try_causal`]'s self-building path does) is the entire point
/// of this function existing separately. `mask` (any dtype; cast to F16
/// internally if needed) must encode exactly the plain shifted-diagonal
/// causal pattern `j <= i + kv_offset` — the fallback path
/// ([`attn_dispatch::causal_with_mask`](super::attn_dispatch::causal_with_mask))
/// reads the same tensor under the same contract, so fused and fallback stay
/// consistent.
///
/// # Errors
///
/// See [`try_causal`].
#[must_use]
pub(crate) fn try_causal_with_mask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: &crate::models::utils::CausalMask,
) -> Option<Result<Tensor>> {
    if q.device().is_cpu() {
        return None;
    }
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    if head_dim_supported(q) {
        return Some((|| {
            let offset = kv_offset(q, k)?;
            run(
                q,
                k,
                v,
                |qf, kf, vf| {
                    fattn_mma_bf16_causal_with_mask(qf, kf, vf, scale, offset, mask.as_tensor())
                },
                |qf, kf, vf| {
                    fattn_mma_causal_with_mask(qf, kf, vf, scale, offset, mask.as_tensor())
                },
                |qf, kf, vf| {
                    fattn_tile_causal_with_mask(qf, kf, vf, scale, offset, mask.as_tensor())
                },
                || super::attn_dispatch::causal_with_mask(q, k, v, scale, mask),
            )
        })());
    }
    Some(super::attn_dispatch::causal_with_mask(q, k, v, scale, mask))
}

/// Tries non-causal (full, bidirectional) flash-attention: every query
/// attends to every key. See [`try_causal`] for the `None`/`Some` contract
/// and the BHSD/GQA shape conventions.
// Not wired into any model's prefill path yet - see `attn_dispatch`'s
// module doc.
#[allow(dead_code)]
#[must_use]
pub(crate) fn try_full(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Option<Result<Tensor>> {
    if q.device().is_cpu() {
        return None;
    }
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    if head_dim_supported(q) {
        return Some(run(
            q,
            k,
            v,
            |qf, kf, vf| fattn_mma_bf16_full(qf, kf, vf, scale),
            |qf, kf, vf| fattn_mma_full(qf, kf, vf, scale),
            |qf, kf, vf| fattn_tile_full(qf, kf, vf, scale),
            || super::attn_dispatch::full_matmul(q, k, v, scale),
        ));
    }
    Some(super::attn_dispatch::full_matmul(q, k, v, scale))
}

/// Tries sliding-window flash-attention: a query at `i` sees keys `j` with
/// `i + kv_offset - window_left <= j <= i + kv_offset + window_right`
/// (`kv_offset` computed the same way as [`try_causal`]). See
/// [`try_causal`] for the `None`/`Some` contract and the BHSD/GQA shape
/// conventions.
#[must_use]
pub(crate) fn try_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    window_left: usize,
    window_right: usize,
) -> Option<Result<Tensor>> {
    if q.device().is_cpu() {
        return None;
    }
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    if head_dim_supported(q) {
        return Some((|| {
            let offset = kv_offset(q, k)?;
            run(
                q,
                k,
                v,
                |qf, kf, vf| {
                    fattn_mma_bf16_windowed(qf, kf, vf, scale, offset, window_left, window_right)
                },
                |qf, kf, vf| {
                    fattn_mma_windowed(qf, kf, vf, scale, offset, window_left, window_right)
                },
                |qf, kf, vf| {
                    fattn_tile_windowed(qf, kf, vf, scale, offset, window_left, window_right)
                },
                || super::attn_dispatch::windowed_matmul(q, k, v, scale, window_left, window_right),
            )
        })());
    }
    Some(super::attn_dispatch::windowed_matmul(
        q,
        k,
        v,
        scale,
        window_left,
        window_right,
    ))
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    use super::*;

    // On a CPU tensor, every entry point must return `None` regardless of
    // which GPU feature (if any) is compiled in, so callers can always fall
    // through to their standard SDPA path without a feature-gated branch of
    // their own.
    #[test]
    fn cpu_device_always_returns_none() {
        let device = Device::Cpu;
        let shape = (1usize, 2usize, 4usize, 128usize);
        let q = Tensor::zeros(shape, DType::F16, &device).unwrap();
        let k = q.clone();
        let v = q.clone();
        let mask = crate::models::utils::CausalMask::new(4, 4, 0, DType::F16, &device).unwrap();

        assert!(try_causal(&q, &k, &v, 0.088).is_none());
        assert!(try_causal_with_mask(&q, &k, &v, 0.088, &mask).is_none());
        assert!(try_full(&q, &k, &v, 0.088).is_none());
        assert!(try_windowed(&q, &k, &v, 0.088, 4, 0).is_none());
    }
}

/// End-to-end coverage of the public BHSD API on a real GPU (this
/// machine's ROCm GPU, or a CUDA GPU when built with `--features cuda`
/// elsewhere), exercising the full path (BHSD->BSHD transpose, dtype cast,
/// BSHD->BHSD transpose back) rather than just the underlying ops in
/// isolation (`ops::fused_ops::fattn_mma`/`fattn_tile`'s own `gpu_tests`
/// already cover those).
#[cfg(all(test, any(feature = "cuda", feature = "rocm")))]
mod gpu_tests {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::ops::softmax_last_dim;

    use crate::ops::fused_ops::fattn::test_support::test_gpu_device;

    use super::{try_causal, try_causal_with_mask, try_full, try_windowed};

    /// `softmax(q @ k^T * scale [+ mask]) @ v` in F32 on the CPU, BHSD
    /// layout (unlike `ops::fused_ops::fattn::test_support`'s BSHD version;
    /// this module tests the BHSD public API directly).
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

    // try_causal on F16 BHSD inputs selects the tensor-core kernel (the
    // only dtype it supports) and must match the naive BHSD reference.
    #[test]
    fn try_causal_f16_matches_reference() {
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

        let got = try_causal(&q, &k, &v, scale)
            .expect("GPU device must return Some")
            .unwrap();
        assert_eq!(got.dims(), &[b, hq, sq, d]);
        assert_close(&got_vec(got), &want, 3e-2);
    }

    // try_full on F16 BHSD inputs.
    #[test]
    fn try_full_f16_matches_reference() {
        let rocm = test_gpu_device();
        let (b, hq, hkv, sq, skv, d) = (1, 2, 2, 20, 20, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();

        let q_f32 = Tensor::randn(0f32, 1f32, (b, hq, sq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();
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

        let got = try_full(&q, &k, &v, scale)
            .expect("GPU device must return Some")
            .unwrap();
        assert_close(&got_vec(got), &want, 3e-2);
    }

    // try_windowed on F16 BHSD inputs, spanning multiple KV chunks.
    #[test]
    fn try_windowed_f16_matches_reference() {
        let rocm = test_gpu_device();
        let (b, hq, hkv, sq, skv, d) = (1, 2, 2, 20, 150, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;

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
            Some((20, 0)),
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

        let got = try_windowed(&q, &k, &v, scale, 20, 0)
            .expect("GPU device must return Some")
            .unwrap();
        assert_close(&got_vec(got), &want, 3e-2);
    }

    // F32 input needs no q cast (already F32) but k/v still narrow to F16;
    // this machine's RDNA4 GPU supports MMA, so this goes through the
    // tensor-core kernel end to end through the public API.
    #[test]
    fn try_causal_f32_uses_mma_kernel() {
        let rocm = test_gpu_device();
        let (b, hq, hkv, sq, skv, d) = (1, 2, 2, 8, 8, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();

        let q_f32 = Tensor::randn(0f32, 1f32, (b, hq, sq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();
        let want = naive_attention_bhsd(&q_f32, &k_f32, &v_f32, scale, true, 0, None);

        let q = q_f32.to_device(&rocm).unwrap();
        let k = k_f32.to_device(&rocm).unwrap();
        let v = v_f32.to_device(&rocm).unwrap();

        let got = try_causal(&q, &k, &v, scale)
            .expect("GPU device must return Some")
            .unwrap();
        assert_eq!(got.dtype(), DType::F32);
        assert_close(&got_vec(got), &want, 6e-2);
    }

    // End-to-end BHSD coverage at GQA ratio 8 (Qwen3-Coder-30B-A3B's ratio):
    // exercises the full dispatch path against the multi-warp MMA kernel's
    // primary target, not just the ratio-2/4 cases above.
    #[test]
    fn try_causal_f16_gqa_ratio_8() {
        let rocm = test_gpu_device();
        let (b, hq, hkv, sq, skv, d) = (1, 8, 1, 20, 150, 128usize);
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

        let got = try_causal(&q, &k, &v, scale)
            .expect("GPU device must return Some")
            .unwrap();
        assert_eq!(got.dims(), &[b, hq, sq, d]);
        assert_close(&got_vec(got), &want, 3e-2);
    }

    // BF16 input needs both q (widen to F32) and k/v (narrow to F16) cast;
    // real models commonly load in BF16, so this is the common real-model
    // path through the tensor-core kernel.
    #[test]
    fn try_causal_bf16_uses_mma_kernel() {
        let rocm = test_gpu_device();
        let (b, hq, hkv, sq, skv, d) = (1, 2, 2, 8, 8, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();

        let q_f32 = Tensor::randn(0f32, 1f32, (b, hq, sq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();
        let want = naive_attention_bhsd(&q_f32, &k_f32, &v_f32, scale, true, 0, None);

        let q = q_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&rocm)
            .unwrap();
        let k = k_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&rocm)
            .unwrap();
        let v = v_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&rocm)
            .unwrap();

        let got = try_causal(&q, &k, &v, scale)
            .expect("GPU device must return Some")
            .unwrap();
        assert_eq!(got.dtype(), DType::BF16);
        assert_close(&got_vec(got), &want, 3e-2);
    }

    // A non-{64,128,256,512} head_dim must fall through to `attn_dispatch`'s
    // matmul SDPA (Some(Ok(..)), not `None` and not a propagated
    // `validate_fattn_bshd` `Err`), and the result must be correct. This
    // exercises the fallback wiring end to end on real (non-CPU) hardware,
    // not just the `attn_dispatch` unit tests that only ever run on CPU.
    #[test]
    fn unsupported_head_dim_falls_back_to_matmul_sdpa() {
        let rocm = test_gpu_device();
        let (b, hq, hkv, sq, skv, d) = (1, 4, 2, 5, 11, 100usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;

        let q_f32 = Tensor::randn(0f32, 1f32, (b, hq, sq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();

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

        let want_causal =
            naive_attention_bhsd(&q_f32, &k_f32, &v_f32, scale, true, kv_offset, None);
        let got_causal = try_causal(&q, &k, &v, scale)
            .expect("GPU device must return Some even for unsupported head_dim")
            .unwrap();
        assert_eq!(got_causal.dims(), &[b, hq, sq, d]);
        assert_close(&got_vec(got_causal), &want_causal, 3e-2);

        let want_full = naive_attention_bhsd(&q_f32, &k_f32, &v_f32, scale, false, 0, None);
        let got_full = try_full(&q, &k, &v, scale)
            .expect("GPU device must return Some even for unsupported head_dim")
            .unwrap();
        assert_close(&got_vec(got_full), &want_full, 3e-2);

        let want_windowed = naive_attention_bhsd(
            &q_f32,
            &k_f32,
            &v_f32,
            scale,
            false,
            kv_offset,
            Some((4, 0)),
        );
        let got_windowed = try_windowed(&q, &k, &v, scale, 4, 0)
            .expect("GPU device must return Some even for unsupported head_dim")
            .unwrap();
        assert_close(&got_vec(got_windowed), &want_windowed, 3e-2);
    }

    // head_dim=64 is structurally supported (head_dim_supported() admits it),
    // so it normally reaches the MMA kernel, which this RDNA4 GPU handles
    // correctly. Forcing CRANE_DISABLE_MMA_FLASH_ATTN=1 skips the MMA
    // attempt, so `run()` reaches the head_dim=64 check with no MMA result
    // to return: before this fix that meant calling the tile kernel, known
    // to produce wrong output at this head_dim on real ROCm hardware; now it
    // must route to `fallback()` and match the naive reference, the same way
    // `unsupported_head_dim_falls_back_to_matmul_sdpa` does for shapes
    // `head_dim_supported()` rejects outright. `mma_disabled()` caches its
    // env var read in a `OnceLock` for the life of the process (same
    // constraint as `bf16_vs_f16_try_causal_with_mask_timing_sweep_fallback`
    // below), so this must run alone in its own process:
    // `cargo test -p crane-core --no-default-features --features "rocm,onnx"
    // -- --ignored --exact
    // models::modules::gpu_flash_attn::gpu_tests::head_dim_64_tile_kernel_routes_to_matmul_fallback
    // --nocapture`
    #[test]
    #[ignore = "sets a process-global env var cached by mma_disabled(); run alone, see doc comment"]
    fn head_dim_64_tile_kernel_routes_to_matmul_fallback() {
        // SAFETY: run with `--exact`, alone in its own process, before any
        // other code reads `CRANE_DISABLE_MMA_FLASH_ATTN`.
        unsafe {
            std::env::set_var("CRANE_DISABLE_MMA_FLASH_ATTN", "1");
        }

        let rocm = test_gpu_device();
        let (b, hq, hkv, sq, skv, d) = (1, 4, 2, 5, 11, 64usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;

        let q_f32 = Tensor::randn(0f32, 1f32, (b, hq, sq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();

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

        let want_causal =
            naive_attention_bhsd(&q_f32, &k_f32, &v_f32, scale, true, kv_offset, None);
        let got_causal = try_causal(&q, &k, &v, scale)
            .expect("GPU device must return Some")
            .unwrap();
        assert_eq!(got_causal.dims(), &[b, hq, sq, d]);
        assert_close(&got_vec(got_causal), &want_causal, 3e-2);

        let want_full = naive_attention_bhsd(&q_f32, &k_f32, &v_f32, scale, false, 0, None);
        let got_full = try_full(&q, &k, &v, scale)
            .expect("GPU device must return Some")
            .unwrap();
        assert_close(&got_vec(got_full), &want_full, 3e-2);

        let want_windowed = naive_attention_bhsd(
            &q_f32,
            &k_f32,
            &v_f32,
            scale,
            false,
            kv_offset,
            Some((4, 0)),
        );
        let got_windowed = try_windowed(&q, &k, &v, scale, 4, 0)
            .expect("GPU device must return Some")
            .unwrap();
        assert_close(&got_vec(got_windowed), &want_windowed, 3e-2);
    }

    // On the fused-kernel (MMA/tile) success path, `try_causal_with_mask` now
    // threads `mask` straight into the kernel (fattn_mma_causal_with_mask/
    // fattn_tile_causal_with_mask), so a wrong-offset mask must change the
    // output relative to the correct causal mask — proving the mask is
    // actually read, not silently ignored the way the old (superseded)
    // Crane kernels were. Mirrors `fattn_mma.rs`'s `garbage_mask_changes_output`
    // at this module's dispatch layer. `q`/`k`/`v`/`scale` and the kernel's
    // own `kv_offset` argument (computed from `q`/`k`'s shapes) are identical
    // between both calls — only the mask tensor's content differs — so this
    // isolates the mask as the one varying input.
    #[test]
    fn try_causal_with_mask_garbage_mask_changes_output() {
        let rocm = test_gpu_device();
        let (b, hq, hkv, sq, skv, d) = (1, 4, 2, 20, 150, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;

        let q_f32 = Tensor::randn(0f32, 1f32, (b, hq, sq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();

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

        let correct_mask =
            crate::models::utils::CausalMask::new(sq, skv, kv_offset, DType::F16, &rocm).unwrap();
        // A causal mask built with a deliberately wrong `kv_offset` (0
        // instead of the correct `skv - sq`): still a valid `CausalMask`
        // (shifted-diagonal pattern), just for the wrong diagonal, so it
        // masks out most of the keys the correct mask allows.
        let wrong_mask =
            crate::models::utils::CausalMask::new(sq, skv, 0, DType::F16, &rocm).unwrap();

        let out_correct = try_causal_with_mask(&q, &k, &v, scale, &correct_mask)
            .expect("GPU device must return Some")
            .unwrap();
        let out_wrong = try_causal_with_mask(&q, &k, &v, scale, &wrong_mask)
            .expect("GPU device must return Some")
            .unwrap();
        let max_diff = got_vec(out_correct)
            .iter()
            .zip(&got_vec(out_wrong))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "wrong-offset mask produced the same output as the correct mask (max diff {max_diff})"
        );
    }

    // A correct, caller-built mask on the fused-kernel path must match
    // `try_causal`'s own (internally-built) result, and the naive reference
    // — proving `try_causal_with_mask` isn't just "mask changes something,"
    // but specifically computes the same correct causal attention.
    #[test]
    fn try_causal_with_mask_correct_mask_matches_try_causal() {
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

        let got = try_causal_with_mask(&q, &k, &v, scale, &mask)
            .expect("GPU device must return Some")
            .unwrap();
        assert_close(&got_vec(got), &want, 3e-2);
    }

    // Deliberately unaligned seq_kv (not a FATTN_KQ_STRIDE=256 multiple) at
    // GQA ratio 8, matching the shape that regressed on real hardware
    // (Qwen3-Coder-30B-A3B/Qwen3.5-4B are both GQA ratio 8, head_dim 128).
    // Must still match the naive reference, proving the K/V padding this
    // unlocks is bit-correct, not just fast.
    #[test]
    fn try_causal_with_mask_unaligned_seq_kv_gqa_ratio_8() {
        let rocm = test_gpu_device();
        let (b, hq, hkv, sq, skv, d) = (1, 8, 1, 37, 1000, 128usize);
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

        let got = try_causal_with_mask(&q, &k, &v, scale, &mask)
            .expect("GPU device must return Some")
            .unwrap();
        assert_close(&got_vec(got), &want, 3e-2);
    }

    // On the matmul-SDPA fallback (unsupported head_dim), `try_causal_with_mask`
    // must actually apply the caller-provided mask rather than ignoring it,
    // matching a naive reference built with the same causal mask.
    #[test]
    fn try_causal_with_mask_unsupported_head_dim_respects_mask() {
        let rocm = test_gpu_device();
        let (b, hq, hkv, sq, skv, d) = (1, 4, 2, 5, 11, 100usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;

        let q_f32 = Tensor::randn(0f32, 1f32, (b, hq, sq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, hkv, skv, d), &Device::Cpu).unwrap();

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

        let want = naive_attention_bhsd(&q_f32, &k_f32, &v_f32, scale, true, kv_offset, None);
        let got = try_causal_with_mask(&q, &k, &v, scale, &mask)
            .expect("GPU device must return Some even for unsupported head_dim")
            .unwrap();
        assert_close(&got_vec(got), &want, 3e-2);
    }

    // Kernel-launch timing comparison between the native-bf16 and
    // cast-to-F16-fallback dispatch paths, at the real chunked-prefill
    // `seq_kv` lengths an 18387-token Qwen3-4B-Instruct-2507 prompt hits
    // (`PREFILL_CHUNK_SIZE=2048`, 9 chunks). Real callers (e.g.
    // `models::qwen3::modeling`'s `Attention::forward`) build Q/K/V in
    // BSHD, transpose to BHSD for the KV cache, and call
    // `try_causal_with_mask` — `run()` then transposes back to BSHD,
    // producing a *non-contiguous* view that `.contiguous()`/`.to_dtype()`
    // must materialize on every call. This test reproduces that exact
    // construction (BSHD alloc -> transpose -> BHSD, matching real
    // provenance) and goes through the real `try_causal_with_mask` entry
    // point (real F32 `CausalMask`, real native-bf16-vs-disabled dispatch
    // via `bf16_disabled()` reading `CRANE_DISABLE_BF16_FLASH_ATTN`),
    // instead of calling the `fattn_mma` kernel wrappers directly. Not a
    // correctness check and not run by default. `bf16_disabled()` caches
    // its env var read in a
    // `OnceLock` for the life of the process, and this worktree's `cargo`
    // proxy does not forward shell-set env vars to the host process it
    // execs, so the fallback variant below sets `CRANE_DISABLE_BF16_FLASH_ATTN`
    // via `std::env::set_var` (operates on the process's own env table
    // directly, unaffected by the proxy) as its very first line, before
    // `bf16_disabled()`'s first read. Run each test name on its own so the
    // two never share a process and the `OnceLock` can't already be
    // initialized by the other variant:
    // `cargo test -p crane-core --no-default-features --features "rocm,onnx"
    // --release -- --ignored --exact
    // models::modules::gpu_flash_attn::gpu_tests::bf16_vs_f16_try_causal_with_mask_timing_sweep_native
    // --nocapture` and the `_fallback` counterpart.
    fn bf16_vs_f16_try_causal_with_mask_timing_sweep_impl(label: &str) {
        let gpu = test_gpu_device();
        let (hq, hkv, d) = (32, 8, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();

        for (sq, skv) in [
            (2048usize, 2048usize),
            (2048, 4096),
            (2048, 6144),
            (2048, 8192),
            (2048, 10240),
            (2048, 12288),
            (2048, 14336),
            (2048, 16384),
            (2003, 18387),
        ] {
            let kv_offset = skv - sq;

            // Build Q/K/V in BSHD (as `qkv_proj.forward` + reshape does in
            // real model code), cast, then transpose to BHSD — the same
            // non-contiguous construction `Attention::forward` feeds into
            // `attn_dispatch::causal`, unlike handing `try_causal_with_mask`
            // an already-contiguous BHSD tensor directly.
            let q_bshd = Tensor::randn(0f32, 1f32, (1, sq, hq, d), &gpu).unwrap();
            let k_bshd = Tensor::randn(0f32, 1f32, (1, skv, hkv, d), &gpu).unwrap();
            let v_bshd = Tensor::randn(0f32, 1f32, (1, skv, hkv, d), &gpu).unwrap();

            let q_bf16 = q_bshd
                .to_dtype(DType::BF16)
                .unwrap()
                .transpose(1, 2)
                .unwrap();
            let k_bf16 = k_bshd
                .to_dtype(DType::BF16)
                .unwrap()
                .transpose(1, 2)
                .unwrap();
            let v_bf16 = v_bshd
                .to_dtype(DType::BF16)
                .unwrap()
                .transpose(1, 2)
                .unwrap();

            let mask = crate::models::utils::CausalMask::new(sq, skv, kv_offset, DType::F32, &gpu)
                .unwrap();

            let time = |label: &str, f: &dyn Fn() -> Tensor| -> f64 {
                for _ in 0..5 {
                    let _ = f();
                }
                gpu.synchronize().unwrap();

                let iters = 15;
                let mut samples = Vec::with_capacity(iters);
                for _ in 0..iters {
                    let t = std::time::Instant::now();
                    let out = f();
                    gpu.synchronize().unwrap();
                    samples.push(t.elapsed().as_secs_f64() * 1e6);
                    drop(out);
                }
                let mean = samples.iter().sum::<f64>() / iters as f64;
                let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / iters as f64;
                println!("{label:<6} mean={mean:9.1}us sd={:.1}us", var.sqrt());
                mean
            };

            println!("--- sq={sq} skv={skv} [{label}] ---");
            let _ = time(label, &|| {
                try_causal_with_mask(&q_bf16, &k_bf16, &v_bf16, scale, &mask)
                    .expect("GPU device must return Some")
                    .unwrap()
            });
        }
    }

    #[test]
    #[ignore = "manual GPU timing comparison, not a correctness check"]
    fn bf16_vs_f16_try_causal_with_mask_timing_sweep_native() {
        bf16_vs_f16_try_causal_with_mask_timing_sweep_impl("native");
    }

    #[test]
    #[ignore = "manual GPU timing comparison, not a correctness check"]
    fn bf16_vs_f16_try_causal_with_mask_timing_sweep_fallback() {
        // SAFETY: run with `--exact`, alone in its own process, before any
        // other code reads `CRANE_DISABLE_BF16_FLASH_ATTN`.
        unsafe {
            std::env::set_var("CRANE_DISABLE_BF16_FLASH_ATTN", "1");
        }
        bf16_vs_f16_try_causal_with_mask_timing_sweep_impl("fallback");
    }

    // Same comparison as `bf16_vs_f16_try_causal_with_mask_timing_sweep_impl`,
    // but grows K/V through the real `kv_cache::update_kv_cache` buffer
    // (`ROOM`-padded, reallocating) instead of a freshly-allocated,
    // tightly-sized tensor. The sweep above's K/V happen to land back on a
    // dense buffer by the time `run()`'s `.contiguous()`/`.to_dtype()` runs
    // on them (its `to_dtype(BF16)` always produces a dense result, and the
    // single `transpose(1, 2)` on top is undone by `run()`'s own transpose),
    // so those calls are free no-ops there. In production,
    // `models::qwen3::modeling::Attention::update_kv_cache` hands
    // `try_causal_with_mask` a `narrow()` view into a larger cache buffer
    // instead, making those same calls real `O(seq_kv)` copies — this test
    // reproduces that by growing the cache across the same 9 chunks a real
    // 18387-token Qwen3-4B-Instruct-2507 prompt hits
    // (`PREFILL_CHUNK_SIZE=2048`). Same single-process-per-variant run
    // instructions as the sweep above apply here.
    fn bf16_vs_f16_real_kv_cache_timing_sweep_impl(label: &str) {
        let gpu = test_gpu_device();
        let (hq, hkv, d) = (32, 8, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();

        let mut cache: Option<(Tensor, Tensor)> = None;
        let mut cache_seq_len = 0usize;

        for chunk_len in [2048usize, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2003] {
            let kv_offset = cache_seq_len;

            // Same BSHD-alloc -> transpose -> BHSD provenance as the sweep
            // above, for Q (never cached) and the new K/V chunk about to be
            // appended to the cache.
            let q_bshd = Tensor::randn(0f32, 1f32, (1, chunk_len, hq, d), &gpu).unwrap();
            let k_bshd = Tensor::randn(0f32, 1f32, (1, chunk_len, hkv, d), &gpu).unwrap();
            let v_bshd = Tensor::randn(0f32, 1f32, (1, chunk_len, hkv, d), &gpu).unwrap();

            let q_bf16 = q_bshd
                .to_dtype(DType::BF16)
                .unwrap()
                .transpose(1, 2)
                .unwrap();
            let k_bf16 = k_bshd
                .to_dtype(DType::BF16)
                .unwrap()
                .transpose(1, 2)
                .unwrap();
            let v_bf16 = v_bshd
                .to_dtype(DType::BF16)
                .unwrap()
                .transpose(1, 2)
                .unwrap();

            let crate::models::modules::kv_cache::KvCacheUpdate {
                buffer,
                seq_len,
                k: k_view,
                v: v_view,
            } = crate::models::modules::kv_cache::update_kv_cache(
                cache.take(),
                cache_seq_len,
                &k_bf16,
                &v_bf16,
            )
            .expect("update_kv_cache");
            cache = Some(buffer);
            cache_seq_len = seq_len;
            let skv = cache_seq_len;

            let mask =
                crate::models::utils::CausalMask::new(chunk_len, skv, kv_offset, DType::F32, &gpu)
                    .unwrap();

            let time = |label: &str, f: &dyn Fn() -> Tensor| -> f64 {
                for _ in 0..5 {
                    let _ = f();
                }
                gpu.synchronize().unwrap();

                let iters = 15;
                let mut samples = Vec::with_capacity(iters);
                for _ in 0..iters {
                    let t = std::time::Instant::now();
                    let out = f();
                    gpu.synchronize().unwrap();
                    samples.push(t.elapsed().as_secs_f64() * 1e6);
                    drop(out);
                }
                let mean = samples.iter().sum::<f64>() / iters as f64;
                let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / iters as f64;
                println!("{label:<6} mean={mean:9.1}us sd={:.1}us", var.sqrt());
                mean
            };

            println!("--- sq={chunk_len} skv={skv} [{label}] ---");
            let _ = time(label, &|| {
                try_causal_with_mask(&q_bf16, &k_view, &v_view, scale, &mask)
                    .expect("GPU device must return Some")
                    .unwrap()
            });
        }
    }

    #[test]
    #[ignore = "manual GPU timing comparison, not a correctness check"]
    fn bf16_vs_f16_real_kv_cache_timing_sweep_native() {
        bf16_vs_f16_real_kv_cache_timing_sweep_impl("native");
    }

    #[test]
    #[ignore = "manual GPU timing comparison, not a correctness check"]
    fn bf16_vs_f16_real_kv_cache_timing_sweep_fallback() {
        // SAFETY: run with `--exact`, alone in its own process, before any
        // other code reads `CRANE_DISABLE_BF16_FLASH_ATTN`.
        unsafe {
            std::env::set_var("CRANE_DISABLE_BF16_FLASH_ATTN", "1");
        }
        bf16_vs_f16_real_kv_cache_timing_sweep_impl("fallback");
    }

    // A different hypothesis from the two sweeps above: both of those time
    // 15 back-to-back calls against the *same* K/V buffer, so the GPU's L2
    // cache stays warm with that one buffer's contents across every sample
    // — a repetition a real forward pass never gets, since each of the
    // model's ~36 decoder layers touches its own distinct KV-cache memory
    // exactly once, cold, while the GPU is also busy running that layer's
    // (and neighboring layers') much larger MLP matmuls in between. This
    // allocates a brand-new, never-touched-before, `ROOM`-padded K/V buffer
    // for every single timed sample (kept alive in `keep_alive` so the
    // allocator can't hand the "next" sample the same physical address and
    // accidentally restore the warm-cache effect this test exists to
    // remove), to check whether losing that warm-cache advantage flips the
    // native-vs-fallback ranking. Builds the padded buffer directly
    // (`(1, hkv, skv+ROOM, dim)` zeros, `slice_set` at offset 0, `narrow`
    // back to `skv`) instead of replaying every prior chunk through
    // `kv_cache::update_kv_cache`, since only the buffer's final
    // non-contiguous shape matters here, not how it was grown. Not a
    // correctness check and not run by default; same
    // `CRANE_DISABLE_BF16_FLASH_ATTN` single-process-per-variant rule as the
    // sweeps above.
    fn bf16_vs_f16_cold_kv_cache_timing_sweep_impl(label: &str) {
        let gpu = test_gpu_device();
        let (hq, hkv, d) = (32, 8, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        const ROOM: usize = 256;

        for (sq, skv) in [
            (2048usize, 2048usize),
            (2048, 4096),
            (2048, 6144),
            (2048, 8192),
            (2048, 10240),
            (2048, 12288),
            (2048, 14336),
            (2048, 16384),
            (2003, 18387),
        ] {
            let kv_offset = skv - sq;
            let mask = crate::models::utils::CausalMask::new(sq, skv, kv_offset, DType::F32, &gpu)
                .unwrap();

            let mut keep_alive: Vec<Tensor> = Vec::new();
            let iters = 15;
            let mut samples = Vec::with_capacity(iters);

            for _ in 0..iters {
                let q_bshd = Tensor::randn(0f32, 1f32, (1, sq, hq, d), &gpu).unwrap();
                let k_bshd = Tensor::randn(0f32, 1f32, (1, skv, hkv, d), &gpu).unwrap();
                let v_bshd = Tensor::randn(0f32, 1f32, (1, skv, hkv, d), &gpu).unwrap();

                let q_bf16 = q_bshd
                    .to_dtype(DType::BF16)
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap();
                let k_bf16 = k_bshd
                    .to_dtype(DType::BF16)
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                let v_bf16 = v_bshd
                    .to_dtype(DType::BF16)
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap()
                    .contiguous()
                    .unwrap();

                let buf_k = Tensor::zeros((1, hkv, skv + ROOM, d), DType::BF16, &gpu).unwrap();
                let buf_v = Tensor::zeros((1, hkv, skv + ROOM, d), DType::BF16, &gpu).unwrap();
                buf_k.slice_set(&k_bf16, 2, 0).unwrap();
                buf_v.slice_set(&v_bf16, 2, 0).unwrap();
                let k_view = buf_k.narrow(2, 0, skv).unwrap();
                let v_view = buf_v.narrow(2, 0, skv).unwrap();

                gpu.synchronize().unwrap();
                let t = std::time::Instant::now();
                let out = try_causal_with_mask(&q_bf16, &k_view, &v_view, scale, &mask)
                    .expect("GPU device must return Some")
                    .unwrap();
                gpu.synchronize().unwrap();
                samples.push(t.elapsed().as_secs_f64() * 1e6);

                keep_alive.push(q_bf16);
                keep_alive.push(k_bf16);
                keep_alive.push(v_bf16);
                keep_alive.push(buf_k);
                keep_alive.push(buf_v);
                keep_alive.push(out);
            }

            let mean = samples.iter().sum::<f64>() / iters as f64;
            let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / iters as f64;
            println!("--- sq={sq} skv={skv} [{label}] cold ---");
            println!("{label:<6} mean={mean:9.1}us sd={:.1}us", var.sqrt());
            drop(keep_alive);
        }
    }

    #[test]
    #[ignore = "manual GPU timing comparison, not a correctness check"]
    fn bf16_vs_f16_cold_kv_cache_timing_sweep_native() {
        bf16_vs_f16_cold_kv_cache_timing_sweep_impl("native");
    }

    #[test]
    #[ignore = "manual GPU timing comparison, not a correctness check"]
    fn bf16_vs_f16_cold_kv_cache_timing_sweep_fallback() {
        // SAFETY: run with `--exact`, alone in its own process, before any
        // other code reads `CRANE_DISABLE_BF16_FLASH_ATTN`.
        unsafe {
            std::env::set_var("CRANE_DISABLE_BF16_FLASH_ATTN", "1");
        }
        bf16_vs_f16_cold_kv_cache_timing_sweep_impl("fallback");
    }
}
