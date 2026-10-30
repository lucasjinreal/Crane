// SPDX-License-Identifier: MIT
//! Opt-in dispatch to the vendored flash-attention prefill kernels
//! (`ops/fused_ops/fattn_mma.rs` tensor-core, `ops/fused_ops/fattn_tile.rs`
//! scalar fallback).
//!
//! The `try_*` functions return `None` whenever they cannot serve the call.
//! The caller then runs its own attention code unchanged. There is no
//! matmul fallback in here. `Some(Err(..))` is reserved for genuine kernel
//! failures.
//!
//! ```ignore
//! let out = match gpu_flash_attn::try_causal(&q, &k, &v, scale) {
//!     Some(out) => out?,
//!     None => sdpa(&q, &k, &v, mask, scale)?,
//! };
//! ```
//!
//! The kernels are compiled under `rocm`, and under `cuda` only with
//! `flash-attn-intree`. Elsewhere every `try_*` returns `None`. BF16 inputs
//! use the native-bf16 MMA kernel and F16/F32 inputs the F16 one. A shape
//! that kernel cannot serve returns `None`.
//!
//! `CRANE_FLASH_ATTN_INTREE=0` turns the kernels off at run time.
//! `CRANE_DISABLE_MMA_FLASH_ATTN=1` forces the tile kernel and
//! `CRANE_DISABLE_BF16_FLASH_ATTN=1` forces the BF16 to F16 cast path. Both
//! are debug switches, not a production fallback.
//!
//! **Known gap:** the tile kernel produces wrong output at `head_dim=64` on
//! real `ROCm` hardware (see `ops::fused_ops::fattn_tile`'s `#[ignore]`d
//! `causal_hd64`/`full_no_mask_hd64` tests). A `head_dim=64` call that
//! cannot reach the MMA kernel returns `None` instead of using it.

use candle_core::{Result, Tensor};

#[cfg(crane_fattn)]
use crate::ops::fused_ops::fattn_mma::{
    fattn_mma_bf16_causal, fattn_mma_bf16_causal_with_mask, fattn_mma_causal,
    fattn_mma_causal_with_mask,
};
#[cfg(crane_fattn)]
use crate::ops::fused_ops::fattn_tile::{fattn_tile_causal, fattn_tile_causal_with_mask};
// Only builds without ROCm need the trait. The ROCm fork's own `Device` has
// an inherent `is_rocm()`.
#[cfg(all(crane_fattn, feature = "cuda", not(feature = "rocm")))]
use crate::utils::DeviceExt;
#[cfg(crane_fattn)]
use candle_core::DType;
#[cfg(crane_fattn)]
use std::sync::Once;

///// Entry points ////////////////////////////////////////////////////////

/// Tries causal flash attention: `j <= i + kv_offset`, where `kv_offset` is
/// `k`'s sequence length minus `q`'s. That is the shifted diagonal that
/// continuation prefill against an append-only KV cache needs.
///
/// `q` is `[B, H_q, S, D]` and `k`/`v` are `[B, H_kv, kv_len, D]`. `H_q` must
/// be a multiple of `H_kv`, which the kernels handle natively.
///
/// The kernel derives `KV_max` analytically from `kv_offset` and skips KV
/// tiles past the causal diagonal. A caller must therefore only use this
/// for plain, unpadded causal attention. F32 inputs run through the kernel
/// in F16, so the result has F16 precision.
///
/// Returns `None` unless the build has the kernels (`rocm`, or `cuda` with
/// `flash-attn-intree`) and `q` is on that device. It also returns `None` for an
/// unsupported `head_dim` or a shape without a usable kernel.
///
/// # Errors
///
/// The inner `Result` errors on a kernel launch failure or on tensor shapes
/// the kernel rejects.
#[must_use]
pub(crate) fn try_causal(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Option<Result<Tensor>> {
    dispatch_causal(q, k, v, scale, None)
}

/// Same as [`try_causal`], for callers that already hold the plain causal
/// mask. The kernel reads `mask` instead of building one per call.
///
/// `mask` may have any float dtype and must be broadcastable to
/// `[.., seq_q, seq_kv]`. The kernel casts it to F16 on the device. It must
/// hide every key past the causal diagonal `j <= i + kv_offset`, because the
/// analytic `KV_max` cuts those off whatever the mask says. Below the
/// diagonal the mask is read in full, so it may hide more, such as the keys
/// outside a sliding window. A mask that unmasks keys past the diagonal is
/// not honored.
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
    mask: &Tensor,
) -> Option<Result<Tensor>> {
    dispatch_causal(q, k, v, scale, Some(mask))
}

/// [`try_causal_with_mask`], or [`try_causal`] when `mask` is `None`, with
/// the result laid out as `[B, S, H*D]` for a decoder's `o_proj`. This is
/// the form every opted-in model wants, so it lives here once.
///
/// # Errors
///
/// See [`try_causal`].
pub(crate) fn try_causal_prefill(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> Result<Option<Tensor>> {
    let out = match mask {
        Some(mask) => try_causal_with_mask(q, k, v, scale, mask),
        None => try_causal(q, k, v, scale),
    };
    out.transpose()?
        .map(|o| {
            let (batch, heads, seq, dim) = o.dims4()?;
            o.transpose(1, 2)?.reshape((batch, seq, heads * dim))
        })
        .transpose()
}

/// Casts the plain causal mask a decoder stack shares across its layers to
/// the F16 the kernels read, so the cast runs once per forward pass instead
/// of once per layer. Pass the result to [`try_causal_with_mask`] in place
/// of the original.
///
/// Returns `None` when there is nothing to gain: `mask` is already F16 or
/// no kernel serves its device. Callers then keep passing the original.
///
/// # Errors
///
/// Returns an error if the device cast fails.
// The cast is fallible on GPU builds. CPU-only builds never reach it.
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn prepare_mask(mask: &Tensor) -> Result<Option<Tensor>> {
    #[cfg(crane_fattn)]
    if on_kernel_device(mask) && mask.dtype() != DType::F16 {
        return mask.to_dtype(DType::F16).map(Some);
    }
    let _ = mask;
    Ok(None)
}

/// Logs a warning, once per process, when `model` opts into flash attention
/// but this build has no flash backend for it, so every call runs the
/// default attention path. Called from each model's
/// `DecoderLayer::enable_flash_causal()`, so downstream crates that call it
/// directly get the warning too.
///
/// `has_fa2` says whether `model` can use the `candle-flash-attn` backend in
/// this build (only Qwen3.5 has an FA2 path).
pub(crate) fn warn_if_no_flash_backend(model: &str, has_fa2: bool) {
    static WARNED: std::sync::Once = std::sync::Once::new();

    if cfg!(crane_fattn) || has_fa2 {
        return;
    }
    WARNED.call_once(|| {
        ribo::utils::log::warn!(
            "{model}: flash attention requested, but this build has no flash-attention backend for it, using the default attention. On CUDA, build with `flash-attn-intree`."
        );
    });
}

/// Shared body of [`try_causal`] and [`try_causal_with_mask`].
#[cfg(crane_fattn)]
fn dispatch_causal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> Option<Result<Tensor>> {
    if intree_env() == Some(false) || !on_kernel_device(q) || !head_dim_supported(q) {
        return None;
    }
    let offset = match kv_offset(q, k) {
        Ok(offset) => offset,
        Err(e) => return Some(Err(e)),
    };
    // The device pointer helpers need a fully contiguous mask. This is an
    // `Arc` clone when it already is, which is the usual case.
    let mask = match mask.map(Tensor::contiguous).transpose() {
        Ok(mask) => mask,
        Err(e) => return Some(Err(e)),
    };
    let mask = mask.as_ref();
    run(
        q,
        k,
        v,
        |qf, kf, vf| match mask {
            Some(m) => fattn_mma_bf16_causal_with_mask(qf, kf, vf, scale, offset, m),
            None => fattn_mma_bf16_causal(qf, kf, vf, scale, offset),
        },
        |qf, kf, vf| match mask {
            Some(m) => fattn_mma_causal_with_mask(qf, kf, vf, scale, offset, m),
            None => fattn_mma_causal(qf, kf, vf, scale, offset),
        },
        |qf, kf, vf| match mask {
            Some(m) => fattn_tile_causal_with_mask(qf, kf, vf, scale, offset, m),
            None => fattn_tile_causal(qf, kf, vf, scale, offset),
        },
    )
    .transpose()
}

/// Builds without a GPU kernel backend never serve a call.
#[cfg(not(crane_fattn))]
fn dispatch_causal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> Option<Result<Tensor>> {
    let _ = (q, k, v, scale, mask);
    None
}

///// Dispatch chain //////////////////////////////////////////////////////

/// Ensures the dtype-cast note in [`run`] logs once per process rather than
/// once per forward pass.
#[cfg(crane_fattn)]
static CAST_WARNED: Once = Once::new();

/// Ensures the unsupported-shape note in [`unsupported_to_none`] logs once
/// per process.
#[cfg(crane_fattn)]
static UNSUPPORTED_LOGGED: Once = Once::new();

/// Ensures the `head_dim=64` tile-kernel note in [`run`] logs once per
/// process.
#[cfg(crane_fattn)]
static TILE_HD64_SKIPPED: Once = Once::new();

/// Runs one kernel over BHSD inputs and returns a BHSD result, or `None`
/// when that kernel has no instance for the shape.
///
/// BF16 inputs go to the native-bf16 MMA kernel with `q`/`k`/`v` passed
/// through unchanged (or to the F16 kernel below when
/// `CRANE_DISABLE_BF16_FLASH_ATTN` is set). F16 and F32 inputs go to the F16
/// kernel: F32 is narrowed to F16, and the F16 result is cast back to the
/// caller's dtype. The tile kernel instead takes `q` in F32. A shape the
/// chosen kernel does not cover returns `None`, so the caller's SDPA serves
/// it. There is no fall-through to another kernel, because the cast path and
/// the tile kernel are slower or less trusted than SDPA. Only the debug
/// variables in this module's docs select them.
///
/// A kernel failure tagged as unsupported-config becomes `Ok(None)`. Any
/// other failure propagates. A kernel that launches but computes wrong
/// values returns `Ok` and is not caught, except for the known
/// `head_dim=64` tile bug, which returns `None` instead.
#[cfg(crane_fattn)]
fn run(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mma_bf16: impl FnOnce(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    mma: impl FnOnce(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    tile: impl FnOnce(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
) -> Result<Option<Tensor>> {
    use ribo::utils::log;

    let orig_dtype = q.dtype();
    let q_bshd = q.transpose(1, 2)?;
    let k_bshd = k.transpose(1, 2)?;
    let v_bshd = v.transpose(1, 2)?;

    if orig_dtype == DType::BF16 && !bf16_disabled() {
        // The kernels read the strided BSHD views directly.
        let q_bf16 = addressable(q_bshd, true)?;
        let k_bf16 = addressable(k_bshd, false)?;
        let v_bf16 = addressable(v_bshd, false)?;
        return match mma_bf16(&q_bf16, &k_bf16, &v_bf16) {
            Ok(out_bshd) => out_bshd.transpose(1, 2).map(Some),
            Err(e) => unsupported_to_none(e),
        };
    }

    // Inputs already in the kernel's dtype go in as strided views. Only the
    // others pay a cast.
    let (k_f16, v_f16) = if orig_dtype == DType::F16 {
        (addressable(k_bshd, false)?, addressable(v_bshd, false)?)
    } else {
        log_cast_once(orig_dtype);
        (k_bshd.to_dtype(DType::F16)?, v_bshd.to_dtype(DType::F16)?)
    };

    // Both F16 kernels always return F16 (accumulation is F32, cast down once
    // at the end), so only an F16 caller skips the cast back.
    let finish = |out_bshd: Tensor| -> Result<Option<Tensor>> {
        let out_bhsd = out_bshd.transpose(1, 2)?;
        if orig_dtype == DType::F16 {
            Ok(Some(out_bhsd))
        } else {
            out_bhsd.to_dtype(orig_dtype).map(Some)
        }
    };

    if !mma_disabled() {
        // The MMA kernels have no F32 Q path, so F32 callers pay a Q cast
        // here as they do for K/V. Models hold Q in BF16 or F16.
        let q_f16 = if orig_dtype == DType::F16 {
            addressable(q_bshd, true)?
        } else {
            q_bshd.to_dtype(DType::F16)?
        };
        return match mma(&q_f16, &k_f16, &v_f16) {
            Ok(out_bshd) => finish(out_bshd),
            Err(e) => unsupported_to_none(e),
        };
    }

    // Only `CRANE_DISABLE_MMA_FLASH_ATTN` gets here.
    if matches!(q_bshd.dim(3), Ok(64)) {
        TILE_HD64_SKIPPED.call_once(|| {
            log::debug!(
                "gpu_flash_attn: the head_dim=64 tile kernel is known to produce wrong output on ROCm, leaving the call to the caller (logged once)"
            );
        });
        return Ok(None);
    }

    let q_f32 = if orig_dtype == DType::F32 {
        addressable(q_bshd, true)?
    } else {
        log_cast_once(orig_dtype);
        q_bshd.to_dtype(DType::F32)?
    };
    match tile(&q_f32, &k_f16, &v_f16) {
        Ok(out_bshd) => finish(out_bshd),
        Err(e) => unsupported_to_none(e),
    }
}

/// Maps a kernel error tagged as unsupported-config to `Ok(None)`, leaving
/// the shape to the caller's SDPA. Any other error propagates, so real
/// kernel faults are not hidden.
#[cfg(crane_fattn)]
fn unsupported_to_none(err: candle_core::Error) -> Result<Option<Tensor>> {
    use crate::ops::fused_ops::fattn::is_unsupported_config;

    if !is_unsupported_config(&err) {
        return Err(err);
    }
    UNSUPPORTED_LOGGED.call_once(|| {
        ribo::utils::log::debug!(
            "gpu_flash_attn: no kernel instance for this shape, leaving the call to the caller (logged once): {err}"
        );
    });
    Ok(None)
}

/// Logs the dtype-cast note once per process.
#[cfg(crane_fattn)]
fn log_cast_once(orig_dtype: DType) {
    CAST_WARNED.call_once(|| {
        ribo::utils::log::debug!(
            "gpu_flash_attn: casting {orig_dtype:?} inputs for the vendored kernels (logged once)"
        );
    });
}

/// Returns `t` itself when the kernels can address its layout, so a view
/// reaches them uncopied. Any other layout gets a fresh contiguous copy.
/// A non-4D `t` passes through, so the kernel's own validation rejects it.
#[cfg(crane_fattn)]
fn addressable(t: Tensor, batch_in_i32: bool) -> Result<Tensor> {
    use crate::ops::fused_ops::fattn::check_kernel_layout;

    let elem_bytes = t.dtype().size_in_bytes();
    if t.rank() != 4 || check_kernel_layout(t.layout(), elem_bytes, batch_in_i32, "").is_ok() {
        return Ok(t);
    }
    copy_unaddressable(&t)
}

/// Copies a view whose offset or strides the kernels cannot address. Plain
/// `contiguous()` would keep a misaligned start offset.
#[cfg(crane_fattn)]
#[cold]
#[inline(never)]
fn copy_unaddressable(t: &Tensor) -> Result<Tensor> {
    t.force_contiguous()
}

///// Helpers /////////////////////////////////////////////////////////////

/// Whether `q` lives on the device this build has kernels for. This is
/// false on CPU, Metal and SYCL, and on a GPU whose backend feature is not
/// compiled in.
#[cfg(crane_fattn)]
fn on_kernel_device(q: &Tensor) -> bool {
    (cfg!(feature = "cuda") && q.device().is_cuda())
        || (cfg!(feature = "rocm") && q.device().is_rocm())
}

/// `k`'s sequence length minus `q`'s. `q`/`k` are BHSD, so the sequence axis
/// is 2.
#[cfg(crane_fattn)]
fn kv_offset(q: &Tensor, k: &Tensor) -> Result<usize> {
    Ok(k.dim(2)?.saturating_sub(q.dim(2)?))
}

/// Whether `q`'s `head_dim` is one the kernels instantiate (64, 128, 256 or
/// 512). On AMD, `head_dim > 256` is excluded. The MMA kernel never
/// launches there, so every call would land on the scalar tile kernel, which
/// measured far slower than a matmul SDPA (29904-token prefill at
/// `head_dim=512`, `gqa_ratio=8`: 904.7 tok/s down to 131.7 tok/s). CUDA has
/// no such restriction.
#[cfg(crane_fattn)]
fn head_dim_supported(q: &Tensor) -> bool {
    use crate::ops::fused_ops::fattn::is_amd_backend;

    match q.dim(3) {
        Ok(d @ (64 | 128 | 256 | 512)) => !(is_amd_backend() && d > 256),
        _ => false,
    }
}

/// `CRANE_FLASH_ATTN_INTREE`: `None` when unset, `Some(false)` for empty or
/// `0`, `Some(true)` otherwise. `Some(false)` turns the in-tree kernels off
/// (a kill switch). Under `flash-attn-ab`, `Some(true)` also moves Qwen3.5
/// off `flash_split` onto them. Read once per process.
#[cfg(crane_fattn)]
pub(crate) fn intree_env() -> Option<bool> {
    static VALUE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *VALUE
        .get_or_init(|| parse_intree_env(std::env::var("CRANE_FLASH_ATTN_INTREE").ok().as_deref()))
}

/// Parses a `CRANE_FLASH_ATTN_INTREE` value, see [`intree_env`].
#[cfg_attr(not(crane_fattn), allow(dead_code))]
fn parse_intree_env(value: Option<&str>) -> Option<bool> {
    value.map(|v| !matches!(v.trim(), "" | "0"))
}

/// Whether `CRANE_DISABLE_MMA_FLASH_ATTN` is set. A kernel that launches but
/// computes wrong values is not caught by the launch-error fallback, so an
/// operator on an unverified architecture needs a way to force the tile
/// kernel without rebuilding. Read once per process.
#[cfg(crane_fattn)]
fn mma_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("CRANE_DISABLE_MMA_FLASH_ATTN").is_ok_and(|v| !matches!(v.trim(), "" | "0"))
    })
}

/// Whether `CRANE_DISABLE_BF16_FLASH_ATTN` is set, forcing the BF16 to F16
/// cast path. Same reasoning as [`mma_disabled`]. Read once per process.
#[cfg(crane_fattn)]
fn bf16_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| {
        std::env::var("CRANE_DISABLE_BF16_FLASH_ATTN").is_ok_and(|v| !matches!(v.trim(), "" | "0"))
    })
}

///// Tests ///////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    use super::*;

    // On a CPU tensor, every entry point returns `None` whatever GPU feature
    // is compiled in, so callers can always fall through to their own path.
    #[test]
    fn cpu_device_always_returns_none() {
        let q = Tensor::zeros((1usize, 2, 4, 128), DType::F16, &Device::Cpu).unwrap();
        let mask = Tensor::zeros((1usize, 1, 4, 4), DType::F16, &Device::Cpu).unwrap();

        assert!(try_causal(&q, &q, &q, 0.088).is_none());
        assert!(try_causal_with_mask(&q, &q, &q, 0.088, &mask).is_none());
    }

    // `CRANE_FLASH_ATTN_INTREE` is tri-state: unset, off (empty or `0`) and
    // on (anything else).
    #[test]
    fn parse_intree_env_is_tri_state() {
        assert_eq!(parse_intree_env(None), None);
        assert_eq!(parse_intree_env(Some("")), Some(false));
        assert_eq!(parse_intree_env(Some(" 0 ")), Some(false));
        assert_eq!(parse_intree_env(Some("1")), Some(true));
        assert_eq!(parse_intree_env(Some("true")), Some(true));
    }
}

/// End-to-end coverage of the BHSD API on a real GPU, through the full
/// transpose, cast and transpose-back path. `ops::fused_ops::fattn_mma` and
/// `fattn_tile` cover the kernels themselves.
#[cfg(all(test, crane_fattn))]
mod gpu_tests {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::ops::softmax_last_dim;

    use crate::ops::fused_ops::fattn::test_support::{narrowed_bhsd, test_gpu_device};

    use super::{try_causal, try_causal_prefill, try_causal_with_mask};

    /// Random F32 reference tensors on the CPU plus their `dtype` copies on
    /// the GPU.
    struct Inputs {
        reference: [Tensor; 3],
        gpu: [Tensor; 3],
        scale: f32,
        kv_offset: usize,
    }

    /// Builds `[b, hq, sq, d]` q and `[b, hkv, skv, d]` k/v.
    fn inputs(
        (b, hq, hkv, sq, skv, d): (usize, usize, usize, usize, usize, usize),
        dtype: DType,
    ) -> Inputs {
        let dev = test_gpu_device();
        let randn = |h, s| Tensor::randn(0f32, 1f32, (b, h, s, d), &Device::Cpu).unwrap();
        let reference = [randn(hq, sq), randn(hkv, skv), randn(hkv, skv)];
        let gpu = reference
            .clone()
            .map(|t| t.to_dtype(dtype).unwrap().to_device(&dev).unwrap());
        Inputs {
            reference,
            gpu,
            scale: 1.0 / (d as f32).sqrt(),
            kv_offset: skv - sq,
        }
    }

    /// Additive `[1, 1, sq, skv]` causal mask: `0` where `j <= i + kv_offset`,
    /// `-inf` elsewhere.
    fn causal_mask(sq: usize, skv: usize, kv_offset: usize, dtype: DType, dev: &Device) -> Tensor {
        windowed_mask(sq, skv, kv_offset, None, dtype, dev)
    }

    /// [`causal_mask`] that also hides keys with `j + window <= i + kv_offset`.
    /// This is the sliding-window mask of `window` keys, the query included.
    fn windowed_mask(
        sq: usize,
        skv: usize,
        kv_offset: usize,
        window: Option<usize>,
        dtype: DType,
        dev: &Device,
    ) -> Tensor {
        let vals: Vec<f32> = (0..sq * skv)
            .map(|idx| {
                let (i, j) = (idx / skv, idx % skv);
                let visible = j <= i + kv_offset && window.is_none_or(|w| j + w > i + kv_offset);
                if visible { 0.0 } else { f32::NEG_INFINITY }
            })
            .collect();
        Tensor::from_vec(vals, (1, 1, sq, skv), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    /// `softmax(q @ k^T * scale + causal mask) @ v` in F32 on the CPU, BHSD.
    fn naive_causal(inp: &Inputs) -> Vec<f32> {
        naive_windowed(inp, None)
    }

    /// [`naive_causal`] that also hides keys more than `window - 1` behind
    /// the query, like Gemma4's sliding layers.
    // q/k/v/b/h/s/d are standard ML tensor-shape notation (query, key, value,
    // batch, heads, seq_len, head_dim).
    #[allow(clippy::many_single_char_names)]
    fn naive_windowed(inp: &Inputs, window: Option<usize>) -> Vec<f32> {
        let [q, k, v] = &inp.reference;
        let (b, hq, sq, d) = q.dims4().unwrap();
        let (_, hkv, skv, _) = k.dims4().unwrap();
        let n_rep = hq / hkv;
        let expand = |t: &Tensor| {
            t.unsqueeze(2)
                .unwrap()
                .expand((b, hkv, n_rep, skv, d))
                .unwrap()
                .reshape((b, hq, skv, d))
                .unwrap()
                .contiguous()
                .unwrap()
        };
        let scores = (q.matmul(&expand(k).transpose(2, 3).unwrap()).unwrap()
            * f64::from(inp.scale))
        .unwrap();
        let mask = windowed_mask(sq, skv, inp.kv_offset, window, DType::F32, &Device::Cpu);
        let probs = softmax_last_dim(&scores.broadcast_add(&mask).unwrap()).unwrap();
        probs
            .matmul(&expand(v))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    fn to_vec(t: &Tensor) -> Vec<f32> {
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

    /// Runs `try_causal` and checks it against the naive reference.
    fn check_try_causal(shape: (usize, usize, usize, usize, usize, usize), dtype: DType, tol: f32) {
        let inp = inputs(shape, dtype);
        let [q, k, v] = &inp.gpu;
        let got = try_causal(q, k, v, inp.scale)
            .expect("GPU device must serve a supported shape")
            .unwrap();
        assert_eq!(got.dtype(), dtype);
        assert_eq!(got.dims(), q.dims());
        assert_close(&to_vec(&got), &naive_causal(&inp), tol);
    }

    // F16 inputs through the tensor-core kernel, with GQA ratio 2 and a KV
    // cache longer than the query.
    #[test]
    fn try_causal_f16_matches_reference() {
        check_try_causal((1, 4, 2, 20, 150, 128), DType::F16, 3e-2);
    }

    // F32 inputs run through the kernel in F16, so the result has F16
    // precision but keeps the caller's dtype.
    #[test]
    fn try_causal_f32_matches_reference() {
        check_try_causal((1, 4, 2, 8, 8, 128), DType::F32, 6e-2);
    }

    // The common real-model path: BF16 inputs through the native kernel.
    #[test]
    fn try_causal_bf16_matches_reference() {
        check_try_causal((1, 4, 2, 8, 8, 128), DType::BF16, 3e-2);
    }

    // GQA ratio 8, the multi-warp kernel's primary target.
    #[test]
    fn try_causal_f16_gqa_ratio_8() {
        check_try_causal((1, 8, 1, 20, 150, 128), DType::F16, 3e-2);
    }

    /// Requires `try_causal` on `q`/`k`/`v` to give the same bits as on
    /// contiguous copies of them.
    fn assert_matches_contiguous_bitwise(q: &Tensor, k: &Tensor, v: &Tensor) {
        let call = |q: &Tensor, k: &Tensor, v: &Tensor| {
            let out = try_causal(q, k, v, 0.125)
                .expect("GPU device must serve a supported shape")
                .unwrap();
            to_vec(&out)
        };
        let got = call(q, k, v);
        let want = call(
            &q.contiguous().unwrap(),
            &k.contiguous().unwrap(),
            &v.contiguous().unwrap(),
        );
        assert_eq!(got.len(), want.len());
        assert!(
            got.iter()
                .zip(&want)
                .all(|(g, w)| g.to_bits() == w.to_bits()),
            "{:?}: outputs differ from contiguous inputs",
            q.dtype()
        );
    }

    // KV-cache-style narrowed BHSD views give the same bits as contiguous
    // copies, for every dtype.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn try_causal_strided_matches_contiguous() {
        let dev = test_gpu_device();
        let (b, hq, hkv, s, d) = (1, 8, 2, 256, 128);
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let q = narrowed_bhsd((b, hq, s, d), dtype, &dev);
            let k = narrowed_bhsd((b, hkv, s, d), dtype, &dev);
            let v = narrowed_bhsd((b, hkv, s, d), dtype, &dev);
            assert!(!k.transpose(1, 2).unwrap().is_contiguous());
            assert_matches_contiguous_bitwise(&q, &k, &v);
        }
    }

    // Q/K/V at an element offset the kernels cannot address are copied and
    // still served by the kernel, not left to the caller's SDPA.
    #[test]
    #[allow(clippy::many_single_char_names)]
    fn try_causal_misaligned_inputs_still_served() {
        let dev = test_gpu_device();
        let (b, hq, hkv, s, d) = (1, 8, 2, 256, 128);
        let misaligned = |heads: usize, dtype: DType| {
            let n = b * heads * s * d;
            Tensor::randn(0f32, 1f32, n + 1, &dev)
                .unwrap()
                .to_dtype(dtype)
                .unwrap()
                .narrow(0, 1, n)
                .unwrap()
                .reshape((b, heads, s, d))
                .unwrap()
        };
        // F32 inputs are cast to fresh F16 tensors, so only these two reach
        // the copy.
        for dtype in [DType::BF16, DType::F16] {
            let q = misaligned(hq, dtype);
            let (k, v) = (misaligned(hkv, dtype), misaligned(hkv, dtype));
            assert_eq!(q.layout().start_offset(), 1);
            assert_eq!(k.layout().start_offset(), 1);
            assert_matches_contiguous_bitwise(&q, &k, &v);
        }
    }

    // A plain causal mask built by the caller must give the same answer as
    // the analytic path, for every float dtype a model may build it in.
    #[test]
    fn try_causal_with_mask_matches_reference_for_any_mask_dtype() {
        let shape = (1, 4, 2, 20, 150, 128);
        let inp = inputs(shape, DType::F16);
        let [q, k, v] = &inp.gpu;
        let want = naive_causal(&inp);
        for mask_dtype in [DType::F16, DType::BF16, DType::F32] {
            let mask = causal_mask(20, 150, inp.kv_offset, mask_dtype, q.device());
            let got = try_causal_with_mask(q, k, v, inp.scale, &mask)
                .expect("GPU device must serve a supported shape")
                .unwrap();
            assert_close(&to_vec(&got), &want, 3e-2);
        }
    }

    // A wrong-diagonal mask must change the output. That proves the kernel
    // reads the mask instead of ignoring it.
    #[test]
    fn try_causal_with_mask_garbage_mask_changes_output() {
        let inp = inputs((1, 4, 2, 20, 150, 128), DType::F16);
        let [q, k, v] = &inp.gpu;
        let correct = causal_mask(20, 150, inp.kv_offset, DType::F16, q.device());
        let wrong = causal_mask(20, 150, 0, DType::F16, q.device());

        let run = |m: &Tensor| {
            to_vec(
                &try_causal_with_mask(q, k, v, inp.scale, m)
                    .expect("GPU device must serve a supported shape")
                    .unwrap(),
            )
        };
        let max_diff = run(&correct)
            .iter()
            .zip(&run(&wrong))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff > 1e-3, "wrong mask matched the correct one");
    }

    // A sliding-window mask hides keys below the diagonal. The kernel must
    // honor it exactly, since only the keys past the diagonal are trimmed
    // analytically. This is what Gemma4's sliding layers rely on, with a
    // window shorter than the prompt and a warm cache.
    #[test]
    fn try_causal_with_mask_honors_sliding_window() {
        let inp = inputs((1, 4, 2, 40, 120, 128), DType::F16);
        let [q, k, v] = &inp.gpu;
        let window = 8;
        let mask = windowed_mask(40, 120, inp.kv_offset, Some(window), DType::F16, q.device());
        let got = try_causal_with_mask(q, k, v, inp.scale, &mask)
            .expect("GPU device must serve a supported shape")
            .unwrap();
        let want = naive_windowed(&inp, Some(window));
        assert_close(&to_vec(&got), &want, 3e-2);

        // The window must matter here, or the check above proves nothing.
        let plain = naive_causal(&inp);
        let max_diff = plain
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff > 1e-2, "window did not change the reference");
    }

    // The prefill form is the same attention in `[B, S, H*D]` layout, with
    // or without a caller-built mask.
    #[test]
    fn try_causal_prefill_is_try_causal_in_flat_layout() {
        let inp = inputs((1, 4, 2, 20, 150, 128), DType::F16);
        let [q, k, v] = &inp.gpu;
        let mask = causal_mask(20, 150, inp.kv_offset, DType::F16, q.device());
        let want = try_causal(q, k, v, inp.scale).unwrap().unwrap();
        let want = to_vec(&want.transpose(1, 2).unwrap().contiguous().unwrap());
        for mask in [None, Some(&mask)] {
            let got = try_causal_prefill(q, k, v, inp.scale, mask)
                .unwrap()
                .expect("GPU device must serve a supported shape");
            assert_eq!(got.dims(), &[1, 20, 4 * 128]);
            assert_close(&to_vec(&got), &want, 3e-2);
        }
    }

    // An unaligned seq_kv at GQA ratio 8 exercises the K/V padding the
    // GQA-batched dispatch needs. It must stay bit-correct, not just fast.
    #[test]
    fn try_causal_with_mask_unaligned_seq_kv_gqa_ratio_8() {
        let inp = inputs((1, 8, 1, 37, 1000, 128), DType::F16);
        let [q, k, v] = &inp.gpu;
        let mask = causal_mask(37, 1000, inp.kv_offset, DType::F16, q.device());
        let got = try_causal_with_mask(q, k, v, inp.scale, &mask)
            .expect("GPU device must serve a supported shape")
            .unwrap();
        assert_close(&to_vec(&got), &naive_causal(&inp), 3e-2);
    }

    // An unsupported head_dim returns `None` so the caller keeps its own
    // path. It is neither an error nor a silent matmul fallback.
    #[test]
    fn unsupported_head_dim_returns_none() {
        let inp = inputs((1, 4, 2, 5, 11, 100), DType::F16);
        let [q, k, v] = &inp.gpu;
        let mask = causal_mask(5, 11, inp.kv_offset, DType::F16, q.device());
        assert!(try_causal(q, k, v, inp.scale).is_none());
        assert!(try_causal_with_mask(q, k, v, inp.scale, &mask).is_none());
    }

    // A shape the MMA kernel has no instance for returns `None` instead of
    // falling through to the tile kernel. Plain MHA is that shape on AMD.
    // NVIDIA serves it, so there it must match the reference.
    #[test]
    fn mha_follows_mma_support() {
        use crate::ops::fused_ops::fattn::{is_amd_backend, select_mma_ncols};

        let shape = (1, 2, 2, 8, 8, 128);
        for dtype in [DType::F16, DType::BF16] {
            if select_mma_ncols(128, 1, 8, 8, is_amd_backend()).is_some() {
                check_try_causal(shape, dtype, 3e-2);
            } else {
                let inp = inputs(shape, dtype);
                let [q, k, v] = &inp.gpu;
                assert!(try_causal(q, k, v, inp.scale).is_none());
            }
        }
    }

    // `head_dim=64` that cannot use MMA would land on the known-bad tile
    // kernel, so it must return `None`. `mma_disabled()` caches its env var
    // for the life of the process, so run this alone:
    // `cargo test -p crane-core --no-default-features --features
    // "rocm,onnx" -- --ignored --exact
    // models::modules::gpu_flash_attn::gpu_tests::head_dim_64_without_mma_returns_none`
    #[test]
    #[ignore = "sets a process-global env var cached by mma_disabled(), run alone"]
    fn head_dim_64_without_mma_returns_none() {
        // SAFETY: run with `--exact`, alone in its own process, before any
        // other code reads `CRANE_DISABLE_MMA_FLASH_ATTN`.
        unsafe {
            std::env::set_var("CRANE_DISABLE_MMA_FLASH_ATTN", "1");
        }
        let inp = inputs((1, 4, 2, 5, 11, 64), DType::F16);
        let [q, k, v] = &inp.gpu;
        assert!(try_causal(q, k, v, inp.scale).is_none());
    }

    // BF16 inputs on the cast path still match the reference, with Q cast
    // straight to F16 like K/V. `bf16_disabled()` caches its env var for
    // the life of the process, so run this alone:
    // `cargo test -p crane-core --no-default-features --features
    // "rocm,onnx" -- --ignored --exact
    // models::modules::gpu_flash_attn::gpu_tests::bf16_disabled_matches_reference`
    #[test]
    #[ignore = "sets a process-global env var cached by bf16_disabled(), run alone"]
    fn bf16_disabled_matches_reference() {
        // SAFETY: run with `--exact`, alone in its own process, before any
        // other code reads `CRANE_DISABLE_BF16_FLASH_ATTN`.
        unsafe {
            std::env::set_var("CRANE_DISABLE_BF16_FLASH_ATTN", "1");
        }
        check_try_causal((1, 4, 2, 8, 8, 128), DType::BF16, 3e-2);
    }

    // The tile kernel takes Q in F32. F32 Q reaches it uncast and F16 Q is
    // widened, and both still match the reference. `mma_disabled()` caches
    // its env var for the life of the process, so run this alone:
    // `cargo test -p crane-core --no-default-features --features
    // "rocm,onnx" -- --ignored --exact
    // models::modules::gpu_flash_attn::gpu_tests::mma_disabled_matches_reference`
    #[test]
    #[ignore = "sets a process-global env var cached by mma_disabled(), run alone"]
    fn mma_disabled_matches_reference() {
        // SAFETY: run with `--exact`, alone in its own process, before any
        // other code reads `CRANE_DISABLE_MMA_FLASH_ATTN`.
        unsafe {
            std::env::set_var("CRANE_DISABLE_MMA_FLASH_ATTN", "1");
        }
        check_try_causal((1, 4, 2, 20, 150, 128), DType::F16, 3e-2);
        check_try_causal((1, 4, 2, 8, 8, 128), DType::F32, 6e-2);
    }

    /// Times `try_causal_with_mask` at the chunked-prefill `seq_kv` lengths
    /// an 18387-token Qwen3-4B prompt hits with 2048-token chunks. Q/K/V are
    /// built in BSHD and transposed, like real model code does.
    // q/k/v/h/s/d are standard ML tensor-shape notation, see `naive_causal`.
    #[allow(clippy::many_single_char_names)]
    fn timing_sweep(label: &str, mask_dtype: DType) {
        let gpu = test_gpu_device();
        let (hq, hkv, d) = (32, 8, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();

        for (sq, skv) in [
            (2048usize, 2048usize),
            (2048, 4096),
            (2048, 8192),
            (2048, 12288),
            (2048, 16384),
            (2003, 18387),
        ] {
            let bhsd = |h: usize, s: usize| {
                Tensor::randn(0f32, 1f32, (1, s, h, d), &gpu)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap()
            };
            let (q, k, v) = (bhsd(hq, sq), bhsd(hkv, skv), bhsd(hkv, skv));
            let mask = causal_mask(sq, skv, skv - sq, mask_dtype, &gpu);

            let call = || {
                try_causal_with_mask(&q, &k, &v, scale, &mask)
                    .expect("GPU device must serve a supported shape")
                    .unwrap()
            };
            for _ in 0..5 {
                drop(call());
            }
            gpu.synchronize().unwrap();

            let iters = 15;
            let mut samples = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t = std::time::Instant::now();
                let out = call();
                gpu.synchronize().unwrap();
                samples.push(t.elapsed().as_secs_f64() * 1e6);
                drop(out);
            }
            let mean = samples.iter().sum::<f64>() / iters as f64;
            println!("sq={sq} skv={skv} [{label}] mean={mean:9.1}us");
        }
    }

    // Timing of the native-bf16 path against the cast-to-F16 path (the
    // `_fallback` variant). Not a correctness check. Each variant caches an
    // env var for the life of the process, so run each by name on its own:
    // `cargo test -p crane-core --no-default-features --features
    // "rocm,onnx" --release -- --ignored --exact
    // models::modules::gpu_flash_attn::gpu_tests::<name> --nocapture`
    #[test]
    #[ignore = "manual GPU timing comparison, not a correctness check"]
    fn bf16_vs_f16_try_causal_with_mask_timing_sweep_native() {
        timing_sweep("native", DType::BF16);
    }

    #[test]
    #[ignore = "manual GPU timing comparison, not a correctness check"]
    fn bf16_vs_f16_try_causal_with_mask_timing_sweep_fallback() {
        // SAFETY: run with `--exact`, alone in its own process, before any
        // other code reads `CRANE_DISABLE_BF16_FLASH_ATTN`.
        unsafe {
            std::env::set_var("CRANE_DISABLE_BF16_FLASH_ATTN", "1");
        }
        timing_sweep("fallback", DType::BF16);
    }

    // Cost of the per-layer mask cast. A BF16 mask pays an on-device F16
    // cast on every call, an F16 mask does not. Compare the two means.
    #[test]
    #[ignore = "manual GPU timing comparison, not a correctness check"]
    fn mask_dtype_cast_cost_timing_sweep() {
        timing_sweep("mask bf16", DType::BF16);
        timing_sweep("mask f16", DType::F16);
    }
}
