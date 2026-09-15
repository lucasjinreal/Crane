// SPDX-License-Identifier: MIT
//! Fused dequantize-in-attention kernels (Phase 5 of the KV-quant work).
//!
//! `QuantKvCache` (`crate::models::modules::quant_kv_cache`) stores K/V as
//! int8/int4 codes plus per-token f32 scales. Reading them back the
//! unfused way means dequantizing into a persistent scratch buffer the
//! size of the full dequantized history — so quantization shrinks storage
//! but not the attention-time working set. These two kernels close that
//! gap for both GPU decode and prefill by reading codes/scales directly,
//! dequantizing per-element inside the matmul:
//!
//! - [`quant_qk_dot`]: `Q_scaled @ dequant(K)^T -> scores`
//! - [`quant_sv_dot`]: `softmax(scores) @ dequant(V) -> output`
//!
//! Both kernels take a query-side dimension `R`: `n_rep` for decode
//! (`seq_len == 1`), or `n_rep * seq_len` for prefill (`seq_len > 1`) —
//! see `crate::models::qwen3::modeling::Attention::forward`'s fused branch
//! for how the caller folds/unfolds `R` around the mask add.
//!
//! Callers do the mask broadcast-add and softmax between the two (plain
//! candle ops — nothing to fuse there). See
//! `crate::models::qwen3::modeling::Attention::forward`'s fused branch for
//! the call site.
//!
//! Three backends, selected at compile time like the rest of `fused_ops`:
//! CUDA (PTX built by `build.rs` from `kernels/cuda/quant_attn.cu`), `ROCm`
//! (the same `.cu` source, compiled by `hipcc` on first use), and a portable
//! CPU fallback that dequantizes the whole cache and does a plain matmul —
//! this is O(total cached) per call, so it exists only so tests can run
//! without a GPU; production attention never reaches it on any device
//! (`KvCache::try_quantized_append` returns `None` unless the device has a
//! CUDA or `ROCm` kernel — Metal included — so attention on anything else
//! keeps using the existing incremental scratch-buffer path instead).
//!
//! Set `CRANE_QUANT_ATTN_FUSED=0` to force the unfused path (full
//! dequantize into a scratch buffer, then a plain matmul) for debugging or
//! A/B correctness checks.

use std::sync::OnceLock;

use candle_core::{DType, Result, Tensor};

use crate::models::modules::quant_kv_cache::QuantizedKvRef;

/// Whether `CRANE_QUANT_ATTN_FUSED=0` was set, cached after the first read.
#[must_use]
pub fn fused_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("CRANE_QUANT_ATTN_FUSED").as_deref() == Ok("0"))
}

/// Whether the fused kernels support `dtype` as a compute dtype: `F32`,
/// `F16`, or `BF16`. Callers must check this before appending into the
/// fused path -- `quant_qk_dot`/`quant_sv_dot` only learn a dtype is
/// unsupported after the cache has already been mutated, which is too late
/// to fall back gracefully.
#[must_use]
pub fn supports_compute_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
}

/// `Q_scaled @ dequant(K)^T -> scores`.
///
/// `q` is `[B, num_kv_heads, R, head_dim]`, pre-scaled by `1/sqrt(head_dim)`.
/// `R` is `n_rep` for decode (`seq_len == 1`) or `n_rep * seq_len` for
/// prefill (`seq_len > 1`) — the caller folds the query's own `seq_len` into
/// `R` alongside `n_rep` before calling this. Returns `scores`,
/// `[B, num_kv_heads, R, cached_len]`, in `q`'s dtype.
///
/// `kv.k_codes`/`kv.k_scale` are read as-is, without forcing contiguity: they
/// are a `narrow` view into a buffer with append headroom past `seq_len` (see
/// `QuantKvCache`'s `grow_append`), and copying that view into a dense buffer
/// on every call would re-materialize the whole cached history — exactly what
/// this fused path exists to avoid. The GPU kernels take the buffer's token
/// stride as an explicit `S_alloc` parameter instead.
///
/// # Errors
///
/// Returns an error if the underlying tensor operations fail, or if `q`'s
/// dtype isn't `F32`/`F16`/`BF16` (GPU backends only).
pub fn quant_qk_dot(q: &Tensor, kv: &QuantizedKvRef) -> Result<Tensor> {
    let q = q.contiguous()?;
    let s_alloc = token_stride(&kv.k_codes)?;
    debug_assert_eq!(
        token_stride(&kv.k_scale)?,
        s_alloc,
        "k_codes and k_scale must share the same allocated token stride"
    );
    dispatch_qk(&q, &kv.k_codes, &kv.k_scale, kv.bits, s_alloc)
}

/// `softmax(scores) @ dequant(V) -> output`.
///
/// `weights` is `[B, num_kv_heads, R, cached_len]`, post-softmax (see
/// [`quant_qk_dot`]'s doc for what `R` is). Returns `output`,
/// `[B, num_kv_heads, R, head_dim]`, in `weights`'s dtype.
///
/// `kv.v_codes`/`kv.v_scale` are read as-is; see [`quant_qk_dot`]'s doc for why.
///
/// # Errors
///
/// Returns an error if the underlying tensor operations fail, or if
/// `weights`'s dtype isn't `F32`/`F16`/`BF16` (GPU backends only).
pub fn quant_sv_dot(weights: &Tensor, kv: &QuantizedKvRef) -> Result<Tensor> {
    let weights = weights.contiguous()?;
    let s_alloc = token_stride(&kv.v_codes)?;
    debug_assert_eq!(
        token_stride(&kv.v_scale)?,
        s_alloc,
        "v_codes and v_scale must share the same allocated token stride"
    );
    dispatch_sv(&weights, &kv.v_codes, &kv.v_scale, kv.bits, s_alloc)
}

/// The physical distance (in tokens) between consecutive positions along the
/// seq-len axis (dim 2) of a `[B, H, S, D]` codes tensor. Equal to `S` when
/// the tensor is contiguous, and to the allocated buffer's width (`S_alloc`,
/// `>= S`) when it's a `narrow`'d view with append headroom past `S` — the
/// GPU kernels need this to address the un-copied buffer directly.
///
/// `grow_append`'s buffer is a single contiguous `[B, H, S_alloc, D]`
/// allocation, so narrowing dim 2 leaves dim 1's stride at `S_alloc * D`
/// while dim 3 (`D`) stays fully contiguous — dividing the two recovers
/// `S_alloc` without touching the un-narrowed buffer.
fn token_stride(codes: &Tensor) -> Result<usize> {
    let stride = codes.stride();
    let dim3 = codes.dim(3)?;
    if dim3 == 0 {
        candle_core::bail!("token_stride: codes tensor has zero-width last dim");
    }
    Ok(stride[1] / dim3)
}

fn dispatch_qk(
    q: &Tensor,
    k_codes: &Tensor,
    k_scale: &Tensor,
    bits: u32,
    s_alloc: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        cuda_impl::quant_qk_dot(q, k_codes, k_scale, bits, s_alloc)
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    {
        rocm_impl::quant_qk_dot(q, k_codes, k_scale, bits, s_alloc)
    }
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        let _ = s_alloc;
        cpu_quant_qk_dot(q, k_codes, k_scale, bits)
    }
}

fn dispatch_sv(
    weights: &Tensor,
    v_codes: &Tensor,
    v_scale: &Tensor,
    bits: u32,
    s_alloc: usize,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        cuda_impl::quant_sv_dot(weights, v_codes, v_scale, bits, s_alloc)
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    {
        rocm_impl::quant_sv_dot(weights, v_codes, v_scale, bits, s_alloc)
    }
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        let _ = s_alloc;
        cpu_quant_sv_dot(weights, v_codes, v_scale, bits)
    }
}

/// Kernel function name suffix for `bits`: `"int8"` or `"int4"`. Only the
/// GPU backends need this — they select a kernel by name; the CPU fallback
/// already branches on `bits` inside `dequantize_per_token`.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn bits_suffix(bits: u32) -> Result<&'static str> {
    match bits {
        8 => Ok("int8"),
        4 => Ok("int4"),
        _ => candle_core::bail!("quant_attn: unsupported bit width {bits}"),
    }
}

/// Kernel function name suffix for a compute dtype: `"f32"`/`"f16"`/`"bf16"`.
/// See [`bits_suffix`] for why this is GPU-only.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        dt => candle_core::bail!("quant_attn: unsupported dtype {dt:?}"),
    }
}

/// CPU fallback: dequantize the whole cache then do a plain matmul. Always
/// compiled (module doc explains why this is test-only, never a hot path).
fn cpu_quant_qk_dot(q: &Tensor, k_codes: &Tensor, k_scale: &Tensor, bits: u32) -> Result<Tensor> {
    let k = crate::models::modules::quant_kv_cache::dequantize_per_token(
        k_codes,
        k_scale,
        bits,
        q.dtype(),
    )?;
    q.matmul(&k.transpose(2, 3)?)
}

/// See [`cpu_quant_qk_dot`].
fn cpu_quant_sv_dot(
    weights: &Tensor,
    v_codes: &Tensor,
    v_scale: &Tensor,
    bits: u32,
) -> Result<Tensor> {
    let v = crate::models::modules::quant_kv_cache::dequantize_per_token(
        v_codes,
        v_scale,
        bits,
        weights.dtype(),
    )?;
    weights.matmul(&v)
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    //! CUDA launcher, dispatching over `bits` (4/8) and compute dtype
    //! (f32/f16/bf16) to the matching `quant_{qk,sv}_dot_{int8,int4}_{dtype}`
    //! kernel compiled from `kernels/cuda/quant_attn.cu`. Mirrors
    //! `crate::ops::gdn::cuda_backend`'s direct-launch pattern (this op needs
    //! 3+ tensor inputs, past `CustomOp1`/`CustomOp2`'s limit).

    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{Result, Storage, Tensor};

    use super::{bits_suffix, dtype_suffix};

    mod ptx {
        include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
    }

    const MODULE_NAME: &str = "crane_quant_attn";

    /// `q: [B, KV, R, D]`, `k_codes: [B, KV, S, D]` (`D/2` for int4),
    /// `k_scale: [B, KV, S, 1]`. `s_alloc` is the physical token stride of
    /// `k_codes`/`k_scale` (see [`super::token_stride`]), possibly `> S` when
    /// they're an un-copied `narrow`'d view. Returns `scores: [B, KV, R, S]`.
    pub fn quant_qk_dot(
        q: &Tensor,
        k_codes: &Tensor,
        k_scale: &Tensor,
        bits: u32,
        s_alloc: usize,
    ) -> Result<Tensor> {
        let (b, kv, r, d) = q.dims4()?;
        let s = k_codes.dim(2)?;
        let dev = q.device().as_cuda_device()?.clone();

        let (kc_s, kc_l) = k_codes.storage_and_layout();
        let kc_slice = match &*kc_s {
            Storage::Cuda(c) => c.as_cuda_slice::<u8>()?,
            _ => candle_core::bail!("quant_qk_dot: k_codes must be a cuda tensor"),
        };
        let kc_v = kc_slice.slice(kc_l.start_offset()..);

        let (ks_s, ks_l) = k_scale.storage_and_layout();
        let ks_slice = match &*ks_s {
            Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle_core::bail!("quant_qk_dot: k_scale must be a cuda tensor"),
        };
        let ks_v = ks_slice.slice(ks_l.start_offset()..);

        let (q_s, q_l) = q.storage_and_layout();
        let fn_name = format!(
            "quant_qk_dot_{}_{}",
            bits_suffix(bits)?,
            dtype_suffix(q.dtype())?
        );
        let func = dev.get_or_load_custom_func(&fn_name, MODULE_NAME, ptx::QUANT_ATTN)?;

        let out_elems = b * kv * r * s;
        let block = 256u32;
        #[allow(clippy::cast_possible_truncation)]
        // out_elems is bounded by B*H*S, far below u32::MAX
        let grid = (out_elems as u32).div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        // model dims, far below i32::MAX
        let (b_i, kv_i, r_i, s_i, d_i) = (b as i32, kv as i32, r as i32, s as i32, d as i32);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        // buffer width, far below i32::MAX
        let s_alloc_i = s_alloc as i32;

        let slice = match &*q_s {
            Storage::Cuda(c) => match &c.slice {
                CudaStorageSlice::F32(qsl) => {
                    let q_v = qsl.slice(q_l.start_offset()..);
                    let dst = unsafe { dev.alloc::<f32>(out_elems)? };
                    let mut builder = func.builder();
                    builder.arg(&q_v);
                    builder.arg(&kc_v);
                    builder.arg(&ks_v);
                    builder.arg(&dst);
                    builder.arg(&b_i);
                    builder.arg(&kv_i);
                    builder.arg(&r_i);
                    builder.arg(&s_i);
                    builder.arg(&d_i);
                    builder.arg(&s_alloc_i);
                    unsafe { builder.launch(cfg) }.w()?;
                    CudaStorageSlice::F32(dst)
                },
                CudaStorageSlice::F16(qsl) => {
                    let q_v = qsl.slice(q_l.start_offset()..);
                    let dst = unsafe { dev.alloc::<half::f16>(out_elems)? };
                    let mut builder = func.builder();
                    builder.arg(&q_v);
                    builder.arg(&kc_v);
                    builder.arg(&ks_v);
                    builder.arg(&dst);
                    builder.arg(&b_i);
                    builder.arg(&kv_i);
                    builder.arg(&r_i);
                    builder.arg(&s_i);
                    builder.arg(&d_i);
                    builder.arg(&s_alloc_i);
                    unsafe { builder.launch(cfg) }.w()?;
                    CudaStorageSlice::F16(dst)
                },
                CudaStorageSlice::BF16(qsl) => {
                    let q_v = qsl.slice(q_l.start_offset()..);
                    let dst = unsafe { dev.alloc::<half::bf16>(out_elems)? };
                    let mut builder = func.builder();
                    builder.arg(&q_v);
                    builder.arg(&kc_v);
                    builder.arg(&ks_v);
                    builder.arg(&dst);
                    builder.arg(&b_i);
                    builder.arg(&kv_i);
                    builder.arg(&r_i);
                    builder.arg(&s_i);
                    builder.arg(&d_i);
                    builder.arg(&s_alloc_i);
                    unsafe { builder.launch(cfg) }.w()?;
                    CudaStorageSlice::BF16(dst)
                },
                _ => candle_core::bail!("quant_qk_dot: unsupported q dtype"),
            },
            _ => candle_core::bail!("quant_qk_dot: q must be a cuda tensor"),
        };

        let dst = CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok(Tensor::from_storage(
            Storage::Cuda(dst),
            (b, kv, r, s),
            BackpropOp::none(),
            false,
        ))
    }

    /// `weights: [B, KV, R, S]`, `v_codes: [B, KV, S, D]` (`D/2` for int4),
    /// `v_scale: [B, KV, S, 1]`. `s_alloc` is the physical token stride of
    /// `v_codes`/`v_scale` (see [`super::token_stride`]), possibly `> S` when
    /// they're an un-copied `narrow`'d view. Returns `output: [B, KV, R, D]`.
    pub fn quant_sv_dot(
        weights: &Tensor,
        v_codes: &Tensor,
        v_scale: &Tensor,
        bits: u32,
        s_alloc: usize,
    ) -> Result<Tensor> {
        let (b, kv, r, s) = weights.dims4()?;
        let d = if bits == 8 {
            v_codes.dim(3)?
        } else {
            v_codes.dim(3)? * 2
        };
        let dev = weights.device().as_cuda_device()?.clone();

        let (vc_s, vc_l) = v_codes.storage_and_layout();
        let vc_slice = match &*vc_s {
            Storage::Cuda(c) => c.as_cuda_slice::<u8>()?,
            _ => candle_core::bail!("quant_sv_dot: v_codes must be a cuda tensor"),
        };
        let vc_v = vc_slice.slice(vc_l.start_offset()..);

        let (vs_s, vs_l) = v_scale.storage_and_layout();
        let vs_slice = match &*vs_s {
            Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle_core::bail!("quant_sv_dot: v_scale must be a cuda tensor"),
        };
        let vs_v = vs_slice.slice(vs_l.start_offset()..);

        let (w_s, w_l) = weights.storage_and_layout();
        let fn_name = format!(
            "quant_sv_dot_{}_{}",
            bits_suffix(bits)?,
            dtype_suffix(weights.dtype())?
        );
        let func = dev.get_or_load_custom_func(&fn_name, MODULE_NAME, ptx::QUANT_ATTN)?;

        let out_elems = b * kv * r * d;
        let block = 256u32;
        #[allow(clippy::cast_possible_truncation)]
        // out_elems is bounded by B*H*D, far below u32::MAX
        let grid = (out_elems as u32).div_ceil(block);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        // model dims, far below i32::MAX
        let (b_i, kv_i, r_i, s_i, d_i) = (b as i32, kv as i32, r as i32, s as i32, d as i32);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        // buffer width, far below i32::MAX
        let s_alloc_i = s_alloc as i32;

        let slice = match &*w_s {
            Storage::Cuda(c) => match &c.slice {
                CudaStorageSlice::F32(wsl) => {
                    let w_v = wsl.slice(w_l.start_offset()..);
                    let dst = unsafe { dev.alloc::<f32>(out_elems)? };
                    let mut builder = func.builder();
                    builder.arg(&w_v);
                    builder.arg(&vc_v);
                    builder.arg(&vs_v);
                    builder.arg(&dst);
                    builder.arg(&b_i);
                    builder.arg(&kv_i);
                    builder.arg(&r_i);
                    builder.arg(&s_i);
                    builder.arg(&d_i);
                    builder.arg(&s_alloc_i);
                    unsafe { builder.launch(cfg) }.w()?;
                    CudaStorageSlice::F32(dst)
                },
                CudaStorageSlice::F16(wsl) => {
                    let w_v = wsl.slice(w_l.start_offset()..);
                    let dst = unsafe { dev.alloc::<half::f16>(out_elems)? };
                    let mut builder = func.builder();
                    builder.arg(&w_v);
                    builder.arg(&vc_v);
                    builder.arg(&vs_v);
                    builder.arg(&dst);
                    builder.arg(&b_i);
                    builder.arg(&kv_i);
                    builder.arg(&r_i);
                    builder.arg(&s_i);
                    builder.arg(&d_i);
                    builder.arg(&s_alloc_i);
                    unsafe { builder.launch(cfg) }.w()?;
                    CudaStorageSlice::F16(dst)
                },
                CudaStorageSlice::BF16(wsl) => {
                    let w_v = wsl.slice(w_l.start_offset()..);
                    let dst = unsafe { dev.alloc::<half::bf16>(out_elems)? };
                    let mut builder = func.builder();
                    builder.arg(&w_v);
                    builder.arg(&vc_v);
                    builder.arg(&vs_v);
                    builder.arg(&dst);
                    builder.arg(&b_i);
                    builder.arg(&kv_i);
                    builder.arg(&r_i);
                    builder.arg(&s_i);
                    builder.arg(&d_i);
                    builder.arg(&s_alloc_i);
                    unsafe { builder.launch(cfg) }.w()?;
                    CudaStorageSlice::BF16(dst)
                },
                _ => candle_core::bail!("quant_sv_dot: unsupported weights dtype"),
            },
            _ => candle_core::bail!("quant_sv_dot: weights must be a cuda tensor"),
        };

        let dst = CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok(Tensor::from_storage(
            Storage::Cuda(dst),
            (b, kv, r, d),
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
mod rocm_impl {
    //! ROCm/HIP launcher, the counterpart of [`super::cuda_impl`] against the
    //! same `kernels/cuda/quant_attn.cu` — candle compiles it with `hipcc` on
    //! first use and caches the code object, so nothing is built at
    //! `cargo build` time. Mirrors `crate::ops::gdn::rocm_backend`'s use of
    //! `crate::ops::rocm`'s shared pointer/launch/wrap helpers.

    use candle_core::{DType, Result, Tensor};

    use super::{bits_suffix, dtype_suffix};
    use crate::ops::rocm::{self, arg};

    const MODULE_NAME: &str = "crane_quant_attn";
    const SOURCE: &str = include_str!("../../../kernels/cuda/quant_attn.cu");

    /// `q: [B, KV, R, D]`, `k_codes: [B, KV, S, D]` (`D/2` for int4),
    /// `k_scale: [B, KV, S, 1]`. `s_alloc` is the physical token stride of
    /// `k_codes`/`k_scale` (see [`super::token_stride`]), possibly `> S` when
    /// they're an un-copied `narrow`'d view. Returns `scores: [B, KV, R, S]`.
    pub fn quant_qk_dot(
        q: &Tensor,
        k_codes: &Tensor,
        k_scale: &Tensor,
        bits: u32,
        s_alloc: usize,
    ) -> Result<Tensor> {
        let (b, kv, r, d) = q.dims4()?;
        let s = k_codes.dim(2)?;
        let dev = q.device().as_rocm_device()?.clone();

        let (kc_s, kc_l) = k_codes.storage_and_layout();
        let kc_p = rocm::device_ptr_strided(&kc_s, kc_l, DType::U8, "quant_qk_dot k_codes")?;
        let (ks_s, ks_l) = k_scale.storage_and_layout();
        let ks_p = rocm::device_ptr_strided(&ks_s, ks_l, DType::F32, "quant_qk_dot k_scale")?;
        let (q_s, q_l) = q.storage_and_layout();
        let q_p = rocm::device_ptr(&q_s, q_l, q.dtype(), "quant_qk_dot q")?;

        let out_elems = b * kv * r * s;
        let kernel = format!(
            "quant_qk_dot_{}_{}",
            bits_suffix(bits)?,
            dtype_suffix(q.dtype())?
        );
        #[allow(clippy::cast_possible_truncation)]
        // out_elems is bounded by B*H*S, far below u32::MAX
        let (grid, block) = ((out_elems as u32).div_ceil(256), 256u32);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        // model dims, far below i32::MAX
        let (b_i, kv_i, r_i, s_i, d_i) = (b as i32, kv as i32, r as i32, s as i32, d as i32);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        // buffer width, far below i32::MAX
        let s_alloc_i = s_alloc as i32;

        macro_rules! launch_and_wrap {
            ($ty:ty, $wrap:ident) => {{
                let dst = dev.alloc::<$ty>(out_elems)?;
                let dst_p = dst.as_ptr();
                let mut args = vec![
                    arg(&q_p),
                    arg(&kc_p),
                    arg(&ks_p),
                    arg(&dst_p),
                    arg(&b_i),
                    arg(&kv_i),
                    arg(&r_i),
                    arg(&s_i),
                    arg(&d_i),
                    arg(&s_alloc_i),
                ];
                // SAFETY: the argument list matches `quant_qk_dot_*`'s
                // signature in `kernels/cuda/quant_attn.cu`; the grid covers
                // every output element; every operand outlives the launch.
                unsafe {
                    rocm::launch(
                        &dev,
                        MODULE_NAME,
                        &kernel,
                        SOURCE,
                        grid,
                        block,
                        0,
                        &mut args,
                    )?;
                }
                rocm::$wrap(dst, &dev, (b, kv, r, s))
            }};
        }

        Ok(match q.dtype() {
            DType::F32 => launch_and_wrap!(f32, wrap_f32),
            DType::F16 => launch_and_wrap!(half::f16, wrap_f16),
            DType::BF16 => launch_and_wrap!(half::bf16, wrap_bf16),
            dt => candle_core::bail!("quant_qk_dot: unsupported dtype {dt:?}"),
        })
    }

    /// `weights: [B, KV, R, S]`, `v_codes: [B, KV, S, D]` (`D/2` for int4),
    /// `v_scale: [B, KV, S, 1]`. `s_alloc` is the physical token stride of
    /// `v_codes`/`v_scale` (see [`super::token_stride`]), possibly `> S` when
    /// they're an un-copied `narrow`'d view. Returns `output: [B, KV, R, D]`.
    pub fn quant_sv_dot(
        weights: &Tensor,
        v_codes: &Tensor,
        v_scale: &Tensor,
        bits: u32,
        s_alloc: usize,
    ) -> Result<Tensor> {
        let (b, kv, r, s) = weights.dims4()?;
        let d = if bits == 8 {
            v_codes.dim(3)?
        } else {
            v_codes.dim(3)? * 2
        };
        let dev = weights.device().as_rocm_device()?.clone();

        let (vc_s, vc_l) = v_codes.storage_and_layout();
        let vc_p = rocm::device_ptr_strided(&vc_s, vc_l, DType::U8, "quant_sv_dot v_codes")?;
        let (vs_s, vs_l) = v_scale.storage_and_layout();
        let vs_p = rocm::device_ptr_strided(&vs_s, vs_l, DType::F32, "quant_sv_dot v_scale")?;
        let (w_s, w_l) = weights.storage_and_layout();
        let w_p = rocm::device_ptr(&w_s, w_l, weights.dtype(), "quant_sv_dot weights")?;

        let out_elems = b * kv * r * d;
        let kernel = format!(
            "quant_sv_dot_{}_{}",
            bits_suffix(bits)?,
            dtype_suffix(weights.dtype())?
        );
        #[allow(clippy::cast_possible_truncation)]
        // out_elems is bounded by B*H*D, far below u32::MAX
        let (grid, block) = ((out_elems as u32).div_ceil(256), 256u32);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        // model dims, far below i32::MAX
        let (b_i, kv_i, r_i, s_i, d_i) = (b as i32, kv as i32, r as i32, s as i32, d as i32);
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        // buffer width, far below i32::MAX
        let s_alloc_i = s_alloc as i32;

        macro_rules! launch_and_wrap {
            ($ty:ty, $wrap:ident) => {{
                let dst = dev.alloc::<$ty>(out_elems)?;
                let dst_p = dst.as_ptr();
                let mut args = vec![
                    arg(&w_p),
                    arg(&vc_p),
                    arg(&vs_p),
                    arg(&dst_p),
                    arg(&b_i),
                    arg(&kv_i),
                    arg(&r_i),
                    arg(&s_i),
                    arg(&d_i),
                    arg(&s_alloc_i),
                ];
                // SAFETY: the argument list matches `quant_sv_dot_*`'s
                // signature in `kernels/cuda/quant_attn.cu`; the grid covers
                // every output element; every operand outlives the launch.
                unsafe {
                    rocm::launch(
                        &dev,
                        MODULE_NAME,
                        &kernel,
                        SOURCE,
                        grid,
                        block,
                        0,
                        &mut args,
                    )?;
                }
                rocm::$wrap(dst, &dev, (b, kv, r, d))
            }};
        }

        Ok(match weights.dtype() {
            DType::F32 => launch_and_wrap!(f32, wrap_f32),
            DType::F16 => launch_and_wrap!(half::f16, wrap_f16),
            DType::BF16 => launch_and_wrap!(half::bf16, wrap_bf16),
            dt => candle_core::bail!("quant_sv_dot: unsupported dtype {dt:?}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{D, DType, Device};

    use super::*;
    use crate::models::modules::quant_kv_cache::QuantKvCache;

    fn rand_kv(b: usize, h: usize, s: usize, d: usize) -> Tensor {
        (Tensor::randn(0f32, 1f32, (b, h, s, d), &Device::Cpu).unwrap() * 4.0).unwrap()
    }

    // The fused CPU path (dequantize + matmul) must match a from-scratch
    // dequantize + matmul done by hand — same math, just called through the
    // fused entry points.
    fn full_attention_matches_reference(bits: u32) {
        let b_sz = 1;
        let kv_heads = 2;
        let n_rep = 3;
        let head_dim = 8;
        let seq_len = 5;

        let mut cache = QuantKvCache::new(bits);
        let k = rand_kv(b_sz, kv_heads, seq_len, head_dim);
        let v = rand_kv(b_sz, kv_heads, seq_len, head_dim);
        let kv_ref = cache.quantized_append(&k, &v).unwrap();

        let q = rand_kv(b_sz, kv_heads, n_rep, head_dim);
        let scale = 1.0 / (head_dim as f64).sqrt();
        let q_scaled = (&q * scale).unwrap();

        let scores = quant_qk_dot(&q_scaled, &kv_ref).unwrap();
        let scores = candle_nn::ops::softmax_last_dim(&scores).unwrap();
        let output = quant_sv_dot(&scores, &kv_ref).unwrap();

        // Reference: dequantize the whole cache, then a plain matmul chain.
        let k_ref = kv_ref.dequantize_k(DType::F32).unwrap();
        let v_ref = kv_ref.dequantize_v(DType::F32).unwrap();
        let ref_scores = q_scaled.matmul(&k_ref.transpose(2, 3).unwrap()).unwrap();
        let ref_scores = candle_nn::ops::softmax_last_dim(&ref_scores).unwrap();
        let ref_output = ref_scores.matmul(&v_ref).unwrap();

        let diff = (&output - &ref_output)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 1e-4, "bits={bits}: fused path diverged: {diff}");

        assert_eq!(output.dims(), &[b_sz, kv_heads, n_rep, head_dim]);
    }

    // Verifies the fused int8 path matches full dequantize + matmul.
    #[test]
    fn cpu_full_attention_matches_reference_int8() {
        full_attention_matches_reference(8);
    }

    // Verifies the fused int4 path matches full dequantize + matmul.
    #[test]
    fn cpu_full_attention_matches_reference_int4() {
        full_attention_matches_reference(4);
    }

    // Mirrors `qwen3::modeling::Attention::forward`'s fused branch for
    // prefill (seq_len > 1): fold (n_rep, seq_len) into R before
    // `quant_qk_dot`, unfold to num_heads layout for a per-query-position
    // causal mask, then unfold back around `quant_sv_dot`. Must match a
    // reference standard SDPA (full dequantize + GQA-expand + the same
    // causal mask) exactly — this is the part that's new relative to the
    // decode-only (seq_len == 1, no mask unfold) case above.
    fn full_attention_matches_reference_prefill(bits: u32) {
        let b_sz = 1;
        let kv_heads = 2;
        let n_rep = 3;
        let num_heads = kv_heads * n_rep;
        let head_dim = 8;
        let seq_len = 4;

        let mut cache = QuantKvCache::new(bits);
        let k = rand_kv(b_sz, kv_heads, seq_len, head_dim);
        let v = rand_kv(b_sz, kv_heads, seq_len, head_dim);
        let kv_ref = cache.quantized_append(&k, &v).unwrap();

        let q = rand_kv(b_sz, num_heads, seq_len, head_dim);
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Causal mask [B, 1, seq_len, seq_len]: 0 on/below diagonal, -inf above.
        let mask_vals: Vec<f32> = (0..seq_len * seq_len)
            .map(|idx| {
                let (i, j) = (idx / seq_len, idx % seq_len);
                if j > i { f32::NEG_INFINITY } else { 0.0 }
            })
            .collect();
        let mask = Tensor::from_vec(mask_vals, (1, 1, seq_len, seq_len), &Device::Cpu).unwrap();

        // Fused path, exactly mirroring the modeling.rs fold/mask/unfold.
        let q_g = (q
            .reshape((b_sz, kv_heads, n_rep * seq_len, head_dim))
            .unwrap()
            * scale)
            .unwrap();
        let scores = quant_qk_dot(&q_g, &kv_ref).unwrap();
        let total_kv = scores.dim(D::Minus1).unwrap();
        let scores = scores
            .reshape((b_sz, num_heads, seq_len, total_kv))
            .unwrap();
        let scores = scores.broadcast_add(&mask).unwrap();
        let scores = candle_nn::ops::softmax_last_dim(&scores)
            .unwrap()
            .reshape((b_sz, kv_heads, n_rep * seq_len, total_kv))
            .unwrap();
        let output = quant_sv_dot(&scores, &kv_ref).unwrap();
        let output = output
            .reshape((b_sz, num_heads, seq_len, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap();

        // Reference: full dequantize + GQA-expand + standard SDPA, no
        // R-folding at all.
        let k_ref = kv_ref
            .dequantize_k(DType::F32)
            .unwrap()
            .unsqueeze(2)
            .unwrap()
            .expand((b_sz, kv_heads, n_rep, seq_len, head_dim))
            .unwrap()
            .reshape((b_sz, num_heads, seq_len, head_dim))
            .unwrap();
        let v_ref = kv_ref
            .dequantize_v(DType::F32)
            .unwrap()
            .unsqueeze(2)
            .unwrap()
            .expand((b_sz, kv_heads, n_rep, seq_len, head_dim))
            .unwrap()
            .reshape((b_sz, num_heads, seq_len, head_dim))
            .unwrap();
        let q_scaled = (&q * scale).unwrap();
        let ref_scores = q_scaled.matmul(&k_ref.transpose(2, 3).unwrap()).unwrap();
        let ref_scores = ref_scores.broadcast_add(&mask).unwrap();
        let ref_scores = candle_nn::ops::softmax_last_dim(&ref_scores).unwrap();
        let ref_output = ref_scores.matmul(&v_ref).unwrap();
        let ref_output = ref_output.transpose(1, 2).unwrap().contiguous().unwrap();

        let diff = (&output - &ref_output)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-4,
            "bits={bits}: fused prefill path diverged: {diff}"
        );
        assert_eq!(output.dims(), &[b_sz, seq_len, num_heads, head_dim]);
    }

    // Verifies the fused int8 path matches reference SDPA for prefill.
    #[test]
    fn cpu_full_attention_matches_reference_prefill_int8() {
        full_attention_matches_reference_prefill(8);
    }

    // Verifies the fused int4 path matches reference SDPA for prefill.
    #[test]
    fn cpu_full_attention_matches_reference_prefill_int4() {
        full_attention_matches_reference_prefill(4);
    }

    // The env var toggle must be readable and reflect its value; the actual
    // caller-side skip behavior is covered by
    // `quant_kv_cache::try_quantized_append`'s own tests.
    #[test]
    fn fused_disabled_reads_env_var() {
        // `fused_disabled` caches its result in a `OnceLock` for the whole
        // process, so this only checks the function doesn't panic and
        // returns a bool consistent with whatever the process-wide value is
        // — it can't independently flip the var mid-test-run.
        let _ = fused_disabled();
    }

    #[test]
    fn qk_dot_rejects_unsupported_dtype() {
        let mut cache = QuantKvCache::new(8);
        let k = rand_kv(1, 1, 2, 8);
        let v = rand_kv(1, 1, 2, 8);
        let kv_ref = cache.quantized_append(&k, &v).unwrap();

        let q = rand_kv(1, 1, 1, 8).to_dtype(DType::U8).unwrap();
        let err = quant_qk_dot(&q, &kv_ref).expect_err("U8 query dtype must error");
        assert!(err.to_string().contains("unsupported"));
    }

    // A contiguous tensor has no append headroom past its visible length, so
    // the physical and visible token counts must match.
    #[test]
    fn token_stride_contiguous() {
        let codes = rand_kv(1, 2, 5, 8);
        assert_eq!(token_stride(&codes).unwrap(), 5);
    }

    // A `narrow`'d view into a buffer with append headroom must report the
    // buffer's full allocated width, not the narrowed (visible) length --
    // this is the exact case the GPU kernels rely on to index un-copied
    // headroom directly.
    #[test]
    fn token_stride_narrowed_with_headroom() {
        let store = rand_kv(1, 2, 10, 8);
        let view = store.narrow(2, 0, 5).unwrap();
        assert_eq!(token_stride(&view).unwrap(), 10);
    }

    #[test]
    fn supports_compute_dtype_matches_kernel_suffixes() {
        assert!(supports_compute_dtype(DType::F32));
        assert!(supports_compute_dtype(DType::F16));
        assert!(supports_compute_dtype(DType::BF16));
        assert!(!supports_compute_dtype(DType::U8));
    }
}
