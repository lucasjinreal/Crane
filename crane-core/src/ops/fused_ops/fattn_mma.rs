// SPDX-License-Identifier: MIT
//! Manual (non-`CustomOp`) dispatch for the vendored tensor-core (MMA)
//! flash-attention kernel (`kernels/cuda/fattn/fattn_mma_f16.cuh`).
//!
//! The kernel takes Q, K, V, an optional mask, and an optional analytic
//! `KV_max` index buffer. That is five tensor-shaped operands, more than
//! `CustomOp3`'s hard cap of three. This follows the manual-dispatch
//! pattern `ops::fused_ops::qsa_mask`/`ops::gdn` use instead: extract each
//! tensor's device pointer directly via `storage_and_layout()`, build the
//! kernel's scalar arguments by hand, and launch through
//! `candle_core::cuda_backend`/`rocm_backend`'s raw launch API.
//!
//! `q` is F32, `k`/`v`/`mask` are F16. This is a mixed-precision contract
//! upstream bakes into this kernel family itself (`GGML_ASSERT(Q->type ==
//! GGML_TYPE_F32)` in llama.cpp's `launch_fattn`, not vendored; the kernel
//! body casts `Q`'s raw bytes straight to `float2`/`float`, confirmed by a
//! real GPU hardware exception before this was caught), unlike every other
//! Crane flash-attention kernel (F16 throughout). `head_dim` in {64, 128,
//! 256, 512}, `parallel_blocks=1` (see `fattn.rs`'s module doc). Unlike
//! `ops::fused_ops::flash_attn_mma`'s
//! kernel (which masks purely via a `kv_offset` scalar and never reads a
//! mask tensor), this kernel family branches on whether `mask`/`KV_max` are
//! null (`if (ncols2 > 1 || mask_h)` in `fattn_mma_f16.cuh`), so `mask`
//! being `None` is a real, intentional "no masking at all" mode (`full`),
//! not an unsupported case. Callers always pass the caller's actual mask
//! intent through, never silently drop it.
//!
//! Gated the same way as `fattn.rs`: every item here exists solely to
//! support CUDA/ROCm kernel dispatch, so there is no CPU-relevant content
//! to keep unconditional.
#![cfg(any(feature = "cuda", feature = "rocm"))]

use candle_core::{DType, Layout, Result, Tensor};

use super::fattn::{
    FATTN_KQ_STRIDE, MAX_GRID_X_DIM, gqa_z_tiles, init_fastdiv_values, is_amd_backend, mma_config,
    mma_shared_mem_bytes, q_tiles, select_mma_ncols,
};

/// Bounds-checked scalar arguments for a `crane_fattn_mma_f16_*` launch,
/// after the five tensor-pointer parameters (`Q`, `K`, `V`, `mask`,
/// `sinks`, `KV_max`, `dst`, `dst_meta`). Field order and types match
/// `fattn_mma_f16.cuh`'s `DECL_FATTN_MMA_F16_CASE`-generated signature.
/// Currently unused (no ALiBi/logit-softcap support): `max_bias`,
/// `m0`, `m1` are fixed at `0.0`/`1.0`/`1.0` and `n_head_log2` is unused
/// (see `crane_fattn_shim.cuh`'s `get_alibi_slope`).
///
/// Shared with [`super::fattn_tile`]: both kernel families take the exact
/// same scalar parameter list after their tensor-pointer operands.
#[allow(clippy::struct_field_names)]
pub(crate) struct LaunchScalars {
    /// Attention scale (`1/sqrt(head_dim)`, or caller-supplied).
    pub(crate) scale: f32,
    /// `ALiBi` max bias; always `0.0` (unused, see this struct's doc comment).
    pub(crate) max_bias: f32,
    /// `ALiBi` `m0` slope base; always `1.0` (unused).
    pub(crate) m0: f32,
    /// `ALiBi` `m1` slope base; always `1.0` (unused).
    pub(crate) m1: f32,
    /// `log2` of the head count, for `ALiBi` slope selection; always `0`
    /// (unused).
    pub(crate) n_head_log2: u32,
    /// Logit soft-cap threshold; always `0.0` (unused).
    pub(crate) logit_softcap: f32,
    /// Q's `head_dim` (ggml `ne[0]`).
    pub(crate) ne00: i32,
    /// Q's `seq_q` (ggml `ne[1]`), packed for the kernel's Barrett-reduction
    /// division (see [`super::fattn::init_fastdiv_values`]).
    pub(crate) ne01: super::fattn::Uint3,
    /// Q's `num_heads_q` (ggml `ne[2]`).
    pub(crate) ne02: i32,
    /// Q's batch (ggml `ne[3]`).
    pub(crate) ne03: i32,
    /// Q's byte stride along `seq_q`.
    pub(crate) nb01: i32,
    /// Q's byte stride along `num_heads_q`.
    pub(crate) nb02: i32,
    /// Q's byte stride along batch.
    pub(crate) nb03: i32,
    /// K's `head_dim` (ggml `ne[0]`).
    pub(crate) ne10: i32,
    /// K's `seq_kv` (ggml `ne[1]`).
    pub(crate) ne11: i32,
    /// K's `num_heads_kv` (ggml `ne[2]`).
    pub(crate) ne12: i32,
    /// K's batch (ggml `ne[3]`).
    pub(crate) ne13: i32,
    /// K's byte stride along `seq_kv`.
    pub(crate) nb11: i32,
    /// K's byte stride along `num_heads_kv`.
    pub(crate) nb12: i32,
    /// K's byte stride along batch.
    pub(crate) nb13: i64,
    /// V's byte stride along `seq_kv`.
    pub(crate) nb21: i32,
    /// V's byte stride along `num_heads_kv`.
    pub(crate) nb22: i32,
    /// V's byte stride along batch.
    pub(crate) nb23: i64,
    /// Mask's `seq_q` axis (`mask->ne[1]`); `0` when no mask is given.
    pub(crate) ne31: i32,
    /// Mask's third axis (broadcast dim); `0` when no mask is given.
    pub(crate) ne32: i32,
    /// Mask's fourth axis (broadcast dim); `0` when no mask is given.
    pub(crate) ne33: i32,
    /// Mask's byte stride along `seq_q`; `0` when no mask is given.
    pub(crate) nb31: i32,
    /// Mask's byte stride along its third axis; `0` when no mask is given.
    pub(crate) nb32: i32,
    /// Mask's byte stride along its fourth axis; `0` when no mask is given.
    pub(crate) nb33: i64,
}

fn to_i32(n: usize, what: &str) -> Result<i32> {
    i32::try_from(n)
        .map_err(|_| candle_core::Error::Msg(format!("fattn_mma: {what} ({n}) exceeds i32::MAX")))
}

fn to_i64(n: usize, what: &str) -> Result<i64> {
    i64::try_from(n)
        .map_err(|_| candle_core::Error::Msg(format!("fattn_mma: {what} ({n}) exceeds i64::MAX")))
}

fn to_u32(n: usize, what: &str) -> Result<u32> {
    u32::try_from(n)
        .map_err(|_| candle_core::Error::Msg(format!("fattn_mma: {what} ({n}) exceeds u32::MAX")))
}

/// Byte stride for axis `axis` of a tensor whose element size is `elem_bytes`.
fn byte_stride(layout: &Layout, axis: usize, elem_bytes: usize, what: &str) -> Result<i64> {
    to_i64(layout.stride()[axis] * elem_bytes, what)
}

#[allow(clippy::too_many_arguments)]
impl LaunchScalars {
    // ne31/ne32/ne33 and nb31/nb32/nb33 match ggml's own naming for these
    // fields (see fattn_mma_f16.cuh's kernel signature) and are kept as-is
    // for traceability back to the vendored kernel rather than renamed to
    // satisfy the lint.
    #[allow(clippy::similar_names)]
    pub(crate) fn new(
        l_q: &Layout,
        l_k: &Layout,
        l_v: &Layout,
        l_mask: Option<&Layout>,
        seq_q: usize,
        h_q: usize,
        d: usize,
        seq_kv: usize,
        h_kv: usize,
        scale: f32,
    ) -> Result<Self> {
        // Upstream requires Q to stay F32 (`GGML_ASSERT(Q->type ==
        // GGML_TYPE_F32)` in `launch_fattn`, not vendored, see this
        // module's doc comment). Only K/V/mask are F16. A real GPU
        // hardware exception (out-of-bounds Q read, since Q's actual
        // buffer is half the byte size the kernel's F32-stride math
        // expects) confirmed this before the fix.
        const F16_BYTES: usize = 2;
        const F32_BYTES: usize = 4;
        let (mp, l, dv) = init_fastdiv_values(to_u64(seq_q)?);
        // ggml's `ne`/`nb` arrays are innermost-axis-first (`ne[0]` = kv_len,
        // the fastest-varying axis), the opposite of Candle's `dims()`
        // (outermost-first). The kernel never reads `ne30`/`nb30` (kv_len):
        // it already has that from K's `ne11`/`seq_kv`, and indexes into the
        // mask purely via `nb31` (the per-query-row stride). So `ne31` here
        // is the mask's seq_q axis, matching llama.cpp's own `mask->ne[1]`,
        // not kv_len.
        let (ne31, ne32, ne33, nb31, nb32, nb33) = if let Some(lm) = l_mask {
            let dims = lm.shape().dims();
            let n = dims.len();
            if n < 2 {
                candle_core::bail!("fattn_mma: mask must be at least 2D, got {dims:?}");
            }
            let axis_dim = |back: usize| if n > back { dims[n - 1 - back] } else { 1 };
            let axis_stride = |back: usize| -> Result<i64> {
                if n > back {
                    byte_stride(lm, n - 1 - back, F16_BYTES, "mask stride")
                } else {
                    Ok(0)
                }
            };
            let i32_stride = |back: usize| -> Result<i32> {
                axis_stride(back)?
                    .try_into()
                    .map_err(|_| candle_core::Error::Msg("fattn_mma: mask stride overflow".into()))
            };
            (
                to_i32(axis_dim(1), "mask ne31 (seq_q)")?,
                to_i32(axis_dim(2), "mask ne32")?,
                to_i32(axis_dim(3), "mask ne33")?,
                i32_stride(1)?,
                i32_stride(2)?,
                axis_stride(3)?,
            )
        } else {
            (0, 0, 0, 0, 0, 0)
        };
        Ok(Self {
            scale,
            max_bias: 0.0,
            m0: 1.0,
            m1: 1.0,
            n_head_log2: 0,
            logit_softcap: 0.0,
            ne00: to_i32(d, "ne00")?,
            ne01: super::fattn::Uint3 { x: mp, y: l, z: dv },
            ne02: to_i32(h_q, "ne02")?,
            ne03: to_i32(l_q.shape().dims4()?.0, "ne03")?,
            nb01: byte_stride(l_q, 1, F32_BYTES, "nb01")?
                .try_into()
                .map_err(|_| candle_core::Error::Msg("fattn_mma: q stride overflow".into()))?,
            nb02: byte_stride(l_q, 2, F32_BYTES, "nb02")?
                .try_into()
                .map_err(|_| candle_core::Error::Msg("fattn_mma: q stride overflow".into()))?,
            nb03: byte_stride(l_q, 0, F32_BYTES, "nb03")?
                .try_into()
                .map_err(|_| candle_core::Error::Msg("fattn_mma: q stride overflow".into()))?,
            ne10: to_i32(d, "ne10")?,
            ne11: to_i32(seq_kv, "ne11")?,
            ne12: to_i32(h_kv, "ne12")?,
            ne13: to_i32(l_k.shape().dims4()?.0, "ne13")?,
            nb11: byte_stride(l_k, 1, F16_BYTES, "nb11")?
                .try_into()
                .map_err(|_| candle_core::Error::Msg("fattn_mma: k stride overflow".into()))?,
            nb12: byte_stride(l_k, 2, F16_BYTES, "nb12")?
                .try_into()
                .map_err(|_| candle_core::Error::Msg("fattn_mma: k stride overflow".into()))?,
            nb13: byte_stride(l_k, 0, F16_BYTES, "nb13")?,
            nb21: byte_stride(l_v, 1, F16_BYTES, "nb21")?
                .try_into()
                .map_err(|_| candle_core::Error::Msg("fattn_mma: v stride overflow".into()))?,
            nb22: byte_stride(l_v, 2, F16_BYTES, "nb22")?
                .try_into()
                .map_err(|_| candle_core::Error::Msg("fattn_mma: v stride overflow".into()))?,
            nb23: byte_stride(l_v, 0, F16_BYTES, "nb23")?,
            ne31,
            ne32,
            ne33,
            nb31,
            nb32,
            nb33,
        })
    }
}

fn to_u64(n: usize) -> Result<u64> {
    u64::try_from(n)
        .map_err(|_| candle_core::Error::Msg(format!("fattn_mma: {n} exceeds u64::MAX")))
}

/// Grid/launch geometry for `parallel_blocks=1`: port of `launch_fattn`'s
/// `block_dim`/`blocks_num` computation (`fattn-common.cuh`, not vendored,
/// see `fattn.rs`'s module doc). Fixed at one block per (query tile, GQA
/// group, KV head, batch element) rather than upstream's occupancy-driven
/// `parallel_blocks` search.
struct LaunchGeometry {
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem_bytes: u32,
}

impl LaunchGeometry {
    /// Upstream's `ggml_cuda_flash_attn_ext_mma_f16_case` always calls
    /// `launch_fattn` with `stream_k=true` (confirmed from `fattn.cu`'s exact
    /// call site: the 9th positional argument is a literal `true`). This
    /// holds even though this dispatcher never uses stream-K's
    /// multi-block-per-tile splitting.
    /// `launch_fattn`'s stream-K branch launches a **1D** grid
    /// (`blocks_num = (ntiles_dst, 1, 1)`) and the kernel body confirms this:
    /// `flash_attn_ext_f16` only ever reads `blockIdx.x`/`gridDim.x`, never
    /// `.y`/`.z`. It decomposes the full `(query tile, GQA group, KV head,
    /// batch)` space from a single linear `kbc` work-range computed purely
    /// from `blockIdx.x`. A 3D grid (as the *non*-stream-K branch, and this
    /// kernel's own tile-kernel sibling, uses) launches redundant blocks
    /// that silently race on the same output and corrupts every later
    /// kernel argument's effective grid-relative addressing. This was
    /// confirmed by a real "`HSA_STATUS_ERROR_EXCEPTION` ... hardware
    /// exception" on real `ROCm` hardware before this fix. One block per unit of work (`ntiles_dst`
    /// total, matching this design's "one block handles its tile's full KV
    /// range, no fixup/combine pass" approach) is the degenerate,
    /// stream-K-disabled case: `gridDim.x = ntiles_dst`, `gridDim.y =
    /// gridDim.z = 1`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        dkq: usize,
        ncols1: usize,
        ncols2: usize,
        seq_q: usize,
        h_kv: usize,
        batch: usize,
        gqa_ratio: usize,
        is_amd: bool,
    ) -> Result<Self> {
        let cfg = mma_config(dkq, ncols1 * ncols2, is_amd)?;
        let nwarps = cfg.nthreads / 32;
        let ntiles_dst = q_tiles(seq_q, ncols1) * gqa_z_tiles(gqa_ratio, ncols2) * h_kv * batch;
        if ntiles_dst > MAX_GRID_X_DIM {
            candle_core::bail!(
                "fattn_mma: grid size ({ntiles_dst}) exceeds the CUDA/HIP grid x-dimension limit ({MAX_GRID_X_DIM})"
            );
        }
        let shared_mem_bytes = mma_shared_mem_bytes(dkq, ncols1, ncols2, &cfg, is_amd);
        Ok(Self {
            grid: (to_u32(ntiles_dst, "ntiles_dst")?, 1, 1),
            block: (32, to_u32(nwarps, "nwarps")?, 1),
            shared_mem_bytes: to_u32(shared_mem_bytes, "shared_mem_bytes")?,
        })
    }
}

/// Tensor-core flash-attention prefill: `softmax(q @ k^T * scale [+ mask])
/// @ v`, always honoring `mask` when given (unlike
/// `ops::fused_ops::flash_attn_mma`'s kernel). `k`/`v` are BSHD F16, `q` is
/// BSHD F32 (see this module's doc comment), `head_dim` in {64, 128, 256,
/// 512}; `mask` (F16, additive) must broadcast as `[.., seq_q, seq_kv]`
/// when given. `kv_max` (I32), when given, is one entry per query tile per
/// batch element. See `ops::fused_ops::fattn`'s `q_tiles`/`gqa_z_tiles`
/// for the tile layout; it bounds each tile's KV loop (`fattn_mma_f16.cuh`'s
/// `kb0_stop`).
///
/// **`mask = None` is only valid when the internally-selected `ncols2`
/// (see [`super::fattn::select_mma_ncols`]) resolves to `1`; this is
/// enforced below, not just documented.** The kernel's own mask-pointer
/// logic (`fattn_mma_f16.cuh`: `ncols2 == 1 && !mask ? nullptr : (const
/// half *) (mask + nb33*(sequence % ne33))`) only skips dereferencing
/// `mask`/`ne33` when `ncols2 == 1`; for `ncols2 > 1` (GQA-batched launches)
/// it unconditionally computes `sequence % ne33` even when the caller
/// passed no mask, and `ne33 = 0` in that case. This was confirmed on real
/// `ROCm` hardware as a null-pointer/modulo-by-zero hardware exception
/// before this was caught. Since `ncols2` is selected internally from
/// `gqa_ratio`/`seq_kv` (not caller-controlled), a caller that always wants
/// "no masking" and might hit a GQA-batched launch must pass a real
/// all-zero mask tensor instead of `None` to stay safe across every
/// `(gqa_ratio, seq_kv)` combination; passing `None` into a `ncols2 > 1`
/// launch now returns a `Result::Err` instead of reaching the kernel.
/// `None` is a true optimization only for the plain-MHA/unpadded-`seq_kv`
/// case where `ncols2 == 1` is guaranteed. (On AMD, `ncols2` never actually
/// resolves to `1` at all. See `select_mma_ncols`'s doc comment, so
/// `mask = None` is never valid there in practice, only on NVIDIA.)
///
/// # Errors
///
/// Returns an error if shapes/dtypes are invalid, `mask = None` while the
/// selected `ncols2 != 1` (see above), no instantiated kernel covers this
/// `(head_dim, GQA ratio, seq_q, seq_kv)` combination (callers
/// should fall back to [`super::fattn_tile::fattn_tile_prefill`] in that
/// case), or the device is CPU (no CPU implementation).
pub fn fattn_mma_prefill(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    kv_max: Option<&Tensor>,
    scale: f32,
) -> Result<Tensor> {
    // See this module's doc comment: upstream requires q to stay F32 while
    // k/v are F16. This is not a uniform-F16 contract like Crane's other kernels.
    if q.dtype() != DType::F32 {
        candle_core::bail!("fattn_mma: q must be F32, got {:?}", q.dtype());
    }
    if k.dtype() != DType::F16 || v.dtype() != DType::F16 {
        candle_core::bail!(
            "fattn_mma: k/v must be F16, got k={:?} v={:?}",
            k.dtype(),
            v.dtype()
        );
    }
    if let Some(m) = mask
        && m.dtype() != DType::F16
    {
        candle_core::bail!("fattn_mma: mask must be F16, got {:?}", m.dtype());
    }
    if let Some(kv) = kv_max
        && kv.dtype() != DType::I32
    {
        candle_core::bail!("fattn_mma: kv_max must be I32, got {:?}", kv.dtype());
    }

    #[cfg(feature = "cuda")]
    {
        cuda::fattn_mma_prefill_cuda(q, k, v, mask, kv_max, scale)
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    {
        rocm::fattn_mma_prefill_rocm(q, k, v, mask, kv_max, scale)
    }
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        let _ = (q, k, v, mask, kv_max, scale);
        candle_core::bail!("fattn_mma_prefill: requires the cuda or rocm feature")
    }
}

/// `(head_dim, gqa_ratio, seq_q, seq_kv, is_amd)` for a `[B, S, H, D]` BSHD
/// `q`/`k` pair, shared by [`mma_ncols1_for`] and [`pad_for_gqa_batching`] so
/// the shape derivation and `num_heads_q`/`num_heads_kv` validation live in
/// one place.
///
/// # Errors
///
/// Returns an error if `q`/`k` aren't 4D, `num_heads_kv` is `0`, or
/// `num_heads_q` isn't a positive multiple of it.
fn mma_shape_params(q: &Tensor, k: &Tensor) -> Result<(usize, usize, usize, usize, bool)> {
    let (_, seq_q, h_q, d) = q.dims4()?;
    let (_, seq_kv, h_kv, _) = k.dims4()?;
    if h_kv == 0 || h_q % h_kv != 0 {
        candle_core::bail!(
            "fattn_mma: num_heads_q ({h_q}) must be a positive multiple of num_heads_kv ({h_kv})"
        );
    }
    Ok((d, h_q / h_kv, seq_q, seq_kv, is_amd_backend()))
}

/// `(ncols1, gqa_ratio)` for a `[B, S, H, D]` BSHD `q`/`k` pair, shared by
/// [`fattn_mma_causal`]/[`fattn_mma_full`]/[`fattn_mma_windowed`] to size
/// their `KV_max` tensor before dispatching to [`fattn_mma_prefill`] (which
/// re-derives the same `ncols1` internally from the same inputs).
///
/// # Errors
///
/// Returns an error if `q`/`k` aren't 4D, `num_heads_kv` is `0`, or
/// `num_heads_q` isn't a multiple of it, or if no instantiated MMA kernel
/// covers this `(head_dim, GQA ratio, seq_q, seq_kv)` combination (callers
/// should fall back to [`super::fattn_tile`] in that case).
fn mma_ncols1_for(q: &Tensor, k: &Tensor) -> Result<usize> {
    let (d, gqa_ratio, seq_q, seq_kv, is_amd) = mma_shape_params(q, k)?;
    select_mma_ncols(d, gqa_ratio, seq_q, seq_kv, is_amd)
        .map(|(ncols1, _)| ncols1)
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "fattn_mma: no instantiated kernel for head_dim={d} gqa_ratio={gqa_ratio} seq_q={seq_q} seq_kv={seq_kv}"
            ))
        })
}

/// If treating `seq_kv` as padded up to the next [`FATTN_KQ_STRIDE`]
/// multiple would unlock a GQA-batched (`ncols2 > 1`) kernel that the live,
/// unpadded length can't (see [`super::fattn::select_mma_ncols`]'s doc
/// comment: that path requires an aligned `seq_kv`, which a live prefill
/// call's real token count almost never has), returns the padded length.
/// Returns `None` when padding wouldn't change the outcome: `gqa_ratio == 1`
/// (no sibling queries to batch, regardless of alignment), `seq_kv` is
/// already aligned, or even the padded length still has no `ncols2 > 1`
/// kernel (e.g. AMD `head_dim > 256`, which has no MMA path at all). This
/// mirrors llama.cpp's own precondition for this path: upstream's `K->ne[1]`
/// is always a KV-cache buffer already allocated padded to this stride, so
/// its equivalent check is realistically always true; Crane's prefill calls
/// build exactly the live-length K/V tensor instead, so this function (and
/// [`pad_for_gqa_batching`]) exist to reproduce that same precondition.
fn padded_seq_kv_for_gqa_batching(
    head_dim: usize,
    gqa_ratio: usize,
    seq_q: usize,
    seq_kv: usize,
    is_amd: bool,
) -> Option<usize> {
    if gqa_ratio == 1 {
        return None;
    }
    if let Some((_, ncols2)) = select_mma_ncols(head_dim, gqa_ratio, seq_q, seq_kv, is_amd)
        && ncols2 > 1
    {
        return None;
    }
    let padded = seq_kv.next_multiple_of(FATTN_KQ_STRIDE);
    if padded == seq_kv {
        return None;
    }
    match select_mma_ncols(head_dim, gqa_ratio, seq_q, padded, is_amd) {
        Some((_, ncols2)) if ncols2 > 1 => Some(padded),
        _ => None,
    }
}

/// Zero-pads `k`/`v`'s sequence axis (BSHD axis 1) from their live length up
/// to `padded_seq_kv`. The kernel's GQA-batched (`ncols2 > 1`) tail tile is
/// read with no bounds check (`oob_check = false` whenever `ncols2 > 1`, see
/// `fattn_mma_f16.cuh`'s `flash_attn_ext_f16_iter` dispatch), so this
/// physically extends the buffer rather than merely widening the logical
/// `KV_max`/mask bound, which alone would leave the tail read past the real
/// allocation. Padding with zeros (not uninitialized memory) matters: the
/// padded columns are always masked to `-inf` by [`pad_mask`] before the
/// softmax, but `-inf + NaN == NaN`, so a garbage bit pattern in the padding
/// could still corrupt a row's softmax reduction even though its
/// contribution is meant to be zero.
fn pad_kv(k: &Tensor, v: &Tensor, padded_seq_kv: usize) -> Result<(Tensor, Tensor)> {
    let (b, seq_kv, h_kv, d) = k.dims4()?;
    let pad_len = padded_seq_kv - seq_kv;
    let zeros = Tensor::zeros((b, pad_len, h_kv, d), DType::F16, k.device())?;
    Ok((Tensor::cat(&[k, &zeros], 1)?, Tensor::cat(&[v, &zeros], 1)?))
}

/// Extends an additive F16 mask's last (kv) axis from its live width up to
/// `padded_seq_kv` with `-inf`, so the padded K/V rows [`pad_kv`] appends
/// never contribute to any query's softmax. `seq_kv` must be `<=
/// padded_seq_kv` (checked by [`pad_for_gqa_batching`] before calling this).
fn pad_mask(mask: &Tensor, padded_seq_kv: usize) -> Result<Tensor> {
    let dims = mask.dims().to_vec();
    let seq_kv = *dims.last().expect("mask must have at least one axis");
    let mut pad_shape = dims.clone();
    *pad_shape
        .last_mut()
        .expect("mask must have at least one axis") = padded_seq_kv - seq_kv;
    let neg_inf =
        Tensor::full(f32::NEG_INFINITY, pad_shape, mask.device())?.to_dtype(DType::F16)?;
    Tensor::cat(&[mask, &neg_inf], dims.len() - 1)
}

/// Pads `k`/`v`/`mask` for GQA-batched dispatch when beneficial (see
/// [`padded_seq_kv_for_gqa_batching`]), otherwise returns clones of the
/// inputs unchanged (a `Tensor` clone is an `Arc` bump, not a data copy).
/// `mask` must already be F16 and cover `k`'s live (unpadded) `seq_kv`
/// width — checked explicitly below rather than left as an unenforced
/// precondition, since a mismatched mask would otherwise make [`pad_mask`]
/// either underflow (`mask`'s kv axis wider than `k`'s) or silently splice
/// `-inf` in the wrong place (`mask`'s kv axis narrower than `k`'s,
/// e.g. a broadcastable width-1 axis). Shared by every
/// `fattn_mma_*`/`fattn_mma_*_with_mask` entry point so the padding decision
/// and mechanics live in one place.
///
/// # Errors
///
/// Returns an error if `q`/`k`'s shapes are invalid (see
/// [`mma_shape_params`]) or `mask`'s last dimension doesn't equal `k`'s live
/// `seq_kv`.
fn pad_for_gqa_batching(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (d, gqa_ratio, seq_q, seq_kv, is_amd) = mma_shape_params(q, k)?;
    let mask_kv = mask.dims().last().copied().unwrap_or(0);
    if mask_kv != seq_kv {
        candle_core::bail!(
            "fattn_mma: mask's last dimension ({mask_kv}) must equal k's seq_kv ({seq_kv})"
        );
    }
    match padded_seq_kv_for_gqa_batching(d, gqa_ratio, seq_q, seq_kv, is_amd) {
        Some(padded) => {
            let (k, v) = pad_kv(k, v, padded)?;
            let mask = pad_mask(mask, padded)?;
            Ok((k, v, mask))
        },
        None => Ok((k.clone(), v.clone(), mask.clone())),
    }
}

/// Causal flash-attention: `j <= i + kv_offset`. Builds the additive mask
/// this kernel family always needs (see this module's doc comment: unlike
/// [`super::flash_attn_mma`], `mask = None` is not a safe universal
/// "no masking" shortcut here) and dispatches to
/// [`fattn_mma_causal_with_mask`].
///
/// # Errors
///
/// See [`fattn_mma_causal_with_mask`].
pub fn fattn_mma_causal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    kv_offset: usize,
) -> Result<Tensor> {
    let (_, seq_q, _, _) = q.dims4()?;
    let (_, seq_kv, _, _) = k.dims4()?;
    let mask =
        super::fattn::build_additive_mask_f16(seq_q, seq_kv, kv_offset, None, 0, q.device())?;
    fattn_mma_causal_with_mask(q, k, v, scale, kv_offset, &mask)
}

/// Same as [`fattn_mma_causal`], but for a caller that already has a mask
/// shared across multiple layers in one forward pass (e.g. a decoder stack
/// whose `forward()` builds the mask once via
/// `models::utils::build_additive_causal_mask` and passes it to every
/// layer) — avoiding this kernel family's `O(seq_q * seq_kv)` mask rebuild
/// on every call that [`fattn_mma_causal`] pays. `mask` must encode exactly
/// `j <= i + kv_offset`, covering `k`'s live `seq_kv` (any dtype; cast to
/// F16 here if needed). `KV_max` is still rebuilt per call — it is
/// `O(batch * n_tiles)`, not `O(seq^2)`, so there is no analogous cost to
/// share.
///
/// # Errors
///
/// Returns a candle error if `q`/`k`/`v`'s shapes are incompatible, if
/// `mask` isn't broadcastable to `[.., seq_q, seq_kv]`, or if any tensor op
/// (including the padding this function may perform, see
/// [`pad_for_gqa_batching`]) fails.
pub fn fattn_mma_causal_with_mask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    kv_offset: usize,
    mask: &Tensor,
) -> Result<Tensor> {
    let (b, seq_q, _, _) = q.dims4()?;
    let mask = if mask.dtype() == DType::F16 {
        mask.clone()
    } else {
        mask.to_dtype(DType::F16)?
    };
    let (k, v, mask) = pad_for_gqa_batching(q, k, v, &mask)?;
    let ncols1 = mma_ncols1_for(q, &k)?;
    let seq_kv = k.dims4()?.1;
    // kv_max_cap rounds seq_kv up to FATTN_KQ_STRIDE rather than capping at
    // the bare seq_kv: the MMA kernel divides KV_max by nbatch_fa
    // (32/64/128, always a FATTN_KQ_STRIDE divisor), and its own kb0_stop is
    // independently bounded by the real buffer length, so this cap only
    // controls alignment, not safety (see build_analytic_kv_max's doc
    // comment).
    let kv_max_cap = seq_kv.next_multiple_of(FATTN_KQ_STRIDE);
    let kv_max = super::fattn::build_analytic_kv_max(
        b,
        seq_q,
        ncols1,
        kv_offset,
        seq_kv,
        0,
        kv_max_cap,
        q.device(),
    )?;
    fattn_mma_prefill(q, &k, &v, Some(&mask), Some(&kv_max), scale)
}

/// Sliding-window flash-attention: `i + kv_offset - window_left <= j <= i +
/// kv_offset + window_right`. See [`fattn_mma_causal`] for the mask/`KV_max`
/// rationale; `window_left` only narrows the mask (there is no lower bound
/// to the kernel's KV loop), while `window_right` also tightens `KV_max`.
///
/// # Errors
///
/// See [`fattn_mma_windowed_with_mask`].
pub fn fattn_mma_windowed(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    kv_offset: usize,
    window_left: usize,
    window_right: usize,
) -> Result<Tensor> {
    let (_, seq_q, _, _) = q.dims4()?;
    let (_, seq_kv, _, _) = k.dims4()?;
    let mask = super::fattn::build_additive_mask_f16(
        seq_q,
        seq_kv,
        kv_offset,
        Some(window_left),
        window_right,
        q.device(),
    )?;
    fattn_mma_windowed_with_mask(q, k, v, scale, kv_offset, window_right, &mask)
}

/// Same as [`fattn_mma_windowed`], but for a caller that already has a
/// shared mask built once per forward pass — see
/// [`fattn_mma_causal_with_mask`]'s doc comment for the rationale. `mask`
/// must encode `i + kv_offset - window_left <= j <= i + kv_offset +
/// window_right`, covering `k`'s live `seq_kv`; `window_left` is not needed
/// here since it only narrows the mask (already baked into `mask`), never
/// `KV_max`'s upper bound (see `build_analytic_kv_max`'s doc comment).
///
/// # Errors
///
/// See [`fattn_mma_causal_with_mask`].
pub fn fattn_mma_windowed_with_mask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    kv_offset: usize,
    window_right: usize,
    mask: &Tensor,
) -> Result<Tensor> {
    let (b, seq_q, _, _) = q.dims4()?;
    let mask = if mask.dtype() == DType::F16 {
        mask.clone()
    } else {
        mask.to_dtype(DType::F16)?
    };
    let (k, v, mask) = pad_for_gqa_batching(q, k, v, &mask)?;
    let ncols1 = mma_ncols1_for(q, &k)?;
    let seq_kv = k.dims4()?.1;
    // kv_max_cap: see fattn_mma_causal_with_mask's comment.
    let kv_max_cap = seq_kv.next_multiple_of(FATTN_KQ_STRIDE);
    let kv_max = super::fattn::build_analytic_kv_max(
        b,
        seq_q,
        ncols1,
        kv_offset,
        seq_kv,
        window_right,
        kv_max_cap,
        q.device(),
    )?;
    fattn_mma_prefill(q, &k, &v, Some(&mask), Some(&kv_max), scale)
}

/// Non-causal (full, bidirectional) flash-attention: every query attends to
/// every key. Passes a real all-zero mask rather than `None` (see this
/// module's doc comment), since `None` is only safe when the
/// internally-selected `ncols2` resolves to `1`, which this entry point
/// cannot guarantee for every `(gqa_ratio, seq_kv)` shape. No `KV_max`:
/// nothing is masked, so there is no tail of the KV loop to skip. Still pads
/// for GQA-batching (see [`pad_for_gqa_batching`]) when beneficial, even
/// though no model wires this entry point up yet, for consistency with
/// [`fattn_mma_causal`]/[`fattn_mma_windowed`].
///
/// # Errors
///
/// See [`fattn_mma_prefill`] and [`pad_for_gqa_batching`].
pub fn fattn_mma_full(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let (_, seq_q, _, _) = q.dims4()?;
    let (_, seq_kv, _, _) = k.dims4()?;
    let mask = Tensor::zeros((1, 1, seq_q, seq_kv), DType::F16, q.device())?;
    let (k, v, mask) = pad_for_gqa_batching(q, k, v, &mask)?;
    fattn_mma_prefill(q, &k, &v, Some(&mask), None, scale)
}

#[cfg(feature = "cuda")]
mod cuda {
    use candle_core::Storage;
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};

    use super::{LaunchGeometry, LaunchScalars};
    use crate::ops::fused_ops::fattn::{mma_kernel_name, select_mma_ncols, validate_fattn_bshd};
    use candle_core::{Result, Tensor};

    mod ptx {
        include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
    }

    fn ptx_for(dkq: usize) -> Result<&'static str> {
        Ok(match dkq {
            64 => ptx::FATTN_MMA_HD64,
            128 => ptx::FATTN_MMA_HD128,
            256 => ptx::FATTN_MMA_HD256,
            512 => ptx::FATTN_MMA_HD512,
            _ => candle_core::bail!("fattn_mma: unsupported head_dim {dkq}"),
        })
    }

    fn f16_slice<'a>(
        storage: &'a Storage,
        what: &str,
    ) -> Result<&'a candle_core::cuda_backend::cudarc::driver::CudaSlice<half::f16>> {
        match storage {
            Storage::Cuda(c) => match &c.slice {
                CudaStorageSlice::F16(s) => Ok(s),
                _ => candle_core::bail!("fattn_mma: {what} must be F16"),
            },
            _ => candle_core::bail!("fattn_mma: {what} must be a cuda tensor"),
        }
    }

    // See this module's doc comment: q stays F32 (never F16) for this
    // kernel family.
    fn f32_slice<'a>(
        storage: &'a Storage,
        what: &str,
    ) -> Result<&'a candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>> {
        match storage {
            Storage::Cuda(c) => match &c.slice {
                CudaStorageSlice::F32(s) => Ok(s),
                _ => candle_core::bail!("fattn_mma: {what} must be F32"),
            },
            _ => candle_core::bail!("fattn_mma: {what} must be a cuda tensor"),
        }
    }

    #[allow(clippy::too_many_lines, clippy::many_single_char_names)]
    pub(super) fn fattn_mma_prefill_cuda(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        kv_max: Option<&Tensor>,
        scale: f32,
    ) -> Result<Tensor> {
        let (q_s, q_l) = q.storage_and_layout();
        let (k_s, k_l) = k.storage_and_layout();
        let (v_s, v_l) = v.storage_and_layout();
        let (b, seq_q, h_q, d, seq_kv, h_kv) = validate_fattn_bshd(q_l, k_l, v_l)?;
        let gqa_ratio = h_q / h_kv;
        let dev = q.device().as_cuda_device()?.clone();
        let is_amd = false;

        let Some((ncols1, ncols2)) = select_mma_ncols(d, gqa_ratio, seq_q, seq_kv, is_amd) else {
            candle_core::bail!(
                "fattn_mma: no instantiated kernel for head_dim={d} gqa_ratio={gqa_ratio} seq_kv={seq_kv}"
            );
        };
        // See fattn_mma_prefill's doc comment: the kernel unconditionally
        // computes `sequence % ne33` whenever ncols2 > 1, which null-derefs/
        // div-by-zeros when mask=None (ne33=0). Only ncols2==1 makes None
        // safe.
        if mask.is_none() && ncols2 != 1 {
            candle_core::bail!(
                "fattn_mma: mask=None requires ncols2 == 1 (got ncols2={ncols2}); \
                 pass an explicit all-zero mask for unmasked attention with this GQA shape"
            );
        }
        let geometry = LaunchGeometry::new(d, ncols1, ncols2, seq_q, h_kv, b, gqa_ratio, is_amd)?;
        let kernel_name = mma_kernel_name(d, ncols1, ncols2);
        let module_name = format!("crane_fattn_mma_hd{d}");
        let func = dev.get_or_load_custom_func(&kernel_name, &module_name, ptx_for(d)?)?;

        let q_sl = f32_slice(&q_s, "q")?.slice(q_l.start_offset()..);
        let k_sl = f16_slice(&k_s, "k")?.slice(k_l.start_offset()..);
        let v_sl = f16_slice(&v_s, "v")?.slice(v_l.start_offset()..);

        let mask_layout = mask.map(Tensor::storage_and_layout);
        let mask_sl = mask_layout
            .as_ref()
            .map(|(s, l)| {
                Ok::<_, candle_core::Error>(f16_slice(s, "mask")?.slice(l.start_offset()..))
            })
            .transpose()?;

        let kv_max_layout = kv_max.map(Tensor::storage_and_layout);
        let kv_max_sl = kv_max_layout
            .as_ref()
            .map(|(s, l)| {
                let Storage::Cuda(c) = &**s else {
                    candle_core::bail!("fattn_mma: kv_max must be a cuda tensor");
                };
                let CudaStorageSlice::I32(sl) = &c.slice else {
                    candle_core::bail!("fattn_mma: kv_max must be I32");
                };
                Ok::<_, candle_core::Error>(sl.slice(l.start_offset()..))
            })
            .transpose()?;

        let p = LaunchScalars::new(
            q_l,
            k_l,
            v_l,
            mask_layout.as_ref().map(|(_, l)| *l),
            seq_q,
            h_q,
            d,
            seq_kv,
            h_kv,
            scale,
        )?;

        let n_out = b * seq_q * h_q * d;
        let dst = unsafe { dev.alloc::<f32>(n_out) }?;

        let cfg = LaunchConfig {
            grid_dim: geometry.grid,
            block_dim: geometry.block,
            shared_mem_bytes: geometry.shared_mem_bytes,
        };

        let mut builder = func.builder();
        builder.arg(&q_sl);
        builder.arg(&k_sl);
        builder.arg(&v_sl);
        // SAFETY: the kernel treats a null mask/sinks/KV_max pointer as "not
        // provided" (`if (ncols2 > 1 || mask_h)` in `fattn_mma_f16.cuh`), the
        // same contract llama.cpp's own host dispatch relies on. Passing a
        // `0u64` in place of the typed slice relies on cudarc copying the
        // argument's raw bytes into the kernel's pointer-sized parameter
        // slot verbatim, matching a null `const char*`/`const int*` on the
        // device side. This specific mechanism has not been verified against
        // a real CUDA build (no nvcc available in this environment). Verify
        // on real CUDA hardware before trusting the `mask`/`kv_max =
        // None` path.
        match &mask_sl {
            Some(s) => builder.arg(s),
            None => builder.arg(&0u64),
        };
        builder.arg(&0u64); // sinks: not wired up yet (no attention-sink models)
        match &kv_max_sl {
            Some(s) => builder.arg(s),
            None => builder.arg(&0u64),
        };
        builder.arg(&dst);
        builder.arg(&0u64); // dst_meta: unused, parallel_blocks=1 never needs the combine pass
        builder.arg(&p.scale);
        builder.arg(&p.max_bias);
        builder.arg(&p.m0);
        builder.arg(&p.m1);
        builder.arg(&p.n_head_log2);
        builder.arg(&p.logit_softcap);
        builder.arg(&p.ne00);
        builder.arg(&p.ne01);
        builder.arg(&p.ne02);
        builder.arg(&p.ne03);
        builder.arg(&p.nb01);
        builder.arg(&p.nb02);
        builder.arg(&p.nb03);
        builder.arg(&p.ne10);
        builder.arg(&p.ne11);
        builder.arg(&p.ne12);
        builder.arg(&p.ne13);
        builder.arg(&p.nb11);
        builder.arg(&p.nb12);
        builder.arg(&p.nb13);
        builder.arg(&p.nb21);
        builder.arg(&p.nb22);
        builder.arg(&p.nb23);
        builder.arg(&p.ne31);
        builder.arg(&p.ne32);
        builder.arg(&p.ne33);
        builder.arg(&p.nb31);
        builder.arg(&p.nb32);
        builder.arg(&p.nb33);
        unsafe { builder.launch(cfg) }.w()?;

        let dst = CudaStorage {
            slice: CudaStorageSlice::F32(dst),
            device: dev,
        };
        Tensor::from_storage(
            Storage::Cuda(dst),
            (b, seq_q, h_q, d),
            candle_core::op::BackpropOp::none(),
            false,
        )
        .to_dtype(candle_core::DType::F16)
    }
}

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
mod rocm {
    use std::ffi::c_void;

    use candle_core::rocm_backend::rocm_rs;
    use candle_core::{DType, Result, Tensor};

    use super::{LaunchGeometry, LaunchScalars};
    use crate::ops::fused_ops::fattn::{mma_kernel_name, select_mma_ncols, validate_fattn_bshd};
    use crate::ops::rocm;

    /// Each leaf vendored header is spliced into the final source exactly
    /// once, at the position its FIRST inclusion occupies in
    /// `fattn_mma_f16.cuh`'s own include list (`crane_fattn_shim.cuh`,
    /// `cp_async.cuh`, `mma.cuh`, `fattn_common.cuh`, `swizzle.cuh`, in that
    /// order). Every *other* occurrence of the same `#include` (whether in
    /// `fattn_mma_f16.cuh` itself or inside one of the leaf headers it
    /// pulls in) is stripped to an empty string instead, so no header's
    /// definitions appear twice in the assembled source. `crane_fattn_shim.cuh`
    /// alone is included from five places (`mma.cuh`, `cp_async.cuh`,
    /// `swizzle.cuh`, `fattn_common.cuh`, and `fattn_mma_f16.cuh` itself);
    /// splicing it at `fattn_mma_f16.cuh`'s own (first) inclusion site
    /// guarantees its macros exist before any later-included leaf needs
    /// them, since nothing before that point in the file requires it.
    /// Splices `with` in place of `pattern`'s single occurrence in `source`.
    /// Debug-asserts `pattern` appears exactly once first: `str::replace`
    /// can't tell "first" from "all" occurrences, so a future header
    /// refresh that adds a second `#include` of an already-spliced leaf
    /// would otherwise duplicate its content silently.
    fn splice_once(source: &str, pattern: &str, with: &str) -> String {
        debug_assert_eq!(
            source.matches(pattern).count(),
            1,
            "expected exactly one `{pattern}` occurrence to splice"
        );
        source.replacen(pattern, with, 1)
    }

    fn rocm_mma_source(instance_source: &'static str) -> String {
        let shim = include_str!("../../../kernels/cuda/fattn/crane_fattn_shim.cuh");
        let mma_h = include_str!("../../../kernels/cuda/fattn/mma.cuh")
            .replace("#include \"crane_fattn_shim.cuh\"", "");
        let cp_async = include_str!("../../../kernels/cuda/fattn/cp_async.cuh")
            .replace("#include \"crane_fattn_shim.cuh\"", "");
        let common = include_str!("../../../kernels/cuda/fattn/fattn_common.cuh")
            .replace("#include \"crane_fattn_shim.cuh\"", "");
        let swizzle = include_str!("../../../kernels/cuda/fattn/swizzle.cuh")
            .replace("#include \"crane_fattn_shim.cuh\"", "")
            .replace("#include \"mma.cuh\"", "");
        let mma_f16 = include_str!("../../../kernels/cuda/fattn/fattn_mma_f16.cuh");
        let mma_f16 = splice_once(mma_f16, "#include \"crane_fattn_shim.cuh\"", shim);
        let mma_f16 = splice_once(&mma_f16, "#include \"cp_async.cuh\"", &cp_async);
        let mma_f16 = splice_once(&mma_f16, "#include \"mma.cuh\"", &mma_h);
        let mma_f16 = splice_once(&mma_f16, "#include \"fattn_common.cuh\"", &common);
        let mma_f16 = splice_once(&mma_f16, "#include \"swizzle.cuh\"", &swizzle);
        splice_once(instance_source, "#include \"fattn_mma_f16.cuh\"", &mma_f16)
    }

    fn source_for(dkq: usize) -> Result<&'static str> {
        use std::sync::OnceLock;
        macro_rules! source_once {
            ($file:literal) => {{
                static SOURCE: OnceLock<String> = OnceLock::new();
                SOURCE.get_or_init(|| rocm_mma_source(include_str!($file)))
            }};
        }
        Ok(match dkq {
            64 => source_once!("../../../kernels/cuda/fattn/fattn_mma_hd64.cu"),
            128 => source_once!("../../../kernels/cuda/fattn/fattn_mma_hd128.cu"),
            256 => source_once!("../../../kernels/cuda/fattn/fattn_mma_hd256.cu"),
            512 => source_once!("../../../kernels/cuda/fattn/fattn_mma_hd512.cu"),
            _ => candle_core::bail!("fattn_mma: unsupported head_dim {dkq}"),
        })
    }

    #[allow(clippy::too_many_lines, clippy::many_single_char_names)]
    pub(super) fn fattn_mma_prefill_rocm(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        kv_max: Option<&Tensor>,
        scale: f32,
    ) -> Result<Tensor> {
        let (q_s, q_l) = q.storage_and_layout();
        let (k_s, k_l) = k.storage_and_layout();
        let (v_s, v_l) = v.storage_and_layout();
        let (b, seq_q, h_q, d, seq_kv, h_kv) = validate_fattn_bshd(q_l, k_l, v_l)?;
        let gqa_ratio = h_q / h_kv;
        let dev = q.device().as_rocm_device()?.clone();
        let is_amd = true;

        let Some((ncols1, ncols2)) = select_mma_ncols(d, gqa_ratio, seq_q, seq_kv, is_amd) else {
            candle_core::bail!(
                "fattn_mma: no instantiated kernel for head_dim={d} gqa_ratio={gqa_ratio} seq_kv={seq_kv}"
            );
        };
        // See fattn_mma_prefill's doc comment: the kernel unconditionally
        // computes `sequence % ne33` whenever ncols2 > 1, which null-derefs/
        // div-by-zeros when mask=None (ne33=0). Only ncols2==1 makes None
        // safe.
        if mask.is_none() && ncols2 != 1 {
            candle_core::bail!(
                "fattn_mma: mask=None requires ncols2 == 1 (got ncols2={ncols2}); \
                 pass an explicit all-zero mask for unmasked attention with this GQA shape"
            );
        }
        let geometry = LaunchGeometry::new(d, ncols1, ncols2, seq_q, h_kv, b, gqa_ratio, is_amd)?;
        let kernel_name = mma_kernel_name(d, ncols1, ncols2);
        let module_name = format!("crane_fattn_mma_hd{d}");
        let source = source_for(d)?;

        // See this module's doc comment: q stays F32 (never F16) for this
        // kernel family.
        let q_ptr = rocm::device_ptr(&q_s, q_l, DType::F32, "fattn_mma q")?;
        let k_ptr = rocm::device_ptr(&k_s, k_l, DType::F16, "fattn_mma k")?;
        let v_ptr = rocm::device_ptr(&v_s, v_l, DType::F16, "fattn_mma v")?;

        let mask_layout = mask.map(Tensor::storage_and_layout);
        let mask_ptr: *mut c_void = match &mask_layout {
            Some((s, l)) => rocm::device_ptr(s, l, DType::F16, "fattn_mma mask")?,
            None => std::ptr::null_mut(),
        };
        let kv_max_layout = kv_max.map(Tensor::storage_and_layout);
        let kv_max_ptr: *mut c_void = match &kv_max_layout {
            Some((s, l)) => rocm::device_ptr(s, l, DType::I32, "fattn_mma kv_max")?,
            None => std::ptr::null_mut(),
        };

        let p = LaunchScalars::new(
            q_l,
            k_l,
            v_l,
            mask_layout.as_ref().map(|(_, l)| *l),
            seq_q,
            h_q,
            d,
            seq_kv,
            h_kv,
            scale,
        )?;

        let n_out = b * seq_q * h_q * d;
        let dst = dev.alloc::<f32>(n_out)?;
        let dst_ptr = dst.as_ptr();
        let dst_meta_ptr: *mut c_void = std::ptr::null_mut();
        let sinks_ptr: *mut c_void = std::ptr::null_mut();
        let grid = rocm_rs::hip::Dim3::new_3d(geometry.grid.0, geometry.grid.1, geometry.grid.2);
        let block =
            rocm_rs::hip::Dim3::new_3d(geometry.block.0, geometry.block.1, geometry.block.2);

        let mut args = [
            rocm::arg(&q_ptr),
            rocm::arg(&k_ptr),
            rocm::arg(&v_ptr),
            rocm::arg(&mask_ptr),
            rocm::arg(&sinks_ptr),
            rocm::arg(&kv_max_ptr),
            rocm::arg(&dst_ptr),
            rocm::arg(&dst_meta_ptr),
            rocm::arg(&p.scale),
            rocm::arg(&p.max_bias),
            rocm::arg(&p.m0),
            rocm::arg(&p.m1),
            rocm::arg(&p.n_head_log2),
            rocm::arg(&p.logit_softcap),
            rocm::arg(&p.ne00),
            rocm::arg(&p.ne01),
            rocm::arg(&p.ne02),
            rocm::arg(&p.ne03),
            rocm::arg(&p.nb01),
            rocm::arg(&p.nb02),
            rocm::arg(&p.nb03),
            rocm::arg(&p.ne10),
            rocm::arg(&p.ne11),
            rocm::arg(&p.ne12),
            rocm::arg(&p.ne13),
            rocm::arg(&p.nb11),
            rocm::arg(&p.nb12),
            rocm::arg(&p.nb13),
            rocm::arg(&p.nb21),
            rocm::arg(&p.nb22),
            rocm::arg(&p.nb23),
            rocm::arg(&p.ne31),
            rocm::arg(&p.ne32),
            rocm::arg(&p.ne33),
            rocm::arg(&p.nb31),
            rocm::arg(&p.nb32),
            rocm::arg(&p.nb33),
        ];

        // SAFETY: the argument list above matches
        // `crane_fattn_mma_f16_d{d}_v{d}_c{ncols1}_s{ncols2}`'s `extern "C"`
        // signature exactly (`DECL_FATTN_MMA_F16_CASE` in
        // `fattn_mma_f16.cuh`). q_ptr/k_ptr/v_ptr/mask_ptr/kv_max_ptr stay
        // valid until the launch completes because `q`/`k`/`v`/`mask`/
        // `kv_max` (and the storage they borrow from) outlive this call; the
        // grid covers every output element (`geometry`'s tile counts).
        unsafe {
            rocm::launch_2d(
                &dev,
                &module_name,
                &kernel_name,
                source,
                grid,
                block,
                geometry.shared_mem_bytes,
                &mut args,
            )
        }?;

        let out = rocm::wrap_f32(dst, &dev, (b, seq_q, h_q, d));
        out.to_dtype(DType::F16)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::*;

    // No CUDA/ROCm feature compiled in: fattn_mma_prefill must bail rather
    // than silently computing a wrong answer.
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    #[test]
    fn bails_without_gpu_feature() {
        let dev = Device::Cpu;
        let q = Tensor::zeros((1, 4, 2, 64), DType::F16, &dev).unwrap();
        let k = q.clone();
        let v = q.clone();
        let err = fattn_mma_prefill(&q, &k, &v, None, None, 0.125)
            .expect_err("must bail without a GPU feature");
        assert!(
            err.to_string()
                .contains("requires the cuda or rocm feature")
        );
    }

    #[test]
    fn rejects_non_f32_q() {
        let dev = Device::Cpu;
        let q = Tensor::zeros((1, 4, 2, 64), DType::F16, &dev).unwrap();
        let k = q.clone();
        let v = q.clone();
        let err = fattn_mma_prefill(&q, &k, &v, None, None, 0.125).expect_err("must reject F16 q");
        assert!(err.to_string().contains("q must be F32"));
    }

    // k/v must be F16. q, by contrast, must be F32 (see rejects_non_f32_q).
    #[test]
    fn rejects_non_f16_kv() {
        let dev = Device::Cpu;
        let q = Tensor::zeros((1, 4, 2, 64), DType::F32, &dev).unwrap();
        let k = Tensor::zeros((1, 4, 2, 64), DType::F32, &dev).unwrap();
        let v = k.clone();
        let err =
            fattn_mma_prefill(&q, &k, &v, None, None, 0.125).expect_err("must reject F32 k/v");
        assert!(err.to_string().contains("k/v must be F16"));
    }
}

/// Correctness against a real GPU (this machine's ROCm GPU, or a CUDA GPU
/// when built with `--features cuda` elsewhere), reusing
/// `fattn::test_support`'s naive matmul/mask/softmax/matmul reference. The
/// `run_*` helpers below build their mask by hand and call
/// [`fattn_mma_prefill`] directly; [`fattn_mma_causal`]/[`fattn_mma_full`]/
/// [`fattn_mma_windowed`] get their own separate end-to-end tests further
/// down, covering the mask/`KV_max`-building logic those wrap around it.
#[cfg(all(test, any(feature = "cuda", feature = "rocm")))]
mod gpu_tests {
    use candle_core::{DType, Device, Tensor};

    use super::super::fattn::test_support::{MaskMode, naive_attention, test_gpu_device};
    use super::{fattn_mma_causal, fattn_mma_full, fattn_mma_prefill, fattn_mma_windowed};

    /// `[1, seq_q, seq_kv]` F16 additive causal mask: `0` where `j <=
    /// i + kv_offset`, `-inf` otherwise. Matches `fattn_mma_f16.cuh`'s
    /// `ne31 = seq_q`/`ne32 = 1`/`ne33 = 1` convention (see
    /// `LaunchScalars::new`'s doc comment).
    fn causal_mask(seq_q: usize, seq_kv: usize, kv_offset: usize, device: &Device) -> Tensor {
        let mut vals = vec![0f32; seq_q * seq_kv];
        for i in 0..seq_q {
            for j in 0..seq_kv {
                let rel = j as i64 - (i as i64 + kv_offset as i64);
                if rel > 0 {
                    vals[i * seq_kv + j] = f32::NEG_INFINITY;
                }
            }
        }
        Tensor::from_vec(vals, (1, seq_q, seq_kv), device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
    }

    // Full (non-causal) attention, via an explicit all-zero mask rather
    // than `mask = None`: see `fattn_mma_prefill`'s doc comment for why
    // `None` is unsafe whenever the launch's internally-selected `ncols2`
    // (driven by `gqa_ratio`/`seq_kv`, not caller-controlled) is `> 1`, as
    // every GQA-ratio-2 shape here selects. An explicit zero mask is
    // semantically identical to "no masking" and exercises the exact same
    // `mask = Some` code path `causal_*` already covers, just with
    // every entry unmasked.
    fn run_full(b: usize, sq: usize, skv: usize, hq: usize, hkv: usize, d: usize) {
        let gpu = test_gpu_device();
        let scale = 1.0f32 / (d as f32).sqrt();

        let q_f32 = Tensor::randn(0f32, 1f32, (b, sq, hq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();

        let want = naive_attention(&q_f32, &k_f32, &v_f32, scale, MaskMode::Full, 0, 0, 0);

        let q = q_f32.to_device(&gpu).unwrap(); // q stays F32 for this kernel family
        let k = k_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let v = v_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let zero_mask = Tensor::zeros((1, sq, skv), DType::F16, &gpu).unwrap();

        let got = fattn_mma_prefill(&q, &k, &v, Some(&zero_mask), None, scale)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 3e-2 * w.abs().max(1.0),
                "[{i}] got {g}, want {w} (full, head_dim={d})"
            );
        }
    }

    fn run_causal(b: usize, sq: usize, skv: usize, hq: usize, hkv: usize, d: usize) {
        let gpu = test_gpu_device();
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv.saturating_sub(sq);

        let q_f32 = Tensor::randn(0f32, 1f32, (b, sq, hq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();

        let want = naive_attention(
            &q_f32,
            &k_f32,
            &v_f32,
            scale,
            MaskMode::Causal,
            kv_offset,
            0,
            0,
        );

        let q = q_f32.to_device(&gpu).unwrap(); // q stays F32 for this kernel family
        let k = k_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let v = v_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let mask = causal_mask(sq, skv, kv_offset, &gpu);

        let got = fattn_mma_prefill(&q, &k, &v, Some(&mask), None, scale)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 3e-2 * w.abs().max(1.0),
                "[{i}] got {g}, want {w} (causal, head_dim={d})"
            );
        }
    }

    // Full (non-causal) attention via an explicit zero mask (see run_full's
    // doc comment for why not `mask = None` at this GQA ratio).
    #[test]
    fn full_zero_mask_hd64() {
        run_full(1, 32, 256, 4, 2, 64);
    }

    #[test]
    fn full_zero_mask_hd128() {
        run_full(1, 32, 256, 4, 2, 128);
    }

    // mask = None is only valid when the selected ncols2 == 1 (see
    // fattn_mma_prefill's doc comment); ncols2 never resolves to 1 on AMD
    // (select_mma_ncols's doc comment), so this must fail cleanly with the
    // "no instantiated kernel" error, never a hardware exception, on this
    // (ROCm) backend.
    #[test]
    fn none_mask_bails_cleanly_when_ncols2_never_resolves_to_one_on_amd() {
        let gpu = test_gpu_device();
        let d = 64usize;
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = Tensor::zeros((1, 16, 2, d), DType::F32, &gpu).unwrap();
        let k = Tensor::zeros((1, 16, 2, d), DType::F16, &gpu).unwrap();
        let v = k.clone();
        let result = fattn_mma_prefill(&q, &k, &v, None, None, scale);
        if let Err(e) = result {
            assert!(
                e.to_string().contains("no instantiated kernel"),
                "unexpected error: {e}"
            );
        }
        // On NVIDIA (not this environment) ncols2 == 1 is valid and this
        // would succeed instead; either outcome is acceptable here, a crash
        // is not.
    }

    // Same (head_dim, gqa_ratio, seq_kv) shape as causal_gqa_hd128, which
    // resolves to ncols2=4 on AMD: mask=None must bail with the explicit
    // "requires ncols2 == 1" error rather than reach the kernel, which
    // would null-deref/modulo-by-zero on `ne33` (see fattn_mma_prefill's
    // doc comment).
    #[test]
    fn none_mask_bails_cleanly_when_ncols2_resolves_above_one() {
        let gpu = test_gpu_device();
        let d = 128usize;
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = Tensor::zeros((1, 32, 8, d), DType::F32, &gpu).unwrap();
        let k = Tensor::zeros((1, 256, 2, d), DType::F16, &gpu).unwrap();
        let v = k.clone();
        let err = fattn_mma_prefill(&q, &k, &v, None, None, scale)
            .expect_err("mask=None at ncols2>1 must bail, not crash");
        assert!(
            err.to_string().contains("requires ncols2 == 1"),
            "unexpected error: {err}"
        );
    }

    // A caller-provided causal mask on the `mask = Some` path.
    #[test]
    fn causal_hd64() {
        run_causal(1, 32, 256, 4, 2, 64);
    }

    #[test]
    fn causal_hd128() {
        run_causal(1, 32, 256, 4, 2, 128);
    }

    // GQA: query heads share KV heads, exercising ncols2 > 1 selection.
    #[test]
    fn causal_gqa_hd128() {
        run_causal(1, 32, 256, 8, 2, 128);
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

    fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= tol * w.abs().max(1.0),
                "[{i}] got {g}, want {w} ({what})"
            );
        }
    }

    // End-to-end `fattn_mma_causal`: builds its own mask/KV_max internally,
    // unlike `run_causal`'s hand-built mask.
    #[test]
    fn causal_fn_matches_reference_hd128() {
        let gpu = test_gpu_device();
        let (b, sq, skv, hq, hkv, d) = (1, 32, 256, 8, 2, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;

        let q_f32 = Tensor::randn(0f32, 1f32, (b, sq, hq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let want = naive_attention(
            &q_f32,
            &k_f32,
            &v_f32,
            scale,
            MaskMode::Causal,
            kv_offset,
            0,
            0,
        );

        let q = q_f32.to_device(&gpu).unwrap();
        let k = k_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let v = v_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();

        let got = fattn_mma_causal(&q, &k, &v, scale, kv_offset).unwrap();
        assert_close(&got_vec(got), &want, 3e-2, "causal_fn");
    }

    // End-to-end `fattn_mma_causal` with a non-FATTN_KQ_STRIDE-aligned
    // seq_kv and GQA ratio > 1: unlike `causal_fn_matches_reference_hd128`'s
    // already-aligned skv=256, this exercises `pad_for_gqa_batching`'s real
    // `pad_kv`/`pad_mask` code path (skv=200 pads up to 256 to unlock the
    // ncols2>1 kernel on AMD), which every other GPU test in this module
    // short-circuits around by only ever using an aligned seq_kv.
    #[test]
    fn causal_fn_matches_reference_padded_gqa_hd128() {
        let gpu = test_gpu_device();
        let (b, sq, skv, hq, hkv, d) = (1, 20, 200, 8, 2, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;

        let q_f32 = Tensor::randn(0f32, 1f32, (b, sq, hq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let want = naive_attention(
            &q_f32,
            &k_f32,
            &v_f32,
            scale,
            MaskMode::Causal,
            kv_offset,
            0,
            0,
        );

        let q = q_f32.to_device(&gpu).unwrap();
        let k = k_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let v = v_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();

        let got = fattn_mma_causal(&q, &k, &v, scale, kv_offset).unwrap();
        assert_close(&got_vec(got), &want, 3e-2, "causal_fn_padded_gqa");
    }

    // End-to-end `fattn_mma_windowed`.
    #[test]
    fn windowed_fn_matches_reference_hd128() {
        // hq != hkv (GQA ratio > 1) and skv a FATTN_KQ_STRIDE (256) multiple:
        // on AMD, ncols2 only batches GQA heads (resolving to > 1) when both
        // hold (see `mma_ncols2`'s `gqa_opt_applies` gate); otherwise it
        // falls back to ncols2 == 1, which the WMMA kernel rejects
        // unconditionally (see `select_mma_ncols`'s doc comment).
        let gpu = test_gpu_device();
        let (b, sq, skv, hq, hkv, d) = (1, 20, 256, 8, 2, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;
        let (window_left, window_right) = (20, 0);

        let q_f32 = Tensor::randn(0f32, 1f32, (b, sq, hq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let want = naive_attention(
            &q_f32,
            &k_f32,
            &v_f32,
            scale,
            MaskMode::Windowed,
            kv_offset,
            window_left,
            window_right,
        );

        let q = q_f32.to_device(&gpu).unwrap();
        let k = k_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let v = v_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();

        let got =
            fattn_mma_windowed(&q, &k, &v, scale, kv_offset, window_left, window_right).unwrap();
        assert_close(&got_vec(got), &want, 3e-2, "windowed_fn");
    }

    // End-to-end `fattn_mma_full`.
    #[test]
    fn full_fn_matches_reference_hd64() {
        let gpu = test_gpu_device();
        let (b, sq, skv, hq, hkv, d) = (1, 32, 256, 4, 2, 64usize);
        let scale = 1.0f32 / (d as f32).sqrt();

        let q_f32 = Tensor::randn(0f32, 1f32, (b, sq, hq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let want = naive_attention(&q_f32, &k_f32, &v_f32, scale, MaskMode::Full, 0, 0, 0);

        let q = q_f32.to_device(&gpu).unwrap();
        let k = k_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let v = v_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();

        let got = fattn_mma_full(&q, &k, &v, scale).unwrap();
        assert_close(&got_vec(got), &want, 3e-2, "full_fn");
    }

    // A deliberately wrong mask (shifted by a large offset, so it allows and
    // denies a disjoint set of positions from the correct one) must produce
    // a different result than the correct causal mask, proving the kernel
    // actually reads `mask` rather than silently ignoring it.
    #[test]
    fn garbage_mask_changes_output() {
        // hq != hkv (GQA ratio > 1): see windowed_fn_matches_reference_hd128's
        // comment for why a ratio-1 shape has no AMD WMMA kernel at all.
        let gpu = test_gpu_device();
        let (sq, skv, hq, hkv, d) = (16, 256, 8, 2, 128usize);
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;

        let q_f32 = Tensor::randn(0f32, 1f32, (1, sq, hq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (1, skv, hkv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (1, skv, hkv, d), &Device::Cpu).unwrap();

        let q = q_f32.to_device(&gpu).unwrap();
        let k = k_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let v = v_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();

        let correct_mask = causal_mask(sq, skv, kv_offset, &gpu);
        // A reversed causal mask: visible exactly where the correct mask is
        // not (and vice versa), so the two results cannot coincidentally
        // match unless the kernel ignores the mask entirely.
        let garbage_vals: Vec<f32> = (0..sq * skv)
            .map(|idx| {
                let (i, j) = (idx / skv, idx % skv);
                let rel = j as i64 - (i as i64 + kv_offset as i64);
                if rel > 0 { 0.0 } else { f32::NEG_INFINITY }
            })
            .collect();
        let garbage_mask = Tensor::from_vec(garbage_vals, (1, sq, skv), &gpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let out_correct = fattn_mma_prefill(&q, &k, &v, Some(&correct_mask), None, scale).unwrap();
        let out_garbage = fattn_mma_prefill(&q, &k, &v, Some(&garbage_mask), None, scale).unwrap();

        let correct_vec = got_vec(out_correct);
        let garbage_vec = got_vec(out_garbage);
        let max_diff = correct_vec
            .iter()
            .zip(&garbage_vec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "garbage mask produced the same output as the correct mask (max diff {max_diff})"
        );
    }
}
