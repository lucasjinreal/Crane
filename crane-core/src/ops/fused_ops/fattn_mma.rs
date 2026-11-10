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
//! `q`, `k` and `v` share one dtype: F16 for the f16 kernel, BF16 for the
//! bf16 one. `mask` is F16 for both. `head_dim` in {64, 128, 256, 512},
//! `parallel_blocks=1` (see `fattn.rs`'s module doc). Unlike
//! `ops::fused_ops::flash_attn_mma`'s kernel (which masks purely via a
//! `kv_offset` scalar and never reads a mask tensor), this kernel family
//! branches on whether `mask`/`KV_max` are null (`if (mask_h)` in
//! `fattn_mma_f16.cuh`), so `mask` being `None` is a real, intentional "no
//! masking at all" mode, not an unsupported case. The plain `*_causal`
//! entries pass no mask and a `causal_kv_offset`, and the kernel masks the
//! diagonal itself. Callers always pass the caller's actual mask intent
//! through, never silently drop it.
//!
//! Gated the same way as `fattn.rs`: every item here exists solely to
//! support CUDA/ROCm kernel dispatch, so there is no CPU-relevant content
//! to keep unconditional.
#![cfg(crane_fattn)]

use candle_core::{DType, Device, Layout, Result, Tensor};

use super::fattn::{
    FATTN_KQ_STRIDE, FattnMmaConfig, MAX_GRID_X_DIM, MmaConfigSet, fixup_scratch_len, gqa_z_tiles,
    has_long_kv_set, init_fastdiv_values, is_amd_backend, mma_config,
    mma_config_ampere_bf16_long_kv, mma_config_bf16, mma_config_rdna_long_kv, mma_config_set,
    mma_shared_mem_bytes, mma_shared_mem_bytes_bf16, narrow_for_underfill, q_tiles,
    select_mma_ncols, select_mma_ncols_bf16, stream_k_grid, stream_k_needs_fixup,
    unsupported_config,
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

/// The kernel's `causal_kv_offset` argument. `None` maps to `-1`, which the
/// kernel reads as "not causal".
fn kernel_causal_offset(causal_kv_offset: Option<usize>) -> Result<i32> {
    causal_kv_offset.map_or(Ok(-1), |o| to_i32(o, "causal_kv_offset"))
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
        q_dtype: DType,
    ) -> Result<Self> {
        // K/V/mask are 16-bit. Q is 16-bit for the MMA kernels but F32 for
        // the tile kernel, so its size comes from its dtype.
        const F16_BYTES: usize = 2;
        let q_elem_bytes = q_dtype.size_in_bytes();
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
            nb01: byte_stride(l_q, 1, q_elem_bytes, "nb01")?
                .try_into()
                .map_err(|_| candle_core::Error::Msg("fattn_mma: q stride overflow".into()))?,
            nb02: byte_stride(l_q, 2, q_elem_bytes, "nb02")?
                .try_into()
                .map_err(|_| candle_core::Error::Msg("fattn_mma: q stride overflow".into()))?,
            nb03: byte_stride(l_q, 0, q_elem_bytes, "nb03")?
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

/// Shape of one launch, everything [`LaunchGeometry::new`] needs besides the
/// kernel choice. `seq_kv` is `k`'s live sequence length.
#[derive(Debug, Clone, Copy)]
struct LaunchShape {
    /// Q's sequence length.
    seq_q: usize,
    /// K/V's sequence length (the live buffer length, never padded).
    seq_kv: usize,
    /// Number of KV heads.
    h_kv: usize,
    /// Batch size.
    batch: usize,
    /// Q heads per KV head (`num_heads_q / h_kv`).
    gqa_ratio: usize,
}

/// Which instantiated MMA kernel a launch targets, everything
/// [`LaunchGeometry::new`] needs to pick the config table and shared-memory
/// sizing besides the launch's shape/target.
#[derive(Debug, Clone, Copy)]
struct KernelVariant {
    /// `head_dim`, one of {64, 128, 256, 512}.
    dkq: usize,
    /// Q rows per tile.
    ncols1: usize,
    /// GQA heads per tile.
    ncols2: usize,
    /// Whether this dispatch targets the AMD/ROCm backend.
    is_amd: bool,
    /// Whether `q`/`k`/`v` are BF16 (picks the BF16 config table).
    bf16: bool,
}

/// What the stream-K grid choice needs from the device.
#[derive(Debug, Clone, Copy)]
struct GridTarget {
    /// SMs (NVIDIA) or CUs (AMD).
    n_sm: usize,
    /// Overrides of the launch policy. Tests only.
    forced: LaunchOverride,
}

/// Overrides of the launch policy, for tests and benchmarks. Production
/// entries pass [`LaunchOverride::NONE`].
#[derive(Debug, Clone, Copy)]
struct LaunchOverride {
    /// Replaces the policy's grid size.
    grid: Option<usize>,
    /// Replaces the policy's tile order, see [`use_heavy_first`]. Passed
    /// to the kernel as is, also on a split grid, where the kernel keeps
    /// the stream-K order. That lets tests reach the kernel's own fallback.
    heavy_first: Option<bool>,
    /// Replaces the policy's kernel set (see [`mma_config_set`]): `Some(true)`
    /// forces [`MmaConfigSet::LongKv`], `Some(false)` forces
    /// [`MmaConfigSet::Default`]. Forcing the long set on a launch that has
    /// none is an error.
    long_kv: Option<bool>,
}

impl LaunchOverride {
    /// The policy's own choices.
    const NONE: Self = Self {
        grid: None,
        heavy_first: None,
        long_kv: None,
    };
}

/// How a tile's cost varies with its Q tile index `jt`, which decides
/// whether reordering the tiles can shorten the last wave.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TileCost {
    /// Every tile walks the same KV range.
    Uniform,
    /// A tile's KV range, and so its cost, grows with `jt`.
    GrowsWithJt,
}

impl TileCost {
    /// The cost shape of a launch with in-kernel causal masking at
    /// `causal_kv_offset`, or with a `kv_max` bound.
    ///
    /// A `kv_max` is assumed to grow with `jt`, as every in-tree one does
    /// (`build_analytic_kv_max`). One that does not only loses the ordering
    /// benefit, never correctness.
    fn of(causal_kv_offset: Option<usize>, kv_max: Option<&Tensor>) -> Self {
        if causal_kv_offset.is_some() || kv_max.is_some() {
            Self::GrowsWithJt
        } else {
            Self::Uniform
        }
    }
}

/// What a launch reads of K and V, for the kernel-set policy.
#[derive(Debug, Clone, Copy)]
struct KvExtent {
    /// K/V rows.
    rows: usize,
    /// Leading K/V rows every query tile reads, see [`mma_config_set`].
    shared_rows: usize,
    /// K/V heads.
    heads: usize,
}

impl KvExtent {
    /// The extent of a launch over `seq_kv` rows of `h_kv` heads. Only the
    /// in-kernel causal entries know their offset. Every other launch counts
    /// as reading nothing for sure, which is correct for an all-rows causal
    /// prefill. It is conservative for a non-causal (full) launch that reads
    /// all of K/V per row. That case should pass `Some(seq_kv)` once a
    /// production caller exists.
    fn new(seq_kv: usize, h_kv: usize, causal_kv_offset: Option<usize>) -> Self {
        Self {
            rows: seq_kv,
            // Off-by-one: row 0 reads o+1 keys; using o is a deliberate
            // conservative approximation, well within the cache-floor slack.
            shared_rows: causal_kv_offset.unwrap_or(0),
            heads: h_kv,
        }
    }
}

/// The tuning config and kernel set of an MMA launch.
///
/// # Errors
///
/// Returns an error if no config exists for `(dkq, ncols)`, or if `forced`
/// asks for the long-KV set where none exists.
fn launch_config(
    (dkq, ncols): (usize, usize),
    kv: KvExtent,
    (is_amd, bf16): (bool, bool),
    forced: LaunchOverride,
) -> Result<(FattnMmaConfig, MmaConfigSet)> {
    let set = match forced.long_kv {
        Some(true) if !has_long_kv_set(dkq, ncols, is_amd, bf16) => {
            candle_core::bail!(
                "fattn_mma: no long-KV kernel set for head_dim={dkq} ncols={ncols} (is_amd={is_amd} bf16={bf16})"
            )
        },
        Some(true) => MmaConfigSet::LongKv,
        Some(false) => MmaConfigSet::Default,
        None => mma_config_set(
            dkq,
            ncols,
            (kv.rows, kv.shared_rows),
            kv.heads,
            is_amd,
            bf16,
        ),
    };
    let cfg = match set {
        MmaConfigSet::LongKv if is_amd => mma_config_rdna_long_kv(dkq, ncols).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "fattn_mma: no long-KV config for head_dim={dkq} ncols={ncols}"
            ))
        })?,
        MmaConfigSet::LongKv => mma_config_ampere_bf16_long_kv(dkq, ncols).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "fattn_mma: no long-KV config for head_dim={dkq} ncols={ncols}"
            ))
        })?,
        MmaConfigSet::Default if bf16 => mma_config_bf16(dkq, ncols, is_amd)?,
        MmaConfigSet::Default => mma_config(dkq, ncols, is_amd)?,
    };
    Ok((cfg, set))
}

/// Grid/launch geometry of an MMA launch, with the stream-K fixup pass when
/// the grid splits tiles across blocks.
struct LaunchGeometry {
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem_bytes: u32,
    /// The fixup launch and its scratch buffer, or `None` when every block
    /// owns whole tiles.
    fixup: Option<FixupPlan>,
    /// The kernel's `heavy_first` argument, see [`use_heavy_first`]. The
    /// kernel ignores it unless the grid has one block per tile.
    heavy_first: bool,
}

/// The fixup launch of a stream-K grid, see `fattn_fixup.cuh`.
struct FixupPlan {
    /// F32 elements of the scratch buffer, passed to the kernel as `dst_meta`.
    scratch_len: usize,
    /// `(main blocks, Q rows per tile, GQA heads per tile)`.
    grid: (u32, u32, u32),
    /// `(head_dim, 1, 1)`: one thread per output element of a row.
    block: (u32, u32, u32),
    /// `seq_q` (ggml `ne[1]`), bounds a block's Q-row/GQA-head index.
    ne01: i32,
    /// `num_heads_q` (`h_kv * gqa_ratio`).
    ne02: i32,
    /// Q heads per KV head.
    gqa_ratio: i32,
    /// Total `(tile, KV block)` units across the whole batch, the kernel's
    /// `b * total_work / gridDim.x` seam formula divides this by.
    total_work: i32,
    /// `Uint3` fast-division constants for `iter_k * iter_j * iter_z * h_kv`.
    fd_k_j_z_ne12: super::fattn::Uint3,
    /// `Uint3` fast-division constants for `iter_k * iter_j * iter_z`.
    fd_k_j_z: super::fattn::Uint3,
    /// `Uint3` fast-division constants for `iter_k * iter_j`.
    fd_k_j: super::fattn::Uint3,
    /// `Uint3` fast-division constants for `iter_k`.
    fd_k: super::fattn::Uint3,
}

impl LaunchGeometry {
    /// Port of `launch_fattn`'s grid choice (`fattn-common.cuh`, not
    /// vendored, see `fattn.rs`'s module doc). The kernel walks one
    /// continuous `(tile, GQA group, KV head, batch, KV block)` space, so a
    /// 1D grid of any size up to the unit count covers it. Never a 3D grid:
    /// the kernel reads only `blockIdx.x`, and extra blocks would race on
    /// the same output.
    ///
    /// The grid is one block per tile, except `target.forced.grid` (tests)
    /// or AMD at `head_dim` 64, which always splits via [`stream_k_grid`]
    /// (llama.cpp's own rule for that shape). With one block per tile
    /// `fixup` is `None` and the kernel never reads `dst_meta`. Otherwise
    /// every block holding a tile's tail needs the fixup pass to merge the
    /// partial softmax states of its tile. `bf16` picks the bf16 config
    /// table and shared-memory sizing.
    ///
    /// `real_occupancy(threads, shared_mem_bytes)` lets a caller substitute
    /// the config table's `occupancy` guess with the driver's own answer
    /// (`None`, or a reported `0`, keeps the table value instead — a grid
    /// sized on zero resident blocks would launch nothing and silently
    /// return an uninitialized `dst`); see
    /// `ops::rocm::occupancy_max_active_blocks`. Only called for AMD at
    /// `head_dim` 64, the one shape that still computes a split grid.
    ///
    /// `cfg` is the `mma_config`/`mma_config_bf16` lookup for this variant's
    /// `(dkq, ncols1*ncols2, is_amd)`, passed in rather than looked up again
    /// here because callers already need it (for the tail-kernel decision)
    /// before they can build the `KernelVariant`/`real_occupancy` closure.
    ///
    /// `cost` feeds the tile order, see [`use_heavy_first`].
    ///
    /// # Errors
    ///
    /// Returns unsupported-config when the tile count exceeds the grid x
    /// limit, an error when the forced grid is outside `1..=ntiles*iter_k`
    /// or a size overflows the kernel's 32-bit arguments.
    fn new(
        variant: &KernelVariant,
        shape: &LaunchShape,
        target: &GridTarget,
        cfg: FattnMmaConfig,
        cost: TileCost,
        real_occupancy: impl FnOnce(usize, usize) -> Option<usize>,
    ) -> Result<Self> {
        let &KernelVariant {
            dkq,
            ncols1,
            ncols2,
            is_amd,
            bf16,
        } = variant;
        let nwarps = cfg.nthreads / 32;
        let (ntiles, iter_k) = tile_counts(&cfg, ncols1, ncols2, shape)?;
        if ntiles > MAX_GRID_X_DIM {
            return Err(unsupported_config(format!(
                "fattn_mma: grid size ({ntiles}) exceeds the CUDA/HIP grid x-dimension limit ({MAX_GRID_X_DIM})"
            )));
        }
        let shared_mem_bytes = if bf16 {
            mma_shared_mem_bytes_bf16(dkq, ncols1, ncols2, &cfg, is_amd)
        } else {
            mma_shared_mem_bytes(dkq, ncols1, ncols2, &cfg, is_amd)
        };
        let total = ntiles.checked_mul(iter_k).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "fattn_mma: ntiles ({ntiles}) * iter_k ({iter_k}) overflows usize"
            ))
        })?;
        let grid = match target.forced.grid {
            Some(g) if (1..=total).contains(&g) => g,
            Some(g) => candle_core::bail!("fattn_mma: forced grid {g} outside 1..={total}"),
            None if is_amd && dkq == 64 => {
                let occupancy = real_occupancy(cfg.nthreads, shared_mem_bytes)
                    .filter(|&o| o > 0)
                    .unwrap_or(cfg.occupancy);
                stream_k_grid(ntiles, iter_k, occupancy * target.n_sm)?
            },
            None => ntiles,
        };
        let fixup = stream_k_needs_fixup(total, grid, iter_k)?
            .then(|| FixupPlan::new(dkq, ncols1, ncols2, shape, iter_k, grid))
            .transpose()?;
        Ok(Self {
            grid: (to_u32(grid, "grid")?, 1, 1),
            block: (32, to_u32(nwarps, "nwarps")?, 1),
            shared_mem_bytes: to_u32(shared_mem_bytes, "shared_mem_bytes")?,
            fixup,
            heavy_first: use_heavy_first(cost, grid, ntiles, target.forced.heavy_first),
        })
    }
}

/// The most tiles a launch can have and still run heaviest-first. Above it,
/// heavy-first measured slower than the stream-K order. Measured on one
/// `ROCm` GPU and an NVIDIA L4 only. It is a tile count, not
/// device-relative, so other GPUs may want another value.
/// `fattn_bench_tile_order` measures it.
const HEAVY_FIRST_MAX_TILES: usize = 3072;

/// Whether a launch of `grid` blocks over `ntiles` tiles hands out its
/// heaviest tiles first, so the light ones fill the last wave.
///
/// Only a one-block-per-tile grid can reorder, because a split grid's fixup
/// slots depend on the stream-K order. Reordering also needs a [`TileCost`]
/// that grows with the Q tile and at most [`HEAVY_FIRST_MAX_TILES`] tiles.
/// `forced` (tests) replaces the whole policy.
///
/// The gain assumes blocks start in ascending `blockIdx.x`. CUDA and HIP do
/// that in practice but do not guarantee it. The output never depends on
/// the order.
fn use_heavy_first(cost: TileCost, grid: usize, ntiles: usize, forced: Option<bool>) -> bool {
    forced.unwrap_or(
        grid == ntiles && cost == TileCost::GrowsWithJt && ntiles <= HEAVY_FIRST_MAX_TILES,
    )
}

impl FixupPlan {
    /// The fixup launch for a `grid`-block stream-K launch: one block per
    /// (main block, Q row, GQA head of the tile), `head_dim` threads each.
    fn new(
        dkq: usize,
        ncols1: usize,
        ncols2: usize,
        shape: &LaunchShape,
        iter_k: usize,
        grid: usize,
    ) -> Result<Self> {
        let iter_j = q_tiles(shape.seq_q, ncols1);
        let iter_z = gqa_z_tiles(shape.gqa_ratio, ncols2);
        let fastdiv = |n: usize| -> Result<super::fattn::Uint3> {
            let (x, y, z) = init_fastdiv_values(to_u64(n)?);
            Ok(super::fattn::Uint3 { x, y, z })
        };
        let mul = |a: usize, b: usize, what: &str| -> Result<usize> {
            a.checked_mul(b).ok_or_else(|| {
                candle_core::Error::Msg(format!("fattn_mma: {what} overflows usize"))
            })
        };
        let iter_k_j = mul(iter_k, iter_j, "iter_k * iter_j")?;
        let per_head = mul(iter_k_j, iter_z, "iter_k * iter_j * iter_z")?;
        let per_head_h_kv = mul(per_head, shape.h_kv, "per_head * h_kv")?;
        let total_work = mul(per_head_h_kv, shape.batch, "total_work")?;
        let ne02 = mul(shape.h_kv, shape.gqa_ratio, "h_kv * gqa_ratio")?;
        let ncols = mul(ncols1, ncols2, "ncols1 * ncols2")?;
        Ok(Self {
            scratch_len: fixup_scratch_len(grid, ncols, dkq)?,
            grid: (
                to_u32(grid, "grid")?,
                to_u32(ncols1, "ncols1")?,
                to_u32(ncols2, "ncols2")?,
            ),
            block: (to_u32(dkq, "head_dim")?, 1, 1),
            ne01: to_i32(shape.seq_q, "ne01")?,
            ne02: to_i32(ne02, "ne02")?,
            gqa_ratio: to_i32(shape.gqa_ratio, "gqa_ratio")?,
            total_work: to_i32(total_work, "total_work")?,
            fd_k_j_z_ne12: fastdiv(per_head_h_kv)?,
            fd_k_j_z: fastdiv(per_head)?,
            fd_k_j: fastdiv(iter_k_j)?,
            fd_k: fastdiv(iter_k)?,
        })
    }
}

/// `(output tiles, KV blocks per tile)` of a launch: the units the grid
/// splits. Matches the kernel's `iter_k` and its tile decomposition.
///
/// # Errors
///
/// Returns an error if the tile count overflows `usize`.
fn tile_counts(
    cfg: &FattnMmaConfig,
    ncols1: usize,
    ncols2: usize,
    shape: &LaunchShape,
) -> Result<(usize, usize)> {
    let ntiles = q_tiles(shape.seq_q, ncols1)
        .checked_mul(gqa_z_tiles(shape.gqa_ratio, ncols2))
        .and_then(|n| n.checked_mul(shape.h_kv))
        .and_then(|n| n.checked_mul(shape.batch))
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "fattn_mma: tile count overflows usize (seq_q={}, ncols1={ncols1}, \
                 gqa_ratio={}, ncols2={ncols2}, h_kv={}, batch={})",
                shape.seq_q, shape.gqa_ratio, shape.h_kv, shape.batch
            ))
        })?;
    Ok((ntiles, shape.seq_kv.div_ceil(cfg.nbatch_fa)))
}

/// The device facts [`LaunchGeometry::new`] needs, queried from `device`.
///
/// # Errors
///
/// Returns an error if the attribute query fails or the device is not a
/// GPU of the enabled backend.
fn grid_target(device: &Device, forced: LaunchOverride) -> Result<GridTarget> {
    #[cfg(feature = "cuda")]
    {
        cuda::grid_target(device, forced)
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    {
        rocm::grid_target(device, forced)
    }
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        let _ = (device, forced);
        candle_core::bail!("fattn_mma: requires the cuda or rocm feature")
    }
}

/// The kernel and module name for an MMA dispatch, cached as `&'static str`
/// after the first call for a given `(dkq, ncols1, ncols2, bf16, tail)`.
/// Every dispatch used to rebuild both strings with `format!`, even though
/// the domain is small and fixed: the config tables in `fattn.rs` only
/// cover `dkq` in {64, 128, 256, 512} and a handful of `ncols1`/`ncols2`
/// pairs per `dkq`, well under 200 combinations total even counting the
/// `tail` variants. Leaking each string once into `&'static str` is a
/// one-time, bounded cost that turns the per-call allocation into a cache
/// hit; shared by both the CUDA and ROCm dispatch functions below, since
/// the naming scheme is backend-agnostic.
///
/// `tail` selects the bounds-checked-tail kernel symbol (see
/// `fattn_mma_f16.cuh`'s `DECL_FATTN_MMA_F16_TAIL_CASE`), which callers
/// should only ever request when `ncols2 > 1` and `seq_kv` isn't a
/// multiple of the selected config's `nbatch_fa` — otherwise the plain
/// (`tail = false`) symbol already covers the launch and has a smaller
/// register footprint.
///
/// `set` selects the module: [`MmaConfigSet::LongKv`] has the same kernel
/// names as the default set, in a module of its own.
fn cached_mma_names(
    dkq: usize,
    ncols1: usize,
    ncols2: usize,
    (bf16, tail): (bool, bool),
    set: MmaConfigSet,
) -> (&'static str, &'static str) {
    use std::collections::HashMap;
    use std::sync::{OnceLock, RwLock};

    type Key = (usize, usize, usize, bool, bool, MmaConfigSet);
    static CACHE: OnceLock<RwLock<HashMap<Key, (&'static str, &'static str)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    let key = (dkq, ncols1, ncols2, bf16, tail, set);
    if let Some(&names) = cache.read().unwrap().get(&key) {
        return names;
    }
    *cache.write().unwrap().entry(key).or_insert_with(|| {
        let kernel_name = match (bf16, tail) {
            (true, true) => {
                crate::ops::fused_ops::fattn::mma_bf16_kernel_name_tail(dkq, ncols1, ncols2)
            },
            (true, false) => {
                crate::ops::fused_ops::fattn::mma_bf16_kernel_name(dkq, ncols1, ncols2)
            },
            (false, true) => {
                crate::ops::fused_ops::fattn::mma_kernel_name_tail(dkq, ncols1, ncols2)
            },
            (false, false) => crate::ops::fused_ops::fattn::mma_kernel_name(dkq, ncols1, ncols2),
        };
        // The long-KV set reuses the kernel names, so only its module differs.
        let long_kv = if set == MmaConfigSet::LongKv {
            "_long_kv"
        } else {
            ""
        };
        let module_name = if bf16 {
            format!("crane_fattn_mma_bf16_hd{dkq}{long_kv}")
        } else {
            format!("crane_fattn_mma_hd{dkq}{long_kv}")
        };
        (&*String::leak(kernel_name), &*String::leak(module_name))
    })
}

/// Tensor-core flash-attention prefill: `softmax(q @ k^T * scale [+ mask]) @
/// v`, always honoring `mask` when given (unlike
/// `ops::fused_ops::flash_attn_mma`'s kernel). `q`/`k`/`v` are BSHD F16,
/// `head_dim` in {64, 128, 256, 512}; `mask` (F16, additive) must broadcast as
/// `[.., seq_q, seq_kv]` when given. `kv_max` (I32), when given, is one entry
/// per query tile per batch element. See `ops::fused_ops::fattn`'s
/// `q_tiles`/`gqa_z_tiles` for the tile layout; it bounds each tile's KV loop
/// (`fattn_mma_f16.cuh`'s `kb0_stop`).
///
/// **`mask = None` is only accepted when the internally-selected `ncols2`
/// (see [`super::fattn::select_mma_ncols`]) resolves to `1`; this is
/// enforced below, not just documented.** The kernel accepts a null mask at
/// any `ncols2`, but this public contract stays as it is for API stability.
/// Since `ncols2` is selected internally from `gqa_ratio`/`seq_q` (not
/// caller-controlled), a caller that always wants "no masking" and might
/// hit a GQA-batched launch must pass a real all-zero mask tensor instead
/// of `None`; passing `None` into a `ncols2 > 1` launch returns a
/// `Result::Err`. `None` is a true optimization only for the
/// plain-MHA/unpadded-`seq_kv` case where `ncols2 == 1` is guaranteed. (On
/// AMD, `ncols2` never actually resolves to `1` at all. See
/// `select_mma_ncols`'s doc comment, so `mask = None` is never valid there
/// in practice, only on NVIDIA.)
///
/// # Errors
///
/// Returns an error if shapes/dtypes are invalid, `mask = None` while the
/// selected `ncols2 != 1` (see above), no instantiated kernel covers this
/// `(head_dim, GQA ratio, seq_q)` combination (callers
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
    if q.dtype() != DType::F16 || k.dtype() != DType::F16 || v.dtype() != DType::F16 {
        candle_core::bail!(
            "fattn_mma: q/k/v must be F16, got q={:?} k={:?} v={:?}",
            q.dtype(),
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
    prefill_dispatch(
        q,
        k,
        v,
        mask,
        kv_max,
        scale,
        None,
        WindowBounds::NONE,
        LaunchOverride::NONE,
    )
}

/// Sliding-window bounds passed to the kernel on top of any causal/mask
/// bound. `window.left = -1` disables the left bound entirely; `window.right
/// = -1` means the causal diagonal itself is the right bound, with no
/// extension past it. A plain named struct instead of two positional `i32`s
/// so `left`/`right` can't be silently transposed across the ~10 call sites
/// that thread them through to the kernel launch.
#[derive(Debug, Clone, Copy)]
struct WindowBounds {
    left: i32,
    right: i32,
}

impl WindowBounds {
    /// No windowing: both bounds disabled.
    const NONE: Self = Self {
        left: -1,
        right: -1,
    };
}

/// Backend dispatch shared by [`fattn_mma_prefill`] and the in-kernel causal/
/// windowed entries. `causal_kv_offset = Some(o)` makes the kernel mask keys
/// `j > i + o` itself, in addition to any `mask`. `window` additionally
/// bounds keys `j < i + o - window.left` and `j > i + o + window.right`; see
/// [`WindowBounds`] for its `-1` sentinel. Callers validate dtypes before
/// calling this.
#[allow(clippy::too_many_arguments)] // one launch's worth of inputs, not separable
fn prefill_dispatch(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    kv_max: Option<&Tensor>,
    scale: f32,
    causal_kv_offset: Option<usize>,
    window: WindowBounds,
    forced: LaunchOverride,
) -> Result<Tensor> {
    let padded_mask;
    let mask = match mask {
        Some(m) => {
            let seq_kv = k.dims4()?.1;
            check_mask_covers_kv(m, seq_kv, "fattn_mma")?;
            padded_mask = pad_mask_stride(m, seq_kv)?;
            Some(&padded_mask)
        },
        None => None,
    };
    #[cfg(feature = "cuda")]
    {
        cuda::fattn_mma_prefill_cuda(
            q,
            k,
            v,
            mask,
            kv_max,
            scale,
            causal_kv_offset,
            window.left,
            window.right,
            forced,
        )
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    {
        rocm::fattn_mma_prefill_rocm(
            q,
            k,
            v,
            mask,
            kv_max,
            scale,
            causal_kv_offset,
            window.left,
            window.right,
            forced,
        )
    }
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        let _ = (
            q,
            k,
            v,
            mask,
            kv_max,
            scale,
            causal_kv_offset,
            window,
            forced,
        );
        candle_core::bail!("fattn_mma_prefill: requires the cuda or rocm feature")
    }
}

/// Native-bf16 counterpart of [`fattn_mma_prefill`]: `k`/`v` stay BF16 instead
/// of being downcast to F16, avoiding the overflow-to-`inf` risk that downcast
/// carries for BF16-range activation values outside F16's narrower dynamic
/// range (see `kernels/cuda/fattn/fattn_mma_bf16.cuh`'s module doc). `q` is
/// BF16 like `k`/`v`, `mask` is still F16 (the mask is an additive bias
/// independent of K/V's dtype). Same masking/`kv_max`/error contract as
/// [`fattn_mma_prefill`]; the one additional failure mode is no instantiated
/// kernel for this `(head_dim, GQA ratio, seq_q)` shape under
/// [`super::fattn::select_mma_ncols_bf16`] (narrower than the f16 kernel's
/// coverage — see that function's doc comment), in which case callers fall
/// back to the F16 cast path, not the tile kernel directly.
///
/// # Errors
///
/// Returns an error if shapes/dtypes are invalid (`q`/`k`/`v` must be
/// BF16, `mask` must be F16, `kv_max` must be I32), `mask = None`
/// while the selected `ncols2 != 1`, no instantiated bf16 kernel covers
/// this shape, or the device is CPU.
pub fn fattn_mma_bf16_prefill(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    kv_max: Option<&Tensor>,
    scale: f32,
) -> Result<Tensor> {
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        candle_core::bail!(
            "fattn_mma_bf16: q/k/v must be BF16, got q={:?} k={:?} v={:?}",
            q.dtype(),
            k.dtype(),
            v.dtype()
        );
    }
    if let Some(m) = mask
        && m.dtype() != DType::F16
    {
        candle_core::bail!("fattn_mma_bf16: mask must be F16, got {:?}", m.dtype());
    }
    if let Some(kv) = kv_max
        && kv.dtype() != DType::I32
    {
        candle_core::bail!("fattn_mma_bf16: kv_max must be I32, got {:?}", kv.dtype());
    }
    prefill_dispatch_bf16(
        q,
        k,
        v,
        mask,
        kv_max,
        scale,
        None,
        WindowBounds::NONE,
        LaunchOverride::NONE,
    )
}

/// bf16 counterpart of [`prefill_dispatch`].
#[allow(clippy::too_many_arguments)] // one launch's worth of inputs, not separable
fn prefill_dispatch_bf16(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    kv_max: Option<&Tensor>,
    scale: f32,
    causal_kv_offset: Option<usize>,
    window: WindowBounds,
    forced: LaunchOverride,
) -> Result<Tensor> {
    let padded_mask;
    let mask = match mask {
        Some(m) => {
            let seq_kv = k.dims4()?.1;
            check_mask_covers_kv(m, seq_kv, "fattn_mma_bf16")?;
            padded_mask = pad_mask_stride(m, seq_kv)?;
            Some(&padded_mask)
        },
        None => None,
    };
    #[cfg(feature = "cuda")]
    {
        cuda::fattn_mma_bf16_prefill_cuda(
            q,
            k,
            v,
            mask,
            kv_max,
            scale,
            causal_kv_offset,
            window.left,
            window.right,
            forced,
        )
    }
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    {
        rocm::fattn_mma_bf16_prefill_rocm(
            q,
            k,
            v,
            mask,
            kv_max,
            scale,
            causal_kv_offset,
            window.left,
            window.right,
            forced,
        )
    }
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        let _ = (
            q,
            k,
            v,
            mask,
            kv_max,
            scale,
            causal_kv_offset,
            window,
            forced,
        );
        candle_core::bail!("fattn_mma_bf16_prefill: requires the cuda or rocm feature")
    }
}

/// `(head_dim, gqa_ratio, seq_q, seq_kv, is_amd)` for a `[B, S, H, D]` BSHD
/// `q`/`k` pair, shared by [`mma_ncols1_for`] and [`causal_in_kernel`] so the
/// shape derivation and `num_heads_q`/`num_heads_kv` validation live in one
/// place.
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

/// Applies the same grid-underfill narrowing [`fattn_mma_prefill_cuda`]/
/// [`fattn_mma_bf16_prefill_cuda`] apply internally, so a caller that needs
/// `ncols1` to size its own tensor (namely [`mma_ncols1_for`]/
/// [`mma_ncols1_for_bf16`], for `KV_max`) never diverges from the `ncols1`
/// the kernel actually launches with. `grid_target`'s `n_sm` query is cached
/// per device, so calling it again here is cheap.
#[allow(clippy::too_many_arguments)] // one narrow_for_underfill call's worth of inputs
fn narrowed_ncols1(
    q: &Tensor,
    ncols1: usize,
    ncols2: usize,
    seq_q: usize,
    gqa_ratio: usize,
    h_kv: usize,
    batch: usize,
    is_amd: bool,
    bf16: bool,
) -> Result<usize> {
    let target = grid_target(q.device(), LaunchOverride::NONE)?;
    Ok(narrow_for_underfill(
        ncols1,
        ncols2,
        seq_q,
        gqa_ratio,
        h_kv,
        batch,
        target.n_sm,
        is_amd,
        bf16,
    ))
}

/// `(ncols1, gqa_ratio)` for a `[B, S, H, D]` BSHD `q`/`k` pair, shared by
/// [`fattn_mma_causal`]/[`fattn_mma_full`]/[`fattn_mma_windowed`] to size
/// their `KV_max` tensor before dispatching to [`fattn_mma_prefill`] (which
/// re-derives the same `ncols1` internally from the same inputs, including
/// the [`narrow_for_underfill`] grid-underfill narrowing via
/// [`narrowed_ncols1`] above).
///
/// # Errors
///
/// Returns an error if `q`/`k` aren't 4D, `num_heads_kv` is `0`, or
/// `num_heads_q` isn't a multiple of it, or if no instantiated MMA kernel
/// covers this `(head_dim, GQA ratio, seq_q)` combination (callers
/// should fall back to [`super::fattn_tile`] in that case).
fn mma_ncols1_for(q: &Tensor, k: &Tensor) -> Result<usize> {
    let (d, gqa_ratio, seq_q, _seq_kv, is_amd) = mma_shape_params(q, k)?;
    let (ncols1, ncols2) = select_mma_ncols(d, gqa_ratio, seq_q, is_amd).ok_or_else(|| {
        unsupported_config(format!(
            "fattn_mma: no instantiated kernel for head_dim={d} gqa_ratio={gqa_ratio} seq_q={seq_q}"
        ))
    })?;
    let (batch, _, h_kv, _) = k.dims4()?;
    narrowed_ncols1(
        q, ncols1, ncols2, seq_q, gqa_ratio, h_kv, batch, is_amd, false,
    )
}

/// bf16 counterpart of [`mma_ncols1_for`], used by the `fattn_mma_bf16_*`
/// entry points below.
fn mma_ncols1_for_bf16(q: &Tensor, k: &Tensor) -> Result<usize> {
    let (d, gqa_ratio, seq_q, _seq_kv, is_amd) = mma_shape_params(q, k)?;
    let (ncols1, ncols2) = select_mma_ncols_bf16(d, gqa_ratio, seq_q, is_amd).ok_or_else(|| {
        unsupported_config(format!(
            "fattn_mma_bf16: no instantiated kernel for head_dim={d} gqa_ratio={gqa_ratio} seq_q={seq_q}"
        ))
    })?;
    let (batch, _, h_kv, _) = k.dims4()?;
    narrowed_ncols1(
        q, ncols1, ncols2, seq_q, gqa_ratio, h_kv, batch, is_amd, true,
    )
}

/// Casts `mask` to F16 if it isn't already (avoiding a no-op cast's clone),
/// since every `fattn_mma_*_with_mask` entry point (F16 and native-BF16
/// alike) requires an F16 mask before dispatch.
fn ensure_f16_mask(mask: &Tensor) -> Result<Tensor> {
    if mask.dtype() == DType::F16 {
        Ok(mask.clone())
    } else {
        mask.to_dtype(DType::F16)
    }
}

/// Validates `mask`'s last (kv) dimension equals `seq_kv`, required by every
/// `fattn_mma_*_with_mask` entry point so the kernel's per-tile mask
/// indexing (`k_VKQ_0 + i` up to `seq_kv - 1`) never reads past the mask
/// buffer.
fn check_mask_covers_kv(mask: &Tensor, seq_kv: usize, what: &str) -> Result<()> {
    let mask_kv = mask.dims().last().copied().unwrap_or(0);
    if mask_kv != seq_kv {
        candle_core::bail!(
            "{what}: mask's last dimension ({mask_kv}) must equal k's seq_kv ({seq_kv})"
        );
    }
    Ok(())
}

/// Pads `mask`'s kv dimension up to a multiple of 8 `half`s (16 bytes).
/// `flash_attn_ext_{f16,bf16}_load_mask`'s fast-path loaders read a row at
/// `mask_h + j_vram*stride_mask + ...` with a vectorized (`cp_async_cg_16`
/// for F16, `half2` for both dtypes) instruction, which requires that
/// address aligned regardless of `j_vram`; that only holds for every
/// `j_vram` if `stride_mask` itself (the mask's real kv-dimension size) is a
/// multiple of 8. The extra columns are never read by any in-bounds tile
/// (the bounds-checked tail path is scalar and guarded by `i_sup`), so their
/// value doesn't matter.
fn pad_mask_stride(mask: &Tensor, seq_kv: usize) -> Result<Tensor> {
    let padded_kv = seq_kv.next_multiple_of(8);
    if padded_kv == seq_kv {
        return Ok(mask.clone());
    }
    let mut dims = mask.dims().to_vec();
    let last = dims.len() - 1;
    dims[last] = padded_kv - seq_kv;
    let pad = Tensor::zeros(dims, mask.dtype(), mask.device())?;
    Tensor::cat(&[mask, &pad], last)
}

/// Causal flash-attention: `j <= i + kv_offset`. The common case builds no
/// mask or `KV_max` tensor: the kernel masks the diagonal and bounds the KV
/// loop itself from `kv_offset`. See [`causal_needs_host_mask`] for when a
/// host mask (and `KV_max`) is still required instead.
/// Output is bitwise equal to [`fattn_mma_causal_with_mask`] given a plain
/// causal mask.
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
    if causal_needs_host_mask(seq_q, seq_kv, kv_offset) {
        let mask =
            super::fattn::build_additive_mask_f16(seq_q, seq_kv, kv_offset, None, 0, q.device())?;
        return fattn_mma_causal_with_mask(q, k, v, scale, kv_offset, &mask);
    }
    causal_in_kernel(q, k, v, scale, kv_offset)
}

/// Same as [`fattn_mma_causal`], but for a caller that already has a mask
/// shared across multiple layers in one forward pass (e.g. a decoder stack
/// whose `forward()` builds the mask once and passes it to every layer).
/// That avoids the `O(seq_q * seq_kv)` mask rebuild on every call that
/// [`fattn_mma_causal`] pays. `mask` must encode exactly
/// `j <= i + kv_offset`, covering `k`'s live `seq_kv` (any dtype, cast to
/// F16 here if needed). `KV_max` is still rebuilt per call. It is
/// `O(batch * n_tiles)`, not `O(seq^2)`, so there is no analogous cost to
/// share.
///
/// # Errors
///
/// Returns a candle error if `q`/`k`/`v`'s shapes are incompatible, if
/// `mask` isn't broadcastable to `[.., seq_q, seq_kv]`, or if any tensor op
/// fails.
pub fn fattn_mma_causal_with_mask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    kv_offset: usize,
    mask: &Tensor,
) -> Result<Tensor> {
    let (b, seq_q, _, _) = q.dims4()?;
    let mask = ensure_f16_mask(mask)?;
    let ncols1 = mma_ncols1_for(q, k)?;
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
    fattn_mma_prefill(q, k, v, Some(&mask), Some(&kv_max), scale)
}

/// Native-bf16 counterpart of [`fattn_mma_causal`]: masks in the kernel and
/// dispatches through [`fattn_mma_bf16_prefill`]'s kernels instead.
///
/// # Errors
///
/// See [`fattn_mma_bf16_causal_with_mask`].
pub fn fattn_mma_bf16_causal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    kv_offset: usize,
) -> Result<Tensor> {
    let (_, seq_q, _, _) = q.dims4()?;
    let (_, seq_kv, _, _) = k.dims4()?;
    if causal_needs_host_mask(seq_q, seq_kv, kv_offset) {
        let mask =
            super::fattn::build_additive_mask_f16(seq_q, seq_kv, kv_offset, None, 0, q.device())?;
        return fattn_mma_bf16_causal_with_mask(q, k, v, scale, kv_offset, &mask);
    }
    causal_in_kernel(q, k, v, scale, kv_offset)
}

/// Whether a causal launch must fall back to the host mask rather than let
/// the kernel bound its own KV loop from `kv_offset`. Returns `true` when
/// the causal diagonal extends past the live keys (`seq_q + kv_offset >
/// seq_kv`). [`causal_in_kernel`]'s bounds-checked tail tile may already
/// handle this case correctly, but this guard is kept as a deliberate
/// conservative safety net: the host-mask path is functionally equivalent
/// and its cost is negligible next to the risk of an out-of-bounds KV read
/// on an untested edge shape.
fn causal_needs_host_mask(seq_q: usize, seq_kv: usize, kv_offset: usize) -> bool {
    seq_q + kv_offset > seq_kv
}

/// Shared body of [`fattn_mma_causal`] and [`fattn_mma_bf16_causal`]:
/// launches `k`/`v` directly with no mask and no `KV_max`, so the kernel
/// masks the diagonal and bounds each tile's KV loop from `kv_offset`.
/// Picks the native-bf16 kernels when `k`/`v` are `BF16`, the plain F16
/// kernels when they are `F16`.
fn causal_in_kernel(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    kv_offset: usize,
) -> Result<Tensor> {
    // Same dtype contract as fattn_mma_prefill/fattn_mma_bf16_prefill: q/k/v
    // all share a single F16 or BF16 dtype.
    if q.dtype() != k.dtype() {
        candle_core::bail!(
            "fattn_mma causal: q must match k/v dtype, got q={:?} k={:?}",
            q.dtype(),
            k.dtype()
        );
    }
    if v.dtype() != k.dtype() {
        candle_core::bail!(
            "fattn_mma causal: k/v dtypes must match, got k={:?} v={:?}",
            k.dtype(),
            v.dtype()
        );
    }
    let bf16 = match k.dtype() {
        DType::F16 => false,
        DType::BF16 => true,
        other => candle_core::bail!("fattn_mma causal: k/v must be F16 or BF16, got {other:?}"),
    };
    if bf16 {
        prefill_dispatch_bf16(
            q,
            k,
            v,
            None,
            None,
            scale,
            Some(kv_offset),
            WindowBounds::NONE,
            LaunchOverride::NONE,
        )
    } else {
        prefill_dispatch(
            q,
            k,
            v,
            None,
            None,
            scale,
            Some(kv_offset),
            WindowBounds::NONE,
            LaunchOverride::NONE,
        )
    }
}

/// Whether a windowed launch must fall back to the host mask rather than
/// let the kernel bound its own KV loop from `kv_offset`/`window_right`.
/// Returns `true` when the window's right edge extends past the live keys
/// (`seq_q + kv_offset + window_right > seq_kv`), the same conservative
/// safety net as [`causal_needs_host_mask`] (which this reduces to when
/// `window_right == 0`).
fn windowed_needs_host_mask(
    seq_q: usize,
    seq_kv: usize,
    kv_offset: usize,
    window_right: usize,
) -> bool {
    seq_q + kv_offset + window_right > seq_kv
}

/// Shared body of [`fattn_mma_windowed`] and [`fattn_mma_bf16_windowed`]:
/// launches `k`/`v` directly with no mask and no `KV_max`, so the kernel
/// masks both edges of the sliding window and bounds each tile's KV loop
/// itself, the windowed counterpart of [`causal_in_kernel`]. Picks the
/// native-bf16 kernels when `k`/`v` are `BF16`, the plain F16 kernels when
/// they are `F16`.
fn windowed_in_kernel(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    kv_offset: usize,
    window_left: usize,
    window_right: usize,
) -> Result<Tensor> {
    // Same dtype contract as causal_in_kernel.
    if q.dtype() != k.dtype() {
        candle_core::bail!(
            "fattn_mma windowed: q must match k/v dtype, got q={:?} k={:?}",
            q.dtype(),
            k.dtype()
        );
    }
    if v.dtype() != k.dtype() {
        candle_core::bail!(
            "fattn_mma windowed: k/v dtypes must match, got k={:?} v={:?}",
            k.dtype(),
            v.dtype()
        );
    }
    let bf16 = match k.dtype() {
        DType::F16 => false,
        DType::BF16 => true,
        other => candle_core::bail!("fattn_mma windowed: k/v must be F16 or BF16, got {other:?}"),
    };
    let window = WindowBounds {
        left: to_i32(window_left, "window_left")?,
        right: to_i32(window_right, "window_right")?,
    };
    if bf16 {
        prefill_dispatch_bf16(
            q,
            k,
            v,
            None,
            None,
            scale,
            Some(kv_offset),
            window,
            LaunchOverride::NONE,
        )
    } else {
        prefill_dispatch(
            q,
            k,
            v,
            None,
            None,
            scale,
            Some(kv_offset),
            window,
            LaunchOverride::NONE,
        )
    }
}

/// Native-bf16 counterpart of [`fattn_mma_causal_with_mask`]: dispatches
/// through [`fattn_mma_bf16_prefill`]'s kernels instead.
///
/// # Errors
///
/// See [`fattn_mma_bf16_prefill`].
pub fn fattn_mma_bf16_causal_with_mask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    kv_offset: usize,
    mask: &Tensor,
) -> Result<Tensor> {
    let (b, seq_q, _, _) = q.dims4()?;
    let mask = ensure_f16_mask(mask)?;
    let ncols1 = mma_ncols1_for_bf16(q, k)?;
    let seq_kv = k.dims4()?.1;
    // kv_max_cap: see fattn_mma_causal_with_mask's comment.
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
    fattn_mma_bf16_prefill(q, k, v, Some(&mask), Some(&kv_max), scale)
}

/// Sliding-window flash-attention: `i + kv_offset - window_left <= j <= i +
/// kv_offset + window_right`. The common case builds no mask or `KV_max`
/// tensor: the kernel masks both window edges and bounds the KV loop
/// itself from `kv_offset`/`window_left`/`window_right`, the windowed
/// counterpart of [`fattn_mma_causal`]. See [`windowed_needs_host_mask`]
/// for when a host mask (and `KV_max`) is still required instead. Output is
/// bitwise equal to [`fattn_mma_windowed_with_mask`] given a mask encoding
/// the same window.
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
    if !windowed_needs_host_mask(seq_q, seq_kv, kv_offset, window_right) {
        return windowed_in_kernel(q, k, v, scale, kv_offset, window_left, window_right);
    }
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
    let mask = ensure_f16_mask(mask)?;
    let ncols1 = mma_ncols1_for(q, k)?;
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
    fattn_mma_prefill(q, k, v, Some(&mask), Some(&kv_max), scale)
}

/// Native-bf16 counterpart of [`fattn_mma_windowed`]: dispatches through
/// [`fattn_mma_bf16_prefill`]'s kernels instead. See
/// [`fattn_mma_bf16_windowed_with_mask`]'s doc comment for the mask-based
/// fallback.
///
/// # Errors
///
/// See [`fattn_mma_bf16_windowed_with_mask`].
pub fn fattn_mma_bf16_windowed(
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
    if !windowed_needs_host_mask(seq_q, seq_kv, kv_offset, window_right) {
        return windowed_in_kernel(q, k, v, scale, kv_offset, window_left, window_right);
    }
    let mask = super::fattn::build_additive_mask_f16(
        seq_q,
        seq_kv,
        kv_offset,
        Some(window_left),
        window_right,
        q.device(),
    )?;
    fattn_mma_bf16_windowed_with_mask(q, k, v, scale, kv_offset, window_right, &mask)
}

/// Native-bf16 counterpart of [`fattn_mma_windowed_with_mask`]: dispatches
/// through [`fattn_mma_bf16_prefill`]'s kernels instead.
///
/// # Errors
///
/// See [`fattn_mma_bf16_prefill`].
pub fn fattn_mma_bf16_windowed_with_mask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    kv_offset: usize,
    window_right: usize,
    mask: &Tensor,
) -> Result<Tensor> {
    let (b, seq_q, _, _) = q.dims4()?;
    let mask = ensure_f16_mask(mask)?;
    let ncols1 = mma_ncols1_for_bf16(q, k)?;
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
    fattn_mma_bf16_prefill(q, k, v, Some(&mask), Some(&kv_max), scale)
}

/// Non-causal (full, bidirectional) flash-attention: every query attends to
/// every key. Passes a real all-zero mask rather than `None` (see this
/// module's doc comment), since `None` is only safe when the
/// internally-selected `ncols2` resolves to `1`, which this entry point
/// cannot guarantee for every `(gqa_ratio, seq_kv)` shape. No `KV_max`:
/// nothing is masked, so there is no tail of the KV loop to skip.
///
/// # Errors
///
/// See [`fattn_mma_prefill`].
pub fn fattn_mma_full(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let (_, seq_q, _, _) = q.dims4()?;
    let (_, seq_kv, _, _) = k.dims4()?;
    let mask = Tensor::zeros((1, 1, seq_q, seq_kv), DType::F16, q.device())?;
    fattn_mma_prefill(q, k, v, Some(&mask), None, scale)
}

/// Native-bf16 counterpart of [`fattn_mma_full`].
///
/// # Errors
///
/// See [`fattn_mma_bf16_prefill`].
pub fn fattn_mma_bf16_full(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let (_, seq_q, _, _) = q.dims4()?;
    let (_, seq_kv, _, _) = k.dims4()?;
    let mask = Tensor::zeros((1, 1, seq_q, seq_kv), DType::F16, q.device())?;
    fattn_mma_bf16_prefill(q, k, v, Some(&mask), None, scale)
}

#[cfg(feature = "cuda")]
mod cuda {
    use candle_core::Storage;
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};

    use super::{
        FixupPlan, GridTarget, KernelVariant, LaunchGeometry, LaunchOverride, LaunchScalars,
        LaunchShape, MmaConfigSet, TileCost,
    };
    use crate::ops::fused_ops::fattn::{
        fixup_kernel_name, narrow_for_underfill, select_mma_ncols, select_mma_ncols_bf16,
        unsupported_config, validate_fattn_bshd,
    };
    use candle_core::{Device, Result, Tensor};

    mod ptx {
        include!(concat!(env!("OUT_DIR"), "/crane_fattn_ptx.rs"));
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

    /// The long-KV set only exists at `head_dim=256`, see
    /// [`crate::ops::fused_ops::fattn::has_long_kv_set`].
    fn ptx_for_bf16(dkq: usize, set: MmaConfigSet) -> Result<&'static str> {
        if set == MmaConfigSet::LongKv {
            return match dkq {
                256 => Ok(ptx::FATTN_MMA_HD256_LONG_KV_BF16),
                _ => candle_core::bail!("fattn_mma_bf16: no long-KV PTX for head_dim {dkq}"),
            };
        }
        Ok(match dkq {
            64 => ptx::FATTN_MMA_HD64_BF16,
            128 => ptx::FATTN_MMA_HD128_BF16,
            256 => ptx::FATTN_MMA_HD256_BF16,
            512 => ptx::FATTN_MMA_HD512_BF16,
            _ => candle_core::bail!("fattn_mma_bf16: unsupported head_dim {dkq}"),
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

    fn bf16_slice<'a>(
        storage: &'a Storage,
        what: &str,
    ) -> Result<&'a candle_core::cuda_backend::cudarc::driver::CudaSlice<half::bf16>> {
        match storage {
            Storage::Cuda(c) => match &c.slice {
                CudaStorageSlice::BF16(s) => Ok(s),
                _ => candle_core::bail!("fattn_mma_bf16: {what} must be BF16"),
            },
            _ => candle_core::bail!("fattn_mma_bf16: {what} must be a cuda tensor"),
        }
    }

    /// Raises `func`'s dynamic shared memory limit to `shared_mem_bytes`.
    ///
    /// `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` is required
    /// whenever a kernel requests more than the default static 48 KiB
    /// per-block limit; `cudarc`'s `launch`/`launch_builder` never calls
    /// it, it only forwards `shared_mem_bytes` to `cuLaunchKernel`, which
    /// fails with `CUDA_ERROR_INVALID_VALUE` past that limit. llama.cpp
    /// (this kernel's origin) raises it unconditionally for every launch,
    /// cached per device (`common.cuh`'s `CUDA_SET_SHARED_MEMORY_LIMIT`).
    /// `head_dim=512` and GQA-narrow F16 tiles with 2-stage `cp.async`
    /// both exceed 48 KiB; `head_dim<=256` with 1 GQA head per tile stays
    /// under it, which is why this never surfaced before real CUDA
    /// hardware ran these shapes.
    ///
    /// Always raises; does not cache by `func`'s raw `CUfunction` pointer.
    /// An earlier version did, keyed on that pointer, but
    /// `CudaDevice::new` (`candle_core`) opens a brand new `CudaContext`
    /// every call with no per-ordinal caching, and `test_gpu_device()`
    /// calls it fresh for every GPU test; once a context is torn down its
    /// module/function addresses can be reused by the next one. A process-
    /// wide cache keyed on that address could then find a stale "already
    /// raised" entry left by a since-dropped context's unrelated, smaller
    /// kernel, and skip the real call for a function whose limit was never
    /// actually raised, launching with an insufficient shared-memory limit
    /// and failing with `CUDA_ERROR_INVALID_VALUE`. `cuFuncSetAttribute` is
    /// a cheap, idempotent driver call (no launch, no sync), so paying it
    /// on every dispatch is the safe default.
    fn raise_dynamic_shared_mem_limit(
        func: &candle_core::cuda_backend::cudarc::driver::CudaFunction,
        shared_mem_bytes: u32,
    ) -> Result<()> {
        use candle_core::cuda_backend::cudarc::driver::result::function::set_function_attribute;
        use candle_core::cuda_backend::cudarc::driver::sys::CUfunction_attribute;

        let bytes = super::to_i32(shared_mem_bytes as usize, "shared_mem_bytes")?;
        // SAFETY: `func` is a function handle freshly resolved from a
        // loaded module (`get_or_load_custom_func`), valid for this call.
        unsafe {
            set_function_attribute(
                func.cu_function(),
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                bytes,
            )
        }
        .w()
    }

    #[allow(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        clippy::too_many_arguments
    )]
    pub(super) fn fattn_mma_prefill_cuda(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        kv_max: Option<&Tensor>,
        scale: f32,
        causal_kv_offset: Option<usize>,
        window_left: i32,
        window_right: i32,
        forced: LaunchOverride,
    ) -> Result<Tensor> {
        let (q_s, q_l) = q.storage_and_layout();
        let (k_s, k_l) = k.storage_and_layout();
        let (v_s, v_l) = v.storage_and_layout();
        let (b, seq_q, h_q, d, seq_kv, h_kv) = validate_fattn_bshd(q_l, k_l, v_l, q.dtype())?;
        let gqa_ratio = h_q / h_kv;
        let dev = q.device().as_cuda_device()?.clone();
        let is_amd = false;

        let Some((mut ncols1, ncols2)) = select_mma_ncols(d, gqa_ratio, seq_q, is_amd) else {
            return Err(unsupported_config(format!(
                "fattn_mma: no instantiated kernel for head_dim={d} gqa_ratio={gqa_ratio} seq_q={seq_q}"
            )));
        };
        // See fattn_mma_prefill's doc comment: a plain mask=None stays
        // limited to ncols2 == 1. The in-kernel causal entries pass no mask
        // and are exempt.
        if mask.is_none() && causal_kv_offset.is_none() && ncols2 != 1 {
            return Err(unsupported_config(format!(
                "fattn_mma: mask=None requires ncols2 == 1 (got ncols2={ncols2}); \
                 pass an explicit all-zero mask for unmasked attention with this GQA shape"
            )));
        }
        let shape = LaunchShape {
            seq_q,
            seq_kv,
            h_kv,
            batch: b,
            gqa_ratio,
        };
        let target = super::grid_target(q.device(), forced)?;
        ncols1 = narrow_for_underfill(
            ncols1,
            ncols2,
            seq_q,
            gqa_ratio,
            h_kv,
            b,
            target.n_sm,
            is_amd,
            false,
        );
        let kv = super::KvExtent::new(seq_kv, h_kv, causal_kv_offset);
        let (cfg, set) = super::launch_config((d, ncols1 * ncols2), kv, (is_amd, false), forced)?;
        let tail = ncols2 > 1 && !seq_kv.is_multiple_of(cfg.nbatch_fa);
        let (kernel_name, module_name) =
            super::cached_mma_names(d, ncols1, ncols2, (false, tail), set);
        let func = dev.get_or_load_custom_func(kernel_name, module_name, ptx_for(d)?)?;
        let variant = KernelVariant {
            dkq: d,
            ncols1,
            ncols2,
            is_amd,
            bf16: false,
        };
        let geometry = LaunchGeometry::new(
            &variant,
            &shape,
            &target,
            cfg,
            TileCost::of(causal_kv_offset, kv_max),
            |_, _| {
                // CUDA's real occupancy query is unverified here (no CUDA
                // device in this environment); keep the config table's
                // guess until it is checked on real hardware.
                None
            },
        )?;

        let q_sl = f16_slice(&q_s, "q")?.slice(q_l.start_offset()..);
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
            q.dtype(),
        )?;

        let causal_kv_offset = super::kernel_causal_offset(causal_kv_offset)?;
        let heavy_first = i32::from(geometry.heavy_first);

        let n_out = b * seq_q * h_q * d;
        let dst = unsafe { dev.alloc::<half::f16>(n_out) }?;
        // The stream-K fixup scratch, written in full by the main kernel
        // before the fixup kernel reads it. Null when no tile is split.
        let scratch = geometry
            .fixup
            .as_ref()
            .map(|f| unsafe { dev.alloc::<f32>(f.scratch_len) })
            .transpose()?;

        raise_dynamic_shared_mem_limit(&func, geometry.shared_mem_bytes)?;
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
        // provided" (`if (mask_h)` in `fattn_mma_f16.cuh`), the
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
        match &scratch {
            Some(s) => builder.arg(s),
            None => builder.arg(&0u64),
        };
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
        builder.arg(&causal_kv_offset);
        builder.arg(&heavy_first);
        builder.arg(&window_left);
        builder.arg(&window_right);
        // SAFETY: the argument list above matches
        // `crane_fattn_mma_f16_d{d}_v{d}_c{ncols1}_s{ncols2}`'s `extern "C"`
        // signature exactly (`DECL_FATTN_MMA_F16_CASE` in
        // `fattn_mma_f16.cuh`). q_sl/k_sl/v_sl/mask_sl/kv_max_sl's underlying
        // storage stays valid until the launch completes because `q`/`k`/
        // `v`/`mask`/`kv_max` outlive this call; `cfg`'s grid covers every
        // output element (`geometry`'s tile counts).
        unsafe { builder.launch(cfg) }.w()?;

        if let (Some(f), Some(scratch)) = (&geometry.fixup, &scratch) {
            let fixup_name = fixup_kernel_name(d, ncols1, ncols2, false);
            let fixup_func = dev.get_or_load_custom_func(&fixup_name, module_name, ptx_for(d)?)?;
            let mut fixup_builder = fixup_func.builder();
            fixup_builder.arg(&dst);
            fixup_builder.arg(scratch);
            fixup_builder.arg(&f.ne01);
            fixup_builder.arg(&f.ne02);
            fixup_builder.arg(&f.gqa_ratio);
            fixup_builder.arg(&f.total_work);
            fixup_builder.arg(&f.fd_k_j_z_ne12);
            fixup_builder.arg(&f.fd_k_j_z);
            fixup_builder.arg(&f.fd_k_j);
            fixup_builder.arg(&f.fd_k);
            // SAFETY: the argument list matches `crane_fattn_fixup_*`'s
            // `extern "C"` signature in `fattn_fixup.cuh`. `dst` and
            // `scratch` outlive the launch, and the grid covers one block
            // per (main block, Q row, GQA head).
            unsafe {
                fixup_builder.launch(LaunchConfig {
                    grid_dim: f.grid,
                    block_dim: f.block,
                    shared_mem_bytes: 0,
                })
            }
            .w()?;
        }

        let dst = CudaStorage {
            slice: CudaStorageSlice::F16(dst),
            device: dev,
        };
        Ok(Tensor::from_storage(
            Storage::Cuda(dst),
            (b, seq_q, h_q, d),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    /// bf16 counterpart of [`fattn_mma_prefill_cuda`]: `k`/`v` stay BF16
    /// throughout (loaded via [`bf16_slice`], not [`f16_slice`]), the PTX
    /// module/kernel name come from the `_bf16` instantiation files, and
    /// the kernel selection uses [`select_mma_ncols_bf16`] (never resolves
    /// to `ncols1*ncols2 == 8`, see that function's doc comment). The
    /// kernel writes the output in BF16 instead of F16.
    #[allow(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        clippy::too_many_arguments
    )]
    pub(super) fn fattn_mma_bf16_prefill_cuda(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        kv_max: Option<&Tensor>,
        scale: f32,
        causal_kv_offset: Option<usize>,
        window_left: i32,
        window_right: i32,
        forced: LaunchOverride,
    ) -> Result<Tensor> {
        let (q_s, q_l) = q.storage_and_layout();
        let (k_s, k_l) = k.storage_and_layout();
        let (v_s, v_l) = v.storage_and_layout();
        let (b, seq_q, h_q, d, seq_kv, h_kv) = validate_fattn_bshd(q_l, k_l, v_l, q.dtype())?;
        let gqa_ratio = h_q / h_kv;
        let dev = q.device().as_cuda_device()?.clone();
        let is_amd = false;

        let Some((mut ncols1, ncols2)) = select_mma_ncols_bf16(d, gqa_ratio, seq_q, is_amd) else {
            return Err(unsupported_config(format!(
                "fattn_mma_bf16: no instantiated kernel for head_dim={d} gqa_ratio={gqa_ratio} seq_q={seq_q}"
            )));
        };
        // See fattn_mma_prefill_cuda's identical check. The in-kernel
        // causal entries pass no mask and are exempt.
        if mask.is_none() && causal_kv_offset.is_none() && ncols2 != 1 {
            return Err(unsupported_config(format!(
                "fattn_mma_bf16: mask=None requires ncols2 == 1 (got ncols2={ncols2}); \
                 pass an explicit all-zero mask for unmasked attention with this GQA shape"
            )));
        }
        let shape = LaunchShape {
            seq_q,
            seq_kv,
            h_kv,
            batch: b,
            gqa_ratio,
        };
        let target = super::grid_target(q.device(), forced)?;
        ncols1 = narrow_for_underfill(
            ncols1,
            ncols2,
            seq_q,
            gqa_ratio,
            h_kv,
            b,
            target.n_sm,
            is_amd,
            true,
        );
        let kv = super::KvExtent::new(seq_kv, h_kv, causal_kv_offset);
        let (cfg, set) = super::launch_config((d, ncols1 * ncols2), kv, (is_amd, true), forced)?;
        let tail = ncols2 > 1 && !seq_kv.is_multiple_of(cfg.nbatch_fa);
        let (kernel_name, module_name) =
            super::cached_mma_names(d, ncols1, ncols2, (true, tail), set);
        let func = dev.get_or_load_custom_func(kernel_name, module_name, ptx_for_bf16(d, set)?)?;
        let variant = KernelVariant {
            dkq: d,
            ncols1,
            ncols2,
            is_amd,
            bf16: true,
        };
        let geometry = LaunchGeometry::new(
            &variant,
            &shape,
            &target,
            cfg,
            TileCost::of(causal_kv_offset, kv_max),
            |_, _| {
                // CUDA's real occupancy query is unverified here (no CUDA
                // device in this environment); keep the config table's
                // guess until it is checked on real hardware.
                None
            },
        )?;

        let q_sl = bf16_slice(&q_s, "q")?.slice(q_l.start_offset()..);
        let k_sl = bf16_slice(&k_s, "k")?.slice(k_l.start_offset()..);
        let v_sl = bf16_slice(&v_s, "v")?.slice(v_l.start_offset()..);

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
                    candle_core::bail!("fattn_mma_bf16: kv_max must be a cuda tensor");
                };
                let CudaStorageSlice::I32(sl) = &c.slice else {
                    candle_core::bail!("fattn_mma_bf16: kv_max must be I32");
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
            q.dtype(),
        )?;

        let causal_kv_offset = super::kernel_causal_offset(causal_kv_offset)?;
        let heavy_first = i32::from(geometry.heavy_first);

        let n_out = b * seq_q * h_q * d;
        let dst = unsafe { dev.alloc::<half::bf16>(n_out) }?;
        // The stream-K fixup scratch, written in full by the main kernel
        // before the fixup kernel reads it. Null when no tile is split.
        let scratch = geometry
            .fixup
            .as_ref()
            .map(|f| unsafe { dev.alloc::<f32>(f.scratch_len) })
            .transpose()?;

        raise_dynamic_shared_mem_limit(&func, geometry.shared_mem_bytes)?;
        let cfg = LaunchConfig {
            grid_dim: geometry.grid,
            block_dim: geometry.block,
            shared_mem_bytes: geometry.shared_mem_bytes,
        };

        let mut builder = func.builder();
        builder.arg(&q_sl);
        builder.arg(&k_sl);
        builder.arg(&v_sl);
        // SAFETY: see fattn_mma_prefill_cuda's identical SAFETY comment on
        // the mask/kv_max null-pointer convention.
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
        match &scratch {
            Some(s) => builder.arg(s),
            None => builder.arg(&0u64),
        };
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
        builder.arg(&causal_kv_offset);
        builder.arg(&heavy_first);
        builder.arg(&window_left);
        builder.arg(&window_right);
        // SAFETY: the argument list above matches
        // `crane_fattn_mma_bf16_d{d}_v{d}_c{ncols1}_s{ncols2}`'s `extern "C"`
        // signature exactly (`DECL_FATTN_MMA_BF16_CASE` in
        // `fattn_mma_bf16.cuh`). q_sl/k_sl/v_sl/mask_sl/kv_max_sl's underlying
        // storage stays valid until the launch completes because `q`/`k`/
        // `v`/`mask`/`kv_max` outlive this call; `cfg`'s grid covers every
        // output element (`geometry`'s tile counts).
        unsafe { builder.launch(cfg) }.w()?;

        if let (Some(f), Some(scratch)) = (&geometry.fixup, &scratch) {
            let fixup_name = fixup_kernel_name(d, ncols1, ncols2, true);
            let fixup_func =
                dev.get_or_load_custom_func(&fixup_name, module_name, ptx_for_bf16(d, set)?)?;
            let mut fixup_builder = fixup_func.builder();
            fixup_builder.arg(&dst);
            fixup_builder.arg(scratch);
            fixup_builder.arg(&f.ne01);
            fixup_builder.arg(&f.ne02);
            fixup_builder.arg(&f.gqa_ratio);
            fixup_builder.arg(&f.total_work);
            fixup_builder.arg(&f.fd_k_j_z_ne12);
            fixup_builder.arg(&f.fd_k_j_z);
            fixup_builder.arg(&f.fd_k_j);
            fixup_builder.arg(&f.fd_k);
            // SAFETY: the argument list matches `crane_fattn_fixup_*`'s
            // `extern "C"` signature in `fattn_fixup.cuh`. `dst` and
            // `scratch` outlive the launch, and the grid covers one block
            // per (main block, Q row, GQA head).
            unsafe {
                fixup_builder.launch(LaunchConfig {
                    grid_dim: f.grid,
                    block_dim: f.block,
                    shared_mem_bytes: 0,
                })
            }
            .w()?;
        }

        let dst = CudaStorage {
            slice: CudaStorageSlice::BF16(dst),
            device: dev,
        };
        Ok(Tensor::from_storage(
            Storage::Cuda(dst),
            (b, seq_q, h_q, d),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    /// The device facts for the stream-K grid choice: the SM count of
    /// `device`, used by [`narrow_for_underfill`] to detect a grid that
    /// leaves SMs idle. `n_sm` is a static fact about a device for the
    /// process's lifetime, so it is cached per gpu_id rather than queried
    /// on every dispatch, mirroring the ROCm `grid_target`.
    pub(super) fn grid_target(device: &Device, forced: LaunchOverride) -> Result<GridTarget> {
        use std::collections::HashMap;
        use std::sync::{OnceLock, RwLock};

        use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;

        static CACHE: OnceLock<RwLock<HashMap<usize, usize>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));

        let candle_core::DeviceLocation::Cuda { gpu_id } = device.location() else {
            candle_core::bail!("grid_target: not a cuda device");
        };
        if let Some(&n_sm) = cache.read().unwrap().get(&gpu_id) {
            return Ok(GridTarget { n_sm, forced });
        }
        let stream = device.as_cuda_device()?.cuda_stream();
        let n_sm = stream
            .context()
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .w()?
            .try_into()
            .unwrap_or(0);
        cache.write().unwrap().insert(gpu_id, n_sm);
        Ok(GridTarget { n_sm, forced })
    }
}

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
mod rocm {
    use std::ffi::c_void;

    use candle_core::rocm_backend::{SendSyncDeviceMemory, rocm_rs};
    use candle_core::{DType, Device, Result, Tensor};

    use super::{
        FixupPlan, GridTarget, KernelVariant, LaunchGeometry, LaunchOverride, LaunchScalars,
        LaunchShape, MmaConfigSet, TileCost,
    };
    use crate::ops::fused_ops::fattn::{
        fixup_kernel_name, narrow_for_underfill, select_mma_ncols, select_mma_ncols_bf16,
        unsupported_config, validate_fattn_bshd,
    };
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
        let fixup = include_str!("../../../kernels/cuda/fattn/fattn_fixup.cuh")
            .replace("#include \"fattn_common.cuh\"", "");
        let swizzle = include_str!("../../../kernels/cuda/fattn/swizzle.cuh")
            .replace("#include \"crane_fattn_shim.cuh\"", "")
            .replace("#include \"mma.cuh\"", "");
        let mma_f16 = include_str!("../../../kernels/cuda/fattn/fattn_mma_f16.cuh");
        let mma_f16 = splice_once(mma_f16, "#include \"crane_fattn_shim.cuh\"", shim);
        let mma_f16 = splice_once(&mma_f16, "#include \"cp_async.cuh\"", &cp_async);
        let mma_f16 = splice_once(&mma_f16, "#include \"mma.cuh\"", &mma_h);
        let mma_f16 = splice_once(&mma_f16, "#include \"fattn_common.cuh\"", &common);
        let mma_f16 = splice_once(&mma_f16, "#include \"fattn_fixup.cuh\"", &fixup);
        let mma_f16 = splice_once(&mma_f16, "#include \"swizzle.cuh\"", &swizzle);
        splice_once(instance_source, "#include \"fattn_mma_f16.cuh\"", &mma_f16)
    }

    fn source_for(dkq: usize, set: MmaConfigSet) -> Result<&'static str> {
        use std::sync::OnceLock;
        macro_rules! source_once {
            ($file:literal) => {{
                static SOURCE: OnceLock<String> = OnceLock::new();
                SOURCE.get_or_init(|| rocm_mma_source(include_str!($file)))
            }};
        }
        if set == MmaConfigSet::LongKv {
            return Ok(match dkq {
                256 => source_once!("../../../kernels/cuda/fattn/fattn_mma_hd256_long_kv.hip"),
                _ => candle_core::bail!("fattn_mma: no long-KV kernel set for head_dim {dkq}"),
            });
        }
        Ok(match dkq {
            64 => source_once!("../../../kernels/cuda/fattn/fattn_mma_hd64.cu"),
            128 => source_once!("../../../kernels/cuda/fattn/fattn_mma_hd128.cu"),
            256 => source_once!("../../../kernels/cuda/fattn/fattn_mma_hd256.cu"),
            512 => source_once!("../../../kernels/cuda/fattn/fattn_mma_hd512.cu"),
            _ => candle_core::bail!("fattn_mma: unsupported head_dim {dkq}"),
        })
    }

    /// bf16 counterpart of [`rocm_mma_source`]: `fattn_mma_bf16.cuh` now
    /// shares `fattn_mma_f16.cuh`'s `cp_async.cuh`/`swizzle.cuh` pipeline
    /// (both NVIDIA-only; stub out under HIP, see those headers' module
    /// docs), so the splice order matches [`rocm_mma_source`]'s. Same
    /// splice-once/strip-elsewhere rule: `crane_fattn_shim.cuh` is included
    /// from five places (`mma.cuh`, `cp_async.cuh`, `swizzle.cuh`,
    /// `fattn_common.cuh`, and `fattn_mma_bf16.cuh` itself), so each leaf's
    /// own copy of that `#include` (and, for `swizzle.cuh`, its `mma.cuh`
    /// include) is stripped before splicing, leaving exactly one copy of
    /// each header's contents in the final source.
    fn rocm_mma_bf16_source(instance_source: &'static str) -> String {
        let shim = include_str!("../../../kernels/cuda/fattn/crane_fattn_shim.cuh");
        let mma_h = include_str!("../../../kernels/cuda/fattn/mma.cuh")
            .replace("#include \"crane_fattn_shim.cuh\"", "");
        let cp_async = include_str!("../../../kernels/cuda/fattn/cp_async.cuh")
            .replace("#include \"crane_fattn_shim.cuh\"", "");
        let common = include_str!("../../../kernels/cuda/fattn/fattn_common.cuh")
            .replace("#include \"crane_fattn_shim.cuh\"", "");
        let fixup = include_str!("../../../kernels/cuda/fattn/fattn_fixup.cuh")
            .replace("#include \"fattn_common.cuh\"", "");
        let swizzle = include_str!("../../../kernels/cuda/fattn/swizzle.cuh")
            .replace("#include \"crane_fattn_shim.cuh\"", "")
            .replace("#include \"mma.cuh\"", "");
        let mma_bf16 = include_str!("../../../kernels/cuda/fattn/fattn_mma_bf16.cuh");
        let mma_bf16 = splice_once(mma_bf16, "#include \"crane_fattn_shim.cuh\"", shim);
        let mma_bf16 = splice_once(&mma_bf16, "#include \"cp_async.cuh\"", &cp_async);
        let mma_bf16 = splice_once(&mma_bf16, "#include \"mma.cuh\"", &mma_h);
        let mma_bf16 = splice_once(&mma_bf16, "#include \"fattn_common.cuh\"", &common);
        let mma_bf16 = splice_once(&mma_bf16, "#include \"fattn_fixup.cuh\"", &fixup);
        let mma_bf16 = splice_once(&mma_bf16, "#include \"swizzle.cuh\"", &swizzle);
        splice_once(
            instance_source,
            "#include \"fattn_mma_bf16.cuh\"",
            &mma_bf16,
        )
    }

    fn source_for_bf16(dkq: usize, set: MmaConfigSet) -> Result<&'static str> {
        use std::sync::OnceLock;
        macro_rules! source_once_bf16 {
            ($file:literal) => {{
                static SOURCE: OnceLock<String> = OnceLock::new();
                SOURCE.get_or_init(|| rocm_mma_bf16_source(include_str!($file)))
            }};
        }
        if set == MmaConfigSet::LongKv {
            return Ok(match dkq {
                256 => {
                    source_once_bf16!(
                        "../../../kernels/cuda/fattn/fattn_mma_hd256_long_kv_bf16.hip"
                    )
                },
                _ => candle_core::bail!("fattn_mma_bf16: no long-KV kernel set for head_dim {dkq}"),
            });
        }
        Ok(match dkq {
            64 => source_once_bf16!("../../../kernels/cuda/fattn/fattn_mma_hd64_bf16.cu"),
            128 => source_once_bf16!("../../../kernels/cuda/fattn/fattn_mma_hd128_bf16.cu"),
            256 => source_once_bf16!("../../../kernels/cuda/fattn/fattn_mma_hd256_bf16.cu"),
            512 => source_once_bf16!("../../../kernels/cuda/fattn/fattn_mma_hd512_bf16.cu"),
            _ => candle_core::bail!("fattn_mma_bf16: unsupported head_dim {dkq}"),
        })
    }

    #[allow(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        clippy::too_many_arguments
    )]
    pub(super) fn fattn_mma_prefill_rocm(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        kv_max: Option<&Tensor>,
        scale: f32,
        causal_kv_offset: Option<usize>,
        window_left: i32,
        window_right: i32,
        forced: LaunchOverride,
    ) -> Result<Tensor> {
        let (q_s, q_l) = q.storage_and_layout();
        let (k_s, k_l) = k.storage_and_layout();
        let (v_s, v_l) = v.storage_and_layout();
        let (b, seq_q, h_q, d, seq_kv, h_kv) = validate_fattn_bshd(q_l, k_l, v_l, q.dtype())?;
        let gqa_ratio = h_q / h_kv;
        let dev = q.device().as_rocm_device()?.clone();
        let is_amd = true;

        let Some((mut ncols1, ncols2)) = select_mma_ncols(d, gqa_ratio, seq_q, is_amd) else {
            return Err(unsupported_config(format!(
                "fattn_mma: no instantiated kernel for head_dim={d} gqa_ratio={gqa_ratio} seq_q={seq_q}"
            )));
        };
        // See fattn_mma_prefill's doc comment: a plain mask=None stays
        // limited to ncols2 == 1. The in-kernel causal entries pass no mask
        // and are exempt.
        if mask.is_none() && causal_kv_offset.is_none() && ncols2 != 1 {
            return Err(unsupported_config(format!(
                "fattn_mma: mask=None requires ncols2 == 1 (got ncols2={ncols2}); \
                 pass an explicit all-zero mask for unmasked attention with this GQA shape"
            )));
        }
        let shape = LaunchShape {
            seq_q,
            seq_kv,
            h_kv,
            batch: b,
            gqa_ratio,
        };
        let target = super::grid_target(q.device(), forced)?;
        ncols1 = narrow_for_underfill(
            ncols1,
            ncols2,
            seq_q,
            gqa_ratio,
            h_kv,
            b,
            target.n_sm,
            is_amd,
            false,
        );
        let kv = super::KvExtent::new(seq_kv, h_kv, causal_kv_offset);
        let (cfg, set) = super::launch_config((d, ncols1 * ncols2), kv, (is_amd, false), forced)?;
        let tail = ncols2 > 1 && !seq_kv.is_multiple_of(cfg.nbatch_fa);
        let (kernel_name, module_name) =
            super::cached_mma_names(d, ncols1, ncols2, (false, tail), set);
        let source = source_for(d, set)?;
        let variant = KernelVariant {
            dkq: d,
            ncols1,
            ncols2,
            is_amd,
            bf16: false,
        };
        let geometry = LaunchGeometry::new(
            &variant,
            &shape,
            &target,
            cfg,
            TileCost::of(causal_kv_offset, kv_max),
            |threads, shared_mem_bytes| {
                real_occupancy(
                    &dev,
                    module_name,
                    kernel_name,
                    source,
                    threads,
                    shared_mem_bytes,
                )
            },
        )?;

        let q_ptr = rocm::strided_device_ptr(&q_s, q_l, DType::F16, "fattn_mma q")?;
        let k_ptr = rocm::strided_device_ptr(&k_s, k_l, DType::F16, "fattn_mma k")?;
        let v_ptr = rocm::strided_device_ptr(&v_s, v_l, DType::F16, "fattn_mma v")?;

        // The mask stays on the contiguous device_ptr. Callers make it
        // contiguous, and its strides are not layout-checked.
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
            q.dtype(),
        )?;

        let causal_kv_offset = super::kernel_causal_offset(causal_kv_offset)?;
        let heavy_first = i32::from(geometry.heavy_first);

        let n_out = b * seq_q * h_q * d;
        let dst = dev.alloc::<half::f16>(n_out)?;
        let dst_ptr = dst.as_ptr();
        // The stream-K fixup scratch, written in full by the main kernel
        // before the fixup kernel reads it. Null when no tile is split.
        let scratch = geometry
            .fixup
            .as_ref()
            .map(|f| dev.alloc::<f32>(f.scratch_len))
            .transpose()?;
        let dst_meta_ptr: *mut c_void = scratch
            .as_ref()
            .map_or(std::ptr::null_mut(), SendSyncDeviceMemory::as_ptr);
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
            rocm::arg(&causal_kv_offset),
            rocm::arg(&heavy_first),
            rocm::arg(&window_left),
            rocm::arg(&window_right),
        ];

        // SAFETY: the argument list above matches
        // `crane_fattn_mma_f16_d{d}_v{d}_c{ncols1}_s{ncols2}`'s `extern "C"`
        // signature exactly (`DECL_FATTN_MMA_F16_CASE` in
        // `fattn_mma_f16.cuh`). q_ptr/k_ptr/v_ptr/mask_ptr/kv_max_ptr stay
        // valid until the launch completes because `q`/`k`/`v`/`mask`/
        // `kv_max` (and the storage they borrow from) outlive this call; the
        // grid covers every output element (`geometry`'s tile counts).
        unsafe {
            rocm::launch_2d_cached(
                &dev,
                module_name,
                kernel_name,
                source,
                grid,
                block,
                geometry.shared_mem_bytes,
                &mut args,
            )
        }?;

        if let Some(f) = &geometry.fixup {
            launch_fixup(
                &dev,
                module_name,
                source,
                (d, ncols1, ncols2, false),
                f,
                (&dst_ptr, &dst_meta_ptr),
            )?;
        }

        Ok(rocm::wrap_f16(dst, &dev, (b, seq_q, h_q, d)))
    }

    /// bf16 counterpart of [`fattn_mma_prefill_rocm`]: `k`/`v` stay BF16
    /// (device pointers extracted with `DType::BF16`, not `DType::F16`),
    /// the kernel source/name come from the `_bf16` instantiation files and
    /// [`rocm_mma_bf16_source`], and the kernel selection uses
    /// [`select_mma_ncols_bf16`] (never resolves to `ncols1*ncols2 == 8`,
    /// see that function's doc comment). The kernel writes the output in
    /// BF16 instead of F16.
    #[allow(
        clippy::too_many_lines,
        clippy::many_single_char_names,
        clippy::too_many_arguments
    )]
    pub(super) fn fattn_mma_bf16_prefill_rocm(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        kv_max: Option<&Tensor>,
        scale: f32,
        causal_kv_offset: Option<usize>,
        window_left: i32,
        window_right: i32,
        forced: LaunchOverride,
    ) -> Result<Tensor> {
        let (q_s, q_l) = q.storage_and_layout();
        let (k_s, k_l) = k.storage_and_layout();
        let (v_s, v_l) = v.storage_and_layout();
        let (b, seq_q, h_q, d, seq_kv, h_kv) = validate_fattn_bshd(q_l, k_l, v_l, q.dtype())?;
        let gqa_ratio = h_q / h_kv;
        let dev = q.device().as_rocm_device()?.clone();
        let is_amd = true;

        let Some((mut ncols1, ncols2)) = select_mma_ncols_bf16(d, gqa_ratio, seq_q, is_amd) else {
            return Err(unsupported_config(format!(
                "fattn_mma_bf16: no instantiated kernel for head_dim={d} gqa_ratio={gqa_ratio} seq_q={seq_q}"
            )));
        };
        // See fattn_mma_prefill_rocm's identical check. The in-kernel
        // causal entries pass no mask and are exempt.
        if mask.is_none() && causal_kv_offset.is_none() && ncols2 != 1 {
            return Err(unsupported_config(format!(
                "fattn_mma_bf16: mask=None requires ncols2 == 1 (got ncols2={ncols2}); \
                 pass an explicit all-zero mask for unmasked attention with this GQA shape"
            )));
        }
        let shape = LaunchShape {
            seq_q,
            seq_kv,
            h_kv,
            batch: b,
            gqa_ratio,
        };
        let target = super::grid_target(q.device(), forced)?;
        ncols1 = narrow_for_underfill(
            ncols1,
            ncols2,
            seq_q,
            gqa_ratio,
            h_kv,
            b,
            target.n_sm,
            is_amd,
            true,
        );
        let kv = super::KvExtent::new(seq_kv, h_kv, causal_kv_offset);
        let (cfg, set) = super::launch_config((d, ncols1 * ncols2), kv, (is_amd, true), forced)?;
        let tail = ncols2 > 1 && !seq_kv.is_multiple_of(cfg.nbatch_fa);
        let (kernel_name, module_name) =
            super::cached_mma_names(d, ncols1, ncols2, (true, tail), set);
        let source = source_for_bf16(d, set)?;
        let variant = KernelVariant {
            dkq: d,
            ncols1,
            ncols2,
            is_amd,
            bf16: true,
        };
        let geometry = LaunchGeometry::new(
            &variant,
            &shape,
            &target,
            cfg,
            TileCost::of(causal_kv_offset, kv_max),
            |threads, shared_mem_bytes| {
                real_occupancy(
                    &dev,
                    module_name,
                    kernel_name,
                    source,
                    threads,
                    shared_mem_bytes,
                )
            },
        )?;

        let q_ptr = rocm::strided_device_ptr(&q_s, q_l, DType::BF16, "fattn_mma_bf16 q")?;
        let k_ptr = rocm::strided_device_ptr(&k_s, k_l, DType::BF16, "fattn_mma_bf16 k")?;
        let v_ptr = rocm::strided_device_ptr(&v_s, v_l, DType::BF16, "fattn_mma_bf16 v")?;

        // The mask stays on the contiguous device_ptr. Callers make it
        // contiguous, and its strides are not layout-checked.
        let mask_layout = mask.map(Tensor::storage_and_layout);
        let mask_ptr: *mut c_void = match &mask_layout {
            Some((s, l)) => rocm::device_ptr(s, l, DType::F16, "fattn_mma_bf16 mask")?,
            None => std::ptr::null_mut(),
        };
        let kv_max_layout = kv_max.map(Tensor::storage_and_layout);
        let kv_max_ptr: *mut c_void = match &kv_max_layout {
            Some((s, l)) => rocm::device_ptr(s, l, DType::I32, "fattn_mma_bf16 kv_max")?,
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
            q.dtype(),
        )?;

        let causal_kv_offset = super::kernel_causal_offset(causal_kv_offset)?;
        let heavy_first = i32::from(geometry.heavy_first);

        let n_out = b * seq_q * h_q * d;
        let dst = dev.alloc::<half::bf16>(n_out)?;
        let dst_ptr = dst.as_ptr();
        // The stream-K fixup scratch, written in full by the main kernel
        // before the fixup kernel reads it. Null when no tile is split.
        let scratch = geometry
            .fixup
            .as_ref()
            .map(|f| dev.alloc::<f32>(f.scratch_len))
            .transpose()?;
        let dst_meta_ptr: *mut c_void = scratch
            .as_ref()
            .map_or(std::ptr::null_mut(), SendSyncDeviceMemory::as_ptr);
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
            rocm::arg(&causal_kv_offset),
            rocm::arg(&heavy_first),
            rocm::arg(&window_left),
            rocm::arg(&window_right),
        ];

        // SAFETY: see fattn_mma_prefill_rocm's identical SAFETY comment;
        // the argument list here matches
        // `crane_fattn_mma_bf16_d{d}_v{d}_c{ncols1}_s{ncols2}`'s `extern
        // "C"` signature (`DECL_FATTN_MMA_BF16_CASE` in
        // `fattn_mma_bf16.cuh`) exactly instead.
        unsafe {
            rocm::launch_2d_cached(
                &dev,
                module_name,
                kernel_name,
                source,
                grid,
                block,
                geometry.shared_mem_bytes,
                &mut args,
            )
        }?;

        if let Some(f) = &geometry.fixup {
            launch_fixup(
                &dev,
                module_name,
                source,
                (d, ncols1, ncols2, true),
                f,
                (&dst_ptr, &dst_meta_ptr),
            )?;
        }

        Ok(rocm::wrap_bf16(dst, &dev, (b, seq_q, h_q, d)))
    }

    /// The device facts for the stream-K grid choice: the CU count of
    /// `device`. `n_sm` is a static fact about a device for the process's
    /// lifetime, but this used to query the driver on every single kernel
    /// dispatch; cached per device ordinal. CUDA has no equivalent cache:
    /// `n_sm` only feeds the AMD `head_dim == 64` split, so CUDA's
    /// `grid_target` skips the query entirely.
    pub(super) fn grid_target(device: &Device, forced: LaunchOverride) -> Result<GridTarget> {
        use std::collections::HashMap;
        use std::sync::{OnceLock, RwLock};

        static CACHE: OnceLock<RwLock<HashMap<usize, usize>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));

        let candle_core::DeviceLocation::Rocm { gpu_id } = device.location() else {
            candle_core::bail!("grid_target: not a rocm device");
        };
        if let Some(&n_sm) = cache.read().unwrap().get(&gpu_id) {
            return Ok(GridTarget { n_sm, forced });
        }
        let n_sm = rocm::multiprocessor_count(device)?;
        cache.write().unwrap().insert(gpu_id, n_sm);
        Ok(GridTarget { n_sm, forced })
    }

    /// The driver's real occupancy for this kernel, or `None` if the query
    /// fails (the config table's guess is then used instead; a failed
    /// query here should not break the launch). A successful query can
    /// still report `0` (register/LDS pressure leaves no room for even one
    /// block); [`LaunchGeometry::new`] is the one that turns that into a
    /// table-value fallback, since it also has to handle a test-supplied
    /// closure, not just this driver-backed one. See
    /// [`rocm::occupancy_max_active_blocks`] and [`LaunchGeometry::new`]'s
    /// doc comment.
    ///
    /// Cached per `(module_name, kernel_name)`: both are the `&'static str`
    /// [`super::cached_mma_names`] leaks once per `(dkq, ncols1, ncols2,
    /// bf16)`, so the same kernel always hits the same cache entry instead
    /// of re-querying the driver on every dispatch.
    fn real_occupancy(
        dev: &candle_core::rocm_backend::RocmDevice,
        module_name: &'static str,
        kernel_name: &'static str,
        source: &'static str,
        threads: usize,
        shared_mem_bytes: usize,
    ) -> Option<usize> {
        use std::collections::HashMap;
        use std::sync::{OnceLock, RwLock};

        static CACHE: OnceLock<RwLock<HashMap<(&'static str, &'static str), Option<usize>>>> =
            OnceLock::new();
        let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
        let key = (module_name, kernel_name);
        if let Some(&occupancy) = cache.read().unwrap().get(&key) {
            return occupancy;
        }
        let occupancy = rocm::occupancy_max_active_blocks(
            dev,
            module_name,
            kernel_name,
            source,
            u32::try_from(threads).ok()?,
            u32::try_from(shared_mem_bytes).ok()?,
        )
        .ok();
        cache.write().unwrap().insert(key, occupancy);
        occupancy
    }

    /// Launches the stream-K fixup kernel that pairs with the MMA kernel of
    /// `(head_dim, ncols1, ncols2, bf16)`. It merges the partials the main
    /// kernel left in `dst_meta` into `dst`.
    fn launch_fixup(
        dev: &candle_core::rocm_backend::RocmDevice,
        module_name: &str,
        source: &'static str,
        (d, ncols1, ncols2, bf16): (usize, usize, usize, bool),
        f: &FixupPlan,
        (dst_ptr, dst_meta_ptr): (&*mut c_void, &*mut c_void),
    ) -> Result<()> {
        let kernel_name = fixup_kernel_name(d, ncols1, ncols2, bf16);
        let grid = rocm_rs::hip::Dim3::new_3d(f.grid.0, f.grid.1, f.grid.2);
        let block = rocm_rs::hip::Dim3::new_3d(f.block.0, f.block.1, f.block.2);
        let mut args = [
            rocm::arg(dst_ptr),
            rocm::arg(dst_meta_ptr),
            rocm::arg(&f.ne01),
            rocm::arg(&f.ne02),
            rocm::arg(&f.gqa_ratio),
            rocm::arg(&f.total_work),
            rocm::arg(&f.fd_k_j_z_ne12),
            rocm::arg(&f.fd_k_j_z),
            rocm::arg(&f.fd_k_j),
            rocm::arg(&f.fd_k),
        ];
        // SAFETY: the argument list matches `crane_fattn_fixup_*`'s `extern
        // "C"` signature in `fattn_fixup.cuh`. `dst_ptr` and `dst_meta_ptr`
        // stay valid until the launch completes because the caller holds
        // `dst` and the scratch buffer across this call. The grid covers
        // one block per (main block, Q row, GQA head).
        unsafe {
            rocm::launch_2d_cached(
                dev,
                module_name,
                &kernel_name,
                source,
                grid,
                block,
                0,
                &mut args,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::*;
    use crate::ops::fused_ops::fattn::{
        AMPERE_BF16_HD256_LONG_KV_MIN_SEQ_KV, RDNA_HD256_LONG_KV_MIN_SEQ_KV, mma_config,
    };

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

    // The policy picks the long-KV set from the threshold on, and a forced
    // override wins over it in both directions.
    #[test]
    fn launch_config_picks_and_forces_the_kernel_set() {
        let min = RDNA_HD256_LONG_KV_MIN_SEQ_KV;
        let pick = |seq_kv, forced| {
            let kv = KvExtent::new(seq_kv, 2, None);
            launch_config((256, 64), kv, (true, false), forced)
                .unwrap()
                .1
        };
        assert_eq!(pick(min - 1, LaunchOverride::NONE), MmaConfigSet::Default);
        assert_eq!(pick(min, LaunchOverride::NONE), MmaConfigSet::LongKv);
        let force = |long_kv| LaunchOverride {
            long_kv: Some(long_kv),
            ..LaunchOverride::NONE
        };
        assert_eq!(pick(min, force(false)), MmaConfigSet::Default);
        assert_eq!(pick(1, force(true)), MmaConfigSet::LongKv);
    }

    // NVIDIA's BF16 hd256 32-column tile mirrors the RDNA policy above, at
    // its own threshold.
    #[test]
    fn launch_config_picks_the_ampere_bf16_long_kv_set() {
        let min = AMPERE_BF16_HD256_LONG_KV_MIN_SEQ_KV;
        let pick = |seq_kv, forced| {
            let kv = KvExtent::new(seq_kv, 2, None);
            launch_config((256, 32), kv, (false, true), forced)
                .unwrap()
                .1
        };
        assert_eq!(pick(min - 1, LaunchOverride::NONE), MmaConfigSet::Default);
        assert_eq!(pick(min, LaunchOverride::NONE), MmaConfigSet::LongKv);
        let force = |long_kv| LaunchOverride {
            long_kv: Some(long_kv),
            ..LaunchOverride::NONE
        };
        assert_eq!(pick(min, force(false)), MmaConfigSet::Default);
        assert_eq!(pick(1, force(true)), MmaConfigSet::LongKv);
    }

    // Only a launch that knows its causal offset can return to the default
    // set for a long KV every tile reads: an unknown offset keeps the
    // long-KV set.
    #[test]
    fn kv_extent_uses_the_causal_offset_only() {
        let seq_kv = 65536;
        let set = |offset| {
            let kv = KvExtent::new(seq_kv, 2, offset);
            launch_config((256, 64), kv, (true, false), LaunchOverride::NONE)
                .unwrap()
                .1
        };
        assert_eq!(set(None), MmaConfigSet::LongKv);
        assert_eq!(set(Some(0)), MmaConfigSet::LongKv);
        assert_eq!(set(Some(seq_kv - 2048)), MmaConfigSet::Default);
    }

    // Each set gets its own tuning config, for both dtypes.
    #[test]
    fn launch_config_returns_the_config_of_the_chosen_set() {
        for bf16 in [false, true] {
            let nbatch_fa = |long_kv| {
                let forced = LaunchOverride {
                    long_kv: Some(long_kv),
                    ..LaunchOverride::NONE
                };
                launch_config((256, 64), KvExtent::new(1, 2, None), (true, bf16), forced)
                    .unwrap()
                    .0
                    .nbatch_fa
            };
            assert_eq!(nbatch_fa(false), 32);
            assert_eq!(nbatch_fa(true), 64);
        }
    }

    // Forcing the long set where no kernels exist is an error, not a silent
    // fallback to the default set.
    #[test]
    fn launch_config_rejects_forced_long_kv_without_a_set() {
        let forced = LaunchOverride {
            long_kv: Some(true),
            ..LaunchOverride::NONE
        };
        for (shape, is_amd) in [((256, 32), true), ((128, 64), true), ((256, 64), false)] {
            let err = launch_config(shape, KvExtent::new(1, 2, None), (is_amd, false), forced)
                .expect_err("no long-KV set exists for this launch");
            assert!(err.to_string().contains("no long-KV kernel set"), "{err}");
        }
    }

    // The long-KV set reuses the kernel names and differs by module only.
    #[test]
    fn cached_mma_names_separates_the_sets_by_module() {
        for bf16 in [false, true] {
            for tail in [false, true] {
                let (kernel, module) =
                    cached_mma_names(256, 16, 4, (bf16, tail), MmaConfigSet::Default);
                let (long_kernel, long_module) =
                    cached_mma_names(256, 16, 4, (bf16, tail), MmaConfigSet::LongKv);
                assert_eq!(kernel, long_kernel);
                assert_eq!(long_module, format!("{module}_long_kv"));
            }
        }
    }

    // q must be in the k/v dtype for both kernel families. F32 q, which
    // upstream takes, is rejected rather than cast.
    #[test]
    fn rejects_q_not_in_kv_dtype() {
        let dev = Device::Cpu;
        let zeros = |dtype| Tensor::zeros((1, 4, 2, 64), dtype, &dev).unwrap();
        let (f32_q, f16, bf16) = (zeros(DType::F32), zeros(DType::F16), zeros(DType::BF16));
        let err = fattn_mma_prefill(&f32_q, &f16, &f16, None, None, 0.125)
            .expect_err("must reject F32 q");
        assert!(err.to_string().contains("q/k/v must be F16"));
        let err = fattn_mma_bf16_prefill(&f32_q, &bf16, &bf16, None, None, 0.125)
            .expect_err("must reject F32 q");
        assert!(err.to_string().contains("q/k/v must be BF16"));
        let err = fattn_mma_bf16_prefill(&f16, &bf16, &bf16, None, None, 0.125)
            .expect_err("must reject F16 q");
        assert!(err.to_string().contains("q/k/v must be BF16"));
    }

    // k/v must be F16 for the f16 kernel family.
    #[test]
    fn rejects_non_f16_kv() {
        let dev = Device::Cpu;
        let q = Tensor::zeros((1, 4, 2, 64), DType::F16, &dev).unwrap();
        let k = Tensor::zeros((1, 4, 2, 64), DType::F32, &dev).unwrap();
        let v = k.clone();
        let err =
            fattn_mma_prefill(&q, &k, &v, None, None, 0.125).expect_err("must reject F32 k/v");
        assert!(err.to_string().contains("q/k/v must be F16"));
    }

    // The causal entries use the host mask exactly when the diagonal of the
    // last query row reaches past the last live key.
    #[test]
    fn causal_needs_host_mask_only_past_live_keys() {
        assert!(!causal_needs_host_mask(20, 200, 179));
        assert!(!causal_needs_host_mask(20, 200, 180));
        assert!(causal_needs_host_mask(20, 200, 181));
        assert!(!causal_needs_host_mask(64, 64, 0));
    }

    // Same rule as `causal_needs_host_mask`, extended by `window_right`:
    // the windowed entries use the host mask exactly when the window's
    // right edge of the last query row reaches past the last live key.
    // Reduces to `causal_needs_host_mask` at `window_right == 0`.
    #[test]
    fn windowed_needs_host_mask_only_past_live_keys() {
        assert!(!windowed_needs_host_mask(20, 200, 179, 0));
        assert!(!windowed_needs_host_mask(20, 200, 180, 0));
        assert!(windowed_needs_host_mask(20, 200, 181, 0));
        assert!(!windowed_needs_host_mask(64, 64, 0, 0));
        assert!(!windowed_needs_host_mask(20, 200, 170, 9));
        assert!(!windowed_needs_host_mask(20, 200, 170, 10));
        assert!(windowed_needs_host_mask(20, 200, 170, 11));
    }

    // `None` maps to the kernel's "not causal" sentinel, `Some` passes the
    // offset through, and an offset beyond `i32::MAX` is rejected.
    #[test]
    fn kernel_causal_offset_maps_none_to_sentinel() {
        assert_eq!(kernel_causal_offset(None).unwrap(), -1);
        assert_eq!(kernel_causal_offset(Some(5)).unwrap(), 5);
        assert!(kernel_causal_offset(Some(usize::MAX)).is_err());
    }

    // The narrow-tile test shapes in `gpu_tests` must really select
    // `ncols1 * ncols2 == 8` on NVIDIA, or they test nothing new.
    #[test]
    fn narrow_bitwise_shapes_select_ncols_8_on_nvidia() {
        assert_eq!(select_mma_ncols(128, 4, 2, false), Some((2, 4)));
        assert_eq!(select_mma_ncols(128, 1, 8, false), Some((8, 1)));
    }

    // A driver occupancy query reporting zero active blocks must fall back
    // to the config table's guess, the same as a failed query. Sizing the
    // stream-K grid on zero resident blocks would launch nothing and still
    // hand back an uninitialized `dst` tensor as if it were real output.
    #[test]
    fn zero_driver_occupancy_falls_back_to_table_value() {
        let variant = KernelVariant {
            dkq: 64,
            ncols1: 8,
            ncols2: 1,
            is_amd: true,
            bf16: false,
        };
        let shape = LaunchShape {
            seq_q: 64,
            seq_kv: 256,
            h_kv: 1,
            batch: 1,
            gqa_ratio: 1,
        };
        let target = GridTarget {
            n_sm: 60,
            forced: LaunchOverride::NONE,
        };
        let cfg = mma_config(variant.dkq, variant.ncols1 * variant.ncols2, variant.is_amd).unwrap();
        let uniform = TileCost::Uniform;
        let zero =
            LaunchGeometry::new(&variant, &shape, &target, cfg, uniform, |_, _| Some(0)).unwrap();
        let table =
            LaunchGeometry::new(&variant, &shape, &target, cfg, uniform, |_, _| None).unwrap();
        assert_eq!(zero.grid, table.grid);
    }

    // Heavy-first needs one block per tile, a cost that grows with the Q
    // tile and a launch that is not too large. A forced order replaces the
    // policy on any grid, so tests can reach the kernel's fallback.
    #[test]
    fn heavy_first_policy() {
        use TileCost::{GrowsWithJt, Uniform};
        let ntiles = 64;
        // (cost, grid, forced, want)
        let rows = [
            (GrowsWithJt, ntiles, None, true),
            (Uniform, ntiles, None, false),
            (GrowsWithJt, ntiles + 1, None, false),
            (GrowsWithJt, ntiles / 2, Some(true), true),
            (Uniform, ntiles, Some(true), true),
            (GrowsWithJt, ntiles, Some(false), false),
        ];
        for (cost, grid, forced, want) in rows {
            assert_eq!(
                use_heavy_first(cost, grid, ntiles, forced),
                want,
                "{cost:?} grid={grid} forced={forced:?}"
            );
        }
        let max = HEAVY_FIRST_MAX_TILES;
        assert!(use_heavy_first(GrowsWithJt, max, max, None));
        assert!(!use_heavy_first(GrowsWithJt, max + 1, max + 1, None));
    }

    // In-kernel causal masking or a `KV_max` make the cost grow with the Q
    // tile. Neither leaves it uniform.
    #[test]
    fn tile_cost_of_launch() {
        let kv_max = Tensor::zeros(4, DType::I32, &Device::Cpu).unwrap();
        assert_eq!(TileCost::of(Some(0), None), TileCost::GrowsWithJt);
        assert_eq!(TileCost::of(None, Some(&kv_max)), TileCost::GrowsWithJt);
        assert_eq!(TileCost::of(None, None), TileCost::Uniform);
    }

    // `LaunchGeometry::new` turns heavy-first on for a growing cost, and off
    // for a uniform cost or too many tiles.
    #[test]
    fn launch_geometry_picks_tile_order() {
        let (h_kv, gqa_ratio) = (2, 4);
        let geometry = |seq: usize, cost| {
            let (ncols1, ncols2) = select_mma_ncols(128, gqa_ratio, seq, false).unwrap();
            let variant = KernelVariant {
                dkq: 128,
                ncols1,
                ncols2,
                is_amd: false,
                bf16: false,
            };
            let shape = LaunchShape {
                seq_q: seq,
                seq_kv: seq,
                h_kv,
                batch: 1,
                gqa_ratio,
            };
            let target = GridTarget {
                n_sm: 0,
                forced: LaunchOverride::NONE,
            };
            let cfg = mma_config(128, ncols1 * ncols2, false).unwrap();
            let (ntiles, _) = tile_counts(&cfg, ncols1, ncols2, &shape).unwrap();
            let got = LaunchGeometry::new(&variant, &shape, &target, cfg, cost, |_, _| None)
                .unwrap()
                .heavy_first;
            (got, ntiles)
        };
        let (got, ntiles) = geometry(512, TileCost::GrowsWithJt);
        assert!(ntiles <= HEAVY_FIRST_MAX_TILES);
        assert!(got);
        assert!(!geometry(512, TileCost::Uniform).0);
        let (got, ntiles) = geometry(64 * 1024, TileCost::GrowsWithJt);
        assert!(ntiles > HEAVY_FIRST_MAX_TILES);
        assert!(!got);
    }

    // The heavy-first mapping gives every tile to exactly one block and
    // hands them out with `jt` never rising. The GPU tests can show the
    // first, not the second.
    #[test]
    fn heavy_first_mapping_is_a_falling_permutation() {
        for iter_j in [1, 3, 8] {
            for nheads in [1, 2, 5, 8] {
                let ntiles = iter_j * nheads;
                let what = format!("iter_j={iter_j} nheads={nheads}");
                let tiles: Vec<usize> = (0..ntiles)
                    .map(|b| heavy_first_tile(b, iter_j, nheads))
                    .collect();
                let mut sorted = tiles.clone();
                sorted.sort_unstable();
                assert!(
                    sorted.into_iter().eq(0..ntiles),
                    "{what}: not a permutation"
                );
                for pair in tiles.windows(2) {
                    assert!(pair[1] % iter_j <= pair[0] % iter_j, "{what}: jt rises");
                }
            }
        }
    }

    /// Mirror of the heavy-first branch of `crane_fattn_block_kbc_range`
    /// (`fattn_fixup.cuh`): the tile block `b` takes, for `nheads` tiles per
    /// Q tile index.
    fn heavy_first_tile(b: usize, iter_j: usize, nheads: usize) -> usize {
        (b % nheads) * iter_j + (iter_j - 1 - b / nheads)
    }
}

/// Correctness against a real GPU (this machine's ROCm GPU, or a CUDA GPU
/// when built with `--features cuda` elsewhere), reusing
/// `fattn::test_support`'s naive matmul/mask/softmax/matmul reference. The
/// `run_*` helpers below build their mask by hand and call
/// [`fattn_mma_prefill`] directly; [`fattn_mma_causal`]/[`fattn_mma_full`]/
/// [`fattn_mma_windowed`] get their own separate end-to-end tests further
/// down, covering the mask/`KV_max`-building logic those wrap around it.
#[cfg(all(test, crane_fattn))]
mod gpu_tests {
    use candle_core::{DType, Device, Tensor};

    use super::super::fattn::is_amd_backend;
    use super::super::fattn::test_support::{
        MaskMode, naive_attention, narrowed_bhsd, test_gpu_device,
    };
    use super::{
        LaunchOverride, WindowBounds, fattn_mma_bf16_causal, fattn_mma_bf16_causal_with_mask,
        fattn_mma_bf16_prefill, fattn_mma_bf16_windowed_with_mask, fattn_mma_causal,
        fattn_mma_causal_with_mask, fattn_mma_full, fattn_mma_prefill, fattn_mma_windowed,
        fattn_mma_windowed_with_mask, prefill_dispatch, prefill_dispatch_bf16, windowed_in_kernel,
        windowed_needs_host_mask,
    };

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

        let q = q_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
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

        let q = q_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
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

    // bf16 counterpart of run_full: k/v stay BF16 (never cast to F16),
    // dispatches through fattn_mma_bf16_prefill. On this AMD environment,
    // ncols2 never resolves to 1 (see fattn_mma_prefill's doc comment), so
    // (b, sq, skv, hq, hkv, d) must pick a GQA ratio and aligned seq_kv that
    // resolves to a (ncols1, ncols2) pair fattn_mma_hd*_bf16.cu actually
    // instantiates (ncols1*ncols2 in {16, 32, 64}, never 8 — see
    // fattn_mma_bf16.cuh's module doc) for this to exercise the real kernel
    // rather than silently bailing with "no instantiated kernel".
    fn run_full_bf16(b: usize, sq: usize, skv: usize, hq: usize, hkv: usize, d: usize) {
        let gpu = test_gpu_device();
        let scale = 1.0f32 / (d as f32).sqrt();

        let q_f32 = Tensor::randn(0f32, 1f32, (b, sq, hq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();

        let want = naive_attention(&q_f32, &k_f32, &v_f32, scale, MaskMode::Full, 0, 0, 0);

        let q = q_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let k = k_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let v = v_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let zero_mask = Tensor::zeros((1, sq, skv), DType::F16, &gpu).unwrap();

        let got = fattn_mma_bf16_prefill(&q, &k, &v, Some(&zero_mask), None, scale)
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
                "[{i}] got {g}, want {w} (bf16 full, head_dim={d})"
            );
        }
    }

    // bf16 counterpart of run_causal: see run_full_bf16's doc comment.
    fn run_causal_bf16(b: usize, sq: usize, skv: usize, hq: usize, hkv: usize, d: usize) {
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

        // fattn_mma_bf16_causal takes BSHD like fattn_mma_causal (see this
        // module's doc comment); q_f32/k_f32/v_f32 above are already BSHD.
        let q = q_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let k = k_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let v = v_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();

        let got = fattn_mma_bf16_causal(&q, &k, &v, scale, kv_offset)
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
                "[{i}] got {g}, want {w} (bf16 causal, head_dim={d})"
            );
        }
    }

    // Native bf16 kernel, GQA (gqa_ratio=2, seq_kv=256 aligned so AMD's
    // ncols2 != 1 requirement is met, see run_full_bf16's doc comment),
    // head_dim=64 and 128.
    #[test]
    fn full_bf16_hd64() {
        run_full_bf16(1, 32, 256, 4, 2, 64);
    }

    #[test]
    fn full_bf16_hd128() {
        run_full_bf16(1, 32, 256, 4, 2, 128);
    }

    #[test]
    fn full_bf16_hd256() {
        run_full_bf16(1, 32, 256, 4, 2, 256);
    }

    #[test]
    fn causal_bf16_hd64() {
        run_causal_bf16(1, 32, 256, 4, 2, 64);
    }

    #[test]
    fn causal_bf16_hd128() {
        run_causal_bf16(1, 32, 256, 4, 2, 128);
    }

    #[test]
    fn causal_bf16_hd256() {
        run_causal_bf16(1, 32, 256, 4, 2, 256);
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

    #[test]
    fn full_zero_mask_hd256() {
        run_full(1, 32, 256, 4, 2, 256);
    }

    // mask = None is only accepted when the selected ncols2 == 1 (see
    // fattn_mma_prefill's doc comment); ncols2 never resolves to 1 on AMD
    // (select_mma_ncols's doc comment), so this must fail cleanly with the
    // "no instantiated kernel" error on this (ROCm) backend.
    #[test]
    fn none_mask_bails_cleanly_when_ncols2_never_resolves_to_one_on_amd() {
        let gpu = test_gpu_device();
        let d = 64usize;
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = Tensor::zeros((1, 16, 2, d), DType::F16, &gpu).unwrap();
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
    // "requires ncols2 == 1" error (see fattn_mma_prefill's doc comment).
    #[test]
    fn none_mask_bails_cleanly_when_ncols2_resolves_above_one() {
        let gpu = test_gpu_device();
        let d = 128usize;
        let scale = 1.0f32 / (d as f32).sqrt();
        let q = Tensor::zeros((1, 32, 8, d), DType::F16, &gpu).unwrap();
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

    #[test]
    fn causal_hd256() {
        run_causal(1, 32, 256, 4, 2, 256);
    }

    // Both kernel sets of the RDNA hd256 64-column tile match the reference
    // at GQA ratios 4 and 8 (`c16_s4`, `c8_s8`), over an aligned length and
    // two unaligned ones (the `_tail` kernels).
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    #[test]
    fn both_kernel_sets_match_reference_hd256() {
        let gpu = test_gpu_device();
        let d = 256;
        // 1 / sqrt(256).
        let scale = 0.0625f32;
        for dtype in [DType::F16, DType::BF16] {
            for (hq, hkv) in [(8, 2), (16, 2)] {
                for seq in [231, 256, 300] {
                    let rand = |heads| {
                        Tensor::randn(0f32, 1f32, (1, seq, heads, d), &Device::Cpu).unwrap()
                    };
                    let (q, k, v) = (rand(hq), rand(hkv), rand(hkv));
                    let want = naive_attention(&q, &k, &v, scale, MaskMode::Causal, 0, 0, 0);
                    let to_gpu = |t: &Tensor| t.to_dtype(dtype).unwrap().to_device(&gpu).unwrap();
                    let (qg, kg, vg) = (to_gpu(&q), to_gpu(&k), to_gpu(&v));
                    for long_kv in [false, true] {
                        let forced = LaunchOverride {
                            long_kv: Some(long_kv),
                            ..LaunchOverride::NONE
                        };
                        let dispatch = if dtype == DType::BF16 {
                            prefill_dispatch_bf16
                        } else {
                            prefill_dispatch
                        };
                        let got = got_vec(
                            dispatch(
                                &qg,
                                &kg,
                                &vg,
                                None,
                                None,
                                scale,
                                Some(0),
                                WindowBounds::NONE,
                                forced,
                            )
                            .unwrap(),
                        );
                        assert_eq!(got.len(), want.len());
                        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                            assert!(
                                (g - w).abs() <= 3e-2 * w.abs().max(1.0),
                                "[{i}] got {g}, want {w} ({dtype:?} {hq}Q/{hkv}KV seq={seq} long_kv={long_kv})"
                            );
                        }
                    }
                }
            }
        }
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

        let q = q_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let k = k_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let v = v_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();

        let got = fattn_mma_causal(&q, &k, &v, scale, kv_offset).unwrap();
        assert_close(&got_vec(got), &want, 3e-2, "causal_fn");
    }

    // End-to-end `fattn_mma_causal` with a non-FATTN_KQ_STRIDE-aligned
    // seq_kv and GQA ratio > 1: unlike `causal_fn_matches_reference_hd128`'s
    // already-aligned skv=256, this exercises the kernel's own partial-tile
    // bounds checks (skv=200 still picks the ncols2>1 kernel on AMD, see
    // `select_mma_ncols`'s doc comment), which every other GPU test in this
    // module short-circuits around by only ever using an aligned seq_kv.
    #[test]
    fn causal_fn_matches_reference_unaligned_gqa_hd128() {
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

        let q = q_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let k = k_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let v = v_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();

        let got = fattn_mma_causal(&q, &k, &v, scale, kv_offset).unwrap();
        assert_close(&got_vec(got), &want, 3e-2, "causal_fn_unaligned_gqa");
    }

    // bf16 counterpart of `causal_fn_matches_reference_unaligned_gqa_hd128`:
    // exercises the bf16 kernel's own partial-tile bounds checks (skv=200
    // still picks the ncols2>1 bf16 kernel on AMD), proving that path is
    // bit-correct, not just fast.
    #[test]
    fn causal_fn_matches_reference_unaligned_gqa_bf16_hd128() {
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

        let q = q_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let k = k_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let v = v_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();

        let got = fattn_mma_bf16_causal(&q, &k, &v, scale, kv_offset).unwrap();
        assert_close(&got_vec(got), &want, 3e-2, "causal_fn_unaligned_gqa_bf16");
    }

    // `causal_fn_matches_reference_unaligned_gqa_hd128`'s skv=200 always
    // spans multiple KV tiles, so it never exercises the single-tile branch
    // of the kernel's tail dispatch (`skip_initial_preload`, see
    // `fattn_mma_f16.cuh`'s `oob_tail` handling), which only fires when a
    // block's entire KV range is one partial tile. hd128 GQA picks
    // ncols=8, whose smallest `nbatch_fa` across backends is 32 (RDNA);
    // skv=20 is below that on every backend, forcing the single-tile case.
    #[test]
    fn causal_fn_single_tile_tail_gqa_hd128() {
        let gpu = test_gpu_device();
        let (b, sq, skv, hq, hkv, d) = (1, 10, 20, 8, 2, 128usize);
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

        let q = q_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let k = k_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let v = v_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();

        let got = fattn_mma_causal(&q, &k, &v, scale, kv_offset).unwrap();
        assert_close(&got_vec(got), &want, 3e-2, "causal_fn_single_tile_tail");
    }

    // bf16 counterpart of `causal_fn_single_tile_tail_gqa_hd128`: proves
    // the bf16 kernel's single-tile tail branch (no `cp.async` pipeline,
    // see `fattn_mma_bf16.cuh`'s `oob_check` branch) is bit-correct too.
    #[test]
    fn causal_fn_single_tile_tail_gqa_bf16_hd128() {
        let gpu = test_gpu_device();
        let (b, sq, skv, hq, hkv, d) = (1, 10, 20, 8, 2, 128usize);
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

        let q = q_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let k = k_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let v = v_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();

        let got = fattn_mma_bf16_causal(&q, &k, &v, scale, kv_offset).unwrap();
        assert_close(
            &got_vec(got),
            &want,
            3e-2,
            "causal_fn_single_tile_tail_bf16",
        );
    }

    // Runs the in-kernel causal entry and the host-mask entry on the same
    // inputs and requires identical output bits.
    #[allow(clippy::many_single_char_names)]
    fn assert_causal_bitwise_equals_with_mask(
        bf16: bool,
        (b, sq, skv, hq, hkv, d): (usize, usize, usize, usize, usize, usize),
    ) {
        let gpu = test_gpu_device();
        // `d` is at most 256, exact in an f32.
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;
        let kv_dtype = if bf16 { DType::BF16 } else { DType::F16 };

        let q = Tensor::randn(0f32, 1f32, (b, sq, hq, d), &gpu)
            .unwrap()
            .to_dtype(kv_dtype)
            .unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &gpu)
            .unwrap()
            .to_dtype(kv_dtype)
            .unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &gpu)
            .unwrap()
            .to_dtype(kv_dtype)
            .unwrap();
        let mask = causal_mask(sq, skv, kv_offset, &gpu);

        let (got, want) = if bf16 {
            (
                fattn_mma_bf16_causal(&q, &k, &v, scale, kv_offset),
                fattn_mma_bf16_causal_with_mask(&q, &k, &v, scale, kv_offset, &mask),
            )
        } else {
            (
                fattn_mma_causal(&q, &k, &v, scale, kv_offset),
                fattn_mma_causal_with_mask(&q, &k, &v, scale, kv_offset, &mask),
            )
        };
        let (got, want) = (got_vec(got.unwrap()), got_vec(want.unwrap()));
        assert_eq!(got.len(), want.len());
        let mismatches = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        assert_eq!(
            mismatches,
            0,
            "bf16={bf16} b={b} sq={sq} skv={skv} hq={hq} hkv={hkv} d={d}: {mismatches} of {} elements differ",
            got.len()
        );
    }

    // Shapes cover an unaligned seq_kv with GQA padding, a decode-like
    // offset, padding with an offset, batch 2, and a partial last Q tile
    // whose in-kernel KV bound stops several tiles before the host
    // `KV_max`'s `FATTN_KQ_STRIDE`-rounded one. MHA (ncols2 == 1) has no AMD
    // kernel, so it only runs on NVIDIA.
    const CAUSAL_BITWISE_SHAPES: [(usize, usize, usize, usize, usize, usize); 5] = [
        (1, 231, 231, 8, 2, 256),
        (1, 16, 256, 8, 2, 128),
        (1, 100, 300, 8, 2, 64),
        (2, 64, 64, 8, 2, 128),
        (1, 300, 1100, 8, 2, 256),
    ];

    // NVIDIA-only f16 shapes with `ncols1 * ncols2 == 8`, the narrow
    // `cols_per_warp == 8` mask site: `(2, 4)` and `(8, 1)`. AMD and the
    // bf16 kernels have no such tile.
    const CAUSAL_BITWISE_NARROW_SHAPES: [(usize, usize, usize, usize, usize, usize); 2] =
        [(1, 2, 256, 8, 2, 128), (1, 8, 64, 4, 4, 128)];

    // `fattn_mma_causal` (kernel masks) equals `fattn_mma_causal_with_mask`
    // (host mask) bit for bit.
    #[test]
    fn causal_matches_causal_with_mask_bitwise() {
        for shape in CAUSAL_BITWISE_SHAPES {
            assert_causal_bitwise_equals_with_mask(false, shape);
        }
        if !is_amd_backend() {
            assert_causal_bitwise_equals_with_mask(false, (1, 64, 64, 4, 4, 128));
            for shape in CAUSAL_BITWISE_NARROW_SHAPES {
                assert_causal_bitwise_equals_with_mask(false, shape);
            }
        }
    }

    // Same check for the native-bf16 kernels.
    #[test]
    fn causal_matches_causal_with_mask_bitwise_bf16() {
        for shape in CAUSAL_BITWISE_SHAPES {
            assert_causal_bitwise_equals_with_mask(true, shape);
        }
        if !is_amd_backend() {
            assert_causal_bitwise_equals_with_mask(true, (1, 64, 64, 4, 4, 128));
        }
    }

    // Runs the in-kernel windowed entry (`windowed_in_kernel`, M1 option 3)
    // and the host-mask entry on the same inputs and requires identical
    // output bits. `kv_offset` must leave enough slack
    // (`sq + kv_offset + window_right <= skv`) that `windowed_needs_host_mask`
    // never triggers the fallback, which would otherwise make both sides
    // call the same mask-based kernel.
    #[allow(clippy::many_single_char_names)]
    fn assert_windowed_bitwise_equals_with_mask(
        bf16: bool,
        (b, sq, skv, hq, hkv, d): (usize, usize, usize, usize, usize, usize),
        (window_left, window_right): (usize, usize),
        kv_offset: usize,
    ) {
        let gpu = test_gpu_device();
        // `d` is at most 256, exact in an f32.
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_dtype = if bf16 { DType::BF16 } else { DType::F16 };
        assert!(!windowed_needs_host_mask(sq, skv, kv_offset, window_right));

        let q = Tensor::randn(0f32, 1f32, (b, sq, hq, d), &gpu)
            .unwrap()
            .to_dtype(kv_dtype)
            .unwrap();
        let k = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &gpu)
            .unwrap()
            .to_dtype(kv_dtype)
            .unwrap();
        let v = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &gpu)
            .unwrap()
            .to_dtype(kv_dtype)
            .unwrap();
        let mask = super::super::fattn::build_additive_mask_f16(
            sq,
            skv,
            kv_offset,
            Some(window_left),
            window_right,
            &gpu,
        )
        .unwrap();

        // windowed_in_kernel auto-detects F16/BF16 from k/v's dtype.
        let got = windowed_in_kernel(&q, &k, &v, scale, kv_offset, window_left, window_right);
        let want = if bf16 {
            fattn_mma_bf16_windowed_with_mask(&q, &k, &v, scale, kv_offset, window_right, &mask)
        } else {
            fattn_mma_windowed_with_mask(&q, &k, &v, scale, kv_offset, window_right, &mask)
        };
        let (got, want) = (got_vec(got.unwrap()), got_vec(want.unwrap()));
        assert_eq!(got.len(), want.len());
        let mismatches = got
            .iter()
            .zip(&want)
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        assert_eq!(
            mismatches,
            0,
            "bf16={bf16} b={b} sq={sq} skv={skv} hq={hq} hkv={hkv} d={d} \
             window=({window_left},{window_right}): {mismatches} of {} elements differ",
            got.len()
        );
    }

    // Shapes cover an unaligned seq_kv with GQA padding, a window narrower
    // than one tile, a window spanning several tiles, and batch 2. Reuses
    // `CAUSAL_BITWISE_SHAPES`' shapes with a window on each. `window_right`
    // stays 0 here (left-edge-only masking); `kv_offset = skv - sq` is the
    // tightest offset that still satisfies `windowed_needs_host_mask`.
    const WINDOWED_BITWISE_CASES: [((usize, usize, usize, usize, usize, usize), (usize, usize));
        5] = [
        ((1, 231, 231, 8, 2, 256), (32, 0)),
        ((1, 16, 256, 8, 2, 128), (8, 0)),
        ((1, 100, 300, 8, 2, 64), (40, 0)),
        ((2, 64, 64, 8, 2, 128), (16, 0)),
        ((1, 300, 1100, 8, 2, 256), (128, 0)),
    ];

    // `windowed_in_kernel` (kernel masks both window edges) equals
    // `fattn_mma_windowed_with_mask` (host mask) bit for bit.
    #[test]
    fn windowed_matches_windowed_with_mask_bitwise() {
        for (shape, window) in WINDOWED_BITWISE_CASES {
            let kv_offset = shape.2 - shape.1;
            assert_windowed_bitwise_equals_with_mask(false, shape, window, kv_offset);
        }
    }

    // Same check for the native-bf16 kernels.
    #[test]
    fn windowed_matches_windowed_with_mask_bitwise_bf16() {
        for (shape, window) in WINDOWED_BITWISE_CASES {
            let kv_offset = shape.2 - shape.1;
            assert_windowed_bitwise_equals_with_mask(true, shape, window, kv_offset);
        }
    }

    // Same shapes, but with `window_right > 0` and `kv_offset` lowered so
    // `sq + kv_offset + window_right <= skv`: `WINDOWED_BITWISE_CASES` can't
    // cover this, since its `kv_offset = skv - sq` leaves no room for a
    // positive `window_right` without tripping `windowed_needs_host_mask`
    // and silently falling back to the host-mask kernel on both sides. This
    // exercises the in-kernel right-edge trim (`right_ext`/`kv_end`) that
    // would otherwise go untested.
    const WINDOWED_RIGHT_BITWISE_CASES: [(
        (usize, usize, usize, usize, usize, usize),
        (usize, usize),
        usize,
    ); 3] = [
        ((1, 100, 300, 8, 2, 64), (40, 20), 170),
        ((1, 64, 256, 8, 2, 128), (16, 8), 180),
        ((1, 231, 500, 8, 2, 256), (32, 10), 250),
    ];

    // `windowed_in_kernel` matches the host-mask entry bit for bit when
    // `window_right > 0` actually reaches the in-kernel path.
    #[test]
    fn windowed_right_matches_windowed_with_mask_bitwise() {
        for (shape, window, kv_offset) in WINDOWED_RIGHT_BITWISE_CASES {
            assert_windowed_bitwise_equals_with_mask(false, shape, window, kv_offset);
        }
    }

    // Same check for the native-bf16 kernels.
    #[test]
    fn windowed_right_matches_windowed_with_mask_bitwise_bf16() {
        for (shape, window, kv_offset) in WINDOWED_RIGHT_BITWISE_CASES {
            assert_windowed_bitwise_equals_with_mask(true, shape, window, kv_offset);
        }
    }

    // A diagonal reaching past the live keys (`kv_offset > skv - sq`) on a
    // padded GQA shape: the causal entry must fall back to the host mask so
    // the zero K/V padding stays masked.
    #[allow(clippy::many_single_char_names)]
    fn assert_causal_past_live_keys_matches_reference(bf16: bool) {
        let gpu = test_gpu_device();
        let (b, sq, skv, hq, hkv, d) = (1, 20, 200, 8, 2, 128usize);
        // `d` is 128, exact in an f32.
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq + 37;
        assert!(super::causal_needs_host_mask(sq, skv, kv_offset));

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

        let kv_dtype = if bf16 { DType::BF16 } else { DType::F16 };
        let q = q_f32.to_dtype(kv_dtype).unwrap().to_device(&gpu).unwrap();
        let k = k_f32.to_dtype(kv_dtype).unwrap().to_device(&gpu).unwrap();
        let v = v_f32.to_dtype(kv_dtype).unwrap().to_device(&gpu).unwrap();

        let got = if bf16 {
            fattn_mma_bf16_causal(&q, &k, &v, scale, kv_offset)
        } else {
            fattn_mma_causal(&q, &k, &v, scale, kv_offset)
        };
        assert_close(&got_vec(got.unwrap()), &want, 3e-2, "causal_past_live_keys");
    }

    // `fattn_mma_causal` past the live keys matches the reference.
    #[test]
    fn causal_past_live_keys_matches_reference() {
        assert_causal_past_live_keys_matches_reference(false);
    }

    // Same check for `fattn_mma_bf16_causal`.
    #[test]
    fn causal_past_live_keys_matches_reference_bf16() {
        assert_causal_past_live_keys_matches_reference(true);
    }

    // Runs the causal entries on strided Q/K/V views and on contiguous
    // copies of them, and requires identical output bits.
    // `d` is at most 256, exact in an f32.
    #[allow(clippy::many_single_char_names, clippy::cast_precision_loss)]
    fn assert_strided_bitwise_equals_contiguous(
        bf16: bool,
        (b, sq, skv, hq, hkv, d): (usize, usize, usize, usize, usize, usize),
    ) {
        let gpu = test_gpu_device();
        let scale = 1.0f32 / (d as f32).sqrt();
        let kv_offset = skv - sq;
        let kv_dtype = if bf16 { DType::BF16 } else { DType::F16 };
        let bshd = |s: usize, h: usize, dtype: DType| {
            narrowed_bhsd((b, h, s, d), dtype, &gpu)
                .transpose(1, 2)
                .unwrap()
        };
        let q = bshd(sq, hq, kv_dtype);
        let k = bshd(skv, hkv, kv_dtype);
        let v = bshd(skv, hkv, kv_dtype);
        assert!(!q.is_contiguous() && !k.is_contiguous() && !v.is_contiguous());
        let (qc, kc, vc) = (
            q.contiguous().unwrap(),
            k.contiguous().unwrap(),
            v.contiguous().unwrap(),
        );
        let mask = causal_mask(sq, skv, kv_offset, &gpu);

        let run = |q: &Tensor, k: &Tensor, v: &Tensor, with_mask: bool| {
            let out = match (bf16, with_mask) {
                (true, false) => fattn_mma_bf16_causal(q, k, v, scale, kv_offset),
                (true, true) => fattn_mma_bf16_causal_with_mask(q, k, v, scale, kv_offset, &mask),
                (false, false) => fattn_mma_causal(q, k, v, scale, kv_offset),
                (false, true) => fattn_mma_causal_with_mask(q, k, v, scale, kv_offset, &mask),
            };
            got_vec(out.unwrap())
        };
        for with_mask in [false, true] {
            let (got, want) = (run(&q, &k, &v, with_mask), run(&qc, &kc, &vc, with_mask));
            assert_eq!(got.len(), want.len());
            let mismatches = got
                .iter()
                .zip(&want)
                .filter(|(g, w)| g.to_bits() != w.to_bits())
                .count();
            assert_eq!(
                mismatches,
                0,
                "bf16={bf16} with_mask={with_mask} b={b} sq={sq} skv={skv} hq={hq} hkv={hkv} d={d}: {mismatches} of {} elements differ",
                got.len()
            );
        }
    }

    // Covers GQA ratios, head dims and an unaligned seq_kv (231). MHA has
    // no AMD kernel, so it only runs on NVIDIA.
    const STRIDED_SHAPES: [(usize, usize, usize, usize, usize, usize); 4] = [
        (1, 256, 256, 8, 2, 256),
        (2, 64, 256, 8, 2, 128),
        (1, 16, 256, 8, 2, 64),
        (1, 231, 231, 8, 2, 256),
    ];

    // Strided Q/K/V views give the same bits as contiguous copies (f16).
    #[test]
    fn strided_inputs_match_contiguous_bitwise() {
        for shape in STRIDED_SHAPES {
            assert_strided_bitwise_equals_contiguous(false, shape);
        }
        if !is_amd_backend() {
            assert_strided_bitwise_equals_contiguous(false, (1, 64, 64, 4, 4, 128));
        }
    }

    // Same check for the native-bf16 kernels.
    #[test]
    fn strided_inputs_match_contiguous_bitwise_bf16() {
        for shape in STRIDED_SHAPES {
            assert_strided_bitwise_equals_contiguous(true, shape);
        }
        if !is_amd_backend() {
            assert_strided_bitwise_equals_contiguous(true, (1, 64, 64, 4, 4, 128));
        }
    }

    // End-to-end `fattn_mma_windowed`.
    #[test]
    fn windowed_fn_matches_reference_hd128() {
        // hq != hkv (GQA ratio > 1), so ncols2 resolves to > 1 regardless of
        // skv's alignment (see `mma_ncols2`'s doc comment); the WMMA kernel
        // would otherwise reject ncols2 == 1 unconditionally on AMD (see
        // `select_mma_ncols`'s doc comment).
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

        let q = q_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
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

        let q = q_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
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

        let q = q_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
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

/// Parity readout against `candle-flash-attn-rocm` (AMD Composable Kernel),
/// gated on its own feature rather than `gpu_tests`' `cuda`/`rocm` cfg since
/// it pulls in an extra optional dependency only needed for this
/// comparison. Reports max-abs, mean-abs, and max-relative error against
/// the shared F32 naive reference for both kernels, plus crane-vs-CK
/// directly — run with `--nocapture` to see the numbers; the assertions
/// only guard against outright divergence, not "the two differ at all".
#[cfg(all(test, feature = "flash-attn-rocm"))]
mod ck_parity {
    use candle_core::{DType, Device, Tensor};

    use super::super::fattn::test_support::{MaskMode, naive_attention, test_gpu_device};
    use super::{fattn_mma_bf16_causal, fattn_mma_bf16_full, fattn_mma_causal, fattn_mma_full};

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

    struct ErrorStats {
        max_abs: f32,
        mean_abs: f32,
        max_rel: f32,
    }

    impl std::fmt::Display for ErrorStats {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "max_abs={:.6} mean_abs={:.6} max_rel={:.6}",
                self.max_abs, self.mean_abs, self.max_rel
            )
        }
    }

    fn error_stats(got: &[f32], want: &[f32]) -> ErrorStats {
        assert_eq!(got.len(), want.len());
        let mut max_abs = 0f32;
        let mut sum_abs = 0f32;
        let mut max_rel = 0f32;
        for (g, w) in got.iter().zip(want) {
            let abs = (g - w).abs();
            max_abs = max_abs.max(abs);
            sum_abs += abs;
            max_rel = max_rel.max(abs / w.abs().max(1e-6));
        }
        ErrorStats {
            max_abs,
            mean_abs: sum_abs / got.len() as f32,
            max_rel,
        }
    }

    // Non-causal: crane's fattn_mma_full vs. candle-flash-attn-rocm's CK
    // kernel, both against the shared F32 naive reference.
    fn run_full_vs_ck(b: usize, sq: usize, skv: usize, hq: usize, hkv: usize, d: usize) {
        let gpu = test_gpu_device();
        let scale = 1.0f32 / (d as f32).sqrt();

        let q_f32 = Tensor::randn(0f32, 1f32, (b, sq, hq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let want = naive_attention(&q_f32, &k_f32, &v_f32, scale, MaskMode::Full, 0, 0, 0);

        // crane's fattn_mma_full: q/k/v cast to F16.
        let q = q_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let k = k_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let v = v_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let crane_out = got_vec(fattn_mma_full(&q, &k, &v, scale).unwrap());

        // candle-flash-attn-rocm requires q/k/v at a uniform dtype.
        let q_f16 = q_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let ck_out =
            got_vec(candle_flash_attn_rocm::flash_attn(&q_f16, &k, &v, scale, false).unwrap());

        let crane_vs_naive = error_stats(&crane_out, &want);
        let ck_vs_naive = error_stats(&ck_out, &want);
        let crane_vs_ck = error_stats(&crane_out, &ck_out);
        println!(
            "full head_dim={d}: crane_vs_naive=({crane_vs_naive}) ck_vs_naive=({ck_vs_naive}) crane_vs_ck=({crane_vs_ck})"
        );

        assert!(
            crane_vs_naive.max_abs <= 2e-3,
            "crane fattn_mma_full diverged from naive reference: {crane_vs_naive}"
        );
        assert!(
            ck_vs_naive.max_abs <= 2e-3,
            "candle-flash-attn-rocm diverged from naive reference: {ck_vs_naive}"
        );
    }

    fn run_causal_vs_ck(b: usize, sq: usize, skv: usize, hq: usize, hkv: usize, d: usize) {
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

        let q = q_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let k = k_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let v = v_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let crane_out = got_vec(fattn_mma_causal(&q, &k, &v, scale, kv_offset).unwrap());

        let q_f16 = q_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let ck_out =
            got_vec(candle_flash_attn_rocm::flash_attn(&q_f16, &k, &v, scale, true).unwrap());

        let crane_vs_naive = error_stats(&crane_out, &want);
        let ck_vs_naive = error_stats(&ck_out, &want);
        let crane_vs_ck = error_stats(&crane_out, &ck_out);
        println!(
            "causal head_dim={d}: crane_vs_naive=({crane_vs_naive}) ck_vs_naive=({ck_vs_naive}) crane_vs_ck=({crane_vs_ck})"
        );

        assert!(
            crane_vs_naive.max_abs <= 2e-3,
            "crane fattn_mma_causal diverged from naive reference: {crane_vs_naive}"
        );
        assert!(
            ck_vs_naive.max_abs <= 2e-3,
            "candle-flash-attn-rocm diverged from naive reference: {ck_vs_naive}"
        );
    }

    #[test]
    fn full_vs_ck_hd64() {
        run_full_vs_ck(1, 32, 256, 4, 2, 64);
    }

    #[test]
    fn full_vs_ck_hd128() {
        run_full_vs_ck(1, 32, 256, 4, 2, 128);
    }

    #[test]
    fn full_vs_ck_hd256() {
        run_full_vs_ck(1, 32, 256, 4, 2, 256);
    }

    #[test]
    fn causal_vs_ck_hd64() {
        run_causal_vs_ck(1, 32, 256, 4, 2, 64);
    }

    #[test]
    fn causal_vs_ck_hd128() {
        run_causal_vs_ck(1, 32, 256, 4, 2, 128);
    }

    #[test]
    fn causal_vs_ck_hd256() {
        run_causal_vs_ck(1, 32, 256, 4, 2, 256);
    }

    // Native-bf16 counterparts: both kernels run end-to-end in BF16 (no F16
    // intermediate), the dtype the maintainer's 1e-4 claim is about. Still
    // measured against the F32 naive reference, since that's the only
    // ground truth available in this repo (see this module's doc comment:
    // there is no torch/FLA/causalconv1d fixture checked in anywhere to
    // compare against directly).
    fn run_full_vs_ck_bf16(b: usize, sq: usize, skv: usize, hq: usize, hkv: usize, d: usize) {
        let gpu = test_gpu_device();
        let scale = 1.0f32 / (d as f32).sqrt();

        let q_f32 = Tensor::randn(0f32, 1f32, (b, sq, hq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let want = naive_attention(&q_f32, &k_f32, &v_f32, scale, MaskMode::Full, 0, 0, 0);

        // crane's fattn_mma_bf16_full: q/k/v cast to BF16.
        let q = q_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let k = k_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let v = v_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let crane_out = got_vec(fattn_mma_bf16_full(&q, &k, &v, scale).unwrap());

        // candle-flash-attn-rocm requires q/k/v at a uniform dtype.
        let q_bf16 = q_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let ck_out =
            got_vec(candle_flash_attn_rocm::flash_attn(&q_bf16, &k, &v, scale, false).unwrap());

        let crane_vs_naive = error_stats(&crane_out, &want);
        let ck_vs_naive = error_stats(&ck_out, &want);
        let crane_vs_ck = error_stats(&crane_out, &ck_out);
        println!(
            "full_bf16 head_dim={d}: crane_vs_naive=({crane_vs_naive}) ck_vs_naive=({ck_vs_naive}) crane_vs_ck=({crane_vs_ck})"
        );

        assert!(
            crane_vs_naive.max_abs <= 1e-2,
            "crane fattn_mma_bf16_full diverged from naive reference: {crane_vs_naive}"
        );
        assert!(
            ck_vs_naive.max_abs <= 1e-2,
            "candle-flash-attn-rocm (bf16) diverged from naive reference: {ck_vs_naive}"
        );
    }

    fn run_causal_vs_ck_bf16(b: usize, sq: usize, skv: usize, hq: usize, hkv: usize, d: usize) {
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

        let q = q_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let k = k_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let v = v_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let crane_out = got_vec(fattn_mma_bf16_causal(&q, &k, &v, scale, kv_offset).unwrap());

        let q_bf16 = q_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let ck_out =
            got_vec(candle_flash_attn_rocm::flash_attn(&q_bf16, &k, &v, scale, true).unwrap());

        let crane_vs_naive = error_stats(&crane_out, &want);
        let ck_vs_naive = error_stats(&ck_out, &want);
        let crane_vs_ck = error_stats(&crane_out, &ck_out);
        println!(
            "causal_bf16 head_dim={d}: crane_vs_naive=({crane_vs_naive}) ck_vs_naive=({ck_vs_naive}) crane_vs_ck=({crane_vs_ck})"
        );

        assert!(
            crane_vs_naive.max_abs <= 1e-2,
            "crane fattn_mma_bf16_causal diverged from naive reference: {crane_vs_naive}"
        );
        assert!(
            ck_vs_naive.max_abs <= 1e-2,
            "candle-flash-attn-rocm (bf16) diverged from naive reference: {ck_vs_naive}"
        );
    }

    #[test]
    fn full_vs_ck_bf16_hd64() {
        run_full_vs_ck_bf16(1, 32, 256, 4, 2, 64);
    }

    #[test]
    fn full_vs_ck_bf16_hd128() {
        run_full_vs_ck_bf16(1, 32, 256, 4, 2, 128);
    }

    #[test]
    fn full_vs_ck_bf16_hd256() {
        run_full_vs_ck_bf16(1, 32, 256, 4, 2, 256);
    }

    #[test]
    fn causal_vs_ck_bf16_hd64() {
        run_causal_vs_ck_bf16(1, 32, 256, 4, 2, 64);
    }

    #[test]
    fn causal_vs_ck_bf16_hd128() {
        run_causal_vs_ck_bf16(1, 32, 256, 4, 2, 128);
    }

    #[test]
    fn causal_vs_ck_bf16_hd256() {
        run_causal_vs_ck_bf16(1, 32, 256, 4, 2, 256);
    }
}

/// Parity readout against `candle-flash-attn` (the upstream NVIDIA
/// flash-attn2 CUDA kernel), gated on `flash-attn-ab`, the one feature that
/// builds both CUDA backends. Exists so a claim about flash-attn2's own
/// accuracy can be checked directly on CUDA hardware instead of taken on
/// faith: this repo's CI and this dev environment are ROCm-only, so these
/// tests cannot run here. Same structure, error metrics, and F32 naive
/// reference as [`ck_parity`] above, just a different reference kernel and
/// backend. Run with `--nocapture` to see the numbers.
#[cfg(all(test, feature = "flash-attn-ab"))]
mod flashattn2_parity {
    use candle_core::{DType, Device, Tensor};

    use super::super::fattn::test_support::{MaskMode, naive_attention, test_gpu_device};
    use super::{fattn_mma_bf16_causal, fattn_mma_bf16_full, fattn_mma_causal, fattn_mma_full};

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

    struct ErrorStats {
        max_abs: f32,
        mean_abs: f32,
        max_rel: f32,
    }

    impl std::fmt::Display for ErrorStats {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "max_abs={:.6} mean_abs={:.6} max_rel={:.6}",
                self.max_abs, self.mean_abs, self.max_rel
            )
        }
    }

    fn error_stats(got: &[f32], want: &[f32]) -> ErrorStats {
        assert_eq!(got.len(), want.len());
        let mut max_abs = 0f32;
        let mut sum_abs = 0f32;
        let mut max_rel = 0f32;
        for (g, w) in got.iter().zip(want) {
            let abs = (g - w).abs();
            max_abs = max_abs.max(abs);
            sum_abs += abs;
            max_rel = max_rel.max(abs / w.abs().max(1e-6));
        }
        ErrorStats {
            max_abs,
            mean_abs: sum_abs / got.len() as f32,
            max_rel,
        }
    }

    // Non-causal: crane's fattn_mma_full vs. candle-flash-attn's flash-attn2
    // kernel, both against the shared F32 naive reference.
    fn run_full_vs_flashattn2(b: usize, sq: usize, skv: usize, hq: usize, hkv: usize, d: usize) {
        let gpu = test_gpu_device();
        let scale = 1.0f32 / (d as f32).sqrt();

        let q_f32 = Tensor::randn(0f32, 1f32, (b, sq, hq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let want = naive_attention(&q_f32, &k_f32, &v_f32, scale, MaskMode::Full, 0, 0, 0);

        // crane's fattn_mma_full: q/k/v cast to F16.
        let q = q_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let k = k_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let v = v_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let crane_out = got_vec(fattn_mma_full(&q, &k, &v, scale).unwrap());

        // candle-flash-attn requires q/k/v at a uniform dtype.
        let q_f16 = q_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let fa2_out = got_vec(candle_flash_attn::flash_attn(&q_f16, &k, &v, scale, false).unwrap());

        let crane_vs_naive = error_stats(&crane_out, &want);
        let fa2_vs_naive = error_stats(&fa2_out, &want);
        let crane_vs_fa2 = error_stats(&crane_out, &fa2_out);
        println!(
            "full head_dim={d}: crane_vs_naive=({crane_vs_naive}) flashattn2_vs_naive=({fa2_vs_naive}) crane_vs_flashattn2=({crane_vs_fa2})"
        );

        assert!(
            crane_vs_naive.max_abs <= 2e-3,
            "crane fattn_mma_full diverged from naive reference: {crane_vs_naive}"
        );
        assert!(
            fa2_vs_naive.max_abs <= 2e-3,
            "candle-flash-attn (flash-attn2) diverged from naive reference: {fa2_vs_naive}"
        );
    }

    fn run_causal_vs_flashattn2(b: usize, sq: usize, skv: usize, hq: usize, hkv: usize, d: usize) {
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

        let q = q_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let k = k_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let v = v_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let crane_out = got_vec(fattn_mma_causal(&q, &k, &v, scale, kv_offset).unwrap());

        let q_f16 = q_f32.to_dtype(DType::F16).unwrap().to_device(&gpu).unwrap();
        let fa2_out = got_vec(candle_flash_attn::flash_attn(&q_f16, &k, &v, scale, true).unwrap());

        let crane_vs_naive = error_stats(&crane_out, &want);
        let fa2_vs_naive = error_stats(&fa2_out, &want);
        let crane_vs_fa2 = error_stats(&crane_out, &fa2_out);
        println!(
            "causal head_dim={d}: crane_vs_naive=({crane_vs_naive}) flashattn2_vs_naive=({fa2_vs_naive}) crane_vs_flashattn2=({crane_vs_fa2})"
        );

        assert!(
            crane_vs_naive.max_abs <= 2e-3,
            "crane fattn_mma_causal diverged from naive reference: {crane_vs_naive}"
        );
        assert!(
            fa2_vs_naive.max_abs <= 2e-3,
            "candle-flash-attn (flash-attn2) diverged from naive reference: {fa2_vs_naive}"
        );
    }

    #[test]
    fn full_vs_flashattn2_hd64() {
        run_full_vs_flashattn2(1, 32, 256, 4, 2, 64);
    }

    #[test]
    fn full_vs_flashattn2_hd128() {
        run_full_vs_flashattn2(1, 32, 256, 4, 2, 128);
    }

    #[test]
    fn full_vs_flashattn2_hd256() {
        run_full_vs_flashattn2(1, 32, 256, 4, 2, 256);
    }

    #[test]
    fn causal_vs_flashattn2_hd64() {
        run_causal_vs_flashattn2(1, 32, 256, 4, 2, 64);
    }

    #[test]
    fn causal_vs_flashattn2_hd128() {
        run_causal_vs_flashattn2(1, 32, 256, 4, 2, 128);
    }

    #[test]
    fn causal_vs_flashattn2_hd256() {
        run_causal_vs_flashattn2(1, 32, 256, 4, 2, 256);
    }

    // Native-bf16 counterparts: both kernels run end-to-end in BF16 (no F16
    // intermediate), the dtype the maintainer's 1e-4 claim is about. Still
    // measured against the F32 naive reference, since that's the only
    // ground truth available in this repo.
    fn run_full_vs_flashattn2_bf16(
        b: usize,
        sq: usize,
        skv: usize,
        hq: usize,
        hkv: usize,
        d: usize,
    ) {
        let gpu = test_gpu_device();
        let scale = 1.0f32 / (d as f32).sqrt();

        let q_f32 = Tensor::randn(0f32, 1f32, (b, sq, hq, d), &Device::Cpu).unwrap();
        let k_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let v_f32 = Tensor::randn(0f32, 1f32, (b, skv, hkv, d), &Device::Cpu).unwrap();
        let want = naive_attention(&q_f32, &k_f32, &v_f32, scale, MaskMode::Full, 0, 0, 0);

        // crane's fattn_mma_bf16_full: q/k/v cast to BF16.
        let q = q_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let k = k_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let v = v_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let crane_out = got_vec(fattn_mma_bf16_full(&q, &k, &v, scale).unwrap());

        // candle-flash-attn requires q/k/v at a uniform dtype.
        let q_bf16 = q_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let fa2_out =
            got_vec(candle_flash_attn::flash_attn(&q_bf16, &k, &v, scale, false).unwrap());

        let crane_vs_naive = error_stats(&crane_out, &want);
        let fa2_vs_naive = error_stats(&fa2_out, &want);
        let crane_vs_fa2 = error_stats(&crane_out, &fa2_out);
        println!(
            "full_bf16 head_dim={d}: crane_vs_naive=({crane_vs_naive}) flashattn2_vs_naive=({fa2_vs_naive}) crane_vs_flashattn2=({crane_vs_fa2})"
        );

        assert!(
            crane_vs_naive.max_abs <= 1e-2,
            "crane fattn_mma_bf16_full diverged from naive reference: {crane_vs_naive}"
        );
        assert!(
            fa2_vs_naive.max_abs <= 1e-2,
            "candle-flash-attn (bf16) diverged from naive reference: {fa2_vs_naive}"
        );
    }

    fn run_causal_vs_flashattn2_bf16(
        b: usize,
        sq: usize,
        skv: usize,
        hq: usize,
        hkv: usize,
        d: usize,
    ) {
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

        let q = q_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let k = k_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let v = v_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let crane_out = got_vec(fattn_mma_bf16_causal(&q, &k, &v, scale, kv_offset).unwrap());

        let q_bf16 = q_f32
            .to_dtype(DType::BF16)
            .unwrap()
            .to_device(&gpu)
            .unwrap();
        let fa2_out = got_vec(candle_flash_attn::flash_attn(&q_bf16, &k, &v, scale, true).unwrap());

        let crane_vs_naive = error_stats(&crane_out, &want);
        let fa2_vs_naive = error_stats(&fa2_out, &want);
        let crane_vs_fa2 = error_stats(&crane_out, &fa2_out);
        println!(
            "causal_bf16 head_dim={d}: crane_vs_naive=({crane_vs_naive}) flashattn2_vs_naive=({fa2_vs_naive}) crane_vs_flashattn2=({crane_vs_fa2})"
        );

        assert!(
            crane_vs_naive.max_abs <= 1e-2,
            "crane fattn_mma_bf16_causal diverged from naive reference: {crane_vs_naive}"
        );
        assert!(
            fa2_vs_naive.max_abs <= 1e-2,
            "candle-flash-attn (bf16) diverged from naive reference: {fa2_vs_naive}"
        );
    }

    #[test]
    fn full_vs_flashattn2_bf16_hd64() {
        run_full_vs_flashattn2_bf16(1, 32, 256, 4, 2, 64);
    }

    #[test]
    fn full_vs_flashattn2_bf16_hd128() {
        run_full_vs_flashattn2_bf16(1, 32, 256, 4, 2, 128);
    }

    #[test]
    fn full_vs_flashattn2_bf16_hd256() {
        run_full_vs_flashattn2_bf16(1, 32, 256, 4, 2, 256);
    }

    #[test]
    fn causal_vs_flashattn2_bf16_hd64() {
        run_causal_vs_flashattn2_bf16(1, 32, 256, 4, 2, 64);
    }

    #[test]
    fn causal_vs_flashattn2_bf16_hd128() {
        run_causal_vs_flashattn2_bf16(1, 32, 256, 4, 2, 128);
    }

    #[test]
    fn causal_vs_flashattn2_bf16_hd256() {
        run_causal_vs_flashattn2_bf16(1, 32, 256, 4, 2, 256);
    }
}

#[cfg(all(test, crane_fattn))]
#[path = "fattn_long_seq_tests.rs"]
mod long_seq_tests;

#[cfg(all(test, crane_fattn))]
#[path = "fattn_golden_tests.rs"]
mod golden_tests;

#[cfg(all(test, crane_fattn))]
#[path = "fattn_bench_tests.rs"]
mod bench_tests;

#[cfg(all(test, crane_fattn))]
#[path = "fattn_stream_k_tests.rs"]
mod stream_k_tests;
