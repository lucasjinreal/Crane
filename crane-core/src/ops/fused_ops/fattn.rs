// SPDX-License-Identifier: MIT
//! Shared types, per-architecture tuning tables, and kernel-selection logic
//! for the vendored flash-attention MMA ([`super::fattn_mma`]) and tile
//! ([`super::fattn_tile`]) kernel dispatchers. Supports (see
//! `kernels/cuda/fattn/`'s module docs): `parallel_blocks=1`, stream-K for the
//! MMA kernels, F16 and BF16, `head_dim` in {64, 128, 256, 512}.
//!
//! Ports llama.cpp's `ggml_cuda_fattn_mma_get_config_{ampere,rdna}` tuning
//! tables (`kernels/cuda/fattn/fattn_mma_f16.cuh`) and its
//! `ggml_cuda_flash_attn_ext_mma_f16_switch_{ncols1,ncols2}` kernel-size
//! selection (llama.cpp's `fattn.cu`, not vendored; it builds a
//! `ggml_tensor` dispatch, replaced entirely here), scoped to the
//! architectures Crane targets (NVIDIA Ampere+, AMD RDNA3+) and the head
//! dims currently instantiated. Does not port the Turing/Volta/CDNA tables,
//! the sparse-mask path, or `max_bias`/`ALiBi`: nothing here ever sets a
//! nonzero `max_bias`, so `get_alibi_slope` in `crane_fattn_shim.cuh` always
//! returns `1.0` regardless of `n_head_log2`.
//!
//! Every item here exists solely to support the CUDA/ROCm kernel
//! dispatchers in [`super::fattn_mma`]/[`super::fattn_tile`] (unlike
//! `flash_attn.rs`, there is no CPU-relevant content to keep unconditional),
//! so the whole module is gated at once rather than per item.
#![cfg(crane_fattn)]

use candle_core::{DType, Device, Layout, Result, Tensor};

/// Prefix marking a candle error built by [`unsupported_config`]. It means
/// this exact shape/GQA-ratio/mask combination has no instantiated kernel.
/// That is an expected static limitation the caller should silently fall
/// back on. It is never a sign of a real runtime failure (a launch
/// rejection, OOM, an invalid argument, ...). `gpu_flash_attn::run`'s
/// dispatch chain checks this prefix (via [`is_unsupported_config`]) to
/// decide whether to try the next kernel tier or propagate the error
/// untouched. Swallowing every
/// `mma`/`mma_bf16` error regardless of cause would hide genuine
/// kernel-execution failures behind the tile-kernel fallback.
pub(crate) const UNSUPPORTED_CONFIG_PREFIX: &str = "crane_fattn_unsupported_config: ";

/// Builds a candle error tagged with [`UNSUPPORTED_CONFIG_PREFIX`]. Use
/// only at call sites where the failure is purely a function of the input
/// shape (no instantiated kernel for this `(head_dim, gqa_ratio, seq_kv)`,
/// a GQA shape `select_mma_ncols` can't batch, a tile grid too large for
/// this `seq_kv`), never for a validation bug or an actual launch/runtime
/// error.
pub(crate) fn unsupported_config(msg: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("{UNSUPPORTED_CONFIG_PREFIX}{msg}"))
}

/// True when `err` was built via [`unsupported_config`]. See that
/// function's doc comment.
pub(crate) fn is_unsupported_config(err: &candle_core::Error) -> bool {
    err.to_string().starts_with(UNSUPPORTED_CONFIG_PREFIX)
}

/// Byte-for-byte layout of CUDA/HIP's built-in `uint3` (`struct { unsigned
/// int x, y, z; }`, 12 bytes, 4-byte aligned, no padding to 16 the way
/// `uint4` has). The kernel's `ne01` parameter is declared `const uint3
/// ne01`, a single 12-byte struct argument, not three independent `u32`
/// parameters. So callers must push exactly one argument of this type, not
/// three separate scalars: pushing three `u32`s shifts every later
/// parameter in the launch's argument list by the struct's size, corrupting
/// the entire call (confirmed by a real "HSAIL operation resulted in a
/// hardware exception" on real `ROCm` hardware before this fix).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Uint3 {
    /// Barrett-reduction multiplier (`mp`).
    pub(crate) x: u32,
    /// Barrett-reduction shift (`L`).
    pub(crate) y: u32,
    /// The divisor itself (`d`), kept alongside `mp`/`L` for the device-side
    /// fastmodulo step.
    pub(crate) z: u32,
}

// SAFETY: `Uint3` is `#[repr(C)]`, three plain `u32`s with no padding and no
// pointers, so copying its bytes into the kernel's argument buffer (what
// `DeviceRepr`'s default `as_kernel_param` does) is exactly what the
// device-side `const uint3 ne01` parameter expects.
#[cfg(feature = "cuda")]
unsafe impl candle_core::cuda_backend::cudarc::driver::DeviceRepr for Uint3 {}

/// Which vendored kernel family a call should dispatch to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FattnKernelType {
    /// `crane_fattn_mma_f16_d{dkq}_v{dkq}_c{ncols1}_s{ncols2}`.
    MmaF16,
    /// `crane_fattn_tile_f16_d{dkq}_v{dkq}_c16_s1`.
    Tile,
}

/// `FATTN_KQ_STRIDE` in `fattn_common.cuh`: KV rows per tile, and the
/// rounding granularity [`build_analytic_kv_max`]'s `kv_max_cap` uses for its
/// tile bounds.
pub(crate) const FATTN_KQ_STRIDE: usize = 256;

/// Mirrors llama.cpp's `fattn_mma_config` (`fattn_mma_f16.cuh`): per-launch
/// tuning knobs selected by `(head_dim, ncols1*ncols2)`. Fields map directly
/// to the kernel's `__launch_bounds__`/shared-memory sizing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FattnMmaConfig {
    /// Threads per block (`block_dim.x * block_dim.y`).
    pub nthreads: usize,
    /// Targeted blocks per SM/CU, the kernel's `__launch_bounds__` hint. It
    /// sizes the stream-K grid (see [`stream_k_grid`]).
    pub occupancy: usize,
    /// KV rows processed per inner-loop iteration. Must match the
    /// kernel's compile-time `NBATCH_FA` (set by each `.cuh` config
    /// macro); a mismatch causes wrong tail-kernel selection and
    /// out-of-bounds device reads since K/V is never padded.
    pub nbatch_fa: usize,
    /// K rows staged in shared memory per `nbatch_fa` chunk.
    pub nbatch_k2: usize,
    /// V rows staged in shared memory per `nbatch_fa` chunk.
    pub nbatch_v2: usize,
    /// Row-tile width used by the softmax/output combine pass.
    pub nbatch_combine: usize,
    /// Target pipeline stage count for `cp.async` prefetching (0 disables
    /// it; only ever nonzero on NVIDIA, see [`mma_shared_mem_bytes`]).
    pub nstages_target: usize,
    /// Whether Q is kept resident in registers rather than shared memory.
    pub q_in_reg: bool,
}

/// NVIDIA Ampere+ tuning table
/// (`ggml_cuda_fattn_mma_get_config_ampere`/`_turing`, the two are
/// identical for the supported head dims, see `fattn_mma_f16.cuh`), scoped to
/// `head_dim` in {64, 128, 256, 512} and `ncols` (`ncols1*ncols2`) in {8, 16,
/// 32, 64}. Returns `None` for any other combination.
#[rustfmt::skip]
fn mma_config_ampere(dkq: usize, ncols: usize) -> Option<FattnMmaConfig> {
    use FattnMmaConfig as C;
    Some(match (dkq, ncols) {
        (64, 8) => C { nthreads: 128, occupancy: 2, nbatch_fa: 128, nbatch_k2: 32, nbatch_v2: 32, nbatch_combine: 32, nstages_target: 2, q_in_reg: true },
        (64, 16 | 32 | 64) => C { nthreads: 128, occupancy: 2, nbatch_fa: 64, nbatch_k2: 32, nbatch_v2: 32, nbatch_combine: 32, nstages_target: 2, q_in_reg: true },
        (128, 8) => C { nthreads: 128, occupancy: 2, nbatch_fa: 128, nbatch_k2: 64, nbatch_v2: 64, nbatch_combine: 64, nstages_target: 2, q_in_reg: true },
        (128, 16 | 32 | 64) => C { nthreads: 128, occupancy: 2, nbatch_fa: 64, nbatch_k2: 64, nbatch_v2: 64, nbatch_combine: 64, nstages_target: 2, q_in_reg: true },
        (256, 8) => C { nthreads: 128, occupancy: 2, nbatch_fa: 64, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 128, nstages_target: 2, q_in_reg: true },
        (256, 16) => C { nthreads: 64, occupancy: 4, nbatch_fa: 32, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 128, nstages_target: 2, q_in_reg: true },
        (256, 32 | 64) => C { nthreads: 128, occupancy: 2, nbatch_fa: 32, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 128, nstages_target: 2, q_in_reg: true },
        (512, 8 | 16) => C { nthreads: 64, occupancy: 4, nbatch_fa: 32, nbatch_k2: 256, nbatch_v2: 256, nbatch_combine: 128, nstages_target: 1, q_in_reg: false },
        (512, 32) => C { nthreads: 128, occupancy: 2, nbatch_fa: 32, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 128, nstages_target: 1, q_in_reg: false },
        (512, 64) => C { nthreads: 256, occupancy: 1, nbatch_fa: 32, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 128, nstages_target: 1, q_in_reg: false },
        _ => return None,
    })
}

/// AMD RDNA3+ (WMMA) tuning table (`ggml_cuda_fattn_mma_get_config_rdna`),
/// same scope as [`mma_config_ampere`].
#[rustfmt::skip]
fn mma_config_rdna(dkq: usize, ncols: usize) -> Option<FattnMmaConfig> {
    use FattnMmaConfig as C;
    Some(match (dkq, ncols) {
        (64, 8 | 16 | 32 | 64) => C { nthreads: 128, occupancy: 2, nbatch_fa: 64, nbatch_k2: 32, nbatch_v2: 32, nbatch_combine: 32, nstages_target: 1, q_in_reg: true },
        (128, 8 | 16) => C { nthreads: 64, occupancy: 2, nbatch_fa: 32, nbatch_k2: 64, nbatch_v2: 64, nbatch_combine: 64, nstages_target: 1, q_in_reg: true },
        (128, 32 | 64) => C { nthreads: 128, occupancy: 2, nbatch_fa: 64, nbatch_k2: 64, nbatch_v2: 64, nbatch_combine: 64, nstages_target: 1, q_in_reg: true },
        (256, 8 | 16) => C { nthreads: 64, occupancy: 2, nbatch_fa: 32, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 128, nstages_target: 1, q_in_reg: true },
        (256, 32) => C { nthreads: 256, occupancy: 2, nbatch_fa: 64, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 64, nstages_target: 1, q_in_reg: true },
        // Q in LDS to cut register spill, with nbatch_fa halved to fit the
        // 64 KB LDS limit (see the matching `.cuh` entry). Long KV uses
        // `mma_config_rdna_long_kv` instead.
        (256, 64) => C { nthreads: 256, occupancy: 2, nbatch_fa: 32, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 64, nstages_target: 1, q_in_reg: false },
        (512, 8 | 16) => C { nthreads: 128, occupancy: 3, nbatch_fa: 64, nbatch_k2: 96, nbatch_v2: 64, nbatch_combine: 128, nstages_target: 1, q_in_reg: true },
        (512, 32 | 64) => C { nthreads: 128, occupancy: 2, nbatch_fa: 32, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 128, nstages_target: 1, q_in_reg: true },
        _ => return None,
    })
}

/// Looks up the tuning config for `head_dim`/`ncols1*ncols2` on the active
/// backend (`is_amd = false` selects the NVIDIA Ampere+ table).
///
/// # Errors
///
/// Returns an error if no config exists for this `(dkq, ncols)` pair.
/// This means [`select_mma_ncols`] picked a combination with no instantiated
/// kernel, a bug in that function rather than an expected runtime
/// condition.
pub(crate) fn mma_config(dkq: usize, ncols: usize, is_amd: bool) -> Result<FattnMmaConfig> {
    let cfg = if is_amd {
        mma_config_rdna(dkq, ncols)
    } else {
        mma_config_ampere(dkq, ncols)
    };
    cfg.ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "fattn: no MMA config for head_dim={dkq} ncols={ncols} ({})",
            if is_amd { "RDNA" } else { "Ampere" }
        ))
    })
}

/// bf16 counterpart of [`mma_config_rdna`]: identical to the F16 RDNA table
/// at every `(dkq, ncols)` pair except `(128, 32 | 64)`, where `nbatch_fa`
/// is halved (64 -> 32). Experimental: bf16's PV accumulator (`T_C_VKQ` in
/// `fattn_mma_bf16.cuh`) is always `float`, twice the register footprint of
/// F16's packed `half2` accumulator at this `(head_dim, ncols)` shape (see
/// that file's module doc), so halving `nbatch_fa` trades tile size for
/// occupancy to compensate. Kept only because
/// `gpu_flash_attn::gpu_tests::bf16_vs_f16_try_causal_with_mask_timing_sweep_native`
/// measured it as a net win; revert to literally reusing [`mma_config_rdna`]
/// if a future change to the bf16 kernel body changes that balance.
#[rustfmt::skip]
fn mma_config_bf16_rdna(dkq: usize, ncols: usize) -> Option<FattnMmaConfig> {
    use FattnMmaConfig as C;
    Some(match (dkq, ncols) {
        (64, 8 | 16 | 32 | 64) => C { nthreads: 128, occupancy: 2, nbatch_fa: 64, nbatch_k2: 32, nbatch_v2: 32, nbatch_combine: 32, nstages_target: 1, q_in_reg: true },
        (128, 8 | 16) => C { nthreads: 64, occupancy: 2, nbatch_fa: 32, nbatch_k2: 64, nbatch_v2: 64, nbatch_combine: 64, nstages_target: 1, q_in_reg: true },
        (128, 32 | 64) => C { nthreads: 128, occupancy: 2, nbatch_fa: 32, nbatch_k2: 64, nbatch_v2: 64, nbatch_combine: 64, nstages_target: 1, q_in_reg: true },
        (256, 8 | 16) => C { nthreads: 64, occupancy: 2, nbatch_fa: 32, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 128, nstages_target: 1, q_in_reg: true },
        (256, 32) => C { nthreads: 256, occupancy: 2, nbatch_fa: 64, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 64, nstages_target: 1, q_in_reg: true },
        (256, 64) => C { nthreads: 256, occupancy: 2, nbatch_fa: 32, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 64, nstages_target: 1, q_in_reg: false },
        (512, 8 | 16) => C { nthreads: 128, occupancy: 3, nbatch_fa: 64, nbatch_k2: 96, nbatch_v2: 64, nbatch_combine: 128, nstages_target: 1, q_in_reg: true },
        (512, 32 | 64) => C { nthreads: 128, occupancy: 2, nbatch_fa: 32, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 128, nstages_target: 1, q_in_reg: true },
        _ => return None,
    })
}

/// bf16 counterpart of [`mma_config_ampere`]: currently an exact copy (the
/// RDNA table is where [`mma_config_bf16_rdna`] differs); kept separate so
/// an NVIDIA-specific bf16 tuning change doesn't require touching the F16
/// table either.
#[rustfmt::skip]
fn mma_config_bf16_ampere(dkq: usize, ncols: usize) -> Option<FattnMmaConfig> {
    mma_config_ampere(dkq, ncols)
}

/// bf16 counterpart of [`mma_config`]: looks up [`mma_config_bf16_rdna`]/
/// [`mma_config_bf16_ampere`] instead of the F16 tables.
///
/// # Errors
///
/// Returns an error if no config exists for this `(dkq, ncols)` pair (see
/// [`mma_config`]'s doc comment).
pub(crate) fn mma_config_bf16(dkq: usize, ncols: usize, is_amd: bool) -> Result<FattnMmaConfig> {
    let cfg = if is_amd {
        mma_config_bf16_rdna(dkq, ncols)
    } else {
        mma_config_bf16_ampere(dkq, ncols)
    };
    cfg.ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "fattn_mma_bf16: no MMA config for head_dim={dkq} ncols={ncols} ({})",
            if is_amd { "RDNA" } else { "Ampere" }
        ))
    })
}

/// Which compiled kernel set an MMA launch uses.
///
/// The RDNA hd256 64-column tile has two configs, and neither wins at every
/// KV length. See [`mma_config_set`] for the policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum MmaConfigSet {
    /// The tuning tables' own entry, in the module built from the
    /// head-dim's regular instance file.
    Default,
    /// The long-KV config of [`mma_config_rdna_long_kv`], in a separate
    /// module built from the `*_long_kv` instance file. Same kernel names.
    LongKv,
}

/// `seq_kv` from which the RDNA hd256 64-column tile may switch to the
/// long-KV set. Below it the default set wins, because the long set's
/// register spill costs more than the loop iterations it saves.
///
/// Measured on gfx1201 only.
pub(crate) const RDNA_HD256_LONG_KV_MIN_SEQ_KV: usize = 4096;

/// Bytes of K and V that every query tile reads, from which the default set
/// wins again. Past it the long set's 64-row KV tiles stream K and V through
/// the last-level cache less well than the default's 32-row tiles, and a
/// chunked prefill (small `seq_q`, large `seq_kv`) runs 4 to 26% slower on
/// it.
///
/// Measured on gfx1201 only, where the Infinity Cache is 64 MiB. The cliff
/// sits between 44 and 56 MiB, so the limit is 48 MiB. A GPU with a smaller
/// cache probably hits it earlier. This policy then keeps the long set where
/// the default set would win, which is the pre-tuning behavior. A GPU with a
/// larger cache probably hits it later. This policy then takes the default
/// set where the long set would win by about 5%.
pub(crate) const RDNA_HD256_LONG_KV_MAX_FLOOR_BYTES: usize = 48 << 20;

/// Whether a long-KV kernel set exists for this launch: only the RDNA hd256
/// 64-column tile has one.
pub(crate) fn has_long_kv_set(dkq: usize, ncols: usize, is_amd: bool) -> bool {
    is_amd && dkq == 256 && ncols == 64
}

/// Picks the kernel set for a launch of `ncols = ncols1 * ncols2` columns
/// over `seq_kv` keys and `h_kv` KV heads.
///
/// `min_kv` is the number of leading K/V rows every query tile reads: the
/// causal offset for in-kernel causal masking, and `0` for any launch that
/// does not know it. An all-rows prefill has `min_kv = 0`, since the first
/// query tiles read only a short prefix. A chunked prefill reads at least
/// `seq_kv - seq_q` rows in every tile.
///
/// The default set keeps Q in LDS with 32-row KV tiles, which avoids the
/// register spill and wins on short KV. The long-KV set wins from
/// [`RDNA_HD256_LONG_KV_MIN_SEQ_KV`] on, unless the `min_kv` rows of K and V
/// already reach [`RDNA_HD256_LONG_KV_MAX_FLOOR_BYTES`]. Every other launch
/// uses [`MmaConfigSet::Default`].
pub(crate) fn mma_config_set(
    dkq: usize,
    ncols: usize,
    (seq_kv, min_kv): (usize, usize),
    h_kv: usize,
    is_amd: bool,
) -> MmaConfigSet {
    // K and V: 4 bytes per (row, head, column).
    let floor_bytes = min_kv
        .saturating_mul(h_kv)
        .saturating_mul(dkq)
        .saturating_mul(4);
    if has_long_kv_set(dkq, ncols, is_amd)
        && seq_kv >= RDNA_HD256_LONG_KV_MIN_SEQ_KV
        && floor_bytes < RDNA_HD256_LONG_KV_MAX_FLOOR_BYTES
    {
        MmaConfigSet::LongKv
    } else {
        MmaConfigSet::Default
    }
}

/// The long-KV config for both dtypes: Q in registers with 64-row KV tiles.
/// It must match the `CRANE_FATTN_LONG_KV` branch of the RDNA tables in
/// `fattn_mma_f16.cuh` and `fattn_mma_bf16.cuh`, because a mismatched
/// `nbatch_fa` picks the wrong tail kernel and reads out of bounds.
///
/// Returns `None` unless a long-KV set exists for `(dkq, ncols)`, see
/// [`has_long_kv_set`].
#[rustfmt::skip]
pub(crate) fn mma_config_rdna_long_kv(dkq: usize, ncols: usize) -> Option<FattnMmaConfig> {
    use FattnMmaConfig as C;
    has_long_kv_set(dkq, ncols, true).then_some(
        C { nthreads: 256, occupancy: 2, nbatch_fa: 64, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 64, nstages_target: 1, q_in_reg: true },
    )
}

/// Picks `ncols2` (how many GQA-sibling Q heads share one block's K/V load):
/// port of `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2`'s non-Volta
/// branches. `ncols2=1` is always valid for any `gqa_ratio` (it divides
/// every integer), so it is the fallback when `gqa_ratio` doesn't divide
/// evenly into a larger tile. The kernel no longer requires `seq_kv` padded
/// to a `FATTN_KQ_STRIDE` multiple to pick `ncols2 > 1` (the GQA-batched
/// kernels handle a partial final KV tile with their own bounds checks, see
/// `fattn_mma_f16.cuh`/`fattn_mma_bf16.cuh`'s `process_tile` dispatch).
fn mma_ncols2(gqa_ratio: usize, is_amd: bool) -> usize {
    if is_amd {
        // RDNA prefers an exact-divisor ncols2 to avoid wasted compute on
        // padding lanes; only falls through to the generic thresholds below
        // when gqa_ratio isn't a clean multiple of 2/4/8.
        if gqa_ratio.is_multiple_of(8) {
            return 8;
        }
        if gqa_ratio.is_multiple_of(4) {
            return 4;
        }
        if gqa_ratio.is_multiple_of(2) {
            return 2;
        }
    }
    if gqa_ratio > 4 {
        8
    } else if gqa_ratio > 2 {
        4
    } else if gqa_ratio > 1 {
        2
    } else {
        1
    }
}

/// Picks `ncols1` (query rows per tile) for the given `ncols2`: port of
/// `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1`. The `turing_mma_available`
/// branch (NVIDIA-only) always applies on CUDA since Crane's minimum target
/// is Ampere (cc 800 > Turing's 750); the `DKQ > 256` shortcut is AMD-only
/// and, for Crane's head dims, only ever applies at `head_dim=512`.
fn mma_ncols1(head_dim: usize, seq_q: usize, ncols2: usize, is_amd: bool) -> usize {
    if !is_amd && ncols2 <= 8 && seq_q <= 8 / ncols2 {
        return 8 / ncols2;
    }
    if ncols2 <= 16 && seq_q <= 16 / ncols2 {
        return 16 / ncols2;
    }
    if seq_q <= 32 / ncols2 || (is_amd && head_dim > 256) {
        return 32 / ncols2;
    }
    64 / ncols2
}

/// Selects `(ncols1, ncols2)` for the MMA kernel at this `(head_dim,
/// gqa_ratio, seq_q)`, or `None` if no instantiated kernel covers it.
/// Callers fall back to the tile kernel in that case. `seq_kv` plays no part
/// in the decision: the kernel handles a partial final KV tile with its own
/// bounds checks (see `fattn_mma_f16.cuh`'s `process_tile` dispatch), so
/// `ncols2 > 1` no longer requires `seq_kv` padded to a `FATTN_KQ_STRIDE`
/// multiple.
///
/// Two `None` cases, both confirmed directly against `fattn_mma_f16.cuh`'s
/// kernel body (the `NO_DEVICE_CODE; return;` guards right after the
/// architecture-availability checks, not just the instance list):
///
/// - **`is_amd` and `head_dim > 256`**: both `AMD_WMMA_AVAILABLE` (RDNA) and
///   `AMD_MFMA_AVAILABLE` (CDNA) branches reject `DKQ > 256` unconditionally.
///   `head_dim=512` (Gemma4) has no working MMA path on AMD at all,
///   regardless of `ncols1`/`ncols2`; a real gfx1201 hardware exception
///   confirmed this is a correctness requirement, not an optimization.
/// - **`is_amd` and `ncols2` resolving to `1`**: `AMD_WMMA_AVAILABLE`
///   additionally rejects `ncols2 == 1` specifically (RDNA always needs the
///   GQA-batching tile shape); `fattn_mma_hd512.cu` only instantiates
///   `ncols2` in `{2, 4, 8}` for this same reason on every backend. A caller
///   with `gqa_ratio=1` (plain MHA) resolves to `ncols2=1` (see
///   [`mma_ncols2`]) and must use the tile kernel on AMD. NVIDIA has no such
///   restriction (confirmed: the `AMD_WMMA_AVAILABLE` guard is absent on the
///   non-AMD path), so `ncols2=1` stays valid there.
pub(crate) fn select_mma_ncols(
    head_dim: usize,
    gqa_ratio: usize,
    seq_q: usize,
    is_amd: bool,
) -> Option<(usize, usize)> {
    if is_amd && head_dim > 256 {
        return None;
    }
    let ncols2 = mma_ncols2(gqa_ratio, is_amd);
    // AMD rejects ncols2==1 unconditionally (kernel-level, see this
    // function's doc comment). NVIDIA allows ncols2==1 in general, but
    // fattn_mma_hd512.cu only instantiates ncols2 in {2, 4, 8} for
    // head_dim=512 specifically (matching upstream's own instance list,
    // see fattn.rs's module doc), so that one combination has no kernel
    // regardless of architecture.
    if ncols2 == 1 && (is_amd || head_dim == 512) {
        return None;
    }
    let ncols1 = mma_ncols1(head_dim, seq_q, ncols2, is_amd);
    // AMD's WMMA kernel traps (NO_DEVICE_CODE) below this product; it
    // currently holds only as an emergent property of mma_ncols1's branch
    // ladder lacking the NVIDIA-only 8/ncols2 tier, so assert it explicitly
    // rather than let a future change to that ladder reintroduce the trap
    // silently.
    if is_amd {
        debug_assert!(
            ncols1 * ncols2 >= 16,
            "AMD WMMA requires ncols1*ncols2 >= 16, got {ncols1}*{ncols2}"
        );
    }
    Some((ncols1, ncols2))
}

/// bf16 counterpart of [`mma_ncols1`]: identical except the NVIDIA-only
/// `ncols2 <= 8 && seq_q <= 8 / ncols2` tier is skipped. `fattn_mma_bf16.cuh`
/// never instantiates `ncols1*ncols2 == 8` (no bf16 counterpart to the
/// narrow `cols_per_warp == 8` tile shape — see that file's module doc), so
/// this always falls through to a tier resolving to a product of at least
/// 16.
fn mma_ncols1_bf16(head_dim: usize, seq_q: usize, ncols2: usize, is_amd: bool) -> usize {
    if ncols2 <= 16 && seq_q <= 16 / ncols2 {
        return 16 / ncols2;
    }
    if seq_q <= 32 / ncols2 || (is_amd && head_dim > 256) {
        return 32 / ncols2;
    }
    64 / ncols2
}

/// bf16 counterpart of [`select_mma_ncols`]: same shape/GQA rejection rules
/// (see that function's doc comment), but calls [`mma_ncols1_bf16`] instead
/// of [`mma_ncols1`] so the result never resolves to `ncols1*ncols2 == 8`.
/// Reuses [`mma_config`]'s existing tuning tables unchanged: `fattn_mma_bf16.cuh`'s
/// config tables are copied from the same Ampere/RDNA values (see that
/// file's module doc), just without the `ncols == 8` entries this function
/// never queries.
pub(crate) fn select_mma_ncols_bf16(
    head_dim: usize,
    gqa_ratio: usize,
    seq_q: usize,
    is_amd: bool,
) -> Option<(usize, usize)> {
    if is_amd && head_dim > 256 {
        return None;
    }
    let ncols2 = mma_ncols2(gqa_ratio, is_amd);
    if ncols2 == 1 && (is_amd || head_dim == 512) {
        return None;
    }
    let ncols1 = mma_ncols1_bf16(head_dim, seq_q, ncols2, is_amd);
    debug_assert!(
        ncols1 * ncols2 >= 16,
        "fattn_mma_bf16 requires ncols1*ncols2 >= 16, got {ncols1}*{ncols2}"
    );
    Some((ncols1, ncols2))
}

/// Which kernel family [`select_mma_ncols`] would pick for, i.e. whether a
/// call with these shapes has an instantiated MMA kernel at all. `seq_kv`
/// is unused (see [`select_mma_ncols`]'s doc comment) but kept in the
/// signature to mirror the dispatch call this answers for.
#[must_use]
pub fn select_kernel(
    head_dim: usize,
    gqa_ratio: usize,
    seq_q: usize,
    _seq_kv: usize,
    is_amd: bool,
) -> FattnKernelType {
    if select_mma_ncols(head_dim, gqa_ratio, seq_q, is_amd).is_some() {
        FattnKernelType::MmaF16
    } else {
        FattnKernelType::Tile
    }
}

/// `crane_fattn_mma_f16_d{dkq}_v{dkq}_c{ncols1}_s{ncols2}`. See
/// `fattn_mma_f16.cuh`'s `DECL_FATTN_MMA_F16_CASE` macro.
pub(crate) fn mma_kernel_name(dkq: usize, ncols1: usize, ncols2: usize) -> String {
    format!("crane_fattn_mma_f16_d{dkq}_v{dkq}_c{ncols1}_s{ncols2}")
}

/// `crane_fattn_mma_bf16_d{dkq}_v{dkq}_c{ncols1}_s{ncols2}`. See
/// `fattn_mma_bf16.cuh`'s `DECL_FATTN_MMA_BF16_CASE` macro.
pub(crate) fn mma_bf16_kernel_name(dkq: usize, ncols1: usize, ncols2: usize) -> String {
    format!("crane_fattn_mma_bf16_d{dkq}_v{dkq}_c{ncols1}_s{ncols2}")
}

/// `crane_fattn_mma_f16_d{dkq}_v{dkq}_c{ncols1}_s{ncols2}_tail`: the
/// bounds-checked-tail counterpart of [`mma_kernel_name`], for launches
/// whose `seq_kv` isn't a multiple of the selected config's `nbatch_fa`. See
/// `fattn_mma_f16.cuh`'s `DECL_FATTN_MMA_F16_TAIL_CASE` macro.
pub(crate) fn mma_kernel_name_tail(dkq: usize, ncols1: usize, ncols2: usize) -> String {
    format!("crane_fattn_mma_f16_d{dkq}_v{dkq}_c{ncols1}_s{ncols2}_tail")
}

/// bf16 counterpart of [`mma_kernel_name_tail`]. See `fattn_mma_bf16.cuh`'s
/// `DECL_FATTN_MMA_BF16_TAIL_CASE` macro.
pub(crate) fn mma_bf16_kernel_name_tail(dkq: usize, ncols1: usize, ncols2: usize) -> String {
    format!("crane_fattn_mma_bf16_d{dkq}_v{dkq}_c{ncols1}_s{ncols2}_tail")
}

/// `crane_fattn_tile_f16_d{dkq}_v{dkq}_c16_s1`. The single `(ncols1,
/// ncols2) = (16, 1)` tier `fattn_tile_instances.cu` instantiates per
/// `head_dim` (see that file's module doc).
pub(crate) fn tile_kernel_name(dkq: usize) -> String {
    format!("crane_fattn_tile_f16_d{dkq}_v{dkq}_c16_s1")
}

/// Rust port of `crane_fattn_shim.cuh`'s `fastdiv`/`fastmodulo`'s host-side
/// producer (upstream's `init_fastdiv_values`, not vendored into the shim;
/// see that file's module doc). Packs `d` into the `<mp, L, d>` triple the
/// kernel's `uint3 ne01` parameter expects for Barrett-reduction division.
///
/// `d` must be nonzero and fit in `u32` (a tensor dimension; Crane never
/// approaches `u32::MAX` elements on any one axis).
pub(crate) fn init_fastdiv_values(d: u64) -> (u32, u32, u32) {
    debug_assert!(d != 0, "init_fastdiv_values: d must be nonzero");
    #[allow(clippy::cast_possible_truncation)]
    let d32 = d as u32;
    let mut l: u32 = 0;
    while l < 32 && (1u32 << l) < d32 {
        l += 1;
    }
    #[allow(clippy::cast_possible_truncation)]
    let mp = ((1u64 << 32) * ((1u64 << l) - u64::from(d32)) / u64::from(d32) + 1) as u32;
    (mp, l, d32)
}

/// Shared-memory size (bytes) the MMA kernel needs for this launch: port of
/// `ggml_cuda_flash_attn_ext_mma_f16_case`'s `nbytes_shared_total`
/// computation (`fattn.cu`, not vendored; see this module's doc comment).
/// `sizeof(half2) == 4`.
pub(crate) fn mma_shared_mem_bytes(
    dkq: usize,
    ncols1: usize,
    ncols2: usize,
    cfg: &FattnMmaConfig,
    is_amd: bool,
) -> usize {
    const HALF2_BYTES: usize = 4;
    let ncols = ncols1 * ncols2;
    let cols_per_warp = ncols.min(16);
    let nwarps = cfg.nthreads / 32;
    // cp.async pipelining is NVIDIA-Ampere+-only (never available under
    // HIP); only takes effect for ncols2 >= 2, matching the device-side
    // `ggml_cuda_fattn_mma_get_nstages`.
    let nstages = if !is_amd && ncols2 >= 2 {
        cfg.nstages_target
    } else {
        0
    };
    let stride_tile_k = tile_stride(cfg.nbatch_k2, is_amd);
    let stride_tile_v = tile_stride(cfg.nbatch_v2, is_amd);
    let nbytes_shared_kv_1stage = cfg.nbatch_fa * stride_tile_k.max(stride_tile_v) * HALF2_BYTES;
    let nbytes_shared_kv_2stage = cfg.nbatch_fa * (stride_tile_k + stride_tile_v) * HALF2_BYTES;
    let nbytes_shared_kv = if nstages <= 1 {
        nbytes_shared_kv_1stage
    } else {
        nbytes_shared_kv_2stage
    };
    let nbytes_shared_q = ncols * (dkq / 2 + 4) * HALF2_BYTES;
    let nbytes_shared_mask = ncols1 * (cfg.nbatch_fa / 2 + 4) * HALF2_BYTES;
    let nbytes_shared_combine = nwarps * cols_per_warp * (cfg.nbatch_combine + 4) * HALF2_BYTES;
    let rest = if cfg.q_in_reg {
        nbytes_shared_q.max(nbytes_shared_kv + nbytes_shared_mask)
    } else {
        nbytes_shared_q + nbytes_shared_kv + nbytes_shared_mask
    };
    nbytes_shared_combine.max(rest)
}

/// bf16 counterpart of [`mma_shared_mem_bytes`]: same formula, forwarding
/// `is_amd` unchanged. `fattn_mma_bf16.cuh` now shares `fattn_mma_f16.cuh`'s
/// shared-memory swizzle and multi-stage `cp.async` pipelining on NVIDIA
/// (see that file's module doc), so BF16's smem sizing must track the real
/// architecture the same way F16's does.
pub(crate) fn mma_shared_mem_bytes_bf16(
    dkq: usize,
    ncols1: usize,
    ncols2: usize,
    cfg: &FattnMmaConfig,
    is_amd: bool,
) -> usize {
    mma_shared_mem_bytes(dkq, ncols1, ncols2, cfg, is_amd)
}

/// `ggml_cuda_fattn_smem_swizzle::tile_stride(nbatch_2, cc)`'s host
/// equivalent: swizzle (and thus the unpadded stride) is only enabled on
/// NVIDIA Turing+ with a bank-aligned stride; always disabled on AMD. Since
/// Crane's CUDA target is Ampere+ (cc 800, already past Turing's 750
/// threshold), `is_amd = false` here always means "Turing+ available".
fn tile_stride(n: usize, is_amd: bool) -> usize {
    if !is_amd && n >= 32 && n.is_multiple_of(32) {
        n
    } else {
        n + 4
    }
}

/// Query-tile count along the grid's x axis: `ceil(seq_q / ncols1)`.
pub(crate) fn q_tiles(seq_q: usize, ncols1: usize) -> usize {
    seq_q.div_ceil(ncols1)
}

/// GQA-group tile count along the grid's z axis: `ceil(gqa_ratio / ncols2)`.
pub(crate) fn gqa_z_tiles(gqa_ratio: usize, ncols2: usize) -> usize {
    gqa_ratio.div_ceil(ncols2)
}

///// Stream-K ////////////////////////////////////////////////////////////

/// Grid size of a stream-K MMA launch splitting `ntiles` output tiles of
/// `iter_k` KV blocks each evenly across `max_blocks` resident blocks.
/// Rounds down to a multiple of `ntiles` when that costs at most 5% of the
/// blocks, otherwise uses `min(max_blocks, ntiles * iter_k)` as-is.
///
/// Port of llama.cpp's `launch_fattn` rule (`fattn-common.cuh`) for AMD at
/// `head_dim` 64, the only shape that still auto-splits (see
/// `LaunchGeometry::new`); every other shape uses one block per tile.
///
/// # Errors
///
/// Returns an error if `ntiles * iter_k` overflows `usize`.
pub(crate) fn stream_k_grid(ntiles: usize, iter_k: usize, max_blocks: usize) -> Result<usize> {
    let total = ntiles.checked_mul(iter_k).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "fattn_mma: ntiles ({ntiles}) * iter_k ({iter_k}) overflows usize"
        ))
    })?;
    let raw = max_blocks.min(total);
    let rounded = raw / ntiles * ntiles;
    let loss_percent = if rounded > 0 {
        100 * (raw - rounded) / raw
    } else {
        100
    };
    Ok(if loss_percent <= 5 { rounded } else { raw })
}

/// Whether a block boundary of a `grid`-block launch over `total` units
/// (`ntiles * iter_k`) falls inside a tile of `iter_k` units. Only then do
/// blocks share a tile and need the fixup pass. The boundaries are the
/// kernel's own, `b * total / grid`.
///
/// # Errors
///
/// Returns an error if `b * total` overflows `usize` for some block index
/// `b` in `1..grid`.
pub(crate) fn stream_k_needs_fixup(total: usize, grid: usize, iter_k: usize) -> Result<bool> {
    for b in 1..grid {
        let boundary = b.checked_mul(total).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "fattn_mma: b ({b}) * total ({total}) overflows usize"
            ))
        })?;
        if !(boundary / grid).is_multiple_of(iter_k) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// F32 elements of the fixup scratch buffer for `grid` blocks of `ncols`
/// columns at `head_dim`. See `kernels/cuda/fattn/fattn_fixup.cuh` for the
/// layout.
///
/// # Errors
///
/// Returns an error if the scratch length overflows `usize`.
pub(crate) fn fixup_scratch_len(grid: usize, ncols: usize, head_dim: usize) -> Result<usize> {
    2usize
        .checked_mul(grid)
        .and_then(|n| n.checked_mul(ncols))
        .and_then(|n| n.checked_mul(2 + head_dim))
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "fattn_mma: fixup scratch length overflows usize \
                 (grid={grid}, ncols={ncols}, head_dim={head_dim})"
            ))
        })
}

/// `crane_fattn_fixup_{f16,bf16}_d{dkq}_c{ncols1}_s{ncols2}`, the fixup entry
/// of the matching MMA kernel. See `fattn_fixup.cuh`'s
/// `DECL_CRANE_FATTN_FIXUP`.
pub(crate) fn fixup_kernel_name(dkq: usize, ncols1: usize, ncols2: usize, bf16: bool) -> String {
    let tag = if bf16 { "bf16" } else { "f16" };
    format!("crane_fattn_fixup_{tag}_d{dkq}_c{ncols1}_s{ncols2}")
}

/// Shared shape/stride validation for the fattn kernels: checks k/v
/// shapes match, `head_dim` is one of {64, 128, 256, 512}, GQA head-count
/// divisibility, and last-axis contiguity. `q_dtype` is Q's dtype, which
/// sets the element size its strides are checked against. Returns `(batch,
/// seq_q, num_heads_q, head_dim, seq_kv, num_heads_kv)`.
///
/// # Errors
///
/// Returns an error if any of the above checks fail.
pub(crate) fn validate_fattn_bshd(
    l_q: &Layout,
    l_k: &Layout,
    l_v: &Layout,
    q_dtype: DType,
) -> Result<(usize, usize, usize, usize, usize, usize)> {
    let (b, seq_q, h_q, d) = l_q.shape().dims4()?;
    let (b_kv, seq_kv, h_kv, d_kv) = l_k.shape().dims4()?;
    let v_dims = l_v.shape().dims4()?;
    if (b_kv, seq_kv, h_kv, d_kv) != v_dims {
        candle_core::bail!(
            "fattn: k {:?} and v {:?} shapes must match",
            (b_kv, seq_kv, h_kv, d_kv),
            v_dims
        );
    }
    if b_kv != b {
        candle_core::bail!("fattn: q batch {b} != k/v batch {b_kv}");
    }
    if d_kv != d {
        candle_core::bail!("fattn: q head_dim {d} != k/v head_dim {d_kv}");
    }
    if !matches!(d, 64 | 128 | 256 | 512) {
        candle_core::bail!("fattn: only head_dim in {{64,128,256,512}} is supported, got {d}");
    }
    if h_q == 0 || h_kv == 0 || h_q % h_kv != 0 {
        candle_core::bail!(
            "fattn: num_heads_q ({h_q}) must be a positive multiple of num_heads_kv ({h_kv})"
        );
    }
    if seq_q == 0 {
        candle_core::bail!("fattn: seq_q must be nonzero");
    }
    for (l, name) in [(l_q, "q"), (l_k, "k"), (l_v, "v")] {
        if l.stride()[3] != 1 {
            candle_core::bail!("fattn: {name} head_dim (last axis) must be contiguous");
        }
    }
    // Q's batch offset is 32-bit in the kernels, K/V's is 64-bit.
    check_kernel_layout(l_q, q_dtype.size_in_bytes(), true, "q")?;
    check_kernel_layout(l_k, KV_ELEM_BYTES, false, "k")?;
    check_kernel_layout(l_v, KV_ELEM_BYTES, false, "v")?;
    Ok((b, seq_q, h_q, d, seq_kv, h_kv))
}

/// Element size of K/V, which every fattn entry takes in F16 or BF16.
pub(crate) const KV_ELEM_BYTES: usize = 2;

/// Byte alignment the kernels' vector loads need for every row and head
/// base address (llama.cpp's `nb[i] % 16` check in `fattn.cu`).
const KERNEL_ALIGN_BYTES: usize = 16;

/// Checks that the fattn kernels can address a `[B, S, H, D]` operand in any
/// layout, not only a contiguous one. The kernels form head offsets (and Q's
/// batch offset) as 32-bit `nb * index` products, and load rows with 16-byte
/// vector reads. Axes of size 1 are skipped, since their stride is never
/// multiplied by a nonzero index.
///
/// # Arguments
///
/// * `layout` - The operand's layout. It must be 4D, with a contiguous last
///   axis (both checked by [`validate_fattn_bshd`]).
/// * `elem_bytes` - Size of one element in bytes.
/// * `batch_in_i32` - Whether the batch offset is also 32-bit (Q's `nb03`).
/// * `what` - Operand name for the error message.
///
/// # Errors
///
/// Returns an unsupported-config error if the start offset or a stride is
/// not a multiple of [`KERNEL_ALIGN_BYTES`], or if the largest 32-bit byte
/// offset exceeds `i32::MAX`. The caller then copies the operand or falls
/// back to SDPA.
pub(crate) fn check_kernel_layout(
    layout: &Layout,
    elem_bytes: usize,
    batch_in_i32: bool,
    what: &str,
) -> Result<()> {
    let dims = layout.dims();
    let stride = layout.stride();
    let aligned = |n: usize| {
        n.checked_mul(elem_bytes)
            .is_some_and(|bytes| bytes % KERNEL_ALIGN_BYTES == 0)
    };
    let misaligned = !aligned(layout.start_offset())
        || (0..3).any(|axis| dims[axis] > 1 && !aligned(stride[axis]));
    if misaligned {
        return Err(unsupported_config(format!(
            "fattn: {what} start offset or strides {stride:?} not {KERNEL_ALIGN_BYTES}-byte aligned"
        )));
    }
    // `None` once any product or sum overflows usize.
    let mut extent = dims[3].checked_mul(elem_bytes);
    for axis in usize::from(!batch_in_i32)..3 {
        if dims[axis] > 1 {
            extent = extent.and_then(|sum| {
                (dims[axis] - 1)
                    .checked_mul(stride[axis])?
                    .checked_mul(elem_bytes)?
                    .checked_add(sum)
            });
        }
    }
    if extent.is_none_or(|bytes| i32::try_from(bytes).is_err()) {
        return Err(unsupported_config(format!(
            "fattn: {what} spans past the kernel's 32-bit offsets (strides {stride:?})"
        )));
    }
    Ok(())
}

/// CUDA/HIP cap the grid's y and z dimensions at 65535. The tile kernel puts
/// `parallel_blocks` (always `1`) on y and `gqa_z_tiles *
/// num_heads_kv * batch` on z; only z can realistically reach this limit.
pub(crate) const MAX_GRID_YZ_DIM: usize = 65535;

/// CUDA/HIP's much larger x-dimension limit (`2^31 - 1`). The MMA kernel
/// puts its entire `(query tile, GQA group, KV head, batch)` work count on
/// x (see `fattn_mma`'s `LaunchGeometry::new` doc comment for why), so it is
/// checked against this limit instead of [`MAX_GRID_YZ_DIM`].
pub(crate) const MAX_GRID_X_DIM: usize = 2_147_483_647;

fn to_i32(n: usize, what: &str) -> Result<i32> {
    i32::try_from(n)
        .map_err(|_| candle_core::Error::Msg(format!("fattn: {what} ({n}) exceeds i32::MAX")))
}

/// Whether this build targets AMD/`ROCm` rather than NVIDIA CUDA, mirroring
/// the same `cuda`-takes-priority-over-`rocm` feature precedence
/// [`super::fattn_mma::fattn_mma_prefill`]/
/// [`super::fattn_tile::fattn_tile_prefill`] use internally. Needed by
/// callers that must know `is_amd` themselves to call
/// [`select_mma_ncols`]/[`mma_config`] ahead of actually dispatching (e.g.
/// to size a `KV_max` tensor before the kernel launch picks its own
/// `ncols1`).
pub(crate) fn is_amd_backend() -> bool {
    cfg!(all(feature = "rocm", not(feature = "cuda")))
}

/// Builds a `[1, 1, seq_q, seq_kv]` `F16` additive mask where position
/// `(i, j)` is `0.0` when `j <= i + kv_offset + window_right` and, if
/// `window_left` is `Some`, also `j >= i + kv_offset - window_left`.
/// Otherwise `f32::NEG_INFINITY`. `window_left = None` means no left bound
/// (plain causal masking). Built on the CPU then uploaded via
/// `Tensor::from_vec`, so every call allocates and uploads. A caller that
/// applies one mask to many layers should build it once and pass it to the
/// `*_with_mask` entry points instead.
pub(crate) fn build_additive_mask_f16(
    seq_q: usize,
    seq_kv: usize,
    kv_offset: usize,
    window_left: Option<usize>,
    window_right: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut data = vec![0f32; seq_q * seq_kv];
    for i in 0..seq_q {
        for j in 0..seq_kv {
            // seq_q/seq_kv/kv_offset/window bounds are sequence lengths, far
            // below i64::MAX, so these never wrap; i64 is needed since `rel`
            // can be negative.
            #[allow(clippy::cast_possible_wrap)]
            let (center, j_i64, window_right_i64, window_left_i64) = (
                (i + kv_offset) as i64,
                j as i64,
                window_right as i64,
                window_left.map(|w| w as i64),
            );
            let rel = j_i64 - center;
            let valid = rel <= window_right_i64
                && match window_left_i64 {
                    Some(w) => rel >= -w,
                    None => true,
                };
            if !valid {
                data[i * seq_kv + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, seq_q, seq_kv), device)?.to_dtype(DType::F16)
}

/// Builds the analytic `KV_max` tensor for causal/windowed masking: one
/// `i32` entry per `(batch, query tile)`, where query tile `t` covers rows
/// `t*ncols1 .. (t+1)*ncols1 - 1`. Matches llama.cpp's own
/// `flash_attn_mask_to_KV_max` host-side semantics (not vendored; see
/// `fattn_mma_f16.cuh`'s `kb0_stop` and `fattn_tile.cuh`'s `k_VKQ_max`): the
/// raw KV-row bound for that tile, rounded up to the next
/// [`FATTN_KQ_STRIDE`] multiple and capped at `kv_max_cap`. There is no
/// `window_left` parameter because a left-side window only narrows which
/// *early* keys are valid, handled by the mask itself, never the upper
/// bound `KV_max` truncates.
///
/// `kv_max_cap` is **not** always `seq_kv` — the two kernel families consume
/// this value differently and need different caps:
/// - The tile kernel (`fattn_tile.cuh`) uses the raw value directly as a loop
///   bound (`k_VKQ_0 &lt; k_VKQ_max`) with no independent clamp against the
///   real K/V buffer length, so an over-wide cap is a real out-of-bounds
///   read. Tile callers must pass the true `seq_kv`.
/// - The MMA kernel (`fattn_mma_f16.cuh`) only ever narrows its own
///   independently-computed `kb0_stop` (`min(kb0_stop, KV_max/nbatch_fa)`,
///   itself already bounded by `iter_k = ceil(real_seq_kv/nbatch_fa)`), so a
///   looser cap can never cause an out-of-bounds read there — but capping at
///   a bare `seq_kv` that isn't a multiple of `nbatch_fa` (32/64/128) makes
///   that integer division truncate `kb0_stop`, silently dropping trailing
///   valid KV rows from the softmax. MMA callers should pass
///   `seq_kv.next_multiple_of(FATTN_KQ_STRIDE)` (safe: `FATTN_KQ_STRIDE` is a
///   multiple of every tabled `nbatch_fa`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_analytic_kv_max(
    batch: usize,
    seq_q: usize,
    ncols1: usize,
    kv_offset: usize,
    seq_kv: usize,
    window_right: usize,
    kv_max_cap: usize,
    device: &Device,
) -> Result<Tensor> {
    let n_tiles = q_tiles(seq_q, ncols1);
    let mut data = vec![0i32; batch * n_tiles];
    for t in 0..n_tiles {
        let last_row = ((t + 1) * ncols1)
            .saturating_sub(1)
            .min(seq_q.saturating_sub(1));
        let last_valid_col = (last_row + kv_offset + window_right).min(seq_kv.saturating_sub(1));
        let bound = (last_valid_col + 1).div_ceil(FATTN_KQ_STRIDE) * FATTN_KQ_STRIDE;
        let bound = to_i32(bound.min(kv_max_cap), "kv_max entry")?;
        for b in 0..batch {
            data[b * n_tiles + t] = bound;
        }
    }
    Tensor::from_vec(data, (batch, n_tiles), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    // init_fastdiv_values must round-trip: n / fastdiv_values and n %
    // fastdiv_values (computed the same way the device-side fastdiv/fastmodulo
    // do) must match plain integer division/modulo for every n in range.
    fn check_fastdiv(d: u64) {
        let (mp, l, dv) = init_fastdiv_values(d);
        assert_eq!(u64::from(dv), d);
        for n in 0..(d as u32 * 3 + 5) {
            let hi = (u64::from(n) * u64::from(mp)) >> 32;
            #[allow(clippy::cast_possible_truncation)]
            let div = ((hi + u64::from(n)) >> l) as u32;
            let modulo = n - div * dv;
            assert_eq!(div, n / d as u32, "fastdiv mismatch for d={d} n={n}");
            assert_eq!(modulo, n % d as u32, "fastmodulo mismatch for d={d} n={n}");
        }
    }

    #[test]
    fn fastdiv_matches_integer_division() {
        for d in [
            1u64, 2, 3, 4, 7, 8, 16, 17, 31, 32, 63, 64, 100, 4096, 19001,
        ] {
            check_fastdiv(d);
        }
    }

    #[test]
    fn select_kernel_falls_back_to_tile_at_hd512_mha() {
        // head_dim=512, gqa_ratio=1 (plain MHA): no ncols2=1 kernel exists.
        assert_eq!(select_kernel(512, 1, 16, 256, false), FattnKernelType::Tile);
        assert_eq!(select_kernel(512, 1, 16, 256, true), FattnKernelType::Tile);
    }

    #[test]
    fn select_kernel_uses_mma_at_hd512_with_gqa_on_nvidia_only() {
        // head_dim=512 with a real GQA ratio and padded seq_kv: ncols2 > 1
        // on NVIDIA, where the MMA kernel supports head_dim=512 at all.
        assert_eq!(
            select_kernel(512, 4, 16, 256, false),
            FattnKernelType::MmaF16
        );
        // AMD's MMA kernel rejects DKQ > 256 unconditionally (both
        // AMD_WMMA_AVAILABLE and AMD_MFMA_AVAILABLE in fattn_mma_f16.cuh),
        // regardless of GQA ratio or padding. Always falls back to tile.
        assert_eq!(select_kernel(512, 4, 16, 256, true), FattnKernelType::Tile);
    }

    #[test]
    fn select_kernel_uses_mma_for_small_head_dims_regardless_of_gqa() {
        // head_dim in {64,128,256} always has a ncols2=1 kernel, so MHA
        // (gqa_ratio=1) or an unpadded seq_kv still resolves to MMA.
        for head_dim in [64, 128, 256] {
            assert_eq!(
                select_kernel(head_dim, 1, 16, 100, false),
                FattnKernelType::MmaF16
            );
        }
    }

    // select_mma_ncols must only ever return (ncols1, ncols2) pairs that
    // fattn_mma_hd{64,128,256,512}.cu actually instantiate.
    fn instantiated_pairs(head_dim: usize) -> &'static [(usize, usize)] {
        const SWEEP: &[(usize, usize)] = &[
            (8, 1),
            (4, 2),
            (2, 4),
            (1, 8),
            (16, 1),
            (8, 2),
            (4, 4),
            (2, 8),
            (1, 16),
            (32, 1),
            (16, 2),
            (8, 4),
            (4, 8),
            (2, 16),
            (64, 1),
            (32, 2),
            (16, 4),
            (8, 8),
            (4, 16),
        ];
        const HD512: &[(usize, usize)] = &[
            (4, 2),
            (8, 2),
            (16, 2),
            (32, 2),
            (2, 4),
            (4, 4),
            (8, 4),
            (16, 4),
            (1, 8),
            (2, 8),
            (4, 8),
            (8, 8),
        ];
        if head_dim == 512 { HD512 } else { SWEEP }
    }

    #[test]
    fn select_mma_ncols_only_picks_instantiated_pairs() {
        for head_dim in [64usize, 128, 256, 512] {
            for is_amd in [false, true] {
                for gqa_ratio in [1usize, 2, 3, 4, 5, 6, 7, 8, 16, 32] {
                    for seq_q in [1usize, 2, 4, 8, 15, 16, 17, 31, 32, 63, 64, 65, 1024] {
                        let Some(pair) = select_mma_ncols(head_dim, gqa_ratio, seq_q, is_amd)
                        else {
                            continue;
                        };
                        assert!(
                            instantiated_pairs(head_dim).contains(&pair),
                            "head_dim={head_dim} is_amd={is_amd} gqa_ratio={gqa_ratio} \
                             seq_q={seq_q} picked uninstantiated pair {pair:?}"
                        );
                        // The chosen config must also actually exist.
                        mma_config(head_dim, pair.0 * pair.1, is_amd).unwrap_or_else(|e| {
                            panic!("head_dim={head_dim} is_amd={is_amd} pair={pair:?}: {e}")
                        });
                    }
                }
            }
        }
    }

    #[test]
    fn mma_kernel_name_matches_decl_macro_naming() {
        assert_eq!(
            mma_kernel_name(128, 8, 1),
            "crane_fattn_mma_f16_d128_v128_c8_s1"
        );
        assert_eq!(
            tile_kernel_name(256),
            "crane_fattn_tile_f16_d256_v256_c16_s1"
        );
    }

    // select_mma_ncols_bf16 must only ever return (ncols1, ncols2) pairs that
    // fattn_mma_hd{64,128,256,512}_bf16.cu actually instantiate: the same
    // sets as instantiated_pairs minus every pair whose product is 8 (no
    // bf16 kernel instantiates that tile shape — see mma_ncols1_bf16's doc
    // comment).
    fn instantiated_pairs_bf16(head_dim: usize) -> &'static [(usize, usize)] {
        const SWEEP_BF16: &[(usize, usize)] = &[
            (16, 1),
            (8, 2),
            (4, 4),
            (2, 8),
            (1, 16),
            (32, 1),
            (16, 2),
            (8, 4),
            (4, 8),
            (2, 16),
            (64, 1),
            (32, 2),
            (16, 4),
            (8, 8),
            (4, 16),
        ];
        const HD512_BF16: &[(usize, usize)] = &[
            (8, 2),
            (16, 2),
            (32, 2),
            (4, 4),
            (8, 4),
            (16, 4),
            (2, 8),
            (4, 8),
            (8, 8),
        ];
        if head_dim == 512 {
            HD512_BF16
        } else {
            SWEEP_BF16
        }
    }

    #[test]
    fn select_mma_ncols_bf16_only_picks_instantiated_pairs() {
        for head_dim in [64usize, 128, 256, 512] {
            for is_amd in [false, true] {
                for gqa_ratio in [1usize, 2, 3, 4, 5, 6, 7, 8, 16, 32] {
                    for seq_q in [1usize, 2, 4, 8, 15, 16, 17, 31, 32, 63, 64, 65, 1024] {
                        let Some(pair) = select_mma_ncols_bf16(head_dim, gqa_ratio, seq_q, is_amd)
                        else {
                            continue;
                        };
                        assert!(
                            instantiated_pairs_bf16(head_dim).contains(&pair),
                            "head_dim={head_dim} is_amd={is_amd} gqa_ratio={gqa_ratio} \
                             seq_q={seq_q} picked uninstantiated bf16 pair {pair:?}"
                        );
                        // The chosen config must also actually exist (bf16
                        // reuses the same tuning tables as f16).
                        mma_config(head_dim, pair.0 * pair.1, is_amd).unwrap_or_else(|e| {
                            panic!("head_dim={head_dim} is_amd={is_amd} pair={pair:?}: {e}")
                        });
                    }
                }
            }
        }
    }

    // Every RDNA config AMD can launch, long-KV set included, needs at most
    // 64 KiB of LDS, the per-workgroup limit on RDNA. AMD rejects ncols2 == 1
    // and head_dim 512, so those pairs are skipped.
    #[test]
    fn rdna_mma_configs_fit_in_amd_lds() {
        const AMD_LDS_PER_WORKGROUP: usize = 64 * 1024;
        for head_dim in [64usize, 128, 256] {
            for &(ncols1, ncols2) in instantiated_pairs(head_dim) {
                if ncols2 == 1 {
                    continue;
                }
                let cfg = mma_config(head_dim, ncols1 * ncols2, true).unwrap();
                let bytes = mma_shared_mem_bytes(head_dim, ncols1, ncols2, &cfg, true);
                assert!(
                    bytes <= AMD_LDS_PER_WORKGROUP,
                    "f16 head_dim={head_dim} c{ncols1}_s{ncols2}: {bytes} B of LDS"
                );
            }
            for &(ncols1, ncols2) in instantiated_pairs_bf16(head_dim) {
                if ncols2 == 1 {
                    continue;
                }
                let cfg = mma_config_bf16(head_dim, ncols1 * ncols2, true).unwrap();
                let bytes = mma_shared_mem_bytes_bf16(head_dim, ncols1, ncols2, &cfg, true);
                assert!(
                    bytes <= AMD_LDS_PER_WORKGROUP,
                    "bf16 head_dim={head_dim} c{ncols1}_s{ncols2}: {bytes} B of LDS"
                );
            }
        }
        // The long-KV config, for every 64-column pair AMD can pick.
        let long = mma_config_rdna_long_kv(256, 64).unwrap();
        for (ncols1, ncols2) in [(32, 2), (16, 4), (8, 8)] {
            let bytes = mma_shared_mem_bytes(256, ncols1, ncols2, &long, true);
            assert!(
                bytes <= AMD_LDS_PER_WORKGROUP,
                "long-KV c{ncols1}_s{ncols2}: {bytes} B of LDS"
            );
        }
    }

    // Only the RDNA hd256 64-column tile has a long-KV set, and the switch
    // happens exactly at the `seq_kv` threshold.
    #[test]
    fn mma_config_set_switches_at_the_threshold() {
        let min = RDNA_HD256_LONG_KV_MIN_SEQ_KV;
        let set = |ncols, seq_kv, dkq, is_amd| mma_config_set(dkq, ncols, (seq_kv, 0), 2, is_amd);
        assert_eq!(set(64, min - 1, 256, true), MmaConfigSet::Default);
        assert_eq!(set(64, min, 256, true), MmaConfigSet::LongKv);
        assert_eq!(set(64, 1 << 20, 256, true), MmaConfigSet::LongKv);
        // Neither another column count, another head_dim, nor NVIDIA.
        assert_eq!(set(32, min, 256, true), MmaConfigSet::Default);
        assert_eq!(set(64, min, 128, true), MmaConfigSet::Default);
        assert_eq!(set(64, min, 256, false), MmaConfigSet::Default);
    }

    // A long KV that every query tile reads in full goes back to the default
    // set, at exactly the byte limit.
    #[test]
    fn mma_config_set_returns_to_default_past_the_cache_floor() {
        // 2 KV heads, head_dim 256: 2 KiB of K and V per row and head, so
        // 4 KiB per row.
        let rows = RDNA_HD256_LONG_KV_MAX_FLOOR_BYTES / (2 * 256 * 2 * 2);
        let seq_kv = 4 * rows;
        let set = |min_kv| mma_config_set(256, 64, (seq_kv, min_kv), 2, true);
        assert_eq!(set(0), MmaConfigSet::LongKv);
        assert_eq!(set(rows - 1), MmaConfigSet::LongKv);
        assert_eq!(set(rows), MmaConfigSet::Default);
        assert_eq!(set(seq_kv), MmaConfigSet::Default);
        // More KV heads reach the limit with fewer rows.
        assert_eq!(
            mma_config_set(256, 64, (seq_kv, rows / 2), 4, true),
            MmaConfigSet::Default
        );
        // Huge inputs saturate instead of wrapping.
        assert_eq!(
            mma_config_set(256, 64, (usize::MAX, usize::MAX), usize::MAX, true),
            MmaConfigSet::Default
        );
    }

    // The long-KV config differs from the default one only in the tile
    // shape, and its kernel set exists exactly where `has_long_kv_set` says.
    #[test]
    fn long_kv_config_matches_has_long_kv_set() {
        for dkq in [64usize, 128, 256, 512] {
            for ncols in [8usize, 16, 32, 64] {
                assert_eq!(
                    mma_config_rdna_long_kv(dkq, ncols).is_some(),
                    has_long_kv_set(dkq, ncols, true),
                    "dkq={dkq} ncols={ncols}"
                );
            }
        }
        let long = mma_config_rdna_long_kv(256, 64).unwrap();
        let default = mma_config(256, 64, true).unwrap();
        assert_eq!((long.nbatch_fa, long.q_in_reg), (64, true));
        assert_eq!((default.nbatch_fa, default.q_in_reg), (32, false));
        assert_eq!(long.nthreads, default.nthreads);
        assert_eq!(long.nbatch_k2, default.nbatch_k2);
    }

    #[test]
    fn mma_bf16_kernel_name_matches_decl_macro_naming() {
        assert_eq!(
            mma_bf16_kernel_name(128, 8, 1),
            "crane_fattn_mma_bf16_d128_v128_c8_s1"
        );
    }

    #[test]
    fn validate_fattn_bshd_rejects_unsupported_head_dim() {
        use candle_core::{DType, Device, Tensor};
        let dev = Device::Cpu;
        let q = Tensor::zeros((1, 4, 2, 100), DType::F16, &dev).unwrap();
        let k = Tensor::zeros((1, 4, 2, 100), DType::F16, &dev).unwrap();
        let v = Tensor::zeros((1, 4, 2, 100), DType::F16, &dev).unwrap();
        let (q_s, q_l) = q.storage_and_layout();
        let (k_s, k_l) = k.storage_and_layout();
        let (v_s, v_l) = v.storage_and_layout();
        drop((q_s, k_s, v_s));
        let err = validate_fattn_bshd(q_l, k_l, v_l, DType::F16)
            .expect_err("head_dim=100 must be rejected");
        assert!(err.to_string().contains("head_dim in"));
    }

    fn assert_unsupported(result: Result<()>) {
        let err = result.expect_err("layout must be rejected");
        assert!(
            is_unsupported_config(&err),
            "not an unsupported-config error: {err}"
        );
    }

    // A misaligned K is rejected as unsupported-config by the shared
    // validator, before any kernel-specific launch setup.
    #[test]
    fn validate_fattn_bshd_rejects_misaligned_k() {
        let q = Layout::contiguous((1, 64, 2, 128));
        let k = Layout::contiguous_with_offset((1, 64, 2, 128), 1);
        assert_unsupported(validate_fattn_bshd(&q, &k, &q, DType::F16).map(|_| ()));
    }

    // Q's strides are checked in Q's own element size. A 132-element head
    // stride is 264 bytes in F16, off the 16-byte grid the kernels' vector
    // loads need, but 528 bytes in F32, which is on it.
    #[test]
    fn validate_fattn_bshd_checks_q_in_its_dtype() {
        let kv = Layout::contiguous((1, 64, 2, 128));
        let q = Layout::new(
            (1, 64, 2, 128).into(),
            vec![64 * 2 * 132, 2 * 132, 132, 1],
            0,
        );
        assert_unsupported(validate_fattn_bshd(&q, &kv, &kv, DType::F16).map(|_| ()));
        validate_fattn_bshd(&q, &kv, &kv, DType::F32).unwrap();
    }

    // A BSHD view of a BHSD tensor, and one of a narrowed KV-cache buffer
    // with a nonzero start offset, are addressable without a copy.
    #[test]
    fn check_kernel_layout_accepts_strided_views() {
        let bhsd = Layout::contiguous((2, 8, 1024, 128));
        check_kernel_layout(&bhsd.transpose(1, 2).unwrap(), 4, true, "q").unwrap();
        let cache = Layout::contiguous((2, 2, 4096, 256))
            .narrow(2, 64, 1024)
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        check_kernel_layout(&cache, 2, false, "k").unwrap();
    }

    // A start offset or a stride off the 16-byte grid is rejected as an
    // unsupported config, not launched.
    #[test]
    fn check_kernel_layout_rejects_misaligned_layouts() {
        let odd_offset = Layout::contiguous_with_offset((1, 64, 2, 128), 1);
        assert_unsupported(check_kernel_layout(&odd_offset, 2, false, "k"));
        // 68 F16 elements per row is 136 bytes, not a multiple of 16.
        let odd_stride = Layout::new((1, 64, 1, 64).into(), vec![64 * 68, 68, 64, 1], 0);
        assert_unsupported(check_kernel_layout(&odd_stride, 2, false, "k"));
    }

    // The stride of a size-1 axis is never used, so a misaligned one (as
    // a single decode step can leave on the seq axis) is accepted.
    #[test]
    fn check_kernel_layout_ignores_size_one_axes() {
        let decode = Layout::new((1, 1, 2, 64).into(), vec![7, 3, 64, 1], 0);
        check_kernel_layout(&decode, 2, false, "k").unwrap();
    }

    // The 32-bit head offset bounds K, and Q's batch offset bounds Q only.
    #[test]
    fn check_kernel_layout_rejects_offsets_past_i32() {
        // Two heads 2^30 F16 elements apart put the second head at 2^31 bytes.
        let far_heads = Layout::new((1, 4, 2, 64).into(), vec![1 << 31, 64, 1 << 30, 1], 0);
        assert_unsupported(check_kernel_layout(&far_heads, 2, false, "k"));
        // Batch 2 of 2^29 F32 elements per item starts at 2^31 bytes.
        let far_batch = Layout::new((2, 4, 2, 64).into(), vec![1 << 29, 128, 64, 1], 0);
        assert_unsupported(check_kernel_layout(&far_batch, 4, true, "q"));
        check_kernel_layout(&far_batch, 4, false, "k").unwrap();
    }

    // Strides whose byte products overflow usize are rejected, not wrapped
    // into a small value that passes the i32 check.
    #[test]
    fn check_kernel_layout_rejects_overflowing_strides() {
        // stride * 2 itself overflows.
        let wrap = Layout::new((1, 4, 2, 64).into(), vec![0, 64, 1 << 63, 1], 0);
        assert_unsupported(check_kernel_layout(&wrap, 2, false, "k"));
        // stride * 2 fits and is aligned, but (heads - 1) * stride * 2 wraps
        // to exactly 0.
        let span = Layout::new((1, 4, 3, 64).into(), vec![0, 64, 1 << 62, 1], 0);
        assert_unsupported(check_kernel_layout(&span, 2, false, "k"));
    }

    // A fully-visible (no kv_offset/window) causal mask: entry (i, j) is 0
    // for j <= i, -inf for j > i.
    #[test]
    fn build_additive_mask_f16_plain_causal() {
        let dev = Device::Cpu;
        let mask = build_additive_mask_f16(4, 4, 0, None, 0, &dev).unwrap();
        assert_eq!(mask.dims(), &[1, 1, 4, 4]);
        assert_eq!(mask.dtype(), DType::F16);
        let vals = mask.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let v = vals[i * 4 + j];
                if j <= i {
                    assert_eq!(v, half::f16::from_f32(0.0), "[{i},{j}] should be visible");
                } else {
                    assert!(
                        v.is_infinite() && v.is_sign_negative(),
                        "[{i},{j}] should be masked"
                    );
                }
            }
        }
    }

    // A sliding window: entry (i, j) is visible only for
    // i - window_left <= j <= i + window_right.
    #[test]
    fn build_additive_mask_f16_windowed() {
        let dev = Device::Cpu;
        let mask = build_additive_mask_f16(1, 10, 5, Some(2), 1, &dev).unwrap();
        let vals = mask.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
        // Row 0, kv_offset=5: center at j=5, visible j in [3, 6].
        for (j, v) in vals.iter().enumerate() {
            let visible = (3..=6).contains(&j);
            if visible {
                assert_eq!(*v, half::f16::from_f32(0.0), "j={j} should be visible");
            } else {
                assert!(
                    v.is_infinite() && v.is_sign_negative(),
                    "j={j} should be masked"
                );
            }
        }
    }

    // Every tile's bound must be a multiple of FATTN_KQ_STRIDE (or capped at
    // kv_max_cap), cover every row the tile is responsible for, and never
    // exceed kv_max_cap. Uses kv_max_cap = seq_kv, the tight cap tile-kernel
    // callers must pass (see build_analytic_kv_max's doc comment).
    #[test]
    fn build_analytic_kv_max_covers_tile_rows_exactly() {
        let dev = Device::Cpu;
        let (batch, seq_q, ncols1, kv_offset, seq_kv) = (2, 40, 16, 100, 300);
        let kv_max =
            build_analytic_kv_max(batch, seq_q, ncols1, kv_offset, seq_kv, 0, seq_kv, &dev)
                .unwrap();
        assert_eq!(kv_max.dims(), &[batch, q_tiles(seq_q, ncols1)]);
        let vals = kv_max.flatten_all().unwrap().to_vec1::<i32>().unwrap();
        let n_tiles = q_tiles(seq_q, ncols1);
        for t in 0..n_tiles {
            let last_row = ((t + 1) * ncols1 - 1).min(seq_q - 1);
            let last_valid_col = (last_row + kv_offset).min(seq_kv - 1);
            for b in 0..batch {
                let bound = vals[b * n_tiles + t];
                assert!(
                    bound as usize > last_valid_col,
                    "tile {t} bound {bound} must cover last valid col {last_valid_col}"
                );
                assert!(
                    bound as usize <= seq_kv,
                    "tile {t} bound {bound} must not exceed kv_max_cap"
                );
                assert!(
                    bound as usize == seq_kv || bound as usize % FATTN_KQ_STRIDE == 0,
                    "tile {t} bound {bound} must be a FATTN_KQ_STRIDE multiple unless capped at kv_max_cap"
                );
            }
        }
    }

    // window_right must extend the bound past kv_offset, same rounding rule
    // as the no-window case.
    #[test]
    fn build_analytic_kv_max_respects_window_right() {
        let dev = Device::Cpu;
        let narrow = build_analytic_kv_max(1, 16, 16, 0, 1000, 0, 1000, &dev).unwrap();
        let wide = build_analytic_kv_max(1, 16, 16, 0, 1000, 500, 1000, &dev).unwrap();
        let narrow_v = narrow.flatten_all().unwrap().to_vec1::<i32>().unwrap()[0];
        let wide_v = wide.flatten_all().unwrap().to_vec1::<i32>().unwrap()[0];
        assert!(
            wide_v > narrow_v,
            "window_right must raise the bound: {narrow_v} vs {wide_v}"
        );
    }

    // R1 regression: an MMA caller passing the loose
    // `seq_kv.next_multiple_of(FATTN_KQ_STRIDE)` cap must always get back a
    // FATTN_KQ_STRIDE multiple, even when the real seq_kv isn't one — this is
    // what keeps `KV_max / nbatch_fa` (fattn_mma_f16.cuh) from truncating and
    // silently dropping trailing valid KV rows. A tile-style tight cap
    // (kv_max_cap = seq_kv) would fail this for a non-aligned seq_kv, which
    // is exactly why MMA/tile callers must pass different caps.
    #[test]
    fn build_analytic_kv_max_mma_cap_stays_stride_aligned_for_unaligned_seq_kv() {
        let dev = Device::Cpu;
        let (batch, seq_q, ncols1, kv_offset, seq_kv): (usize, usize, usize, usize, usize) =
            (1, 20, 16, 0, 150);
        let kv_max_cap = seq_kv.next_multiple_of(FATTN_KQ_STRIDE);
        let kv_max =
            build_analytic_kv_max(batch, seq_q, ncols1, kv_offset, seq_kv, 0, kv_max_cap, &dev)
                .unwrap();
        let vals = kv_max.flatten_all().unwrap().to_vec1::<i32>().unwrap();
        for (t, &bound) in vals.iter().enumerate() {
            assert!(
                bound as usize % FATTN_KQ_STRIDE == 0,
                "tile {t} bound {bound} must be a FATTN_KQ_STRIDE multiple, not truncated to the unaligned seq_kv={seq_kv}"
            );
        }
    }

    // With more tiles than resident blocks the grid is the resident block
    // count.
    #[test]
    fn stream_k_grid_uses_the_resident_block_count() {
        assert_eq!(stream_k_grid(512, 4, 140).unwrap(), 140);
        assert_eq!(stream_k_grid(128, 4, 128).unwrap(), 128);
    }

    // The grid rounds down to a multiple of `ntiles` only when that costs at
    // most 5% of the blocks.
    #[test]
    fn stream_k_rounds_to_whole_tiles_within_five_percent() {
        assert_eq!(stream_k_grid(30, 8, 124).unwrap(), 120);
        assert_eq!(stream_k_grid(30, 8, 128).unwrap(), 128);
    }

    // The grid never exceeds the number of units to hand out.
    #[test]
    fn stream_k_grid_is_capped_by_the_work() {
        assert_eq!(stream_k_grid(4, 2, 128).unwrap(), 8);
    }

    // `ntiles * iter_k` overflowing `usize` is reported, not wrapped.
    #[test]
    fn stream_k_grid_reports_overflow() {
        assert!(stream_k_grid(usize::MAX, 2, 128).is_err());
    }

    // A seam inside a tile exists exactly when some block boundary is not a
    // multiple of `iter_k`.
    #[test]
    fn needs_fixup_only_for_seams_inside_tiles() {
        // 30 tiles of 4 blocks.
        assert!(stream_k_needs_fixup(120, 120, 4).unwrap());
        assert!(stream_k_needs_fixup(120, 60, 4).unwrap());
        assert!(stream_k_needs_fixup(120, 31, 4).unwrap());
        assert!(stream_k_needs_fixup(120, 29, 4).unwrap());
        assert!(!stream_k_needs_fixup(120, 30, 4).unwrap());
        assert!(!stream_k_needs_fixup(120, 1, 4).unwrap());
        assert!(!stream_k_needs_fixup(120, 15, 4).unwrap());
    }

    // The scratch holds both meta arrays and both data regions.
    #[test]
    fn fixup_scratch_len_covers_meta_and_both_data_regions() {
        let (grid, ncols, d) = (120, 64, 256);
        let meta_floats = 2 * 2 * grid * ncols;
        let data_floats = 2 * grid * ncols * d;
        assert_eq!(
            fixup_scratch_len(grid, ncols, d).unwrap(),
            meta_floats + data_floats
        );
    }

    // The fixup entry carries the dtype tag and the kernel's tile shape.
    #[test]
    fn fixup_kernel_name_matches_decl_macro_naming() {
        assert_eq!(
            fixup_kernel_name(256, 16, 4, false),
            "crane_fattn_fixup_f16_d256_c16_s4"
        );
        assert_eq!(
            fixup_kernel_name(128, 8, 2, true),
            "crane_fattn_fixup_bf16_d128_c8_s2"
        );
    }
}

/// Shared real-GPU test utilities for [`super::fattn_mma`]'s/
/// [`super::fattn_tile`]'s `gpu_tests`: the naive matmul/mask/softmax/matmul
/// reference and the test-only GPU device picker. `pub(crate)` so both
/// sibling modules reuse the same reference rather than duplicating it.
#[cfg(all(test, crane_fattn))]
pub(crate) mod test_support {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::ops::softmax_last_dim;

    /// The real GPU device these tests run against: CUDA when built with
    /// that feature, `ROCm` otherwise. The two features are mutually
    /// exclusive, so exactly one of these applies in any given build.
    #[cfg(feature = "cuda")]
    pub(crate) fn test_gpu_device() -> Device {
        Device::new_cuda(0).expect("cuda device")
    }

    /// See [`test_gpu_device`].
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    pub(crate) fn test_gpu_device() -> Device {
        Device::new_rocm(0).expect("rocm device")
    }

    /// Random `[b, h, s, d]` view of a `[b, h, s + 16, d]` buffer narrowed
    /// past its first 16 rows, like a model's KV cache. Its BSHD transpose
    /// is the strided layout the kernels read without a copy.
    pub(crate) fn narrowed_bhsd(
        (b, h, s, d): (usize, usize, usize, usize),
        dtype: DType,
        dev: &Device,
    ) -> Tensor {
        Tensor::randn(0f32, 1f32, (b, h, s + 16, d), dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
            .narrow(2, 16, s)
            .unwrap()
    }

    /// Mask mode for [`naive_attention`], mirroring the fattn kernels' own
    /// causal/full/windowed contract.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum MaskMode {
        Causal,
        Full,
        Windowed,
    }

    /// `softmax(q @ k^T * scale [+ mask]) @ v` in F32 on `q`'s device,
    /// expanding `k`/`v`'s heads for GQA. `q`, `k`, `v` are BSHD.
    pub(crate) fn naive_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        mask_mode: MaskMode,
        kv_offset: usize,
        window_left: usize,
        window_right: usize,
    ) -> Vec<f32> {
        let (b, sq, hq, d) = q.dims4().unwrap();
        let (_, skv, hkv, _) = k.dims4().unwrap();
        let n_rep = hq / hkv;

        let q = q.transpose(1, 2).unwrap().contiguous().unwrap();
        let k = k.transpose(1, 2).unwrap().contiguous().unwrap();
        let v = v.transpose(1, 2).unwrap().contiguous().unwrap();

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
        let k = expand_kv(&k);
        let v = expand_kv(&v);

        let scores = (q.matmul(&k.transpose(2, 3).unwrap()).unwrap() * f64::from(scale)).unwrap();
        let scores = if mask_mode == MaskMode::Full {
            scores
        } else {
            let mut mask_vals = vec![0f32; sq * skv];
            for i in 0..sq {
                for j in 0..skv {
                    let rel = j as i64 - (i as i64 + kv_offset as i64);
                    let valid = if mask_mode == MaskMode::Causal {
                        rel <= 0
                    } else {
                        rel >= -(window_left as i64) && rel <= window_right as i64
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
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    ///// Long-sequence inputs ////////////////////////////////////////////

    /// Q heads of the long-sequence inputs. With [`LONG_SEQ_HKV`] and
    /// `head_dim=256` this is Qwen3.5's full-attention shape.
    pub(crate) const LONG_SEQ_HQ: usize = 8;
    /// KV heads of the long-sequence inputs.
    pub(crate) const LONG_SEQ_HKV: usize = 2;

    /// Value distribution of [`long_seq_inputs`]. The logit sizes below hold
    /// at every `head_dim`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum LongSeqDist {
        /// q, k, v ~ N(0, 1).
        Normal,
        /// q, k ~ N(0, 4), which gives logits a std of about 4. v ~ N(0, 1).
        Scaled,
        /// [`LongSeqDist::Normal`] plus two keys every query aligns with. The
        /// second one is larger and sits at 60% of the sequence, so rows past
        /// it see a new running max many KV tiles into the loop.
        Spike,
    }

    /// Seeded BSHD `[1, seq, LONG_SEQ_HQ, d]` q and `[1, seq, LONG_SEQ_HKV,
    /// d]` k/v, flattened, where `d` is `head_dim`. Candle's `randn` cannot
    /// be seeded, which a reproducible regression test needs.
    pub(crate) fn long_seq_inputs(
        seq: usize,
        d: usize,
        dist: LongSeqDist,
        seed: u64,
    ) -> [Vec<f32>; 3] {
        let qk_std = if dist == LongSeqDist::Scaled {
            2.0
        } else {
            1.0
        };
        let seed = seed.wrapping_mul(4);
        let mut q = seeded_normals(seed, seq * LONG_SEQ_HQ * d, qk_std);
        let mut k = seeded_normals(seed + 1, seq * LONG_SEQ_HKV * d, qk_std);
        let mut v = seeded_normals(seed + 2, seq * LONG_SEQ_HKV * d, 1.0);
        if dist == LongSeqDist::Spike {
            let n_rep = LONG_SEQ_HQ / LONG_SEQ_HKV;
            // head_dim is at most 512, exact in an f32.
            #[allow(clippy::cast_precision_loss)]
            let sqrt_d = (d as f32).sqrt();
            let mut dirs = seeded_normals(seed + 3, LONG_SEQ_HKV * d, 1.0);
            for (g, u) in dirs.chunks_exact_mut(d).enumerate() {
                let norm = u.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in u.iter_mut() {
                    *x /= norm;
                }
                for row in q.chunks_exact_mut(LONG_SEQ_HQ * d) {
                    for h in row[g * n_rep * d..][..n_rep * d].chunks_exact_mut(d) {
                        h.iter_mut().zip(&*u).for_each(|(x, e)| *x += 4.0 * e);
                    }
                }
                // Logits 4 * mag / sqrt(d) of about 8 and 16, against a
                // typical std of 1.
                for (pos, mag) in [(3, 2.0 * sqrt_d), (seq * 3 / 5, 4.0 * sqrt_d)] {
                    let at = (pos * LONG_SEQ_HKV + g) * d;
                    k[at..at + d]
                        .iter_mut()
                        .zip(&*u)
                        .for_each(|(x, e)| *x = mag * e);
                    v[at..at + d].iter_mut().for_each(|x| *x *= 3.0);
                }
            }
        }
        [q, k, v]
    }

    /// Causal `softmax(q @ k^T * scale) @ v` in F64 on the CPU, for the given
    /// query `rows` and every head. `q`, `k`, `v` are flat long-sequence
    /// inputs (see [`long_seq_inputs`]) with `seq_q == seq_kv` and
    /// `head_dim` `d`. Returns `[rows.len(), LONG_SEQ_HQ, d]`, flattened.
    // q/k/v/h/g/d are standard ML tensor-shape notation, see `naive_attention`.
    #[allow(clippy::many_single_char_names)]
    pub(crate) fn attention_rows_f64(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        d: usize,
        scale: f32,
        rows: &[usize],
    ) -> Vec<f64> {
        fn kv_row(t: &[f32], j: usize, g: usize, d: usize) -> &[f32] {
            &t[(j * LONG_SEQ_HKV + g) * d..][..d]
        }

        let n_rep = LONG_SEQ_HQ / LONG_SEQ_HKV;
        let mut out = Vec::with_capacity(rows.len() * LONG_SEQ_HQ * d);
        let mut logits = Vec::new();
        let mut acc = vec![0f64; d];
        for &i in rows {
            for h in 0..LONG_SEQ_HQ {
                let g = h / n_rep;
                let qi = &q[(i * LONG_SEQ_HQ + h) * d..][..d];
                logits.clear();
                logits.extend((0..=i).map(|j| {
                    let dot: f64 = qi
                        .iter()
                        .zip(kv_row(k, j, g, d))
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    dot * f64::from(scale)
                }));
                let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                acc.fill(0.0);
                let mut sum = 0.0;
                for (j, &l) in logits.iter().enumerate() {
                    let p = (l - max).exp();
                    sum += p;
                    let vj = kv_row(v, j, g, d);
                    acc.iter_mut()
                        .zip(vj)
                        .for_each(|(a, &x)| *a += p * f64::from(x));
                }
                out.extend(acc.iter().map(|a| a / sum));
            }
        }
        out
    }

    /// `n` samples from `N(0, std^2)`, deterministic in `seed`.
    pub(crate) fn seeded_normals(seed: u64, n: usize, std: f32) -> Vec<f32> {
        let mut state = seed;
        // Top 53 bits mapped to (0, 1], so the log below stays finite. The
        // u64 -> f64 casts keep at most 53 bits, exact in an f64.
        #[allow(clippy::cast_precision_loss)]
        let mut uniform = || ((splitmix64(&mut state) >> 11) + 1) as f64 / (1u64 << 53) as f64;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            // Box-Muller: two uniforms give two independent normals.
            let r = (-2.0 * uniform().ln()).sqrt() * f64::from(std);
            let (sin, cos) = (std::f64::consts::TAU * uniform()).sin_cos();
            // Rounds a sample of a few units to f32, which is the intent.
            #[allow(clippy::cast_possible_truncation)]
            let pair = [(r * cos) as f32, (r * sin) as f32];
            out.push(pair[0]);
            if out.len() < n {
                out.push(pair[1]);
            }
        }
        out
    }

    /// Advances `state` and returns the next splitmix64 output.
    pub(crate) fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}
