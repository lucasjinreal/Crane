// SPDX-License-Identifier: MIT
//! Shared types, per-architecture tuning tables, and kernel-selection logic
//! for the vendored flash-attention MMA ([`super::fattn_mma`]) and tile
//! ([`super::fattn_tile`]) kernel dispatchers. Currently supports only (see
//! `kernels/cuda/fattn/`'s module docs): `parallel_blocks=1`, no stream-K,
//! F16 only, `head_dim` in {64, 128, 256, 512}.
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
#![cfg(any(feature = "cuda", feature = "rocm"))]

use candle_core::{Layout, Result};

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
/// padding granularity `select_mma_ncols` requires of `seq_kv` before
/// picking a GQA-batched `ncols2 > 1` (`gqa_opt_applies` below).
pub(crate) const FATTN_KQ_STRIDE: usize = 256;

/// Mirrors llama.cpp's `fattn_mma_config` (`fattn_mma_f16.cuh`): per-launch
/// tuning knobs selected by `(head_dim, ncols1*ncols2)`. Fields map directly
/// to the kernel's `__launch_bounds__`/shared-memory sizing. Upstream's
/// `occupancy` field (a hint for `cudaOccupancyMaxActiveBlocksPerMultiprocessor`-
/// driven `parallel_blocks` tuning) is dropped: `parallel_blocks=1`
/// unconditionally (see this module's doc comment), so occupancy is never
/// consulted.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FattnMmaConfig {
    /// Threads per block (`block_dim.x * block_dim.y`).
    pub nthreads: usize,
    /// KV rows processed per inner-loop iteration.
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
        (64, 8) => C { nthreads: 128, nbatch_fa: 128, nbatch_k2: 32, nbatch_v2: 32, nbatch_combine: 32, nstages_target: 2, q_in_reg: true },
        (64, 16 | 32 | 64) => C { nthreads: 128, nbatch_fa: 64, nbatch_k2: 32, nbatch_v2: 32, nbatch_combine: 32, nstages_target: 2, q_in_reg: true },
        (128, 8) => C { nthreads: 128, nbatch_fa: 128, nbatch_k2: 64, nbatch_v2: 64, nbatch_combine: 64, nstages_target: 2, q_in_reg: true },
        (128, 16 | 32 | 64) => C { nthreads: 128, nbatch_fa: 64, nbatch_k2: 64, nbatch_v2: 64, nbatch_combine: 64, nstages_target: 2, q_in_reg: true },
        (256, 8) => C { nthreads: 128, nbatch_fa: 64, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 128, nstages_target: 2, q_in_reg: true },
        (256, 16) => C { nthreads: 64, nbatch_fa: 32, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 128, nstages_target: 2, q_in_reg: true },
        (256, 32 | 64) => C { nthreads: 128, nbatch_fa: 32, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 128, nstages_target: 2, q_in_reg: true },
        (512, 8 | 16) => C { nthreads: 64, nbatch_fa: 32, nbatch_k2: 256, nbatch_v2: 256, nbatch_combine: 128, nstages_target: 1, q_in_reg: false },
        (512, 32) => C { nthreads: 128, nbatch_fa: 32, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 128, nstages_target: 1, q_in_reg: false },
        (512, 64) => C { nthreads: 256, nbatch_fa: 32, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 128, nstages_target: 1, q_in_reg: false },
        _ => return None,
    })
}

/// AMD RDNA3+ (WMMA) tuning table (`ggml_cuda_fattn_mma_get_config_rdna`),
/// same scope as [`mma_config_ampere`].
#[rustfmt::skip]
fn mma_config_rdna(dkq: usize, ncols: usize) -> Option<FattnMmaConfig> {
    use FattnMmaConfig as C;
    Some(match (dkq, ncols) {
        (64, 8 | 16 | 32 | 64) => C { nthreads: 128, nbatch_fa: 64, nbatch_k2: 32, nbatch_v2: 32, nbatch_combine: 32, nstages_target: 1, q_in_reg: true },
        (128, 8 | 16) => C { nthreads: 64, nbatch_fa: 32, nbatch_k2: 64, nbatch_v2: 64, nbatch_combine: 64, nstages_target: 1, q_in_reg: true },
        (128, 32 | 64) => C { nthreads: 128, nbatch_fa: 64, nbatch_k2: 64, nbatch_v2: 64, nbatch_combine: 64, nstages_target: 1, q_in_reg: true },
        (256, 8 | 16) => C { nthreads: 64, nbatch_fa: 32, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 128, nstages_target: 1, q_in_reg: true },
        (256, 32 | 64) => C { nthreads: 256, nbatch_fa: 64, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 64, nstages_target: 1, q_in_reg: true },
        (512, 8 | 16) => C { nthreads: 128, nbatch_fa: 64, nbatch_k2: 96, nbatch_v2: 64, nbatch_combine: 128, nstages_target: 1, q_in_reg: true },
        (512, 32 | 64) => C { nthreads: 128, nbatch_fa: 32, nbatch_k2: 128, nbatch_v2: 128, nbatch_combine: 128, nstages_target: 1, q_in_reg: true },
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

/// Picks `ncols2` (how many GQA-sibling Q heads share one block's K/V load):
/// port of `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2`'s non-Volta
/// branches. `ncols2=1` is always valid for any `gqa_ratio` (it divides
/// every integer), so it is the universal fallback when `gqa_opt_applies`
/// is `false` or `gqa_ratio` doesn't divide evenly into a larger tile.
fn mma_ncols2(gqa_ratio: usize, gqa_opt_applies: bool, is_amd: bool) -> usize {
    if !gqa_opt_applies {
        return 1;
    }
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
/// gqa_ratio, seq_q, seq_kv)`, or `None` if no instantiated kernel covers
/// it. Callers fall back to the tile kernel in that case.
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
///   with `gqa_ratio=1` (plain MHA) or an unpadded `seq_kv` resolves to
///   `ncols2=1` (see [`mma_ncols2`]) and must use the tile kernel on AMD.
///   NVIDIA has no such restriction (confirmed: the `AMD_WMMA_AVAILABLE`
///   guard is absent on the non-AMD path), so `ncols2=1` stays valid there.
pub(crate) fn select_mma_ncols(
    head_dim: usize,
    gqa_ratio: usize,
    seq_q: usize,
    seq_kv: usize,
    is_amd: bool,
) -> Option<(usize, usize)> {
    if is_amd && head_dim > 256 {
        return None;
    }
    let gqa_opt_applies = seq_kv.is_multiple_of(FATTN_KQ_STRIDE);
    let ncols2 = mma_ncols2(gqa_ratio, gqa_opt_applies, is_amd);
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

/// Which kernel family [`select_mma_ncols`] would pick for, i.e. whether a
/// call with these shapes has an instantiated MMA kernel at all.
#[must_use]
pub fn select_kernel(
    head_dim: usize,
    gqa_ratio: usize,
    seq_q: usize,
    seq_kv: usize,
    is_amd: bool,
) -> FattnKernelType {
    if select_mma_ncols(head_dim, gqa_ratio, seq_q, seq_kv, is_amd).is_some() {
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

/// Shared shape/stride validation for the fattn kernels: checks k/v
/// shapes match, `head_dim` is one of {64, 128, 256, 512}, GQA head-count
/// divisibility, and last-axis contiguity. Returns `(batch, seq_q,
/// num_heads_q, head_dim, seq_kv, num_heads_kv)`.
///
/// # Errors
///
/// Returns an error if any of the above checks fail.
pub(crate) fn validate_fattn_bshd(
    l_q: &Layout,
    l_k: &Layout,
    l_v: &Layout,
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
    Ok((b, seq_q, h_q, d, seq_kv, h_kv))
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
                        for seq_kv in [1usize, 255, 256, 257, 512, 4096] {
                            let Some(pair) =
                                select_mma_ncols(head_dim, gqa_ratio, seq_q, seq_kv, is_amd)
                            else {
                                continue;
                            };
                            assert!(
                                instantiated_pairs(head_dim).contains(&pair),
                                "head_dim={head_dim} is_amd={is_amd} gqa_ratio={gqa_ratio} \
                                 seq_q={seq_q} seq_kv={seq_kv} picked uninstantiated pair {pair:?}"
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
        let err = validate_fattn_bshd(&q_l, &k_l, &v_l).expect_err("head_dim=100 must be rejected");
        assert!(err.to_string().contains("head_dim in"));
    }
}

/// Shared real-GPU test utilities for [`super::fattn_mma`]'s/
/// [`super::fattn_tile`]'s `gpu_tests`: the naive matmul/mask/softmax/matmul
/// reference and the test-only GPU device picker. `pub(crate)` so both
/// sibling modules reuse the same reference rather than duplicating it.
#[cfg(all(test, any(feature = "cuda", feature = "rocm")))]
pub(crate) mod test_support {
    use candle_core::{Device, Tensor};
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

    /// Mask mode for [`naive_attention`], mirroring the fattn kernels' own
    /// causal/full/windowed contract.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum MaskMode {
        Causal,
        Full,
        Windowed,
    }

    /// `softmax(q @ k^T * scale [+ mask]) @ v` in F32 on the CPU, expanding
    /// `k`/`v`'s heads for GQA. `q`, `k`, `v` are BSHD.
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
}
