// SPDX-License-Identifier: MIT

//! Launcher for the exact Metal top-k in `kernels/metal/topk.metal`, the port
//! of `kernels/cuda/topk.cu` (same two stages, same total order, same `k`
//! limit; see [`super::cuda_topk`] on CUDA builds).

use candle_core::{DType, Result, Tensor};
use candle_metal_kernels::metal::ComputeCommandEncoder;
use objc2_metal::MTLSize;

use crate::ops::metal_util;

/// `TOPK_THREADS` in the kernel.
const THREADS: usize = 256;
/// Elements a threadgroup consumes between two flush checks:
/// `TOPK_THREADS * TOPK_UNROLL`.
const STEP: usize = 256 * 4;
/// Largest `k` the kernel accepts (`TOPK_CAP / 4`).
pub const MAX_K: usize = 2048 / 4;

/// `topk_args` in the kernel.
#[repr(C)]
struct Args {
    n: u32,
    k: u32,
    items_per_group: u32,
}

fn u32_of(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("metal topk: {n} exceeds u32")))
}

/// Top-`k` indices of every row of `x: [rows, n]` f32 on Metal, as `[rows, k]`
/// U32: value descending, index ascending among equal values. `None` above
/// [`MAX_K`].
///
/// # Errors
///
/// Returns an error if `x` is not a Metal f32 matrix, `k` is out of range, or
/// a dispatch fails.
pub fn topk_indices_rows(x: &Tensor, k: usize) -> Result<Option<Tensor>> {
    let (rows, n) = x.dims2()?;
    if k == 0 || k > n {
        candle_core::bail!("topk_indices expects 0 < k <= n");
    }
    if k > MAX_K || x.dtype() != DType::F32 {
        return Ok(None);
    }
    let x = x.contiguous()?;
    let dev = x.device().as_metal_device()?.clone();
    let source = || include_str!("../../../kernels/metal/topk.metal").to_string();
    let stage1 = metal_util::pipeline(&dev, "topk", source, "topk_stage1_f32")?;
    let stage2 = metal_util::pipeline(&dev, "topk", source, "topk_stage2_u64")?;

    // Grid first, then the span each threadgroup scans (sizing the span first
    // and clamping the grid would leave the tail of the row unvisited).
    //
    // Stage 2 is the cost: one threadgroup per row merges `groups * k` keys.
    // Long stage-1 spans keep that small; with several rows, stage 1 has
    // parallelism to spare, so the spans grow with the row count. At
    // n = 248320, k = 256 on an M3 Pro, against CUDA's 4-step spans: 1 row
    // ~0.75 -> ~0.6 ms (16 spans), 15 rows ~2.6 -> ~0.9 ms (4 spans per row).
    let groups = (n.div_ceil(STEP * 16) / rows.clamp(1, 4)).clamp(1, 256);
    let items = n.div_ceil(groups);
    // 8-byte keys; candle has no U64, I64 is the same size.
    let keys = metal_util::output(&dev, rows * groups * k, DType::I64, "topk_keys")?;
    let out = metal_util::output(&dev, rows * k, DType::U32, "topk_idx")?;
    let size = |width, height| MTLSize {
        width,
        height,
        depth: 1,
    };
    let threads = size(THREADS, 1);
    {
        let (storage, _) = x.storage_and_layout();
        let (buf, offset) = metal_util::buffer(&storage, &x, 0, "x")?;
        let encoder = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = encoder.as_ref();
        enc.set_compute_pipeline_state(&stage1);
        enc.set_input_buffer(0, Some(buf), offset);
        enc.set_output_buffer(1, Some(&keys), 0);
        enc.set_bytes(
            2,
            &Args {
                n: u32_of(n)?,
                k: u32_of(k)?,
                items_per_group: u32_of(items)?,
            },
        );
        enc.dispatch_thread_groups(size(groups, rows), threads);
    }
    {
        let encoder = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = encoder.as_ref();
        enc.set_compute_pipeline_state(&stage2);
        enc.set_input_buffer(0, Some(&keys), 0);
        enc.set_output_buffer(1, Some(&out), 0);
        enc.set_bytes(
            2,
            &Args {
                n: u32_of(groups * k)?,
                k: u32_of(k)?,
                items_per_group: 0,
            },
        );
        enc.dispatch_thread_groups(size(1, rows), threads);
    }
    Ok(Some(metal_util::wrap(&dev, out, (rows, k), DType::U32)))
}
