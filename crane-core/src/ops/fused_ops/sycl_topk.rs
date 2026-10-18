// SPDX-License-Identifier: MIT

//! Launcher for the exact SYCL top-k in `kernels/sycl/topk.cpp`, the port of
//! `kernels/metal/topk.metal` (same two stages, same total order, same `k`
//! limit).

use std::ffi::c_void;

use candle_core::op::BackpropOp;
use candle_core::{DType, Result, Storage, SyclStorage, Tensor};

/// Elements a work-group consumes between two flush checks:
/// `TOPK_THREADS * TOPK_UNROLL`.
const STEP: usize = 256 * 4;
/// Largest `k` the kernel accepts (`TOPK_CAP / 4`).
pub const MAX_K: usize = 2048 / 4;

// libcrane_gdn_sycl.so — linked by build.rs when `--features sycl`.
unsafe extern "C" {
    fn crane_topk_rows_sycl(
        queue: *mut c_void,
        x: *const f32,
        keys: *mut u64,
        out: *mut u32,
        rows: u32,
        n: u32,
        k: u32,
        groups: u32,
        items_per_group: u32,
    ) -> i32;
}

fn u32_of(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("sycl topk: {n} exceeds u32")))
}

/// Top-`k` indices of every row of `x: [rows, n]` f32 on SYCL, as `[rows, k]`
/// U32: value descending, index ascending among equal values. `None` above
/// [`MAX_K`] or for a non-f32 input.
///
/// # Errors
///
/// Returns an error if `x` is not a SYCL matrix, `k` is out of range, or the
/// launch fails.
pub fn topk_indices_rows(x: &Tensor, k: usize) -> Result<Option<Tensor>> {
    let (rows, n) = x.dims2()?;
    if k == 0 || k > n {
        candle_core::bail!("topk_indices expects 0 < k <= n");
    }
    if k > MAX_K || x.dtype() != DType::F32 {
        return Ok(None);
    }
    let x = x.contiguous()?;
    let dev = x.device().as_sycl_device()?.clone();

    // Same grid sizing as the Metal launcher: long stage-1 spans keep the
    // single-work-group stage 2 small, and more rows give stage 1 enough
    // parallelism to afford longer spans.
    let groups = (n.div_ceil(STEP * 16) / rows.clamp(1, 4)).clamp(1, 256);
    let items = n.div_ceil(groups);
    let keys = dev.alloc_bytes(rows * groups * k * std::mem::size_of::<u64>())?;
    let out = dev.alloc_bytes(rows * k * std::mem::size_of::<u32>())?;

    let (storage, layout) = x.storage_and_layout();
    let x_ptr = match &*storage {
        Storage::Sycl(st) => unsafe { st.buf().as_ptr().cast::<f32>().add(layout.start_offset()) },
        _ => candle_core::bail!("sycl topk: input must be a sycl tensor"),
    };
    let status = unsafe {
        crane_topk_rows_sycl(
            dev.queue().native_ptr(),
            x_ptr,
            keys.as_mut_ptr().cast(),
            out.as_mut_ptr().cast(),
            u32_of(rows)?,
            u32_of(n)?,
            u32_of(k)?,
            u32_of(groups)?,
            u32_of(items)?,
        )
    };
    drop(storage);
    if status != 0 {
        candle_core::bail!("crane_topk_rows_sycl failed (status {status})");
    }
    let storage = Storage::Sycl(SyclStorage::from_buffer(&dev, out, DType::U32, rows * k));
    Ok(Some(Tensor::from_storage(
        storage,
        (rows, k),
        BackpropOp::none(),
        false,
    )))
}
