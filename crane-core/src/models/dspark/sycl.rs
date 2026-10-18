// SPDX-License-Identifier: MIT

//! Launcher for `kernels/sycl/dspark.cpp`: the Markov chain of
//! [`super::DSparkDrafter::markov_chain`] in one submission (the SYCL
//! counterpart of the CUDA and Metal launchers).

use std::ffi::c_void;

use candle_core::op::BackpropOp;
use candle_core::{DType, Result, Storage, SyclStorage, Tensor};

/// `DS_MAX_RANK` / `DS_MAX_K` in `dspark.cpp`.
const MAX_RANK: usize = 1024;
const MAX_K: usize = 1024;

// libcrane_gdn_sycl.so — linked by build.rs when `--features sycl`.
unsafe extern "C" {
    fn crane_dspark_markov_chain_sycl(
        queue: *mut c_void,
        dtype: i32,
        ids: *const u32,
        logits: *const f32,
        w2: *const c_void,
        w1: *const c_void,
        tokens: *mut u32,
        prevs: *mut u32,
        anchor: u32,
        blk: u32,
        k: u32,
        r: u32,
    ) -> i32;
}

/// Pointer to the first element of a contiguous SYCL tensor.
fn ptr(t: &Tensor) -> Result<*const c_void> {
    let (s, l) = t.storage_and_layout();
    match &*s {
        Storage::Sycl(st) => Ok(unsafe {
            st.buf()
                .as_ptr()
                .cast::<u8>()
                .add(l.start_offset() * t.dtype().size_in_bytes())
                .cast()
        }),
        _ => candle_core::bail!("dspark: expected a sycl tensor"),
    }
}

/// `ids` `[blk, k]` U32, `logits` `[blk, k]` F32, `w2` `[blk, k, r]` and `w1`
/// `[vocab, r]` in one float dtype, all on one SYCL device. Returns the
/// drafted tokens and the token each position was conditioned on, both `[blk]`
/// U32 on the device. `None` when the shapes or dtypes don't fit the kernel.
///
/// # Errors
///
/// Returns an error if the launch fails or the inputs are not on SYCL.
pub fn markov_chain(
    ids: &Tensor,
    logits: &Tensor,
    w2: &Tensor,
    w1: &Tensor,
    anchor: u32,
) -> Result<Option<(Tensor, Tensor)>> {
    let (blk, k) = ids.dims2()?;
    let r = w1.dim(1)?;
    let dtype_tag = match w1.dtype() {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        _ => return Ok(None),
    };
    if k == 0 || k > MAX_K || r > MAX_RANK || w2.dtype() != w1.dtype() || ids.dtype() != DType::U32
    {
        return Ok(None);
    }
    let dev = ids.device().as_sycl_device()?.clone();
    let (ids, logits) = (
        ids.contiguous()?,
        logits.to_dtype(DType::F32)?.contiguous()?,
    );
    let (w2, w1) = (w2.contiguous()?, w1.contiguous()?);
    let to_u32 = |v: usize| {
        u32::try_from(v).map_err(|_| candle_core::Error::Msg(format!("{v} does not fit u32")))
    };
    let tokens = dev.alloc_bytes(blk * std::mem::size_of::<u32>())?;
    let prevs = dev.alloc_bytes(blk * std::mem::size_of::<u32>())?;
    let status = unsafe {
        crane_dspark_markov_chain_sycl(
            dev.queue().native_ptr(),
            dtype_tag,
            ptr(&ids)?.cast(),
            ptr(&logits)?.cast(),
            ptr(&w2)?,
            ptr(&w1)?,
            tokens.as_mut_ptr().cast(),
            prevs.as_mut_ptr().cast(),
            anchor,
            to_u32(blk)?,
            to_u32(k)?,
            to_u32(r)?,
        )
    };
    if status != 0 {
        candle_core::bail!("crane_dspark_markov_chain_sycl failed (status {status})");
    }
    let wrap = |buf| {
        let storage = Storage::Sycl(SyclStorage::from_buffer(&dev, buf, DType::U32, blk));
        Tensor::from_storage(storage, blk, BackpropOp::none(), false)
    };
    Ok(Some((wrap(tokens), wrap(prevs))))
}
