// SPDX-License-Identifier: MIT

//! Launcher for `kernels/cuda/dspark.cu`: the Markov chain of
//! [`super::DSparkDrafter::markov_chain`] in one launch.

use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};
use candle_core::op::BackpropOp;
use candle_core::{CudaStorage, DType, Result, Storage, Tensor, bail};

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

const MODULE_NAME: &str = "crane_dspark";
/// `DS_MAX_RANK` / `DS_MAX_K` / `DS_THREADS` in `dspark.cu`.
const MAX_RANK: usize = 1024;
const MAX_K: usize = 1024;
const THREADS: u32 = 256;

/// `ids` `[blk, k]` U32, `logits` `[blk, k]` F32, `w2` `[blk, k, r]` and `w1`
/// `[vocab, r]` in one float dtype, all on one CUDA device. Returns the
/// drafted tokens and the token each position was conditioned on, both `[blk]`
/// U32 on the device. `None` when the shapes don't fit the kernel.
///
/// # Errors
///
/// Returns an error if a launch fails or the inputs are not on CUDA.
pub fn markov_chain(
    ids: &Tensor,
    logits: &Tensor,
    w2: &Tensor,
    w1: &Tensor,
    anchor: u32,
) -> Result<Option<(Tensor, Tensor)>> {
    let (blk, k) = ids.dims2()?;
    let r = w1.dim(1)?;
    // Rows are read 8 weights (16 bytes for half types) at a time.
    if k > MAX_K || r > MAX_RANK || r % 8 != 0 || w2.dtype() != w1.dtype() {
        return Ok(None);
    }
    let dev = ids.device().as_cuda_device()?.clone();
    let suffix = match w1.dtype() {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => return Ok(None),
    };
    let (ids, logits) = (
        ids.contiguous()?,
        logits.to_dtype(DType::F32)?.contiguous()?,
    );
    let (w2, w1) = (w2.contiguous()?, w1.contiguous()?);
    let tokens = dev.alloc_zeros::<u32>(blk)?;
    let prevs = dev.alloc_zeros::<u32>(blk)?;
    {
        let (ids_s, ids_l) = ids.storage_and_layout();
        let (lg_s, lg_l) = logits.storage_and_layout();
        let (w2_s, w2_l) = w2.storage_and_layout();
        let (w1_s, w1_l) = w1.storage_and_layout();
        let (Storage::Cuda(ids_c), Storage::Cuda(lg_c), Storage::Cuda(w2_c), Storage::Cuda(w1_c)) =
            (&*ids_s, &*lg_s, &*w2_s, &*w1_s)
        else {
            bail!("DSpark Markov chain inputs must be on CUDA")
        };
        let ids_v = ids_c.as_cuda_slice::<u32>()?.slice(ids_l.start_offset()..);
        let lg_v = lg_c.as_cuda_slice::<f32>()?.slice(lg_l.start_offset()..);
        let func = dev.get_or_load_custom_func(
            &format!("dspark_markov_chain_{suffix}"),
            MODULE_NAME,
            ptx::DSPARK,
        )?;
        let (blk_i, k_i, r_i) = (to_i32(blk)?, to_i32(k)?, to_i32(r)?);
        let mut builder = func.builder();
        builder.arg(&ids_v);
        builder.arg(&lg_v);
        let (w2_f32, w1_f32, w2_f16, w1_f16, w2_bf16, w1_bf16);
        match (&w2_c.slice, &w1_c.slice) {
            (CudaStorageSlice::F32(a), CudaStorageSlice::F32(b)) => {
                w2_f32 = a.slice(w2_l.start_offset()..);
                w1_f32 = b.slice(w1_l.start_offset()..);
                builder.arg(&w2_f32);
                builder.arg(&w1_f32);
            },
            (CudaStorageSlice::F16(a), CudaStorageSlice::F16(b)) => {
                w2_f16 = a.slice(w2_l.start_offset()..);
                w1_f16 = b.slice(w1_l.start_offset()..);
                builder.arg(&w2_f16);
                builder.arg(&w1_f16);
            },
            (CudaStorageSlice::BF16(a), CudaStorageSlice::BF16(b)) => {
                w2_bf16 = a.slice(w2_l.start_offset()..);
                w1_bf16 = b.slice(w1_l.start_offset()..);
                builder.arg(&w2_bf16);
                builder.arg(&w1_bf16);
            },
            _ => bail!("DSpark Markov chain: mismatched weight storage"),
        }
        builder.arg(&anchor);
        builder.arg(&tokens);
        builder.arg(&prevs);
        builder.arg(&blk_i);
        builder.arg(&k_i);
        builder.arg(&r_i);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
    }
    let wrap = |s| {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(s, dev.clone())),
            blk,
            BackpropOp::none(),
            false,
        )
    };
    Ok(Some((wrap(tokens), wrap(prevs))))
}

fn to_i32(v: usize) -> Result<i32> {
    i32::try_from(v).map_err(|_| candle_core::Error::Msg(format!("{v} does not fit i32")))
}
