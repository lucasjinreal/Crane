// SPDX-License-Identifier: MIT

//! Launcher for `kernels/metal/dspark.metal`: the Markov chain of
//! [`super::DSparkDrafter::markov_chain`] in one dispatch (the Metal
//! counterpart of [`super::cuda`]).

use candle_core::{DType, Result, Tensor};
use candle_metal_kernels::metal::ComputeCommandEncoder;
use objc2_metal::MTLSize;

use crate::ops::metal_util;

/// `DS_MAX_RANK` / `DS_MAX_K` / `DS_THREADS` in `dspark.metal`.
const MAX_RANK: usize = 1024;
const MAX_K: usize = 1024;
const THREADS: usize = 256;

/// `ds_args` in the kernel.
#[repr(C)]
struct Args {
    anchor: u32,
    blk: u32,
    k: u32,
    r: u32,
}

/// `ids` `[blk, k]` U32, `logits` `[blk, k]` F32, `w2` `[blk, k, r]` and `w1`
/// `[vocab, r]` in one float dtype, all on one Metal device. Returns the
/// drafted tokens and the token each position was conditioned on, both `[blk]`
/// U32 on the device. `None` when the shapes don't fit the kernel.
///
/// # Errors
///
/// Returns an error if the dispatch fails or the inputs are not on Metal.
pub fn markov_chain(
    ids: &Tensor,
    logits: &Tensor,
    w2: &Tensor,
    w1: &Tensor,
    anchor: u32,
) -> Result<Option<(Tensor, Tensor)>> {
    let (blk, k) = ids.dims2()?;
    let r = w1.dim(1)?;
    if k > MAX_K || r > MAX_RANK || w2.dtype() != w1.dtype() || ids.dtype() != DType::U32 {
        return Ok(None);
    }
    let dev = ids.device().as_metal_device()?.clone();
    let name = format!("dspark_markov_chain_{}", metal_util::float_tag(w1.dtype())?);
    let pipeline = metal_util::pipeline(
        &dev,
        "dspark",
        || include_str!("../../../kernels/metal/dspark.metal").to_string(),
        &name,
    )?;
    let (ids, logits) = (
        ids.contiguous()?,
        logits.to_dtype(DType::F32)?.contiguous()?,
    );
    let (w2, w1) = (w2.contiguous()?, w1.contiguous()?);
    let to_u32 = |v: usize| {
        u32::try_from(v).map_err(|_| candle_core::Error::Msg(format!("{v} does not fit u32")))
    };
    let args = Args {
        anchor,
        blk: to_u32(blk)?,
        k: to_u32(k)?,
        r: to_u32(r)?,
    };
    let tokens = metal_util::output(&dev, blk, DType::U32, "dspark_tokens")?;
    let prevs = metal_util::output(&dev, blk, DType::U32, "dspark_prevs")?;
    {
        let (ids_s, _) = ids.storage_and_layout();
        let (lg_s, _) = logits.storage_and_layout();
        let (w2_s, _) = w2.storage_and_layout();
        let (w1_s, _) = w1.storage_and_layout();
        let (ids_b, ids_o) = metal_util::buffer(&ids_s, &ids, 0, "ids")?;
        let (lg_b, lg_o) = metal_util::buffer(&lg_s, &logits, 0, "logits")?;
        let (w2_b, w2_o) = metal_util::buffer(&w2_s, &w2, 0, "w2")?;
        let (w1_b, w1_o) = metal_util::buffer(&w1_s, &w1, 0, "w1")?;
        let encoder = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = encoder.as_ref();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_input_buffer(0, Some(ids_b), ids_o);
        enc.set_input_buffer(1, Some(lg_b), lg_o);
        enc.set_input_buffer(2, Some(w2_b), w2_o);
        enc.set_input_buffer(3, Some(w1_b), w1_o);
        enc.set_output_buffer(4, Some(&tokens), 0);
        enc.set_output_buffer(5, Some(&prevs), 0);
        enc.set_bytes(6, &args);
        let one = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        enc.dispatch_thread_groups(
            one,
            MTLSize {
                width: THREADS,
                ..one
            },
        );
    }
    Ok(Some((
        metal_util::wrap(&dev, tokens, blk, DType::U32),
        metal_util::wrap(&dev, prevs, blk, DType::U32),
    )))
}
