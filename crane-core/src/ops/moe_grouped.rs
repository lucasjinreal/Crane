// SPDX-License-Identifier: MIT

//! Launchers for `kernels/cuda/moe_grouped.cu`: the grouped-by-expert `MoE`
//! matvec for small token batches over packed k-quant experts.
//!
//! [`group_pairs`] buckets a batch's (token, expert) pairs by expert on the
//! device; [`grouped_matvec`] then reads each used expert's weights once and
//! dots them against every activation row routed to it, where candle's
//! `indexed_moe_forward` streams the expert once per pair. One grouping serves
//! both projections of a layer. [`swiglu_quantize`] and [`combine`] fold the
//! glue between and after the projections into one launch each, which matters
//! as much as the matvec at decode, where the block is launch-bound.

use candle_core::cuda_backend::WrapErr;
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use candle_core::op::BackpropOp;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{CudaDevice, CudaStorage, DType, Result, Storage, Tensor, bail};

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

const MODULE_NAME: &str = "crane_moe_grouped";
/// Module holding `iq4_quantize_q8_*` (shared with `ops::quant_iq`).
const QUANTIZE_MODULE: &str = "crane_quant_iq4";
/// `MG_WARPS` / `MG_GROUP_THREADS` in `moe_grouped.cu`.
const WARPS: usize = 4;
const GROUP_THREADS: u32 = 256;

/// Whether [`grouped_matvec`] has a kernel for this weight type.
#[must_use]
pub fn supported(dtype: GgmlDType) -> bool {
    matches!(dtype, GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K)
}

/// A batch's (token, expert) pairs bucketed by expert, on the device.
pub struct ExpertGroups {
    buf: CudaSlice<u32>,
    dev: CudaDevice,
    n_pairs: usize,
    n_experts: usize,
}

/// Bucket the pairs of `ids` (`[tokens, top_k]` U32 on CUDA, values below
/// `n_experts`) by expert, in one single-block launch.
///
/// # Errors
///
/// Returns an error if `ids` is not a CUDA U32 tensor or the launch fails.
pub fn group_pairs(ids: &Tensor, n_experts: usize) -> Result<ExpertGroups> {
    if ids.dtype() != DType::U32 {
        bail!("expert ids must be U32, got {:?}", ids.dtype())
    }
    let dev = ids.device().as_cuda_device()?.clone();
    let ids = ids.flatten_all()?.contiguous()?;
    let n_pairs = ids.elem_count();
    let buf = unsafe { dev.alloc::<u32>(3 * n_pairs + 2) }?;
    {
        let (storage, layout) = ids.storage_and_layout();
        let Storage::Cuda(cuda) = &*storage else {
            bail!("expert ids must be on CUDA")
        };
        let ids_view = cuda.as_cuda_slice::<u32>()?.slice(layout.start_offset()..);
        let func = dev.get_or_load_custom_func("moe_group_pairs", MODULE_NAME, ptx::MOE_GROUPED)?;
        let (n_pairs_i, n_experts_i) = (int::<i32>(n_pairs)?, int::<i32>(n_experts)?);
        let mut builder = func.builder();
        builder.arg(&ids_view);
        builder.arg(&buf);
        builder.arg(&n_pairs_i);
        builder.arg(&n_experts_i);
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (GROUP_THREADS, 1, 1),
                shared_mem_bytes: int::<u32>(8 * n_experts)?,
            })
        }
        .w()?;
    }
    Ok(ExpertGroups {
        buf,
        dev,
        n_pairs,
        n_experts,
    })
}

/// For every pair `p` of `groups`, expert `ids[p]`'s `[rows, cols]` slice of
/// the packed `weights` (`[experts, rows, cols]`, Q4_K/Q5_K/Q6_K on the same
/// device) times activation row `p / x_div` of `xs` (`[_, cols]`, f32/f16/bf16).
/// Returns `[pairs, rows]` f32.
///
/// # Errors
///
/// Returns an error on an unsupported weight type, mismatched shapes or
/// devices, or a failed launch.
pub fn grouped_matvec(
    xs: &Tensor,
    groups: &ExpertGroups,
    weights: &QTensor,
    x_div: usize,
) -> Result<Tensor> {
    grouped_matvec_q8(&quantize(xs)?, groups, weights, x_div)
}

/// Activations quantized for [`grouped_matvec_q8`]: int8 values `[rows, cols]`
/// and one f32 scale per 32 of them (llama.cpp's `q8_1` minus the sums).
pub struct Q8Acts {
    xq: CudaSlice<i8>,
    xd: CudaSlice<f32>,
    rows: usize,
    cols: usize,
}

/// [`grouped_matvec`] on activations already in [`Q8Acts`] form.
///
/// # Errors
///
/// Returns an error on an unsupported weight type, mismatched shapes or
/// devices, or a failed launch.
pub fn grouped_matvec_q8(
    acts: &Q8Acts,
    groups: &ExpertGroups,
    weights: &QTensor,
    x_div: usize,
) -> Result<Tensor> {
    let ty = match weights.dtype() {
        GgmlDType::Q4K => "q4_k",
        GgmlDType::Q5K => "q5_k",
        GgmlDType::Q6K => "q6_k",
        other => bail!("grouped MoE matvec: unsupported weight type {other:?}"),
    };
    let (experts, rows, cols) = weights.shape().dims3()?;
    if experts != groups.n_experts {
        bail!(
            "grouped MoE matvec: {experts} experts in the weights, {} grouped",
            groups.n_experts
        )
    }
    if cols == 0 || cols % 256 != 0 || acts.cols != cols {
        bail!("grouped MoE matvec: activation width does not match {cols} weight columns")
    }
    // One warp per row reads the whole row and reduces it with shuffles only
    // (1.25-1.7x `indexed_moe_forward` from 1 to 32 tokens at 35B-A3B shapes;
    // splitting a row over 2 or 4 warps was slower everywhere). Rows wider
    // than the 8 blocks one thread can take are split over 2 warps. This picks
    // the `_i{ITERS}_w{WPR}` variant of `moe_grouped.cu`.
    let blocks = cols / 256;
    let (iters, warps_per_row) = match blocks {
        1 => (1, 1),
        2 => (2, 1),
        3 | 4 => (4, 1),
        5..=8 => (8, 1),
        9..=16 => (8, 2),
        _ => bail!("grouped MoE matvec: {cols} columns is wider than supported"),
    };
    if groups.n_pairs.div_ceil(x_div) != acts.rows {
        bail!(
            "grouped MoE matvec: {} activation rows for {} pairs / {x_div}",
            acts.rows,
            groups.n_pairs
        )
    }
    let dev = &groups.dev;

    let block_bytes = weights.dtype().type_size();
    let expert_stride = (rows * (cols / 256) * block_bytes) as u64;
    let weights_ptr = weights.device_ptr()? as u64;
    let out = unsafe { dev.alloc::<f32>(groups.n_pairs * rows) }?;
    let kernel = format!("moe_grouped_matvec_{ty}_i{iters}_w{warps_per_row}");
    let func = dev.get_or_load_custom_func(&kernel, MODULE_NAME, ptx::MOE_GROUPED)?;
    let rows_i = int::<i32>(rows)?;
    let cols_i = int::<i32>(cols)?;
    let x_div_i = int::<i32>(x_div)?;
    let n_pairs_i = int::<i32>(groups.n_pairs)?;
    let mut builder = func.builder();
    builder.arg(&weights_ptr);
    builder.arg(&expert_stride);
    builder.arg(&acts.xq);
    builder.arg(&acts.xd);
    builder.arg(&groups.buf);
    builder.arg(&out);
    builder.arg(&rows_i);
    builder.arg(&cols_i);
    builder.arg(&x_div_i);
    builder.arg(&n_pairs_i);
    // `rows_per_tb` in the kernel; one grid row per possible used expert, rows
    // past the actual count exit.
    let rows_per_block = WARPS / warps_per_row;
    let slots = groups.n_pairs.min(groups.n_experts);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (rows.div_ceil(rows_per_block) as u32, int::<u32>(slots)?, 1),
            block_dim: (32 * WARPS as u32, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        (groups.n_pairs, rows),
        BackpropOp::none(),
        false,
    ))
}

/// Quantize `xs` (`[_, cols]`, f32/f16/bf16 on CUDA, `cols` a multiple of 32)
/// to [`Q8Acts`].
///
/// # Errors
///
/// Returns an error on an unsupported dtype or a failed launch.
pub fn quantize(xs: &Tensor) -> Result<Q8Acts> {
    let dev = xs.device().as_cuda_device()?.clone();
    let cols = xs.dim(candle_core::D::Minus1)?;
    let rows = xs.elem_count() / cols;
    let n = rows * cols;
    let xs = xs.contiguous()?;
    let n_blocks = n / 32;
    let xq = unsafe { dev.alloc::<i8>(n) }?;
    let xd = unsafe { dev.alloc::<f32>(n_blocks) }?;
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Cuda(cuda) = &*storage else {
        bail!("grouped MoE matvec activations must be on CUDA")
    };
    let offset = layout.start_offset();
    let name = match xs.dtype() {
        DType::F32 => "iq4_quantize_q8_f32",
        DType::F16 => "iq4_quantize_q8_f16",
        DType::BF16 => "iq4_quantize_q8_bf16",
        other => bail!("grouped MoE matvec: unsupported activation dtype {other:?}"),
    };
    let func = dev.get_or_load_custom_func(name, QUANTIZE_MODULE, ptx::QUANT_IQ4)?;
    let n_blocks_i = int::<i32>(n_blocks)?;
    let (x_f32, x_f16, x_bf16);
    let mut builder = func.builder();
    match xs.dtype() {
        DType::F32 => {
            x_f32 = cuda.as_cuda_slice::<f32>()?.slice(offset..);
            builder.arg(&x_f32);
        },
        DType::F16 => {
            x_f16 = cuda.as_cuda_slice::<half::f16>()?.slice(offset..);
            builder.arg(&x_f16);
        },
        _ => {
            x_bf16 = cuda.as_cuda_slice::<half::bf16>()?.slice(offset..);
            builder.arg(&x_bf16);
        },
    }
    builder.arg(&xq);
    builder.arg(&xd);
    builder.arg(&n_blocks_i);
    // 8 warps per thread block, one warp per 32 activations.
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (n_blocks.div_ceil(8) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    drop(storage);
    Ok(Q8Acts { xq, xd, rows, cols })
}

/// SwiGLU of `gate_up` (`[pairs, 2 * inter]` f32 on CUDA, gate half first,
/// `inter` a multiple of 32) straight into [`Q8Acts`] `[pairs, inter]`: one
/// launch for what was two copies, `swiglu` and a quantize, bit-identical.
///
/// # Errors
///
/// Returns an error on a bad shape or dtype or a failed launch.
pub fn swiglu_quantize(gate_up: &Tensor) -> Result<Q8Acts> {
    let dev = gate_up.device().as_cuda_device()?.clone();
    let (pairs, width) = gate_up.dims2()?;
    let inter = width / 2;
    if gate_up.dtype() != DType::F32 || width % 64 != 0 {
        bail!("swiglu_quantize: want [pairs, 2 * inter] f32 with inter % 32 == 0")
    }
    let gate_up = gate_up.contiguous()?;
    let n_blocks = pairs * inter / 32;
    let xq = unsafe { dev.alloc::<i8>(pairs * inter) }?;
    let xd = unsafe { dev.alloc::<f32>(n_blocks) }?;
    {
        let (storage, layout) = gate_up.storage_and_layout();
        let Storage::Cuda(cuda) = &*storage else {
            bail!("swiglu_quantize input must be on CUDA")
        };
        let x = cuda.as_cuda_slice::<f32>()?.slice(layout.start_offset()..);
        let func =
            dev.get_or_load_custom_func("moe_swiglu_quantize_f32", MODULE_NAME, ptx::MOE_GROUPED)?;
        let (pairs_i, inter_i) = (int::<i32>(pairs)?, int::<i32>(inter)?);
        let mut builder = func.builder();
        builder.arg(&x);
        builder.arg(&xq);
        builder.arg(&xd);
        builder.arg(&pairs_i);
        builder.arg(&inter_i);
        // 8 warps per thread block, one warp per 32 outputs.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (int::<u32>(n_blocks.div_ceil(8))?, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .w()?;
    }
    Ok(Q8Acts {
        xq,
        xd,
        rows: pairs,
        cols: inter,
    })
}

/// `out[t] = sum_k weights[t, k] * y[t * top_k + k]` (`y` `[tokens * top_k,
/// hidden]` f32 in pair order, `weights` `[tokens, top_k]` f32), summed over
/// `k` in order in f32 and returned as `[tokens, hidden]` in `out_dtype`, in
/// one launch.
///
/// # Errors
///
/// Returns an error on a bad shape or dtype or a failed launch.
pub fn combine(y: &Tensor, weights: &Tensor, out_dtype: DType) -> Result<Tensor> {
    let dev = y.device().as_cuda_device()?.clone();
    let (tokens, top_k) = weights.dims2()?;
    let (n, hidden) = y.dims2()?;
    if n != tokens * top_k || y.dtype() != DType::F32 || weights.dtype() != DType::F32 {
        bail!("combine: want y [tokens * top_k, hidden] and weights [tokens, top_k], both f32")
    }
    let (y, weights) = (y.contiguous()?, weights.contiguous()?);
    let (ys, yl) = y.storage_and_layout();
    let (ws, wl) = weights.storage_and_layout();
    let (Storage::Cuda(yc), Storage::Cuda(wc)) = (&*ys, &*ws) else {
        bail!("combine inputs must be on CUDA")
    };
    let y_view = yc.as_cuda_slice::<f32>()?.slice(yl.start_offset()..);
    let w_view = wc.as_cuda_slice::<f32>()?.slice(wl.start_offset()..);
    let total = tokens * hidden;
    let config = LaunchConfig {
        grid_dim: (int::<u32>(total.div_ceil(256))?, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (tokens_i, top_k_i, hidden_i) =
        (int::<i32>(tokens)?, int::<i32>(top_k)?, int::<i32>(hidden)?);
    macro_rules! launch {
        ($ty:ty, $name:literal) => {{
            let out = unsafe { dev.alloc::<$ty>(total) }?;
            let func = dev.get_or_load_custom_func($name, MODULE_NAME, ptx::MOE_GROUPED)?;
            let mut builder = func.builder();
            builder.arg(&y_view);
            builder.arg(&w_view);
            builder.arg(&out);
            builder.arg(&tokens_i);
            builder.arg(&top_k_i);
            builder.arg(&hidden_i);
            unsafe { builder.launch(config) }.w()?;
            CudaStorage::wrap_cuda_slice(out, dev.clone())
        }};
    }
    let storage = match out_dtype {
        DType::F32 => launch!(f32, "moe_combine_f32"),
        DType::F16 => launch!(half::f16, "moe_combine_f16"),
        DType::BF16 => launch!(half::bf16, "moe_combine_bf16"),
        other => bail!("combine: unsupported output dtype {other:?}"),
    };
    Ok(Tensor::from_storage(
        Storage::Cuda(storage),
        (tokens, hidden),
        BackpropOp::none(),
        false,
    ))
}

fn int<T: TryFrom<usize>>(v: usize) -> Result<T> {
    T::try_from(v)
        .map_err(|_| candle_core::Error::Msg(format!("{v} does not fit a kernel argument")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantized::iquant::IQuantType;
    use crate::quantized::test_util::random_blocks;
    use candle_core::Device;
    use candle_core::quantized::QStorage;
    use std::borrow::Cow;

    /// Routing with the overlap a real verify block has: a few hot experts
    /// shared by most tokens, the rest spread out.
    fn routing(tokens: usize, top_k: usize, experts: usize) -> Vec<u32> {
        (0..tokens)
            .flat_map(|t| {
                (0..top_k).map(move |k| {
                    let e = if k < 3 {
                        k * 5
                    } else {
                        (t * 37 + k * 11) % experts
                    };
                    u32::try_from(e).unwrap()
                })
            })
            .collect()
    }

    fn packed(
        device: &Device,
        ty: IQuantType,
        dtype: GgmlDType,
        shape: (usize, usize, usize),
        seed: u32,
    ) -> Result<QTensor> {
        let data = random_blocks(ty, shape.0 * shape.1 * shape.2 / 256, seed);
        QTensor::new(QStorage::from_data(Cow::Owned(data), device, dtype)?, shape)
    }

    fn q8_to_host(q: &Q8Acts) -> Result<(Vec<i8>, Vec<f32>)> {
        let dev = q.xq.context();
        let stream = dev.default_stream();
        Ok((
            stream.memcpy_dtov(&q.xq).w()?,
            stream.memcpy_dtov(&q.xd).w()?,
        ))
    }

    /// Bit-identical to the `swiglu` kernel followed by [`quantize`].
    #[test]
    fn swiglu_quantize_matches_swiglu_then_quantize() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let (pairs, inter) = (24usize, 512usize);
        let gate_up = Tensor::randn(0f32, 2.0, (pairs, 2 * inter), &device)?;
        let fused = swiglu_quantize(&gate_up)?;
        let act = crate::ops::fused_ops::swiglu::swiglu(
            &gate_up.narrow(1, 0, inter)?,
            &gate_up.narrow(1, inter, inter)?,
        )?;
        let reference = quantize(&act)?;
        let (fq, fd) = q8_to_host(&fused)?;
        let (rq, rd) = q8_to_host(&reference)?;
        assert_eq!(fq, rq);
        assert_eq!(
            fd.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            rd.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        Ok(())
    }

    /// Matches the weight / sum / cast chain it replaces.
    #[test]
    fn combine_matches_weighted_sum() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let (tokens, top_k, hidden) = (5usize, 8usize, 2048usize);
        let y = Tensor::randn(0f32, 1.0, (tokens * top_k, hidden), &device)?;
        let w = Tensor::rand(0f32, 1.0, (tokens, top_k), &device)?;
        let want = y
            .reshape((tokens, top_k, hidden))?
            .broadcast_mul(&w.unsqueeze(2)?)?
            .sum(1)?;
        for dtype in [DType::F32, DType::BF16] {
            let got = combine(&y, &w, dtype)?.to_dtype(DType::F32)?;
            let diff = (&got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let tol = if dtype == DType::F32 { 1e-5 } else { 3e-2 };
            assert!(diff < tol, "{dtype:?}: max diff {diff}");
        }
        Ok(())
    }

    /// Matches `indexed_moe_forward` (the same int8 activation quantization,
    /// so only the summation order differs) for both projection layouts.
    #[test]
    fn grouped_matvec_matches_indexed_moe_forward() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let (experts, top_k) = (64usize, 8usize);
        for (ty, dtype) in [
            (IQuantType::Q4K, GgmlDType::Q4K),
            (IQuantType::Q5K, GgmlDType::Q5K),
            (IQuantType::Q6K, GgmlDType::Q6K),
        ] {
            for tokens in [1usize, 3, 17] {
                let ids: Vec<u32> = routing(tokens, top_k, experts);
                let ids = Tensor::from_vec(ids, (tokens, top_k), &device)?;
                let groups = group_pairs(&ids, experts)?;
                // gate_up layout: every routed expert of a token shares its row.
                let w = packed(&device, ty, dtype, (experts, 96, 512), 3)?;
                let xs = (Tensor::randn(0f32, 1.0, (tokens, 512), &device)? * 0.1)?;
                let got = grouped_matvec(&xs, &groups, &w, top_k)?;
                let want = w
                    .indexed_moe_forward(&xs.unsqueeze(1)?, &ids)?
                    .reshape((tokens * top_k, 96))?;
                let exact = exact_product(&w, &ids, &xs, top_k)?;
                check(
                    &got,
                    &want,
                    &exact,
                    &format!("{ty:?} gate_up tokens={tokens}"),
                )?;
                // down layout: one activation row per pair.
                let w = packed(&device, ty, dtype, (experts, 64, 256), 5)?;
                let xs = (Tensor::randn(0f32, 1.0, (tokens, top_k, 256), &device)? * 0.1)?;
                let xs = xs.reshape((tokens * top_k, 256))?;
                let got = grouped_matvec(&xs, &groups, &w, 1)?;
                let want = w
                    .indexed_moe_forward(&xs.reshape((tokens, top_k, 256))?, &ids)?
                    .reshape((tokens * top_k, 64))?;
                let exact = exact_product(&w, &ids, &xs, 1)?;
                check(&got, &want, &exact, &format!("{ty:?} down tokens={tokens}"))?;
            }
        }
        Ok(())
    }

    /// Row `p` = dequantized expert `ids[p]` times activation row `p / x_div`.
    fn exact_product(w: &QTensor, ids: &Tensor, xs: &Tensor, x_div: usize) -> Result<Tensor> {
        let device = xs.device().clone();
        let dense = w.dequantize(&Device::Cpu)?;
        let xs = xs.to_device(&Device::Cpu)?;
        let ids = ids.flatten_all()?.to_vec1::<u32>()?;
        let rows = ids
            .iter()
            .enumerate()
            .map(|(p, &e)| {
                let x = xs.get(p / x_div)?.unsqueeze(1)?;
                dense.get(e as usize)?.matmul(&x)?.squeeze(1)
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::stack(&rows, 0)?.to_device(&device)
    }

    /// `got` (this kernel) and `candle` (`indexed_moe_forward`) against the
    /// exact f32 product. Both quantize the activations to int8 (the error that
    /// dominates), so ours must be within 10% of candle's.
    fn check(got: &Tensor, candle: &Tensor, exact: &Tensor, what: &str) -> Result<()> {
        let scale = exact.abs()?.max_all()?.to_scalar::<f32>()?;
        let err = |t: &Tensor| -> Result<f32> {
            Ok((t - exact)?.abs()?.max_all()?.to_scalar::<f32>()? / scale)
        };
        let (ours, theirs) = (err(got)?, err(candle)?);
        assert!(scale > 0.0, "{what}: reference is all zero");
        assert!(
            ours <= 1.1 * theirs + 1e-5,
            "{what}: rel err {ours:.2e} vs indexed_moe_forward {theirs:.2e}"
        );
        Ok(())
    }

    /// Both projections of one `MoE` layer at MiniCPM-V-4.7-35B-A3B shapes
    /// (256 experts, top-8, 2048 hidden, 512 expert FFN; `Q4_K` gate/up,
    /// `Q6_K` down): grouped kernels (grouping and activation quantization
    /// included) against `indexed_moe_forward`. `spread` routes every pair
    /// to a distinct expert where possible, `overlap` shares hot experts.
    #[test]
    #[ignore = "benchmark"]
    fn grouped_matvec_bench() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let (experts, hidden, inter, top_k) = (256usize, 2048usize, 512usize, 8usize);
        let gate_up = packed(
            &device,
            IQuantType::Q4K,
            GgmlDType::Q4K,
            (experts, 2 * inter, hidden),
            3,
        )?;
        let down = packed(
            &device,
            IQuantType::Q6K,
            GgmlDType::Q6K,
            (experts, hidden, inter),
            7,
        )?;
        for tokens in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] {
            for (name, ids) in [
                (
                    "spread ",
                    (0..tokens * top_k)
                        .map(|p| ((p * 37 + p / top_k * 11) % experts) as u32)
                        .collect::<Vec<_>>(),
                ),
                ("overlap", routing(tokens, top_k, experts)),
            ] {
                let ids = Tensor::from_vec(ids, (tokens, top_k), &device)?;
                let xs = (Tensor::randn(0f32, 1.0, (tokens, hidden), &device)? * 0.1)?;
                let hs = (Tensor::randn(0f32, 1.0, (tokens * top_k, inter), &device)? * 0.1)?;
                let iters = 50;
                let time = |f: &dyn Fn() -> Result<Tensor>| -> Result<f64> {
                    f()?;
                    device.synchronize()?;
                    let start = std::time::Instant::now();
                    for _ in 0..iters {
                        f()?;
                    }
                    device.synchronize()?;
                    Ok(start.elapsed().as_secs_f64() * 1e3 / f64::from(iters))
                };
                let candle = time(&|| {
                    gate_up.indexed_moe_forward(&xs.unsqueeze(1)?, &ids)?;
                    down.indexed_moe_forward(&hs.reshape((tokens, top_k, inter))?, &ids)
                })?;
                let grouped = time(&|| {
                    let groups = group_pairs(&ids, experts)?;
                    grouped_matvec(&xs, &groups, &gate_up, top_k)?;
                    grouped_matvec(&hs, &groups, &down, 1)
                })?;
                let used = ids
                    .flatten_all()?
                    .to_vec1::<u32>()?
                    .into_iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len();
                println!(
                    "tokens {tokens:4} {name} ({used:3} experts): indexed_moe_forward {candle:7.3} ms, grouped {grouped:7.3} ms, {:.2}x",
                    candle / grouped
                );
            }
        }
        Ok(())
    }
}
