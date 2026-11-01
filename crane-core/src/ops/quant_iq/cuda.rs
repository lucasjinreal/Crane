// SPDX-License-Identifier: MIT

//! Launchers for `kernels/cuda/quant_iq4.cu` (IQ4_XS `dp4a` decode, IQ4 dequant)
//! and `kernels/cuda/quant_iq.cu` (every other type, plus the by-expert-id
//! matvec packed `MoE` experts need; the k-quants are only reached by those).

use candle_core::cuda_backend::cudarc::driver::{DeviceRepr, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaDType, WrapErr};
use candle_core::op::BackpropOp;
use candle_core::{CudaDevice, CudaStorage, DType, Result, Storage, StorageRef, Tensor, bail};

use crate::quantized::iquant::IQuantType;

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

const MODULE_NAME: &str = "crane_quant_iq4";
/// Module of the type-generic float kernels (`kernels/cuda/quant_iq.cu`).
const GENERIC_MODULE: &str = "crane_quant_iq";
/// Output rows per thread block in the matvec kernels (one warp each).
const WARPS: usize = 4;
/// `ROWS_PER_BLOCK` in `quant_iq.cu`.
const ROWS_PER_BLOCK: usize = 4;

fn packed_slice(packed: &Tensor) -> Result<(StorageRef<'_>, usize)> {
    let (storage, layout) = packed.storage_and_layout();
    if !matches!(&*storage, Storage::Cuda(_)) {
        bail!("i-quant packed weight must be a CUDA u8 tensor")
    }
    Ok((storage, layout.start_offset()))
}

/// `input` (`[rows, cols]`, CUDA f32/f16/bf16) times the transposed packed
/// weight (`[output_rows, cols]`), returning `[rows, output_rows]` in
/// `out_dtype`.
///
/// IQ4_XS quantizes the activations to int8 (one scale per 32 values) and
/// runs an integer `dp4a` dot product, like llama.cpp, writing the output
/// directly in `out_dtype`. IQ4_NL uses a float kernel. Meant for
/// decode-sized `rows`; prefill should use [`dequantize`] and a regular
/// matmul instead.
pub fn matvec(
    input: &Tensor,
    packed: &Tensor,
    ty: IQuantType,
    output_rows: usize,
    cols: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    match ty {
        IQuantType::Iq4Xs => {
            let dev = input.device().as_cuda_device()?.clone();
            match out_dtype {
                DType::F32 => matvec_xs::<f32>(&dev, input, packed, output_rows, cols, out_dtype),
                DType::F16 => {
                    matvec_xs::<half::f16>(&dev, input, packed, output_rows, cols, out_dtype)
                },
                DType::BF16 => {
                    matvec_xs::<half::bf16>(&dev, input, packed, output_rows, cols, out_dtype)
                },
                other => bail!("i-quant matvec: unsupported output dtype {other:?}"),
            }
        },
        _ => {
            let rows = input.elem_count() / cols;
            let out = launch_matvec(
                input,
                packed,
                ty,
                None,
                1,
                rows,
                output_rows,
                cols,
                out_dtype,
            )?;
            let mut dims = input.dims().to_vec();
            *dims.last_mut().unwrap() = output_rows;
            out.reshape(dims)
        },
    }
}

fn launch_config(rows: usize, output_rows: usize) -> (usize, LaunchConfig) {
    let nr = if rows >= 4 { 4 } else { 1 };
    let config = LaunchConfig {
        grid_dim: (
            output_rows.div_ceil(WARPS) as u32,
            rows.div_ceil(nr) as u32,
            1,
        ),
        block_dim: (32 * WARPS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    (nr, config)
}

fn output_tensor<T: CudaDType + DeviceRepr>(
    buf: candle_core::cuda_backend::cudarc::driver::CudaSlice<T>,
    dev: CudaDevice,
    input: &Tensor,
    output_rows: usize,
) -> Tensor {
    let mut dims = input.dims().to_vec();
    *dims.last_mut().unwrap() = output_rows;
    Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(buf, dev)),
        dims,
        BackpropOp::none(),
        false,
    )
}

fn matvec_xs<T: CudaDType + DeviceRepr>(
    dev: &CudaDevice,
    input: &Tensor,
    packed: &Tensor,
    output_rows: usize,
    cols: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    let input = input.contiguous()?;
    let rows = input.elem_count() / cols;
    let n_blocks = rows * cols / 32;
    let (nr, config) = launch_config(rows, output_rows);

    // int8 activations followed by one f32 scale per 32 of them; `rows * cols`
    // is a multiple of 256, so the scales stay 4-byte aligned.
    let scratch = unsafe { dev.alloc::<u8>(rows * cols + 4 * n_blocks) }?;
    let xq = scratch.slice(..rows * cols);
    let xd = scratch.slice(rows * cols..);
    {
        let (input_storage, input_layout) = input.storage_and_layout();
        let Storage::Cuda(input_cuda) = &*input_storage else {
            bail!("i-quant matvec input must be on CUDA")
        };
        let offset = input_layout.start_offset();
        let quantize = match input.dtype() {
            DType::F32 => "iq4_quantize_q8_f32",
            DType::F16 => "iq4_quantize_q8_f16",
            DType::BF16 => "iq4_quantize_q8_bf16",
            other => bail!("i-quant kernels do not support {other:?} activations"),
        };
        let func = dev.get_or_load_custom_func(quantize, MODULE_NAME, ptx::QUANT_IQ4)?;
        let n_blocks_i = n_blocks as i32;
        let (x_f32, x_f16, x_bf16);
        let mut builder = func.builder();
        match input.dtype() {
            DType::F32 => {
                x_f32 = input_cuda.as_cuda_slice::<f32>()?.slice(offset..);
                builder.arg(&x_f32);
            },
            DType::F16 => {
                x_f16 = input_cuda.as_cuda_slice::<half::f16>()?.slice(offset..);
                builder.arg(&x_f16);
            },
            _ => {
                x_bf16 = input_cuda.as_cuda_slice::<half::bf16>()?.slice(offset..);
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
    }

    let kernel = match (nr, out_dtype) {
        (1, DType::F32) => "iq4_xs_matvec_q8_1_f32",
        (1, DType::F16) => "iq4_xs_matvec_q8_1_f16",
        (1, _) => "iq4_xs_matvec_q8_1_bf16",
        (_, DType::F32) => "iq4_xs_matvec_q8_4_f32",
        (_, DType::F16) => "iq4_xs_matvec_q8_4_f16",
        _ => "iq4_xs_matvec_q8_4_bf16",
    };
    let (packed_storage, packed_offset) = packed_slice(packed)?;
    let Storage::Cuda(packed_cuda) = &*packed_storage else {
        unreachable!()
    };
    let packed_slice = packed_cuda.as_cuda_slice::<u8>()?.slice(packed_offset..);
    let output_buf = unsafe { dev.alloc::<T>(rows * output_rows) }?;
    let func = dev.get_or_load_custom_func(kernel, MODULE_NAME, ptx::QUANT_IQ4)?;
    let rows_i = rows as i32;
    let output_rows_i = output_rows as i32;
    let cols_i = cols as i32;
    let mut builder = func.builder();
    builder.arg(&packed_slice);
    builder.arg(&xq);
    builder.arg(&xd);
    builder.arg(&output_buf);
    builder.arg(&rows_i);
    builder.arg(&output_rows_i);
    builder.arg(&cols_i);
    unsafe { builder.launch(config) }.w()?;
    drop(packed_storage);
    Ok(output_tensor(output_buf, dev.clone(), &input, output_rows))
}

/// Expand weight rows `row_start..row_start + n_rows` of a packed
/// `[_, cols]` tensor into a dense `[n_rows, cols]` tensor of `dtype`.
pub fn dequantize(
    packed: &Tensor,
    ty: IQuantType,
    row_start: usize,
    n_rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<Tensor> {
    if !matches!(ty, IQuantType::Iq4Nl | IQuantType::Iq4Xs) {
        let row_bytes = cols / ty.block_size() * ty.block_bytes();
        return launch_dequant(
            PackedSource::Tensor(packed, row_start * row_bytes),
            ty,
            None,
            1,
            n_rows,
            cols,
            dtype,
        )?
        .reshape((n_rows, cols));
    }
    let dev = packed.device().as_cuda_device()?.clone();
    match dtype {
        DType::F32 => dequantize_as::<f32>(&dev, packed, ty, row_start, n_rows, cols, "f32"),
        DType::F16 => dequantize_as::<half::f16>(&dev, packed, ty, row_start, n_rows, cols, "f16"),
        DType::BF16 => {
            dequantize_as::<half::bf16>(&dev, packed, ty, row_start, n_rows, cols, "bf16")
        },
        other => bail!("i-quant dequantize: unsupported output dtype {other:?}"),
    }
}

fn dequantize_as<T: CudaDType + DeviceRepr>(
    dev: &CudaDevice,
    packed: &Tensor,
    ty: IQuantType,
    row_start: usize,
    n_rows: usize,
    cols: usize,
    suffix: &str,
) -> Result<Tensor> {
    let row_bytes = cols / ty.block_size() * ty.block_bytes();
    let (packed_storage, packed_offset) = packed_slice(packed)?;
    let Storage::Cuda(packed_cuda) = &*packed_storage else {
        unreachable!()
    };
    let start = packed_offset + row_start * row_bytes;
    let packed_slice = packed_cuda
        .as_cuda_slice::<u8>()?
        .slice(start..start + n_rows * row_bytes);

    let output_buf = unsafe { dev.alloc::<T>(n_rows * cols) }?;
    // One warp per 256 weights: an IQ4_XS block or eight IQ4_NL blocks.
    let n_blocks = n_rows * cols / ty.block_size();
    let warps = n_rows * cols / 256 + usize::from(!(n_rows * cols).is_multiple_of(256));
    let kernel = match (ty, suffix) {
        (IQuantType::Iq4Xs, "f32") => "iq4_xs_dequant_f32",
        (IQuantType::Iq4Xs, "f16") => "iq4_xs_dequant_f16",
        (IQuantType::Iq4Xs, _) => "iq4_xs_dequant_bf16",
        (IQuantType::Iq4Nl, "f32") => "iq4_nl_dequant_f32",
        (IQuantType::Iq4Nl, "f16") => "iq4_nl_dequant_f16",
        (IQuantType::Iq4Nl, _) => "iq4_nl_dequant_bf16",
        (other, _) => bail!("no CUDA i-quant kernel for {}", other.name()),
    };
    let func = dev.get_or_load_custom_func(kernel, MODULE_NAME, ptx::QUANT_IQ4)?;
    let n_blocks_i = n_blocks as i32;
    let mut builder = func.builder();
    builder.arg(&packed_slice);
    builder.arg(&output_buf);
    builder.arg(&n_blocks_i);
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (warps.div_ceil(8) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    drop(packed_storage);

    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output_buf, dev.clone())),
        (n_rows, cols),
        BackpropOp::none(),
        false,
    ))
}

/// The type part of the generic kernel names (`IQ_ALL_TYPES` in `quant_iq.cu`).
fn type_tag(ty: IQuantType) -> &'static str {
    match ty {
        IQuantType::Iq4Nl => "iq4_nl",
        IQuantType::Iq4Xs => "iq4_xs",
        IQuantType::Iq2S => "iq2_s",
        IQuantType::Iq3Xxs => "iq3_xxs",
        IQuantType::Iq3S => "iq3_s",
        IQuantType::Q2_0 => "q2_0",
        IQuantType::Q4K => "q4_k",
        IQuantType::Q5K => "q5_k",
        IQuantType::Q6K => "q6_k",
    }
}

fn float_tag(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        other => bail!("i-quant kernels: unsupported dtype {other:?}"),
    }
}

fn to_i32(n: usize, what: &str) -> Result<i32> {
    i32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("{what} {n} exceeds i32")))
}

/// Flattened contiguous `U32` expert ids on a CUDA device.
fn ids_slice(ids: &Tensor) -> Result<(StorageRef<'_>, usize)> {
    let (storage, layout) = ids.storage_and_layout();
    if !matches!(&*storage, Storage::Cuda(_)) || ids.dtype() != DType::U32 {
        bail!("expert ids must be a CUDA U32 tensor")
    }
    Ok((storage, layout.start_offset()))
}

/// `MoE` matmul by expert id: for each of the `ids.len()` pairs `p`, row `p`
/// of the result is expert `ids[p]`'s `[output_rows, cols]` matrix (from the
/// packed `[experts, output_rows, cols]` tensor) times activation row
/// `p / x_div` of `input` (`[_, cols]`). Returns `[ids.len(), output_rows]`.
///
/// `ids` is never read on the host. Prefill-sized routings take
/// [`super::GemmPlan`] instead (see `IQuantExperts`).
///
/// # Errors
///
/// Returns an error if the tensors are not on a CUDA device, `ids` is not
/// `U32`, `out_dtype` isn't F32/F16/BF16, or the kernel launch fails.
#[allow(clippy::too_many_arguments)]
pub fn matvec_indexed(
    input: &Tensor,
    packed: &Tensor,
    ty: IQuantType,
    ids: &Tensor,
    x_div: usize,
    output_rows: usize,
    cols: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    if ids.dtype() != DType::U32 {
        bail!("expert ids must be U32, got {:?}", ids.dtype())
    }
    let ids = ids.flatten_all()?.contiguous()?;
    let pairs = ids.elem_count();
    launch_matvec(
        input,
        packed,
        ty,
        Some(&ids),
        x_div,
        pairs,
        output_rows,
        cols,
        out_dtype,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_matvec(
    input: &Tensor,
    packed: &Tensor,
    ty: IQuantType,
    ids: Option<&Tensor>,
    x_div: usize,
    pairs: usize,
    output_rows: usize,
    cols: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    match out_dtype {
        DType::F32 => launch_matvec_as::<f32>(
            input,
            packed,
            ty,
            ids,
            x_div,
            pairs,
            output_rows,
            cols,
            out_dtype,
        ),
        DType::F16 => launch_matvec_as::<half::f16>(
            input,
            packed,
            ty,
            ids,
            x_div,
            pairs,
            output_rows,
            cols,
            out_dtype,
        ),
        DType::BF16 => launch_matvec_as::<half::bf16>(
            input,
            packed,
            ty,
            ids,
            x_div,
            pairs,
            output_rows,
            cols,
            out_dtype,
        ),
        other => bail!("i-quant matvec: unsupported output dtype {other:?}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_matvec_as<T: CudaDType + DeviceRepr>(
    input: &Tensor,
    packed: &Tensor,
    ty: IQuantType,
    ids: Option<&Tensor>,
    x_div: usize,
    pairs: usize,
    output_rows: usize,
    cols: usize,
    out_dtype: DType,
) -> Result<Tensor> {
    let dev = input.device().as_cuda_device()?.clone();
    let name = format!("matvec_{}_{}", type_tag(ty), float_tag(out_dtype)?);
    let func = dev.get_or_load_custom_func(&name, GENERIC_MODULE, ptx::QUANT_IQ)?;
    let expert_stride = (output_rows * (cols / ty.block_size()) * ty.block_bytes()) as u64;
    let x_div_i = to_i32(x_div.max(1), "x_div")?;
    let out_rows_i = to_i32(output_rows, "matvec output rows")?;
    let cols_i = to_i32(cols, "matvec cols")?;
    let has_ids = i32::from(ids.is_some());

    // Activations are F32 in the kernel (matches the SYCL / Metal paths).
    let input = input.to_dtype(DType::F32)?.contiguous()?;
    let (input_storage, input_layout) = input.storage_and_layout();
    let Storage::Cuda(input_cuda) = &*input_storage else {
        bail!("i-quant matvec input must be on CUDA")
    };
    let input_slice = input_cuda
        .as_cuda_slice::<f32>()?
        .slice(input_layout.start_offset()..);
    let (packed_storage, packed_offset) = packed_slice(packed)?;
    let Storage::Cuda(packed_cuda) = &*packed_storage else {
        unreachable!()
    };
    let packed_slice = packed_cuda.as_cuda_slice::<u8>()?.slice(packed_offset..);
    let ids_guard = ids.map(ids_slice).transpose()?;
    // Unread without ids; any valid u32 buffer will do.
    let dummy = if ids.is_none() {
        Some(dev.alloc_zeros::<u32>(1)?)
    } else {
        None
    };
    let ids_view = match (&ids_guard, &dummy) {
        (Some((storage, offset)), _) => {
            let Storage::Cuda(cuda) = &**storage else {
                unreachable!()
            };
            cuda.as_cuda_slice::<u32>()?.slice(*offset..)
        },
        (None, Some(d)) => d.slice(..),
        (None, None) => unreachable!(),
    };

    let config = LaunchConfig {
        grid_dim: (
            u32::try_from(pairs).map_err(|_| candle_core::Error::Msg("too many pairs".into()))?,
            output_rows.div_ceil(ROWS_PER_BLOCK) as u32,
            1,
        ),
        block_dim: (32 * ROWS_PER_BLOCK as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let output_buf = unsafe { dev.alloc::<T>(pairs * output_rows) }?;
    let mut builder = func.builder();
    builder.arg(&packed_slice);
    builder.arg(&ids_view);
    builder.arg(&input_slice);
    builder.arg(&output_buf);
    builder.arg(&expert_stride);
    builder.arg(&x_div_i);
    builder.arg(&out_rows_i);
    builder.arg(&cols_i);
    builder.arg(&has_ids);
    unsafe { builder.launch(config) }.w()?;
    drop(input_storage);
    drop(packed_storage);
    drop(ids_guard);
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output_buf, dev)),
        (pairs, output_rows),
        BackpropOp::none(),
        false,
    ))
}

/// Where [`launch_dequant`] reads the packed blocks.
#[derive(Clone, Copy)]
enum PackedSource<'a> {
    /// A `u8` tensor, from this byte offset.
    Tensor(&'a Tensor, usize),
    /// A raw device allocation on this device, e.g. a `QTensor`'s.
    Raw(&'a CudaDevice, u64),
}

/// Expand weight rows `row_start..row_start + n_rows` -- see [`dequantize`];
/// this is the type-generic kernel behind it, optionally over experts.
#[allow(clippy::too_many_arguments)]
fn launch_dequant(
    packed: PackedSource<'_>,
    ty: IQuantType,
    ids: Option<&Tensor>,
    n_mats: usize,
    n_rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<Tensor> {
    match dtype {
        DType::F32 => launch_dequant_as::<f32>(packed, ty, ids, n_mats, n_rows, cols, dtype),
        DType::F16 => launch_dequant_as::<half::f16>(packed, ty, ids, n_mats, n_rows, cols, dtype),
        DType::BF16 => {
            launch_dequant_as::<half::bf16>(packed, ty, ids, n_mats, n_rows, cols, dtype)
        },
        other => bail!("i-quant dequantize: unsupported output dtype {other:?}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_dequant_as<T: CudaDType + DeviceRepr>(
    packed: PackedSource<'_>,
    ty: IQuantType,
    ids: Option<&Tensor>,
    n_mats: usize,
    n_rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<Tensor> {
    let dev = match packed {
        PackedSource::Tensor(t, _) => t.device().as_cuda_device()?.clone(),
        PackedSource::Raw(dev, _) => dev.clone(),
    };
    let name = format!("dequant_{}_{}", type_tag(ty), float_tag(dtype)?);
    let func = dev.get_or_load_custom_func(&name, GENERIC_MODULE, ptx::QUANT_IQ)?;
    let expert_stride = (n_rows * (cols / ty.block_size()) * ty.block_bytes()) as u64;
    let n_mats_i = to_i32(n_mats, "dequantize matrices")?;
    let n_rows_i = to_i32(n_rows, "dequantize rows")?;
    let cols_i = to_i32(cols, "dequantize cols")?;
    let has_ids = i32::from(ids.is_some());

    let packed_guard = match packed {
        PackedSource::Tensor(t, byte_offset) => {
            let (storage, offset) = packed_slice(t)?;
            Some((storage, offset + byte_offset))
        },
        PackedSource::Raw(..) => None,
    };
    let packed_view = match &packed_guard {
        Some((storage, offset)) => {
            let Storage::Cuda(cuda) = &**storage else {
                unreachable!()
            };
            Some(cuda.as_cuda_slice::<u8>()?.slice(*offset..))
        },
        None => None,
    };
    let ids = ids.map(|t| t.flatten_all()?.contiguous()).transpose()?;
    let ids_guard = ids.as_ref().map(ids_slice).transpose()?;
    let dummy = if ids.is_none() {
        Some(dev.alloc_zeros::<u32>(1)?)
    } else {
        None
    };
    let ids_view = match (&ids_guard, &dummy) {
        (Some((storage, offset)), _) => {
            let Storage::Cuda(cuda) = &**storage else {
                unreachable!()
            };
            cuda.as_cuda_slice::<u32>()?.slice(*offset..)
        },
        (None, Some(d)) => d.slice(..),
        (None, None) => unreachable!(),
    };

    let chunks = cols / 32;
    let config = LaunchConfig {
        grid_dim: (
            u32::try_from(n_mats * n_rows)
                .map_err(|_| candle_core::Error::Msg("too many rows".into()))?,
            chunks.div_ceil(256) as u32,
            1,
        ),
        block_dim: (chunks.min(256) as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let output_buf = unsafe { dev.alloc::<T>(n_mats * n_rows * cols) }?;
    let mut builder = func.builder();
    match (&packed_view, &packed) {
        (Some(view), _) => builder.arg(view),
        (None, PackedSource::Raw(_, ptr)) => builder.arg(ptr),
        (None, PackedSource::Tensor(..)) => unreachable!(),
    };
    builder.arg(&ids_view);
    builder.arg(&output_buf);
    builder.arg(&expert_stride);
    builder.arg(&n_mats_i);
    builder.arg(&n_rows_i);
    builder.arg(&cols_i);
    builder.arg(&has_ids);
    unsafe { builder.launch(config) }.w()?;
    drop(packed_view);
    drop(packed_guard);
    drop(ids_guard);
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output_buf, dev)),
        (n_mats * n_rows, cols),
        BackpropOp::none(),
        false,
    ))
}

/// Decode experts `ids` (`U32`, on the device) of a packed `[experts, rows,
/// cols]` tensor into a dense `[ids.len(), rows, cols]` tensor of `dtype`,
/// in one launch.
///
/// # Errors
///
/// Returns an error if the tensors are not on a CUDA device, `dtype` isn't
/// F32/F16/BF16, or the kernel launch fails.
pub fn dequantize_experts(
    packed: &Tensor,
    ty: IQuantType,
    ids: &Tensor,
    rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<Tensor> {
    let n = ids.elem_count();
    launch_dequant(
        PackedSource::Tensor(packed, 0),
        ty,
        Some(ids),
        n,
        rows,
        cols,
        dtype,
    )?
    .reshape((n, rows, cols))
}

/// [`dequantize_experts`] reading the packed `[experts, rows, cols]` blocks
/// straight out of a `QTensor`'s device buffer (candle stores them
/// contiguously, padding only at the end), so `MoE` experts held for
/// `indexed_moe_forward` can take the batched-GEMM path without a second
/// copy.
///
/// # Errors
///
/// Returns an error if `packed` is not on `dev`, `dtype` isn't
/// F32/F16/BF16, or the kernel launch fails.
pub fn dequantize_qtensor_experts(
    dev: &CudaDevice,
    packed: &candle_core::quantized::QTensor,
    ty: IQuantType,
    ids: &Tensor,
    rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<Tensor> {
    if !packed
        .device()
        .same_device(&candle_core::Device::Cuda(dev.clone()))
    {
        bail!("packed experts are not on the activations' CUDA device")
    }
    let ptr = packed.device_ptr()? as u64;
    let n = ids.elem_count();
    launch_dequant(
        PackedSource::Raw(dev, ptr),
        ty,
        Some(ids),
        n,
        rows,
        cols,
        dtype,
    )?
    .reshape((n, rows, cols))
}
