//! Float32 transcendental helpers with explicit rounding boundaries.
use candle_core::{DType, Result, Tensor};
mod scan;
pub use scan::cumsum_last_dim;
#[cfg(feature = "cuda")]
#[allow(dead_code)]
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

/// `base ** exponent`, using powf directly rather than exp(log(base)*x).
/// The distinction matters when the result is subsequently rounded to BF16.
pub fn scalar_power(base: f32, exponent: &Tensor) -> Result<Tensor> {
    if !base.is_finite() || base <= 0. {
        candle_core::bail!("scalar_power requires a positive finite base");
    }
    if exponent.dtype() != DType::F32 {
        candle_core::bail!("scalar_power requires F32 exponents");
    }
    #[cfg(feature = "cuda")]
    if exponent.device().is_cuda() {
        return cuda_power(base, exponent);
    }
    if exponent.device().is_cpu() {
        let values = exponent
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|x| base.powf(x))
            .collect::<Vec<_>>();
        return Tensor::from_vec(values, exponent.shape().clone(), exponent.device());
    }
    exponent.affine((base as f64).ln(), 0.)?.exp()
}
#[cfg(feature = "cuda")]
fn cuda_power(base: f32, exponent: &Tensor) -> Result<Tensor> {
    use candle_core::cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    };
    use candle_core::{CudaStorage, Storage, op::BackpropOp};
    let exponent = exponent.contiguous()?;
    let dev = exponent.device().as_cuda_device()?.clone();
    let (storage, layout) = exponent.storage_and_layout();
    let slice = match &*storage {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => unreachable!(),
    };
    let input = slice.slice(layout.start_offset()..);
    let n = u32::try_from(exponent.elem_count()).map_err(candle_core::Error::wrap)?;
    if n == 0 {
        return Ok(exponent.clone());
    }
    let out = unsafe { dev.alloc::<f32>(n as usize) }?;
    let function =
        dev.get_or_load_custom_func("scalar_power_f32", "crane_float_math", ptx::FLOAT_MATH)?;
    let mut builder = function.builder();
    builder.arg(&input);
    builder.arg(&out);
    builder.arg(&n);
    builder.arg(&base);
    // SAFETY: input/output hold n contiguous F32 elements on this stream.
    unsafe { builder.launch(LaunchConfig::for_num_elems(n)) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)),
        exponent.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    fn check(device: &Device) -> Result<()> {
        let x = Tensor::new(&[[0f32, 1.], [0.5, -1.]], device)?;
        assert_eq!(
            scalar_power(4., &x)?.to_vec2::<f32>()?,
            vec![vec![1., 4.], vec![2., 0.25]]
        );
        assert!(scalar_power(-4., &x).is_err());
        Ok(())
    }
    #[test]
    fn cpu_known_powers() -> Result<()> {
        check(&Device::Cpu)
    }
    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_known_powers() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        check(&device)
    }
}

/// F32 RMS normalization with explicit square/reduction/multiply rounding.
/// A missing alpha requests unit-weight normalization.
pub fn rms_norm(x: &Tensor, alpha: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let width = x.dim(candle_core::D::Minus1)?;
    if x.dtype() != DType::F32
        || alpha.is_some_and(|a| {
            a.dtype() != DType::F32
                || a.dims() != [width]
                || a.device().location() != x.device().location()
        })
    {
        candle_core::bail!("ordered RMS norm requires F32 inputs and a width-sized F32 alpha");
    }
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() && width >= 128 && width % 128 == 0 {
        return cuda_rms_norm(x, alpha, eps);
    }
    let alpha = match alpha {
        Some(a) => a.clone(),
        None => Tensor::ones(width, DType::F32, x.device())?,
    };
    candle_nn::ops::rms_norm(&x.contiguous()?, &alpha, eps)
}
#[cfg(feature = "cuda")]
fn cuda_rms_norm(x: &Tensor, alpha: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    use candle_core::cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    };
    use candle_core::{CudaStorage, Storage, op::BackpropOp};
    let x = x.contiguous()?;
    let width = u32::try_from(x.dim(candle_core::D::Minus1)?).map_err(candle_core::Error::wrap)?;
    let _count = u32::try_from(x.elem_count()).map_err(candle_core::Error::wrap)?;
    let rows = u32::try_from(x.elem_count() / width as usize).map_err(candle_core::Error::wrap)?;
    if rows == 0 {
        return Ok(x.clone());
    }
    let dev = x.device().as_cuda_device()?.clone();
    let (storage, layout) = x.storage_and_layout();
    let slice = match &*storage {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => unreachable!(),
    };
    let input = slice.slice(layout.start_offset()..);
    let alpha = alpha.map(Tensor::contiguous).transpose()?;
    let a_storage = alpha.as_ref().map(Tensor::storage_and_layout);
    let a_view = match &a_storage {
        Some((storage, layout)) => match &**storage {
            Storage::Cuda(c) => Some(c.as_cuda_slice::<f32>()?.slice(layout.start_offset()..)),
            _ => candle_core::bail!("ordered RMS norm alpha must be on CUDA"),
        },
        None => None,
    };
    let out = unsafe { dev.alloc::<f32>(x.elem_count()) }?;
    let function =
        dev.get_or_load_custom_func("rms_norm_f32_ordered", "crane_float_math", ptx::FLOAT_MATH)?;
    let mut builder = function.builder();
    builder.arg(&input);
    let null = 0u64;
    match &a_view {
        Some(view) => {
            builder.arg(view);
        },
        None => {
            builder.arg(&null);
        },
    }
    builder.arg(&out);
    builder.arg(&rows);
    builder.arg(&width);
    builder.arg(&eps);
    // SAFETY: one warp owns each contiguous row; all tensor guards remain
    // alive through launch and alpha is either null or width F32 values.
    unsafe { builder.launch(LaunchConfig::for_num_elems(rows * 32)) }.w()?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)),
        x.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}
