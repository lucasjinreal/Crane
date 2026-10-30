//! CUDA vector-four Welford LayerNorm with explicit rounding order.
#[cfg(feature = "cuda")]
use candle_core::DType;
use candle_core::{Module, Result, Tensor};

pub fn forward(norm: &candle_nn::LayerNorm, x: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() && matches!(x.dtype(), DType::F32 | DType::BF16) {
        let width = x.dim(candle_core::D::Minus1)?;
        if let Some(beta) = norm.bias() {
            if width > 0
                && width % 4 == 0
                && norm.weight().dims() == [width]
                && beta.dims() == [width]
                && norm.weight().device().location() == x.device().location()
                && beta.device().location() == x.device().location()
            {
                return cuda_forward(x, norm.weight(), beta, norm.eps() as f32);
            }
        }
    }
    norm.forward(x)
}

#[cfg(feature = "cuda")]
fn cuda_forward(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
    use candle_core::cuda_backend::{
        WrapErr,
        cudarc::driver::{LaunchConfig, PushKernelArg},
    };
    use candle_core::{CudaStorage, Storage, op::BackpropOp};
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?.contiguous()?;
    let gamma = gamma.to_dtype(DType::F32)?.contiguous()?;
    let beta = beta.to_dtype(DType::F32)?.contiguous()?;
    let width = u32::try_from(x.dim(candle_core::D::Minus1)?).map_err(candle_core::Error::wrap)?;
    let rows = u32::try_from(x.elem_count() / width as usize).map_err(candle_core::Error::wrap)?;
    if rows == 0 {
        return x.to_dtype(dtype);
    }
    let dev = x.device().as_cuda_device()?.clone();
    let views = [&x, &gamma, &beta].map(Tensor::storage_and_layout);
    let mut slices = Vec::new();
    for (storage, layout) in &views {
        match &**storage {
            Storage::Cuda(c) => {
                slices.push(c.as_cuda_slice::<f32>()?.slice(layout.start_offset()..))
            },
            _ => candle_core::bail!("Welford LayerNorm parameters must be on CUDA"),
        }
    }
    let out = unsafe { dev.alloc::<f32>(x.elem_count()) }?;
    let func = dev.get_or_load_custom_func(
        "layer_norm_f32_welford",
        "crane_float_math",
        super::ptx::FLOAT_MATH,
    )?;
    let mut builder = func.builder();
    builder
        .arg(&slices[0])
        .arg(&slices[1])
        .arg(&slices[2])
        .arg(&out)
        .arg(&width)
        .arg(&eps);
    // One block owns a contiguous row; input and parameter guards span launch.
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .w()?;
    Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)),
        x.shape().clone(),
        BackpropOp::none(),
        false,
    )
    .to_dtype(dtype)
}
