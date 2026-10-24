//! Full-patch 3D convolution through the public cuDNN descriptor API.
use candle_core::cuda_backend::cudarc::cudnn::{ConvForward, Cudnn, CudnnDataType, sys};
use candle_core::{DType, Result, Tensor};
/// BF16 NCTHW patches with one output voxel per complete convolution filter.
pub fn full_patch_conv3d(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    full_patch_conv3d_with_algorithm(x, weight, None)
}
/// Explicit algorithm selection is useful when comparing backend numerics.
pub fn full_patch_conv3d_with_algorithm(
    x: &Tensor,
    weight: &Tensor,
    algorithm: Option<sys::cudnnConvolutionFwdAlgo_t>,
) -> Result<Tensor> {
    match x.dtype() {
        DType::BF16 => forward_t::<half::bf16>(x, weight, algorithm),
        DType::F32 => forward_t::<f32>(x, weight, algorithm),
        _ => candle_core::bail!("patch conv3d requires BF16 or F32"),
    }
}
fn forward_t<
    T: candle_core::WithDType
        + candle_core::cuda_backend::CudaDType
        + CudnnDataType
        + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
>(
    x: &Tensor,
    weight: &Tensor,
    algorithm: Option<sys::cudnnConvolutionFwdAlgo_t>,
) -> Result<Tensor> {
    use candle_core::{CudaStorage, Storage, op::BackpropOp};
    let (n, c, t, h, w) = x.dims5()?;
    let (out, wc, wt, wh, ww) = weight.dims5()?;
    if !x.device().is_cuda()
        || x.dtype() != T::DTYPE
        || weight.dtype() != T::DTYPE
        || !x.device().same_device(weight.device())
        || (c, t, h, w) != (wc, wt, wh, ww)
        || [n, c, t, h, w, out].contains(&0)
    {
        candle_core::bail!(
            "patch conv3d requires CUDA BF16 NCTHW inputs and a full-patch OCTHW filter"
        );
    }
    let convert = |v: usize| i32::try_from(v).map_err(candle_core::Error::wrap);
    let xd = [n, c, t, h, w]
        .map(convert)
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let wd = [out, c, t, h, w]
        .map(convert)
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let strides = [c * t * h * w, t * h * w, h * w, w, 1]
        .map(convert)
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let yd = [convert(n)?, convert(out)?, 1, 1, 1];
    let ystr = [convert(out)?, 1, 1, 1, 1];
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let dev = x.device().as_cuda_device()?.clone();
    let dnn = Cudnn::new(dev.cuda_stream()).map_err(candle_core::Error::wrap)?;
    let xdesc = dnn
        .create_nd_tensor::<T>(&xd, &strides)
        .map_err(candle_core::Error::wrap)?;
    let wdesc = dnn
        .create_nd_filter::<T>(sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, &wd)
        .map_err(candle_core::Error::wrap)?;
    let ydesc = dnn
        .create_nd_tensor::<T>(&yd, &ystr)
        .map_err(candle_core::Error::wrap)?;
    let mut conv = dnn
        .create_convnd::<f32>(
            &[0, 0, 0],
            &xd[2..],
            &[1, 1, 1],
            sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )
        .map_err(candle_core::Error::wrap)?;
    conv.set_math_type(sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH)
        .map_err(candle_core::Error::wrap)?;
    let operation = ConvForward {
        conv: &conv,
        x: &xdesc,
        w: &wdesc,
        y: &ydesc,
    };
    let algorithm = match algorithm {
        Some(a) => a,
        None => operation
            .pick_algorithm()
            .map_err(candle_core::Error::wrap)?,
    };
    let workspace_size = operation
        .get_workspace_size(algorithm)
        .map_err(candle_core::Error::wrap)?;
    if workspace_size > 256 * 1024 * 1024 {
        candle_core::bail!("patch convolution workspace exceeds 256 MiB");
    }
    let mut workspace = unsafe { dev.alloc::<u8>(workspace_size.max(1)) }?;
    let mut output = unsafe { dev.alloc::<T>(n * out) }?;
    let (xs, xl) = x.storage_and_layout();
    let (ws, wl) = weight.storage_and_layout();
    let Storage::Cuda(xs) = &*xs else {
        unreachable!()
    };
    let Storage::Cuda(ws) = &*ws else {
        unreachable!()
    };
    let input = xs.as_cuda_slice::<T>()?.slice(xl.start_offset()..);
    let filter = ws.as_cuda_slice::<T>()?.slice(wl.start_offset()..);
    // SAFETY: the validated contiguous buffers match all cuDNN descriptors.
    unsafe {
        operation.launch(
            algorithm,
            Some(&mut workspace),
            (T::from_f64(1.), T::from_f64(0.)),
            &input,
            &filter,
            &mut output,
        )
    }
    .map_err(candle_core::Error::wrap)?;
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev)),
        (n, out),
        BackpropOp::none(),
        false,
    ))
}
