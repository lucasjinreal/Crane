//! Linear without bias, with an explicit nonbatched CUDA GEMM precision policy.
use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;
pub struct LinearNoBias {
    inner: candle_nn::Linear,
    tf32: bool,
}
impl LinearNoBias {
    pub fn load(input: usize, output: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            inner: candle_nn::linear_no_bias(input, output, vb)?,
            tf32: false,
        })
    }
    pub fn set_tf32(&mut self, enabled: bool) {
        self.tf32 = enabled;
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.tf32 && x.device().is_cuda() && x.dtype() == candle_core::DType::F32 {
            return self.forward_cuda(x);
        }
        self.inner.forward(x)
    }
    #[cfg(feature = "cuda")]
    fn forward_cuda(&self, x: &Tensor) -> Result<Tensor> {
        use candle_core::cuda_backend::cudarc::{
            cublas::{result, sys},
            driver::{DevicePtr, DevicePtrMut},
        };
        use candle_core::{CudaStorage, DType, Storage, op::BackpropOp};
        let w = self.inner.weight();
        let (outputs, inputs) = w.dims2()?;
        if x.dim(candle_core::D::Minus1)? != inputs
            || w.dtype() != DType::F32
            || !x.device().same_device(w.device())
        {
            candle_core::bail!(
                "CUDA unbiased GEMM requires matching F32 inputs and weights on one device"
            );
        }
        let x = x.contiguous()?;
        let w = w.contiguous()?;
        let rows = x.elem_count() / inputs;
        let m = i32::try_from(outputs).map_err(candle_core::Error::wrap)?;
        let n = i32::try_from(rows).map_err(candle_core::Error::wrap)?;
        let k = i32::try_from(inputs).map_err(candle_core::Error::wrap)?;
        let dev = x.device().as_cuda_device()?.clone();
        let stream = dev.cuda_stream();
        let (xs, xl) = x.storage_and_layout();
        let (ws, wl) = w.storage_and_layout();
        let Storage::Cuda(xs) = &*xs else {
            unreachable!()
        };
        let Storage::Cuda(ws) = &*ws else {
            unreachable!()
        };
        let xv = xs.as_cuda_slice::<f32>()?.slice(xl.start_offset()..);
        let wv = ws.as_cuda_slice::<f32>()?.slice(wl.start_offset()..);
        let mut out = unsafe { dev.alloc::<f32>(rows * outputs) }?;
        {
            let (xp, _x_guard) = xv.device_ptr(&stream);
            let (wp, _w_guard) = wv.device_ptr(&stream);
            let (op, _o_guard) = out.device_ptr_mut(&stream);
            let blas = dev.cublas_handle();
            // SAFETY: row-major W and X are represented as transposed column-major matrices.
            unsafe {
                result::gemm_ex(
                    *blas.handle(),
                    sys::cublasOperation_t::CUBLAS_OP_T,
                    sys::cublasOperation_t::CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    (&1f32 as *const f32).cast(),
                    wp as _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    k,
                    xp as _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    k,
                    (&0f32 as *const f32).cast(),
                    op as _,
                    sys::cudaDataType_t::CUDA_R_32F,
                    m,
                    sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
                    sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                )
            }
            .map_err(candle_core::Error::wrap)?;
        }
        let mut shape = x.dims().to_vec();
        *shape.last_mut().expect("nonempty linear shape") = outputs;
        Ok(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)),
            shape,
            BackpropOp::none(),
            false,
        ))
    }
}
