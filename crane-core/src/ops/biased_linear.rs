//! Biased linear with a native cuBLASLt BF16 epilogue.
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::cublaslt::CudaBlasLT;
use candle_core::{DType, Module, Result, Tensor};
use candle_nn::VarBuilder;
#[cfg(feature = "cuda")]
use std::sync::Arc;

pub struct BiasedLinear {
    inner: candle_nn::Linear,
    #[cfg(feature = "cuda")]
    blas: Option<Arc<CudaBlasLT>>,
}
impl BiasedLinear {
    pub fn load(input: usize, output: usize, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::linear(input, output, vb.clone())?;
        #[cfg(feature = "cuda")]
        let blas = if vb.device().is_cuda() && matches!(vb.dtype(), DType::BF16 | DType::F32) {
            use std::{cell::RefCell, collections::HashMap, sync::Weak};
            thread_local! {
                static HANDLES: RefCell<HashMap<usize, Weak<CudaBlasLT>>> = RefCell::new(HashMap::new());
            }
            let stream = vb.device().as_cuda_device()?.cuda_stream();
            let key = Arc::as_ptr(&stream) as usize;
            Some(HANDLES.with(|handles| -> Result<_> {
                let mut handles = handles.borrow_mut();
                if let Some(handle) = handles.get(&key).and_then(Weak::upgrade) {
                    return Ok(handle);
                }
                let handle = Arc::new(CudaBlasLT::new(stream).map_err(candle_core::Error::wrap)?);
                handles.insert(key, Arc::downgrade(&handle));
                Ok(handle)
            })?)
        } else {
            None
        };
        Ok(Self {
            inner,
            #[cfg(feature = "cuda")]
            blas,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if xs.dim(candle_core::D::Minus1)? != self.inner.weight().dim(1)? {
            candle_core::bail!("biased linear input width differs from its weights");
        }
        if xs.device().location() != self.inner.weight().device().location() {
            candle_core::bail!("biased linear input and weights must share a device");
        }
        #[cfg(feature = "cuda")]
        if let Some(blas) = &self.blas {
            if self.inner.weight().dtype() == DType::BF16
                || candle_core::cuda_backend::gemm_reduced_precision_f32()
            {
                return match self.inner.weight().dtype() {
                    DType::BF16 => self.forward_cuda::<half::bf16>(xs, blas),
                    DType::F32 => self.forward_cuda::<f32>(xs, blas),
                    _ => unreachable!(),
                };
            }
        }
        self.inner.forward(xs)
    }
    #[cfg(feature = "cuda")]
    fn forward_cuda<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::WithDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        xs: &Tensor,
        blas: &CudaBlasLT,
    ) -> Result<Tensor> {
        use candle_core::{CudaStorage, Storage, op::BackpropOp};
        let xs = xs.to_dtype(T::DTYPE)?.contiguous()?;
        let (output, input) = self.inner.weight().dims2()?;
        let rows = xs.elem_count() / input;
        let dev = xs.device().as_cuda_device()?.clone();
        let (x_store, x_layout) = xs.storage_and_layout();
        let (w_store, w_layout) = self.inner.weight().storage_and_layout();
        let bias = self.inner.bias().expect("biased linear bias");
        let (b_store, b_layout) = bias.storage_and_layout();
        fn get<T: candle_core::cuda_backend::CudaDType>(
            storage: &Storage,
        ) -> Result<&candle_core::cuda_backend::cudarc::driver::CudaSlice<T>> {
            match storage {
                Storage::Cuda(c) => c.as_cuda_slice::<T>(),
                _ => candle_core::bail!("biased linear requires CUDA tensors"),
            }
        }
        let x = get::<T>(&x_store)?.slice(x_layout.start_offset()..);
        let w = get::<T>(&w_store)?.slice(w_layout.start_offset()..);
        let b = get::<T>(&b_store)?.slice(b_layout.start_offset()..);
        let mut out = unsafe { dev.alloc::<T>(rows * output) }?;
        cuda::matmul(
            blas,
            &w,
            &x,
            &b,
            &mut out,
            output,
            rows,
            input,
            T::DTYPE == DType::F32,
        )?;
        let mut shape = xs.dims().to_vec();
        *shape.last_mut().expect("linear input rank") = output;
        Ok(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev)),
            shape,
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    fn check(device: &Device, dtype: DType) -> Result<()> {
        let w = (1..=8)
            .flat_map(|r| std::iter::repeat_n(r as f32, 8))
            .collect::<Vec<_>>();
        let weights = std::collections::HashMap::from([
            (
                "weight".into(),
                Tensor::from_vec(w, (8, 8), device)?.to_dtype(dtype)?,
            ),
            (
                "bias".into(),
                Tensor::arange(0f32, 8f32, device)?.to_dtype(dtype)?,
            ),
        ]);
        let vb = VarBuilder::from_tensors(weights, dtype, device);
        let layer = BiasedLinear::load(8, 8, vb)?;
        let input = Tensor::from_vec([vec![1f32; 8], vec![-1f32; 8]].concat(), (2, 8), device)?
            .to_dtype(dtype)?;
        let result = layer
            .forward(&input)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        assert_eq!(result[0], vec![8., 17., 26., 35., 44., 53., 62., 71.]);
        assert_eq!(
            result[1],
            vec![-8., -15., -22., -29., -36., -43., -50., -57.]
        );
        Ok(())
    }
    #[test]
    fn cpu_bias_and_layout() -> Result<()> {
        check(&Device::Cpu, DType::F32)
    }
    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_bf16_bias_and_layout() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        check(&device, DType::BF16)
    }
}
