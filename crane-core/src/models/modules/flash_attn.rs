//! Shared CPU flash-attention dispatch helper.
//!
//! `candle_nn::attention::flash_attn` is generic over the element type and
//! must be monomorphised per dtype at the call site; this centralizes that
//! dispatch so callers just pass a `Tensor`.

use candle_core::{DType, Result, Tensor};
use candle_nn::attention::{flash_attn, AttnMask};

/// Dispatch `flash_attn` to the monomorphised instantiation matching `q`'s
/// dtype.
///
/// CPU only: candle's `flash_attn` kernels read `Storage::Cpu` directly and
/// error on any other device, so callers must gate on `q.device().is_cpu()`
/// and fall back to a matmul-based SDPA on GPU.
///
/// # Errors
///
/// Returns an error if `q`'s dtype is not `F32`, `F16`, or `BF16`, or if the
/// underlying `flash_attn` kernel fails (e.g. non-CPU storage, shape
/// mismatch).
pub fn dispatch_flash_attn(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, mask: AttnMask) -> Result<Tensor> {
    match q.dtype() {
        DType::F32 => flash_attn::<f32>(q, k, v, scale, mask, None, None),
        DType::F16 => flash_attn::<half::f16>(q, k, v, scale, mask, None, None),
        DType::BF16 => flash_attn::<half::bf16>(q, k, v, scale, mask, None, None),
        dt => candle_core::bail!("flash_attn: unsupported dtype {dt:?}"),
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    fn bshd(b: usize, s: usize, h: usize, d: usize, dtype: DType, device: &Device) -> Tensor {
        Tensor::randn(0f32, 1f32, (b, s, h, d), device)
            .expect("randn")
            .to_dtype(dtype)
            .expect("to_dtype")
    }

    #[test]
    fn dispatch_f32_produces_correct_shape() {
        let device = Device::Cpu;
        let (b, s, h, d) = (1, 5, 2, 4);
        let q = bshd(b, s, h, d, DType::F32, &device);
        let k = bshd(b, s, h, d, DType::F32, &device);
        let v = bshd(b, s, h, d, DType::F32, &device);

        let out = dispatch_flash_attn(&q, &k, &v, 0.5, AttnMask::None).expect("dispatch");
        assert_eq!(out.dims(), &[b, h, s, d]);
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    fn dispatch_f16_produces_correct_shape() {
        let device = Device::Cpu;
        let (b, s, h, d) = (1, 5, 2, 4);
        let q = bshd(b, s, h, d, DType::F16, &device);
        let k = bshd(b, s, h, d, DType::F16, &device);
        let v = bshd(b, s, h, d, DType::F16, &device);

        // candle's CPU flash_attn kernels always accumulate in f32 and
        // return an f32 tensor, regardless of the input dtype.
        let out = dispatch_flash_attn(&q, &k, &v, 0.5, AttnMask::None).expect("dispatch");
        assert_eq!(out.dims(), &[b, h, s, d]);
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    fn dispatch_bf16_produces_correct_shape() {
        let device = Device::Cpu;
        let (b, s, h, d) = (1, 5, 2, 4);
        let q = bshd(b, s, h, d, DType::BF16, &device);
        let k = bshd(b, s, h, d, DType::BF16, &device);
        let v = bshd(b, s, h, d, DType::BF16, &device);

        // candle's CPU flash_attn kernels always accumulate in f32 and
        // return an f32 tensor, regardless of the input dtype.
        let out = dispatch_flash_attn(&q, &k, &v, 0.5, AttnMask::None).expect("dispatch");
        assert_eq!(out.dims(), &[b, h, s, d]);
        assert_eq!(out.dtype(), DType::F32);
    }

    #[test]
    fn dispatch_unsupported_dtype_errors() {
        let device = Device::Cpu;
        let (b, s, h, d) = (1, 5, 2, 4);
        let q = bshd(b, s, h, d, DType::U32, &device);
        let k = bshd(b, s, h, d, DType::U32, &device);
        let v = bshd(b, s, h, d, DType::U32, &device);

        let err = dispatch_flash_attn(&q, &k, &v, 0.5, AttnMask::None)
            .expect_err("unsupported dtype must error");
        assert!(
            err.to_string().contains("unsupported dtype"),
            "unexpected error message: {err}"
        );
    }
}
