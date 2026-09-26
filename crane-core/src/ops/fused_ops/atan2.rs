// SPDX-License-Identifier: MIT
//! Crane Added 20260807: fused Atan2, reusable across callers (currently the
//! ONNX evaluator's `Atan2` op).
//!
//! `candle_core` has no native `.atan2()` tensor method, so this is
//! implemented via `CustomOp2` operating directly on tensor storage.
//! `cpu_fwd` is always compiled; `cuda_fwd` is gated behind the `cuda`
//! feature and dispatches to the kernel compiled from
//! `kernels/cuda/atan2.cu`, following the `snake` op's pattern in this same
//! module. `rocm_fwd` is gated behind the `rocm` feature and runs the
//! *same* `.cu` source through `hipcc` at runtime (see [`crate::ops::rocm`]).
//! Upstream candle has an open PR adding `atan`/`atan2`
//! (<https://github.com/huggingface/candle/pull/3338>); once that ships in a
//! released version this crate upgrades to, `cpu_fwd` can be replaced with a
//! direct tensor method call. Callers broadcast `y`/`x` to matching shapes
//! before calling `atan2()`.

#[cfg(feature = "cuda")]
use candle_core::DType;
#[cfg(feature = "cuda")]
use candle_core::backend::BackendStorage;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
use candle_core::rocm_backend::RocmStorage;
use candle_core::{CpuStorage, CustomOp2, Layout, Result, Shape, Tensor, WithDType};

// PTX compiled from kernels/cuda/atan2.cu — embedded at build time.
#[cfg(feature = "cuda")]
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

#[cfg(feature = "cuda")]
const MODULE_NAME: &str = "crane_atan2";

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
const ROCM_MODULE_NAME: &str = "crane_atan2";
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
const ROCM_SOURCE: &str = include_str!("../../../kernels/cuda/atan2.cu");

/// Element-wise `atan2(y, x)`. IEEE 754 compliant: `atan2(0, 0) = 0`,
/// handling the zero-magnitude case without a special branch.
struct Atan2Op;

impl CustomOp2 for Atan2Op {
    fn name(&self) -> &'static str {
        "atan2"
    }

    fn cpu_fwd(
        &self,
        s_y: &CpuStorage,
        l_y: &Layout,
        s_x: &CpuStorage,
        l_x: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: WithDType>(
            y: &[T],
            l_y: &Layout,
            x: &[T],
            l_x: &Layout,
        ) -> (CpuStorage, Shape) {
            let dst = candle_core::cpu_backend::binary_map(l_y, l_x, y, x, |a, b| {
                T::from_f64(a.to_f64().atan2(b.to_f64()))
            });
            (T::to_cpu_storage_owned(dst), l_y.shape().clone())
        }

        if l_y.shape() != l_x.shape() {
            candle_core::bail!("atan2: y and x must have the same shape");
        }

        match (s_y, s_x) {
            (CpuStorage::BF16(y), CpuStorage::BF16(x)) => Ok(inner(y, l_y, x, l_x)),
            (CpuStorage::F16(y), CpuStorage::F16(x)) => Ok(inner(y, l_y, x, l_x)),
            (CpuStorage::F32(y), CpuStorage::F32(x)) => Ok(inner(y, l_y, x, l_x)),
            (CpuStorage::F64(y), CpuStorage::F64(x)) => Ok(inner(y, l_y, x, l_x)),
            _ => candle_core::bail!("unsupported or mismatched dtypes for Atan2"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_y: &CudaStorage,
        l_y: &Layout,
        s_x: &CudaStorage,
        l_x: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = s_y.device();
        let n = l_y.shape().elem_count();

        let (yo1, yo2) = l_y
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("atan2: y must be contiguous".into()))?;
        let (xo1, xo2) = l_x
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("atan2: x must be contiguous".into()))?;
        if yo2 - yo1 != n || xo2 - xo1 != n {
            candle_core::bail!("atan2: y and x must have the same element count");
        }

        let fn_name = match s_y.dtype() {
            DType::BF16 => "atan2_bf16",
            DType::F16 => "atan2_f16",
            DType::F32 => "atan2_f32",
            dt => candle_core::bail!("atan2: unsupported dtype {dt:?}"),
        };
        let func = dev.get_or_load_custom_func(fn_name, MODULE_NAME, ptx::ATAN2)?;

        let n_u32 = n as u32;
        let block_size = 256u32;
        let grid_size = n_u32.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let slice = match (&s_y.slice, &s_x.slice) {
            (CudaStorageSlice::BF16(y), CudaStorageSlice::BF16(x)) => {
                let y = y.slice(yo1..yo2);
                let x = x.slice(xo1..xo2);
                let dst = unsafe { dev.alloc::<half::bf16>(n)? };
                let mut builder = func.builder();
                builder.arg(&y);
                builder.arg(&x);
                builder.arg(&dst);
                builder.arg(&n_u32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::BF16(dst)
            },
            (CudaStorageSlice::F16(y), CudaStorageSlice::F16(x)) => {
                let y = y.slice(yo1..yo2);
                let x = x.slice(xo1..xo2);
                let dst = unsafe { dev.alloc::<half::f16>(n)? };
                let mut builder = func.builder();
                builder.arg(&y);
                builder.arg(&x);
                builder.arg(&dst);
                builder.arg(&n_u32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F16(dst)
            },
            (CudaStorageSlice::F32(y), CudaStorageSlice::F32(x)) => {
                let y = y.slice(yo1..yo2);
                let x = x.slice(xo1..xo2);
                let dst = unsafe { dev.alloc::<f32>(n)? };
                let mut builder = func.builder();
                builder.arg(&y);
                builder.arg(&x);
                builder.arg(&dst);
                builder.arg(&n_u32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F32(dst)
            },
            _ => candle_core::bail!("atan2: unsupported or mismatched CUDA storage types"),
        };

        let dst = CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l_y.shape().clone()))
    }

    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    fn rocm_fwd(
        &self,
        s_y: &RocmStorage,
        l_y: &Layout,
        s_x: &RocmStorage,
        l_x: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        // SAFETY: atan2_{bf16,f16,f32} in ROCM_SOURCE take (const T*, const
        // T*, T*, uint32_t) for dtype T, matching binary_elementwise_fwd's
        // contract.
        unsafe {
            crate::ops::rocm::binary_elementwise_fwd(
                s_y,
                l_y,
                s_x,
                l_x,
                ROCM_MODULE_NAME,
                "atan2",
                ROCM_SOURCE,
            )
        }
    }
}

/// Fused `Atan2`: element-wise `atan2(y, x)`.
///
/// Computes the two-argument arctangent directly rather than through the
/// decomposed `Div(y,x) → Atan → quadrant-correction Where` chain that ONNX
/// exporters emit for opsets without a native `Atan2` op. This avoids the
/// numerical instability that decomposition introduces near the origin
/// (where `Div(0,0) = NaN` and the quadrant-decision `Less(x, 0)` is
/// noise-sensitive). `y` and `x` must already have matching shapes
/// (broadcast by the caller); both are made contiguous here so the CUDA
/// kernel can index them as flat buffers.
///
/// # Errors
///
/// Returns an error if `y`/`x` have a dtype other than `BF16`/`F16`/`F32`/
/// `F64` (`cpu_fwd`), or — on CUDA — other than `BF16`/`F16`/`F32`, or if
/// either input's element count doesn't match after broadcasting.
pub fn atan2(y: &Tensor, x: &Tensor) -> Result<Tensor> {
    let y = y.contiguous()?;
    let x = x.contiguous()?;
    y.apply_op2_no_bwd(&x, &Atan2Op)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    use super::atan2;

    // Verifies all four quadrants of atan2.
    #[test]
    fn atan2_all_quadrants() -> Result<()> {
        let y_vals = Tensor::new(&[1.0f32, 1.0, -1.0, -1.0], &Device::Cpu)?;
        let x_vals = Tensor::new(&[1.0f32, -1.0, -1.0, 1.0], &Device::Cpu)?;

        let result = atan2(&y_vals, &x_vals)?;

        let got = result.to_vec1::<f32>()?;
        let expected: Vec<f32> = vec![
            1.0f32.atan2(1.0),
            1.0f32.atan2(-1.0),
            (-1.0f32).atan2(-1.0),
            (-1.0f32).atan2(1.0),
        ];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < f32::EPSILON);
        }
        Ok(())
    }

    // IEEE 754: atan2(0, 0) = 0.
    #[test]
    fn atan2_zero_zero_is_zero() -> Result<()> {
        let y = Tensor::new(&[0.0f32], &Device::Cpu)?;
        let x = Tensor::new(&[0.0f32], &Device::Cpu)?;

        let result = atan2(&y, &x)?;

        assert_eq!(result.to_vec1::<f32>()?, vec![0.0]);
        Ok(())
    }

    // Verifies atan2 on the axes (y=0 or x=0).
    #[test]
    fn atan2_on_axes() -> Result<()> {
        let y_vals = Tensor::new(&[0.0f32, 1.0, 0.0, -1.0], &Device::Cpu)?;
        let x_vals = Tensor::new(&[1.0f32, 0.0, -1.0, 0.0], &Device::Cpu)?;

        let result = atan2(&y_vals, &x_vals)?;

        let got = result.to_vec1::<f32>()?;
        let expected: Vec<f32> = vec![
            0.0f32.atan2(1.0),
            1.0f32.atan2(0.0),
            0.0f32.atan2(-1.0),
            (-1.0f32).atan2(0.0),
        ];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < f32::EPSILON);
        }
        Ok(())
    }
}
