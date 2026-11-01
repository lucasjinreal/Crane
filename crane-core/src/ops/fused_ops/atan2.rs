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
//! Metal (`--features metal`) dispatches `kernels/metal/atan2.metal` and
//! SYCL (`--features sycl`, F32/F16) `crane_atan2_sycl` in
//! `kernels/sycl/fused_ops.cpp` directly; any other device or dtype without
//! an implementation (SYCL BF16, or Metal without the feature) computes on
//! the CPU and copies back.
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

#[cfg(not(feature = "rocm"))]
use crate::utils::DeviceExt;

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
    let device = y.device();
    #[cfg(feature = "metal")]
    if device.is_metal() {
        return metal_kernel::atan2(&y, &x);
    }
    #[cfg(feature = "sycl")]
    if device.is_sycl() && matches!(y.dtype(), candle_core::DType::F32 | candle_core::DType::F16) {
        return sycl_kernel::atan2(&y, &x);
    }
    // `Atan2Op` runs on the CPU, CUDA and ROCm only, and candle has no
    // `atan`/`atan2` to compose a portable chain from: elsewhere, round-trip
    // through the CPU.
    if device.is_cpu() || device.is_cuda() || device.is_rocm() {
        return y.apply_op2_no_bwd(&x, &Atan2Op);
    }
    let cpu = candle_core::Device::Cpu;
    y.to_device(&cpu)?
        .apply_op2_no_bwd(&x.to_device(&cpu)?, &Atan2Op)?
        .to_device(device)
}

/// Launcher for `crane_atan2_sycl` (`kernels/sycl/fused_ops.cpp`).
#[cfg(feature = "sycl")]
mod sycl_kernel {
    use std::ffi::c_void;

    use candle_core::op::BackpropOp;
    use candle_core::{DType, Result, Storage, SyclStorage, Tensor};

    // libcrane_gdn_sycl.so — linked by build.rs when `--features sycl`.
    unsafe extern "C" {
        fn crane_atan2_sycl(
            queue: *mut c_void,
            dtype: i32,
            y: *const c_void,
            x: *const c_void,
            out: *mut c_void,
            n: usize,
        ) -> i32;
    }

    fn ptr(t: &Tensor) -> Result<*const c_void> {
        let (storage, layout) = t.storage_and_layout();
        match &*storage {
            Storage::Sycl(st) => Ok(unsafe {
                st.buf()
                    .as_ptr()
                    .cast::<u8>()
                    .add(layout.start_offset() * t.dtype().size_in_bytes())
                    .cast::<c_void>()
            }),
            _ => candle_core::bail!("atan2: expected a sycl tensor"),
        }
    }

    /// `y` and `x` contiguous, same shape, F32 or F16.
    pub fn atan2(y: &Tensor, x: &Tensor) -> Result<Tensor> {
        if y.shape() != x.shape() || y.dtype() != x.dtype() {
            candle_core::bail!(
                "atan2: y {:?} {:?} and x {:?} {:?} must match",
                y.dtype(),
                y.dims(),
                x.dtype(),
                x.dims()
            );
        }
        let dtype = y.dtype();
        let tag = i32::from(dtype == DType::F16);
        let dev = y.device().as_sycl_device()?.clone();
        let n = y.elem_count();
        let out = dev.alloc_bytes(n * dtype.size_in_bytes())?;
        let status = unsafe {
            crane_atan2_sycl(
                dev.queue().native_ptr(),
                tag,
                ptr(y)?,
                ptr(x)?,
                out.as_mut_ptr(),
                n,
            )
        };
        if status != 0 {
            candle_core::bail!("crane_atan2_sycl failed (status {status})");
        }
        Ok(Tensor::from_storage(
            Storage::Sycl(SyclStorage::from_buffer(&dev, out, dtype, n)),
            y.shape().clone(),
            BackpropOp::none(),
            false,
        ))
    }
}

/// Launcher for `atan2_{f32,f16,bf16}` (`kernels/metal/atan2.metal`).
#[cfg(feature = "metal")]
mod metal_kernel {
    use candle_core::{Result, Tensor};
    use candle_metal_kernels::metal::ComputeCommandEncoder;
    use objc2_metal::MTLSize;

    use crate::ops::metal_util;

    /// `y` and `x` contiguous, same shape and dtype.
    pub fn atan2(y: &Tensor, x: &Tensor) -> Result<Tensor> {
        if y.shape() != x.shape() || y.dtype() != x.dtype() {
            candle_core::bail!("atan2: y and x must have the same shape and dtype");
        }
        let dev = y.device().as_metal_device()?.clone();
        let dtype = y.dtype();
        let pipeline = metal_util::pipeline(
            &dev,
            "atan2",
            || include_str!("../../../kernels/metal/atan2.metal").to_string(),
            &format!("atan2_{}", metal_util::float_tag(dtype)?),
        )?;
        let n = y.elem_count();
        let n_u32 = u32::try_from(n)
            .map_err(|_| candle_core::Error::Msg(format!("atan2: {n} elements exceed u32")))?;
        let out = metal_util::output(&dev, n, dtype, "atan2")?;
        let (y_s, _) = y.storage_and_layout();
        let (x_s, _) = x.storage_and_layout();
        {
            let (y_buf, y_off) = metal_util::buffer(&y_s, y, 0, "y")?;
            let (x_buf, x_off) = metal_util::buffer(&x_s, x, 0, "x")?;
            let encoder = dev.command_encoder()?;
            let enc: &ComputeCommandEncoder = encoder.as_ref();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_input_buffer(0, Some(y_buf), y_off);
            enc.set_input_buffer(1, Some(x_buf), x_off);
            enc.set_output_buffer(2, Some(&out), 0);
            enc.set_bytes(3, &n_u32);
            enc.dispatch_threads(
                MTLSize {
                    width: n.max(1),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: n.clamp(1, 256),
                    height: 1,
                    depth: 1,
                },
            );
        }
        drop(y_s);
        drop(x_s);
        Ok(metal_util::wrap(&dev, out, y.shape().clone(), dtype))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    use super::atan2;

    /// Within one ulp at `e`'s magnitude. The reference comes from the
    /// platform's `atan2f`, which need not be correctly rounded (Apple's
    /// returns the float just below pi for `atan2(0, -1)`), while the op
    /// rounds an f64 result.
    fn within_ulp(g: f32, e: f32) -> bool {
        (g - e).abs() <= f32::EPSILON * e.abs().max(1.0)
    }

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
            assert!(within_ulp(*g, *e), "got {g}, expected {e}");
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

    /// The Metal kernel against the CPU op across dtypes, quadrants, the axes
    /// and signed zeros.
    #[cfg(feature = "metal")]
    #[test]
    fn atan2_metal_matches_cpu() -> Result<()> {
        use candle_core::DType;

        let Ok(metal) = Device::new_metal(0) else {
            return Ok(());
        };
        let mut ys = vec![0.0f32, -0.0, 0.0, -0.0, 1.0, -1.0, 0.0, 0.0, 1e-30, -3.5];
        let mut xs = vec![0.0f32, 0.0, -0.0, -0.0, 0.0, 0.0, 1.0, -1.0, -1e-30, 2.25];
        let ry = Tensor::randn(0f32, 3.0, 1000, &Device::Cpu)?.to_vec1::<f32>()?;
        let rx = Tensor::randn(0f32, 3.0, 1000, &Device::Cpu)?.to_vec1::<f32>()?;
        ys.extend(ry);
        xs.extend(rx);
        let n = ys.len();
        let y = Tensor::from_vec(ys, (2, n / 2), &Device::Cpu)?;
        let x = Tensor::from_vec(xs, (2, n / 2), &Device::Cpu)?;
        for (dtype, tol) in [
            (DType::F32, 2e-6f32),
            (DType::F16, 2e-3),
            (DType::BF16, 1e-2),
        ] {
            let (yd, xd) = (y.to_dtype(dtype)?, x.to_dtype(dtype)?);
            let want = atan2(&yd, &xd)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let got = atan2(&yd.to_device(&metal)?, &xd.to_device(&metal)?)?
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for (i, (g, e)) in got.iter().zip(&want).enumerate() {
                // Signed zeros and the +-pi branch must match exactly in sign.
                assert!(
                    (g - e).abs() <= tol * e.abs().max(1.0)
                        && g.is_sign_negative() == e.is_sign_negative(),
                    "{dtype:?} [{i}]: got {g}, expected {e}"
                );
            }
        }
        Ok(())
    }

    /// The SYCL kernel against the CPU op across quadrants, the axes and
    /// signed zeros (F32 and F16; BF16 takes the CPU round-trip).
    #[cfg(feature = "sycl")]
    #[test]
    fn atan2_sycl_matches_cpu() -> Result<()> {
        use candle_core::DType;

        if !candle_core::utils::sycl_is_available() {
            return Ok(());
        }
        let sycl = Device::new_sycl(0)?;
        let mut ys = vec![0.0f32, -0.0, 0.0, -0.0, 1.0, -1.0, 0.0, 0.0, 1e-30, -3.5];
        let mut xs = vec![0.0f32, 0.0, -0.0, -0.0, 0.0, 0.0, 1.0, -1.0, -1e-30, 2.25];
        ys.extend(Tensor::randn(0f32, 3.0, 1000, &Device::Cpu)?.to_vec1::<f32>()?);
        xs.extend(Tensor::randn(0f32, 3.0, 1000, &Device::Cpu)?.to_vec1::<f32>()?);
        let n = ys.len();
        let y = Tensor::from_vec(ys, (2, n / 2), &Device::Cpu)?;
        let x = Tensor::from_vec(xs, (2, n / 2), &Device::Cpu)?;
        for (dtype, tol) in [
            (DType::F32, 2e-6f32),
            (DType::F16, 2e-3),
            (DType::BF16, 1e-2),
        ] {
            let (yd, xd) = (y.to_dtype(dtype)?, x.to_dtype(dtype)?);
            let want = atan2(&yd, &xd)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let got = atan2(&yd.to_device(&sycl)?, &xd.to_device(&sycl)?)?;
            assert!(got.device().is_sycl());
            let got = got
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            for (i, (g, e)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - e).abs() <= tol * e.abs().max(1.0)
                        && g.is_sign_negative() == e.is_sign_negative(),
                    "{dtype:?} [{i}]: got {g}, expected {e}"
                );
            }
        }
        Ok(())
    }

    /// A device `Atan2Op` cannot run on (here Metal without crane's `metal`
    /// feature, which Apple Silicon builds still get from candle) falls back
    /// to the CPU instead of failing.
    #[cfg(not(feature = "metal"))]
    #[test]
    fn atan2_without_kernel_falls_back_to_cpu() -> Result<()> {
        let Ok(device) = Device::new_metal(0) else {
            return Ok(());
        };
        let y = Tensor::new(&[1.0f32, -1.0, 0.0], &device)?;
        let x = Tensor::new(&[-1.0f32, 0.0, -1.0], &device)?;
        let got = atan2(&y, &x)?;
        assert!(got.device().is_metal());
        let want = [1.0f32.atan2(-1.0), (-1.0f32).atan2(0.0), 0.0f32.atan2(-1.0)];
        for (g, e) in got.to_vec1::<f32>()?.into_iter().zip(want) {
            assert!(within_ulp(g, e), "got {g}, expected {e}");
        }
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
            assert!(within_ulp(*g, *e), "got {g}, expected {e}");
        }
        Ok(())
    }
}
