// SPDX-License-Identifier: MIT
//! Fused `SwiGLU` activation: `silu(gate) * up` in a single pass.
//!
//! Replaces the two-op `Silu.forward(&gate)? * up` chain with one kernel
//! launch, eliminating the intermediate tensor that `Silu` allocates.
//! `cpu_fwd` is always compiled; `cuda_fwd` is gated behind the `cuda`
//! feature and dispatches to the kernel compiled from
//! `kernels/cuda/swiglu.cu`, following the [`super::snake`] pattern.
//! `rocm_fwd` is gated behind the `rocm` feature and runs the *same*
//! `.cu` source through `hipcc` at runtime (see [`crate::ops::rocm`]).

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

#[cfg(feature = "cuda")]
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

#[cfg(feature = "cuda")]
const MODULE_NAME: &str = "crane_swiglu";

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
const ROCM_MODULE_NAME: &str = "crane_swiglu";
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
const ROCM_SOURCE: &str = include_str!("../../../kernels/cuda/swiglu.cu");

/// Fused `SwiGLU`: `silu(gate) * up`.
struct SwigluOp;

impl CustomOp2 for SwigluOp {
    fn name(&self) -> &'static str {
        "swiglu"
    }

    fn cpu_fwd(
        &self,
        s_gate: &CpuStorage,
        l_gate: &Layout,
        s_up: &CpuStorage,
        l_up: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: WithDType>(
            gate: &[T],
            l_gate: &Layout,
            up: &[T],
            l_up: &Layout,
        ) -> (CpuStorage, Shape) {
            let dst =
                candle_core::cpu_backend::binary_map(l_gate, l_up, gate, up, |g_val, u_val| {
                    let g = g_val.to_f64();
                    let u = u_val.to_f64();
                    T::from_f64((g / (1.0 + (-g).exp())) * u)
                });
            (T::to_cpu_storage_owned(dst), l_gate.shape().clone())
        }

        match (s_gate, s_up) {
            (CpuStorage::BF16(gate), CpuStorage::BF16(up)) => Ok(inner(gate, l_gate, up, l_up)),
            (CpuStorage::F16(gate), CpuStorage::F16(up)) => Ok(inner(gate, l_gate, up, l_up)),
            (CpuStorage::F32(gate), CpuStorage::F32(up)) => Ok(inner(gate, l_gate, up, l_up)),
            (CpuStorage::F64(gate), CpuStorage::F64(up)) => Ok(inner(gate, l_gate, up, l_up)),
            _ => candle_core::bail!("swiglu: unsupported or mismatched dtypes"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_gate: &CudaStorage,
        l_gate: &Layout,
        s_up: &CudaStorage,
        l_up: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = s_gate.device();
        let n = l_gate.shape().elem_count();

        let (go1, go2) = l_gate
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("swiglu: gate must be contiguous".into()))?;
        let (uo1, uo2) = l_up
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("swiglu: up must be contiguous".into()))?;
        if go2 - go1 != n || uo2 - uo1 != n {
            candle_core::bail!("swiglu: gate and up must have the same element count");
        }

        let fn_name = match s_gate.dtype() {
            DType::BF16 => "swiglu_bf16",
            DType::F16 => "swiglu_f16",
            DType::F32 => "swiglu_f32",
            dt => candle_core::bail!("swiglu: unsupported dtype {dt:?}"),
        };
        let func = dev.get_or_load_custom_func(fn_name, MODULE_NAME, ptx::SWIGLU)?;

        let n_u32 = n as u32;
        let block_size = 256u32;
        let grid_size = n_u32.div_ceil(block_size);
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let slice = match (&s_gate.slice, &s_up.slice) {
            (CudaStorageSlice::BF16(gate), CudaStorageSlice::BF16(up)) => {
                let gate = gate.slice(go1..go2);
                let up = up.slice(uo1..uo2);
                let dst = unsafe { dev.alloc::<half::bf16>(n)? };
                let mut builder = func.builder();
                builder.arg(&gate);
                builder.arg(&up);
                builder.arg(&dst);
                builder.arg(&n_u32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::BF16(dst)
            },
            (CudaStorageSlice::F16(gate), CudaStorageSlice::F16(up)) => {
                let gate = gate.slice(go1..go2);
                let up = up.slice(uo1..uo2);
                let dst = unsafe { dev.alloc::<half::f16>(n)? };
                let mut builder = func.builder();
                builder.arg(&gate);
                builder.arg(&up);
                builder.arg(&dst);
                builder.arg(&n_u32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F16(dst)
            },
            (CudaStorageSlice::F32(gate), CudaStorageSlice::F32(up)) => {
                let gate = gate.slice(go1..go2);
                let up = up.slice(uo1..uo2);
                let dst = unsafe { dev.alloc::<f32>(n)? };
                let mut builder = func.builder();
                builder.arg(&gate);
                builder.arg(&up);
                builder.arg(&dst);
                builder.arg(&n_u32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F32(dst)
            },
            _ => candle_core::bail!("swiglu: unsupported or mismatched CUDA storage types"),
        };

        let dst = CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, l_gate.shape().clone()))
    }

    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    fn rocm_fwd(
        &self,
        s_gate: &RocmStorage,
        l_gate: &Layout,
        s_up: &RocmStorage,
        l_up: &Layout,
    ) -> Result<(RocmStorage, Shape)> {
        // SAFETY: swiglu_{bf16,f16,f32} in ROCM_SOURCE take (const T*, const
        // T*, T*, uint32_t) for dtype T, matching binary_elementwise_fwd's
        // contract.
        unsafe {
            crate::ops::rocm::binary_elementwise_fwd(
                s_gate,
                l_gate,
                s_up,
                l_up,
                ROCM_MODULE_NAME,
                "swiglu",
                ROCM_SOURCE,
            )
        }
    }
}

/// Fused `SwiGLU` activation: `silu(gate) * up`.
///
/// Computes `gate / (1 + exp(-gate)) * up` in a single pass, avoiding the
/// intermediate tensor that a separate `Silu` + `Mul` chain allocates.
/// `gate` and `up` must have the same shape. On CPU, `cpu_fwd` walks each
/// input's own strides via `binary_map`, so non-contiguous views (e.g. a
/// `narrow` of a shared `gate_up` tensor) are passed through as-is. On
/// CUDA/ROCm the kernel indexes flat buffers, so both inputs are made
/// contiguous first.
///
/// # Errors
///
/// Returns an error if `gate`/`up` have a dtype other than `BF16`/`F16`/
/// `F32`/`F64` (`cpu_fwd`), or on CUDA/ROCm other than `BF16`/`F16`/`F32`,
/// or if the shapes don't match.
pub fn swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if gate.shape() != up.shape() {
        candle_core::bail!("swiglu: gate and up must have the same shape");
    }
    if gate.device().is_cpu() {
        return gate.apply_op2_no_bwd(up, &SwigluOp);
    }
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    gate.apply_op2_no_bwd(&up, &SwigluOp)
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::swiglu;

    /// Naive reference: separate silu + mul.
    fn naive_swiglu(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        let activated = candle_nn::ops::silu(gate)?;
        activated.broadcast_mul(up)
    }

    // Verifies the fused kernel matches silu(gate) * up for 1-D f32 values.
    #[test]
    fn swiglu_matches_naive_1d_f32() -> Result<()> {
        let gate = Tensor::new(&[0.0f32, 1.0, -1.0, 2.5, -0.5], &Device::Cpu)?;
        let up = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0, 1.5], &Device::Cpu)?;

        let got = swiglu(&gate, &up)?.to_vec1::<f32>()?;
        let expected = naive_swiglu(&gate, &up)?.to_vec1::<f32>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
        Ok(())
    }

    // Verifies correctness on a 2-D shape (simulates a batch of tokens).
    #[test]
    fn swiglu_2d_shape() -> Result<()> {
        let gate = Tensor::new(&[[0.5f32, -0.3, 1.2], [2.0, -1.0, 0.0]], &Device::Cpu)?;
        let up = Tensor::new(&[[1.0f32, 0.8, -0.5], [0.3, 2.0, 1.0]], &Device::Cpu)?;

        let got = swiglu(&gate, &up)?.flatten_all()?.to_vec1::<f32>()?;
        let expected = naive_swiglu(&gate, &up)?.flatten_all()?.to_vec1::<f32>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
        Ok(())
    }

    // Verifies the BF16 dtype branch.
    #[test]
    fn swiglu_bf16() -> Result<()> {
        let device = Device::Cpu;
        let gate = Tensor::new(&[0.0f32, 1.0, -1.0, 2.5], &device)?.to_dtype(DType::BF16)?;
        let up = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?.to_dtype(DType::BF16)?;

        let got = swiglu(&gate, &up)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let expected = naive_swiglu(&gate, &up)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-2, "got {g}, expected {e}");
        }
        Ok(())
    }

    // Verifies the F16 dtype branch.
    #[test]
    fn swiglu_f16() -> Result<()> {
        let device = Device::Cpu;
        let gate = Tensor::new(&[0.0f32, 1.0, -1.0, 2.5], &device)?.to_dtype(DType::F16)?;
        let up = Tensor::new(&[1.0f32, 2.0, 0.5, 3.0], &device)?.to_dtype(DType::F16)?;

        let got = swiglu(&gate, &up)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let expected = naive_swiglu(&gate, &up)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-3, "got {g}, expected {e}");
        }
        Ok(())
    }

    // Verifies the F64 dtype branch.
    #[test]
    fn swiglu_f64() -> Result<()> {
        let gate = Tensor::new(&[0.0f64, 1.0, -1.0, 2.5], &Device::Cpu)?;
        let up = Tensor::new(&[1.0f64, 2.0, 0.5, 3.0], &Device::Cpu)?;

        let got = swiglu(&gate, &up)?.to_vec1::<f64>()?;
        let expected = naive_swiglu(&gate, &up)?.to_vec1::<f64>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < f64::EPSILON * 100.0,
                "got {g}, expected {e}"
            );
        }
        Ok(())
    }

    // Verifies that unsupported dtypes return an error.
    #[test]
    fn swiglu_unsupported_dtype_errors() -> Result<()> {
        let gate = Tensor::new(&[1u32, 2, 3], &Device::Cpu)?;
        let up = Tensor::new(&[1u32, 2, 3], &Device::Cpu)?;

        let err = swiglu(&gate, &up).expect_err("unsupported dtype must error");

        assert!(
            err.to_string().contains("unsupported"),
            "unexpected error message: {err}"
        );
        Ok(())
    }

    // Verifies mismatched shapes error even when element counts match.
    #[test]
    fn swiglu_shape_mismatch_errors() -> Result<()> {
        let gate = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu)?;
        let up = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu)?;

        let err = swiglu(&gate, &up).expect_err("mismatched shapes must error");

        assert!(
            err.to_string().contains("same shape"),
            "unexpected error message: {err}"
        );
        Ok(())
    }

    // Verifies silu(0) * up == 0 (gate=0 means sigmoid(0)=0.5, so silu(0)=0).
    #[test]
    fn swiglu_gate_zero() -> Result<()> {
        let gate = Tensor::new(&[0.0f32], &Device::Cpu)?;
        let up = Tensor::new(&[42.0f32], &Device::Cpu)?;

        let got = swiglu(&gate, &up)?.to_vec1::<f32>()?;

        assert!((got[0]).abs() < 1e-10);
        Ok(())
    }

    // Verifies NaN gate produces NaN output.
    #[test]
    fn swiglu_nan_passthrough() -> Result<()> {
        let gate = Tensor::new(&[f32::NAN, 1.0], &Device::Cpu)?;
        let up = Tensor::new(&[1.0f32, 1.0], &Device::Cpu)?;

        let got = swiglu(&gate, &up)?.to_vec1::<f32>()?;
        let expected = naive_swiglu(&gate, &up)?.to_vec1::<f32>()?;

        assert!(got[0].is_nan());
        assert!((got[1] - expected[1]).abs() < 1e-6);
        Ok(())
    }

    // Verifies non-contiguous inputs (e.g. narrow()'d halves of a shared
    // gate_up tensor, as used by the MoE expert forward paths) produce the
    // same result as contiguous ones.
    #[test]
    fn swiglu_non_contiguous_narrow() -> Result<()> {
        let gate_up = Tensor::new(
            &[
                [0.5f32, -0.3, 1.2, 1.0, 0.8, -0.5],
                [2.0, -1.0, 0.0, 0.3, 2.0, 1.0],
            ],
            &Device::Cpu,
        )?;
        let gate = gate_up.narrow(1, 0, 3)?;
        let up = gate_up.narrow(1, 3, 3)?;
        assert!(!gate.is_contiguous());
        assert!(!up.is_contiguous());

        let got = swiglu(&gate, &up)?.flatten_all()?.to_vec1::<f32>()?;
        let expected = naive_swiglu(&gate, &up)?.flatten_all()?.to_vec1::<f32>()?;

        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
        Ok(())
    }
}
