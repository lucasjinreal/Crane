//! SYCL implementations of the fused ops.
//!
//! The kernel is `kernels/sycl/fused_ops.cpp`, built by `crane-core`'s own
//! `build.rs` into `libcrane_gdn_sycl.so` alongside the GDN recurrence kernel
//! (see `ops/gdn/sycl_backend.rs` for the launch pattern this mirrors).
//!
//! Only `fused_silu_mul` has a native kernel so far; every other entry point
//! falls back to [`super::portable`] (or to the op chain it replaces) off a
//! non-SYCL device, matching `cuda_impl.rs` / `rocm_impl.rs`.

use std::ffi::c_void;

use candle_core::op::BackpropOp;
use candle_core::{DType, Result, Storage, SyclStorage, Tensor};

use super::portable;

pub use portable::{copy_from_slice_u32, copy_from_tensor_f32, gpu_argmax, topk_indices};

// libcrane_gdn_sycl.so — linked by build.rs when `--features sycl`.
unsafe extern "C" {
    fn crane_fused_silu_mul_sycl(
        queue: *mut c_void,
        dtype: i32,
        gate_up: *const c_void,
        out: *mut c_void,
        n_rows: i64,
        intermediate_size: i32,
    ) -> i32;
}

/// Fused `SiLU(gate) * up` where `gate_up`'s last dim is `[gate; up]`
/// concatenated (`2 * intermediate_size` wide).
///
/// Replaces the `narrow + narrow + silu + mul` op chain (2 real kernel
/// launches on SYCL, since `narrow` on the last dim is a free view) with one.
///
/// # Errors
///
/// Returns an error if the last dimension isn't `2 * intermediate_size`, if
/// the (SYCL) input isn't contiguous, if the dtype isn't `F32`/`F16`, or if
/// the kernel launch fails.
pub fn fused_silu_mul(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    let dims = gate_up.dims();
    let last = *dims
        .last()
        .ok_or_else(|| candle_core::Error::Msg("fused_silu_mul: scalar input".into()))?;
    if last != 2 * intermediate_size {
        candle_core::bail!(
            "fused_silu_mul: last dim {last} != 2*intermediate_size {}",
            2 * intermediate_size
        );
    }
    if !gate_up.device().is_sycl() {
        return silu_mul_ops(gate_up, intermediate_size);
    }
    if !gate_up.is_contiguous() {
        candle_core::bail!("fused_silu_mul: input must be contiguous");
    }

    let dtype = gate_up.dtype();
    let dtype_tag: i32 = match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        dt => candle_core::bail!("fused_silu_mul: unsupported dtype {dt:?} on SYCL"),
    };

    let dev = gate_up.device().as_sycl_device()?.clone();
    let queue = dev.queue().native_ptr();
    let n_rows = gate_up.elem_count() / last;
    let out_el = n_rows * intermediate_size;

    let (storage, layout) = gate_up.storage_and_layout();
    let src_ptr: *const c_void = match &*storage {
        Storage::Sycl(st) => unsafe {
            (st.buf().as_ptr() as *const u8).add(layout.start_offset() * dtype.size_in_bytes())
                as *const c_void
        },
        _ => candle_core::bail!("fused_silu_mul: input must be a sycl tensor"),
    };

    let out_buf = dev.alloc_bytes(out_el * dtype.size_in_bytes())?;

    let status = unsafe {
        crane_fused_silu_mul_sycl(
            queue,
            dtype_tag,
            src_ptr,
            out_buf.as_mut_ptr(),
            n_rows as i64,
            intermediate_size as i32,
        )
    };
    if status != 0 {
        candle_core::bail!("crane_fused_silu_mul_sycl failed (status {status})");
    }

    let mut out_dims = dims.to_vec();
    if let Some(d) = out_dims.last_mut() {
        *d = intermediate_size;
    }

    let storage = Storage::Sycl(SyclStorage::from_buffer(&dev, out_buf, dtype, out_el));
    Ok(Tensor::from_storage(
        storage,
        out_dims,
        BackpropOp::none(),
        false,
    ))
}

/// `narrow + narrow + silu + mul` off a SYCL device — e.g. a `sycl` build
/// still runs on CPU when no Intel GPU is present.
fn silu_mul_ops(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    use candle_core::D;
    let gate = gate_up.narrow(D::Minus1, 0, intermediate_size)?;
    let up = gate_up.narrow(D::Minus1, intermediate_size, intermediate_size)?;
    candle_nn::ops::silu(&gate)? * up
}
