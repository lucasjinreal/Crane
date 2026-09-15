//! SYCL launcher for the fused Gated Delta Net recurrence kernel.
//!
//! The counterpart of [`super::cuda_backend`] / [`super::rocm_backend`], against
//! `kernels/sycl/gdn.cpp` (built into `libcrane_gdn_sycl.so` by `build.rs` with
//! `icpx`). Collapses the per-timestep candle op graph into one submission onto
//! candle's in-order SYCL queue.
//!
//! Inputs must be contiguous f32 SYCL tensors in the layouts documented on
//! [`gdn_recurrence_sycl`]; `q` is expected pre-scaled by `1/sqrt(K)` (the
//! caller does this, matching the CPU reference). No explicit `synchronize`:
//! candle's SYCL queue is in-order, so the wrapped result tensors are correctly
//! ordered against every later op on the same queue.

use std::ffi::c_void;

use candle_core::op::BackpropOp;
use candle_core::{DType, Result, Storage, SyclStorage, Tensor};

// libcrane_gdn_sycl.so — linked by build.rs when `--features sycl`.
unsafe extern "C" {
    fn crane_gdn_recurrence_sycl(
        queue: *mut c_void,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state_in: *const f32,
        state_out: *mut f32,
        y: *mut f32,
        bh: i32,
        s: i32,
        kdim: i32,
        vdim: i32,
        v_tile: i32,
    ) -> i32;
}

/// Run the gated delta rule recurrence on SYCL.
///
/// Shapes: `q,k = [BH,S,K]`, `v = [BH,S,V]`, `g,beta = [BH,S]`,
/// `state = [BH,K,V]`. Returns `(y = [BH,S,V], state_out = [BH,K,V])`.
///
/// # Errors
///
/// Returns an error if `head_k_dim > 256` (the kernel's staging limit), if any
/// operand is not an f32 SYCL tensor, or if the submission fails.
pub fn gdn_recurrence_sycl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (bh, s, kdim) = q.dims3()?;
    let vdim = v.dim(2)?;
    if kdim > 256 {
        candle_core::bail!("gdn sycl kernel supports head_k_dim <= 256, got {kdim}");
    }

    let dev = q.device().as_sycl_device()?.clone();
    let queue = dev.queue().native_ptr();

    // The storage guards must outlive the submission — the raw pointers borrow
    // from them — so they are all bound here rather than inside a helper.
    let (q_s, q_l) = q.storage_and_layout();
    let (k_s, k_l) = k.storage_and_layout();
    let (v_s, v_l) = v.storage_and_layout();
    let (g_s, g_l) = g.storage_and_layout();
    let (beta_s, beta_l) = beta.storage_and_layout();
    let (state_s, state_l) = state.storage_and_layout();

    let ptr = |s: &Storage, offset: usize, name: &str| -> Result<*const f32> {
        match s {
            Storage::Sycl(st) => Ok(unsafe { (st.buf().as_ptr() as *const f32).add(offset) }),
            _ => candle_core::bail!("gdn: {name} must be a sycl tensor"),
        }
    };
    let q_ptr = ptr(&q_s, q_l.start_offset(), "q")?;
    let k_ptr = ptr(&k_s, k_l.start_offset(), "k")?;
    let v_ptr = ptr(&v_s, v_l.start_offset(), "v")?;
    let g_ptr = ptr(&g_s, g_l.start_offset(), "g")?;
    let beta_ptr = ptr(&beta_s, beta_l.start_offset(), "beta")?;
    let state_ptr = ptr(&state_s, state_l.start_offset(), "state")?;

    let y_elems = bh * s * vdim;
    let state_elems = bh * kdim * vdim;
    let y_buf = dev.alloc_bytes(y_elems * std::mem::size_of::<f32>())?;
    let state_out_buf = dev.alloc_bytes(state_elems * std::mem::size_of::<f32>())?;

    let status = unsafe {
        crane_gdn_recurrence_sycl(
            queue,
            q_ptr,
            k_ptr,
            v_ptr,
            g_ptr,
            beta_ptr,
            state_ptr,
            state_out_buf.as_mut_ptr() as *mut f32,
            y_buf.as_mut_ptr() as *mut f32,
            bh as i32,
            s as i32,
            kdim as i32,
            vdim as i32,
            vdim as i32, // V_TILE == V: one work-group per (batch*head)
        )
    };
    if status != 0 {
        candle_core::bail!("crane_gdn_recurrence_sycl failed (status {status})");
    }

    let y = Tensor::from_storage(
        Storage::Sycl(SyclStorage::from_buffer(&dev, y_buf, DType::F32, y_elems)),
        (bh, s, vdim),
        BackpropOp::none(),
        false,
    );
    let state_out = Tensor::from_storage(
        Storage::Sycl(SyclStorage::from_buffer(
            &dev,
            state_out_buf,
            DType::F32,
            state_elems,
        )),
        (bh, kdim, vdim),
        BackpropOp::none(),
        false,
    );
    Ok((y, state_out))
}
