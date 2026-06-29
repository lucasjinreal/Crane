//! CUDA launcher for the fused Gated Delta Net recurrence kernel.
//!
//! Collapses the per-timestep Candle op graph into a single kernel launch
//! (`kernels/gdn.cu`). Inputs must be contiguous f32 CUDA tensors in the
//! layouts documented on [`gdn_recurrence_cuda`]; `q` is expected pre-scaled
//! by `1/sqrt(K)` (the caller does this, matching the CPU reference).

use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::WrapErr;
use candle_core::op::BackpropOp;
use candle_core::{CudaStorage, Result, Storage, Tensor};

// PTX compiled from kernels/gdn.cu — embedded at build time.
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

const MODULE_NAME: &str = "crane_gdn";

/// Run the gated delta rule recurrence on CUDA.
///
/// Shapes: `q,k = [BH,S,K]`, `v = [BH,S,V]`, `g,beta = [BH,S]`,
/// `state = [BH,K,V]`. Returns `(y = [BH,S,V], state_out = [BH,K,V])`.
pub fn gdn_recurrence_cuda(
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
        candle_core::bail!("gdn cuda kernel supports head_k_dim <= 256, got {kdim}");
    }
    let dev = q.device().as_cuda_device()?.clone();

    // Hold the storage guards alive until after the launch; the slice views
    // borrow from them.
    let (q_s, q_l) = q.storage_and_layout();
    let q_sl = match &*q_s {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("gdn: q must be a cuda tensor"),
    };
    let (k_s, k_l) = k.storage_and_layout();
    let k_sl = match &*k_s {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("gdn: k must be a cuda tensor"),
    };
    let (v_s, v_l) = v.storage_and_layout();
    let v_sl = match &*v_s {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("gdn: v must be a cuda tensor"),
    };
    let (g_s, g_l) = g.storage_and_layout();
    let g_sl = match &*g_s {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("gdn: g must be a cuda tensor"),
    };
    let (beta_s, beta_l) = beta.storage_and_layout();
    let beta_sl = match &*beta_s {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("gdn: beta must be a cuda tensor"),
    };
    let (state_s, state_l) = state.storage_and_layout();
    let state_sl = match &*state_s {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("gdn: state must be a cuda tensor"),
    };

    // Offsets (tensors are passed contiguous; honor any start offset anyway).
    let q_v = q_sl.slice(q_l.start_offset()..);
    let k_v = k_sl.slice(k_l.start_offset()..);
    let v_v = v_sl.slice(v_l.start_offset()..);
    let g_v = g_sl.slice(g_l.start_offset()..);
    let beta_v = beta_sl.slice(beta_l.start_offset()..);
    let state_v = state_sl.slice(state_l.start_offset()..);

    let y_buf = unsafe { dev.alloc::<f32>(bh * s * vdim) }?;
    let state_out_buf = unsafe { dev.alloc::<f32>(bh * kdim * vdim) }?;

    let func = dev.get_or_load_custom_func("gdn_recurrence_f32", MODULE_NAME, ptx::GDN)?;
    let cfg = LaunchConfig {
        grid_dim: (bh as u32, 1, 1),
        block_dim: (vdim as u32, 1, 1),
        shared_mem_bytes: (2 * kdim * std::mem::size_of::<f32>()) as u32,
    };

    let (bh_i, s_i, k_i, v_i) = (bh as i32, s as i32, kdim as i32, vdim as i32);
    let mut builder = func.builder();
    builder.arg(&q_v);
    builder.arg(&k_v);
    builder.arg(&v_v);
    builder.arg(&g_v);
    builder.arg(&beta_v);
    builder.arg(&state_v);
    builder.arg(&state_out_buf);
    builder.arg(&y_buf);
    builder.arg(&bh_i);
    builder.arg(&s_i);
    builder.arg(&k_i);
    builder.arg(&v_i);
    unsafe { builder.launch(cfg) }.w()?;

    let y = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(y_buf, dev.clone())),
        (bh, s, vdim),
        BackpropOp::none(),
        false,
    );
    let state_out = Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(state_out_buf, dev.clone())),
        (bh, kdim, vdim),
        BackpropOp::none(),
        false,
    );
    Ok((y, state_out))
}
