//! Launchers for the fused multi-token GDN stages in
//! `kernels/cuda/gdn_fused.cu` (bf16 activations, head dim 128, batch 1).
//!
//! Short passes through a GDN layer -- speculative-decoding verification
//! blocks, prefill chunks -- are dispatch-bound: the op-by-op conv / split /
//! norm / gates / layout chain is ~45 launches per layer. Three kernels replace
//! it. They reproduce the op chain's rounding points, so the layer output is
//! the same; `GatedDeltaNet::forward` falls back to the op chain whenever a
//! precondition ([`supported`]) does not hold.

use candle_core::cuda_backend::WrapErr;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::op::BackpropOp;
use candle_core::{CudaStorage, DType, Result, Storage, Tensor};
use half::bf16;

use super::config::{GdnDims, VHeadOrder};

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

const MODULE: &str = "crane_gdn_fused";
const HEAD_DIM: usize = 128;
const CONV_K: usize = 4;
/// Warps (value heads) per block in the prep / post kernels.
const WARPS: usize = 4;

/// Whether the fused path can run for this layer and input: CUDA, bf16, batch 1,
/// more than one token, 128-wide heads and a 4-tap conv.
pub fn supported(x: &Tensor, dims: &GdnDims) -> bool {
    x.device().is_cuda()
        && x.dtype() == DType::BF16
        && x.dim(0).is_ok_and(|b| b == 1)
        && x.dim(1).is_ok_and(|s| s > 1)
        && dims.head_k_dim == HEAD_DIM
        && dims.head_v_dim == HEAD_DIM
        && dims.conv_kernel_size == CONV_K
        && std::env::var("CRANE_GDN_FUSED").map_or(true, |v| v != "0")
}

/// Borrow a CUDA tensor's data as a device view starting at its layout offset.
macro_rules! cuda_view {
    ($name:ident, $guard:ident, $tensor:expr, $ty:ty) => {
        let ($guard, layout) = $tensor.storage_and_layout();
        let $name = match &*$guard {
            Storage::Cuda(c) => c
                .as_cuda_slice::<$ty>()
                .map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "gdn fused: `{}` is {:?}, expected {}: {e}",
                        stringify!($tensor),
                        $tensor.dtype(),
                        stringify!($ty)
                    ))
                })?
                .slice(layout.start_offset()..),
            _ => candle_core::bail!("gdn fused: expected a CUDA tensor"),
        };
    };
}

fn wrap<T: candle_core::cuda_backend::CudaDType>(
    buf: candle_core::cuda_backend::cudarc::driver::CudaSlice<T>,
    dev: &candle_core::CudaDevice,
    shape: impl Into<candle_core::Shape>,
) -> Tensor {
    Tensor::from_storage(
        Storage::Cuda(CudaStorage::wrap_cuda_slice(buf, dev.clone())),
        shape,
        BackpropOp::none(),
        false,
    )
}

/// Result of [`conv_silu`].
pub struct ConvOut {
    /// `[S, C]` bf16, silu already applied.
    pub y: Tensor,
    /// `[1, C, 4]` conv state after all `S` tokens.
    pub state: Tensor,
    /// State after the first `ckpt_at` tokens, if one was requested.
    pub ckpt: Option<Tensor>,
}

/// Causal conv1d + silu over `x: [S, C]` (contiguous bf16) continuing from
/// `state: [1, C, 4]` (contiguous bf16). `weight: [C, 4]` is f32, already
/// rounded through bf16 as the op-by-op path does.
///
/// # Errors
///
/// Returns an error if a tensor is not a contiguous CUDA tensor of the
/// expected dtype or the launch fails.
pub fn conv_silu(
    x: &Tensor,
    state: &Tensor,
    weight: &Tensor,
    ckpt_at: Option<usize>,
) -> Result<ConvOut> {
    let (s, c) = x.dims2()?;
    let dev = x.device().as_cuda_device()?.clone();
    cuda_view!(x_v, x_g, x, bf16);
    cuda_view!(state_v, state_g, state, bf16);
    cuda_view!(w_v, w_g, weight, f32);
    let y = unsafe { dev.alloc::<bf16>(s * c) }?;
    let new_state = unsafe { dev.alloc::<bf16>(c * CONV_K) }?;
    let want_ckpt = ckpt_at.filter(|&p| p > 0 && p < s);
    let ckpt_buf = match want_ckpt {
        Some(_) => Some(unsafe { dev.alloc::<bf16>(c * CONV_K) }?),
        None => None,
    };

    let func = dev.get_or_load_custom_func("gdn_conv_silu_bf16", MODULE, ptx::GDN_FUSED)?;
    let (s_i, c_i, xs_i, p_i) = (s as i32, c as i32, c as i32, want_ckpt.unwrap_or(0) as i32);
    let null_ckpt;
    let mut b = func.builder();
    b.arg(&x_v);
    b.arg(&state_v);
    b.arg(&w_v);
    b.arg(&y);
    b.arg(&new_state);
    match &ckpt_buf {
        Some(buf) => {
            b.arg(buf);
        },
        None => {
            null_ckpt = 0u64; // nullptr
            b.arg(&null_ckpt);
        },
    }
    b.arg(&s_i);
    b.arg(&c_i);
    b.arg(&xs_i);
    b.arg(&p_i);
    unsafe { b.launch(LaunchConfig::for_num_elems(c as u32)) }.w()?;

    Ok(ConvOut {
        y: wrap(y, &dev, (s, c)),
        state: wrap(new_state, &dev, (1, c, CONV_K)),
        ckpt: ckpt_buf.map(|buf| wrap(buf, &dev, (1, c, CONV_K))),
    })
}

/// Tensors for the recurrence kernel, covering one run of tokens.
pub struct Prepped {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub g: Tensor,
    pub beta: Tensor,
}

/// Token stride (elements between consecutive tokens) of a `[1, S, H]` tensor
/// whose head dim is packed, or `None` if it is not.
fn token_stride(t: &Tensor) -> Option<usize> {
    let layout = t.layout();
    (layout.dims().len() == 3 && layout.stride()[2] == 1).then(|| layout.stride()[1])
}

/// Split Q|K|V of `conv: [S, C]`, L2-normalise Q and K, scale Q, derive the
/// gates from `a`/`b` (`[1, S, H]` bf16, any token stride) and lay everything
/// out as the recurrence kernel reads it, for tokens `t0 .. t0 + len`.
///
/// # Errors
///
/// Returns an error if an input is not a CUDA tensor with packed trailing
/// dims, or the launch fails.
#[allow(clippy::too_many_arguments)]
pub fn prep(
    conv: &Tensor,
    a: &Tensor,
    b: &Tensor,
    neg_exp_a_log: &Tensor,
    dt_bias: &Tensor,
    dims: &GdnDims,
    t0: usize,
    len: usize,
    scale: f32,
) -> Result<Prepped> {
    let c = conv.dim(1)?;
    let h = dims.num_v_heads;
    let dev = conv.device().as_cuda_device()?.clone();
    let (Some(as_), Some(bs_)) = (token_stride(a), token_stride(b)) else {
        candle_core::bail!("gdn fused prep: a/b must have packed head dims");
    };
    cuda_view!(conv_v, conv_g, conv, bf16);
    cuda_view!(a_v, a_g, a, bf16);
    cuda_view!(b_v, b_g, b, bf16);
    cuda_view!(nea_v, nea_g, neg_exp_a_log, f32);
    cuda_view!(dt_v, dt_g, dt_bias, f32);
    let q3 = unsafe { dev.alloc::<f32>(h * len * HEAD_DIM) }?;
    let k3 = unsafe { dev.alloc::<f32>(h * len * HEAD_DIM) }?;
    let v3 = unsafe { dev.alloc::<f32>(h * len * HEAD_DIM) }?;
    let g2 = unsafe { dev.alloc::<f32>(h * len) }?;
    let beta2 = unsafe { dev.alloc::<f32>(h * len) }?;

    let func = dev.get_or_load_custom_func("gdn_prep_bf16", MODULE, ptx::GDN_FUSED)?;
    let order = i32::from(dims.v_head_order == VHeadOrder::Chunked);
    let ints = [
        t0 as i32,
        len as i32,
        h as i32,
        dims.num_k_heads as i32,
        dims.v_per_group as i32,
        order,
        dims.key_dim as i32,
        c as i32,
        as_ as i32,
        bs_ as i32,
    ];
    let mut bl = func.builder();
    bl.arg(&conv_v);
    bl.arg(&a_v);
    bl.arg(&b_v);
    bl.arg(&nea_v);
    bl.arg(&dt_v);
    bl.arg(&q3);
    bl.arg(&k3);
    bl.arg(&v3);
    bl.arg(&g2);
    bl.arg(&beta2);
    for i in &ints {
        bl.arg(i);
    }
    bl.arg(&scale);
    let cfg = LaunchConfig {
        grid_dim: (len as u32, h.div_ceil(WARPS) as u32, 1),
        block_dim: (32 * WARPS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { bl.launch(cfg) }.w()?;

    Ok(Prepped {
        q: wrap(q3, &dev, (h, len, HEAD_DIM)),
        k: wrap(k3, &dev, (h, len, HEAD_DIM)),
        v: wrap(v3, &dev, (h, len, HEAD_DIM)),
        g: wrap(g2, &dev, (h, len)),
        beta: wrap(beta2, &dev, (h, len)),
    })
}

/// Allocate the `[S, H*128]` bf16 output of [`post_norm`].
///
/// # Errors
///
/// Returns an error if the allocation fails.
pub fn alloc_post_out(
    like: &Tensor,
    seq_len: usize,
    h: usize,
) -> Result<candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>> {
    let dev = like.device().as_cuda_device()?;
    Ok(unsafe { dev.alloc::<bf16>(seq_len * h * HEAD_DIM) }?)
}

/// Recurrence output `y: [H, len, 128]` f32 for tokens `t0 .. t0 + len` ->
/// bf16 -> gated RMSNorm with `z: [1, S, H, 128]` -> rows of `out`.
///
/// # Errors
///
/// Returns an error if `z` is not packed per head or the launch fails.
#[allow(clippy::too_many_arguments)]
pub fn post_norm(
    out: &candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
    y: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f32,
    t0: usize,
    len: usize,
    h: usize,
) -> Result<()> {
    let dev = y.device().as_cuda_device()?.clone();
    let zs = z.layout().stride()[1];
    if z.layout().stride()[2] != HEAD_DIM || z.layout().stride()[3] != 1 {
        candle_core::bail!("gdn fused post: z must be packed per head");
    }
    cuda_view!(y_v, y_g, y, f32);
    cuda_view!(z_v, z_g, z, bf16);
    cuda_view!(w_v, w_g, weight, f32);
    let func = dev.get_or_load_custom_func("gdn_post_norm_bf16", MODULE, ptx::GDN_FUSED)?;
    let (t0_i, len_i, h_i, zs_i) = (t0 as i32, len as i32, h as i32, zs as i32);
    let mut b = func.builder();
    b.arg(&y_v);
    b.arg(&z_v);
    b.arg(&w_v);
    b.arg(out);
    b.arg(&t0_i);
    b.arg(&len_i);
    b.arg(&h_i);
    b.arg(&zs_i);
    b.arg(&eps);
    let cfg = LaunchConfig {
        grid_dim: (len as u32, h.div_ceil(WARPS) as u32, 1),
        block_dim: (32 * WARPS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { b.launch(cfg) }.w()?;
    Ok(())
}

/// Wrap the buffer filled by [`post_norm`] as a `[S, H*128]` tensor.
pub fn post_out_tensor(
    buf: candle_core::cuda_backend::cudarc::driver::CudaSlice<bf16>,
    like: &Tensor,
    seq_len: usize,
    h: usize,
) -> Result<Tensor> {
    let dev = like.device().as_cuda_device()?.clone();
    Ok(wrap(buf, &dev, (seq_len, h * HEAD_DIM)))
}
