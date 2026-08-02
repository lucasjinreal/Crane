//! ROCm/HIP launcher for the fused Gated Delta Net recurrence kernel.
//!
//! The counterpart of [`super::cuda_backend`], against the same
//! `kernels/cuda/gdn.cu`: candle compiles it with `hipcc` on first use and caches
//! the code object on disk, so nothing is built at `cargo build` time.
//!
//! This is what keeps hybrid (GDN) models off the portable path, where the
//! recurrence costs ~10 candle ops *per timestep per layer* — 1024 prompt
//! tokens across 18 linear-attention layers is ~184 000 launches, and the
//! measured cost was 5.47 ms per prompt token. Here it is one launch per layer
//! per forward pass. Set `CRANE_GDN_PORTABLE=1` to force the op-by-op path back
//! on and cross-check numerics on the same device.
//!
//! Inputs must be contiguous f32 ROCm tensors in the layouts documented on
//! [`gdn_recurrence_rocm`]; `q` is expected pre-scaled by `1/sqrt(K)` (the
//! caller does this, matching the CPU reference).

use candle_core::{DType, Result, Tensor};

use crate::ops::rocm::{self, arg};

const MODULE_NAME: &str = "crane_gdn";
const SOURCE: &str = include_str!("../../../kernels/cuda/gdn.cu");

/// Run the gated delta rule recurrence on ROCm.
///
/// Shapes: `q,k = [BH,S,K]`, `v = [BH,S,V]`, `g,beta = [BH,S]`,
/// `state = [BH,K,V]`. Returns `(y = [BH,S,V], state_out = [BH,K,V])`.
///
/// # Errors
///
/// Returns an error if `head_k_dim > 256` (the kernel's staging limit), if any
/// operand is not a contiguous f32 ROCm tensor, or if the launch fails.
pub fn gdn_recurrence_rocm(
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
        candle_core::bail!("gdn rocm kernel supports head_k_dim <= 256, got {kdim}");
    }
    let dev = q.device().as_rocm_device()?.clone();

    // The storage guards must outlive the launch — the pointers borrow from
    // them — so they are all bound here rather than inside a helper.
    let (q_s, q_l) = q.storage_and_layout();
    let (k_s, k_l) = k.storage_and_layout();
    let (v_s, v_l) = v.storage_and_layout();
    let (g_s, g_l) = g.storage_and_layout();
    let (beta_s, beta_l) = beta.storage_and_layout();
    let (state_s, state_l) = state.storage_and_layout();
    let q_p = rocm::device_ptr(&q_s, q_l, DType::F32, "gdn q")?;
    let k_p = rocm::device_ptr(&k_s, k_l, DType::F32, "gdn k")?;
    let v_p = rocm::device_ptr(&v_s, v_l, DType::F32, "gdn v")?;
    let g_p = rocm::device_ptr(&g_s, g_l, DType::F32, "gdn g")?;
    let beta_p = rocm::device_ptr(&beta_s, beta_l, DType::F32, "gdn beta")?;
    let state_p = rocm::device_ptr(&state_s, state_l, DType::F32, "gdn state")?;

    let y_buf = dev.alloc::<f32>(bh * s * vdim)?;
    let state_out_buf = dev.alloc::<f32>(bh * kdim * vdim)?;
    let y_p = y_buf.as_ptr();
    let state_out_p = state_out_buf.as_ptr();

    // One block per (head, V-tile). V columns are independent recurrences, so
    // they can be tiled, but one block per head (V_TILE == V) keeps the whole
    // state column resident and hides the serial inner loop best; the knob is
    // kept for shapes where that block would be too wide.
    let v_tile = std::env::var("CRANE_GDN_VTILE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|t| *t > 0)
        .unwrap_or(vdim)
        .min(vdim);
    let tiles = vdim.div_ceil(v_tile);
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let (grid, block, shared) = (
        (bh * tiles) as u32,
        v_tile as u32,
        (2 * kdim * size_of::<f32>()) as u32,
    );
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let (bh_i, s_i, k_i, v_i, vt_i) =
        (bh as i32, s as i32, kdim as i32, vdim as i32, v_tile as i32);

    // Specialized register-resident kernel for K == 128 (the state column lives
    // in registers); runtime-K fallback otherwise, which spills to scratch.
    let mut args = vec![
        arg(&q_p),
        arg(&k_p),
        arg(&v_p),
        arg(&g_p),
        arg(&beta_p),
        arg(&state_p),
        arg(&state_out_p),
        arg(&y_p),
        arg(&bh_i),
        arg(&s_i),
    ];
    let kernel = if kdim == 128 {
        "gdn_recurrence_f32_k128"
    } else {
        args.push(arg(&k_i));
        "gdn_recurrence_f32"
    };
    args.push(arg(&v_i));
    args.push(arg(&vt_i));

    // SAFETY: the argument list above matches the selected kernel's signature
    // in `kernels/cuda/gdn.cu`, the grid covers every (head, tile) pair the kernel
    // writes, and every operand outlives the launch.
    unsafe {
        rocm::launch(
            &dev,
            MODULE_NAME,
            kernel,
            SOURCE,
            grid,
            block,
            shared,
            &mut args,
        )?;
    }

    Ok((
        rocm::wrap_f32(y_buf, &dev, (bh, s, vdim)),
        rocm::wrap_f32(state_out_buf, &dev, (bh, kdim, vdim)),
    ))
}
