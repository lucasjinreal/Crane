// SPDX-License-Identifier: MIT

//! Metal launchers for `kernels/metal/gdn_fused.metal`: the gated delta rule
//! recurrence, the chunked path's triangular inverse, and the three stages
//! around the recurrence (conv + silu, Q/K/V prep, gated norm).
//!
//! The stage launchers mirror [`super::gdn_fused`]'s CUDA API, so
//! `GatedDeltaNet::forward_fused` drives either backend. Unlike CUDA, Metal
//! takes that path for every pass, a single decode token included: one
//! dispatch per stage is the whole point on a dispatch-bound decode, and it
//! gives a token the same GDN arithmetic whether it is decoded alone or inside
//! a speculative-decoding verification block.

use std::sync::Arc;

use candle_core::{DType, Result, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputeCommandEncoder, ComputePipeline};
use objc2_metal::MTLSize;

use super::config::{GdnDims, VHeadOrder};
use crate::ops::metal_util;

const LIBRARY: &str = "gdn_fused";
const HEAD_DIM: usize = 128;
const CONV_K: usize = 4;
/// `GDN_LANES` in the kernel: threads per state column in the recurrence.
const LANES: usize = 4;
/// `GDN_REC_COLS` in the kernel: state columns per recurrence threadgroup.
const REC_COLS: usize = 32;
/// `GDN_MAX_CHUNK` in the kernel.
pub const MAX_CHUNK: usize = 64;
/// SIMD-groups (value heads) per threadgroup in the prep / post kernels.
const GROUPS: usize = 4;

/// `gdn_rec_args` in the kernel.
#[repr(C)]
struct RecArgs {
    s: u32,
    v: u32,
    tiles: u32,
}

/// `gdn_conv_args` in the kernel.
#[repr(C)]
struct ConvArgs {
    s: u32,
    c: u32,
    xs: u32,
    ckpt_at: u32,
}

/// `gdn_prep_args` in the kernel.
#[repr(C)]
struct PrepArgs {
    t0: u32,
    len: u32,
    h: u32,
    num_k: u32,
    vpg: u32,
    order: u32,
    key_dim: u32,
    c: u32,
    as_: u32,
    bs_: u32,
    scale: f32,
}

/// `gdn_post_args` in the kernel.
#[repr(C)]
struct PostArgs {
    t0: u32,
    len: u32,
    h: u32,
    zs: u32,
    eps: f32,
}

fn pipeline(dev: &candle_core::MetalDevice, name: &str) -> Result<ComputePipeline> {
    metal_util::pipeline(
        dev,
        LIBRARY,
        || include_str!("../../../kernels/metal/gdn_fused.metal").to_string(),
        name,
    )
}

fn size(width: usize, height: usize) -> MTLSize {
    MTLSize {
        width,
        height,
        depth: 1,
    }
}

fn u32_of(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("gdn metal: {n} exceeds u32")))
}

/// Whether the fused path can run for this layer and input: Metal, a float
/// activation dtype, batch 1, 128-wide heads and a 4-tap conv.
pub fn supported(x: &Tensor, dims: &GdnDims) -> bool {
    x.device().is_metal()
        && matches!(x.dtype(), DType::F16 | DType::BF16 | DType::F32)
        && x.dim(0).is_ok_and(|b| b == 1)
        && dims.head_k_dim == HEAD_DIM
        && dims.head_v_dim == HEAD_DIM
        && dims.conv_kernel_size == CONV_K
        && std::env::var("CRANE_GDN_FUSED").map_or(true, |v| v != "0")
        && std::env::var("CRANE_GDN_PORTABLE").is_err()
}

/// Bind `$tensor`'s buffer (at its layout offset) as input `$idx` of `$enc`.
/// The storage guard lives in `$guard` until the end of the enclosing scope.
macro_rules! bind_input {
    ($enc:expr, $idx:expr, $guard:ident, $tensor:expr) => {
        let ($guard, _) = $tensor.storage_and_layout();
        let (buf, offset) = metal_util::buffer(&$guard, $tensor, 0, stringify!($tensor))?;
        $enc.set_input_buffer($idx, Some(buf), offset);
    };
}

// ─────────────────────────────────────────────────────────────────────
//  Recurrence
// ─────────────────────────────────────────────────────────────────────

/// The gated delta rule over a whole pass, on Metal.
///
/// Shapes (contiguous f32): `q,k = [BH,S,K]` (`q` pre-scaled by `1/sqrt(K)`),
/// `v = [BH,S,V]`, `g,beta = [BH,S]`, `state = [BH,K,V]`. Returns
/// `(y = [BH,S,V], state_out = [BH,K,V])`; `state` is left untouched.
/// `None` for head dims without a kernel (`K` other than 64 / 128).
///
/// # Errors
///
/// Returns an error if a tensor is not on Metal or the dispatch fails.
#[allow(clippy::many_single_char_names)]
pub fn gdn_recurrence_metal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    let (bh, s, kdim) = q.dims3()?;
    let vdim = v.dim(2)?;
    let name = match kdim {
        64 => "gdn_recurrence_f32_k64",
        128 => "gdn_recurrence_f32_k128",
        _ => return Ok(None),
    };
    let dev = q.device().as_metal_device()?.clone();
    let pipeline = pipeline(&dev, name)?;
    let (q, k, v, g, beta, state) = (
        q.contiguous()?,
        k.contiguous()?,
        v.contiguous()?,
        g.contiguous()?,
        beta.contiguous()?,
        state.contiguous()?,
    );
    let y = metal_util::output(&dev, bh * s * vdim, DType::F32, "gdn_y")?;
    let state_out = metal_util::output(&dev, bh * kdim * vdim, DType::F32, "gdn_state")?;
    let tiles = vdim.div_ceil(REC_COLS);
    let args = RecArgs {
        s: u32_of(s)?,
        v: u32_of(vdim)?,
        tiles: u32_of(tiles)?,
    };
    {
        let encoder = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = encoder.as_ref();
        enc.set_compute_pipeline_state(&pipeline);
        bind_input!(enc, 0, q_g, &q);
        bind_input!(enc, 1, k_g, &k);
        bind_input!(enc, 2, v_g, &v);
        bind_input!(enc, 3, g_g, &g);
        bind_input!(enc, 4, b_g, &beta);
        bind_input!(enc, 5, s_g, &state);
        enc.set_output_buffer(6, Some(&state_out), 0);
        enc.set_output_buffer(7, Some(&y), 0);
        enc.set_bytes(8, &args);
        enc.dispatch_thread_groups(size(bh * tiles, 1), size(REC_COLS * LANES, 1));
    }
    Ok(Some((
        metal_util::wrap(&dev, y, (bh, s, vdim), DType::F32),
        metal_util::wrap(&dev, state_out, (bh, kdim, vdim), DType::F32),
    )))
}

/// `(I + kb)^-1` for a strictly lower-triangular `kb: [B, cs, cs]` f32, as the
/// chunked recurrence needs it. `None` above [`MAX_CHUNK`].
///
/// # Errors
///
/// Returns an error if `kb` is not on Metal or the dispatch fails.
pub fn gdn_chunk_tri_inverse_metal(kb: &Tensor) -> Result<Option<Tensor>> {
    let (b, cs, _) = kb.dims3()?;
    if cs > MAX_CHUNK || kb.dtype() != DType::F32 {
        return Ok(None);
    }
    let dev = kb.device().as_metal_device()?.clone();
    let pipeline = pipeline(&dev, "gdn_tri_inverse_f32")?;
    let kb = kb.contiguous()?;
    let out = metal_util::output(&dev, b * cs * cs, DType::F32, "gdn_tri_inv")?;
    let cs_u = u32_of(cs)?;
    {
        let encoder = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = encoder.as_ref();
        enc.set_compute_pipeline_state(&pipeline);
        bind_input!(enc, 0, kb_g, &kb);
        enc.set_output_buffer(1, Some(&out), 0);
        enc.set_bytes(2, &cs_u);
        enc.dispatch_thread_groups(size(b, 1), size(cs, 1));
    }
    Ok(Some(metal_util::wrap(&dev, out, (b, cs, cs), DType::F32)))
}

// ─────────────────────────────────────────────────────────────────────
//  Stages around the recurrence
// ─────────────────────────────────────────────────────────────────────

/// Result of [`conv_silu`].
pub struct ConvOut {
    /// `[S, C]`, silu already applied.
    pub y: Tensor,
    /// `[1, C, 4]` conv state after all `S` tokens.
    pub state: Tensor,
    /// State after the first `ckpt_at` tokens, if one was requested.
    pub ckpt: Option<Tensor>,
}

/// Causal conv1d + silu over `x: [S, C]` continuing from `state: [1, C, 4]`
/// (both contiguous, in the activation dtype). `weight: [C, 4]` is f32,
/// already rounded through the activation dtype as the op-by-op path does.
///
/// # Errors
///
/// Returns an error if a tensor is not on Metal or the dispatch fails.
pub fn conv_silu(
    x: &Tensor,
    state: &Tensor,
    weight: &Tensor,
    ckpt_at: Option<usize>,
) -> Result<ConvOut> {
    let (s, c) = x.dims2()?;
    let dtype = x.dtype();
    let dev = x.device().as_metal_device()?.clone();
    let pipeline = pipeline(
        &dev,
        &format!("gdn_conv_silu_{}", metal_util::float_tag(dtype)?),
    )?;
    let y = metal_util::output(&dev, s * c, dtype, "gdn_conv_y")?;
    let new_state = metal_util::output(&dev, c * CONV_K, dtype, "gdn_conv_state")?;
    let want_ckpt = ckpt_at.filter(|&p| p > 0 && p < s);
    let ckpt = want_ckpt
        .map(|_| metal_util::output(&dev, c * CONV_K, dtype, "gdn_conv_ckpt"))
        .transpose()?;
    let args = ConvArgs {
        s: u32_of(s)?,
        c: u32_of(c)?,
        xs: u32_of(c)?,
        ckpt_at: u32_of(want_ckpt.unwrap_or(0))?,
    };
    let rows = s + CONV_K + if want_ckpt.is_some() { CONV_K } else { 0 };
    {
        let encoder = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = encoder.as_ref();
        enc.set_compute_pipeline_state(&pipeline);
        bind_input!(enc, 0, x_g, x);
        bind_input!(enc, 1, st_g, state);
        bind_input!(enc, 2, w_g, weight);
        enc.set_output_buffer(3, Some(&y), 0);
        enc.set_output_buffer(4, Some(&new_state), 0);
        // Never written without a checkpoint; any buffer satisfies the binding.
        enc.set_output_buffer(5, Some(ckpt.as_ref().unwrap_or(&new_state)), 0);
        enc.set_bytes(6, &args);
        enc.dispatch_threads(size(c, rows), size(256.min(c), 1));
    }
    Ok(ConvOut {
        y: metal_util::wrap(&dev, y, (s, c), dtype),
        state: metal_util::wrap(&dev, new_state, (1, c, CONV_K), dtype),
        ckpt: ckpt.map(|buf| metal_util::wrap(&dev, buf, (1, c, CONV_K), dtype)),
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

/// Token stride of a `[1, S, H]` tensor whose head dim is packed.
fn token_stride(t: &Tensor) -> Option<usize> {
    let layout = t.layout();
    (layout.dims().len() == 3 && layout.stride()[2] == 1).then(|| layout.stride()[1])
}

/// Split Q|K|V of `conv: [S, C]`, L2-normalise Q and K, scale Q, derive the
/// gates from `a`/`b` (`[1, S, H]`, any token stride) and lay everything out
/// as the recurrence kernel reads it, for tokens `t0 .. t0 + len`.
///
/// # Errors
///
/// Returns an error if an input is not on Metal with packed trailing dims, or
/// the dispatch fails.
#[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
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
    let dtype = conv.dtype();
    let dev = conv.device().as_metal_device()?.clone();
    let (Some(as_), Some(bs_)) = (token_stride(a), token_stride(b)) else {
        candle_core::bail!("gdn metal prep: a/b must have packed head dims");
    };
    if a.dtype() != dtype || b.dtype() != dtype {
        candle_core::bail!("gdn metal prep: a/b must be {dtype:?}");
    }
    let pipeline = pipeline(&dev, &format!("gdn_prep_{}", metal_util::float_tag(dtype)?))?;
    let n = h * len * HEAD_DIM;
    let q3 = metal_util::output(&dev, n, DType::F32, "gdn_q3")?;
    let k3 = metal_util::output(&dev, n, DType::F32, "gdn_k3")?;
    let v3 = metal_util::output(&dev, n, DType::F32, "gdn_v3")?;
    let g2 = metal_util::output(&dev, h * len, DType::F32, "gdn_g2")?;
    let beta2 = metal_util::output(&dev, h * len, DType::F32, "gdn_beta2")?;
    let args = PrepArgs {
        t0: u32_of(t0)?,
        len: u32_of(len)?,
        h: u32_of(h)?,
        num_k: u32_of(dims.num_k_heads)?,
        vpg: u32_of(dims.v_per_group)?,
        order: u32::from(dims.v_head_order == VHeadOrder::Chunked),
        key_dim: u32_of(dims.key_dim)?,
        c: u32_of(c)?,
        as_: u32_of(as_)?,
        bs_: u32_of(bs_)?,
        scale,
    };
    {
        let encoder = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = encoder.as_ref();
        enc.set_compute_pipeline_state(&pipeline);
        bind_input!(enc, 0, conv_g, conv);
        bind_input!(enc, 1, a_g, a);
        bind_input!(enc, 2, b_g, b);
        bind_input!(enc, 3, nea_g, neg_exp_a_log);
        bind_input!(enc, 4, dt_g, dt_bias);
        enc.set_output_buffer(5, Some(&q3), 0);
        enc.set_output_buffer(6, Some(&k3), 0);
        enc.set_output_buffer(7, Some(&v3), 0);
        enc.set_output_buffer(8, Some(&g2), 0);
        enc.set_output_buffer(9, Some(&beta2), 0);
        enc.set_bytes(10, &args);
        enc.dispatch_thread_groups(size(len, h.div_ceil(GROUPS)), size(32 * GROUPS, 1));
    }
    let wrap = |buf, shape: &[usize]| metal_util::wrap(&dev, buf, shape, DType::F32);
    Ok(Prepped {
        q: wrap(q3, &[h, len, HEAD_DIM]),
        k: wrap(k3, &[h, len, HEAD_DIM]),
        v: wrap(v3, &[h, len, HEAD_DIM]),
        g: wrap(g2, &[h, len]),
        beta: wrap(beta2, &[h, len]),
    })
}

/// Allocate the `[S, H*128]` output of [`post_norm`], in `like`'s dtype.
///
/// # Errors
///
/// Returns an error if the allocation fails.
pub fn alloc_post_out(like: &Tensor, seq_len: usize, h: usize) -> Result<Arc<Buffer>> {
    let dev = like.device().as_metal_device()?;
    metal_util::output(dev, seq_len * h * HEAD_DIM, like.dtype(), "gdn_post_out")
}

/// Recurrence output `y: [H, len, 128]` f32 for tokens `t0 .. t0 + len` ->
/// gated `RMSNorm` with `z: [1, S, H, 128]` -> rows of `out`.
///
/// # Errors
///
/// Returns an error if `z` is not packed per head or the dispatch fails.
#[allow(clippy::too_many_arguments)]
pub fn post_norm(
    out: &Arc<Buffer>,
    y: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f32,
    t0: usize,
    len: usize,
    h: usize,
) -> Result<()> {
    let dev = y.device().as_metal_device()?.clone();
    let zs = z.layout().stride()[1];
    if z.layout().stride()[2] != HEAD_DIM || z.layout().stride()[3] != 1 {
        candle_core::bail!("gdn metal post: z must be packed per head");
    }
    let pipeline = pipeline(
        &dev,
        &format!("gdn_post_norm_{}", metal_util::float_tag(z.dtype())?),
    )?;
    let y = y.contiguous()?;
    let args = PostArgs {
        t0: u32_of(t0)?,
        len: u32_of(len)?,
        h: u32_of(h)?,
        zs: u32_of(zs)?,
        eps,
    };
    {
        let encoder = dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = encoder.as_ref();
        enc.set_compute_pipeline_state(&pipeline);
        bind_input!(enc, 0, y_g, &y);
        bind_input!(enc, 1, z_g, z);
        bind_input!(enc, 2, w_g, weight);
        enc.set_output_buffer(3, Some(out), 0);
        enc.set_bytes(4, &args);
        enc.dispatch_thread_groups(size(len, h.div_ceil(GROUPS)), size(32 * GROUPS, 1));
    }
    Ok(())
}

/// Wrap the buffer filled by [`post_norm`] as a `[S, H*128]` tensor.
///
/// # Errors
///
/// Never; `Result` for parity with the CUDA launcher.
pub fn post_out_tensor(
    buf: Arc<Buffer>,
    like: &Tensor,
    seq_len: usize,
    h: usize,
) -> Result<Tensor> {
    let dev = like.device().as_metal_device()?;
    Ok(metal_util::wrap(
        dev,
        buf,
        (seq_len, h * HEAD_DIM),
        like.dtype(),
    ))
}
