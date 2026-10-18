//! Top-level `GatedDeltaNet` layer: orchestrates input projection, causal
//! `Conv1D`, gated delta rule recurrence, gated `RMSNorm`, and output projection.

#[cfg(any(feature = "cuda", feature = "metal"))]
use candle_core::DType;
use candle_core::quantized::GgmlDType;
use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::backend::{GdnGateConsts, apply_recurrence, compute_beta_g, l2_alpha, l2_norm_fused};
use super::cache::GdnLayerCache;
use super::config::{GdnConfig, GdnDims, VHeadOrder};
use super::conv::causal_conv1d_ckpt;
use super::norm::RmsNormGated;
use super::projection::{GdnInputProjection, GdnInputProjectionKind};
use crate::ops::linear::{LinearLayer, linear_layer};

/// The fused multi-token stages for this build's GPU backend (same API on both).
#[cfg(feature = "cuda")]
use super::gdn_fused as gf;
#[cfg(all(feature = "metal", not(feature = "cuda")))]
use super::metal_backend as gf;

/// One linear-attention layer as used by Qwen 3.5 (and similar hybrid models).
///
/// Field names follow the HF safetensors layout (`linear_attn.in_proj_qkv.weight`,
/// `linear_attn.conv1d.weight`, `linear_attn.A_log`, …).
pub struct GatedDeltaNet {
    pub input_proj: GdnInputProjection,
    pub conv1d_weight: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
    pub norm: RmsNormGated,
    pub out_proj: LinearLayer,
    /// Per-token-invariant values derived from `a_log` / `dt_bias` / the head
    /// dim. Built by [`GatedDeltaNet::with_derived`]; kept out of the public
    /// constructors' argument lists because they are pure functions of fields
    /// that are already there.
    derived: GdnDerived,
}

/// Constants the forward pass would otherwise rebuild on every token.
struct GdnDerived {
    gates: GdnGateConsts,
    /// The `[head_k_dim]` scale vector for the fused Q/K L2 norm.
    l2_alpha: Tensor,
    /// Weights pre-rounded for the fused path (CUDA, Metal).
    #[cfg(any(feature = "cuda", feature = "metal"))]
    fused: Option<FusedConsts>,
}

/// What the fused kernels need, derived once: the op-by-op path rounds the
/// conv and norm weights through the activation dtype (bf16 on CUDA, the model
/// dtype on Metal) on every call; doing it here is the same arithmetic without
/// the per-call launches.
#[cfg(any(feature = "cuda", feature = "metal"))]
struct FusedConsts {
    /// `conv1d.weight` as `[conv_dim, kernel]` f32 (via the activation dtype).
    conv_w: Tensor,
    /// Gated-norm weight `[head_v_dim]` f32 (via the activation dtype).
    norm_w: Tensor,
}

impl GatedDeltaNet {
    /// Load weights from `vb.pp("linear_attn")`.
    ///
    /// `cfg` supplies the dimensions; `projection_kind` selects the QKV/Z/B/A
    /// weight layout (Qwen 3.5 uses [`GdnInputProjectionKind::Split`]).
    /// `quant` requests in-situ quantization of the large projections
    /// (conv1d / `dt_bias` / `A_log` / norm always stay in full precision).
    ///
    /// # Errors
    ///
    /// Returns an error if a required weight is missing from `vb`, the
    /// config names an unsupported gate activation, or a loaded tensor's
    /// shape does not match `cfg`'s dimensions.
    pub fn load(
        vb: &VarBuilder,
        cfg: &dyn GdnConfig,
        projection_kind: GdnInputProjectionKind,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let dims = GdnDims::new(cfg);
        let vb_la = vb.pp("linear_attn");

        let input_proj = GdnInputProjection::load(&vb_la, &dims, projection_kind, quant)?;
        let conv1d_weight =
            vb_la.get((dims.conv_dim, 1, dims.conv_kernel_size), "conv1d.weight")?;
        let dt_bias = vb_la.get(dims.num_v_heads, "dt_bias")?;
        let a_log = vb_la.get(dims.num_v_heads, "A_log")?;

        let norm = RmsNormGated::new(
            dims.head_v_dim,
            cfg.rms_norm_eps(),
            cfg.output_gate_activation()?,
            &vb_la.pp("norm"),
        )?;
        let out_proj = linear_layer(
            dims.value_dim,
            dims.hidden_size,
            vb_la.pp("out_proj"),
            quant,
        )?;

        Self::with_derived(
            input_proj,
            conv1d_weight,
            dt_bias,
            a_log,
            norm,
            out_proj,
            &dims,
        )
    }

    /// Load linear-attention block `layer_idx` from a GGUF file in llama.cpp's
    /// delta-net layout (`qwen35`, `qwen4exp`), returning it with its dims.
    ///
    /// The split projections are `attn_qkv` (Q|K|V), `attn_gate` (z),
    /// `ssm_beta` (β) and `ssm_alpha` (A); `ssm_conv1d` is 2-D
    /// `[conv_dim, kernel]`, `ssm_a` stores `-exp(A_log)` and `ssm_dt.bias` is
    /// `dt_bias`; the gated `ssm_norm` is a plain weight as in HF.
    ///
    /// The converter orders the value-head axis differently from HF
    /// ([`VHeadOrder::Chunked`] vs `Interleaved`); rather than permuting the
    /// affected weights — which would mean dequantizing and re-quantizing
    /// them, and `Q6_K`'s quantizer is not idempotent — the returned dims
    /// record the order and the Q/K expansion adapts to it.
    ///
    /// # Errors
    ///
    /// Returns an error if a required tensor is missing or the config names an
    /// unsupported gate activation.
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        gg: &mut crate::quantized::gguf_file::Gguf<R>,
        layer_idx: usize,
        cfg: &dyn GdnConfig,
    ) -> Result<(Self, GdnDims)> {
        let prefix = format!("blk.{layer_idx}");
        let dims = GdnDims::new(cfg).with_v_head_order(VHeadOrder::Chunked);
        let input_proj = Self::gguf_input_projection(gg, &prefix)?;
        // GGUF stores the conv kernel 2-D; crane expects HF's
        // `[conv_dim, 1, kernel]`.
        let conv1d_weight = gg
            .dequant_tensor(&format!("{prefix}.ssm_conv1d.weight"))?
            .unsqueeze(1)?;
        let dt_bias = gg.dequant_tensor(&format!("{prefix}.ssm_dt.bias"))?;
        let a_log = gg
            .dequant_tensor(&format!("{prefix}.ssm_a"))?
            .neg()?
            .log()?;
        let norm = RmsNormGated::from_weight(
            gg.dequant_tensor(&format!("{prefix}.ssm_norm.weight"))?,
            cfg.rms_norm_eps(),
            cfg.output_gate_activation()?,
        );
        let out_proj = gg.linear(&format!("{prefix}.ssm_out.weight"))?;
        let gdn = Self::with_derived(
            input_proj,
            conv1d_weight,
            dt_bias,
            a_log,
            norm,
            out_proj,
            &dims,
        )?;
        Ok((gdn, dims))
    }

    /// The four split projections, stacked where they can run as one matvec
    /// (see [`GdnInputProjection::Fused`]): `attn_qkv` with `attn_gate` when
    /// they share a quant type, and `ssm_beta` with `ssm_alpha` as one dense
    /// matrix, dequantized like the safetensors path keeps them (they feed
    /// the numerically sensitive β/decay path).
    ///
    /// Decode results are bit-identical to the separate layers; prefill's
    /// tiled quantized matmul rounds differently for the taller matrix
    /// (~1e-6 relative). Quantizing β/α too, so all four run as one matvec,
    /// measured another ~1.5% of decode on Ornith-1.5-35B but is not done,
    /// for that numerical reason.
    fn gguf_input_projection<R: std::io::Read + std::io::Seek>(
        gg: &mut crate::quantized::gguf_file::Gguf<R>,
        prefix: &str,
    ) -> Result<GdnInputProjection> {
        use super::projection::GdnPart;

        let [qkv, z, b, a] = ["attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha"]
            .map(|n| format!("{prefix}.{n}.weight"));
        let mut groups = Vec::with_capacity(3);
        match gg.linear_fused(&[&qkv, &z])? {
            Some(l) => groups.push((l, vec![GdnPart::Qkv, GdnPart::Z])),
            None => {
                groups.push((gg.linear(&qkv)?, vec![GdnPart::Qkv]));
                groups.push((gg.linear(&z)?, vec![GdnPart::Z]));
            },
        }
        let ba = Tensor::cat(&[gg.dequant_tensor(&b)?, gg.dequant_tensor(&a)?], 0)?;
        groups.push((
            LinearLayer::Standard(candle_nn::Linear::new(ba, None)),
            vec![GdnPart::B, GdnPart::A],
        ));
        Ok(GdnInputProjection::Fused(groups))
    }

    /// Assemble from already-loaded parts, deriving the per-token constants.
    ///
    /// This rather than a struct literal: `derived` has to stay consistent with
    /// `a_log` and `dt_bias`, and a literal would let a caller forget.
    ///
    /// # Errors
    ///
    /// Returns an error if deriving the gate constants from `a_log`/`dt_bias`
    /// or building the L2-norm scale vector fails.
    #[allow(clippy::too_many_arguments)]
    pub fn with_derived(
        input_proj: GdnInputProjection,
        conv1d_weight: Tensor,
        dt_bias: Tensor,
        a_log: Tensor,
        norm: RmsNormGated,
        out_proj: LinearLayer,
        dims: &GdnDims,
    ) -> Result<Self> {
        // `conv1d_weight` is the dtype/device witness: it is always present and
        // always carries the model's compute dtype on both load paths.
        #[cfg(any(feature = "cuda", feature = "metal"))]
        let fused = {
            let device = conv1d_weight.device();
            // CUDA's fused pass is bf16-only; Metal's runs in the model dtype.
            let act = if device.is_cuda() {
                Some(DType::BF16)
            } else if device.is_metal() {
                Some(conv1d_weight.dtype())
            } else {
                None
            };
            match act {
                Some(act) if norm.activation() == super::norm::GateActivation::Silu => {
                    let via = |t: &Tensor| t.to_dtype(act)?.to_dtype(DType::F32)?.contiguous();
                    Some(FusedConsts {
                        conv_w: via(&conv1d_weight.squeeze(1)?)?,
                        norm_w: via(norm.weight())?,
                    })
                },
                _ => None,
            }
        };
        let derived = GdnDerived {
            gates: GdnGateConsts::new(&a_log, &dt_bias)?,
            l2_alpha: l2_alpha(
                dims.head_k_dim,
                conv1d_weight.dtype(),
                conv1d_weight.device(),
            )?,
            #[cfg(any(feature = "cuda", feature = "metal"))]
            fused,
        };
        Ok(Self {
            input_proj,
            conv1d_weight,
            dt_bias,
            a_log,
            norm,
            out_proj,
            derived,
        })
    }

    /// Forward pass over a sequence: a whole prompt, one prefill chunk, or a
    /// single decode step — all continue from `cache`, which is updated in
    /// place.
    ///
    /// `x: [B, S, hidden_size]`. Returns `[B, S, hidden_size]`.
    ///
    /// # Errors
    ///
    /// Returns an error if `x` is not 3-dimensional or if any stage —
    /// input projection, causal `Conv1D`, the gated delta rule recurrence,
    /// gated `RMSNorm`, or output projection — fails on a shape/dtype
    /// mismatch.
    pub fn forward(&self, x: &Tensor, dims: &GdnDims, cache: &mut GdnLayerCache) -> Result<Tensor> {
        use crate::utils::prof::{Span, timed};

        let (batch_size, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();

        // Speculative decoding asks for a copy of the state after the first
        // `p` tokens. Projections and the conv run once over the whole input;
        // only the conv-state capture and the recurrence are split at `p`.
        let ckpt = cache.checkpoint_at.take();
        if ckpt == Some(0) {
            cache.save_checkpoint()?;
        }
        let split = ckpt.filter(|&p| p > 0 && p < seq_len);

        // 1. Input projections → Q, K, V, Z, B, A.
        let projected = timed(Span::GdnProj, || {
            self.input_proj.forward(x, dims, batch_size, seq_len)
        })?;

        // Fused path: conv, Q/K/V prep, recurrence and gated norm in a
        // handful of launches instead of ~45 (see `gdn_fused`; on Metal every
        // pass, see `metal_backend`).
        #[cfg(any(feature = "cuda", feature = "metal"))]
        if let Some(fused) = &self.derived.fused
            && projected.mixed_qkv.is_some()
            && gf::supported(x, dims)
        {
            let out = self.forward_fused(&projected, fused, dims, cache, split)?;
            if ckpt.is_some_and(|p| p >= seq_len) {
                cache.save_checkpoint()?;
            }
            return Ok(out);
        }

        // 2. Causal Conv1D over [Q|K|V].
        let (mixed_qkv, conv_ckpt) = timed(Span::GdnConv, || {
            let mixed_qkv = projected.conv_input(dims, batch_size, seq_len)?;
            causal_conv1d_ckpt(&mixed_qkv, &self.conv1d_weight, dims, cache, split)
        })?;

        // 3-6 in one launch on SYCL (two around a speculative checkpoint).
        #[cfg(all(feature = "sycl", not(feature = "cuda"), not(feature = "rocm")))]
        if mixed_qkv.device().is_sycl() && std::env::var("CRANE_GDN_PORTABLE").is_err() {
            let fused = timed(Span::GdnRecur, || -> Result<Option<Tensor>> {
                let run = |state: &Tensor, t0: usize, len: usize| {
                    super::sycl_backend::gdn_fused_sycl(
                        &mixed_qkv.narrow(1, t0, len)?,
                        &projected.a.narrow(1, t0, len)?,
                        &projected.b.narrow(1, t0, len)?,
                        &self.derived.gates.neg_exp_a_log,
                        &self.derived.gates.dt_bias,
                        state,
                        dims,
                    )
                };
                let p = split.unwrap_or(seq_len);
                let Some((head, state)) = run(&cache.recurrent_state, 0, p)? else {
                    return Ok(None);
                };
                cache.recurrent_state = state;
                if p == seq_len {
                    return Ok(Some(head));
                }
                let conv_state = conv_ckpt
                    .clone()
                    .expect("conv checkpoint computed for 0 < p < seq_len");
                cache.checkpoint = Some((conv_state, cache.recurrent_state.copy()?));
                let (tail, state) = run(&cache.recurrent_state, p, seq_len - p)?
                    .expect("the tail has the head's shapes and dtype");
                cache.recurrent_state = state;
                Ok(Some(Tensor::cat(&[&head, &tail], 1)?))
            })?;
            if let Some(y) = fused {
                if ckpt.is_some_and(|p| p >= seq_len) {
                    cache.save_checkpoint()?;
                }
                return timed(Span::GdnFinish, || {
                    self.finish_forward(&y, &projected.z, batch_size, seq_len)
                });
            }
        }

        // 3-5. Per-head split, QK L2-norm, and the β/g gates.
        let (q, k, v, beta, g) = timed(Span::GdnQkv, || -> Result<_> {
            // Split → per-head Q, K, V; expand K from num_k_heads → num_v_heads.
            let (q, k, v) = split_qkv(&mixed_qkv, dims, batch_size, seq_len)?;
            let (q, k) = repeat_kv_heads(&q, &k, dims)?;
            // L2-normalize Q and K (Qwen 3.5 uses QK-norm = L2).
            let q = l2_norm_fused(&q, &self.derived.l2_alpha, 1e-6)?;
            let k = l2_norm_fused(&k, &self.derived.l2_alpha, 1e-6)?;
            // β (write strength) and g (decay) from B, A, and the gate consts.
            let (beta, g) = compute_beta_g(&projected.b, &projected.a, &self.derived.gates, dtype)?;
            Ok((q, k, v, beta, g))
        })?;

        // 6. Run the gated delta rule recurrence.
        let y = timed(Span::GdnRecur, || -> Result<Tensor> {
            let Some(p) = split else {
                return apply_recurrence(
                    &q, &k, &v, &g, &beta, dims, batch_size, seq_len, cache, dtype,
                );
            };
            let seg = |t: &Tensor, from: usize, len: usize| t.narrow(1, from, len);
            let head = apply_recurrence(
                &seg(&q, 0, p)?,
                &seg(&k, 0, p)?,
                &seg(&v, 0, p)?,
                &seg(&g, 0, p)?,
                &seg(&beta, 0, p)?,
                dims,
                batch_size,
                p,
                cache,
                dtype,
            )?;
            let conv_state = conv_ckpt.expect("conv checkpoint computed for 0 < p < seq_len");
            cache.checkpoint = Some((conv_state, cache.recurrent_state.copy()?));
            let tail_len = seq_len - p;
            let tail = apply_recurrence(
                &seg(&q, p, tail_len)?,
                &seg(&k, p, tail_len)?,
                &seg(&v, p, tail_len)?,
                &seg(&g, p, tail_len)?,
                &seg(&beta, p, tail_len)?,
                dims,
                batch_size,
                tail_len,
                cache,
                dtype,
            )?;
            Tensor::cat(&[&head, &tail], 1)
        })?;

        if ckpt.is_some_and(|p| p >= seq_len) {
            cache.save_checkpoint()?;
        }

        // 7. Gated RMSNorm + output projection.
        timed(Span::GdnFinish, || {
            self.finish_forward(&y, &projected.z, batch_size, seq_len)
        })
    }

    /// The fused path (a multi-token bf16 pass on CUDA, any pass on Metal);
    /// `split` is the speculative-decoding checkpoint position
    /// (`0 < p < seq_len`).
    #[cfg(any(feature = "cuda", feature = "metal"))]
    fn forward_fused(
        &self,
        projected: &super::projection::GdnProjection,
        fused: &FusedConsts,
        dims: &GdnDims,
        cache: &mut GdnLayerCache,
        split: Option<usize>,
    ) -> Result<Tensor> {
        use crate::utils::prof::{Span, timed};

        let mixed = projected.mixed_qkv.as_ref().expect("checked by the caller");
        let seq_len = mixed.dim(1)?;
        let h = dims.num_v_heads;
        let x2 = mixed.reshape((seq_len, dims.conv_dim))?.contiguous()?;
        let act = x2.dtype();
        if cache.conv_state.dtype() != act {
            cache.conv_state = cache.conv_state.to_dtype(act)?;
        }
        let state = cache.conv_state.contiguous()?;
        let mut conv = timed(Span::GdnConv, || {
            gf::conv_silu(&x2, &state, &fused.conv_w, split)
        })?;
        cache.conv_state = conv.state.clone();

        let segments: Vec<(usize, usize)> = match split {
            Some(p) => vec![(0, p), (p, seq_len - p)],
            None => vec![(0, seq_len)],
        };
        let out_buf = gf::alloc_post_out(&x2, seq_len, h)?;
        #[allow(clippy::cast_possible_truncation)]
        let scale = (1.0 / (dims.head_k_dim as f64).sqrt()) as f32;
        for (i, &(t0, len)) in segments.iter().enumerate() {
            if i == 1 {
                // State after the first `p` tokens.
                let conv_state = conv
                    .ckpt
                    .take()
                    .expect("conv checkpoint for 0 < p < seq_len");
                cache.checkpoint = Some((conv_state, cache.recurrent_state.copy()?));
            }
            let pre = timed(Span::GdnQkv, || {
                gf::prep(
                    &conv.y,
                    &projected.a,
                    &projected.b,
                    &self.derived.gates.neg_exp_a_log,
                    &self.derived.gates.dt_bias,
                    dims,
                    t0,
                    len,
                    scale,
                )
            })?;
            let y = timed(Span::GdnRecur, || {
                super::backend::recurrence_prepped(
                    &pre.q, &pre.k, &pre.v, &pre.g, &pre.beta, dims, 1, len, cache, act,
                )
            })?;
            // The chunked (long-prefill) recurrence yields bf16 on CUDA, the
            // serial one f32; the kernel reads f32 and re-rounding a bf16 value
            // is exact.
            let y = if y.dtype() == DType::F32 {
                y
            } else {
                y.to_dtype(DType::F32)?
            };
            timed(Span::GdnFinish, || {
                gf::post_norm(
                    &out_buf,
                    &y,
                    &projected.z,
                    &fused.norm_w,
                    self.norm.eps(),
                    t0,
                    len,
                    h,
                )
            })?;
        }
        let out =
            gf::post_out_tensor(out_buf, &x2, seq_len, h)?.reshape((1, seq_len, dims.value_dim))?;
        timed(Span::GdnFinish, || self.out_proj.forward(&out))
    }

    fn finish_forward(
        &self,
        y: &Tensor,
        z: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        // y comes out of the recurrence as `[B, S, num_v_heads, head_v_dim]`.
        // Flatten the head + head_v dims, gated-RMSNorm along head_v, then
        // reshape back to `[B, S, value_dim]`.
        let head_v_dim = self.norm_head_dim();
        let value_dim = y.dim(2)? * head_v_dim;
        let y_2d = y.reshape(((), head_v_dim))?;
        let z_2d = z.reshape(((), head_v_dim))?;
        let y_2d = self.norm.forward(&y_2d, &z_2d)?;
        let y = y_2d.reshape((batch_size, seq_len, value_dim))?;
        self.out_proj.forward(&y)
    }

    fn norm_head_dim(&self) -> usize {
        // out_proj.weight shape is [value_dim, hidden_size]; head_v_dim is
        // stored indirectly as value_dim / num_v_heads. We recover it from
        // the norm's RmsNorm weight length.
        self.norm.weight_len()
    }
}

/// Split the conv1d output `[B, S, conv_dim]` into Q, K, V along the last
/// dim, then reshape each into `[B, S, num_heads, head_dim]`.
fn split_qkv(
    mixed_qkv: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    use candle_core::D;
    let q = mixed_qkv.narrow(D::Minus1, 0, dims.key_dim)?;
    let k = mixed_qkv.narrow(D::Minus1, dims.key_dim, dims.key_dim)?;
    let v = mixed_qkv.narrow(D::Minus1, dims.key_dim * 2, dims.value_dim)?;
    let q = q.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?;
    let k = k.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?;
    let v = v.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?;
    Ok((q, k, v))
}

/// Expand K (and Q, for symmetry) from `num_k_heads` to `num_v_heads` by
/// repeating each key head `v_per_group` times (GQA-style expansion for
/// linear attention).
///
/// The expansion must place each key head's replicas at the same indices the
/// checkpoint's value heads occupy, so it follows `dims.v_head_order`:
/// `Interleaved` (HF) repeats along a trailing axis, putting a key head's
/// replicas next to each other; `Chunked` (llama.cpp GGUF) repeats along a
/// leading axis, laying out one full pass of key heads per replica. Both
/// produce the same multiset of rows — only the pairing differs.
///
/// When `num_k_heads == num_v_heads` (`v_per_group == 1`) this is a no-op and
/// the two orders coincide.
fn repeat_kv_heads(q: &Tensor, k: &Tensor, dims: &GdnDims) -> Result<(Tensor, Tensor)> {
    if dims.v_per_group == 1 {
        return Ok((q.clone(), k.clone()));
    }
    // `expand` + one `contiguous` copy rather than `Tensor::repeat`, which
    // lowers to a `cat` of `v_per_group` copies — one launch instead of
    // `v_per_group`, per tensor per GDN layer per token.
    let expand = |t: &Tensor| -> Result<Tensor> {
        let (b, s, h, d) = t.dims4()?;
        let v = dims.v_per_group;
        let repeated = match dims.v_head_order {
            VHeadOrder::Interleaved => t.unsqueeze(3)?.expand((b, s, h, v, d))?,
            VHeadOrder::Chunked => t.unsqueeze(2)?.expand((b, s, v, h, d))?,
        };
        repeated
            .contiguous()?
            .reshape((b, s, dims.num_v_heads, dims.head_k_dim))
    };
    Ok((expand(q)?, expand(k)?))
}

// ─────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// `num_k_heads=2, v_per_group=2` -> 4 value heads. Each key head must end
    /// up at exactly the indices its value heads occupy, which differs per
    /// order. Tagging each key head with a distinct value makes the resulting
    /// pairing directly readable.
    fn dims_2x2(order: VHeadOrder) -> GdnDims {
        GdnDims {
            hidden_size: 8,
            num_k_heads: 2,
            num_v_heads: 4,
            head_k_dim: 1,
            head_v_dim: 1,
            conv_kernel_size: 4,
            key_dim: 2,
            value_dim: 4,
            conv_dim: 8,
            v_per_group: 2,
            v_head_order: order,
        }
    }

    fn expanded(order: VHeadOrder) -> Vec<f32> {
        // [B=1, S=1, num_k_heads=2, head_k_dim=1]: key head 0 -> 10, head 1 -> 20
        let t = Tensor::new(&[[[[10.0f32], [20.0f32]]]], &Device::Cpu).unwrap();
        let dims = dims_2x2(order);
        let (q, k) = repeat_kv_heads(&t, &t, &dims).unwrap();
        assert_eq!(q.dims(), &[1, 1, 4, 1]);
        assert_eq!(k.dims(), &[1, 1, 4, 1]);
        q.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// HF: a key head's replicas are adjacent.
    #[test]
    fn repeat_kv_heads_interleaved_groups_replicas_together() {
        assert_eq!(
            expanded(VHeadOrder::Interleaved),
            vec![10.0, 10.0, 20.0, 20.0]
        );
    }

    /// llama.cpp GGUF: one full pass over key heads per replica. Mixing this
    /// up with the interleaved order is the Qwen3.5-GGUF value-head bug.
    #[test]
    fn repeat_kv_heads_chunked_repeats_whole_key_head_pass() {
        assert_eq!(expanded(VHeadOrder::Chunked), vec![10.0, 20.0, 10.0, 20.0]);
    }

    /// Ratios other than 2 must follow the same rule. Qwen3.5-27B is
    /// `16 key / 48 value` heads (v_per_group == 3) and the 122B/397B are
    /// `16 / 64` (== 4), so the formulas must not assume 2 or a power of two.
    /// Modelled here at `2 key / 6 value` heads.
    #[test]
    fn repeat_kv_heads_generalizes_beyond_v_per_group_2() {
        let t = Tensor::new(&[[[[10.0f32], [20.0f32]]]], &Device::Cpu).unwrap();
        let base = GdnDims {
            num_v_heads: 6,
            value_dim: 6,
            v_per_group: 3,
            ..dims_2x2(VHeadOrder::Interleaved)
        };

        // key_head * v_per_group + replica
        let (q, _) = repeat_kv_heads(&t, &t, &base).unwrap();
        assert_eq!(
            q.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
        );

        // replica * num_k_heads + key_head
        let chunked = GdnDims {
            v_head_order: VHeadOrder::Chunked,
            ..base
        };
        let (q, _) = repeat_kv_heads(&t, &t, &chunked).unwrap();
        assert_eq!(
            q.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0]
        );
    }

    /// With `v_per_group == 1` the two orders coincide and expansion is a no-op.
    #[test]
    fn repeat_kv_heads_is_noop_without_grouping() {
        let t = Tensor::new(&[[[[10.0f32], [20.0f32]]]], &Device::Cpu).unwrap();
        for order in [VHeadOrder::Interleaved, VHeadOrder::Chunked] {
            let mut dims = dims_2x2(order);
            dims.num_v_heads = 2;
            dims.value_dim = 2;
            dims.v_per_group = 1;
            let (q, _) = repeat_kv_heads(&t, &t, &dims).unwrap();
            assert_eq!(
                q.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                vec![10.0, 20.0]
            );
        }
    }
}
