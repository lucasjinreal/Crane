//! Qwen 3.5 transformer layer: hybrid full-attention + linear-attention stack.
//!
//! Layout (per layer):
//!   residual = input_layernorm(x)
//!   attn_or_gdn_out = full_or_linear(residual, …)   // dispatched by layer_types
//!   x = x + attn_or_gdn_out
//!   residual2 = post_attention_layernorm(x)
//!   x = x + mlp(residual2)
//!
//! The full-attention path uses MRoPE-interleaved rotary embeddings and gated
//! output (`attn_output_gate: true` in the config). The linear-attention path
//! uses [`crate::ops::gdn::GatedDeltaNet`].
//!
//! Per-layer GDN state is held by [`super::Qwen3_5TextModel`] (not by the
//! layer), so that continuous-batching can save/restore state per request.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, VarBuilder};

// ── Qwen 3.5 RMSNorm (unit-offset) ───────────────────────────────────────

/// RMSNorm as used by Qwen 3.5: `x / rms(x) * (1 + weight)`.
///
/// Unlike the standard (Llama/Qwen3) RMSNorm — which scales by `weight` — Qwen
/// 3.5 adds a unit offset (`1 + weight`, Gemma-style). HF source:
/// `output = self._norm(x.float()) * (1.0 + self.weight.float())`. The stored
/// weights have mean ~0.24, so omitting the `+1` shrinks every normalized
/// activation ~5x and compounds across layers. Computed in f32 then cast back.
///
/// This is NOT used by the GDN gated norm ([`crate::ops::gdn::RmsNormGated`]), which
/// scales by plain `weight` in HF.
#[derive(Clone)]
pub struct Qwen35RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen35RmsNorm {
    pub fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Module for Qwen35RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        // `1 + weight` (unit offset), in f32.
        let scale = self.weight.to_dtype(DType::F32)?.affine(1.0, 1.0)?;
        x_normed.broadcast_mul(&scale)?.to_dtype(dtype)
    }
}

use super::config::{LayerType, TextConfig};
use super::kv_cache::KvCache;
use crate::ops::gdn::{GatedDeltaNet, GdnDims, GdnInputProjectionKind, GdnLayerCache};

// ── MRoPE rotary embedding ─────────────────────────────────────────────

/// Multimodal Rotary Position Embedding (interleaved variant, used by Qwen 3.5).
///
/// For text-only inference the position_ids are identical across all three
/// sections, so this reduces to standard RoPE applied to the first
/// `rot_dim = head_dim * partial_rotary_factor` components of each head.
///
/// Precomputes `cos` and `sin` tables of shape `[max_pos, rot_dim/2]`. Use
/// [`apply_mrope`] to rotate a query/key tensor.
pub struct MRotaryEmbedding {
    cos_table: Tensor,
    sin_table: Tensor,
    rot_dim: usize,
}

impl MRotaryEmbedding {
    pub fn new(cfg: &TextConfig, device: &Device) -> Result<Self> {
        let rot_dim = cfg.rot_dim();
        let base = cfg.rope_theta() as f32;
        let max_pos = cfg.max_position_embeddings;

        // cos/sin tables have shape `[S, rot_dim/2]` — exactly the slice of the
        // head that receives rotary embeddings. `apply_mrope` rotates only the
        // first `rot_dim` components of each head (HF's partial-rotary scheme:
        // `q_rot = q[..., :rot_dim]`), so the tables must NOT be padded to
        // `head_dim/2`. Padding them to the full head and rotating the whole
        // head (the previous approach) pairs dim `i` with dim `i+head_dim/2`,
        // whereas HF pairs `i` with `i+rot_dim/2` inside the rotary slice — a
        // different rotation entirely.
        let half_rot = rot_dim / 2;
        let inv: Vec<f32> = (0..half_rot)
            .map(|i| 1.0 / base.powf(i as f32 * 2.0 / rot_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv.as_slice(), device)?;

        let positions: Vec<f32> = (0..max_pos).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?; // [max_pos, half_rot]

        let cos_table = freqs.cos()?.contiguous()?;
        let sin_table = freqs.sin()?.contiguous()?;

        Ok(Self {
            cos_table,
            sin_table,
            rot_dim,
        })
    }

    /// Slice cos/sin for positions `[start, start+seq_len)`.
    pub fn cos_sin(&self, start: usize, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos_table.narrow(0, start, seq_len)?;
        let sin = self.sin_table.narrow(0, start, seq_len)?;
        Ok((cos, sin))
    }

    pub fn rot_dim(&self) -> usize {
        self.rot_dim
    }
}

/// Apply rotary embeddings to a query/key tensor `[B, H, S, D]`.
///
/// Only the first `rot_dim` components of each head are rotated; the remaining
/// `D - rot_dim` components are passed through unchanged. This mirrors HF's
/// partial-rotary scheme (`q_rot = q[..., :rot_dim]`, `q_pass = q[..., rot_dim:]`).
/// `cos`/`sin` tables have shape `[S, rot_dim/2]`.
///
/// `candle_nn::rotary_emb::rope` is the rotate-half (non-interleaved / GPT-NeoX)
/// variant — it pairs component `i` with `i + rot_dim/2` *within the slice we
/// hand it*, which matches HF's `rotate_half` over the rotary slice. We must
/// slice first; rotating the full head would pair `i` with `i + head_dim/2`.
pub fn apply_mrope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rot_dim: usize,
) -> Result<Tensor> {
    let (_b, _h, _seq_len, head_dim) = x.dims4()?;
    let dtype = x.dtype();
    // `cos`/`sin` are kept in f32; the rope op requires its inputs to share a
    // dtype, so for half-precision activations (F16/BF16 on GPU) we rotate in
    // f32 and cast back. This also matches HF, which applies RoPE in float.
    let rope_f32 = |t: &Tensor| -> Result<Tensor> {
        let r = candle_nn::rotary_emb::rope(&t.to_dtype(DType::F32)?.contiguous()?, cos, sin)?;
        r.to_dtype(dtype)
    };
    if rot_dim == head_dim {
        return rope_f32(x);
    }
    let x_rot = rope_f32(&x.narrow(D::Minus1, 0, rot_dim)?)?;
    let x_pass = x.narrow(D::Minus1, rot_dim, head_dim - rot_dim)?;
    Tensor::cat(&[&x_rot, &x_pass], D::Minus1)?.contiguous()
}

// ── Full-attention layer ────────────────────────────────────────────────

/// Standard softmax attention layer for Qwen 3.5's `full_attention` blocks.
///
/// Differences from the regular Qwen 3 attention:
/// - `q_proj` outputs `num_heads * head_dim * 2`; the second half is a sigmoid
///   gate applied to the attention output.
/// - Per-head QK-norm is always present (`q_norm`, `k_norm` of size `head_dim`).
/// - RoPE is MRoPE-interleaved applied only to the first `rot_dim` components.
pub struct FullAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    q_norm: Qwen35RmsNorm,
    k_norm: Qwen35RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    has_output_gate: bool,
}

impl FullAttention {
    pub fn load(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        let q_out = if cfg.attn_output_gate {
            num_heads * head_dim * 2
        } else {
            num_heads * head_dim
        };
        let q_proj = linear_no_bias(cfg.hidden_size, q_out, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;

        let q_norm = Qwen35RmsNorm::load(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = Qwen35RmsNorm::load(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            has_output_gate: cfg.attn_output_gate,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rot_dim: usize,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;


        let q_out = self.q_proj.forward(x)?;
        let k_proj_out = self.k_proj.forward(x)?;
        let v_proj_out = self.v_proj.forward(x)?;
        let k = k_proj_out;
        let v = v_proj_out;

        let (q, gate) = if self.has_output_gate {
            // HF splits `[query | gate]` PER HEAD, not on the flat axis:
            //   q_proj(x).view(B, S, num_heads, head_dim*2).chunk(2, dim=-1)
            // so for each head the first `head_dim` is the query and the next
            // `head_dim` is the gate. Splitting the flat 4096 in half instead
            // interleaves heads' query/gate and scrambles q_norm. Re-flatten
            // both back to `[B, S, num_heads*head_dim]` in head order.
            let flat = self.num_heads * self.head_dim;
            let qh = q_out.reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?;
            let q = qh
                .narrow(D::Minus1, 0, self.head_dim)?
                .contiguous()?
                .reshape((b_sz, seq_len, flat))?;
            let gate = qh
                .narrow(D::Minus1, self.head_dim, self.head_dim)?
                .contiguous()?
                .reshape((b_sz, seq_len, flat))?;
            (q, Some(gate))
        } else {
            (q_out, None)
        };


        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q = apply_mrope(&q, cos, sin, rot_dim)?;
        let k = apply_mrope(&k, cos, sin, rot_dim)?;

        // Append this step's K/V to the cache (post-RoPE, pre-GQA-expand) and
        // continue with the full cached K/V. During incremental decode this is
        // what lets the attention see the whole context from a single token.
        let (k, v) = match kv_cache {
            Some(cache) => cache.append(&k, &v)?,
            None => (k, v),
        };

        let n_rep = self.num_heads / self.num_kv_heads;
        let scale = 1.0 / (self.head_dim as f64).sqrt();

        let k_rep = if n_rep > 1 {
            let (b, kv_heads, s, d) = k.dims4()?;
            k.unsqueeze(2)?
                .expand((b, kv_heads, n_rep, s, d))?
                .contiguous()?
                .reshape((b, self.num_heads, s, d))?
        } else {
            k
        };
        let v_rep = if n_rep > 1 {
            let (b, kv_heads, s, d) = v.dims4()?;
            v.unsqueeze(2)?
                .expand((b, kv_heads, n_rep, s, d))?
                .contiguous()?
                .reshape((b, self.num_heads, s, d))?
        } else {
            v
        };

        let k_t = k_rep.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn_logits = (q.matmul(&k_t)? * scale)?;
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights_with_mask(&attn_logits, mask)?,
            None => attn_logits,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let y = attn_weights.matmul(&v_rep)?;

        let y = y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?;

        let y = if let Some(g) = gate {
            let gate = candle_nn::ops::sigmoid(&g.to_dtype(y.dtype())?)?;
            y.broadcast_mul(&gate)?
        } else {
            y
        };

        self.o_proj.forward(&y)
    }
}

fn attn_weights_with_mask(
    attn_logits: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    // HF applies the mask (shape `[B, 1, S_q, S_k]` for additive causal mask)
    // via `attn_weights + mask` and softmax. We do the same.
    Ok(attn_logits.broadcast_add(mask)?)
}

// ── MLP ────────────────────────────────────────────────────────────────

/// Standard SwiGLU MLP: `down(silu(gate(x)) * up(x))`.
pub struct Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Mlp {
    pub fn load(cfg: &TextConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let h = gate.broadcast_mul(&up)?;
        let out = self.down_proj.forward(&h)?;
        Ok(out)
    }
}

// ── DecoderLayer ────────────────────────────────────────────────────────

/// One transformer block. `LayerImpl` selects which attention path runs.
///
/// Per-layer GDN state is passed in via the [`DecoderLayer::forward`] signature
/// (a `&mut Option<GdnLayerCache>`); the model holds the canonical cache array.
pub struct DecoderLayer {
    layer_impl: LayerImpl,
    input_layernorm: Qwen35RmsNorm,
    post_attention_layernorm: Qwen35RmsNorm,
    mlp: Mlp,
    /// Pre-computed dims for the GDN path. `None` for full-attention blocks.
    gdn_dims: Option<GdnDims>,
}

enum LayerImpl {
    FullAttention(FullAttention),
    LinearAttention(GatedDeltaNet),
}

impl DecoderLayer {
    pub fn load(
        cfg: &TextConfig,
        layer_type: LayerType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_layernorm = Qwen35RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = Qwen35RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = Mlp::load(cfg, vb.pp("mlp"))?;

        let (layer_impl, gdn_dims) = match layer_type {
            LayerType::FullAttention => (
                LayerImpl::FullAttention(FullAttention::load(cfg, vb.pp("self_attn"))?),
                None,
            ),
            LayerType::LinearAttention => {
                let dims = GdnDims::new(cfg);
                let gdn = GatedDeltaNet::load(vb, cfg, GdnInputProjectionKind::Split)?;
                (LayerImpl::LinearAttention(gdn), Some(dims))
            }
        };

        Ok(Self {
            layer_impl,
            input_layernorm,
            post_attention_layernorm,
            mlp,
            gdn_dims,
        })
    }

    pub fn is_linear(&self) -> bool {
        matches!(self.layer_impl, LayerImpl::LinearAttention(_))
    }

    /// Forward pass. For `LinearAttention` blocks pass `Some(gdn_cache)` and
    /// `None` for `attn_cache`; for `FullAttention` blocks pass `Some(attn_cache)`
    /// and `None` for `gdn_cache`.
    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        rot_dim: usize,
        attention_mask: Option<&Tensor>,
        gdn_cache: Option<&mut GdnLayerCache>,
        attn_cache: Option<&mut KvCache>,
        is_decode_step: bool,
    ) -> Result<Tensor> {
        let residual = x;
        let normed = self.input_layernorm.forward(x)?;

        let attn_out = match &self.layer_impl {
            LayerImpl::FullAttention(attn) => {
                debug_assert!(gdn_cache.is_none(), "full-attention layer should not receive a GDN cache");
                attn.forward(&normed, cos, sin, rot_dim, attention_mask, attn_cache)?
            }
            LayerImpl::LinearAttention(gdn) => {
                debug_assert!(attn_cache.is_none(), "linear-attention layer should not receive a KV cache");
                let cache = gdn_cache.ok_or_else(|| {
                    candle_core::Error::Msg("GDN cache missing for linear-attention layer".into())
                })?;
                let dims = self.gdn_dims.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("GDN dims missing for linear-attention layer".into())
                })?;
                gdn.forward(&normed, dims, cache, is_decode_step)?
            }
        };

        let x = (residual + attn_out)?;

        let residual2 = &x;
        let normed2 = self.post_attention_layernorm.forward(&x)?;

        let mlp_out = self.mlp.forward(&normed2)?;

        residual2 + mlp_out
    }
}
