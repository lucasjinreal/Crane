//! Shared MiniCPM4 decoder-block stack, used by four of VoxCPM2's five
//! sub-networks (`base_lm`, `residual_lm`, `feat_encoder`, `feat_decoder`'s
//! DiT) with different [`MiniCpm4Config`] values. Ported from
//! `voxcpm/modules/minicpm4/model.py`.
//!
//! `embed_tokens` is exposed but never called from [`MiniCpm4Model::forward`]
//! / [`MiniCpm4Model::forward_step`] — both always take precomputed
//! `inputs_embeds`, matching the Python (`VoxCPM2Model` calls
//! `self.base_lm.embed_tokens(text_tokens)` directly, then separately calls
//! `self.base_lm(inputs_embeds=combined_embed, ...)`; `feat_encoder`/
//! `feat_decoder`'s inner `MiniCPMModel`s have `vocab_size == 0` and no
//! embedding at all).

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, rms_norm, Activation, Embedding, RmsNorm, VarBuilder};

use super::config::MiniCpm4Config;
use crate::models::modules::attention::{AttentionConfig, GqaAttention, RopeMode};
use crate::models::modules::ffn::SwiGluFfn;

// ── LongRoPE ─────────────────────────────────────────────────────────────

/// Two-regime (`short_factor` / `long_factor`) frequency-scaled RoPE table.
/// Ported from `MiniCPMLongRoPE`.
///
/// Produces half-width `[max_pos, head_dim/2]` cos/sin tables for
/// [`GqaAttention`]'s `RopeMode::HalfSplit` — the Python builds full-width
/// `head_dim` tables (`cat((freqs, freqs))`) because HF's `rotate_half`
/// convention there expects that duplication baked in; candle's fused
/// `rope()` op does the equivalent duplication internally and wants the
/// half-width `freqs` table directly (same convention already documented in
/// `modules::rotary::RotaryEmbedding`).
pub struct LongRoPE {
    cos_table: Tensor,
    sin_table: Tensor,
}

impl LongRoPE {
    /// `table_len`: how many rows to precompute (a pure memory/efficiency
    /// choice — this crate uses the checkpoint's `max_length`, e.g. 8192,
    /// rather than Python's `max_position_embeddings`, e.g. 32768, since
    /// nothing here ever queries beyond `max_length`). **Not** the same
    /// value as the branch-selection / `scaling_factor` math below, which
    /// Python always derives from the fixed `config.max_position_embeddings`
    /// architectural constant — conflating the two would be a real
    /// divergence from upstream for any config where they differ (they
    /// happen to be equal, 32768, for this checkpoint).
    pub fn new(cfg: &MiniCpm4Config, table_len: usize, device: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let half = dim / 2;
        let scaling = &cfg.rope_scaling;
        let orig_max = scaling.original_max_position_embeddings as f64;

        let ext_factors: &[f64] = if cfg.max_position_embeddings > scaling.original_max_position_embeddings {
            &scaling.long_factor
        } else {
            &scaling.short_factor
        };
        assert_eq!(ext_factors.len(), half, "rope_scaling factor length must equal head_dim/2");

        // Matches `MiniCPMLongRoPE.__init__` exactly: computed unconditionally
        // (no `if scale > 1` guard) — `scale == 1.0` for this checkpoint makes
        // `ln(scale) == 0` and `scaling_factor == 1.0` for the same reason,
        // not because of a special case.
        let scale = cfg.max_position_embeddings as f64 / orig_max;
        let scaling_factor = (1.0 + scale.ln() / orig_max.ln()).sqrt();

        let inv_freq: Vec<f32> = (0..half)
            .map(|i| {
                let base_inv = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / dim as f64);
                (base_inv / ext_factors[i]) as f32
            })
            .collect();
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

        let positions: Vec<f32> = (0..table_len).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?; // [table_len, half]

        let cos_table = (freqs.cos()?.contiguous()? * scaling_factor)?;
        let sin_table = (freqs.sin()?.contiguous()? * scaling_factor)?;

        Ok(Self { cos_table, sin_table })
    }

    /// cos/sin for positions `[start, start+seq_len)`.
    pub fn forward(&self, start: usize, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos_table.narrow(0, start, seq_len)?;
        let sin = self.sin_table.narrow(0, start, seq_len)?;
        Ok((cos, sin))
    }
}

// ── Decoder layer ────────────────────────────────────────────────────────

struct DecoderLayer {
    self_attn: GqaAttention,
    mlp: SwiGluFfn,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    /// `scale_depth / sqrt(num_hidden_layers)` when `use_mup`, else `1.0`
    /// (plain residual) — see `MiniCPMDecoderLayer.forward`'s `use_mup` gate.
    residual_scale: f64,
}

impl DecoderLayer {
    fn new(cfg: &MiniCpm4Config, vb: VarBuilder) -> Result<Self> {
        let attn_cfg = AttentionConfig {
            dim: cfg.hidden_size,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim(),
            qkv_bias: false,
            o_bias: false,
            rope_mode: if cfg.no_rope { RopeMode::None } else { RopeMode::HalfSplit },
            use_qk_norm: false,
            norm_eps: cfg.rms_norm_eps,
        };
        let self_attn = GqaAttention::new(attn_cfg, vb.pp("self_attn"))?;
        let mlp = SwiGluFfn::new(cfg.hidden_size, cfg.intermediate_size, Activation::Silu, vb.pp("mlp"))?;
        let input_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;
        #[allow(clippy::cast_precision_loss)]
        let residual_scale = if cfg.use_mup {
            cfg.scale_depth / (cfg.num_hidden_layers as f64).sqrt()
        } else {
            1.0
        };
        Ok(Self { self_attn, mlp, input_layernorm, post_attention_layernorm, residual_scale })
    }

    fn forward(&mut self, xs: &Tensor, cos_sin: Option<(&Tensor, &Tensor)>, mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let hidden = self.input_layernorm.forward(xs)?;
        let hidden = self.self_attn.forward(&hidden, cos_sin, mask)?;
        let hidden = (residual + (hidden * self.residual_scale)?)?;

        let residual = &hidden;
        let mlp_in = self.post_attention_layernorm.forward(&hidden)?;
        let mlp_out = self.mlp.forward(&mlp_in)?;
        residual + (mlp_out * self.residual_scale)?
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ── Model ────────────────────────────────────────────────────────────────

pub struct MiniCpm4Model {
    pub embed_tokens: Option<Embedding>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    rotary: Option<LongRoPE>,
}

impl MiniCpm4Model {
    /// `rope_table_len`: rows to precompute for [`LongRoPE`] (e.g. the
    /// checkpoint's `max_length`) — ignored when `cfg.no_rope`.
    pub fn new(cfg: &MiniCpm4Config, rope_table_len: usize, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = if cfg.vocab_size > 0 {
            Some(embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, vb_layers.pp(i))?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary =
            if cfg.no_rope { None } else { Some(LongRoPE::new(cfg, rope_table_len, vb.device())?) };

        Ok(Self { embed_tokens, layers, norm, rotary })
    }

    /// Full-sequence forward. `inputs_embeds`: `[B, S, H]`. `is_causal`
    /// controls the mask built internally (`None` mask = full bidirectional
    /// attention, matching `feat_encoder`/`feat_decoder`'s non-causal use).
    pub fn forward(&mut self, inputs_embeds: &Tensor, is_causal: bool) -> Result<Tensor> {
        let (_b, seq_len, _h) = inputs_embeds.dims3()?;
        let cos_sin = match &self.rotary {
            Some(r) => {
                let (cos, sin) = r.forward(0, seq_len)?;
                Some((cos, sin))
            }
            None => None,
        };
        let cos_sin_ref = cos_sin.as_ref().map(|(c, s)| (c, s));

        let mask = if is_causal && seq_len > 1 {
            Some(build_causal_mask(seq_len, inputs_embeds.device(), inputs_embeds.dtype())?)
        } else {
            None
        };

        let mut hidden = inputs_embeds.clone();
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, cos_sin_ref, mask.as_ref())?;
        }
        self.norm.forward(&hidden)
    }

    /// Single-position incremental step. `inputs_embeds`: `[B, H]` (one
    /// position, no explicit seq dim). `position_id`: absolute position for
    /// both the RoPE table lookup and (implicitly, via each layer's
    /// [`GqaAttention`] internal cache) the KV-cache write slot — the caller
    /// is responsible for calling this only in strictly increasing position
    /// order (matching `GqaAttention`'s append-only cache).
    pub fn forward_step(&mut self, inputs_embeds: &Tensor, position_id: usize) -> Result<Tensor> {
        let hidden = inputs_embeds.unsqueeze(1)?; // [B, 1, H]
        let cos_sin = match &self.rotary {
            Some(r) => {
                let (cos, sin) = r.forward(position_id, 1)?;
                Some((cos, sin))
            }
            None => None,
        };
        let cos_sin_ref = cos_sin.as_ref().map(|(c, s)| (c, s));

        let mut hidden = hidden;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, cos_sin_ref, None)?;
        }
        let hidden = self.norm.forward(&hidden)?;
        hidden.squeeze(1)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

/// Additive causal mask `[1, 1, seq_len, seq_len]` (0 where attend, `-inf`
/// where masked) — same convention every other Crane model builds locally.
fn build_causal_mask(seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Tensor::from_slice(&mask, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::voxcpm2::config::RopeScalingConfig;

    // Reference values computed independently in Python against the real
    // VoxCPM2 checkpoint's config.json (theta=10000, kv_channels=128,
    // max_position_embeddings == original_max_position_embeddings == 32768,
    // so scale==1.0 and scaling_factor==1.0 exactly):
    //   pos=5, i=0:  cos=0.2942175021533906  sin=-0.9557384900832547
    //   pos=5, i=1:  cos=-0.4305646386043557 sin=-0.902559744273752
    //   pos=5, i=63: cos=0.9999999998268125  sin=1.8611153426248152e-05
    #[test]
    fn long_rope_matches_python_reference() {
        let device = Device::Cpu;
        // Per-index computation is independent, so only the checked indices
        // (0, 1, 63) need the checkpoint's real `short_factor` values —
        // placeholders elsewhere don't affect those three results.
        let mut short_factor: Vec<f64> = (0..64).map(|_| 1.0).collect();
        short_factor[0] = 0.9977997200264581;
        short_factor[1] = 1.014658295992452;
        short_factor[63] = 31.02392307921529;
        let cfg = MiniCpm4Config {
            hidden_size: 2048,
            intermediate_size: 6144,
            max_position_embeddings: 32768,
            num_attention_heads: 16,
            num_hidden_layers: 28,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-5,
            rope_scaling: RopeScalingConfig {
                rope_type: "longrope".to_string(),
                long_factor: short_factor.clone(),
                short_factor,
                original_max_position_embeddings: 32768,
            },
            rope_theta: 10000.0,
            vocab_size: 0,
            use_mup: false,
            scale_emb: 12.0,
            scale_depth: 1.4,
            kv_channels: Some(128),
            no_rope: false,
        };
        let rope = LongRoPE::new(&cfg, 16, &device).expect("build LongRoPE");
        let (cos, sin) = rope.forward(5, 1).expect("forward");
        let cos: Vec<f32> = cos.flatten_all().unwrap().to_vec1().unwrap();
        let sin: Vec<f32> = sin.flatten_all().unwrap().to_vec1().unwrap();

        let check = |i: usize, exp_cos: f32, exp_sin: f32| {
            assert!((cos[i] - exp_cos).abs() < 1e-4, "cos[{i}]: got {}, expected {exp_cos}", cos[i]);
            assert!((sin[i] - exp_sin).abs() < 1e-4, "sin[{i}]: got {}, expected {exp_sin}", sin[i]);
        };
        check(0, 0.2942175, -0.9557385);
        check(1, -0.4305646, -0.9025597);
        check(63, 0.9999999998, 0.0000186112);
    }
}
