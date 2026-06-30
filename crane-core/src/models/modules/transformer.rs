//! Shared pre-norm transformer decoder block (`TransformerBlock`).
//!
//! Combines [`GqaAttention`] and [`SwiGluFfn`] with two [`RmsNorm`] layers into
//! the standard pre-norm decoder pattern used across Qwen3-TTS and Qwen2.5.
//!
//! # Forward pass
//!
//! ```text
//! residual = x
//! x = input_layernorm(x)
//! x = self_attn(x, cos_sin, mask)
//! x = residual + x
//! residual = x
//! x = post_attention_layernorm(x)
//! x = mlp(x)
//! x = residual + x
//! ```
//!
//! # Weight names
//!
//! The [`VarBuilder`] passed to [`TransformerBlock::new`] is expected to hold:
//! - `self_attn.{q,k,v,o}_proj.weight` (and bias variants based on [`AttentionConfig`])
//! - `self_attn.{q,k}_norm.weight` (only when `cfg.use_qk_norm = true`)
//! - `mlp.{gate,up,down}_proj.weight`
//! - `input_layernorm.weight`
//! - `post_attention_layernorm.weight`
//!
//! These names match the safetensors checkpoint layout for Qwen3-TTS and Qwen2.5.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Activation, VarBuilder};

use crate::models::with_tracing::RmsNorm;

use super::attention::{AttentionConfig, GqaAttention};
use super::ffn::SwiGluFfn;

/// Pre-norm transformer decoder block combining attention and FFN.
///
/// See the [module documentation](self) for the full forward-pass description
/// and the expected weight-name layout.
///
/// Call [`TransformerBlock::clear_kv_cache`] between independent sequences.
#[derive(Debug)]
pub struct TransformerBlock {
    self_attn: GqaAttention,
    ffn: SwiGluFfn,
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
}

impl Clone for TransformerBlock {
    /// Clone the block with the KV cache reset to empty.
    ///
    /// Weights and norms are cloned as-is, but the accumulated KV cache is
    /// **not** carried over — the clone starts with a fresh cache. This
    /// prevents an accidental deep-copy of potentially large cached tensors.
    /// See also [`GqaAttention`]'s `Clone` implementation.
    fn clone(&self) -> Self {
        Self {
            self_attn: self.self_attn.clone(),
            ffn: self.ffn.clone(),
            input_norm: self.input_norm.clone(),
            post_attn_norm: self.post_attn_norm.clone(),
        }
    }
}

impl TransformerBlock {
    /// Create a new `TransformerBlock`.
    ///
    /// # Arguments
    ///
    /// * `cfg` — Attention configuration controlling dimensions, bias, `RoPE`, and
    ///   QK-norm. The norm epsilon (`cfg.norm_eps`) is also used for the block's
    ///   layer norms.
    /// * `intermediate_size` — Hidden dimension of the FFN intermediate layer.
    /// * `activation` — Activation function for the FFN gate branch.
    /// * `vb` — [`VarBuilder`] scoped to this block. Loads the weights listed in
    ///   the [module documentation](self).
    ///
    /// # Errors
    ///
    /// Returns a candle error if any weight-loading operation fails.
    // VarBuilder::pp() consumes self, so it must be taken by value.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        cfg: &AttentionConfig,
        intermediate_size: usize,
        activation: Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = GqaAttention::new(*cfg, vb.pp("self_attn"))?;
        let ffn = SwiGluFfn::new(cfg.dim, intermediate_size, activation, vb.pp("mlp"))?;
        let input_norm = RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("input_layernorm"))?;
        let post_attn_norm =
            RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("post_attention_layernorm"))?;
        Ok(Self {
            self_attn,
            ffn,
            input_norm,
            post_attn_norm,
        })
    }

    /// Run the pre-norm transformer block.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` — Input tensor of shape `[batch, seq_len, dim]`.
    /// * `cos_sin` — Pre-computed `RoPE` tables `(cos, sin)`. **Must be `Some` when
    ///   `cfg.use_rope = true`.**
    /// * `attention_mask` — Optional additive causal attention mask broadcastable to
    ///   `[batch, n_heads, q_seq_len, kv_seq_len]`.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, seq_len, dim]`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `cos_sin` is `None` but `cfg.use_rope = true`, or
    /// if any tensor operation fails.
    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        cos_sin: Option<(&Tensor, &Tensor)>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.input_norm.forward(hidden_states)?;
        let hidden_states = self
            .self_attn
            .forward(&hidden_states, cos_sin, attention_mask)?;
        let hidden_states = (residual + hidden_states)?;

        let residual = &hidden_states;
        let hidden_states = self.post_attn_norm.forward(&hidden_states)?;
        let hidden_states = self.ffn.forward(&hidden_states)?;
        residual + hidden_states
    }

    /// Clear the accumulated KV cache.
    ///
    /// Must be called between independent sequences to avoid stale context
    /// contaminating the next sequence.
    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    // dim=16, 4 heads, head_dim=4, intermediate_size=32, no bias/rope/qk_norm.
    fn base_cfg() -> AttentionConfig {
        AttentionConfig {
            dim: 16,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 4,
            qkv_bias: false,
            o_bias: false,
            use_rope: false,
            use_qk_norm: false,
            norm_eps: 1e-6,
        }
    }

    fn base_intermediate() -> usize {
        32
    }

    /// Build a `VarBuilder` with zero-valued weights for the given config and
    /// intermediate size. Norm weights are initialised to ones (RMSNorm identity).
    fn make_vb(cfg: &AttentionConfig, intermediate_size: usize) -> VarBuilder<'static> {
        let device = &Device::Cpu;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let q_out = cfg.n_heads * cfg.head_dim;
        let kv_out = cfg.n_kv_heads * cfg.head_dim;
        let dim = cfg.dim;

        let zeros =
            |shape: (usize, usize)| Tensor::zeros(shape, DType::F32, device).expect("zeros");
        let ones1 = |n: usize| Tensor::ones(n, DType::F32, device).expect("ones");

        // self_attn projections
        t.insert("self_attn.q_proj.weight".into(), zeros((q_out, dim)));
        t.insert("self_attn.k_proj.weight".into(), zeros((kv_out, dim)));
        t.insert("self_attn.v_proj.weight".into(), zeros((kv_out, dim)));
        t.insert("self_attn.o_proj.weight".into(), zeros((dim, q_out)));

        // mlp projections
        t.insert(
            "mlp.gate_proj.weight".into(),
            zeros((intermediate_size, dim)),
        );
        t.insert("mlp.up_proj.weight".into(), zeros((intermediate_size, dim)));
        t.insert(
            "mlp.down_proj.weight".into(),
            zeros((dim, intermediate_size)),
        );

        // layer norms (weight=1 → identity scale)
        t.insert("input_layernorm.weight".into(), ones1(dim));
        t.insert("post_attention_layernorm.weight".into(), ones1(dim));

        VarBuilder::from_tensors(t, DType::F32, device)
    }

    fn zeros_block(cfg: AttentionConfig, intermediate_size: usize) -> TransformerBlock {
        TransformerBlock::new(
            &cfg,
            intermediate_size,
            Activation::Silu,
            make_vb(&cfg, intermediate_size),
        )
        .expect("TransformerBlock::new")
    }

    fn varying_input(batch: usize, seq: usize, dim: usize) -> Tensor {
        let data: Vec<f32> = (0..batch * seq * dim)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        Tensor::from_vec(data, (batch, seq, dim), &Device::Cpu).expect("varying_input")
    }

    // -------------------------------------------------------------------------
    // Shape tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_output_shape() {
        let mut block = zeros_block(base_cfg(), base_intermediate());
        let x = Tensor::zeros((2, 5, 16), DType::F32, &Device::Cpu).expect("zeros");
        let y = block.forward(&x, None, None).expect("forward");
        assert_eq!(y.dims(), &[2, 5, 16]);
    }

    #[test]
    fn test_single_token_decode_shape() {
        let mut block = zeros_block(base_cfg(), base_intermediate());
        let x = Tensor::zeros((1, 1, 16), DType::F32, &Device::Cpu).expect("zeros");
        let y = block.forward(&x, None, None).expect("forward");
        assert_eq!(y.dims(), &[1, 1, 16]);
    }

    // -------------------------------------------------------------------------
    // Residual connection test
    // -------------------------------------------------------------------------

    #[test]
    fn test_zero_weights_preserve_residual() {
        // With zero projection weights the attention and FFN outputs are zero,
        // so the residual connections return the input unchanged.
        let mut block = zeros_block(base_cfg(), base_intermediate());
        let x = varying_input(1, 4, 16);
        let y = block.forward(&x, None, None).expect("forward");
        let diff: f32 = (&y - &x)
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            diff < 1e-5,
            "zero weights: residual must preserve input, max_diff={diff}"
        );
    }

    // -------------------------------------------------------------------------
    // KV cache tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_kv_cache_decode_step_shape() {
        // Prefill 4 tokens then decode 1; output must be [1, 1, dim].
        let mut block = zeros_block(base_cfg(), base_intermediate());
        let device = &Device::Cpu;
        let prefill = Tensor::zeros((1, 4, 16), DType::F32, device).expect("prefill");
        block
            .forward(&prefill, None, None)
            .expect("prefill forward");
        let token = Tensor::zeros((1, 1, 16), DType::F32, device).expect("token");
        let y = block.forward(&token, None, None).expect("decode forward");
        assert_eq!(y.dims(), &[1, 1, 16]);
    }

    #[test]
    fn test_clear_kv_cache_resets_to_fresh_state() {
        let mut block = zeros_block(base_cfg(), base_intermediate());
        let device = &Device::Cpu;
        let x = Tensor::ones((1, 3, 16), DType::F32, device).expect("ones");

        let y_fresh = block.forward(&x, None, None).expect("first forward");
        block.clear_kv_cache();
        let y_reset = block.forward(&x, None, None).expect("forward after clear");

        let diff: f32 = (&y_fresh - &y_reset)
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            diff < 1e-6,
            "output after clear_kv_cache must match fresh forward, max_diff={diff}"
        );
    }

    // -------------------------------------------------------------------------
    // RoPE handling tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rope_disabled_accepts_none_cos_sin() {
        let cfg = base_cfg(); // use_rope: false
        let mut block = zeros_block(cfg, base_intermediate());
        let x = Tensor::zeros((1, 3, 16), DType::F32, &Device::Cpu).expect("zeros");
        let y = block.forward(&x, None, None).expect("forward without rope");
        assert_eq!(y.dims(), &[1, 3, 16]);
    }

    #[test]
    fn test_rope_enabled_none_cos_sin_errors() {
        let cfg = AttentionConfig {
            use_rope: true,
            ..base_cfg()
        };
        let mut block = zeros_block(cfg, base_intermediate());
        let x = Tensor::zeros((1, 3, 16), DType::F32, &Device::Cpu).expect("zeros");
        assert!(
            block.forward(&x, None, None).is_err(),
            "use_rope=true with cos_sin=None must return an error"
        );
    }

    #[test]
    fn test_rope_enabled_output_shape() {
        let cfg = AttentionConfig {
            use_rope: true,
            ..base_cfg()
        };
        let mut block = zeros_block(cfg, base_intermediate());
        let x = Tensor::zeros((1, 3, 16), DType::F32, &Device::Cpu).expect("zeros");
        let half_dim = cfg.head_dim / 2;
        let cos = Tensor::ones((3, half_dim), DType::F32, &Device::Cpu).expect("cos");
        let sin = Tensor::zeros((3, half_dim), DType::F32, &Device::Cpu).expect("sin");
        let y = block
            .forward(&x, Some((&cos, &sin)), None)
            .expect("forward with rope");
        assert_eq!(y.dims(), &[1, 3, 16]);
    }
}
