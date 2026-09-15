//! The Qwen2 decoder backbone (`model.language_model.*` in the checkpoint —
//! `decoder_config` in `config.json`, `model_type: "qwen2"`).
//!
//! A thin wrapper around the shared [`TransformerBlock`]/[`RotaryEmbedding`]
//! stack (the same building blocks `qwen25::qwen2` and `qwen3_tts` use) —
//! see that module's doc comment for the weight-name contract. The one
//! difference from `qwen25::qwen2::Model` this wrapper needs:
//! [`KugelAudioDecoder::forward_embeds`] takes `inputs_embeds` directly and
//! returns hidden states for **every** position, not just the last one —
//! KugelAudio splices [`super::connector::SpeechConnector`] output into the
//! input embeddings at specific positions (text tokens still go through
//! [`KugelAudioDecoder::embed_tokens`]) and needs the hidden state at every
//! speech position (as the diffusion head's condition), not only the final
//! token's logits. `lm_head` lives one level up (`model.rs`) since
//! `lm_head.weight` sits at the checkpoint's root, a sibling of `model.*`,
//! not nested under `language_model.*`.
//!
//! No sliding-window attention (unlike `qwen25::qwen2::Model`, which
//! implements it for Qwen2's smaller variants): the reference checkpoint's
//! `decoder_config` has `use_sliding_window: false`, `sliding_window: null`,
//! so this wrapper only ever builds a plain causal mask.

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::models::modules::attention::{AttentionConfig, RopeMode};
use crate::models::modules::rotary::RotaryEmbedding;
use crate::models::modules::transformer::TransformerBlock;
use crate::models::with_tracing::RmsNorm;

use super::config::DecoderConfig;

/// `Clone` resets every layer's KV cache (see [`TransformerBlock`]'s `Clone`
/// impl) while sharing the underlying weight tensors (`Tensor` is
/// `Arc`-backed — cloning bumps a refcount, it doesn't copy ~7B parameters).
/// [`super::model::KugelAudioModel::generate`] uses this to run a second,
/// independently-cached "negative" (CFG-unconditional) decoder stream
/// without doubling memory.
#[derive(Clone)]
pub struct KugelAudioDecoder {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<TransformerBlock>,
    norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    device: Device,
    dtype: DType,
}

impl KugelAudioDecoder {
    pub fn new(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let head_dim = cfg.head_dim();
        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.device(),
        )?;
        let attn_cfg = AttentionConfig {
            dim: cfg.hidden_size,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim,
            // Qwen2 (unlike Qwen2.5's smaller variants and Qwen3) keeps
            // biased Q/K/V projections and no output bias.
            qkv_bias: true,
            o_bias: false,
            rope_mode: RopeMode::HalfSplit,
            use_qk_norm: false,
            norm_eps: cfg.rms_norm_eps,
        };
        let vb_l = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(TransformerBlock::new(
                &attn_cfg,
                cfg.intermediate_size,
                cfg.hidden_act,
                vb_l.pp(i),
            )?);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Text-token embedding lookup — the half of KugelAudio's input
    /// embedding that isn't overwritten by spliced-in speech features.
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    fn causal_mask(&self, b_size: usize, tgt_len: usize, seqlen_offset: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    /// Run the decoder on pre-built input embeddings (`[batch, seq_len,
    /// hidden_size]`), returning hidden states at every position (post
    /// final norm, pre `lm_head`) — `[batch, seq_len, hidden_size]`.
    /// `seqlen_offset` is the number of already-cached positions (0 for a
    /// fresh prefill).
    pub fn forward_embeds(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.causal_mask(b_size, seq_len, seqlen_offset)?)
        };
        let (cos, sin) = self.rotary_emb.forward(seqlen_offset, seq_len)?;
        let cos = cos.to_dtype(self.dtype)?;
        let sin = sin.to_dtype(self.dtype)?;
        let mut xs = inputs_embeds.clone();
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, Some((&cos, &sin)), attention_mask.as_ref())?;
        }
        xs.apply(&self.norm)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::Activation;
    use std::collections::HashMap;

    fn small_cfg() -> DecoderConfig {
        DecoderConfig {
            vocab_size: 50,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
        }
    }

    /// Populate every tensor [`KugelAudioDecoder::new`] needs, keyed by exact
    /// weight name (see [`super::super::modules::transformer`]'s doc
    /// comment for the shared layout Qwen2's biased-QKV/no-bias-O attention
    /// follows here).
    fn make_vb(cfg: &DecoderConfig, dtype: DType, device: &Device) -> VarBuilder<'static> {
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let fill = |shape: &[usize]| -> Tensor {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| 0.02 * ((i % 13) as f32 - 6.0)).collect();
            Tensor::from_vec(data, shape, device).unwrap()
        };
        let ones = |n: usize| Tensor::ones(n, DType::F32, device).unwrap();
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;

        t.insert("embed_tokens.weight".into(), fill(&[cfg.vocab_size, h]));
        for i in 0..cfg.num_hidden_layers {
            let p = format!("layers.{i}");
            t.insert(format!("{p}.self_attn.q_proj.weight"), fill(&[q_dim, h]));
            t.insert(format!("{p}.self_attn.q_proj.bias"), fill(&[q_dim]));
            t.insert(format!("{p}.self_attn.k_proj.weight"), fill(&[kv_dim, h]));
            t.insert(format!("{p}.self_attn.k_proj.bias"), fill(&[kv_dim]));
            t.insert(format!("{p}.self_attn.v_proj.weight"), fill(&[kv_dim, h]));
            t.insert(format!("{p}.self_attn.v_proj.bias"), fill(&[kv_dim]));
            t.insert(format!("{p}.self_attn.o_proj.weight"), fill(&[h, q_dim]));
            t.insert(format!("{p}.mlp.gate_proj.weight"), fill(&[cfg.intermediate_size, h]));
            t.insert(format!("{p}.mlp.up_proj.weight"), fill(&[cfg.intermediate_size, h]));
            t.insert(format!("{p}.mlp.down_proj.weight"), fill(&[h, cfg.intermediate_size]));
            t.insert(format!("{p}.input_layernorm.weight"), ones(h));
            t.insert(format!("{p}.post_attention_layernorm.weight"), ones(h));
        }
        t.insert("norm.weight".into(), ones(h));

        VarBuilder::from_tensors(t, dtype, device)
    }

    /// Prefill (4 tokens) then a decode step (1 token, `seqlen_offset = 4`)
    /// through a small synthetic decoder — checks shapes and finiteness of
    /// the GQA + biased-QKV attention, RoPE, and causal mask this wrapper
    /// builds around the shared [`TransformerBlock`].
    fn prefill_then_decode(device: &Device, dtype: DType) {
        let cfg = small_cfg();
        let vb = make_vb(&cfg, dtype, device);
        let mut decoder = KugelAudioDecoder::new(&cfg, vb).expect("build decoder");

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), device).unwrap();
        let embeds = decoder
            .embed_tokens(&input_ids)
            .expect("embed_tokens")
            .to_dtype(dtype)
            .unwrap();
        assert_eq!(embeds.dims(), &[1, 4, cfg.hidden_size]);

        let hidden = decoder.forward_embeds(&embeds, 0).expect("prefill");
        assert_eq!(hidden.dims(), &[1, 4, cfg.hidden_size]);
        let max_abs: f32 = hidden
            .to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(max_abs.is_finite());

        let next_id = Tensor::from_vec(vec![5u32], (1, 1), device).unwrap();
        let next_embed = decoder
            .embed_tokens(&next_id)
            .expect("embed next token")
            .to_dtype(dtype)
            .unwrap();
        let decode_hidden = decoder
            .forward_embeds(&next_embed, 4)
            .expect("decode step");
        assert_eq!(decode_hidden.dims(), &[1, 1, cfg.hidden_size]);
        let decode_max_abs: f32 = decode_hidden
            .to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(decode_max_abs.is_finite());
    }

    #[test]
    fn prefill_then_decode_shapes_and_finite_on_cpu() {
        prefill_then_decode(&Device::Cpu, DType::F32);
    }

    /// Same prefill/decode check on Metal with F16 — the dtype/device this
    /// crate actually picks for KugelAudio on macOS (see
    /// `kugelaudio_simple.rs`/`kugelaudio_load.rs`). Skipped where Metal
    /// isn't available (e.g. non-macOS CI).
    #[test]
    fn prefill_then_decode_shapes_and_finite_on_metal() {
        if !candle_core::utils::metal_is_available() {
            return;
        }
        let device = Device::new_metal(0).expect("metal device");
        prefill_then_decode(&device, DType::F16);
    }
}
