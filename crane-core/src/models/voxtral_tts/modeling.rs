//! LLM backbone and audio codebook embedding for Voxtral TTS.
//!
//! Provides two public types:
//! - [`VoxtralLlm`] — 26-layer Mistral-style decoder with GQA and `RoPE`.
//! - [`AudioCodebookEmbedding`] — summed lookup over 37 parallel codebooks.
//!
//! The Voxtral checkpoint uses Mistral-style weight names that differ from the
//! `HuggingFace` convention expected by the shared [`TransformerBlock`]. Call
//! [`rename_voxtral_transformer_keys`] on the raw `HashMap<String, Tensor>`
//! before constructing a [`VarBuilder`] that is passed to [`VoxtralLlm::new`].

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, Activation, Embedding, VarBuilder};

use crate::models::modules::attention::AttentionConfig;
use crate::models::modules::rotary::RotaryEmbedding;
use crate::models::modules::transformer::TransformerBlock;
use crate::models::with_tracing::RmsNorm;

use super::model::VoxtralConfig;

// ── Key renaming ──────────────────────────────────────────────────────────────

/// Rename Mistral-style safetensors keys to HuggingFace-style keys expected
/// by the shared [`TransformerBlock`].
///
/// Only keys under `"layers."` (LLM backbone) and
/// `"acoustic_transformer.layers."` (flow-matching transformer) are modified.
/// Codec tensors (`"audio_tokenizer.*"`) and embedding tensors
/// (`"mm_audio_embeddings.*"`) are passed through unchanged.
///
/// | Original fragment | Renamed to |
/// |---|---|
/// | `.attention.wq.` | `.self_attn.q_proj.` |
/// | `.attention.wk.` | `.self_attn.k_proj.` |
/// | `.attention.wv.` | `.self_attn.v_proj.` |
/// | `.attention.wo.` | `.self_attn.o_proj.` |
/// | `.attention_norm.` | `.input_layernorm.` |
/// | `.feed_forward.w1.` | `.mlp.gate_proj.` |
/// | `.feed_forward.w2.` | `.mlp.down_proj.` |
/// | `.feed_forward.w3.` | `.mlp.up_proj.` |
/// | `.ffn_norm.` | `.post_attention_layernorm.` |
#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn rename_voxtral_transformer_keys(
    tensors: HashMap<String, Tensor>,
) -> HashMap<String, Tensor> {
    tensors
        .into_iter()
        .map(|(k, v)| {
            let new_k = if k.starts_with("layers.") || k.starts_with("acoustic_transformer.layers.")
            {
                k.replace(".attention.wq.", ".self_attn.q_proj.")
                    .replace(".attention.wk.", ".self_attn.k_proj.")
                    .replace(".attention.wv.", ".self_attn.v_proj.")
                    .replace(".attention.wo.", ".self_attn.o_proj.")
                    .replace(".attention_norm.", ".input_layernorm.")
                    .replace(".feed_forward.w1.", ".mlp.gate_proj.")
                    .replace(".feed_forward.w2.", ".mlp.down_proj.")
                    .replace(".feed_forward.w3.", ".mlp.up_proj.")
                    .replace(".ffn_norm.", ".post_attention_layernorm.")
            } else {
                k
            };
            (new_k, v)
        })
        .collect()
}

// ── Causal mask ───────────────────────────────────────────────────────────────

/// Build a square additive causal attention mask.
///
/// Returns shape `[1, 1, seq_len, seq_len]` in `dtype`. Allowed positions
/// (diagonal and lower triangle) are `0.0`; future positions (upper triangle)
/// are `f32::NEG_INFINITY` cast to `dtype`.
fn build_causal_mask(seq_len: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (1, 1, seq_len, seq_len), device)?.to_dtype(dtype)
}

// ── VoxtralLlm ────────────────────────────────────────────────────────────────

/// 26-layer Mistral-style LLM decoder backbone for Voxtral TTS.
///
/// Implements causal autoregressive decoding with grouped-query attention (GQA)
/// and rotary positional embeddings (`RoPE`, `theta=1e6`). The KV cache is
/// managed internally by each [`TransformerBlock`].
///
/// # Usage
///
/// 1. Call [`VoxtralLlm::embed_tokens`] to convert text/special token IDs to
///    embeddings, then mix in voice embeddings to build the full prompt.
/// 2. Call [`VoxtralLlm::forward`] with `start_pos=0` during prefill, then
///    with an incrementing `start_pos` for each decode step.
/// 3. Call [`VoxtralLlm::clear_kv_cache`] between independent sequences.
///
/// # Weight names
///
/// The `VarBuilder` must use HuggingFace-style names. Use
/// [`rename_voxtral_transformer_keys`] to convert the raw safetensors map.
///
/// Expected paths under `vb`:
/// - `mm_audio_embeddings.tok_embeddings.weight`
/// - `layers.{i}.self_attn.{q,k,v,o}_proj.weight`
/// - `layers.{i}.{input,post_attention}_layernorm.weight`
/// - `layers.{i}.mlp.{gate,up,down}_proj.weight`
/// - `norm.weight`
#[derive(Debug)]
pub struct VoxtralLlm {
    tok_embeddings: Embedding,
    layers: Vec<TransformerBlock>,
    norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
}

impl VoxtralLlm {
    /// Construct the LLM backbone.
    ///
    /// `vb` must contain `HuggingFace`-style weight names as documented above.
    /// `device` is used for the pre-computed `RoPE` tables.
    ///
    /// # Errors
    ///
    /// Returns a candle error if any weight is missing or has an unexpected shape.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(cfg: &VoxtralConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let rotary_emb =
            RotaryEmbedding::new(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, device)?;

        let tok_embeddings = embedding(
            cfg.vocab_size,
            cfg.dim,
            vb.pp("mm_audio_embeddings").pp("tok_embeddings"),
        )?;

        let attn_cfg = AttentionConfig {
            dim: cfg.dim,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            qkv_bias: false,
            o_bias: false,
            use_rope: true,
            use_qk_norm: false,
            norm_eps: cfg.norm_eps,
        };

        let layers = (0..cfg.n_layers)
            .map(|i| {
                TransformerBlock::new(
                    &attn_cfg,
                    cfg.hidden_dim,
                    Activation::Silu,
                    vb.pp(format!("layers.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let norm = RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("norm"))?;

        Ok(Self {
            tok_embeddings,
            layers,
            norm,
            rotary_emb,
        })
    }

    /// Embed token IDs into the LLM input space.
    ///
    /// `token_ids` may have any shape; each element is looked up in the token
    /// embedding table and the output has the same leading dims plus `[dim]`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the lookup fails.
    pub fn embed_tokens(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.tok_embeddings.forward(token_ids)
    }

    /// Run the LLM decoder on pre-computed input embeddings.
    ///
    /// `input_embeds` has shape `[batch, seq_len, dim]`. `start_pos` is the
    /// absolute sequence offset of the first token (0 during prefill,
    /// accumulated during autoregressive decode).
    ///
    /// Returns hidden states of shape `[batch, seq_len, dim]`.
    ///
    /// A causal attention mask is applied during prefill (`seq_len > 1`); no
    /// explicit mask is used for single-token decode steps because the KV cache
    /// already restricts visibility to past tokens.
    ///
    /// # Errors
    ///
    /// Returns a candle error if any layer or the final norm fails.
    pub fn forward(&mut self, input_embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_, seq_len, _) = input_embeds.dims3()?;

        // Multi-token decode with a non-zero start_pos is not supported: the
        // causal mask would be [seq_len × seq_len] but the KV cache makes the
        // actual key length cache_len + seq_len, causing a shape mismatch.
        debug_assert!(
            start_pos == 0 || seq_len == 1,
            "multi-token decode with non-zero start_pos is not supported \
             (start_pos={start_pos}, seq_len={seq_len})"
        );

        let (cos, sin) = self.rotary_emb.forward(start_pos, seq_len)?;

        let mask = if seq_len > 1 {
            Some(build_causal_mask(
                seq_len,
                input_embeds.dtype(),
                input_embeds.device(),
            )?)
        } else {
            None
        };

        let mut h = input_embeds.clone();
        for layer in &mut self.layers {
            h = layer.forward(&h, Some((&cos, &sin)), mask.as_ref())?;
        }

        self.norm.forward(&h)
    }

    /// Clear the KV cache in every transformer layer.
    ///
    /// Must be called between independent generation sequences.
    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

// ── AudioCodebookEmbedding ────────────────────────────────────────────────────

/// Summed audio codebook embedding over 37 parallel codebooks (1 semantic + 36 acoustic).
///
/// The weight is a single concatenated `[N, dim]` lookup table (N may be padded):
/// - Codebook 0 (semantic VQ): 8192 codes + 2 special tokens = 8194 entries
/// - Codebooks 1–36 (acoustic FSQ): 21 levels + 2 special tokens = 23 entries each
///
/// [`AudioCodebookEmbedding::forward`] adds per-codebook offsets to 37 local
/// code indices, retrieves the corresponding embeddings, and sums them to
/// produce a single `[dim]` vector for the next LLM input step.
///
/// # Weight name
///
/// `mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight`
#[derive(Debug)]
pub struct AudioCodebookEmbedding {
    /// Concatenated embedding table, shape `[total_entries, dim]`.
    embeddings: Tensor,
    /// Start offset in `embeddings` for each codebook (length = `n_codebooks`).
    offsets: Vec<u32>,
    /// Total number of codebooks (37 = 1 semantic + 36 acoustic).
    n_codebooks: usize,
}

impl AudioCodebookEmbedding {
    /// Load the embedding table from `vb`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the weight tensor is missing or has the wrong shape.
    ///
    /// # Panics
    ///
    /// Panics if a codebook offset exceeds `u32::MAX`, which cannot happen with
    /// the production config (max offset is 8999).
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(cfg: &VoxtralConfig, vb: VarBuilder) -> Result<Self> {
        let audio = &cfg.multimodal.audio_model_args;
        let n_acoustic = audio.n_acoustic_codebook; // 36
        let n_codebooks = 1 + n_acoustic; // 37

        // Codebook entry counts (code values + 2 special tokens each)
        let semantic_size = audio.semantic_codebook_size + 2; // 8194
        let acoustic_size = audio.acoustic_codebook_size + 2; // 23
        let total_entries = semantic_size + n_acoustic * acoustic_size; // 9022

        // Cumulative start offsets for each codebook in the concatenated table
        let mut offsets = Vec::with_capacity(n_codebooks);
        offsets.push(0u32);
        for i in 0..n_acoustic {
            offsets.push(
                u32::try_from(semantic_size + i * acoustic_size).expect("codebook offset fits u32"),
            );
        }

        let embeddings = vb
            .pp("mm_audio_embeddings")
            .pp("audio_codebook_embeddings")
            .pp("embeddings")
            .get_unchecked("weight")?;
        let dims = embeddings.dims2()?;
        if dims.0 < total_entries || dims.1 != cfg.dim {
            candle_core::bail!(
                "audio_codebook_embeddings shape mismatch: \
                 need at least [{total_entries}, {}], got [{}, {}]",
                cfg.dim,
                dims.0,
                dims.1,
            );
        }

        Ok(Self {
            embeddings,
            offsets,
            n_codebooks,
        })
    }

    /// Embed `codes` and sum across the codebook dimension.
    ///
    /// `codes` must be a 1-D tensor with `n_codebooks` (37) elements,
    /// convertible to `U32`. Each element is a local code index within its
    /// codebook (values starting at 0; `EMPTY_AUDIO=0`, `END_AUDIO=1`, then
    /// actual codes from 2 onwards).
    ///
    /// Returns a tensor of shape `[dim]`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `index_select` or `sum` fails.
    pub fn forward(&self, codes: &Tensor) -> Result<Tensor> {
        let device = codes.device();

        // Offset each local code into the global concatenated table.
        // `to_vec1()` is a synchronising device-to-host copy, but the vector
        // has only 37 elements so the overhead is negligible. We stay on the
        // CPU for the arithmetic to avoid u32 elementwise-add limitations on
        // some candle backends.
        let local: Vec<u32> = codes.to_dtype(DType::U32)?.to_vec1()?;
        let global: Vec<u32> = local
            .iter()
            .zip(self.offsets.iter())
            .map(|(c, o)| c + o)
            .collect();
        let indices = Tensor::from_vec(global, self.n_codebooks, device)?;

        self.embeddings.index_select(&indices, 0)?.sum(0)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    /// Minimal config for unit tests: tiny dimensions, 2 layers.
    /// Uses real codebook sizes so offset and shape arithmetic matches production.
    fn tiny_config() -> VoxtralConfig {
        let json = r#"{
            "dim": 16, "n_layers": 2, "head_dim": 4, "hidden_dim": 32,
            "n_heads": 4, "n_kv_heads": 4, "rope_theta": 10000.0,
            "norm_eps": 1e-5, "vocab_size": 64, "max_seq_len": 256,
            "tied_embeddings": true,
            "multimodal": {
                "bos_token_id": 1,
                "audio_model_args": {
                    "semantic_codebook_size": 8192, "acoustic_codebook_size": 21,
                    "n_acoustic_codebook": 36, "audio_token_id": 24,
                    "begin_audio_token_id": 25,
                    "acoustic_transformer_args": {
                        "dim": 16, "n_layers": 2, "head_dim": 4, "hidden_dim": 32,
                        "n_heads": 4, "n_kv_heads": 4
                    }
                },
                "audio_tokenizer_args": {
                    "sampling_rate": 24000, "pretransform_patch_size": 240,
                    "semantic_codebook_size": 8192, "semantic_dim": 256,
                    "acoustic_codebook_size": 21, "acoustic_dim": 36,
                    "dim": 1024, "hidden_dim": 4096, "head_dim": 128,
                    "n_heads": 8, "n_kv_heads": 8, "qk_norm_eps": 1e-6,
                    "norm_eps": 0.01, "layer_scale_init": 0.01,
                    "decoder_transformer_lengths_str": "2,2,2,2",
                    "decoder_convs_kernels_str": "3,4,4,4",
                    "decoder_convs_strides_str": "1,2,2,2",
                    "voice": {}
                }
            }
        }"#;
        serde_json::from_str(json).expect("tiny_config parse")
    }

    /// Build a VarBuilder with zero/one weights for a tiny LLM (HF-style names).
    fn llm_vb(cfg: &VoxtralConfig) -> VarBuilder<'static> {
        let device = &Device::Cpu;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let dim = cfg.dim;
        let q_out = cfg.n_heads * cfg.head_dim;
        let kv_out = cfg.n_kv_heads * cfg.head_dim;

        let zeros = |s: (usize, usize)| Tensor::zeros(s, DType::F32, device).expect("zeros");
        let ones1 = |n: usize| Tensor::ones(n, DType::F32, device).expect("ones");

        t.insert(
            "mm_audio_embeddings.tok_embeddings.weight".into(),
            zeros((cfg.vocab_size, dim)),
        );
        for i in 0..cfg.n_layers {
            let p = format!("layers.{i}.");
            t.insert(format!("{p}self_attn.q_proj.weight"), zeros((q_out, dim)));
            t.insert(format!("{p}self_attn.k_proj.weight"), zeros((kv_out, dim)));
            t.insert(format!("{p}self_attn.v_proj.weight"), zeros((kv_out, dim)));
            t.insert(format!("{p}self_attn.o_proj.weight"), zeros((dim, q_out)));
            t.insert(format!("{p}input_layernorm.weight"), ones1(dim));
            t.insert(format!("{p}post_attention_layernorm.weight"), ones1(dim));
            t.insert(
                format!("{p}mlp.gate_proj.weight"),
                zeros((cfg.hidden_dim, dim)),
            );
            t.insert(
                format!("{p}mlp.up_proj.weight"),
                zeros((cfg.hidden_dim, dim)),
            );
            t.insert(
                format!("{p}mlp.down_proj.weight"),
                zeros((dim, cfg.hidden_dim)),
            );
        }
        t.insert("norm.weight".into(), ones1(dim));

        VarBuilder::from_tensors(t, DType::F32, device)
    }

    /// Build a VarBuilder for AudioCodebookEmbedding with a zero dummy tensor.
    fn codebook_vb(cfg: &VoxtralConfig) -> VarBuilder<'static> {
        let device = &Device::Cpu;
        let audio = &cfg.multimodal.audio_model_args;
        let semantic_size = audio.semantic_codebook_size + 2;
        let acoustic_size = audio.acoustic_codebook_size + 2;
        let total = semantic_size + audio.n_acoustic_codebook * acoustic_size;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert(
            "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight".into(),
            Tensor::zeros((total, cfg.dim), DType::F32, device).expect("zeros"),
        );
        VarBuilder::from_tensors(t, DType::F32, device)
    }

    // ── Rename tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_rename_llm_layer_keys() {
        fn check(orig: &str, expected: &str) {
            let mut m: HashMap<String, Tensor> = HashMap::new();
            m.insert(
                orig.to_string(),
                Tensor::zeros(1usize, DType::F32, &Device::Cpu).expect("zeros"),
            );
            let renamed = rename_voxtral_transformer_keys(m);
            assert!(
                renamed.contains_key(expected),
                "rename {orig:?} -> expected {expected:?}, got: {:?}",
                renamed.keys().collect::<Vec<_>>(),
            );
        }

        check(
            "layers.0.attention.wq.weight",
            "layers.0.self_attn.q_proj.weight",
        );
        check(
            "layers.5.attention.wk.weight",
            "layers.5.self_attn.k_proj.weight",
        );
        check(
            "layers.0.attention.wv.weight",
            "layers.0.self_attn.v_proj.weight",
        );
        check(
            "layers.0.attention.wo.weight",
            "layers.0.self_attn.o_proj.weight",
        );
        check(
            "layers.0.attention_norm.weight",
            "layers.0.input_layernorm.weight",
        );
        check(
            "layers.0.feed_forward.w1.weight",
            "layers.0.mlp.gate_proj.weight",
        );
        check(
            "layers.0.feed_forward.w2.weight",
            "layers.0.mlp.down_proj.weight",
        );
        check(
            "layers.0.feed_forward.w3.weight",
            "layers.0.mlp.up_proj.weight",
        );
        check(
            "layers.0.ffn_norm.weight",
            "layers.0.post_attention_layernorm.weight",
        );
    }

    #[test]
    fn test_rename_acoustic_transformer_keys() {
        let orig = "acoustic_transformer.layers.0.attention.wq.weight";
        let expected = "acoustic_transformer.layers.0.self_attn.q_proj.weight";
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert(
            orig.to_string(),
            Tensor::zeros(1usize, DType::F32, &Device::Cpu).expect("zeros"),
        );
        let renamed = rename_voxtral_transformer_keys(m);
        assert!(
            renamed.contains_key(expected),
            "acoustic transformer rename failed; got: {:?}",
            renamed.keys().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_rename_leaves_codec_keys_unchanged() {
        let key = "audio_tokenizer.decoder_blocks.1.layers.0.attention.wq.weight";
        let mut m: HashMap<String, Tensor> = HashMap::new();
        m.insert(
            key.to_string(),
            Tensor::zeros(1usize, DType::F32, &Device::Cpu).expect("zeros"),
        );
        let renamed = rename_voxtral_transformer_keys(m);
        assert!(
            renamed.contains_key(key),
            "codec key must not be renamed; got: {:?}",
            renamed.keys().collect::<Vec<_>>(),
        );
    }

    // ── Causal mask tests ─────────────────────────────────────────────────────

    #[test]
    fn test_causal_mask_shape() {
        let mask = build_causal_mask(4, DType::F32, &Device::Cpu).expect("mask");
        assert_eq!(mask.dims(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_causal_mask_values() {
        // Squeeze to [3, 3] and verify upper triangle is -inf, rest is 0.
        let mask = build_causal_mask(3, DType::F32, &Device::Cpu)
            .expect("mask")
            .squeeze(0)
            .expect("sq0")
            .squeeze(0)
            .expect("sq1");
        let vals: Vec<f32> = mask
            .to_vec2::<f32>()
            .expect("vec2")
            .into_iter()
            .flatten()
            .collect();
        // Row 0: [0, -inf, -inf]
        assert_eq!(vals[0], 0.0);
        assert!(vals[1].is_infinite() && vals[1] < 0.0);
        assert!(vals[2].is_infinite() && vals[2] < 0.0);
        // Row 1: [0, 0, -inf]
        assert_eq!(vals[3], 0.0);
        assert_eq!(vals[4], 0.0);
        assert!(vals[5].is_infinite() && vals[5] < 0.0);
        // Row 2: [0, 0, 0]
        assert_eq!(vals[6], 0.0);
        assert_eq!(vals[7], 0.0);
        assert_eq!(vals[8], 0.0);
    }

    // ── Codebook offset tests ─────────────────────────────────────────────────

    #[test]
    fn test_codebook_offsets() {
        let cfg = tiny_config();
        let emb = AudioCodebookEmbedding::new(&cfg, codebook_vb(&cfg)).expect("emb");
        let audio = &cfg.multimodal.audio_model_args;
        let n_codebooks = 1 + audio.n_acoustic_codebook; // 37
        let semantic_size = audio.semantic_codebook_size + 2; // 8194
        let acoustic_size = audio.acoustic_codebook_size + 2; // 23

        assert_eq!(emb.offsets.len(), n_codebooks);
        assert_eq!(emb.offsets[0], 0, "semantic starts at 0");
        assert_eq!(
            emb.offsets[1], semantic_size as u32,
            "first acoustic starts after semantic"
        );
        assert_eq!(
            emb.offsets[2],
            (semantic_size + acoustic_size) as u32,
            "second acoustic"
        );
        assert_eq!(
            *emb.offsets.last().expect("last"),
            (semantic_size + (n_codebooks - 2) * acoustic_size) as u32,
            "last acoustic codebook offset"
        );
    }

    // ── Codebook forward shape test ───────────────────────────────────────────

    #[test]
    fn test_codebook_forward_shape() {
        let cfg = tiny_config();
        let emb = AudioCodebookEmbedding::new(&cfg, codebook_vb(&cfg)).expect("emb");
        let codes = Tensor::zeros(emb.n_codebooks, DType::U32, &Device::Cpu).expect("codes");
        let out = emb.forward(&codes).expect("forward");
        assert_eq!(out.dims(), &[cfg.dim]);
    }

    // ── LLM forward shape tests ───────────────────────────────────────────────

    #[test]
    fn test_llm_prefill_shape() {
        let cfg = tiny_config();
        let mut llm = VoxtralLlm::new(&cfg, llm_vb(&cfg), &Device::Cpu).expect("llm");
        let input = Tensor::zeros((1, 5, cfg.dim), DType::F32, &Device::Cpu).expect("input");
        let out = llm.forward(&input, 0).expect("forward");
        assert_eq!(out.dims(), &[1, 5, cfg.dim]);
    }

    #[test]
    fn test_llm_decode_step_shape() {
        let cfg = tiny_config();
        let mut llm = VoxtralLlm::new(&cfg, llm_vb(&cfg), &Device::Cpu).expect("llm");

        // Prefill 5 tokens
        let prefill = Tensor::zeros((1, 5, cfg.dim), DType::F32, &Device::Cpu).expect("prefill");
        llm.forward(&prefill, 0).expect("prefill");

        // Single decode step
        let step = Tensor::zeros((1, 1, cfg.dim), DType::F32, &Device::Cpu).expect("step");
        let out = llm.forward(&step, 5).expect("decode");
        assert_eq!(out.dims(), &[1, 1, cfg.dim]);
    }

    #[test]
    fn test_llm_clear_kv_cache() {
        let cfg = tiny_config();
        let mut llm = VoxtralLlm::new(&cfg, llm_vb(&cfg), &Device::Cpu).expect("llm");
        let input = Tensor::zeros((1, 3, cfg.dim), DType::F32, &Device::Cpu).expect("input");

        let y1 = llm.forward(&input, 0).expect("first");
        llm.clear_kv_cache();
        let y2 = llm.forward(&input, 0).expect("second");

        // With zero weights both outputs are identical; just check shape.
        assert_eq!(y1.dims(), y2.dims());
    }

    // ── Integration tests (require local checkpoint) ──────────────────────────

    fn checkpoint_path() -> Option<std::path::PathBuf> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .join("checkpoints/Voxtral-4B-TTS-2603");
        p.is_dir().then_some(p)
    }

    #[test]
    #[ignore = "requires local checkpoint at checkpoints/Voxtral-4B-TTS-2603"]
    fn test_llm_load_real() {
        use super::super::model::VoxtralConfig;
        let Some(cp) = checkpoint_path() else {
            return;
        };
        let cfg = VoxtralConfig::from_model_dir(&cp).expect("config");
        let raw =
            candle_core::safetensors::load(&cp.join("consolidated.safetensors"), &Device::Cpu)
                .expect("load safetensors");
        let renamed = rename_voxtral_transformer_keys(raw);
        let vb = VarBuilder::from_tensors(renamed, DType::BF16, &Device::Cpu);
        VoxtralLlm::new(&cfg, vb, &Device::Cpu).expect("VoxtralLlm::new");
    }

    #[test]
    #[ignore = "requires local checkpoint at checkpoints/Voxtral-4B-TTS-2603"]
    fn test_codebook_load_real() {
        use super::super::model::VoxtralConfig;
        let Some(cp) = checkpoint_path() else {
            return;
        };
        let cfg = VoxtralConfig::from_model_dir(&cp).expect("config");
        let raw =
            candle_core::safetensors::load(&cp.join("consolidated.safetensors"), &Device::Cpu)
                .expect("load safetensors");
        let vb = VarBuilder::from_tensors(raw, DType::BF16, &Device::Cpu);
        let emb = AudioCodebookEmbedding::new(&cfg, vb).expect("AudioCodebookEmbedding::new");
        assert_eq!(emb.embeddings.dims()[1], cfg.dim);
    }
}
