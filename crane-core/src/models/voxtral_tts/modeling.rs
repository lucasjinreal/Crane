//! LLM backbone, audio codebook embedding, and acoustic transformer for Voxtral TTS.
//!
//! Provides three public types:
//! - [`VoxtralLlm`] — 26-layer Mistral-style decoder with GQA and `RoPE`.
//! - [`AudioCodebookEmbedding`] — summed lookup over 37 parallel codebooks.
//! - [`AcousticTransformer`] — 3-layer bidirectional transformer with flow matching.
//!
//! The Voxtral checkpoint uses Mistral-style weight names that differ from the
//! `HuggingFace` convention expected by the shared [`TransformerBlock`]. Call
//! [`rename_voxtral_transformer_keys`] on the raw `HashMap<String, Tensor>`
//! before constructing a [`VarBuilder`] that is passed to [`VoxtralLlm::new`]
//! and [`AcousticTransformer::new`].

use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Activation, Embedding, Linear, VarBuilder};

use crate::models::modules::attention::{AttentionConfig, RopeMode};
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
    const RENAMES: &[(&str, &str)] = &[
        (".attention.wq.", ".self_attn.q_proj."),
        (".attention.wk.", ".self_attn.k_proj."),
        (".attention.wv.", ".self_attn.v_proj."),
        (".attention.wo.", ".self_attn.o_proj."),
        (".attention_norm.", ".input_layernorm."),
        (".feed_forward.w1.", ".mlp.gate_proj."),
        (".feed_forward.w2.", ".mlp.down_proj."),
        (".feed_forward.w3.", ".mlp.up_proj."),
        (".ffn_norm.", ".post_attention_layernorm."),
    ];
    tensors
        .into_iter()
        .map(|(k, v)| {
            let new_k = if k.starts_with("layers.") || k.starts_with("acoustic_transformer.layers.")
            {
                RENAMES
                    .iter()
                    .find_map(|(old, new)| k.contains(old).then(|| k.replace(old, new)))
                    .unwrap_or(k)
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
            rope_mode: RopeMode::Interleaved,
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
            .map(|(c, o)| {
                debug_assert!(
                    c.checked_add(*o).is_some(),
                    "codebook code {c} + offset {o} overflows u32"
                );
                c + o
            })
            .collect();
        let indices = Tensor::from_vec(global, self.n_codebooks, device)?;

        self.embeddings.index_select(&indices, 0)?.sum(0)
    }
}

// ── AcousticTransformer ───────────────────────────────────────────────────────

/// Padded semantic head output dimension (8192 + 2 specials, rounded to 128).
const SEMANTIC_PADDED: usize = 8320;
/// Classifier-free guidance scale for flow matching.
const CFG_ALPHA: f32 = 1.2;
/// Number of Euler integration timesteps (7 intervals between 8 points).
const FLOW_INTERVALS: u8 = 7;
/// Base frequency for sinusoidal time embeddings.
const TIME_EMB_THETA: f32 = 10_000.0;

/// 3-layer bidirectional acoustic transformer with flow-matching inference.
///
/// Per audio frame, it receives the LLM hidden state `h ∈ R^{dim}` and produces:
/// - 1 semantic code via [`AcousticTransformer::predict_semantic_code`]
/// - 36 acoustic codes via [`AcousticTransformer::flow_match_inference`]
///
/// The transformer is **bidirectional** (no `RoPE`, no causal mask). Each
/// velocity prediction uses a fresh 3-token sequence (noise, timestep, LLM
/// conditioning), so the KV cache is cleared before every forward pass.
///
/// # Weight names
///
/// The `VarBuilder` must use `HuggingFace`-style names after
/// [`rename_voxtral_transformer_keys`]. Expected paths under `vb`:
/// - `acoustic_transformer.input_projection.weight`
/// - `acoustic_transformer.time_projection.weight`
/// - `acoustic_transformer.llm_projection.weight`
/// - `acoustic_transformer.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
/// - `acoustic_transformer.layers.{i}.{input,post_attention}_layernorm.weight`
/// - `acoustic_transformer.layers.{i}.mlp.{gate,up,down}_proj.weight`
/// - `acoustic_transformer.norm.weight`
/// - `acoustic_transformer.semantic_codebook_output.weight`
/// - `acoustic_transformer.acoustic_codebook_output.weight`
#[derive(Debug)]
pub struct AcousticTransformer {
    layers: Vec<TransformerBlock>,
    input_projection: Linear,
    time_projection: Linear,
    llm_projection: Linear,
    norm: RmsNorm,
    /// Semantic output head: `[dim] -> [SEMANTIC_PADDED]`, no bias.
    semantic_head: Linear,
    /// Acoustic velocity head: `[dim] -> [n_acoustic]`, no bias.
    acoustic_head: Linear,
    /// Precomputed `inv_freq[i] = TIME_EMB_THETA^(-i / half_dim)`.
    time_inv_freq: Vec<f32>,
    /// Number of acoustic codebooks (36).
    pub n_acoustic: usize,
    /// First index past the valid semantic range (8194 = 8192 codes + 2 specials).
    pub semantic_valid_end: usize,
    /// Hidden dimension (3072).
    pub dim: usize,
}

impl AcousticTransformer {
    /// Construct the acoustic transformer.
    ///
    /// `vb` must contain `HuggingFace`-style weight names as documented above.
    ///
    /// # Errors
    ///
    /// Returns a candle error if any weight is missing or has an unexpected shape.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(cfg: &VoxtralConfig, vb: VarBuilder) -> Result<Self> {
        let ac = &cfg.multimodal.audio_model_args.acoustic_transformer_args;
        let audio = &cfg.multimodal.audio_model_args;

        let attn_cfg = AttentionConfig {
            dim: ac.dim,
            n_heads: ac.n_heads,
            n_kv_heads: ac.n_kv_heads,
            head_dim: ac.head_dim,
            qkv_bias: false,
            o_bias: false,
            rope_mode: RopeMode::None,
            use_qk_norm: false,
            norm_eps: cfg.norm_eps,
        };

        let ac_vb = vb.pp("acoustic_transformer");

        let layers = (0..ac.n_layers)
            .map(|i| {
                TransformerBlock::new(
                    &attn_cfg,
                    ac.hidden_dim,
                    Activation::Silu,
                    ac_vb.pp("layers").pp(i),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let input_projection = linear_no_bias(
            audio.n_acoustic_codebook,
            ac.dim,
            ac_vb.pp("input_projection"),
        )?;
        let time_projection = linear_no_bias(ac.dim, ac.dim, ac_vb.pp("time_projection"))?;
        let llm_projection = linear_no_bias(ac.dim, ac.dim, ac_vb.pp("llm_projection"))?;
        let norm = RmsNorm::new(ac.dim, cfg.norm_eps, ac_vb.pp("norm"))?;
        let semantic_head = linear_no_bias(
            ac.dim,
            SEMANTIC_PADDED,
            ac_vb.pp("semantic_codebook_output"),
        )?;
        let acoustic_head = linear_no_bias(
            ac.dim,
            audio.n_acoustic_codebook,
            ac_vb.pp("acoustic_codebook_output"),
        )?;

        // Precompute sinusoidal inverse frequencies (half the embedding dim)
        let half_dim = ac.dim / 2;
        let log_theta = TIME_EMB_THETA.ln();
        #[allow(clippy::cast_precision_loss)]
        let time_inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| (-log_theta * i as f32 / half_dim as f32).exp())
            .collect();

        let semantic_valid_end = audio.semantic_codebook_size + 2;

        Ok(Self {
            layers,
            input_projection,
            time_projection,
            llm_projection,
            norm,
            semantic_head,
            acoustic_head,
            time_inv_freq,
            n_acoustic: audio.n_acoustic_codebook,
            semantic_valid_end,
            dim: ac.dim,
        })
    }

    /// Predict the semantic code from LLM hidden state `h`.
    ///
    /// `llm_hidden` has shape `[dim]`. Returns a code in `[1, semantic_valid_end)`.
    /// `EMPTY_AUDIO` (index 0) and padding indices `≥ semantic_valid_end` are
    /// masked to `-1e30` before greedy argmax. Returns `END_AUDIO_CODE` (1) on
    /// the (impossible) empty-iterator edge case.
    ///
    /// # Panics
    ///
    /// Panics if the argmax index cannot be represented as `u32`, which cannot
    /// happen in practice because the vocabulary size is well below `u32::MAX`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the linear forward or device-to-host copy fails.
    pub fn predict_semantic_code(&self, llm_hidden: &Tensor) -> Result<u32> {
        // semantic_head expects 2-D input
        let logits = self
            .semantic_head
            .forward(&llm_hidden.unsqueeze(0)?)?
            .squeeze(0)?; // [SEMANTIC_PADDED]
        let mut logits: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;

        // Mask EMPTY_AUDIO (index 0) — generation must never produce it
        logits[0] = f32::NEG_INFINITY;
        // Mask padding beyond the valid semantic range
        for v in &mut logits[self.semantic_valid_end..] {
            *v = f32::NEG_INFINITY;
        }

        let code = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(super::model::END_AUDIO_CODE, |(i, _)| {
                u32::try_from(i).expect("semantic code index fits u32")
            });

        Ok(code)
    }

    /// Compute a sinusoidal time embedding for scalar timestep `t`.
    ///
    /// Returns a 1-D tensor of shape `[dim]`:
    /// `[cos(t·f₀), …, cos(t·f_{H-1}), sin(t·f₀), …, sin(t·f_{H-1})]`
    /// where `H = dim/2` and `fᵢ = TIME_EMB_THETA^(-i/H)`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if tensor construction fails.
    fn compute_time_embedding(&self, t: f32, device: &Device) -> Result<Tensor> {
        let half_dim = self.dim / 2;
        let mut emb = vec![0f32; self.dim];
        for i in 0..half_dim {
            let angle = t * self.time_inv_freq[i];
            emb[i] = angle.cos();
            emb[half_dim + i] = angle.sin();
        }
        Tensor::from_vec(emb, self.dim, device)
    }

    /// Run flow-matching inference: 7 Euler steps with CFG (`alpha=1.2`).
    ///
    /// Starting from Gaussian noise `x ~ N(0,1) ∈ R^{n_acoustic}`, integrates
    /// the ODE conditioned on `llm_hidden` via classifier-free guidance.
    ///
    /// `llm_hidden` has shape `[dim]`. Returns a `U32` tensor of shape
    /// `[n_acoustic]` with FSQ codes in `[2, 22]` (raw code `[0, 20]` plus
    /// the 2-slot special-token offset used by [`AudioCodebookEmbedding`]).
    ///
    /// # Errors
    ///
    /// Returns a candle error if any tensor operation fails.
    pub fn flow_match_inference(&mut self, llm_hidden: &Tensor) -> Result<Tensor> {
        let dtype = llm_hidden.dtype();
        let device = llm_hidden.device();

        let mut x = Tensor::randn(0f32, 1f32, self.n_acoustic, device)?.to_dtype(dtype)?;

        let dt = 1.0f32 / f32::from(FLOW_INTERVALS);

        // Hoist loop-invariant projections: llm_hidden is the same for all 7
        // steps, and llm_projection has no bias so projection(zeros) = zeros.
        // Note: tok0 and tok1 still vary per step, so h_uncond itself is NOT
        // constant — only tok2_uncond is.
        debug_assert!(
            self.llm_projection.bias().is_none(),
            "tok2_uncond shortcut requires llm_projection to have no bias"
        );
        let tok2_cond = self
            .llm_projection
            .forward(&llm_hidden.unsqueeze(0)?)?
            .squeeze(0)?;
        let tok2_uncond = Tensor::zeros(self.dim, dtype, device)?;

        // Precompute time tokens for all 7 steps (fixed schedule).
        let mut time_tokens = Vec::with_capacity(FLOW_INTERVALS as usize);
        for step in 0u8..FLOW_INTERVALS {
            let t = f32::from(step) * dt;
            let emb = self.compute_time_embedding(t, device)?.to_dtype(dtype)?;
            let tok = self
                .time_projection
                .forward(&emb.unsqueeze(0)?)?
                .squeeze(0)?;
            time_tokens.push(tok);
        }

        for tok1 in &time_tokens {
            for layer in &mut self.layers {
                layer.clear_kv_cache();
            }

            let tok0 = self
                .input_projection
                .forward(&x.unsqueeze(0)?)?
                .squeeze(0)?;
            let h_cond = Tensor::stack(&[&tok0, tok1, &tok2_cond], 0)?;
            let h_uncond = Tensor::stack(&[&tok0, tok1, &tok2_uncond], 0)?;
            let mut h = Tensor::stack(&[&h_cond, &h_uncond], 0)?;

            for layer in &mut self.layers {
                h = layer.forward(&h, None, None)?;
            }

            let pos0 = h.i((.., 0, ..))?; // [2, dim]
            let normed = self.norm.forward(&pos0)?;
            let out = self.acoustic_head.forward(&normed)?; // [2, n_acoustic]
            let v_cond = out.i(0)?;
            let v_uncond = out.i(1)?;

            let v = (v_cond.affine(f64::from(CFG_ALPHA), 0.0)?
                + v_uncond.affine(f64::from(1.0 - CFG_ALPHA), 0.0)?)?;
            x = (x + v.affine(f64::from(dt), 0.0)?)?;
        }

        // FSQ quantize: clamp to [-1,1], scale to [0,20], round to integer, add +2 offset
        let x_f32 = x.to_dtype(DType::F32)?.clamp(-1f32, 1f32)?;
        let codes_f32 = x_f32.affine(10.0, 10.0)?.round()?.affine(1.0, 2.0)?;

        codes_f32.to_dtype(DType::U32)
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

    // ── Acoustic transformer helpers ──────────────────────────────────────────

    /// Build a `VarBuilder` with zero/one dummy weights for `AcousticTransformer`.
    ///
    /// Uses `HuggingFace`-style names (post-rename) so that
    /// `AcousticTransformer::new` can load them without running
    /// `rename_voxtral_transformer_keys`.
    fn acoustic_vb(cfg: &VoxtralConfig) -> VarBuilder<'static> {
        let device = &Device::Cpu;
        let ac = &cfg.multimodal.audio_model_args.acoustic_transformer_args;
        let audio = &cfg.multimodal.audio_model_args;
        let dim = ac.dim;
        let q_out = ac.n_heads * ac.head_dim;
        let kv_out = ac.n_kv_heads * ac.head_dim;

        let mut t: HashMap<String, Tensor> = HashMap::new();
        let zeros2 = |s: (usize, usize)| Tensor::zeros(s, DType::F32, device).expect("zeros");
        let ones1 = |n: usize| Tensor::ones(n, DType::F32, device).expect("ones");

        let p = "acoustic_transformer.";

        // Projection linears (no bias)
        t.insert(
            format!("{p}input_projection.weight"),
            zeros2((dim, audio.n_acoustic_codebook)),
        );
        t.insert(format!("{p}time_projection.weight"), zeros2((dim, dim)));
        t.insert(format!("{p}llm_projection.weight"), zeros2((dim, dim)));

        // Semantic head (no bias), acoustic head (no bias)
        t.insert(
            format!("{p}semantic_codebook_output.weight"),
            zeros2((SEMANTIC_PADDED, dim)),
        );
        t.insert(
            format!("{p}acoustic_codebook_output.weight"),
            zeros2((audio.n_acoustic_codebook, dim)),
        );

        // Final norm
        t.insert(format!("{p}norm.weight"), ones1(dim));

        // Transformer layers (post-rename HF-style names)
        for i in 0..ac.n_layers {
            let lp = format!("{p}layers.{i}.");
            t.insert(format!("{lp}self_attn.q_proj.weight"), zeros2((q_out, dim)));
            t.insert(
                format!("{lp}self_attn.k_proj.weight"),
                zeros2((kv_out, dim)),
            );
            t.insert(
                format!("{lp}self_attn.v_proj.weight"),
                zeros2((kv_out, dim)),
            );
            t.insert(format!("{lp}self_attn.o_proj.weight"), zeros2((dim, q_out)));
            t.insert(format!("{lp}input_layernorm.weight"), ones1(dim));
            t.insert(format!("{lp}post_attention_layernorm.weight"), ones1(dim));
            t.insert(
                format!("{lp}mlp.gate_proj.weight"),
                zeros2((ac.hidden_dim, dim)),
            );
            t.insert(
                format!("{lp}mlp.up_proj.weight"),
                zeros2((ac.hidden_dim, dim)),
            );
            t.insert(
                format!("{lp}mlp.down_proj.weight"),
                zeros2((dim, ac.hidden_dim)),
            );
        }

        VarBuilder::from_tensors(t, DType::F32, device)
    }

    // ── Acoustic transformer tests ────────────────────────────────────────────

    #[test]
    fn test_acoustic_transformer_new() {
        let cfg = tiny_config();
        let vb = acoustic_vb(&cfg);
        AcousticTransformer::new(&cfg, vb).expect("AcousticTransformer::new");
    }

    #[test]
    fn test_time_embedding_shape() {
        let cfg = tiny_config();
        let at = AcousticTransformer::new(&cfg, acoustic_vb(&cfg)).expect("new");
        let emb = at
            .compute_time_embedding(0.5, &Device::Cpu)
            .expect("time_emb");
        assert_eq!(emb.dims(), &[at.dim]);
    }

    #[test]
    fn test_time_embedding_cos_sin_split() {
        // At t=0, cos(0) = 1.0 and sin(0) = 0.0 for all frequencies.
        let cfg = tiny_config();
        let at = AcousticTransformer::new(&cfg, acoustic_vb(&cfg)).expect("new");
        let emb: Vec<f32> = at
            .compute_time_embedding(0.0, &Device::Cpu)
            .expect("emb")
            .to_vec1()
            .expect("to_vec1");
        let half = at.dim / 2;
        for &v in &emb[..half] {
            assert!((v - 1.0).abs() < 1e-5, "cos(0) should be 1, got {v}");
        }
        for &v in &emb[half..] {
            assert!(v.abs() < 1e-5, "sin(0) should be 0, got {v}");
        }
    }

    #[test]
    fn test_semantic_code_valid_range() {
        let cfg = tiny_config();
        let at = AcousticTransformer::new(&cfg, acoustic_vb(&cfg)).expect("new");
        let h = Tensor::zeros(at.dim, DType::F32, &Device::Cpu).expect("h");
        let code = at.predict_semantic_code(&h).expect("predict_semantic_code");
        // Must not be EMPTY_AUDIO (0) and must be within valid range
        assert!(code >= 1, "code must not be EMPTY_AUDIO");
        assert!(
            (code as usize) < at.semantic_valid_end,
            "code {code} must be below semantic_valid_end {}",
            at.semantic_valid_end,
        );
    }

    #[test]
    fn test_semantic_code_non_end_audio() {
        // Build a VarBuilder where row 42 of the semantic head weight is all
        // ones. With h = ones(dim), logit[42] = dim while all others are 0,
        // so argmax picks code 42.
        let cfg = tiny_config();
        let device = &Device::Cpu;
        let ac = &cfg.multimodal.audio_model_args.acoustic_transformer_args;
        let audio = &cfg.multimodal.audio_model_args;
        let dim = ac.dim;
        let q_out = ac.n_heads * ac.head_dim;
        let kv_out = ac.n_kv_heads * ac.head_dim;

        let mut t: HashMap<String, Tensor> = HashMap::new();
        let zeros2 = |s: (usize, usize)| Tensor::zeros(s, DType::F32, device).expect("zeros");
        let ones1 = |n: usize| Tensor::ones(n, DType::F32, device).expect("ones");

        let p = "acoustic_transformer.";
        t.insert(
            format!("{p}input_projection.weight"),
            zeros2((dim, audio.n_acoustic_codebook)),
        );
        t.insert(format!("{p}time_projection.weight"), zeros2((dim, dim)));
        t.insert(format!("{p}llm_projection.weight"), zeros2((dim, dim)));
        // Row 42 = ones, all other rows = zeros -> logit[42] = dim
        let mut w = vec![0f32; SEMANTIC_PADDED * dim];
        for j in 0..dim {
            w[42 * dim + j] = 1.0;
        }
        t.insert(
            format!("{p}semantic_codebook_output.weight"),
            Tensor::from_vec(w, (SEMANTIC_PADDED, dim), device).expect("weight"),
        );
        t.insert(
            format!("{p}acoustic_codebook_output.weight"),
            zeros2((audio.n_acoustic_codebook, dim)),
        );
        t.insert(format!("{p}norm.weight"), ones1(dim));
        for i in 0..ac.n_layers {
            let lp = format!("{p}layers.{i}.");
            t.insert(format!("{lp}self_attn.q_proj.weight"), zeros2((q_out, dim)));
            t.insert(
                format!("{lp}self_attn.k_proj.weight"),
                zeros2((kv_out, dim)),
            );
            t.insert(
                format!("{lp}self_attn.v_proj.weight"),
                zeros2((kv_out, dim)),
            );
            t.insert(format!("{lp}self_attn.o_proj.weight"), zeros2((dim, q_out)));
            t.insert(format!("{lp}input_layernorm.weight"), ones1(dim));
            t.insert(format!("{lp}post_attention_layernorm.weight"), ones1(dim));
            t.insert(
                format!("{lp}mlp.gate_proj.weight"),
                zeros2((ac.hidden_dim, dim)),
            );
            t.insert(
                format!("{lp}mlp.up_proj.weight"),
                zeros2((ac.hidden_dim, dim)),
            );
            t.insert(
                format!("{lp}mlp.down_proj.weight"),
                zeros2((dim, ac.hidden_dim)),
            );
        }
        let vb = VarBuilder::from_tensors(t, DType::F32, device);

        let at = AcousticTransformer::new(&cfg, vb).expect("new");
        let h = Tensor::ones(at.dim, DType::F32, device).expect("h");
        let code = at.predict_semantic_code(&h).expect("predict_semantic_code");
        assert_eq!(code, 42, "expected argmax at index 42");
    }

    #[test]
    fn test_flow_match_output_shape_and_range() {
        let cfg = tiny_config();
        let mut at = AcousticTransformer::new(&cfg, acoustic_vb(&cfg)).expect("new");
        let h = Tensor::zeros(at.dim, DType::F32, &Device::Cpu).expect("h");
        let codes = at.flow_match_inference(&h).expect("flow_match_inference");

        assert_eq!(codes.dims(), &[at.n_acoustic]);
        assert_eq!(codes.dtype(), DType::U32);

        let codes: Vec<u32> = codes.to_vec1().expect("to_vec1");
        for &c in &codes {
            assert!(c >= 2 && c <= 22, "FSQ code {c} out of range [2, 22]");
        }
    }

    #[test]
    fn test_llm_projection_zeros_in_zeros_out() {
        let cfg = tiny_config();
        let at = AcousticTransformer::new(&cfg, acoustic_vb(&cfg)).expect("new");
        let zeros = Tensor::zeros(at.dim, DType::F32, &Device::Cpu).expect("zeros");
        let out = at
            .llm_projection
            .forward(&zeros.unsqueeze(0).expect("unsqueeze"))
            .expect("forward")
            .squeeze(0)
            .expect("squeeze");
        let vals: Vec<f32> = out.to_vec1().expect("to_vec1");
        assert!(
            vals.iter().all(|&v| v == 0.0),
            "projection of zeros must be zeros (no-bias invariant)"
        );
    }

    // ── Integration tests (require local checkpoint) ──────────────────────────

    fn checkpoint_path() -> Option<std::path::PathBuf> {
        let p = if let Ok(dir) = std::env::var("VOXTRAL_CHECKPOINT_DIR") {
            std::path::PathBuf::from(dir)
        } else {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()?
                .join("checkpoints/Voxtral-4B-TTS-2603")
        };
        p.is_dir().then_some(p)
    }

    #[test]
    #[ignore = "requires checkpoint dir (set VOXTRAL_CHECKPOINT_DIR or place at checkpoints/Voxtral-4B-TTS-2603)"]
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
    #[ignore = "requires checkpoint dir (set VOXTRAL_CHECKPOINT_DIR or place at checkpoints/Voxtral-4B-TTS-2603)"]
    fn test_acoustic_transformer_load_real() {
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
        AcousticTransformer::new(&cfg, vb).expect("AcousticTransformer::new");
    }

    #[test]
    #[ignore = "requires checkpoint dir (set VOXTRAL_CHECKPOINT_DIR or place at checkpoints/Voxtral-4B-TTS-2603)"]
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
