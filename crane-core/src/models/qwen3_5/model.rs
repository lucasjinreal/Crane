//! Top-level Qwen 3.5 text-only transformer + the high-level `Model`
//! wrapper used by the engine (config + weights + tokenizer).

use std::io::Write;

use anyhow::{Context, Error as E, Result};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder, embedding};

use crate::ops::linear::{LinearLayer, parse_ggml_dtype, quantize_linear};
use crate::utils::DeviceExt;
// TODO(candle-transformers-removal): Generation helpers only; see CANDLE_TRANSFORMERS.md.
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

use super::config::{Config, HiddenAct, LayerType, RopeParameters, TextConfig, load_config};
use super::kv_cache::{KvCache, KvCacheKind};
use super::modeling::{DecoderLayer, MRotaryEmbedding, Qwen35RmsNorm, RopeSlice};
use crate::generation::GenerationConfig;
use crate::generation::based::ModelForCausalLM;
use crate::models::modules::embedding::EmbeddingLayer;
use crate::quantized::gguf_file::Gguf;
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::utils;

/// A point-in-time capture of every layer's cache, for resuming a shared
/// prompt prefix. See [`Qwen3_5TextModel::snapshot_state`].
#[derive(Clone)]
pub struct StateSnapshot {
    gdn: Vec<Option<crate::ops::gdn::GdnLayerCache>>,
    /// Fill level per attention layer; `None` for GDN blocks.
    attn_lens: Vec<Option<usize>>,
}

/// Text-only Qwen 3.5 transformer.
///
/// `gdn_caches` is indexed by layer; `None` for full-attention blocks, `Some`
/// for linear-attention blocks. The engine is responsible for cloning/saving
/// these caches across context switches (continuous batching).
pub struct Qwen3_5TextModel {
    cfg: TextConfig,
    embed_tokens: EmbeddingLayer,
    layers: Vec<DecoderLayer>,
    norm: Qwen35RmsNorm,
    lm_head: LinearLayer,
    rotary: MRotaryEmbedding,
    gdn_caches: Vec<Option<crate::ops::gdn::GdnLayerCache>>,
    /// Per-layer K/V cache; `Some` for full-attention blocks, `None` for GDN.
    attn_caches: Vec<Option<KvCache>>,
    device: Device,
    dtype: DType,
}

impl Qwen3_5TextModel {
    /// Load a text-only Qwen 3.5 model from a HF checkpoint directory.
    ///
    /// The HF layout has a top-level `language_model.*` prefix when the model
    /// was saved with `Qwen3_5ForConditionalGeneration`. We probe for that
    /// prefix and fall back to a flat layout.
    ///
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing or has an
    /// unexpected shape.
    pub fn new(
        cfg: &Config,
        vb: &VarBuilder,
        device: &Device,
        dtype: DType,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let text_cfg = cfg.text().clone();
        // HF saves Qwen 3.5 weights with the prefix
        //   model.language_model.layers.{i}.{linear_attn,self_attn}.*
        // (the `model.` is the inner multimodal `Qwen3_5Model`, of which
        // `language_model` is the text component). Older checkpoints and
        // standalone text exports may drop the leading `model.`.
        let vb_lm = if vb.contains_tensor("model.language_model.embed_tokens.weight") {
            vb.pp("model").pp("language_model")
        } else if vb.contains_tensor("language_model.embed_tokens.weight") {
            vb.pp("language_model")
        } else if vb.contains_tensor("model.embed_tokens.weight") {
            vb.pp("model")
        } else {
            // Last-resort: assume a flat layout, no prefix.
            vb.clone()
        };

        let embed_tokens = embedding(
            text_cfg.vocab_size,
            text_cfg.hidden_size,
            vb_lm.pp("embed_tokens"),
        )?;
        let embed_weight = embed_tokens.embeddings().clone();
        let embed_tokens = EmbeddingLayer::Dense(embed_tokens);

        let layer_types = text_cfg.layer_types();
        let mut layers = Vec::with_capacity(text_cfg.num_hidden_layers);
        for (idx, &layer_type) in layer_types.iter().enumerate() {
            layers.push(DecoderLayer::load(
                &text_cfg,
                layer_type,
                vb_lm.pp("layers").pp(idx),
                quant,
            )?);
        }

        let norm = Qwen35RmsNorm::load(
            text_cfg.hidden_size,
            text_cfg.rms_norm_eps,
            &vb_lm.pp("norm"),
        )?;

        // Resolve the output projection. With `tie_word_embeddings: true`
        // (0.8B/4B) it's the embedding table; untied models (e.g. Ornith-9B)
        // ship a dedicated `lm_head.weight` of shape `[vocab, hidden]`. That
        // tensor lives at the checkpoint ROOT, not under the `language_model`
        // prefix, so probe `vb` first and fall back to `vb_lm`.
        let (lm_head_raw, is_tied) = if cfg.tie_word_embeddings {
            (embed_weight, true)
        } else {
            let shape = (text_cfg.vocab_size, text_cfg.hidden_size);
            let mut tied = false;
            let w = vb
                .get(shape, "lm_head.weight")
                .or_else(|_| vb_lm.get(shape, "lm_head.weight"))
                .or_else(|_| {
                    eprintln!(
                        "[qwen3_5] tie_word_embeddings=false but no lm_head.weight found; \
                         falling back to tied embeddings"
                    );
                    tied = true;
                    Ok::<_, candle_core::Error>(embed_weight)
                })?;
            (w, tied)
        };
        // Quantizing a tied lm_head would only ADD memory: the fp embedding
        // table must stay resident for lookups, so the quantized copy is pure
        // overhead. Quantize only a dedicated (untied) output projection.
        let lm_head = match quant {
            Some(dt) if !is_tied => quantize_linear(Linear::new(lm_head_raw, None), dt)?,
            _ => {
                // Pre-store in F32 only when the compute dtype is F16: at
                // this model's 248k vocab, raw logits overflow F16's 65504
                // max even more readily than smaller-vocab models (see
                // `forward_logits`). BF16/F32 stay native.
                let w = if dtype == DType::F16 {
                    lm_head_raw.to_dtype(DType::F32)?
                } else {
                    lm_head_raw
                };
                LinearLayer::Standard(Linear::new(w, None))
            },
        };

        let rotary = MRotaryEmbedding::new(&text_cfg, device)?;

        let (gdn_caches, attn_caches) = build_layer_caches(&layers, &text_cfg, dtype, device)?;

        Ok(Self {
            cfg: text_cfg,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
            gdn_caches,
            attn_caches,
            device: device.clone(),
            dtype,
        })
    }

    /// Load from a parsed GGUF file (llama.cpp `qwen35` arch).
    ///
    /// The model config is reconstructed entirely from GGUF metadata; the
    /// per-layer full/linear attention layout is derived from tensor presence
    /// (`blk.{i}.ssm_a` ⇒ linear) rather than trusting the interval field.
    ///
    /// # Errors
    ///
    /// Returns an error if required GGUF metadata or tensors are missing or malformed.
    // This function's length comes from reconstructing every TextConfig field
    // one-by-one from GGUF metadata keys; splitting it up would scatter that
    // field-by-field mapping across several small functions without
    // simplifying it.
    #[allow(clippy::too_many_lines)]
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        ct: candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        Self::from_gguf_impl(ct, reader, device, None)
    }

    pub fn from_gguf_extended<R: std::io::Read + std::io::Seek>(
        ct: candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        device: &Device,
        extended: crate::quantized::extended_gguf::ExtendedGgufInfo,
    ) -> Result<Self> {
        Self::from_gguf_impl(ct, reader, device, Some(extended))
    }

    fn from_gguf_impl<R: std::io::Read + std::io::Seek>(
        ct: candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        device: &Device,
        extended: Option<crate::quantized::extended_gguf::ExtendedGgufInfo>,
    ) -> Result<Self> {
        // QMatMul handles quantized weights internally; dequantized side
        // tensors (norms, conv kernels, embeddings) use a compute dtype of
        // BF16 on CUDA and F16 on Metal (the F32 embedding alone would cost
        // ~1 GB at Qwen3.5's 248k vocab), F32 on CPU.
        let dtype = if device.is_cuda() {
            DType::BF16
        } else if device.is_metal() || device.is_rocm() || device.is_sycl() {
            DType::F16
        } else {
            DType::F32
        };
        let mut gg = match extended {
            Some(info) => Gguf::new_extended(ct, reader, device.clone(), dtype, info)?,
            None => Gguf::new(ct, reader, device.clone(), dtype),
        };

        let arch = gg
            .metadata()
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_else(|| "qwen35".to_string());
        let md_u32 = |gg: &Gguf<&mut R>, key: &str| -> Result<usize> {
            gg.metadata()
                .get(&format!("{arch}.{key}"))
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!("missing GGUF metadata {arch}.{key}"))
                })?
                .to_u32()
                .map(|v| v as usize)
                .map_err(Into::into)
        };
        let md_u32_or = |gg: &Gguf<&mut R>, key: &str, default: usize| -> usize {
            gg.metadata()
                .get(&format!("{arch}.{key}"))
                .and_then(|v| v.to_u32().ok())
                .map_or(default, |v| v as usize)
        };

        let head_dim = md_u32(&gg, "attention.key_length")?;
        let hidden_size = md_u32(&gg, "embedding_length")?;
        let num_hidden_layers = md_u32(&gg, "block_count")?;
        let rms_norm_eps = gg
            .metadata()
            .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.to_f32().ok())
            .map_or(1e-6, f64::from);
        let rope_theta = gg
            .metadata()
            .get(&format!("{arch}.rope.freq_base"))
            .and_then(|v| v.to_f32().ok())
            .map_or(10_000_000.0, f64::from);
        let rot_dim = md_u32_or(&gg, "rope.dimension_count", head_dim / 4);
        let mrope_section: Vec<usize> = gg
            .metadata()
            .get(&format!("{arch}.rope.dimension_sections"))
            .and_then(|v| v.to_vec().ok())
            .map(|vals| {
                vals.iter()
                    .filter_map(|v| v.to_i32().ok().map(|x| usize::try_from(x).unwrap_or(0)))
                    .collect()
            })
            .unwrap_or_default();
        let num_v_heads = md_u32(&gg, "ssm.time_step_rank")?;
        let inner_size = md_u32(&gg, "ssm.inner_size")?;

        // Vocab from the embedding table shape (metadata has no vocab_size).
        let vocab_size = gg
            .ct
            .tensor_infos
            .get("token_embd.weight")
            .ok_or_else(|| candle_core::Error::Msg("GGUF missing token_embd.weight".into()))?
            .shape
            .dims()[0];
        let tie_word_embeddings = !gg.contains_tensor("output.weight");

        // Per-layer attention layout from tensor presence.
        let layer_types: Vec<LayerType> = (0..num_hidden_layers)
            .map(|i| {
                if gg.contains_tensor(&format!("blk.{i}.ssm_a")) {
                    LayerType::LinearAttention
                } else {
                    LayerType::FullAttention
                }
            })
            .collect();

        // The q projection is 2× wide when the sigmoid output gate is fused in.
        let attn_output_gate = layer_types
            .iter()
            .position(|t| *t == LayerType::FullAttention)
            .is_none_or(|i| {
                let q_rows = gg
                    .ct
                    .tensor_infos
                    .get(&format!("blk.{i}.attn_q.weight"))
                    .map_or(0, |info| info.shape.dims()[0]);
                let num_heads = md_u32_or(&gg, "attention.head_count", 0);
                q_rows == 2 * num_heads * head_dim
            });

        // rot_dim and head_dim are small model dimensions (well under 2^52),
        // so this division cannot lose precision.
        #[allow(clippy::cast_precision_loss)]
        let partial_rotary_factor = rot_dim as f64 / head_dim as f64;

        let text_cfg = TextConfig {
            head_dim,
            vocab_size,
            hidden_size,
            intermediate_size: md_u32(&gg, "feed_forward_length")?,
            num_hidden_layers,
            num_attention_heads: md_u32(&gg, "attention.head_count")?,
            num_key_value_heads: md_u32(&gg, "attention.head_count_kv")?,
            hidden_act: HiddenAct::Silu,
            max_position_embeddings: md_u32_or(&gg, "context_length", 262_144),
            rms_norm_eps,
            rope_parameters: RopeParameters {
                rope_theta,
                mrope_section,
                partial_rotary_factor,
                mrope_interleaved: true,
            },
            full_attention_interval: md_u32_or(&gg, "full_attention_interval", 4),
            linear_conv_kernel_dim: md_u32(&gg, "ssm.conv_kernel")?,
            linear_key_head_dim: md_u32(&gg, "ssm.state_size")?,
            linear_value_head_dim: inner_size / num_v_heads,
            linear_num_key_heads: md_u32(&gg, "ssm.group_count")?,
            linear_num_value_heads: num_v_heads,
            tie_word_embeddings,
            attn_output_gate,
            // GGUF carries no gate-activation key; the conversion only ever
            // targets the swish gate this code implements.
            output_gate_type: None,
        };

        // The 248k-row table is the single largest dequantization in the
        // checkpoint; keep it in its GGUF format and gather rows on demand.
        let embed_tokens = gg.quantized_embedding("token_embd.weight", hidden_size)?;

        let mut layers = Vec::with_capacity(num_hidden_layers);
        for (idx, &layer_type) in layer_types.iter().enumerate() {
            layers.push(DecoderLayer::from_gguf(
                &text_cfg, layer_type, &mut gg, idx,
            )?);
        }

        let norm =
            Qwen35RmsNorm::from_folded(gg.dequant_tensor("output_norm.weight")?, rms_norm_eps);
        let lm_head = if tie_word_embeddings {
            // Tied: the output projection is this very table, so it reuses the
            // same buffer rather than materializing a dense copy.
            let tied = embed_tokens.tied_output()?;
            // Pre-store a `Standard` tied head as F32 only for F16, same
            // overflow reasoning as the safetensors path above. A
            // `Quantized` table already computes in F32 internally via
            // QMatMul, so it needs no change.
            match tied {
                LinearLayer::Standard(l) if dtype == DType::F16 => {
                    LinearLayer::Standard(Linear::new(l.weight().to_dtype(DType::F32)?, None))
                },
                other => other,
            }
        } else {
            gg.linear("output.weight")?
        };

        let rotary = MRotaryEmbedding::new(&text_cfg, device)?;
        let (gdn_caches, attn_caches) = build_layer_caches(&layers, &text_cfg, dtype, device)?;

        Ok(Self {
            cfg: text_cfg,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
            gdn_caches,
            attn_caches,
            device: device.clone(),
            dtype,
        })
    }

    #[must_use]
    pub fn config(&self) -> &TextConfig {
        &self.cfg
    }

    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Total bytes held by the full-attention K/V caches across all layers
    /// (incl. headroom + quant scales). The context-scaling memory term.
    #[must_use]
    pub fn attn_cache_bytes(&self) -> usize {
        self.attn_caches
            .iter()
            .flatten()
            .map(KvCache::byte_size)
            .sum()
    }

    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Capture the layer state so a later request replaying this token prefix
    /// can resume from it.
    ///
    /// Cheap: the GDN entries are `Tensor` clones (refcount bumps, and the
    /// recurrence assigns rather than writes in place), and the attention side
    /// only records a fill level — its buffer is append-only, so rewinding to
    /// that level restores exactly these positions.
    ///
    /// Valid only until something resets the caches.
    #[must_use]
    pub fn snapshot_state(&self) -> StateSnapshot {
        StateSnapshot {
            gdn: self.gdn_caches.clone(),
            attn_lens: self
                .attn_caches
                .iter()
                .map(|c| c.as_ref().map(KvCache::len))
                .collect(),
        }
    }

    /// Restore a [`snapshot_state`](Self::snapshot_state).
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot does not describe this model's layers.
    pub fn restore_state(&mut self, snap: &StateSnapshot) -> Result<()> {
        if snap.gdn.len() != self.gdn_caches.len() || snap.attn_lens.len() != self.attn_caches.len()
        {
            anyhow::bail!(
                "state snapshot has {} gdn / {} attn layers, model has {} / {}",
                snap.gdn.len(),
                snap.attn_lens.len(),
                self.gdn_caches.len(),
                self.attn_caches.len(),
            );
        }
        self.gdn_caches = snap.gdn.clone();
        for (cache, len) in self.attn_caches.iter_mut().zip(&snap.attn_lens) {
            match (cache.as_mut(), len) {
                (Some(c), Some(n)) => c.truncate(*n),
                (None, None) => {},
                _ => anyhow::bail!("state snapshot layer kinds do not match the model"),
            }
        }
        Ok(())
    }

    /// Reset all per-layer GDN caches. Called between unrelated requests
    /// sharing the same pre-allocated layer set.
    ///
    /// # Errors
    ///
    /// Returns an error if resetting a layer's recurrent state fails.
    pub fn reset_gdn_caches(&mut self) -> Result<()> {
        for slot in self.gdn_caches.iter_mut().flatten() {
            slot.reset()?;
        }
        for slot in self.attn_caches.iter_mut().flatten() {
            slot.reset();
        }
        Ok(())
    }

    /// Embed `input_ids` and return the hidden states `[B, S, hidden_size]`.
    /// Used by the multimodal wrapper to compute text embeddings before
    /// splicing image features over the `<|image_pad|>` placeholders.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedding lookup fails.
    pub fn embed_only(&self, input_ids: &Tensor) -> Result<Tensor> {
        Ok(self.embed_tokens.forward(input_ids)?)
    }

    /// Forward pass over `input_ids` of shape `[B, S]`. `start_pos` is the
    /// absolute position of the first token (used for rotary slicing).
    /// `attention_mask` is broadcastable to `[B, 1, S, S_total]` (or `None`).
    ///
    /// Returns next-token logits of shape `[B, vocab_size]` — only the last
    /// position is projected through `lm_head`.
    ///
    /// Long prompts are prefilled in chunks; see [`super::prefill`].
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        start_pos: usize,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        super::prefill::forward(self, input_ids, start_pos, attention_mask)
    }

    /// Embed `input_ids` `[B, S]` and run every decoder layer, returning the
    /// pre-final-norm hidden states `[B, S, hidden_size]`.
    ///
    /// `start_pos` is the absolute position of the first token: `RoPE` and the
    /// causal mask are indexed from it, so a prefill chunk starting mid-prompt
    /// sees its true positions rather than chunk-relative ones.
    pub(super) fn forward_layers(
        &mut self,
        input_ids: &Tensor,
        start_pos: usize,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        use crate::utils::prof::{Span, timed};

        let seq_len = input_ids.dim(1)?;
        let xs = timed(Span::Embed, || self.embed_tokens.forward(input_ids))?;

        let (cos, sin) = self.rotary.cos_sin(start_pos, seq_len)?;
        let rope = RopeSlice {
            cos: &cos,
            sin: &sin,
            rot_dim: self.rotary.rot_dim(),
        };
        self.run_layers(xs, rope, attention_mask)
    }

    /// Forward pass with **pre-computed hidden states** (vision path).
    ///
    /// `hidden_states` already has image embeddings spliced in at the
    /// `image_token_id` positions; the caller is responsible for embedding
    /// the text tokens (or a placeholder at image positions) and stitching
    /// them together. `position_ids` of shape `[3, S]` enables per-token
    /// 3D `MRoPE` for vision tokens (T/H/W positions).
    ///
    /// `start_pos` is still required for the K/V cache offset. For
    /// prefill with images this is `0`; for decode it advances one per
    /// generated token (the caller already supplies sequential positions
    /// on the three axes).
    ///
    /// Returns logits of shape `[B, V]` (only the last position is projected,
    /// matching the text-only forward).
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward_embeds(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        start_pos: usize,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_b, seq_len, _h) = hidden_states.dims3()?;
        let is_decode_step = seq_len == 1;

        let mask = match attention_mask {
            Some(m) => Some(m.clone()),
            None if !is_decode_step => Some(super::prefill::causal_mask(
                seq_len,
                start_pos,
                hidden_states.device(),
                hidden_states.dtype(),
            )?),
            None => None,
        };

        let (cos, sin) = self.rotary.cos_sin_with_position_ids(position_ids)?;
        let rope = RopeSlice {
            cos: &cos,
            sin: &sin,
            rot_dim: self.rotary.rot_dim(),
        };

        let xs = self.run_layers(hidden_states.clone(), rope, mask.as_ref())?;
        let s = xs.dim(1)?;
        let last = xs.narrow(1, s - 1, 1)?.contiguous()?;
        self.head(&last)
    }

    /// Run every decoder layer over already-embedded hidden states `xs`,
    /// returning the pre-final-norm hidden states.
    fn run_layers(
        &mut self,
        mut xs: Tensor,
        rope: RopeSlice<'_>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let debug = std::env::var_os("CRANE_QWEN35_DEBUG_LAYERS").is_some();
        for i in 0..self.layers.len() {
            let layer = &self.layers[i];
            let gdn_slot = self.gdn_caches[i].as_mut();
            let attn_slot = self.attn_caches[i].as_mut();
            xs = layer.forward(&xs, rope, attention_mask, gdn_slot, attn_slot)?;
            if debug {
                let last = xs
                    .narrow(1, xs.dim(1)? - 1, 1)?
                    .flatten_all()?
                    .to_dtype(DType::F32)?;
                let v = last.to_vec1::<f32>()?;
                // v is a single hidden-state vector (hidden_size elements, a small
                // model dimension), nowhere near f32's 24-bit exact-integer range.
                #[allow(clippy::cast_precision_loss)]
                let n = v.len() as f32;
                let mean = v.iter().sum::<f32>() / n;
                let min = v.iter().copied().fold(f32::INFINITY, f32::min);
                let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let nonfinite = v.iter().filter(|x| !x.is_finite()).count();
                eprintln!(
                    "[qwen3_5:debug] layer[{i}]: min={min:.4} max={max:.4} mean={mean:.4} non_finite={nonfinite}/{}",
                    v.len()
                );
            }
        }
        Ok(xs)
    }

    /// Final norm + `lm_head` over a single position `[B, 1, hidden_size]`,
    /// returning `[B, vocab_size]`.
    ///
    /// Prefill only ever needs the last position, and at Qwen 3.5's 248k vocab
    /// running the head over the whole prompt would put the quadratic memory
    /// term straight back after chunking removed it from attention.
    pub(super) fn head(&self, hidden: &Tensor) -> Result<Tensor> {
        crate::utils::prof::timed(crate::utils::prof::Span::Head, || {
            let (b, _s, _h) = hidden.dims3()?;
            let xs = self.norm.forward(hidden)?.reshape((b, ()))?;
            Ok(self.lm_head.forward_logits(&xs)?)
        })
    }
}

/// Pre-allocate per-layer caches: GDN recurrent state for linear blocks,
/// K/V cache for full-attention blocks (mutually exclusive per layer).
/// The K/V representation (fp / int8 / …) is chosen once here.
#[allow(clippy::type_complexity)]
fn build_layer_caches(
    layers: &[DecoderLayer],
    cfg: &TextConfig,
    dtype: DType,
    device: &Device,
) -> Result<(
    Vec<Option<crate::ops::gdn::GdnLayerCache>>,
    Vec<Option<KvCache>>,
)> {
    let kv_kind = KvCacheKind::from_env();
    let mut gdn_caches = Vec::with_capacity(layers.len());
    let mut attn_caches = Vec::with_capacity(layers.len());
    for layer in layers {
        if layer.is_linear() {
            gdn_caches.push(Some(crate::ops::gdn::GdnLayerCache::new(
                cfg, dtype, device,
            )?));
            attn_caches.push(None);
        } else {
            gdn_caches.push(None);
            attn_caches.push(Some(KvCache::new(kv_kind)));
        }
    }
    Ok((gdn_caches, attn_caches))
}

/// Append Qwen3.5's canonical stop-token ids (`<|im_end|>`, `<|endoftext|>`)
/// to `eos_token_ids` by resolving them by name in `vocab`, skipping any
/// already present. Used as a GGUF fallback: GGUF's `tokenizer.ggml.eos_token_id`
/// metadata field can only hold one id, so a GGUF-only export (no sidecar
/// `generation_config.json`) silently drops whichever canonical stop token the
/// exporter didn't pick.
fn merge_canonical_eos_ids(
    eos_token_ids: &mut Vec<u32>,
    vocab: &std::collections::HashMap<String, u32>,
) {
    for name in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(&id) = vocab.get(name)
            && !eos_token_ids.contains(&id)
        {
            eos_token_ids.push(id);
        }
    }
}

/// Read EOS token id(s) from `generation_config.json` (preferred) then
/// `config.json`. The field may be a single integer or a list; returns an empty
/// vec if absent.
fn read_eos_token_ids(model_path: &str) -> Vec<u32> {
    fn from_value(v: &serde_json::Value) -> Vec<u32> {
        match v {
            serde_json::Value::Number(n) => n
                .as_u64()
                .and_then(|x| u32::try_from(x).ok())
                .map(|x| vec![x])
                .unwrap_or_default(),
            serde_json::Value::Array(a) => a
                .iter()
                .filter_map(|e| e.as_u64().and_then(|x| u32::try_from(x).ok()))
                .collect(),
            _ => Vec::new(),
        }
    }
    for fname in ["generation_config.json", "config.json"] {
        let path = std::path::Path::new(model_path).join(fname);
        let Ok(data) = std::fs::read(&path) else {
            continue;
        };
        let Ok(json) = serde_json::from_slice::<serde_json::Value>(&data) else {
            continue;
        };
        if let Some(eos) = json.get("eos_token_id") {
            let ids = from_value(eos);
            if !ids.is_empty() {
                return ids;
            }
        }
    }
    Vec::new()
}

/// Format of model weights on disk. `Auto` picks GGUF when the path is a
/// `.gguf` file, safetensors otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Auto,
    Safetensors,
    Gguf,
}

/// holds a tokenizer, device, dtype, and the inner [`Qwen3_5TextModel`].
pub struct Model {
    pub tokenizer: TokenOutputStream,
    pub device: Device,
    pub dtype: DType,
    /// Stop tokens read from `generation_config.json` / `config.json` (Qwen3.5
    /// and Ornith both use multi-id EOS, e.g. `[248044, 248046]`). Used when the
    /// caller's `GenerationConfig` doesn't pin its own `eos_token_id`.
    eos_token_ids: Vec<u32>,
    inner: Qwen3_5TextModel,
}

/// Read the in-situ quantization level from `CRANE_ISQ` (e.g. `q4k`, `q8_0`).
/// Invalid values abort with a clear message rather than silently loading fp.
fn isq_from_env() -> Option<GgmlDType> {
    let name = std::env::var("CRANE_ISQ").ok()?;
    if name.trim().is_empty() {
        return None;
    }
    match parse_ggml_dtype(&name) {
        Ok(dt) => Some(dt),
        Err(e) => panic!("invalid CRANE_ISQ: {e}"),
    }
}

impl Model {
    /// Load a Qwen 3.5 model from a HF checkpoint directory (or `.gguf` file).
    ///
    /// In-situ quantization is picked up from the `CRANE_ISQ` env var; use
    /// [`Model::new_with_options`] to set it explicitly (e.g. from a CLI flag).
    ///
    /// # Errors
    ///
    /// Returns an error if the model or tokenizer cannot be loaded from `model_path`.
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        Self::new_with_format(model_path, device, dtype, ModelFormat::Auto)
    }

    /// # Errors
    ///
    /// Returns an error if the model or tokenizer cannot be loaded from `model_path`.
    pub fn new_with_format(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        format: ModelFormat,
    ) -> Result<Self> {
        Self::new_with_options(model_path, device, dtype, format, isq_from_env())
    }

    /// Load with an explicit in-situ quantization level (`quant`). `None`
    /// keeps the checkpoint dtype. Quantization only applies to the
    /// safetensors path — GGUF weights are already quantized.
    ///
    /// # Errors
    ///
    /// Returns an error if the model or tokenizer cannot be loaded from `model_path`.
    pub fn new_with_options(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        format: ModelFormat,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let format = match format {
            ModelFormat::Auto => {
                let is_gguf = std::path::Path::new(model_path)
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
                if is_gguf {
                    ModelFormat::Gguf
                } else {
                    ModelFormat::Safetensors
                }
            },
            other => other,
        };
        match format {
            ModelFormat::Safetensors => Self::from_pretrained(model_path, device, *dtype, quant),
            ModelFormat::Gguf => {
                if quant.is_some() {
                    eprintln!(
                        "[qwen3_5] --quant/CRANE_ISQ ignored: GGUF weights are already quantized"
                    );
                }
                Self::from_gguf_file(model_path, device)
            },
            ModelFormat::Auto => unreachable!("Auto is resolved above"),
        }
    }

    /// Load from a `.gguf` file. The tokenizer is read from the GGUF itself
    /// (`tokenizer.ggml.tokens` / `tokenizer.ggml.merges` / `token_type`);
    /// a sibling `tokenizer.json` is only consulted if the GGUF lacks the
    /// embedded metadata (older / third-party quantizers).
    fn from_gguf_file(model_path: &str, device: &Device) -> Result<Self> {
        use crate::utils::tokenizer_utils::resolve_gguf_tokenizer;

        let gguf_path = std::path::Path::new(model_path);
        let parent = gguf_path.parent().unwrap_or(gguf_path);

        let mmap = crate::quantized::gguf_file::mmap_gguf_file(gguf_path)
            .with_context(|| format!("mmap GGUF file {model_path}"))?;
        let mut cursor = std::io::Cursor::new(mmap.as_ref());
        let (ct, extended) = crate::quantized::extended_gguf::read_content(mmap.as_ref())?;
        eprintln!(
            "[qwen3_5] GGUF loaded: {} tensors, {} metadata entries",
            ct.tensor_infos.len(),
            ct.metadata.len()
        );

        let tokenizer = resolve_gguf_tokenizer(&ct, gguf_path)?;

        // EOS: sibling generation_config.json wins (may hold the full multi-id
        // set); fall back to the single id in GGUF metadata, then fill in any
        // other canonical stop token by name (see `merge_canonical_eos_ids`).
        let mut eos_token_ids = read_eos_token_ids(&parent.to_string_lossy());
        if eos_token_ids.is_empty()
            && let Some(id) = ct
                .metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(|v| v.to_u32().ok())
        {
            eos_token_ids.push(id);
        }
        merge_canonical_eos_ids(&mut eos_token_ids, &tokenizer.get_vocab(true));

        let inner = if extended.is_ternary() {
            eprintln!(
                "[qwen3_5] detected Prism ternary GGUF: {} PTQ1_0/PQ2_0 tensors",
                extended.tensors.len()
            );
            Qwen3_5TextModel::from_gguf_extended(ct, &mut cursor, device, extended)?
        } else {
            Qwen3_5TextModel::from_gguf(ct, &mut cursor, device)?
        };
        let dtype = inner.dtype();

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype,
            eos_token_ids,
            inner,
        })
    }

    fn from_pretrained(
        model_path: &str,
        device: &Device,
        dtype: DType,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        // When ISQ is requested, memory is the caller's priority: on Metal,
        // keep the non-quantized side tensors (embedding, norms, conv) in F16
        // instead of the server's F32 default — the 248k-vocab embedding
        // alone is ~1 GB in F32 vs ~0.5 GB in F16.
        let dtype = if quant.is_some() && device.is_metal() && dtype == DType::F32 {
            DType::F16
        } else {
            dtype
        };
        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer not found at {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let filenames = utils::get_safetensors_files(model_path)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device) }?;

        let config_path = std::path::Path::new(model_path).join("config.json");
        let cfg = load_config(config_path.to_str().context("non-UTF8 model path")?)?;

        let eos_token_ids = read_eos_token_ids(model_path);

        if let Some(dt) = quant {
            eprintln!("[qwen3_5] in-situ quantization enabled: {dt:?}");
        }
        let inner = Qwen3_5TextModel::new(&cfg, &vb, device, dtype, quant)?;

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype,
            eos_token_ids,
            inner,
        })
    }

    /// Stop-token ids read from `generation_config.json` / `config.json`
    /// (safetensors) or `tokenizer.ggml.eos_token_id` (GGUF).
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Tokenize a prompt string into input IDs (mirrors `qwen3::Model`).
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails.
    pub fn prepare_inputs(&self, inputs: &str) -> Result<Vec<u32>> {
        let input_ids = self
            .tokenizer
            .tokenizer
            .encode(inputs, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        Ok(input_ids)
    }

    /// Run a single forward step, returning next-token logits `[1, vocab]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> Result<Tensor> {
        let input = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        self.inner.forward(&input, start_pos, None)
    }

    /// Reset all per-layer GDN and attention caches (between unrelated
    /// requests).
    ///
    /// # Errors
    ///
    /// Returns an error if a layer's state could not be reset. This is not
    /// merely in-memory bookkeeping: on an accelerator the reset issues device
    /// memsets, so it fails for the same reasons any other device op does.
    pub fn clear_kv_cache(&mut self) -> Result<()> {
        self.inner.reset_gdn_caches()
    }

    /// Capture the current layer state; see
    /// [`Qwen3_5TextModel::snapshot_state`].
    #[must_use]
    pub fn snapshot_state(&self) -> StateSnapshot {
        self.inner.snapshot_state()
    }

    /// Restore a previously captured state.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot does not match this model's layers.
    pub fn restore_state(&mut self, snap: &StateSnapshot) -> Result<()> {
        self.inner.restore_state(snap)
    }

    pub fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    /// Total bytes held by the full-attention K/V caches (context-scaling term).
    pub fn attn_cache_bytes(&self) -> usize {
        self.inner.attn_cache_bytes()
    }

    /// Warm up the model with a small forward pass.
    pub fn warmup(&mut self) {
        // Run a tiny decode-only forward (single token) so warmup succeeds
        // regardless of how the generate loop is implemented.
        if let Err(e) = self.generate(&[45], &GenerationConfig::with_max_tokens(5), None) {
            eprintln!("warmup failed (non-fatal): {e}");
        }
        if let Err(e) = self.clear_kv_cache() {
            eprintln!("warmup cache reset failed (non-fatal): {e}");
        }
    }
}

impl ModelForCausalLM for Model {
    fn device(&self) -> &Device {
        &self.device
    }

    fn generate(
        &mut self,
        input_ids: &[u32],
        config: &GenerationConfig,
        mut streamer: Option<&mut dyn crate::generation::streamer::TokenStreamer>,
    ) -> Result<Vec<u32>> {
        self.tokenizer.clear();
        self.clear_kv_cache()?;

        let mut logits_processor = LogitsProcessor::new(1024, config.temperature, config.top_p);

        let mut tokens = input_ids.to_vec();
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        // Stop tokens: an explicit `eos_token_id` on the request wins; otherwise
        // use the model's configured EOS set (Qwen3.5/Ornith use multiple, e.g.
        // [248044, 248046]); last-resort, look up `<|im_end|>` in the tokenizer.
        let mut stop_ids: Vec<u32> = match config.eos_token_id {
            Some(e) => vec![e],
            None if !self.eos_token_ids.is_empty() => self.eos_token_ids.clone(),
            None => self.tokenizer.get_token("<|im_end|>").into_iter().collect(),
        };
        stop_ids.sort_unstable();
        stop_ids.dedup();

        // Incremental decode: GDN layers carry recurrent state and
        // full-attention layers carry a K/V cache, so feeding one token per
        // step is correct. `CRANE_FULL_RECOMPUTE=1` forces the O(n²)
        // reset-and-reprocess path instead (kept as a debugging cross-check
        // for the incremental path).
        let full_recompute = std::env::var("CRANE_FULL_RECOMPUTE").is_ok();

        let start_gen = std::time::Instant::now();
        let mut finalized = false;
        for index in 0..config.max_new_tokens {
            let context_size = if index > 0 && !full_recompute {
                1
            } else {
                tokens.len()
            };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];

            if full_recompute {
                self.clear_kv_cache()?;
            }
            let logits = self.forward_step(ctxt, start_pos)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

            // repetition_penalty is compared against the exact default sentinel 1.0,
            // not a computed value, so exact float equality is intentional here.
            #[allow(clippy::float_cmp)]
            let no_repetition_penalty = config.repetition_penalty == 1.;
            let logits = if no_repetition_penalty {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(config.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    config.repetition_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;

            if stop_ids.binary_search(&next_token).is_ok() {
                if let Some(ref mut s) = streamer {
                    s.finalize()?;
                    finalized = true;
                }
                break;
            }

            if let Some(ref mut s) = streamer {
                s.append(next_token)?;
            }
        }

        if !finalized && let Some(ref mut s) = streamer {
            s.finalize()?;
        }

        let dt = start_gen.elapsed();
        if config.report_speed {
            // generated_tokens is a small per-request token count; f64 has ample
            // precision for it, this is purely a display metric.
            #[allow(clippy::cast_precision_loss)]
            let tokens_per_sec = generated_tokens as f64 / dt.as_secs_f64();
            println!("\n{generated_tokens} tokens generated ({tokens_per_sec:.2} token/s)\n");
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::merge_canonical_eos_ids;
    use std::collections::HashMap;

    #[test]
    // A GGUF export that only recorded <|im_end|> (248046) must get
    // <|endoftext|> (248044) added from the vocab, matching Qwen3.5's
    // documented multi-id EOS set.
    fn merge_adds_missing_canonical_id() {
        let mut eos_token_ids = vec![248_046];
        let vocab: HashMap<String, u32> = [
            ("<|im_end|>".to_string(), 248_046),
            ("<|endoftext|>".to_string(), 248_044),
        ]
        .into_iter()
        .collect();
        merge_canonical_eos_ids(&mut eos_token_ids, &vocab);
        assert_eq!(eos_token_ids, vec![248_046, 248_044]);
    }

    #[test]
    // An id already present (from generation_config.json or GGUF metadata)
    // must not be duplicated.
    fn merge_does_not_duplicate_existing_id() {
        let mut eos_token_ids = vec![248_046, 248_044];
        let vocab: HashMap<String, u32> = [
            ("<|im_end|>".to_string(), 248_046),
            ("<|endoftext|>".to_string(), 248_044),
        ]
        .into_iter()
        .collect();
        merge_canonical_eos_ids(&mut eos_token_ids, &vocab);
        assert_eq!(eos_token_ids, vec![248_046, 248_044]);
    }

    #[test]
    // A vocab lacking either canonical name (e.g. a different tokenizer)
    // must leave the existing ids untouched rather than erroring.
    fn merge_is_noop_when_vocab_lacks_canonical_names() {
        let mut eos_token_ids = vec![7];
        let vocab: HashMap<String, u32> = [("<|other|>".to_string(), 1)].into_iter().collect();
        merge_canonical_eos_ids(&mut eos_token_ids, &vocab);
        assert_eq!(eos_token_ids, vec![7]);
    }
}
