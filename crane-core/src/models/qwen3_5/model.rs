//! Top-level Qwen 3.5 text-only transformer + the high-level `Model`
//! wrapper used by the engine (config + weights + tokenizer).

use std::io::Write;

use anyhow::{Context, Error as E, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{embedding, Embedding, VarBuilder};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

use super::config::{load_config, Config, TextConfig};
use super::kv_cache::{KvCache, KvCacheKind};
use super::modeling::{DecoderLayer, MRotaryEmbedding, Qwen35RmsNorm};
use crate::generation::based::ModelForCausalLM;
use crate::generation::GenerationConfig;
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::utils;

/// Text-only Qwen 3.5 transformer.
///
/// `gdn_caches` is indexed by layer; `None` for full-attention blocks, `Some`
/// for linear-attention blocks. The engine is responsible for cloning/saving
/// these caches across context switches (continuous batching).
pub struct Qwen3_5TextModel {
    cfg: TextConfig,
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: Qwen35RmsNorm,
    lm_head: Tensor,
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
    pub fn new(cfg: &Config, vb: VarBuilder, device: &Device, dtype: DType) -> Result<Self> {
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

        let embed_tokens = embedding(text_cfg.vocab_size, text_cfg.hidden_size, vb_lm.pp("embed_tokens"))?;

        let layer_types = text_cfg.layer_types();
        let mut layers = Vec::with_capacity(text_cfg.num_hidden_layers);
        for (idx, &layer_type) in layer_types.iter().enumerate() {
            layers.push(DecoderLayer::load(&text_cfg, layer_type, vb_lm.pp("layers").pp(idx))?);
        }

        let norm = Qwen35RmsNorm::load(text_cfg.hidden_size, text_cfg.rms_norm_eps, vb_lm.pp("norm"))?;

        // Resolve the output projection. With `tie_word_embeddings: true`
        // (0.8B/4B) it's the embedding table; untied models (e.g. Ornith-9B)
        // ship a dedicated `lm_head.weight` of shape `[vocab, hidden]`. That
        // tensor lives at the checkpoint ROOT, not under the `language_model`
        // prefix, so probe `vb` first and fall back to `vb_lm`.
        let embed_weight = embed_tokens.embeddings().clone();
        let lm_head_raw = if cfg.tie_word_embeddings {
            embed_weight
        } else {
            let shape = (text_cfg.vocab_size, text_cfg.hidden_size);
            vb.get(shape, "lm_head.weight")
                .or_else(|_| vb_lm.get(shape, "lm_head.weight"))
                .or_else(|_| {
                    eprintln!(
                        "[qwen3_5] tie_word_embeddings=false but no lm_head.weight found; \
                         falling back to tied embeddings"
                    );
                    Ok::<_, candle_core::Error>(embed_weight)
                })?
        };
        // Store `lm_head` already transposed (`[hidden, vocab]`) so the per-step
        // matmul doesn't have to.
        let lm_head = lm_head_raw.t()?.contiguous()?;

        let rotary = MRotaryEmbedding::new(&text_cfg, device)?;

        // Pre-allocate per-layer caches: GDN recurrent state for linear blocks,
        // K/V cache for full-attention blocks (mutually exclusive per layer).
        // The K/V representation (fp / int8 / …) is chosen once here.
        let kv_kind = KvCacheKind::from_env();
        let mut gdn_caches = Vec::with_capacity(layers.len());
        let mut attn_caches = Vec::with_capacity(layers.len());
        for layer in &layers {
            if layer.is_linear() {
                gdn_caches.push(Some(crate::ops::gdn::GdnLayerCache::new(
                    &text_cfg, dtype, device,
                )?));
                attn_caches.push(None);
            } else {
                gdn_caches.push(None);
                attn_caches.push(Some(KvCache::new(kv_kind)));
            }
        }

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

    pub fn config(&self) -> &TextConfig {
        &self.cfg
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Total bytes held by the full-attention K/V caches across all layers
    /// (incl. headroom + quant scales). The context-scaling memory term.
    pub fn attn_cache_bytes(&self) -> usize {
        self.attn_caches
            .iter()
            .flatten()
            .map(|c| c.byte_size())
            .sum()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn lm_head(&self) -> &Tensor {
        &self.lm_head
    }

    /// Reset all per-layer GDN caches. Called between unrelated requests
    /// sharing the same pre-allocated layer set.
    pub fn reset_gdn_caches(&mut self) -> Result<()> {
        for slot in self.gdn_caches.iter_mut().flatten() {
            slot.reset()?;
        }
        for slot in self.attn_caches.iter_mut().flatten() {
            slot.reset();
        }
        Ok(())
    }

    /// Forward pass over `input_ids` of shape `[B, S]`. `start_pos` is the
    /// absolute position of the first token (used for rotary slicing).
    /// `attention_mask` is broadcastable to `[B, 1, S, S_total]` (or `None`).
    ///
    /// Returns logits of shape `[B, S, vocab_size]`.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        start_pos: usize,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let is_decode_step = seq_len == 1;

        let mut xs = self.embed_tokens.forward(input_ids)?;

        // If no mask was supplied, build a causal mask so prefill doesn't
        // leak future tokens into past positions. (Decode is `seq_len==1`,
        // where every position trivially attends only to itself — the mask is
        // a no-op there.)
        let mask = match attention_mask {
            Some(m) => Some(m.clone()),
            None if !is_decode_step => {
                let total = start_pos + seq_len;
                Some(build_causal_mask(seq_len, total, xs.device(), xs.dtype())?)
            }
            None => None,
        };
        let mask_ref = mask.as_ref();

        let (cos, sin) = self.rotary.cos_sin(start_pos, seq_len)?;
        let rot_dim = self.rotary.rot_dim();

        for i in 0..self.layers.len() {
            let layer = &self.layers[i];
            let gdn_slot = self.gdn_caches[i].as_mut();
            let attn_slot = self.attn_caches[i].as_mut();
            xs = layer.forward(
                &xs,
                &cos,
                &sin,
                rot_dim,
                mask_ref,
                gdn_slot,
                attn_slot,
                is_decode_step,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        // Matmul with lm_head on the LAST position only. The engine's
        // sampling step expects `[B, V]` (1D logits for the next token).
        let (b, s, _) = xs.dims3()?;
        let last = xs.narrow(1, s - 1, 1)?.reshape((b, ()))?;
        let logits = last.matmul(&self.lm_head)?;
        Ok(logits)
    }
}

/// Build a causal mask of shape `[1, 1, S_q, S_k]` where row `i` has `-inf`
/// for columns `j > start_pos + i` (future tokens). Returned as f32 cast to
/// the model's dtype.
fn build_causal_mask(
    seq_q: usize,
    total_k: usize,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> candle_core::Result<candle_core::Tensor> {
    let mut data: Vec<f32> = Vec::with_capacity(total_k * total_k);
    for q in 0..total_k {
        for k in 0..total_k {
            data.push(if k > q { f32::NEG_INFINITY } else { 0.0 });
        }
    }
    let full = candle_core::Tensor::from_vec(data, (total_k, total_k), device)?
        .to_dtype(dtype)?;
    Ok(full.narrow(0, 0, seq_q)?.unsqueeze(0)?.unsqueeze(0)?)
}

/// Read EOS token id(s) from `generation_config.json` (preferred) then
/// `config.json`. The field may be a single integer or a list; returns an empty
/// vec if absent.
fn read_eos_token_ids(model_path: &str) -> Vec<u32> {
    fn from_value(v: &serde_json::Value) -> Vec<u32> {
        match v {
            serde_json::Value::Number(n) => {
                n.as_u64().map(|x| vec![x as u32]).unwrap_or_default()
            }
            serde_json::Value::Array(a) => {
                a.iter().filter_map(|e| e.as_u64().map(|x| x as u32)).collect()
            }
            _ => Vec::new(),
        }
    }
    for fname in ["generation_config.json", "config.json"] {
        let path = std::path::Path::new(model_path).join(fname);
        let Ok(data) = std::fs::read(&path) else { continue };
        let Ok(json) = serde_json::from_slice::<serde_json::Value>(&data) else { continue };
        if let Some(eos) = json.get("eos_token_id") {
            let ids = from_value(eos);
            if !ids.is_empty() {
                return ids;
            }
        }
    }
    Vec::new()
}


/// Public-facing `Model` for Qwen 3.5 text-only inference.
///
/// Mirrors the structure of `crane_core::models::qwen3::Model`:
/// Format of model weights on disk. Qwen 3.5 currently only ships
/// safetensors; GGUF support is left for a future PR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Auto,
    Safetensors,
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

impl Model {
    /// Load a Qwen 3.5 model from a HF checkpoint directory.
    ///
    /// Only the safetensors path is wired. GGUF support is left as a follow-up.
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        Self::new_with_format(model_path, device, dtype, ModelFormat::Auto)
    }

    pub fn new_with_format(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        format: ModelFormat,
    ) -> Result<Self> {
        let format = match format {
            ModelFormat::Auto => ModelFormat::Safetensors,
            other => other,
        };
        match format {
            ModelFormat::Safetensors => Self::from_pretrained(model_path, device, dtype),
            // GGUF support is a follow-up — the dense checkpoints ship as
            // safetensors, and the GDN path's lazy-eviction story is enough
            // complexity for one PR.
            ModelFormat::Auto => unreachable!("Auto resolves to Safetensors above"),
        }
    }

    fn from_pretrained(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer not found at {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let filenames = utils::get_safetensors_files(model_path)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, *dtype, device) }?;

        let config_path = std::path::Path::new(model_path).join("config.json");
        let cfg = load_config(config_path.to_str().context("non-UTF8 model path")?)?;

        let eos_token_ids = read_eos_token_ids(model_path);

        let inner = Qwen3_5TextModel::new(&cfg, vb, device, *dtype)?;

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype: *dtype,
            eos_token_ids,
            inner,
        })
    }

    /// Tokenize a prompt string into input IDs (mirrors `qwen3::Model`).
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

    /// Run a single forward step, returning raw logits `[1, S, vocab]`.
    pub fn forward_step(
        &mut self,
        input_ids: &[u32],
        start_pos: usize,
    ) -> Result<Tensor> {
        let input = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        Ok(self.inner.forward(&input, start_pos, None)?)
    }

    /// Reset all per-layer GDN caches (between unrelated requests).
    pub fn clear_kv_cache(&mut self) {
        self.inner
            .reset_gdn_caches()
            .expect("GDN cache reset failed");
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
        if let Err(e) = self.generate(
            &[45],
            &GenerationConfig::with_max_tokens(5),
            None,
        ) {
            eprintln!("warmup failed (non-fatal): {e}");
        }
        self.clear_kv_cache();
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
        self.clear_kv_cache();

        let mut logits_processor =
            LogitsProcessor::new(1024, config.temperature, config.top_p);

        let mut tokens = input_ids.to_vec();
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        // Stop tokens: an explicit `eos_token_id` on the request wins; otherwise
        // use the model's configured EOS set (Qwen3.5/Ornith use multiple, e.g.
        // [248044, 248046]); last-resort, look up `<|im_end|>` in the tokenizer.
        let mut stop_ids: Vec<u32> = match config.eos_token_id {
            Some(e) => vec![e],
            None if !self.eos_token_ids.is_empty() => self.eos_token_ids.clone(),
            None => self
                .tokenizer
                .get_token("<|im_end|>")
                .into_iter()
                .collect(),
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
        for index in 0..config.max_new_tokens {
            let context_size = if index > 0 && !full_recompute { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];

            if full_recompute {
                self.clear_kv_cache();
            }
            let logits = self.forward_step(ctxt, start_pos)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

            let logits = if config.repetition_penalty == 1. {
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
                }
                break;
            }

            if let Some(ref mut s) = streamer {
                s.append(next_token)?;
            }
        }

        let dt = start_gen.elapsed();
        if config.report_speed {
            println!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)\n",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
        }
        Ok(tokens)
    }
}

