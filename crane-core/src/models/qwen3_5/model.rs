//! Top-level Qwen 3.5 text-only transformer + the high-level `Model`
//! wrapper used by the engine (config + weights + tokenizer).

use std::io::Write;

use anyhow::{Context, Error as E, Result};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{embedding, Embedding, Linear, VarBuilder};

use crate::ops::linear::{parse_ggml_dtype, quantize_linear, LinearLayer};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

use super::config::{load_config, Config, HiddenAct, LayerType, RopeParameters, TextConfig};
use super::kv_cache::{KvCache, KvCacheKind};
use super::modeling::{DecoderLayer, MRotaryEmbedding, Qwen35RmsNorm};
use crate::generation::based::ModelForCausalLM;
use crate::generation::GenerationConfig;
use crate::models::hunyuan_dense::modeling::Gguf;
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
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
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

        let embed_tokens = embedding(text_cfg.vocab_size, text_cfg.hidden_size, vb_lm.pp("embed_tokens"))?;

        let layer_types = text_cfg.layer_types();
        let mut layers = Vec::with_capacity(text_cfg.num_hidden_layers);
        for (idx, &layer_type) in layer_types.iter().enumerate() {
            layers.push(DecoderLayer::load(&text_cfg, layer_type, vb_lm.pp("layers").pp(idx), quant)?);
        }

        let norm = Qwen35RmsNorm::load(text_cfg.hidden_size, text_cfg.rms_norm_eps, vb_lm.pp("norm"))?;

        // Resolve the output projection. With `tie_word_embeddings: true`
        // (0.8B/4B) it's the embedding table; untied models (e.g. Ornith-9B)
        // ship a dedicated `lm_head.weight` of shape `[vocab, hidden]`. That
        // tensor lives at the checkpoint ROOT, not under the `language_model`
        // prefix, so probe `vb` first and fall back to `vb_lm`.
        let embed_weight = embed_tokens.embeddings().clone();
        let (lm_head_raw, is_tied) = if cfg.tie_word_embeddings {
            (embed_weight, true)
        } else {
            let shape = (text_cfg.vocab_size, text_cfg.hidden_size);
            let mut tied = false;
            let w = vb.get(shape, "lm_head.weight")
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
            _ => LinearLayer::Standard(Linear::new(lm_head_raw, None)),
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
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        ct: candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // QMatMul handles quantized weights internally; dequantized side
        // tensors (norms, conv kernels, embeddings) use a compute dtype of
        // BF16 on CUDA and F16 on Metal (the F32 embedding alone would cost
        // ~1 GB at Qwen3.5's 248k vocab), F32 on CPU.
        let dtype = if device.is_cuda() {
            DType::BF16
        } else if device.is_metal() {
            DType::F16
        } else {
            DType::F32
        };
        let mut gg = Gguf::new(ct, reader, device.clone(), dtype);

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
                    .filter_map(|v| v.to_i32().ok().map(|x| x.max(0) as usize))
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
            .map(|i| {
                let q_rows = gg
                    .ct
                    .tensor_infos
                    .get(&format!("blk.{i}.attn_q.weight"))
                    .map_or(0, |info| info.shape.dims()[0]);
                let num_heads = md_u32_or(&gg, "attention.head_count", 0);
                q_rows == 2 * num_heads * head_dim
            })
            .unwrap_or(true);

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
                partial_rotary_factor: rot_dim as f64 / head_dim as f64,
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
        };

        let embed_tokens = gg.embedding("token_embd.weight", hidden_size)?;

        let mut layers = Vec::with_capacity(num_hidden_layers);
        for (idx, &layer_type) in layer_types.iter().enumerate() {
            layers.push(DecoderLayer::from_gguf(&text_cfg, layer_type, &mut gg, idx)?);
        }

        let norm = Qwen35RmsNorm::from_folded(
            gg.dequant_tensor("output_norm.weight")?,
            rms_norm_eps,
        );
        let lm_head = if tie_word_embeddings {
            LinearLayer::Standard(Linear::new(embed_tokens.embeddings().clone(), None))
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
        // Project with lm_head on the LAST position only. The engine's
        // sampling step expects `[B, V]` (1D logits for the next token).
        let (b, s, _) = xs.dims3()?;
        let last = xs.narrow(1, s - 1, 1)?.reshape((b, ()))?;
        let logits = self.lm_head.forward(&last)?;
        Ok(logits)
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
            gdn_caches.push(Some(crate::ops::gdn::GdnLayerCache::new(cfg, dtype, device)?));
            attn_caches.push(None);
        } else {
            gdn_caches.push(None);
            attn_caches.push(Some(KvCache::new(kv_kind)));
        }
    }
    Ok((gdn_caches, attn_caches))
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
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        Self::new_with_format(model_path, device, dtype, ModelFormat::Auto)
    }

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
                    .map(|e| e.eq_ignore_ascii_case("gguf"))
                    .unwrap_or(false);
                if is_gguf { ModelFormat::Gguf } else { ModelFormat::Safetensors }
            }
            other => other,
        };
        match format {
            ModelFormat::Safetensors => Self::from_pretrained(model_path, device, dtype, quant),
            ModelFormat::Gguf => {
                if quant.is_some() {
                    eprintln!(
                        "[qwen3_5] --quant/CRANE_ISQ ignored: GGUF weights are already quantized"
                    );
                }
                Self::from_gguf_file(model_path, device)
            }
            ModelFormat::Auto => unreachable!("Auto is resolved above"),
        }
    }

    /// Load from a `.gguf` file. The tokenizer is read from the GGUF itself
    /// (`tokenizer.ggml.tokens` / `tokenizer.ggml.merges` / `token_type`);
    /// a sibling `tokenizer.json` is only consulted if the GGUF lacks the
    /// embedded metadata (older / third-party quantizers).
    fn from_gguf_file(model_path: &str, device: &Device) -> Result<Self> {
        use crate::utils::tokenizer_utils::{
            build_tokenizer_from_gguf_path, gguf_has_embedded_tokenizer,
        };

        let gguf_path = std::path::Path::new(model_path);
        let parent = gguf_path.parent().unwrap_or(gguf_path);

        let mut file = std::fs::File::open(gguf_path)
            .with_context(|| format!("open GGUF file {model_path}"))?;
        let ct = candle_core::quantized::gguf_file::Content::read(&mut file)?;
        eprintln!(
            "[qwen3_5] GGUF loaded: {} tensors, {} metadata entries",
            ct.tensor_infos.len(),
            ct.metadata.len()
        );

        // Prefer the embedded tokenizer; fall back to a sibling tokenizer.json
        // only when the GGUF lacks the necessary metadata.
        let tokenizer = if gguf_has_embedded_tokenizer(&ct) {
            build_tokenizer_from_gguf_path(gguf_path)?
                .ok_or_else(|| anyhow::anyhow!("GGUF reports embedded tokenizer but build returned None"))?
        } else {
            let tokenizer_path = parent.join("tokenizer.json");
            if !tokenizer_path.exists() {
                anyhow::bail!(
                    "GGUF lacks `tokenizer.ggml.tokens`/`merges` metadata and no sibling \
                     tokenizer.json was found at {}. Re-export the model with a current \
                     llama.cpp to get the embedded tokenizer.",
                    tokenizer_path.display()
                );
            }
            eprintln!(
                "[qwen3_5] GGUF has no embedded tokenizer; falling back to {}",
                tokenizer_path.display()
            );
            Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?
        };

        // EOS: sibling generation_config.json wins (may hold the full multi-id
        // set); fall back to the single id in GGUF metadata.
        let mut eos_token_ids = read_eos_token_ids(&parent.to_string_lossy());
        if eos_token_ids.is_empty()
            && let Some(id) = ct
                .metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(|v| v.to_u32().ok())
        {
            eos_token_ids.push(id);
        }

        let inner = Qwen3_5TextModel::from_gguf(ct, &mut file, device)?;
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
        dtype: &DType,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        // When ISQ is requested, memory is the caller's priority: on Metal,
        // keep the non-quantized side tensors (embedding, norms, conv) in F16
        // instead of the server's F32 default — the 248k-vocab embedding
        // alone is ~1 GB in F32 vs ~0.5 GB in F16.
        let dtype = if quant.is_some() && device.is_metal() && *dtype == DType::F32 {
            &DType::F16
        } else {
            dtype
        };
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

        if let Some(dt) = quant {
            eprintln!("[qwen3_5] in-situ quantization enabled: {dt:?}");
        }
        let inner = Qwen3_5TextModel::new(&cfg, vb, device, *dtype, quant)?;

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype: *dtype,
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

