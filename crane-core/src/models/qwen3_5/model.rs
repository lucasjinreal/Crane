//! Top-level Qwen 3.5 text-only transformer + the high-level `Model`
//! wrapper used by the engine (config + weights + tokenizer).

use std::io::Write;
use std::sync::Arc;

use anyhow::{Context, Error as E, Result};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder, embedding};
use ribo::utils::log;

use crate::device::{DeviceAssignment, format_budget, kv_vram_overhead, query_gpu_memory};
use crate::models::modules::expert_cache::{ExpertCache, ExpertCacheBudget, blob_bytes_per_expert};
use crate::ops::linear::{LinearLayer, parse_ggml_dtype, quantize_linear};
use crate::quantized::gguf_metadata::GgufMetadata;
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

/// Longest speculative pass that runs the GDN recurrence on the serial kernel.
const SPEC_SERIAL_MAX: usize = 64;

/// Fixed margin for allocator fragmentation and transient compute
/// allocations on top of an [`ExpertCache`]'s own pool byte budget. Does not
/// cover KV-cache growth — `runtime_reservation_bytes` already does that
/// (see [`Model::seed_expert_cache`]'s caller).
const EXPERT_CACHE_SAFETY_MARGIN_BYTES: u64 = 512 * (1 << 20);

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
    /// Dtype the full-attention layers attend and cache K/V in when it
    /// differs from `dtype` (see [`FullAttention::set_attention_dtype`]).
    ///
    /// [`FullAttention::set_attention_dtype`]: super::FullAttention::set_attention_dtype
    attn_dtype: Option<DType>,
    /// Single-token CUDA graph per run of GDN layers, keyed by the run's first
    /// layer (`CRANE_QWEN35_CUDA_GRAPH=1`); `None` once capture has failed.
    /// Graphs hold no request state, see [`Self::replay_gdn_run`]. GDN env
    /// flags (`CRANE_GDN_FUSED`, …) are read at capture, so they are frozen.
    #[cfg(feature = "cuda")]
    decode_graphs: std::collections::HashMap<usize, Option<crate::ops::cuda_graph::GraphPlan>>,
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
        Self::new_with_expert_device(cfg, vb, device, dtype, quant, device)
    }

    /// Like [`Self::new`], but places `MoE` expert weights on `expert_device`
    /// rather than `device`. Used for `MoE` `CPU` offload (see
    /// [`crate::device::DeviceAssignment`]).
    ///
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing or has an
    /// unexpected shape.
    pub fn new_with_expert_device(
        cfg: &Config,
        vb: &VarBuilder,
        device: &Device,
        dtype: DType,
        quant: Option<GgmlDType>,
        expert_device: &Device,
    ) -> Result<Self> {
        disable_cuda_event_tracking(device);
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
            let mut layer = DecoderLayer::load_with_expert_device(
                &text_cfg,
                idx,
                layer_type,
                vb_lm.pp("layers").pp(idx),
                quant,
                expert_device,
            )?;
            layers.push(layer);
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
            attn_dtype: None,
            #[cfg(feature = "cuda")]
            decode_graphs: Default::default(),
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
        Self::from_gguf_impl(ct, reader, &DeviceAssignment::uniform(device), None)
    }

    /// # Errors
    ///
    /// Returns an error if required GGUF metadata or tensors are missing or malformed.
    pub fn from_gguf_extended<R: std::io::Read + std::io::Seek>(
        ct: candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        device: &Device,
        extended: crate::quantized::extended_gguf::ExtendedGgufInfo,
    ) -> Result<Self> {
        Self::from_gguf_impl(
            ct,
            reader,
            &DeviceAssignment::uniform(device),
            Some(extended),
        )
    }

    /// Like [`Self::from_gguf`], but places `MoE` expert weights on
    /// `devices.expert` rather than `devices.main`. Used for `MoE` `CPU`
    /// offload (see [`DeviceAssignment`]).
    ///
    /// # Errors
    ///
    /// Returns an error if required GGUF metadata or tensors are missing or malformed.
    pub fn from_gguf_with_devices<R: std::io::Read + std::io::Seek>(
        ct: candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        devices: &DeviceAssignment,
    ) -> Result<Self> {
        Self::from_gguf_impl(ct, reader, devices, None)
    }

    /// Like [`Self::from_gguf_extended`], but places `MoE` expert weights on
    /// `devices.expert` rather than `devices.main`.
    ///
    /// # Errors
    ///
    /// Returns an error if required GGUF metadata or tensors are missing or malformed.
    pub fn from_gguf_extended_with_devices<R: std::io::Read + std::io::Seek>(
        ct: candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        devices: &DeviceAssignment,
        extended: crate::quantized::extended_gguf::ExtendedGgufInfo,
    ) -> Result<Self> {
        Self::from_gguf_impl(ct, reader, devices, Some(extended))
    }

    fn from_gguf_impl<R: std::io::Read + std::io::Seek>(
        ct: candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        devices: &DeviceAssignment,
        extended: Option<crate::quantized::extended_gguf::ExtendedGgufInfo>,
    ) -> Result<Self> {
        let device = &devices.main;
        disable_cuda_event_tracking(device);
        // QMatMul handles quantized weights internally; dequantized side
        // tensors (norms, conv kernels) use the compute dtype, while the
        // embedding stays quantized whatever it is. F16 on Metal / ROCm /
        // SYCL, F32 on CPU.
        //
        // On CUDA decode activations are F32: candle's quantized matvec only
        // takes F32, so BF16 activations cost a cast into and out of every
        // linear layer (~700 of a token's ~3000 launches on Ornith-1.5-35B,
        // where decode is launch-bound: 53.4 -> 59.6 tok/s on an RTX 3090).
        // The K/V cache, attention and multi-token (prefill) activations stay
        // BF16 (`attn_dtype`): in F32 they cost enough memory that a 24 GB
        // card ran out at 18k tokens of context where BF16 reaches ~27k.
        let (dtype, attn_dtype) = if device.is_cuda() {
            (DType::F32, Some(DType::BF16))
        } else if device.is_metal() || device.is_rocm() || device.is_sycl() {
            (DType::F16, None)
        } else {
            (DType::F32, None)
        };
        let mut gg = match extended {
            Some(info) => Gguf::new_extended(ct, reader, device.clone(), dtype, info)?,
            None => Gguf::new(ct, reader, device.clone(), dtype),
        };

        let md = GgufMetadata::new(gg.metadata());
        let arch = md
            .opt_string("general.architecture")
            .unwrap_or_else(|| "qwen35".to_string());
        let key = |k: &str| format!("{arch}.{k}");

        let head_dim = md.usize(&key("attention.key_length"))?;
        let hidden_size = md.usize(&key("embedding_length"))?;
        // Newer converters append the MTP draft layer(s) as extra `blk.N`
        // blocks and count them in `block_count`; they are not decoder layers.
        let num_hidden_layers = md.usize(&key("block_count"))?
            - md.opt_usize(&key("nextn_predict_layers")).unwrap_or(0);
        let rms_norm_eps = md
            .opt_f32(&key("attention.layer_norm_rms_epsilon"))
            .map_or(1e-6, f64::from);
        let rope_theta = md
            .opt_f32(&key("rope.freq_base"))
            .map_or(10_000_000.0, f64::from);
        let rot_dim = md
            .opt_usize(&key("rope.dimension_count"))
            .unwrap_or(head_dim / 4);
        let mrope_section: Vec<usize> =
            md.0.get(&key("rope.dimension_sections"))
                .and_then(|v| v.to_vec().ok())
                .map(|vals| {
                    vals.iter()
                        .filter_map(|v| v.to_i32().ok().map(|x| usize::try_from(x).unwrap_or(0)))
                        .collect()
                })
                .unwrap_or_default();
        let num_v_heads = md.usize(&key("ssm.time_step_rank"))?;
        let inner_size = md.usize(&key("ssm.inner_size"))?;

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
                let num_heads = md.opt_usize(&key("attention.head_count")).unwrap_or(0);
                q_rows == 2 * num_heads * head_dim
            });

        let partial_rotary_factor = rot_dim as f64 / head_dim as f64;

        let text_cfg = TextConfig {
            head_dim,
            vocab_size,
            hidden_size,
            // MoE checkpoints have no dense FFN width; their layers load
            // `ffn_gate_inp` + experts instead (see `DecoderLayer::from_gguf`).
            intermediate_size: md.opt_usize(&key("feed_forward_length")).unwrap_or(0),
            num_hidden_layers,
            num_attention_heads: md.usize(&key("attention.head_count"))?,
            num_key_value_heads: md.usize(&key("attention.head_count_kv"))?,
            hidden_act: HiddenAct::Silu,
            max_position_embeddings: md.opt_usize(&key("context_length")).unwrap_or(262_144),
            rms_norm_eps,
            rope_parameters: RopeParameters {
                rope_theta,
                mrope_section,
                partial_rotary_factor,
                mrope_interleaved: true,
            },
            full_attention_interval: md.opt_usize(&key("full_attention_interval")).unwrap_or(4),
            linear_conv_kernel_dim: md.usize(&key("ssm.conv_kernel"))?,
            linear_key_head_dim: md.usize(&key("ssm.state_size"))?,
            linear_value_head_dim: inner_size / num_v_heads,
            linear_num_key_heads: md.usize(&key("ssm.group_count"))?,
            linear_num_value_heads: num_v_heads,
            tie_word_embeddings,
            attn_output_gate,
            // GGUF carries no gate-activation key; the conversion only ever
            // targets the swish gate this code implements.
            output_gate_type: None,
            // Zero for a dense model (key absent or 0).
            num_experts: md.opt_usize(&key("expert_count")).unwrap_or(0),
            num_experts_per_tok: md.opt_usize(&key("expert_used_count")).unwrap_or(0),
            moe_intermediate_size: md
                .opt_usize(&key("expert_feed_forward_length"))
                .unwrap_or(0),
        };

        // The 248k-row table is the single largest dequantization in the
        // checkpoint; keep it in its GGUF format and gather rows on demand.
        let embed_tokens = gg.quantized_embedding("token_embd.weight", hidden_size)?;

        let mut layers = Vec::with_capacity(num_hidden_layers);
        for (idx, &layer_type) in layer_types.iter().enumerate() {
            let mut layer =
                DecoderLayer::from_gguf(&text_cfg, layer_type, &mut gg, idx, &devices.expert)?;
            layer.set_attention_dtype(attn_dtype);
            layers.push(layer);
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
            attn_dtype,
            #[cfg(feature = "cuda")]
            decode_graphs: Default::default(),
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

    /// Dtype attention runs in, and so the dtype of the causal mask.
    #[must_use]
    pub fn attention_dtype(&self) -> DType {
        self.attn_dtype.unwrap_or(self.dtype)
    }

    /// Runs a probe forward pass, live-queries free VRAM, and seeds a
    /// per-expert [`ExpertCache`] sized to fit within `vram_ceiling_bytes`
    /// (minus `runtime_reservation_bytes` and a `MoE` `CPU`-offload
    /// reservation for a `chunk_tokens`-token prefill chunk — see
    /// [`Self::moe_offload_reservation_bytes`]). `Ok(None)` for non-`MoE`
    /// checkpoints. See `crate::models::qwen3::modeling::Qwen3Model::
    /// seed_expert_cache` for the pattern this mirrors.
    ///
    /// # Errors
    ///
    /// Returns an error only if a tensor op unrelated to the probe fails; an
    /// out-of-memory probe forward pass is caught and logged instead,
    /// falling back to `Ok(None)`.
    // One sequential pipeline (probe -> cost estimate -> budget -> seed)
    // sharing local state (`gpu_location`, `moe_layer_count`) throughout;
    // splitting it up would scatter that shared context across several
    // small functions without simplifying the control flow itself.
    #[allow(clippy::too_many_lines)]
    pub fn seed_expert_cache(
        &mut self,
        main_device: &Device,
        vram_ceiling_bytes: u64,
        runtime_reservation_bytes: u64,
        chunk_tokens: usize,
    ) -> Result<Option<Arc<ExpertCache>>> {
        let Some(moe_config) = self.cfg.moe_config() else {
            return Ok(None);
        };
        if matches!(main_device, Device::Cpu) {
            return Ok(None);
        }
        let gpu_location = main_device.location();

        let moe_layer_count = self
            .layers
            .iter()
            .filter(|l| l.moe_block().is_some())
            .count();
        if moe_layer_count == 0 {
            return Ok(None);
        }
        #[allow(clippy::redundant_closure_for_method_calls)]
        let all_packed = self
            .layers
            .iter()
            .all(|l| l.moe_block().is_none_or(|b| b.has_packed_qtensor_experts()));
        if !all_packed {
            eprintln!(
                "[qwen3_5] Expert cache: not every MoE layer stores experts as a packed QTensor \
                 pair (i-quant or dequantized per-expert storage); skipping per-expert caching, \
                 all experts stay on CPU"
            );
            return Ok(None);
        }

        eprintln!(
            "[qwen3_5] Expert cache: running probe forward pass + live VRAM query on \
             {gpu_location:?} before sizing the per-expert cache (may take a few seconds)"
        );
        let probe_ids = Tensor::new(&[45u32, 546, 456], main_device)?.unsqueeze(0)?;
        if let Err(e) = self.forward(&probe_ids, 0, None) {
            let _ = self.reset_gdn_caches();
            eprintln!(
                "[qwen3_5] expert-cache probe forward failed on {gpu_location:?} (non-fatal, \
                 all experts stay on CPU): {e}"
            );
            return Ok(None);
        }
        self.reset_gdn_caches()?;

        let Some((free, total)) = query_gpu_memory(main_device) else {
            eprintln!(
                "[qwen3_5] No live VRAM query available for {gpu_location:?}; skipping expert \
                 cache (all experts stay on CPU)"
            );
            return Ok(None);
        };
        let used = total.saturating_sub(free);
        let ceiling = vram_ceiling_bytes.min(total);
        let remaining = ceiling.saturating_sub(used);
        let moe_offload_reservation = self.moe_offload_reservation_bytes(chunk_tokens, main_device);
        let available = remaining
            .saturating_sub(runtime_reservation_bytes)
            .saturating_sub(moe_offload_reservation);
        eprintln!(
            "[qwen3_5] Live VRAM on {gpu_location:?} after probe: free={}, total={}, used={}, \
             available_for_experts={} (configured limit={}, moe_offload_reservation={})",
            format_budget(free),
            format_budget(total),
            format_budget(used),
            format_budget(available),
            format_budget(vram_ceiling_bytes),
            format_budget(moe_offload_reservation),
        );

        let (gate_up_sources, down_sources): (Vec<Arc<QTensor>>, Vec<Arc<QTensor>>) = self
            .layers
            .iter()
            .filter_map(|l| {
                let block = l.moe_block()?;
                Some((
                    block.packed_gate_up_source()?.clone(),
                    block.packed_down_source()?.clone(),
                ))
            })
            .unzip();
        debug_assert_eq!(gate_up_sources.len(), moe_layer_count);

        let gate_up_dims = gate_up_sources[0].shape().dims().to_vec();
        let down_dims = down_sources[0].shape().dims().to_vec();
        let gate_up_mismatch = gate_up_sources
            .iter()
            .position(|t| t.shape().dims() != gate_up_dims);
        let down_mismatch = down_sources
            .iter()
            .position(|t| t.shape().dims() != down_dims);
        if let Some(i) = gate_up_mismatch {
            eprintln!(
                "[qwen3_5] Expert cache: the {i}th MoE layer's gate_up projection has shape \
                 {:?}, but the first MoE layer's is shape {gate_up_dims:?}; ExpertCache requires \
                 a uniform tensor shape across layers, skipping per-expert caching, all experts \
                 stay on CPU",
                gate_up_sources[i].shape().dims(),
            );
            return Ok(None);
        }
        if let Some(i) = down_mismatch {
            eprintln!(
                "[qwen3_5] Expert cache: the {i}th MoE layer's down projection has shape {:?}, \
                 but the first MoE layer's is shape {down_dims:?}; ExpertCache requires a \
                 uniform tensor shape across layers, skipping per-expert caching, all experts \
                 stay on CPU",
                down_sources[i].shape().dims(),
            );
            return Ok(None);
        }

        let gate_up_blob_bytes = gate_up_sources
            .iter()
            .map(|t| blob_bytes_per_expert(t))
            .max()
            .unwrap_or(0);
        let down_blob_bytes = down_sources
            .iter()
            .map(|t| blob_bytes_per_expert(t))
            .max()
            .unwrap_or(0);

        let Some(budget) = ExpertCacheBudget::compute(
            available,
            EXPERT_CACHE_SAFETY_MARGIN_BYTES,
            gate_up_blob_bytes,
            down_blob_bytes,
            moe_layer_count,
            moe_config.num_experts,
        ) else {
            eprintln!(
                "[qwen3_5] Expert cache: {} available after reservations is too little for even \
                 one slot per layer ({moe_layer_count} layers, {} bytes/expert); all experts \
                 stay on CPU",
                format_budget(available),
                format_budget((gate_up_blob_bytes + down_blob_bytes) as u64),
            );
            return Ok(None);
        };

        let cache = Arc::new(ExpertCache::new(
            budget,
            gate_up_sources,
            down_sources,
            main_device,
        )?);

        for layer in &mut self.layers {
            if let Some(block) = layer.moe_block_mut() {
                block.set_expert_cache(cache.clone());
            }
        }

        eprintln!(
            "[qwen3_5] Expert cache: seeded {} of {} experts/layer on {gpu_location:?} across \
             {moe_layer_count} MoE layers",
            budget.slots_per_layer, moe_config.num_experts,
        );
        Ok(Some(cache))
    }

    /// Drops every `MoE` layer's [`ExpertCache`] reference, reverting to
    /// all-CPU expert dispatch. Call before [`Self::seed_expert_cache`] to
    /// re-run it with an updated policy, so the old cache's GPU memory is
    /// freed before the new one's live VRAM query runs.
    pub(crate) fn clear_expert_cache(&mut self) {
        for layer in &mut self.layers {
            if let Some(block) = layer.moe_block_mut() {
                block.clear_expert_cache();
            }
        }
    }

    /// Worst-case transient VRAM one in-flight `MoE` CPU-offload call needs
    /// for a `chunk_tokens`-token prefill chunk. 0 if every `MoE` layer is
    /// already GPU-resident, this isn't a `MoE` model, or `main_device`
    /// doesn't support the GPU offload burst (CUDA/ROCm only).
    #[must_use]
    pub(crate) fn moe_offload_reservation_bytes(
        &self,
        chunk_tokens: usize,
        main_device: &Device,
    ) -> u64 {
        if !(main_device.is_cuda() || main_device.is_rocm()) {
            return 0;
        }
        let hidden_size = self.cfg.hidden_size;
        self.layers
            .iter()
            .filter_map(|l| {
                l.moe_block()?
                    .offload_reservation_bytes(chunk_tokens, hidden_size)
            })
            .max()
            .unwrap_or(0)
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
                self.attention_dtype(),
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

    /// Speculative-decoding verification pass over `hidden_states`
    /// `[1, S, hidden]` (already embedded). Unlike [`Self::forward_embeds`] it
    /// returns logits for the **last `n_logits` positions** and the
    /// post-layer hidden states at `tap_layers` for **every** position.
    ///
    /// Every GDN layer snapshots its recurrent state after the first
    /// `checkpoint_at` tokens; [`Self::commit_spec`] later restores it. That is
    /// what lets a verification block be `[already-accepted tokens] + [drafts]`
    /// without the rejected drafts leaving their mark on the recurrent state.
    ///
    /// `serial_gdn` runs the GDN recurrence on the serial kernel (one launch,
    /// exact f32) when the block is short; pass `false` for prefill so it uses
    /// the same kernels as a plain prefill.
    ///
    /// Returns `(logits [n_logits, vocab] f32, taps [S, tap_layers.len() * hidden])`.
    ///
    /// # Errors
    ///
    /// Returns an error if any layer, the head, or the tap concatenation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_spec(
        &mut self,
        hidden_states: &Tensor,
        position_ids: &Tensor,
        start_pos: usize,
        checkpoint_at: usize,
        tap_layers: &[usize],
        n_logits: usize,
        serial_gdn: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (_b, seq_len, _h) = hidden_states.dims3()?;
        let timer = crate::utils::prof::pass(seq_len, hidden_states.device());
        for slot in self.gdn_caches.iter_mut().flatten() {
            slot.checkpoint_at = Some(checkpoint_at);
            slot.checkpoint = None;
            // A verification block is short: one serial launch beats the
            // chunked formulation's dozens (long prefill chunks keep it).
            slot.prefer_serial = serial_gdn && seq_len <= SPEC_SERIAL_MAX;
        }
        let mask = super::prefill::causal_mask(
            seq_len,
            start_pos,
            hidden_states.device(),
            self.attention_dtype(),
        )?;
        let (cos, sin) = self.rotary.cos_sin_with_position_ids(position_ids)?;
        let rope = RopeSlice {
            cos: &cos,
            sin: &sin,
            rot_dim: self.rotary.rot_dim(),
        };
        let mut xs = hidden_states.clone();
        if xs.dtype() != self.attention_dtype() {
            xs = xs.to_dtype(self.attention_dtype())?;
        }
        let mut taps = Vec::with_capacity(tap_layers.len());
        for i in 0..self.layers.len() {
            let layer = &self.layers[i];
            let gdn_slot = self.gdn_caches[i].as_mut();
            let attn_slot = self.attn_caches[i].as_mut();
            xs = layer.forward(&xs, rope, Some(&mask), gdn_slot, attn_slot)?;
            if tap_layers.contains(&i) {
                taps.push(xs.clone());
            }
        }
        let taps = Tensor::cat(&taps, 2)?.squeeze(0)?;
        let n = n_logits.min(seq_len);
        if n == 0 {
            let empty = Tensor::zeros((0, 1), DType::F32, xs.device())?;
            if let Some(t) = timer {
                t.finish(xs.device());
            }
            return Ok((empty, taps));
        }
        let tail = xs.narrow(1, seq_len - n, n)?.contiguous()?;
        let logits = crate::utils::prof::timed(crate::utils::prof::Span::Head, || {
            let normed = self.norm.forward(&tail)?.reshape((n, ()))?;
            self.lm_head.forward_logits(&normed)
        })?;
        if let Some(t) = timer {
            t.finish(xs.device());
        }
        Ok((logits.to_dtype(DType::F32)?, taps))
    }

    /// Finish a [`Self::forward_spec`] round: every GDN layer reverts to its
    /// checkpoint and every full-attention K/V cache is truncated to
    /// `attn_len` tokens, i.e. the model state is "everything before
    /// `checkpoint_at`" again.
    pub fn commit_spec(&mut self, attn_len: usize) {
        for slot in self.gdn_caches.iter_mut().flatten() {
            slot.take_checkpoint();
        }
        for slot in self.attn_caches.iter_mut().flatten() {
            slot.truncate(attn_len);
        }
    }

    /// `lm_head` alone (no final norm) over rows `[N, hidden]` -> `[N, vocab]`.
    /// For drafters that share the target's output head.
    ///
    /// # Errors
    ///
    /// Returns an error if the head matmul fails.
    pub fn lm_head_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        Ok(self.lm_head.forward_logits(hidden)?)
    }

    /// Run every decoder layer over already-embedded hidden states `xs`,
    /// returning the pre-final-norm hidden states.
    fn run_layers(
        &mut self,
        mut xs: Tensor,
        rope: RopeSlice<'_>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Multi-token passes are bound by memory and GEMMs, not launches, so
        // they keep the half-precision activations (see `attn_dtype`).
        if xs.dim(1)? > 1 && xs.dtype() != self.attention_dtype() {
            xs = xs.to_dtype(self.attention_dtype())?;
        }
        let debug = std::env::var_os("CRANE_QWEN35_DEBUG_LAYERS").is_some();
        #[cfg(feature = "cuda")]
        let graphs = xs.dim(1)? == 1
            && xs.dim(0)? == 1
            && !debug
            && xs.device().is_cuda()
            && !crate::utils::prof::enabled()
            && std::env::var("CRANE_QWEN35_CUDA_GRAPH").is_ok_and(|v| v == "1");
        let mut i = 0;
        while i < self.layers.len() {
            #[cfg(feature = "cuda")]
            if graphs && let Some((end, out)) = self.replay_gdn_run(i, &xs, rope)? {
                xs = out;
                i = end;
                continue;
            }
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
            i += 1;
        }
        Ok(xs)
    }

    /// Run the GDN layers `start..end` (up to the next full-attention layer) as
    /// one CUDA graph, capturing it on first use. The graph's inputs are `xs`
    /// and each layer's conv/recurrent state and its outputs the new ones, so
    /// resets, snapshots and state swaps need no invalidation. Returns `None`
    /// to run the layers eagerly: not a GDN run, a speculative checkpoint or
    /// serial pass pending, the prefill-to-decode conv-state cast still due, or
    /// capture failed. Capture on a thread that outlives the model: candle's
    /// kernel-param cache the graph reads from is thread-local.
    #[cfg(feature = "cuda")]
    fn replay_gdn_run(
        &mut self,
        start: usize,
        xs: &Tensor,
        rope: RopeSlice<'_>,
    ) -> Result<Option<(usize, Tensor)>> {
        if start > 0 && self.layers[start - 1].is_linear() {
            return Ok(None);
        }
        let n = self.layers.len();
        let end = (start..n)
            .find(|&j| !self.layers[j].is_linear())
            .unwrap_or(n);
        let caches: Vec<_> = self.gdn_caches[start..end]
            .iter()
            .flatten()
            .cloned()
            .collect();
        if caches.is_empty()
            || caches.iter().any(|c| {
                c.checkpoint_at.is_some() || c.prefer_serial || c.conv_state.dtype() != xs.dtype()
            })
        {
            return Ok(None);
        }
        let mut inputs = vec![xs];
        for c in &caches {
            inputs.extend([&c.conv_state, &c.recurrent_state]);
        }
        if let Some(Some(g)) = self.decode_graphs.get(&start)
            && !g.matches(&inputs)
        {
            self.decode_graphs.remove(&start);
        }
        if !self.decode_graphs.contains_key(&start) {
            let layers = &self.layers[start..end];
            // Safety: the captured layers' weights live as long as `self`, and
            // their decode path reads no host values.
            let plan = unsafe {
                crate::ops::cuda_graph::GraphPlan::capture(&inputs, |v| {
                    let mut xs = v[0].clone();
                    let mut out = vec![];
                    for (j, (layer, c)) in layers.iter().zip(&caches).enumerate() {
                        let mut c = crate::ops::gdn::GdnLayerCache {
                            conv_state: v[2 * j + 1].clone(),
                            recurrent_state: v[2 * j + 2].clone(),
                            ..c.clone()
                        };
                        xs = layer.forward(&xs, rope, None, Some(&mut c), None)?;
                        out.extend([c.conv_state, c.recurrent_state]);
                    }
                    out.insert(0, xs);
                    Ok(out)
                })
            };
            if let Err(e) = &plan {
                eprintln!(
                    "[qwen3_5] CUDA graph for layers {start}..{end} failed, running them eagerly: {e}"
                );
            }
            self.decode_graphs.insert(start, plan.ok());
        }
        let Some(Some(plan)) = self.decode_graphs.get_mut(&start) else {
            return Ok(None);
        };
        let mut out = plan.replay(&inputs)?.into_iter();
        let xs = out.next().expect("graph returns xs");
        for c in self.gdn_caches[start..end].iter_mut().flatten() {
            c.conv_state = out.next().expect("graph returns conv state");
            c.recurrent_state = out.next().expect("graph returns recurrent state");
        }
        Ok(Some((end, xs)))
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
pub(crate) fn merge_canonical_eos_ids(
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
pub(crate) fn read_eos_token_ids(model_path: &str) -> Vec<u32> {
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
        Self::new_with_format_and_devices(
            model_path,
            &DeviceAssignment::uniform(device),
            dtype,
            format,
        )
    }

    /// Like [`Self::new_with_format`], but `devices.expert` places `MoE`
    /// expert weights on a different device than the rest of the model
    /// (`MoE` `CPU` offload); callers that don't need that should pass
    /// [`DeviceAssignment::uniform`].
    ///
    /// # Errors
    ///
    /// Returns an error if the model or tokenizer cannot be loaded from `model_path`.
    pub fn new_with_format_and_devices(
        model_path: &str,
        devices: &DeviceAssignment,
        dtype: &DType,
        format: ModelFormat,
    ) -> Result<Self> {
        Self::new_with_options_and_devices(model_path, devices, dtype, format, isq_from_env())
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
        Self::new_with_options_and_devices(
            model_path,
            &DeviceAssignment::uniform(device),
            dtype,
            format,
            quant,
        )
    }

    /// Like [`Self::new_with_options`], but `devices.expert` places `MoE`
    /// expert weights on a different device than the rest of the model
    /// (`MoE` `CPU` offload); callers that don't need that should pass
    /// [`DeviceAssignment::uniform`].
    ///
    /// # Errors
    ///
    /// Returns an error if the model or tokenizer cannot be loaded from `model_path`.
    pub fn new_with_options_and_devices(
        model_path: &str,
        devices: &DeviceAssignment,
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
            ModelFormat::Safetensors => Self::from_pretrained(model_path, devices, *dtype, quant),
            ModelFormat::Gguf => {
                if quant.is_some() {
                    eprintln!(
                        "[qwen3_5] --quant/CRANE_ISQ ignored: GGUF weights are already quantized"
                    );
                }
                Self::from_gguf_file(model_path, devices)
            },
            ModelFormat::Auto => unreachable!("Auto is resolved above"),
        }
    }

    /// Load from a `.gguf` file. The tokenizer is read from the GGUF itself
    /// (`tokenizer.ggml.tokens` / `tokenizer.ggml.merges` / `token_type`);
    /// a sibling `tokenizer.json` is only consulted if the GGUF lacks the
    /// embedded metadata (older / third-party quantizers).
    fn from_gguf_file(model_path: &str, devices: &DeviceAssignment) -> Result<Self> {
        let device = &devices.main;
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
            && let Some(id) = GgufMetadata::new(&ct.metadata).opt_u32("tokenizer.ggml.eos_token_id")
        {
            eos_token_ids.push(id);
        }
        merge_canonical_eos_ids(&mut eos_token_ids, &tokenizer.get_vocab(true));

        let inner = if !extended.is_empty() {
            if extended.is_ternary() {
                eprintln!(
                    "[qwen3_5] detected Prism ternary GGUF: {} PTQ1_0/PQ2_0 tensors",
                    extended.tensors.len()
                );
            }
            Qwen3_5TextModel::from_gguf_extended_with_devices(ct, &mut cursor, devices, extended)?
        } else {
            Qwen3_5TextModel::from_gguf_with_devices(ct, &mut cursor, devices)?
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
        devices: &DeviceAssignment,
        dtype: DType,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let device = &devices.main;
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
        let inner = Qwen3_5TextModel::new_with_expert_device(
            &cfg,
            &vb,
            device,
            dtype,
            quant,
            &devices.expert,
        )?;

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

    /// Native maximum context length this checkpoint was trained/configured for.
    pub fn max_position_embeddings(&self) -> usize {
        self.inner.config().max_position_embeddings
    }

    /// Total bytes held by the full-attention K/V caches (context-scaling term).
    pub fn attn_cache_bytes(&self) -> usize {
        self.inner.attn_cache_bytes()
    }

    /// Bytes of KV cache one sequence consumes per generated token. Counts
    /// only full-attention layers — GDN layers carry constant-size state
    /// instead (see [`TextConfig::kv_bytes_per_token`]). Uses the KV cache's
    /// actual storage dtype ([`Qwen3_5TextModel::attention_dtype`]), not the
    /// model's compute dtype — on CUDA these differ (BF16 cache, F32 compute).
    #[must_use]
    pub fn kv_bytes_per_token(&self) -> u64 {
        self.inner
            .config()
            .kv_bytes_per_token(self.inner.attention_dtype().size_in_bytes())
    }

    /// Runs a probe forward pass, live-queries free VRAM, and seeds a
    /// per-expert [`ExpertCache`] sized to fit within `vram_ceiling_bytes`
    /// (minus a KV-cache reservation derived from `max_concurrent`/
    /// `max_seq_len`, and minus a `MoE` `CPU`-offload reservation sized for a
    /// `chunk_tokens`-token prefill chunk). `Ok(None)` for non-`MoE`
    /// checkpoints.
    ///
    /// The returned cache must still be driven by the caller: advance its
    /// token clock once per scheduler step via [`ExpertCache::on_step`], or
    /// placement never improves past the arbitrary initial seed.
    ///
    /// # Errors
    ///
    /// Returns an error only if a tensor op unrelated to the probe fails; an
    /// out-of-memory probe forward pass is caught and logged instead,
    /// falling back to `Ok(None)`.
    pub fn seed_expert_cache(
        &mut self,
        vram_ceiling_bytes: u64,
        max_concurrent: Option<usize>,
        max_seq_len: Option<usize>,
        chunk_tokens: usize,
    ) -> Result<Option<Arc<ExpertCache>>> {
        let config = self.inner.config();
        let max_concurrent = max_concurrent.unwrap_or(1);
        let effective_seq_len = max_seq_len
            .filter(|&n| n > 0)
            .unwrap_or(config.max_position_embeddings) as u64;
        let kv_dtype_bytes = self.inner.attention_dtype().size_in_bytes();
        let kv_storage = effective_seq_len * config.kv_bytes_per_token(kv_dtype_bytes);
        let runtime_reservation_bytes = kv_vram_overhead(kv_storage, max_concurrent);
        log::tracing::trace!(
            "KV reservation: vram_ceiling={}, max_concurrent={max_concurrent}, \
             effective_seq_len={effective_seq_len}, kv_storage={} \
             (batch_factor={}, safety_margin={}), runtime_reservation={}",
            format_budget(vram_ceiling_bytes),
            format_budget(kv_storage),
            crate::device::kv_batch_factor(max_concurrent),
            format_budget(crate::device::KV_SAFETY_MARGIN_BYTES),
            format_budget(runtime_reservation_bytes),
        );
        self.inner.seed_expert_cache(
            &self.device,
            vram_ceiling_bytes,
            runtime_reservation_bytes,
            chunk_tokens,
        )
    }

    /// Drops every `MoE` layer's [`ExpertCache`] reference, reverting to
    /// all-CPU expert dispatch. Call before [`Self::seed_expert_cache`] to
    /// re-run it with an updated policy, so the old cache's GPU memory is
    /// freed before the new one's live VRAM query runs.
    pub fn clear_expert_cache(&mut self) {
        self.inner.clear_expert_cache();
    }

    /// Worst-case transient VRAM one in-flight `MoE` CPU-offload call needs
    /// for a `chunk_tokens`-token prefill chunk.
    #[must_use]
    pub fn moe_offload_reservation_bytes(&self, chunk_tokens: usize) -> u64 {
        self.inner
            .moe_offload_reservation_bytes(chunk_tokens, &self.device)
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
            let tokens_per_sec = generated_tokens as f64 / dt.as_secs_f64();
            println!("\n{generated_tokens} tokens generated ({tokens_per_sec:.2} token/s)\n");
        }
        Ok(tokens)
    }
}

/// Turn off candle's per-tensor CUDA event tracking (cudarc records and waits
/// on an event around every tensor access, for multi-stream safety). Crane
/// runs each model on one stream, so the events are pure overhead: on
/// MiniCPM-V-4.7-35B-A3B they were ~31k driver calls (~8.5 ms of CPU) per
/// DSpark round. The other decoders do the same (`EventTrackingGuard`).
fn disable_cuda_event_tracking(device: &Device) {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(dev) = device {
        if dev.is_event_tracking() {
            // Safety: all work for this model is issued on the device's single
            // stream, so ordering is already guaranteed without events.
            unsafe { dev.disable_event_tracking() };
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = device;
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Shape;
    use candle_nn::Init;
    use candle_nn::var_builder::SimpleBackend;
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

    /// Random weights for any requested tensor name: good enough for a
    /// construction-only test that never runs a forward pass.
    struct RandWeights;

    impl SimpleBackend for RandWeights {
        fn get(
            &self,
            s: Shape,
            _name: &str,
            _h: Init,
            dtype: DType,
            dev: &Device,
        ) -> candle_core::Result<Tensor> {
            let fan_in = s.dims().last().copied().unwrap_or(1) as f64;
            Tensor::randn(0f32, (1.0 / fan_in.sqrt()) as f32, s, dev)?.to_dtype(dtype)
        }

        fn get_unchecked(&self, _n: &str, _d: DType, _v: &Device) -> candle_core::Result<Tensor> {
            candle_core::bail!("RandWeights requires a shape")
        }

        fn contains_tensor(&self, _name: &str) -> bool {
            false
        }
    }

    /// Small hybrid MoE config: 8 layers, 2 full-attention (interval 4),
    /// every layer `MoE` (`num_experts: 2`).
    fn moe_tiny_config() -> Config {
        Config {
            text_config: TextConfig {
                head_dim: 16,
                vocab_size: 64,
                hidden_size: 32,
                intermediate_size: 48,
                num_hidden_layers: 8,
                num_attention_heads: 4,
                num_key_value_heads: 2,
                hidden_act: HiddenAct::Silu,
                max_position_embeddings: 128,
                rms_norm_eps: 1e-6,
                rope_parameters: RopeParameters {
                    rope_theta: 10_000.0,
                    mrope_section: Vec::new(),
                    partial_rotary_factor: 0.5,
                    mrope_interleaved: true,
                },
                full_attention_interval: 4,
                linear_conv_kernel_dim: 4,
                linear_key_head_dim: 8,
                linear_value_head_dim: 8,
                linear_num_key_heads: 2,
                linear_num_value_heads: 4,
                tie_word_embeddings: true,
                attn_output_gate: true,
                output_gate_type: None,
                num_experts: 2,
                num_experts_per_tok: 1,
                moe_intermediate_size: 32,
            },
            vision_config: None,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
            tie_word_embeddings: true,
        }
    }

    // On a CPU-only model, `seed_expert_cache` must return `Ok(None)`
    // without panicking (no live VRAM query is possible on CPU, so it
    // takes the early-return path before running the probe forward pass).
    #[test]
    fn seed_expert_cache_is_noop_on_cpu() {
        let cfg = moe_tiny_config();
        let dev = Device::Cpu;
        let vb = VarBuilder::from_backend(Box::new(RandWeights), DType::F32, dev.clone());
        let mut model = Qwen3_5TextModel::new(&cfg, &vb, &dev, DType::F32, None).expect("build");

        let cache = model
            .seed_expert_cache(&dev, 1 << 30, 0, 2048)
            .expect("seed_expert_cache");
        assert!(cache.is_none());
    }
}

#[cfg(all(test, feature = "cuda"))]
#[path = "fla_decode_tests.rs"]
mod fla_decode_tests;
