// SPDX-License-Identifier: MIT

//! The Qwen4-Exp text model: embedding, decoder layers over the
//! hyper-connection streams, final mixer and `lm_head`, with per-sequence
//! caches for one sequence at a time (like the Qwen 3.5 backend).
//!
//! One decoder layer (HF `Qwen4ExpTextDecoderLayer`):
//!
//! ```text
//! streams += PLE(streams, token n-grams)          (the PLE layer only)
//! x, w = attn_mixer.mix(streams)
//! streams = combine(streams, GDN(x) | indexed attention(x), w)
//! x, w = ffn_mixer.mix(streams)
//! streams = combine(streams, MoE(x), w)
//! ```

use std::io::{Read, Seek};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Module, Result, Tensor, bail};
use candle_nn::{Linear, VarBuilder, embedding};

use super::config::{LayerType, TextConfig};
use super::hyper_connection::{GatedResidual, Injection, combine, expand_streams};
use super::indexer::{IndexerCache, QsaIndexer};
use super::ple::{NgramTable, PleLayer, PleState, Residency};
use crate::models::modules::embedding::EmbeddingLayer;
use crate::models::modules::moe::SparseMoeBlock;
use crate::models::qwen3_5::{
    AttentionDims, FullAttention, KvCache, KvCacheKind, MRotaryEmbedding, RopeSlice,
    attn_query_slice,
};
use crate::ops::gdn::{GatedDeltaNet, GdnDims, GdnInputProjectionKind, GdnLayerCache};
use crate::ops::linear::LinearLayer;
use crate::quantized::gguf_file::Gguf;
use crate::utils::prof::{Span, timed};

/// Default prefill chunk. The packed `MoE` decodes every routed expert once
/// per chunk, so larger chunks amortize that (see `ops::quant_iq`), but a
/// chunk's temporaries must fit beside the weights: on a 32 GB Arc Pro B70
/// running Qwen3.8-Flash-Next ~2.7 GiB is left, which a 2048-token chunk
/// overruns and a 1024-token one fits (0.2 GiB to spare at 4k context).
/// `CRANE_PREFILL_CHUNK` overrides it (`0` disables chunking).
const DEFAULT_PREFILL_CHUNK: usize = 1024;

enum TokenMixer {
    Linear {
        gdn: GatedDeltaNet,
        dims: GdnDims,
    },
    Attention {
        attn: FullAttention,
        indexer: QsaIndexer,
    },
}

/// One decoder layer.
pub struct DecoderLayer {
    ple: Option<PleLayer>,
    attn_mixer: GatedResidual,
    mixer: TokenMixer,
    ffn_mixer: GatedResidual,
    moe: SparseMoeBlock,
}

/// Per-sequence state of one layer.
enum LayerCache {
    Linear {
        gdn: GdnLayerCache,
        ple: Option<PleState>,
    },
    Attention {
        kv: KvCache,
        indexer: IndexerCache,
    },
}

/// Rotary tables for the cells a forward call can see.
struct Rope<'a> {
    rotary: &'a MRotaryEmbedding,
    /// `[cells, rot_dim / 2]` for cells `0..start + seq`.
    cos: Tensor,
    sin: Tensor,
}

impl DecoderLayer {
    fn load_hf(cfg: &TextConfig, idx: usize, vb: &VarBuilder) -> Result<Self> {
        let mixer_hf = |name: &str| {
            GatedResidual::load(
                cfg.hidden_size,
                cfg.hc_count,
                cfg.hc_lowrank,
                cfg.rms_norm_eps,
                true,
                &vb.pp(name),
            )
        };
        let mixer = match cfg.layer_types[idx] {
            LayerType::LinearAttention => TokenMixer::Linear {
                gdn: GatedDeltaNet::load(vb.clone(), cfg, GdnInputProjectionKind::Split, None)?,
                dims: GdnDims::new(cfg),
            },
            LayerType::IndexedAttention => {
                let vb = vb.pp("self_attn");
                TokenMixer::Attention {
                    attn: FullAttention::load_dims(
                        attention_dims(cfg),
                        cfg.hidden_size,
                        &vb,
                        None,
                        // QSA indexer mask isn't plain causal; the fused
                        // flash-attn kernel can't represent it.
                        false,
                    )?,
                    indexer: QsaIndexer::load(
                        cfg.indexer()?,
                        cfg.hidden_size,
                        cfg.rms_norm_eps,
                        &vb.pp("indexer"),
                    )?,
                }
            },
        };
        let ple = (cfg.ple_layer() == Some(idx))
            .then(|| PleLayer::load(cfg, 0, &vb.pp("ple")))
            .transpose()?;
        Ok(Self {
            ple,
            attn_mixer: mixer_hf("attn_hyper_connection")?,
            mixer,
            ffn_mixer: mixer_hf("mlp_hyper_connection")?,
            moe: SparseMoeBlock::new(
                &cfg.moe_config(),
                idx,
                cfg.hidden_size,
                vb.pp("mlp"),
                vb.device(),
            )?,
        })
    }

    fn load_gguf<R: Read + Seek>(
        cfg: &TextConfig,
        idx: usize,
        gg: &mut Gguf<R>,
        ple_table: &mut Option<NgramTable>,
        device: &Device,
    ) -> Result<Self> {
        let mixer_gguf = |gg: &mut Gguf<R>, name: &str| {
            GatedResidual::from_gguf(
                gg,
                &format!("blk.{idx}.{name}"),
                cfg.hidden_size,
                cfg.hc_count,
                cfg.rms_norm_eps,
                true,
            )
        };
        let attn_mixer = mixer_gguf(gg, "hc_attn")?;
        let ffn_mixer = mixer_gguf(gg, "hc_ffn")?;
        let mixer = match cfg.layer_types[idx] {
            LayerType::LinearAttention => {
                let (gdn, dims) = GatedDeltaNet::from_gguf(gg, idx, cfg)?;
                TokenMixer::Linear { gdn, dims }
            },
            LayerType::IndexedAttention => TokenMixer::Attention {
                // QSA indexer mask isn't plain causal; the fused flash-attn
                // kernel can't represent it.
                attn: FullAttention::from_gguf_dims(attention_dims(cfg), gg, idx, false)?,
                indexer: QsaIndexer::from_gguf(cfg.indexer()?, cfg.rms_norm_eps, gg, idx)?,
            },
        };
        let ple = if cfg.ple_layer() == Some(idx) {
            let Some(table) = ple_table.take() else {
                bail!("layer {idx} has a PLE module but no n-gram table was found")
            };
            Some(PleLayer::from_gguf(cfg, gg, idx, table)?)
        } else {
            None
        };
        Ok(Self {
            ple,
            attn_mixer,
            mixer,
            ffn_mixer,
            moe: SparseMoeBlock::new_from_gguf(&cfg.moe_config(), gg, idx, device)?,
        })
    }

    fn new_cache(&self, cfg: &TextConfig, dtype: DType, device: &Device) -> Result<LayerCache> {
        Ok(match &self.mixer {
            TokenMixer::Linear { .. } => LayerCache::Linear {
                gdn: GdnLayerCache::new(cfg, dtype, device)?,
                ple: self.ple.as_ref().map(|p| PleState::new(p.hash())),
            },
            TokenMixer::Attention { .. } => LayerCache::Attention {
                kv: KvCache::new(KvCacheKind::from_env()),
                indexer: IndexerCache::default(),
            },
        })
    }

    /// `streams` is `[1, seq, hc * hidden]` for cache positions
    /// `start..start + seq`; `tokens` are those positions' ids.
    fn forward(
        &self,
        streams: &Tensor,
        tokens: &[u32],
        start: usize,
        rope: &Rope<'_>,
        cache: &mut LayerCache,
    ) -> Result<Tensor> {
        let mut streams = streams.clone();
        if let (
            Some(ple),
            LayerCache::Linear {
                ple: Some(state), ..
            },
        ) = (&self.ple, &mut *cache)
        {
            streams = timed(Span::Ple, || -> Result<Tensor> {
                let emb = ple.embed(state, tokens, streams.device())?;
                let delta = ple.forward(&streams.squeeze(0)?, &emb, state)?;
                streams + delta.unsqueeze(0)?
            })?;
        }

        let (x, inject) = timed(Span::Resid, || self.attn_mixer.mix(&streams))?;
        let out = match (&self.mixer, cache) {
            (TokenMixer::Linear { gdn, dims }, LayerCache::Linear { gdn: gdn_cache, .. }) => {
                timed(Span::Gdn, || gdn.forward(&x, dims, gdn_cache))?
            },
            (
                TokenMixer::Attention { attn, indexer },
                LayerCache::Attention { kv, indexer: ic },
            ) => timed(Span::Attn, || {
                indexed_attention(attn, indexer, &x, start, rope, kv, ic)
            })?,
            _ => bail!("layer cache does not match the layer type"),
        };
        let (x, streams, inject) = timed(Span::Resid, || -> Result<_> {
            let streams = combine(&streams, &out, &injection(inject)?)?;
            let (x, inject) = self.ffn_mixer.mix(&streams)?;
            Ok((x, streams, inject))
        })?;
        let out = timed(Span::Mlp, || self.moe.forward(&x))?;
        timed(Span::Resid, || combine(&streams, &out, &injection(inject)?))
    }
}

/// Softmax attention restricted by the QSA indexer, over `x` `[1, seq,
/// hidden]` in slices of [`attn_query_slice`] queries (each slice sees the
/// ones before it through the caches): the indexer builds each slice's mask.
fn indexed_attention(
    attn: &FullAttention,
    indexer: &QsaIndexer,
    x: &Tensor,
    start: usize,
    rope: &Rope<'_>,
    kv: &mut KvCache,
    idx_cache: &mut IndexerCache,
) -> Result<Tensor> {
    let seq = x.dim(1)?;
    let rot_dim = rope.rotary.rot_dim();
    let slice = attn_query_slice(attn.num_heads(), start + seq);
    let mut outs = Vec::with_capacity(seq.div_ceil(slice));
    let mut offset = 0;
    while offset < seq {
        let len = slice.min(seq - offset);
        let xs = x.narrow(1, offset, len)?;
        let pos = start + offset;
        let mask = indexer
            .select(&xs.squeeze(0)?, &rope.cos, &rope.sin, rot_dim, idx_cache)?
            .to_dtype(x.dtype())?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let cos = rope.cos.narrow(0, pos, len)?;
        let sin = rope.sin.narrow(0, pos, len)?;
        let slice = RopeSlice {
            cos: &cos,
            sin: &sin,
            rot_dim,
        };
        outs.push(attn.forward(&xs, slice, None, Some(&mask), Some(kv))?);
        offset += len;
        // Each slice's scores grow with the cache, so the backend's exact-size
        // cache cannot reuse them (see `release_cached_memory`).
        if seq > 1 {
            crate::device::release_cached_memory(x.device());
        }
    }
    Tensor::cat(&outs, 1)
}

/// Block mixers are always loaded with injection weights.
fn injection(injection: Option<Injection>) -> Result<Injection> {
    injection.ok_or_else(|| candle_core::Error::Msg("block mixer has no injection weights".into()))
}

fn attention_dims(cfg: &TextConfig) -> AttentionDims {
    AttentionDims {
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        rms_norm_eps: cfg.rms_norm_eps,
        // Qwen4-Exp always gates the attention output (sigmoid, per head).
        output_gate: true,
    }
}

/// Text-only Qwen4-Exp with the caches of one sequence.
pub struct Qwen4ExpTextModel {
    cfg: TextConfig,
    embed: EmbeddingLayer,
    layers: Vec<DecoderLayer>,
    final_mixer: GatedResidual,
    lm_head: LinearLayer,
    rotary: MRotaryEmbedding,
    caches: Vec<LayerCache>,
    /// Tokens already in the caches.
    seen: usize,
    device: Device,
    dtype: DType,
}

impl Qwen4ExpTextModel {
    /// Load from a HF checkpoint (`model.*` + root `lm_head.weight`). Meant
    /// for small checkpoints: the n-gram table is read densely.
    ///
    /// # Errors
    ///
    /// Returns an error if a weight is missing or has the wrong shape.
    pub fn load_hf(cfg: TextConfig, vb: &VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let lm = vb.pp("model");
        let embed = EmbeddingLayer::Dense(embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            lm.pp("embed_tokens"),
        )?);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| DecoderLayer::load_hf(&cfg, i, &lm.pp(format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let final_mixer = GatedResidual::load(
            cfg.hidden_size,
            cfg.hc_count,
            cfg.hc_lowrank,
            cfg.rms_norm_eps,
            false,
            &lm.pp("hyper_connection_mixer"),
        )?;
        let lm_head = if cfg.tie_word_embeddings {
            lm.pp("embed_tokens")
                .get((cfg.vocab_size, cfg.hidden_size), "weight")?
        } else {
            vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")?
        };
        Self::assemble(
            cfg,
            embed,
            layers,
            final_mixer,
            LinearLayer::Standard(Linear::new(lm_head, None)),
            device,
            dtype,
        )
    }

    /// Load from the first shard of a llama.cpp `qwen4exp` GGUF. The PLE
    /// n-gram table (`per_layer_token_embd.weight`) is memory-mapped from
    /// whichever shard holds it and never uploaded.
    ///
    /// # Errors
    ///
    /// Returns an error if a shard, tensor or metadata key is missing.
    pub fn load_gguf(first_shard: &Path, device: &Device) -> Result<Self> {
        let mmap = crate::quantized::gguf_file::mmap_gguf_file(first_shard)?;
        let (ct, extended) = crate::quantized::extended_gguf::read_content(mmap.as_ref())?;
        let cfg = TextConfig::from_gguf(&ct.metadata)?;
        let mut ple_table = if cfg.ple_layer().is_some() {
            Some(open_ple_table(first_shard, &ct)?)
        } else {
            None
        };
        // Same compute dtypes as the Qwen 3.5 GGUF path.
        let dtype = if device.is_cuda() {
            DType::BF16
        } else if device.is_cpu() {
            DType::F32
        } else {
            DType::F16
        };
        let mut gg = Gguf::new_extended(
            ct,
            std::io::Cursor::new(mmap.as_ref()),
            device.clone(),
            dtype,
            extended,
        )?;
        let embed = gg.quantized_embedding("token_embd.weight", cfg.hidden_size)?;
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| DecoderLayer::load_gguf(&cfg, i, &mut gg, &mut ple_table, device))
            .collect::<Result<Vec<_>>>()?;
        let final_mixer = GatedResidual::from_gguf(
            &mut gg,
            "output_hc",
            cfg.hidden_size,
            cfg.hc_count,
            cfg.rms_norm_eps,
            false,
        )?;
        let lm_head = gg.linear("output.weight")?;
        Self::assemble(
            cfg,
            embed,
            layers,
            final_mixer,
            lm_head,
            device.clone(),
            dtype,
        )
    }

    fn assemble(
        cfg: TextConfig,
        embed: EmbeddingLayer,
        layers: Vec<DecoderLayer>,
        final_mixer: GatedResidual,
        lm_head: LinearLayer,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let rotary = MRotaryEmbedding::from_params(
            cfg.rot_dim(),
            cfg.rope_parameters.rope_theta,
            cfg.max_position_embeddings,
            &cfg.rope_parameters.mrope_section,
            &device,
        )?;
        let caches = layers
            .iter()
            .map(|l| l.new_cache(&cfg, dtype, &device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            cfg,
            embed,
            layers,
            final_mixer,
            lm_head,
            rotary,
            caches,
            seen: 0,
            device,
            dtype,
        })
    }

    #[must_use]
    pub fn config(&self) -> &TextConfig {
        &self.cfg
    }

    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Tokens currently in the caches.
    #[must_use]
    pub fn seen_tokens(&self) -> usize {
        self.seen
    }

    /// Drop every cache, starting a new sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if a cache cannot be allocated.
    pub fn reset(&mut self) -> Result<()> {
        self.caches = self
            .layers
            .iter()
            .map(|l| l.new_cache(&self.cfg, self.dtype, &self.device))
            .collect::<Result<Vec<_>>>()?;
        self.seen = 0;
        Ok(())
    }

    /// Feed the next `tokens` of the sequence and return the logits
    /// (`[vocab]`, F32) after its last token. Long inputs are prefilled in
    /// chunks (`CRANE_PREFILL_CHUNK`, default [`DEFAULT_PREFILL_CHUNK`]).
    ///
    /// # Errors
    ///
    /// Returns an error if `tokens` is empty or a forward step fails.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Tensor> {
        if tokens.is_empty() {
            bail!("forward needs at least one token")
        }
        let chunk = prefill_chunk();
        let step = if chunk == 0 { tokens.len() } else { chunk };
        let mut logits = None;
        for part in tokens.chunks(step) {
            logits = Some(self.forward_chunk(part, true)?);
        }
        logits.ok_or_else(|| candle_core::Error::Msg("forward needs at least one token".into()))
    }

    /// Logits for every position of `tokens` (`[tokens.len(), vocab]`), in one
    /// pass. For tests and scoring; generation uses [`Self::forward`].
    ///
    /// # Errors
    ///
    /// Returns an error if a forward step fails.
    pub fn forward_all(&mut self, tokens: &[u32]) -> Result<Tensor> {
        self.forward_chunk(tokens, false)
    }

    fn forward_chunk(&mut self, tokens: &[u32], last_only: bool) -> Result<Tensor> {
        let timer = crate::utils::prof::pass(tokens.len(), &self.device);
        let out = self.forward_chunk_inner(tokens, last_only);
        if let Some(timer) = timer {
            timer.finish(&self.device);
        }
        out
    }

    fn forward_chunk_inner(&mut self, tokens: &[u32], last_only: bool) -> Result<Tensor> {
        let start = self.seen;
        let seq = tokens.len();
        let mut streams = timed(Span::Embed, || -> Result<Tensor> {
            let ids = Tensor::from_slice(tokens, (1, seq), &self.device)?;
            let embedded = self.embed.forward(&ids)?.to_dtype(self.dtype)?;
            expand_streams(&embedded, self.cfg.hc_count)
        })?;

        let (cos, sin) = self.rotary.cos_sin(0, start + seq)?;
        let rope = Rope {
            rotary: &self.rotary,
            cos,
            sin,
        };
        for (layer, cache) in self.layers.iter().zip(self.caches.iter_mut()) {
            streams = layer.forward(&streams, tokens, start, &rope, cache)?;
            // A prefill layer frees hundreds of MB of temporaries; on SYCL they
            // would otherwise sit in the backend's caches while the ~2 GB left
            // beside the weights runs out (see `release_cached_memory`).
            if seq > 1 {
                crate::device::release_cached_memory(&self.device);
            }
        }
        self.seen += seq;

        timed(Span::Head, || {
            let streams = if last_only {
                streams.narrow(1, seq - 1, 1)?
            } else {
                streams
            };
            let (hidden, _) = self.final_mixer.mix(&streams)?;
            let rows = hidden.dim(1)?;
            let logits = self
                .lm_head
                .forward_logits(&hidden.reshape((rows, self.cfg.hidden_size))?)?
                .to_dtype(DType::F32)?;
            if last_only {
                logits.squeeze(0)
            } else {
                Ok(logits)
            }
        })
    }
}

fn prefill_chunk() -> usize {
    match std::env::var("CRANE_PREFILL_CHUNK") {
        Ok(v) => v.trim().parse().unwrap_or(0),
        Err(_) => DEFAULT_PREFILL_CHUNK,
    }
}

/// The shard holding `per_layer_token_embd.weight`: the first shard itself,
/// or a sibling `-0000K-of-0000N` shard (llama.cpp split naming).
fn open_ple_table(
    first_shard: &Path,
    ct: &candle_core::quantized::gguf_file::Content,
) -> Result<NgramTable> {
    const NAME: &str = "per_layer_token_embd.weight";
    let residency = Residency::from_env()?;
    if ct.tensor_infos.contains_key(NAME) {
        return NgramTable::open_gguf(first_shard, NAME, residency);
    }
    for shard in sibling_shards(first_shard, ct)? {
        if !shard.exists() {
            bail!("GGUF shard {} is missing", shard.display())
        }
        let map = crate::quantized::gguf_file::mmap_gguf_file(&shard)?;
        let (shard_ct, _) = crate::quantized::extended_gguf::read_content(map.as_ref())?;
        if shard_ct.tensor_infos.contains_key(NAME) {
            return NgramTable::open_gguf(&shard, NAME, residency);
        }
    }
    bail!("no shard of {} holds {NAME}", first_shard.display())
}

/// Paths of the other shards of a split GGUF, from `split.count` and the
/// `-00001-of-0000N.gguf` file name.
fn sibling_shards(
    first_shard: &Path,
    ct: &candle_core::quantized::gguf_file::Content,
) -> Result<Vec<PathBuf>> {
    let count = match ct.metadata.get("split.count") {
        Some(v) => u32::from(v.to_u16()?),
        None => 1,
    };
    let name = first_shard
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let suffix = format!("-00001-of-{count:05}.gguf");
    let Some(stem) = name.strip_suffix(&suffix) else {
        bail!("{name} is not the first shard of a {count}-way split GGUF")
    };
    Ok((2..=count)
        .map(|k| first_shard.with_file_name(format!("{stem}-{k:05}-of-{count:05}.gguf")))
        .collect())
}

/// [`Qwen4ExpTextModel`] with its tokenizer and stop tokens, loaded from a
/// GGUF; the unit the serving backend and examples drive.
pub struct Model {
    pub tokenizer: crate::utils::token_output_stream::TokenOutputStream,
    eos_token_ids: Vec<u32>,
    pub inner: Qwen4ExpTextModel,
}

impl Model {
    /// Load from the first shard of a `qwen4exp` GGUF; tokenizer and chat
    /// template come from its metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the model fails to load.
    pub fn from_gguf_file(first_shard: &Path, device: &Device) -> anyhow::Result<Self> {
        use crate::models::qwen3_5::model::{merge_canonical_eos_ids, read_eos_token_ids};
        use crate::utils::tokenizer_utils::resolve_gguf_tokenizer;

        let mmap = crate::quantized::gguf_file::mmap_gguf_file(first_shard)?;
        let (ct, _) = crate::quantized::extended_gguf::read_content(mmap.as_ref())?;
        let tokenizer = resolve_gguf_tokenizer(&ct, first_shard)?;
        let parent = first_shard.parent().unwrap_or(first_shard);
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
        drop(mmap);

        let inner = Qwen4ExpTextModel::load_gguf(first_shard, device)?;
        Ok(Self {
            tokenizer: crate::utils::token_output_stream::TokenOutputStream::new(tokenizer),
            eos_token_ids,
            inner,
        })
    }

    #[must_use]
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Next-token logits `[1, vocab]` after feeding `input_ids` at
    /// `start_pos`. The caches hold one sequence: `start_pos` must continue
    /// it, or be `0` to start a new one.
    ///
    /// # Errors
    ///
    /// Returns an error if `start_pos` neither continues the cached sequence
    /// nor restarts it, or the forward pass fails.
    pub fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> anyhow::Result<Tensor> {
        if start_pos == 0 && self.inner.seen_tokens() > 0 {
            self.inner.reset()?;
        } else if start_pos != self.inner.seen_tokens() {
            anyhow::bail!(
                "qwen4_exp: start_pos {start_pos} does not continue the {} cached tokens",
                self.inner.seen_tokens()
            );
        }
        Ok(self.inner.forward(input_ids)?.unsqueeze(0)?)
    }

    /// # Errors
    ///
    /// Returns an error if the caches cannot be reallocated.
    pub fn clear_kv_cache(&mut self) -> anyhow::Result<()> {
        Ok(self.inner.reset()?)
    }

    /// Native maximum context length this checkpoint was trained/configured for.
    pub fn max_position_embeddings(&self) -> usize {
        self.inner.config().max_position_embeddings
    }

    /// One short forward so kernels are compiled before the first request.
    pub fn warmup(&mut self) {
        if let Err(e) = self.inner.forward(&[45]) {
            eprintln!("[qwen4_exp] warmup failed (non-fatal): {e}");
        }
        if let Err(e) = self.inner.reset() {
            eprintln!("[qwen4_exp] warmup cache reset failed (non-fatal): {e}");
        }
    }
}
