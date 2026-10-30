//! `MuScriptor` `LMModel` + weight loading + chunked transcription driver.
//!
//! Three public surfaces:
//!
//! * [`LMModel`] — the architecture (embedding, streaming transformer,
//!   out norm, lm head, conditioner). `forward` and `generate` only
//!   touch tensors; no WAV, no MIDI.
//! * [`Model`] — owns `LMModel` + the MT3 tokenizer; the bridge between
//!   the architecture and one 5-second chunk. Loads weights from a
//!   `config.json` + `model.safetensors` directory with the
//!   legacy-`emb.0.` → `emb.` key remap from the upstream.
//! * [`TranscriptionModel`] — owns `Model` and orchestrates chunked
//!   transcription: WAV load → per-chunk mel conditioner → autoregressive
//!   generate (with tie-prologue forcing across chunk boundaries) → event
//!   decode → Standard MIDI File writer. `transcribe_to_midi` handles
//!   audio of any length, splitting it into `SEGMENT_DURATION` chunks.
//!
//! **Not implemented**: classifier-free guidance (`cfg_coef != 1.0` is
//! rejected at generation time — the upstream runs a doubled cond/uncond
//! batch through a doubled KV cache, which this port's single-sequence
//! `LMModel::generate` doesn't support) and beam search (greedy/sampled
//! decoding only). Neither changes the architecture; both are plausible
//! follow-ups if quality at the low-CFG end of the model's published
//! numbers turns out to matter.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::quantized::GgmlDType;
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Embedding, LayerNorm, VarBuilder, embedding, layer_norm};

use crate::models::muscriptor::conditioner::{
    ClassConditioner, ConditioningAttributes, ConditioningProvider, MelSpectrogramConditioner,
    WavCondition,
};
use crate::models::muscriptor::config::{SAMPLE_RATE, SEGMENT_DURATION, VariantConfig};
use crate::models::muscriptor::midi::{MidiNote, MidiWriter};
use crate::models::muscriptor::mt3::{
    DRUM_PROGRAM, EOS_ID, MT3Tokenizer, Token, instrument_group_from_names,
    resolve_instrument_names,
};
use crate::models::muscriptor::transformer::{StreamingTransformer, TransformerState};
use crate::ops::linear::LinearLayer;
use crate::utils::float_cast::{f32_to_usize, f64_to_usize};

// ── Public note/event stream types ────────────────────────────────────

/// One decoded (and time-stamped) MIDI event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoteEvent {
    Pitched {
        is_drum: bool,
        program: u8,
        pitch: u8,
        time: f32,
        velocity: u8,
    },
    Tie,
    Eos,
}

/// Public transcription controls. Mirrors the upstream's `transcribe`
/// keyword arguments.
///
/// `cfg_coef` (classifier-free guidance) is not implemented — the upstream
/// runs a doubled cond/uncond batch through a doubled KV cache, which this
/// port's single-sequence `LMModel::generate` doesn't support. Values other
/// than `1.0` are rejected at generation time rather than silently ignored.
#[derive(Debug, Clone)]
pub struct TranscribeConfig {
    pub instruments: Option<Vec<String>>,
    pub hard_mask_instruments: bool,
    pub use_sampling: bool,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub cfg_coef: f32,
    pub max_gen_len: usize,
    /// RNG seed for sampled decoding (`use_sampling = true`). Ignored under
    /// greedy decoding, which is already deterministic.
    pub seed: u64,
}

impl Default for TranscribeConfig {
    fn default() -> Self {
        Self {
            instruments: None,
            hard_mask_instruments: false,
            use_sampling: false,
            temperature: 1.0,
            top_k: 0,
            top_p: 0.0,
            cfg_coef: 1.0,
            max_gen_len: 2000,
            seed: 0,
        }
    }
}

/// Build the [`candle_transformers::generation::LogitsProcessor`]
/// `TranscribeConfig`'s sampling knobs describe. Mirrors the upstream's
/// `sample_from_probs` branch order: top-p first, then top-k, then plain
/// temperature; greedy argmax when sampling is off or `temperature <= 0`.
fn build_logits_processor(
    config: &TranscribeConfig,
) -> candle_transformers::generation::LogitsProcessor {
    use candle_transformers::generation::{LogitsProcessor, Sampling};
    let sampling = if !config.use_sampling || config.temperature <= 0.0 {
        Sampling::ArgMax
    } else if config.top_p > 0.0 {
        Sampling::TopP {
            p: f64::from(config.top_p),
            temperature: f64::from(config.temperature),
        }
    } else if config.top_k > 0 {
        Sampling::TopK {
            k: config.top_k,
            temperature: f64::from(config.temperature),
        }
    } else {
        Sampling::All {
            temperature: f64::from(config.temperature),
        }
    };
    LogitsProcessor::from_sampling(config.seed, sampling)
}

// ── ScaledEmbedding ────────────────────────────────────────────────────

struct ScaledEmbedding {
    inner: Embedding,
    zero_idx: i64,
}

impl ScaledEmbedding {
    fn new(num_embeddings: usize, embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner = embedding(num_embeddings, embedding_dim, vb)?;
        Ok(Self {
            inner,
            zero_idx: -1,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Map the `zero_idx` sentinel (negatives) to zero vectors:
        // look up via clamped indices, then zero out the rows whose
        // original value matched the sentinel.
        let safe = input.clamp(0, i64::MAX)?;
        let y = self.inner.forward(&safe)?;
        let is_zero = input.eq(self.zero_idx)?.to_dtype(y.dtype())?;
        // `is_zero` is rank-2 `[B, S]`; `y` is rank-3 `[B, S, D]`.
        // Broadcast-multiplying keeps `where_cond` honest.
        let keep = is_zero.affine(-1.0, 1.0)?; // 1 where real, 0 where zero
        let out = y.broadcast_mul(&keep.unsqueeze(D::Minus1)?)?;
        Ok(out)
    }
}

type ConditionTensors = std::collections::HashMap<String, (Tensor, Tensor)>;

// ── LMModel ────────────────────────────────────────────────────────────

pub struct LMModel {
    emb: ScaledEmbedding,
    out_norm: LayerNorm,
    linear: LinearLayer,
    transformer: StreamingTransformer,
    /// LM head output dim. Always equals the published `card` (1393 for
    /// small, 1395 for medium/large). For medium/large the trailing
    /// `[1393..card)` logits are "reserved / OOV" slots that must
    /// never be sampled — see `vocab_size` for the tokenizer-derived
    /// cap used in the keep-mask.
    card: usize,
    embed_dim: usize,
    /// Cached off `vb.device()` at construction — `generate()` needs it to
    /// build the BOS/decode-step tensors, and `LinearLayer` (unlike the
    /// `candle_nn::Linear` it replaced for quantization support) doesn't
    /// expose a `.weight()` to read it back off.
    device: Device,
}

impl LMModel {
    /// `quant` in-situ quantizes every big `Linear` (attention in/out
    /// projections, FFN, LM head) to the given GGML level; embeddings and
    /// norms always stay in the surrounding compute dtype (matching the
    /// Qwen 3.5 ISQ convention — quantizing those would only add memory,
    /// not save it, since lookups need the dequantized values anyway).
    ///
    /// # Errors
    ///
    /// Returns an error if building the embedding, transformer, output norm,
    /// or LM head linear fails, typically from missing or mismatched weights
    /// in `vb`.
    pub fn new(vb: &VarBuilder, config: &VariantConfig, quant: Option<GgmlDType>) -> Result<Self> {
        let device = vb.device().clone();
        let emb = ScaledEmbedding::new(config.card + 1, config.dim, vb.pp("emb"))?;
        let transformer = StreamingTransformer::new(&vb.pp("transformer"), config, quant)?;
        let out_norm = layer_norm(config.dim, 1e-5, vb.pp("out_norm"))?;
        let linear =
            crate::ops::linear::linear_layer(config.dim, config.card, vb.pp("linear"), quant)?;
        Ok(Self {
            emb,
            out_norm,
            linear,
            transformer,
            card: config.card,
            embed_dim: config.dim,
            device,
        })
    }

    /// Run the full stack on `sequence` (`[B, S]`). On `first_step` only,
    /// `condition_tensors` (each `(embed [B, T_i, D], mask [B, T_i])`) is
    /// prepended to the embedded sequence — matching the upstream's
    /// `first_step` gate in `LMModel.forward`. Every subsequent decode step
    /// must NOT re-prepend: the conditioning was already written into the
    /// KV cache during the prefill call, so re-concatenating it here would
    /// both corrupt the cache (the write cursor only advances by the new
    /// token count, not by the re-sent prefix) and reprocess the whole
    /// ~500-frame mel prefix through every layer on every single decode
    /// step. Returns `[B, S, card]` logits. The caller's `state` is
    /// mutated to carry the new prefix forward.
    ///
    /// # Errors
    ///
    /// Returns an error if any tensor op in the embed/prepend/transformer/
    /// projection pipeline fails, e.g. from a shape mismatch between
    /// `sequence` and a condition tensor.
    pub fn forward(
        &self,
        sequence: &Tensor,
        condition_tensors: &ConditionTensors,
        first_step: bool,
        state: &mut TransformerState,
    ) -> Result<Tensor> {
        let (_, s) = sequence.dims2()?;
        let mut input = self.emb.forward(sequence)?;

        let mut prepend_total = 0;
        if first_step {
            for (cond, _mask) in condition_tensors.values() {
                let t = cond.dim(1)?;
                prepend_total += t;
                // Move to the model's device (mel conditioner is built on
                // CPU for `rustfft`; class conditioner is built on the
                // model's VarBuilder device but kept as-is here) and
                // cast to the same dtype as the embedded tokens so `cat`
                // accepts both sides.
                let cond_d = cond.to_device(input.device())?.to_dtype(input.dtype())?;
                input = Tensor::cat(&[&cond_d, &input], 1)?;
            }
        }

        let transformer_out = self.transformer.forward(&input, state)?;
        let transformer_out = if prepend_total > 0 {
            transformer_out.narrow(1, prepend_total, s)?
        } else {
            transformer_out
        };
        let transformer_out = self.out_norm.forward(&transformer_out)?;
        let logits = self.linear.forward(&transformer_out)?;
        Ok(logits)
    }

    /// Autoregressive generation, single sequence (batch = 1) only.
    ///
    /// `prompt` is teacher-forced: it's fed straight into the prefill
    /// alongside the BOS token (not sampled), and echoed verbatim at the
    /// front of the returned sequence. Multi-chunk transcription uses this
    /// for the tie-prologue that declares notes sustained from the previous
    /// chunk (see `TranscriptionModel::transcribe_to_midi`); pass `&[]` for
    /// a fresh chunk with nothing to force.
    ///
    /// `logits_processor` controls greedy vs. sampled decoding (construct it
    /// with `Sampling::ArgMax` for the old greedy-only behavior).
    ///
    /// `forbidden_tokens` (optional) is applied every step as a constant
    /// -inf mask added to the raw logits before sampling.
    ///
    /// Returns up to `max_gen_len` token ids (`prompt.len()` of them are the
    /// echoed prompt); stops early if EOS is sampled, in which case EOS is
    /// the last id returned. Total length is at most `max_gen_len` — pass
    /// a prompt no longer than that or the prefill has nothing left to
    /// generate.
    ///
    /// # Errors
    ///
    /// Returns an error if any condition tensor's batch dim isn't 1, if
    /// `prompt.len() >= max_gen_len`, or if any tensor op during prefill or
    /// decoding fails.
    pub fn generate(
        &self,
        condition_tensors: &ConditionTensors,
        prompt: &[u32],
        max_gen_len: usize,
        logits_processor: &mut candle_transformers::generation::LogitsProcessor,
        forbidden_tokens: Option<&[u32]>,
        state: &mut TransformerState,
    ) -> Result<Vec<u32>> {
        let device = self.device.clone();
        let card = self.card;
        let vocab_size = crate::models::muscriptor::mt3::CARD;

        // Additive `[card]` mask: 0.0 where a token may be sampled,
        // -inf for forbidden ids and for the trailing reserved/OOV slots
        // `[vocab_size..card)` that only medium/large have (so those can
        // never be sampled either). Built once, reused every step.
        let mut forbid_add = vec![0f32; card];
        for slot in forbid_add.iter_mut().skip(vocab_size) {
            *slot = f32::NEG_INFINITY;
        }
        if let Some(forbidden) = forbidden_tokens {
            for &t in forbidden {
                if let Some(slot) = forbid_add.get_mut(t as usize) {
                    *slot = f32::NEG_INFINITY;
                }
            }
        }
        let forbid_add = Tensor::from_vec(forbid_add, (card,), &device)?;

        let batch = condition_tensors
            .values()
            .next()
            .map_or(1, |(c, _)| c.dim(0).unwrap_or(1));
        anyhow::ensure!(
            batch == 1,
            "LMModel::generate only supports batch=1, got {batch}"
        );
        anyhow::ensure!(
            prompt.len() < max_gen_len,
            "prompt ({} tokens) leaves nothing to generate within max_gen_len ({})",
            prompt.len(),
            max_gen_len
        );

        let sample = |logits: &Tensor,
                      lp: &mut candle_transformers::generation::LogitsProcessor|
         -> Result<u32> {
            let last = logits
                .narrow(1, logits.dim(1)? - 1, 1)?
                .squeeze(1)?
                .squeeze(0)?;
            let masked = last.to_dtype(DType::F32)?.broadcast_add(&forbid_add)?;
            Ok(lp.sample(&masked)?)
        };

        // Prefill: [BOS, prompt...] plus all prepended conditions.
        // `first_step=true` is what makes `forward` prepend the conditions
        // at all — every following decode step passes `false`, both to
        // skip re-prepending (already in the cache) and to hit the cheap
        // `T_q == 1` attention fast path instead of reprocessing the whole
        // mel prefix.
        let prepend_total: usize = condition_tensors
            .values()
            .map(|(cond, _mask)| cond.dim(1))
            .collect::<candle_core::Result<Vec<_>>>()?
            .into_iter()
            .sum();
        let mut prefill_ids: Vec<i64> = Vec::with_capacity(1 + prompt.len());
        prefill_ids.push(self.card as i64);
        prefill_ids.extend(prompt.iter().map(|&t| i64::from(t)));
        let prefill_t = Tensor::from_vec(prefill_ids.clone(), (batch, prefill_ids.len()), &device)?;
        let prefill_logits = self.forward(&prefill_t, condition_tensors, true, state)?;
        // The cache absorbed `prepend_total` condition rows plus
        // `prefill_ids.len()` BOS+prompt rows; advancing by only 1 here
        // (the old bug) left the cursor pointing ~500 positions short of
        // what was actually written, so every later step's cache
        // write/read collided with the prefix.
        self.transformer
            .advance(state, prepend_total + prefill_ids.len());

        let mut generated: Vec<u32> = prompt.to_vec();
        let mut next = sample(&prefill_logits, logits_processor)?;
        generated.push(next);

        while generated.len() < max_gen_len && next != EOS_ID {
            let input = Tensor::from_vec(vec![i64::from(next)], (1usize, 1usize), &device)?;
            let logits = self.forward(&input, condition_tensors, false, state)?;
            self.transformer.advance(state, 1);
            next = sample(&logits, logits_processor)?;
            generated.push(next);
        }

        Ok(generated)
    }

    #[must_use]
    pub fn dim(&self) -> usize {
        self.embed_dim
    }

    #[must_use]
    pub fn card(&self) -> usize {
        self.card
    }
}

// ── Model (weights + helpers) ──────────────────────────────────────────

pub struct Model {
    pub(super) inner: LMModel,
    pub(super) conditioner: Arc<ConditioningProvider>,
    tokenizer: MT3Tokenizer,
    device: Device,
    dtype: DType,
}

impl Model {
    /// Load `model.safetensors` + `config.json` from `model_dir`.
    /// `dtype` is the compute dtype for the transformer only — see
    /// [`Self::new_with_options`].
    ///
    /// # Errors
    ///
    /// See [`Self::new_with_options`].
    pub fn new(model_dir: &str, device: &Device, dtype: DType) -> Result<Self> {
        Self::new_with_options(model_dir, device, dtype, None)
    }

    /// `quant` in-situ quantizes the transformer's linear projections
    /// (`--quant q4k|q8_0|…`, same levels as Qwen 3.5's ISQ); `None` keeps
    /// them in `dtype`.
    ///
    /// `dtype` governs the transformer (embedding table, attention/FFN
    /// projections, LM head) — pass `F16`/`BF16` to roughly halve its VRAM
    /// even without `quant`, or combine both for a bigger cut. The
    /// conditioning pipeline (mel-spectrogram projection, class embeddings)
    /// **always loads and computes in F32 regardless of `dtype`** — log-mel
    /// of quiet passages underflows in fp16, and it's a tiny fraction of
    /// total weights (≈1.6M params) so there's nothing worth saving there
    /// anyway. `LMModel::forward` casts the conditioner's F32 output to the
    /// transformer's dtype right before splicing it in, so this costs
    /// nothing at the seam.
    ///
    /// # Errors
    ///
    /// Returns an error if `config.json` or `model.safetensors` can't be
    /// read or parsed, if the checkpoint has more than one codebook
    /// (`n_q > 1`, unsupported), if a required conditioner tensor (mel
    /// filterbank or class-embedding weights) is missing, or if building the
    /// conditioners or [`LMModel`] fails.
    pub fn new_with_options(
        model_dir: &str,
        device: &Device,
        dtype: DType,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let dir = Path::new(model_dir);
        let cfg_bytes = std::fs::read(dir.join("config.json"))
            .with_context(|| format!("read {}/config.json", model_dir))?;
        let config = VariantConfig::from_json_bytes(&cfg_bytes)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("parse {}/config.json", model_dir))?;

        // Read all tensors to CPU via candle's safetensors loader, then
        // rewrite the legacy `emb.0.*` / `linears.0.*` keys to `emb.*`
        // / `linear.*`. The upstream flattens an `nn.ModuleList([Embedding])`
        // to a single-element list under the `emb.0.` prefix; published
        // checkpoints are single-stream so the `.0.` is always present
        // and uniquely maps.
        let st_path = dir.join("model.safetensors");
        let st_bytes =
            std::fs::read(&st_path).with_context(|| format!("read {}", st_path.display()))?;
        let raw = candle_core::safetensors::load_buffer(&st_bytes, &Device::Cpu)
            .with_context(|| format!("parse {}", st_path.display()))?;

        if raw
            .keys()
            .any(|k| k.starts_with("emb.1.") || k.starts_with("linears.1."))
        {
            anyhow::bail!(
                "checkpoint has more than one codebook (n_q > 1); \
                 only single-stream models are supported"
            );
        }

        // Kept in the checkpoint's native dtype here (F32 for every
        // published variant) — dtype conversion happens per-destination
        // below (transformer vs. always-F32 conditioners), not here.
        let mut tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();
        for (name, t) in raw {
            let mut new_name = name;
            if let Some(rest) = new_name.strip_prefix("emb.0.") {
                new_name = format!("emb.{rest}");
            } else if let Some(rest) = new_name.strip_prefix("linears.0.") {
                new_name = format!("linear.{rest}");
            }
            tensors.insert(new_name, t);
        }

        // Pull out the non-parameter buffers the VarBuilder can't
        // reach through `pp(...)` paths before handing the rest off.
        // Forced to F32 regardless of the checkpoint's on-disk dtype or
        // the caller's requested `dtype` — these feed the conditioners,
        // which must always run in F32 (see `new_with_options` doc).
        let fb_name = "condition_provider.conditioners.self_wav.mel_spec_transform.mel_scale.fb";
        let filterbank = tensors
            .remove(fb_name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor {fb_name}"))?
            .to_dtype(DType::F32)?;
        let lg_name = "condition_provider.conditioners.instrument_group.embed.weight";
        let lg_emb = tensors
            .remove(lg_name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor {lg_name}"))?
            .to_dtype(DType::F32)?;
        let ds_name = "condition_provider.conditioners.dataset_name.embed.weight";
        let ds_emb = tensors
            .remove(ds_name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor {ds_name}"))?
            .to_dtype(DType::F32)?;

        // Two views over the same (cheaply `Tensor::clone`d — Arc bump, no
        // data copy) weight map: `vb` casts to the transformer's compute
        // `dtype` on fetch, `vb_f32` always casts to F32. Only the mel
        // conditioner's `output_proj.{weight,bias}` are fetched through
        // `vb_f32` (everything else the conditioners need was already
        // pulled out above as raw F32 tensors); every other key goes
        // through `vb`. `VarBuilder::from_tensors`' backend forces every
        // fetched tensor to the builder's own configured dtype regardless
        // of what's actually stored in the map, so this split — not a
        // per-tensor dtype in the map — is what keeps the conditioner path
        // off the transformer's (possibly quantized/half-precision) dtype.
        let vb_f32 = VarBuilder::from_tensors(tensors.clone(), DType::F32, device);
        let vb = VarBuilder::from_tensors(tensors, dtype, device);

        let mut mel = std::collections::HashMap::new();
        let mut class = std::collections::HashMap::new();
        mel.insert(
            "self_wav".to_string(),
            Arc::new(MelSpectrogramConditioner::new(
                config.dim,
                device,
                DType::F32,
                &vb_f32.pp("condition_provider.conditioners.self_wav"),
                filterbank,
            )?),
        );
        class.insert(
            "instrument_group".to_string(),
            Arc::new(ClassConditioner::from_embedding_tensor(
                1000, config.dim, &lg_emb,
            )?),
        );
        class.insert(
            "dataset_name".to_string(),
            Arc::new(ClassConditioner::from_embedding_tensor(
                4, config.dim, &ds_emb,
            )?),
        );
        let conditioner = Arc::new(ConditioningProvider::new(mel, class));

        let inner = LMModel::new(&vb, &config, quant)?;
        Ok(Self {
            inner,
            conditioner,
            tokenizer: MT3Tokenizer::new(),
            device: device.clone(),
            dtype,
        })
    }

    /// Resample `samples` from `in_sr` to [`SAMPLE_RATE`] Hz (linear).
    #[must_use]
    pub fn resample_to_target(samples: &[f32], in_sr: u32) -> Vec<f32> {
        if in_sr == SAMPLE_RATE as u32 {
            return samples.to_vec();
        }
        let ratio = SAMPLE_RATE as f64 / in_sr as f64;
        let out_len = f64_to_usize((samples.len() as f64 * ratio).round());
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src = i as f64 / ratio;
            let i0 = f64_to_usize(src.floor());
            let i1 = (i0 + 1).min(samples.len().saturating_sub(1));
            let frac = (src - i0 as f64) as f32;
            out.push(samples[i0] * (1.0 - frac) + samples[i1] * frac);
        }
        out
    }

    #[must_use]
    pub fn inner(&self) -> &LMModel {
        &self.inner
    }

    #[must_use]
    pub fn conditioner(&self) -> &Arc<ConditioningProvider> {
        &self.conditioner
    }

    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[must_use]
    pub fn tokenizer(&self) -> &MT3Tokenizer {
        &self.tokenizer
    }

    /// Convenience: instrument-group token IDs that must be masked.
    #[must_use]
    pub fn forbidden_tokens_for(&self, instruments: &[&str]) -> Vec<u32> {
        self.tokenizer.forbidden_token_ids(instruments)
    }

    /// # Errors
    ///
    /// Returns an error if an entry in `names` matches more than one known
    /// instrument (ambiguous prefix) or matches none, in which case the
    /// message suggests close matches.
    pub fn resolve_instrument_names(names: &[&str]) -> Result<Vec<String>, String> {
        resolve_instrument_names(names)
    }

    /// # Errors
    ///
    /// Returns an error listing the valid instrument names if any entry in
    /// `names` isn't a recognized MT3 instrument name.
    pub fn instrument_group_from_names(names: &[&str]) -> Result<String, String> {
        instrument_group_from_names(names)
    }

    #[must_use]
    pub const fn default_max_gen_len() -> usize {
        2000
    }

    fn wav_condition(&self, samples: &[f32]) -> Result<WavCondition> {
        let wav = Tensor::from_vec(samples.to_vec(), (1, samples.len()), &self.device)?
            .to_dtype(self.dtype)?
            .unsqueeze(0)?;
        let length = Tensor::from_vec(vec![samples.len() as u32], (1,), &self.device)?;
        Ok(WavCondition {
            wav,
            length,
            sample_rate: vec![SAMPLE_RATE as u32],
        })
    }

    fn build_attrs(
        &self,
        samples: &[f32],
        sample_rate: u32,
        instrument_group: Option<String>,
    ) -> Result<ConditioningAttributes> {
        let target = if sample_rate != SAMPLE_RATE as u32 {
            Self::resample_to_target(samples, sample_rate)
        } else {
            samples.to_vec()
        };
        let wav_cond = self.wav_condition(&target)?;
        Ok(ConditioningAttributes {
            wav: vec![("self_wav".to_string(), wav_cond)],
            text: vec![
                ("instrument_group".to_string(), instrument_group),
                ("dataset_name".to_string(), None),
            ],
        })
    }
}

// ── TranscriptionModel ────────────────────────────────────────────────

pub struct TranscriptionModel {
    model: Model,
}

impl TranscriptionModel {
    #[must_use]
    pub fn new(model: Model) -> Self {
        Self { model }
    }

    /// # Errors
    ///
    /// See [`Model::new`].
    pub fn load(model_dir: &str, device: &Device, dtype: DType) -> Result<Self> {
        let m = Model::new(model_dir, device, dtype)?;
        Ok(Self::new(m))
    }

    /// `quant` in-situ quantizes the transformer's linear projections —
    /// see [`Model::new_with_options`].
    ///
    /// # Errors
    ///
    /// See [`Model::new_with_options`].
    pub fn load_with_options(
        model_dir: &str,
        device: &Device,
        dtype: DType,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let m = Model::new_with_options(model_dir, device, dtype, quant)?;
        Ok(Self::new(m))
    }

    #[must_use]
    pub fn inner(&self) -> &Model {
        &self.model
    }

    /// Transcribe a single ≤ 5-second chunk of audio into a Standard
    /// MIDI File byte buffer. Audio longer than 5 seconds is
    /// truncated; shorter is zero-padded.
    ///
    /// Note reconstruction is a *very* simplified pass: every emitted
    /// `Pitch` token becomes one short `MidiNote` (onset = current
    /// frame time, offset = onset + 0.1 s). Drum tokens become
    /// drum-channel onsets. This matches the upstream's most basic
    /// decoding path and produces audibly right output for monophonic
    /// solo instruments; a full upstream-equivalent reconstruction
    /// (program + pitch + velocity → onset/offset pairing) is left as
    /// a follow-up PR.
    /// Transcribe audio of any length into a Standard MIDI File byte
    /// buffer. Audio is split into consecutive `SEGMENT_DURATION` (5 s)
    /// chunks (the last is zero-padded); each chunk gets an independent
    /// forward pass — the model has no cross-chunk KV cache — but chunk
    /// boundaries are kept coherent via *tie-prologue forcing*: the
    /// `(program, pitch)` pairs still sounding at the end of chunk N are
    /// teacher-forced as chunk N+1's opening tokens (mirrors the upstream's
    /// `prelude_forcing=True` default), so a note straddling a boundary
    /// stays attributed to the same instrument instead of the model
    /// re-guessing it.
    ///
    /// # Errors
    ///
    /// Returns an error if `config.cfg_coef != 1.0`, if any requested
    /// instrument name is unrecognized, or if building conditioning tensors,
    /// initializing transformer state, or generation fails for any chunk.
    pub fn transcribe_to_midi(
        &self,
        samples: &[f32],
        sample_rate: u32,
        config: &TranscribeConfig,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            config.cfg_coef == 1.0,
            "cfg_coef={} requested but classifier-free guidance isn't implemented \
             (LMModel::generate is single-sequence only); use 1.0",
            config.cfg_coef
        );

        let m = &self.model;
        let full = if sample_rate != SAMPLE_RATE as u32 {
            Model::resample_to_target(samples, sample_rate)
        } else {
            samples.to_vec()
        };

        let segment_samples = f32_to_usize(SEGMENT_DURATION * SAMPLE_RATE as f32);
        let num_chunks = full.len().div_ceil(segment_samples).max(1);

        let instrument_group = if let Some(names) = &config.instruments {
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            Some(Model::instrument_group_from_names(&refs).map_err(anyhow::Error::msg)?)
        } else {
            None
        };
        let forbidden = if config.hard_mask_instruments {
            config
                .instruments
                .as_ref()
                .map(|names| {
                    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
                    m.forbidden_tokens_for(&refs)
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let forbidden = if forbidden.is_empty() {
            None
        } else {
            Some(&forbidden[..])
        };

        let mut logits_processor = build_logits_processor(config);
        let mut tracker = OpenNoteTracker::new();
        let mut notes: Vec<MidiNote> = Vec::new();

        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * segment_samples;
            let end = (start + segment_samples).min(full.len());
            let mut chunk = full[start..end].to_vec();
            if chunk.len() < segment_samples {
                chunk.resize(segment_samples, 0.0);
            }

            let attrs = m.build_attrs(&chunk, SAMPLE_RATE as u32, instrument_group.clone())?;
            let conditions = m.conditioner().forward(&attrs)?;

            // Fresh chunk 0 has nothing sustained yet; every later chunk
            // forces the previous one's still-open notes as its prologue.
            let prompt: Vec<u32> = if chunk_idx == 0 {
                Vec::new()
            } else {
                crate::models::muscriptor::mt3::tie_section_token_ids(
                    &m.tokenizer,
                    &tracker.open_keys(),
                )
            };

            // Exact upper bound on how many KV rows this chunk's `generate`
            // call will ever write: the prepended mel/class conditioning
            // (~500 frames for a 5 s chunk) + the BOS token + up to
            // `max_gen_len` prompt-or-generated tokens. Sizing this off
            // `max_gen_len` alone (dropping the conditioning prepend) is
            // exactly the overflow this comment is warning against — it
            // shipped once already and blew up as "KV cache overflow: tried
            // to write at offset 2020, max is 2019" on a 10-chunk piece.
            let prepend_total: usize = conditions
                .values()
                .map(|(cond, _mask)| cond.dim(1))
                .collect::<candle_core::Result<Vec<_>>>()?
                .into_iter()
                .sum();
            let mut state = m.inner.transformer.init_state(
                1,
                prepend_total + 1 + config.max_gen_len,
                m.dtype,
                m.device(),
            )?;
            let generated = m.inner.generate(
                &conditions,
                &prompt,
                config.max_gen_len,
                &mut logits_processor,
                forbidden,
                &mut state,
            )?;

            let chunk_start_secs = chunk_idx as f32 * SEGMENT_DURATION;
            // The last chunk has no following chunk to bleed events into;
            // every other chunk drops any event whose local time drifted
            // past its own 5 s window (mirrors the upstream's
            // `next_seek_time` bound in `OpenNoteTracker.feed`).
            let chunk_end_bound = (chunk_idx + 1 < num_chunks).then_some(SEGMENT_DURATION);

            tracker.start_chunk();
            for (tok, secs) in m.tokenizer.iter(&generated) {
                if matches!(tok, Token::Eos) {
                    break;
                }
                tracker.feed(
                    tok,
                    secs as f32,
                    chunk_start_secs,
                    chunk_end_bound,
                    &mut notes,
                );
            }
        }
        tracker.finish(&mut notes);

        let writer = MidiWriter::new();
        Ok(writer.write(&notes))
    }
}

/// Minimum duration (seconds) assigned to a drum hit's synthetic offset, or
/// to any note still open at the end of the stream. Matches the upstream's
/// `MINIMUM_NOTE_DURATION_SEC`.
const MIN_NOTE_DURATION_SECS: f32 = 0.01;

/// Cross-chunk open-note state machine: turns the raw MT3 token stream
/// (Program/Velocity as *state*, Pitch as the *trigger* that opens or
/// closes a note — see the module-level notes on `LMModel::generate`) into
/// timestamped [`MidiNote`]s, carrying still-sounding notes across chunk
/// boundaries. A reduced port of the upstream `OpenNoteTracker`
/// (events.py) for this port's `prelude_forcing=True`, batch-size-1 case:
/// since every non-initial chunk's tie prologue is *teacher-forced* from
/// `open_keys()` (not model-generated), the upstream's tie-set bookkeeping
/// — which exists to reconcile a *guessed* prologue against reality — is
/// unnecessary here: prologue and reality are identical by construction, so
/// nothing ever needs to close purely because of a chunk transition.
struct OpenNoteTracker {
    /// `(program, pitch)` → onset time, in seconds from the start of the
    /// whole piece (not chunk-local).
    open: std::collections::HashMap<(u8, u8), f32>,
    current_program: Option<u8>,
    current_velocity: Option<bool>,
    in_prologue: bool,
}

impl OpenNoteTracker {
    fn new() -> Self {
        Self {
            open: Default::default(),
            current_program: None,
            current_velocity: None,
            in_prologue: true,
        }
    }

    /// Sorted `(program, pitch)` pairs currently held open — the exact
    /// input `mt3::tie_section_token_ids` needs to force the next chunk's
    /// prologue.
    fn open_keys(&self) -> Vec<(u8, u8)> {
        let mut keys: Vec<(u8, u8)> = self.open.keys().copied().collect();
        keys.sort_unstable();
        keys
    }

    /// Reset per-chunk state. `program`/`velocity` don't carry meaning
    /// across chunks (each chunk re-declares them via its own tokens); only
    /// `open` persists.
    fn start_chunk(&mut self) {
        self.current_program = None;
        self.current_velocity = None;
        self.in_prologue = true;
    }

    /// Feed one decoded token at chunk-local time `local_secs` (already
    /// offset by `chunk_start_secs` for the emitted `MidiNote`s).
    /// `chunk_end_bound`, when set, drops any event at or past that
    /// chunk-local time (the chunk's own window ends there; a later chunk
    /// owns that time instead). Call `finish` once after the last chunk to
    /// close anything still open.
    fn feed(
        &mut self,
        tok: Token,
        local_secs: f32,
        chunk_start_secs: f32,
        chunk_end_bound: Option<f32>,
        notes: &mut Vec<MidiNote>,
    ) {
        if self.in_prologue {
            match tok {
                Token::Tie => self.in_prologue = false,
                // The upstream's prologue handler *does* update `_program`
                // on a `program` token (events.py `feed`, `_in_prologue`
                // branch) — it's how the body picks up the right
                // instrument for the very first post-prologue `Pitch`
                // without waiting for a fresh `Program` token. Skipping
                // this (an earlier version of this port did) left
                // `current_program` stuck at `None` through the whole
                // prologue, silently dropping every pitch event up to the
                // next `Program` restatement — which, for a chunk that
                // never restates it, is the rest of the chunk.
                Token::Program(p) => self.current_program = Some(p),
                _ => {},
            }
            return;
        }
        if let Some(bound) = chunk_end_bound
            && local_secs >= bound
        {
            return;
        }
        let global_time = chunk_start_secs + local_secs;
        match tok {
            Token::Program(p) => self.current_program = Some(p),
            Token::Velocity(v) => self.current_velocity = Some(v),
            Token::Pitch(p) => {
                let (Some(prog), Some(vel)) = (self.current_program, self.current_velocity) else {
                    return;
                };
                let key = (prog, p);
                if let Some(onset) = self.open.remove(&key) {
                    notes.push(MidiNote {
                        program: prog,
                        pitch: p,
                        onset,
                        offset: global_time.max(onset),
                        instrument: crate::models::muscriptor::mt3::instrument_name_for_program(
                            prog,
                        ),
                    });
                }
                if vel {
                    self.open.insert(key, global_time);
                }
            },
            Token::Drum(d) => {
                notes.push(MidiNote {
                    program: DRUM_PROGRAM,
                    pitch: d,
                    onset: global_time,
                    offset: global_time + MIN_NOTE_DURATION_SECS,
                    instrument: Some("drums"),
                });
            },
            _ => {},
        }
    }

    /// End of stream: anything still open gets the minimum-duration
    /// fallback (matches the upstream's `OpenNoteTracker.finish`) rather
    /// than being dropped or extended to the last chunk's end.
    fn finish(&mut self, notes: &mut Vec<MidiNote>) {
        for ((prog, pitch), onset) in self.open.drain() {
            notes.push(MidiNote {
                program: prog,
                pitch,
                onset,
                offset: onset + MIN_NOTE_DURATION_SECS,
                instrument: crate::models::muscriptor::mt3::instrument_name_for_program(prog),
            });
        }
    }
}

#[cfg(test)]
mod open_note_tracker_tests {
    use super::*;

    /// Regression test for a real bug: an earlier version of `feed`
    /// ignored `Program` tokens during the prologue, so `current_program`
    /// stayed `None` through the whole prologue and every `Pitch` event in
    /// the body was silently dropped until (if ever) a fresh `Program`
    /// token happened to reappear — on a real transcription this made
    /// every chunk after the first produce empty output whenever the
    /// instrument didn't change mid-chunk.
    #[test]
    fn program_token_in_prologue_carries_into_the_body() {
        let mut tracker = OpenNoteTracker::new();
        let mut notes = Vec::new();
        tracker.start_chunk();

        // Prologue: "program 0, pitch 60 sustained, tie" — no notes emitted
        // during the prologue itself.
        tracker.feed(
            Token::Program(0),
            0.0,
            5.0,
            Some(SEGMENT_DURATION),
            &mut notes,
        );
        tracker.feed(
            Token::Pitch(60),
            0.0,
            5.0,
            Some(SEGMENT_DURATION),
            &mut notes,
        );
        tracker.feed(Token::Tie, 0.0, 5.0, Some(SEGMENT_DURATION), &mut notes);
        assert!(
            notes.is_empty(),
            "prologue tokens must never emit notes directly"
        );
        assert!(!tracker.in_prologue, "Tie must end the prologue");

        // Body: velocity-on, pitch 62 — with no Program token restated,
        // this must still resolve against program 0 from the prologue.
        tracker.feed(
            Token::Velocity(true),
            0.5,
            5.0,
            Some(SEGMENT_DURATION),
            &mut notes,
        );
        tracker.feed(
            Token::Pitch(62),
            0.5,
            5.0,
            Some(SEGMENT_DURATION),
            &mut notes,
        );

        assert_eq!(
            tracker.open_keys(),
            vec![(0, 62)],
            "pitch 62 must have opened under program 0"
        );
    }
}
