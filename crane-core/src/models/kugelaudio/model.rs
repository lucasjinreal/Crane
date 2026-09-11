//! `KugelAudioModel`: loads and wires every sub-network from a checkpoint
//! directory, plus the mechanical building blocks a generation loop needs
//! (prefill/decode forward, one diffusion-denoising step, tokenizer
//! encode/decode). All weight-name prefixes below were verified directly
//! against the real `kugelaudio/kugelaudio-0-open`
//! `model.safetensors.index.json` (1205 tensors), not just derived from the
//! Python source:
//!
//! ```text
//! lm_head.weight
//! model.language_model.{embed_tokens,layers,norm}.*
//! model.acoustic_tokenizer.{encoder,decoder}.*
//! model.semantic_tokenizer.encoder.*
//! model.{acoustic,semantic}_connector.{fc1,fc2,norm}.*
//! model.prediction_head.{cond_proj,final_layer,layers,noisy_images_proj,t_embedder}.*
//! model.speech_scaling_factor, model.speech_bias_factor   (scalar buffers)
//! ```
//!
//! **Not yet implemented here: the autoregressive `generate()` loop.**
//! `kugelaudio_inference.py`'s `generate()` (lines ~358-720 of
//! `models/kugelaudio_inference.py` in the `kugelaudio-open` package)
//! interleaves text-token decoding with diffusion-based speech-latent
//! generation, using a *second*, CFG-negative decoder stream whose KV cache
//! is kept in sync with only the diffusion-token positions via retroactive
//! per-token cache/mask/id splicing (lines 601-635) — a batched
//! optimization with no equivalent in this crate's shared
//! [`crate::models::modules::attention::GqaAttention`] (append-only cache,
//! no mid-sequence splice). Porting that exactly needs either extending the
//! shared cache or accepting a batch-size-1 simplification (the negative
//! stream only ever needs to contain `speech_start` plus previously
//! generated audio embeddings — see the doc comment above wherever that
//! lands) — deferred until the port has real weights to validate against
//! (see [`KugelAudioModel::from_pretrained`]'s doc comment for status).

use anyhow::{Context, Result};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;

use crate::utils::utils::get_safetensors_files;

use super::config::{KugelAudioConfig, load_config};
use super::connector::SpeechConnector;
use super::conv_layers::{TokenizerDecoder, TokenizerEncoder, load_decoder, load_encoder};
use super::decoder::KugelAudioDecoder;
use super::diffusion_head::DiffusionHead;
use super::dpm_solver::DpmSolverScheduler;
use super::prompt::PromptResult;

use crate::models::with_tracing::{Linear, linear_no_bias};

/// Speech-related special token ids. Hardcoded in the Python
/// (`KugelAudioProcessor.__call__` / `generate()`'s `getattr(self.config,
/// "speech_start_id", None) or 151652` pattern) rather than sourced from
/// `config.json` — these reuse Qwen2's vision special tokens
/// (`<|vision_start|>`, `<|vision_end|>`, `<|vision_pad|>`) since KugelAudio
/// ships no tokenizer of its own (`kugelaudio_processor.py` loads
/// `Qwen/Qwen2.5-1.5B`'s tokenizer by default and adds no new vocab).
pub mod special_tokens {
    pub const SPEECH_START_ID: u32 = 151_652;
    pub const SPEECH_END_ID: u32 = 151_653;
    pub const SPEECH_DIFFUSION_ID: u32 = 151_654;
    pub const EOS_TOKEN_ID: u32 = 151_643;
}

/// Every sub-network wired to one checkpoint's weights, plus the config
/// values callers need for prompt construction and generation bookkeeping
/// (`speech_compression_ratio`, `acoustic_vae_dim`, …).
pub struct KugelAudioModel {
    pub config: KugelAudioConfig,
    decoder: KugelAudioDecoder,
    lm_head: Linear,
    acoustic_encoder: TokenizerEncoder,
    acoustic_decoder: TokenizerDecoder,
    semantic_encoder: TokenizerEncoder,
    acoustic_connector: SpeechConnector,
    semantic_connector: SpeechConnector,
    diffusion_head: DiffusionHead,
    /// `model.speech_scaling_factor` / `model.speech_bias_factor`: scalar
    /// buffers fit once at training time (`1/std`, `-mean` of the training
    /// acoustic-latent distribution) and saved into the checkpoint. Applied
    /// as `(latent + bias) * scale` before the acoustic connector, and
    /// inverted (`latent / scale - bias`) after diffusion sampling — see
    /// `kugelaudio_inference.py`'s `_process_speech_inputs`/
    /// `generate()`. Always present (non-NaN) in a trained checkpoint; the
    /// Python's NaN-guarded "skip scaling" branch only matters mid-training
    /// and is not ported.
    speech_scaling_factor: f64,
    speech_bias_factor: f64,
    device: Device,
    dtype: DType,
}

impl KugelAudioModel {
    /// Load every sub-network from `model_dir` (a directory containing
    /// `config.json` and `model.safetensors.index.json` + shards, i.e. a
    /// local clone of `kugelaudio/kugelaudio-0-open`).
    ///
    /// Verified against the real checkpoint: this constructs successfully
    /// and produces the expected per-module tensor counts when pointed at a
    /// full local clone (see `crane-core/tests/kugelaudio_load.rs`).
    pub fn from_pretrained(model_dir: &str, device: &Device, dtype: DType) -> Result<Self> {
        let config_path = std::path::Path::new(model_dir).join("config.json");
        let config = load_config(config_path.to_str().context("non-UTF8 model path")?)
            .context("kugelaudio: load config.json")?;

        let filenames = get_safetensors_files(model_dir)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device)? };
        let vb_model = vb.pp("model");

        let decoder = KugelAudioDecoder::new(&config.decoder_config, vb_model.pp("language_model"))
            .context("kugelaudio: load language_model")?;
        let lm_head = linear_no_bias(
            config.decoder_config.hidden_size,
            config.decoder_config.vocab_size,
            vb.pp("lm_head"),
        )
        .context("kugelaudio: load lm_head")?;

        let vb_acoustic = vb_model.pp("acoustic_tokenizer");
        let acoustic_encoder = load_encoder(&config.acoustic_tokenizer_config, vb_acoustic.pp("encoder"))
            .context("kugelaudio: load acoustic_tokenizer.encoder")?;
        let acoustic_decoder = load_decoder(&config.acoustic_tokenizer_config, vb_acoustic.pp("decoder"))
            .context("kugelaudio: load acoustic_tokenizer.decoder")?;

        let semantic_encoder = load_encoder(
            &config.semantic_tokenizer_config,
            vb_model.pp("semantic_tokenizer").pp("encoder"),
        )
        .context("kugelaudio: load semantic_tokenizer.encoder")?;

        let acoustic_connector = SpeechConnector::load(
            config.acoustic_vae_dim,
            config.decoder_config.hidden_size,
            vb_model.pp("acoustic_connector"),
        )
        .context("kugelaudio: load acoustic_connector")?;
        let semantic_connector = SpeechConnector::load(
            config.semantic_vae_dim,
            config.decoder_config.hidden_size,
            vb_model.pp("semantic_connector"),
        )
        .context("kugelaudio: load semantic_connector")?;

        let diffusion_head = DiffusionHead::load(&config.diffusion_head_config, vb_model.pp("prediction_head"))
            .context("kugelaudio: load prediction_head")?;

        let speech_scaling_factor: f64 = vb_model
            .get((), "speech_scaling_factor")
            .context("kugelaudio: load speech_scaling_factor")?
            .to_dtype(DType::F64)?
            .to_scalar()?;
        let speech_bias_factor: f64 = vb_model
            .get((), "speech_bias_factor")
            .context("kugelaudio: load speech_bias_factor")?
            .to_dtype(DType::F64)?
            .to_scalar()?;

        Ok(Self {
            config,
            decoder,
            lm_head,
            acoustic_encoder,
            acoustic_decoder,
            semantic_encoder,
            acoustic_connector,
            semantic_connector,
            diffusion_head,
            speech_scaling_factor,
            speech_bias_factor,
            device: device.clone(),
            dtype,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Text-token embedding lookup (`get_input_embeddings()(ids)` in the
    /// Python) — `input_ids`: `[batch, seq_len]` → `[batch, seq_len, hidden_size]`.
    pub fn embed_text_tokens(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        self.decoder.embed_tokens(input_ids)
    }

    /// Run the decoder backbone on pre-built input embeddings and project
    /// to vocabulary logits at every position. `seqlen_offset` is the
    /// number of already-cached positions (0 for a fresh prefill).
    /// Returns `(hidden_states, logits)`, both `[batch, seq_len, *]`.
    pub fn forward(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let hidden = self.decoder.forward_embeds(inputs_embeds, seqlen_offset)?;
        let logits = self.lm_head.forward(&hidden)?;
        Ok((hidden, logits))
    }

    pub fn clear_kv_cache(&mut self) {
        self.decoder.clear_kv_cache();
    }

    /// Encode a raw waveform (`[batch, 1, num_samples]`) through the
    /// acoustic tokenizer, returning the **unscaled** latent mean
    /// (`encoder_output.mode()` in the Python — this port always takes the
    /// distribution's mean rather than the Gaussian-sampled path; see
    /// `config.rs`'s `TokenizerConfig::std_dist_type` doc comment).
    /// `[batch, vae_dim, T]`.
    pub fn encode_acoustic(&self, waveform: &Tensor) -> candle_core::Result<Tensor> {
        self.acoustic_encoder.encode(&waveform.to_dtype(self.dtype)?)
    }

    /// Encode a raw waveform through the semantic tokenizer.
    /// `[batch, semantic_vae_dim, T]`.
    pub fn encode_semantic(&self, waveform: &Tensor) -> candle_core::Result<Tensor> {
        self.semantic_encoder.encode(&waveform.to_dtype(self.dtype)?)
    }

    /// Decode acoustic latents (`[batch, vae_dim, T]`, **already** run
    /// through [`Self::unscale_acoustic_latent`] if they came from
    /// diffusion sampling) back to a waveform (`[batch, 1, num_samples]`).
    pub fn decode_acoustic(&self, latents: &Tensor) -> candle_core::Result<Tensor> {
        self.acoustic_decoder.decode(&latents.to_dtype(self.dtype)?)
    }

    /// `(latent + speech_bias_factor) * speech_scaling_factor` — applied to
    /// encoder output before the acoustic connector projects it into the
    /// decoder's embedding space (voice-prompt conditioning path).
    pub fn scale_acoustic_latent(&self, latent: &Tensor) -> candle_core::Result<Tensor> {
        latent.affine(self.speech_scaling_factor, self.speech_bias_factor * self.speech_scaling_factor)
    }

    /// `latent / speech_scaling_factor - speech_bias_factor` — inverts
    /// [`Self::scale_acoustic_latent`], applied to a diffusion-sampled
    /// latent before it's decoded to audio or re-encoded through the
    /// semantic tokenizer for the next autoregressive step.
    pub fn unscale_acoustic_latent(&self, latent: &Tensor) -> candle_core::Result<Tensor> {
        latent.affine(1.0 / self.speech_scaling_factor, -self.speech_bias_factor)
    }

    pub fn acoustic_connector(&self) -> &SpeechConnector {
        &self.acoustic_connector
    }

    pub fn semantic_connector(&self) -> &SpeechConnector {
        &self.semantic_connector
    }

    /// Build a fresh [`DpmSolverScheduler`] and run the full
    /// `ddpm_num_inference_steps`-step denoising chain, producing one
    /// speech-latent vector per row of `condition`.
    ///
    /// `condition`: `[N, hidden_size]` decoder hidden states (one per
    /// speech position being generated this step). No classifier-free
    /// guidance here — CFG (interpolating between a `condition` and
    /// `neg_condition` forward pass) is the caller's responsibility, same
    /// split as the Python's `sample_speech_tokens` vs. its caller in
    /// `generate()`; this only implements the single-condition diffusion
    /// loop those `cfg_scale == 1.0` and `!= 1.0` branches both bottom out
    /// in (`self.model.prediction_head(...)` / `noise_scheduler.step(...)`).
    /// Returns `[N, latent_size]` — **still in the scaled latent space**;
    /// callers must apply [`Self::unscale_acoustic_latent`] before decoding
    /// to audio.
    pub fn sample_speech_latents(&self, condition: &Tensor) -> candle_core::Result<Tensor> {
        let mut scheduler = DpmSolverScheduler::new(&self.config.diffusion_head_config)
            .map_err(|e| candle_core::Error::Msg(format!("kugelaudio: build DPM scheduler: {e}")))?;
        scheduler.set_timesteps(self.config.diffusion_head_config.ddpm_num_inference_steps);

        let n = condition.dim(0)?;
        let latent_size = self.config.diffusion_head_config.latent_size;
        let mut sample = Tensor::randn(0f32, 1f32, (n, latent_size), condition.device())?
            .to_dtype(condition.dtype())?;

        for &t in scheduler.timesteps().to_vec().iter() {
            let timesteps = Tensor::full(t as f32, n, condition.device())?;
            let eps = self.diffusion_head.forward(&sample, &timesteps, condition)?;
            sample = scheduler.step(&eps, &sample)?;
        }
        Ok(sample)
    }

    /// Classifier-free-guided variant of [`Self::sample_speech_latents`],
    /// for `N == 1` (one speech position; batch>1 is not implemented — see
    /// [`Self::generate`]'s doc comment). `condition`/`neg_condition`:
    /// `[1, hidden_size]`. Returns `[1, latent_size]`, still scaled.
    ///
    /// Deviates from `kugelaudio_inference.py`'s `sample_speech_tokens`'s
    /// CFG branch in one respect: the Python maintains *two* independently
    /// noised sample rows (`speech = torch.randn(2, vae_dim)`) through the
    /// whole denoising loop, feeding the diffusion head only row 0's noisy
    /// value (duplicated into both rows) at every step, then discarding row
    /// 1 entirely (`return speech[:1]` at the end) — row 1's own
    /// `scheduler.step` trajectory is computed but never read. This port
    /// only ever tracks the one trajectory that's actually returned,
    /// duplicating it on the fly for the conditional/unconditional forward
    /// pass — mathematically identical output, without the wasted half.
    pub fn sample_speech_latents_cfg(
        &self,
        condition: &Tensor,
        neg_condition: &Tensor,
        cfg_scale: f64,
    ) -> candle_core::Result<Tensor> {
        let mut scheduler = DpmSolverScheduler::new(&self.config.diffusion_head_config)
            .map_err(|e| candle_core::Error::Msg(format!("kugelaudio: build DPM scheduler: {e}")))?;
        scheduler.set_timesteps(self.config.diffusion_head_config.ddpm_num_inference_steps);

        let latent_size = self.config.diffusion_head_config.latent_size;
        let combined_condition = Tensor::cat(&[condition, neg_condition], 0)?; // [2, hidden]
        let mut sample = Tensor::randn(0f32, 1f32, (1, latent_size), condition.device())?
            .to_dtype(condition.dtype())?;

        for &t in scheduler.timesteps().to_vec().iter() {
            let combined_sample = Tensor::cat(&[&sample, &sample], 0)?; // [2, latent]
            let timesteps = Tensor::full(t as f32, 2, condition.device())?;
            let eps = self.diffusion_head.forward(&combined_sample, &timesteps, &combined_condition)?; // [2, latent]
            let cond_eps = eps.narrow(0, 0, 1)?;
            let uncond_eps = eps.narrow(0, 1, 1)?;
            let half_eps = (&uncond_eps + ((&cond_eps - &uncond_eps)? * cfg_scale)?)?;
            sample = scheduler.step(&half_eps, &sample)?;
        }
        Ok(sample)
    }

    /// Splice `replacement` (`[T, hidden_size]`) into `embeds`
    /// (`[1, N, hidden_size]`) at the single contiguous run of `true`
    /// positions in `mask` (length `N`) — `build_prompt`'s voice-prompt
    /// placeholder run. No-op (returns `embeds` unchanged) if `mask` is all
    /// `false`.
    fn splice_speech_embeds(embeds: &Tensor, mask: &[bool], replacement: &Tensor) -> candle_core::Result<Tensor> {
        let Some(start) = mask.iter().position(|&b| b) else {
            return Ok(embeds.clone());
        };
        let count = mask.iter().filter(|&&b| b).count();
        let n = mask.len();
        let before = embeds.narrow(1, 0, start)?;
        let after = embeds.narrow(1, start + count, n - start - count)?;
        let replacement = replacement.unsqueeze(0)?; // [1, T, hidden]
        Tensor::cat(&[&before, &replacement, &after], 1)
    }

    /// Pad/truncate `x` (`[1, T, dim]`) along the time axis to exactly
    /// `target_len` — mirrors `_process_speech_inputs`'s acoustic/semantic
    /// length-alignment belt-and-suspenders (in practice a no-op for this
    /// checkpoint: both tokenizers share `encoder_ratios`, so `T` already
    /// matches).
    fn align_time_len(x: &Tensor, target_len: usize) -> candle_core::Result<Tensor> {
        let t = x.dim(1)?;
        match t.cmp(&target_len) {
            std::cmp::Ordering::Equal => Ok(x.clone()),
            std::cmp::Ordering::Greater => x.narrow(1, 0, target_len),
            std::cmp::Ordering::Less => {
                let dim = x.dim(2)?;
                let pad = Tensor::zeros((1, target_len - t, dim), x.dtype(), x.device())?;
                Tensor::cat(&[x, &pad], 1)
            },
        }
    }

    /// Generate speech for `prompt`, batch size 1 only.
    ///
    /// Ports `kugelaudio_inference.py`'s `generate()` (lines ~358-720 of
    /// `models/kugelaudio_inference.py`), simplified for batch size 1 — see
    /// `model.rs`'s module doc comment for why the batched version's KV
    /// cache splicing has no equivalent here. The simplification is exact
    /// (not approximate) for the realistic single-utterance case: with one
    /// sample, the Python's per-step "is this sample mid-diffusion or not"
    /// bookkeeping across a batch collapses to a single always-true-or-false
    /// flag, and the retroactive cache-correction logic it drives is
    /// provably dead code (see the reasoning trail in this port's
    /// development notes) whenever `speech_start` occurs at most once (the
    /// non-multi-speaker case this implements) and `speech_end`/`eos` are
    /// terminal. Multi-speaker prompts that re-emit `speech_start` mid-
    /// generation (`generate()`'s `speech_start_mask` handling, lines
    /// 550-570 of the Python) are **not** ported — such a token is treated
    /// like any other non-diffusion token here (see the loop body).
    ///
    /// No streaming tokenizer cache either (see `conv_layers.rs`'s doc
    /// comment): every diffusion step re-decodes the *entire* accumulated
    /// latent sequence through the acoustic decoder and re-encodes the
    /// *entire* resulting waveform through the semantic encoder from
    /// scratch, keeping only the newest frame's output — correct (causal
    /// convs give identical results for a position whether computed via a
    /// streaming cache or a full non-streaming recompute) but `O(steps²)`
    /// instead of `O(steps)`. Fine for short clips; revisit for long-form
    /// generation.
    #[allow(clippy::too_many_lines)]
    pub fn generate(
        &mut self,
        prompt: &PromptResult,
        voice_waveform: Option<&Tensor>,
        cfg: &KugelAudioGenerationConfig,
    ) -> Result<KugelAudioGenerationOutput> {
        use special_tokens::{EOS_TOKEN_ID, SPEECH_DIFFUSION_ID, SPEECH_END_ID, SPEECH_START_ID};

        self.clear_kv_cache();
        let device = self.device.clone();

        let ids_u32: Vec<u32> = prompt.token_ids.clone();
        let ids_tensor = Tensor::from_vec(ids_u32.clone(), (1, ids_u32.len()), &device)?;
        let mut text_embeds = self.embed_text_tokens(&ids_tensor)?;

        if prompt.voice_frame_count > 0 {
            let waveform = voice_waveform
                .context("kugelaudio generate: prompt has voice-prompt frames but no voice_waveform was given")?;
            let acoustic = self.encode_acoustic(waveform)?.transpose(1, 2)?.contiguous()?; // [1, T, vae_dim]
            let acoustic_scaled = self.scale_acoustic_latent(&acoustic)?;
            let semantic = self.encode_semantic(waveform)?.transpose(1, 2)?.contiguous()?; // [1, T_sem, sem_dim]
            let semantic = Self::align_time_len(&semantic, acoustic_scaled.dim(1)?)?;
            let acoustic_embed = self.acoustic_connector.forward(&acoustic_scaled.squeeze(0)?)?;
            let semantic_embed = self.semantic_connector.forward(&semantic.squeeze(0)?)?;
            let combined = (acoustic_embed + semantic_embed)?; // [T, hidden]
            text_embeds = Self::splice_speech_embeds(&text_embeds, &prompt.speech_input_mask, &combined)?;
        }

        let embed_one = |model: &Self, id: u32| -> candle_core::Result<Tensor> {
            let t = Tensor::from_vec(vec![id], (1, 1), &device)?;
            model.embed_text_tokens(&t)
        };
        let speech_start_embed = embed_one(self, SPEECH_START_ID)?;
        let inputs_embeds = Tensor::cat(&[&text_embeds, &speech_start_embed], 1)?;
        let mut generated_ids: Vec<u32> = vec![SPEECH_START_ID];

        let (mut hidden, mut logits) = self.forward(&inputs_embeds, 0)?;
        let mut pos_seqlen = inputs_embeds.dim(1)?;

        let use_cfg = cfg.cfg_scale != 1.0;
        let mut neg_decoder = self.decoder.clone();
        let mut neg_last_embed = speech_start_embed;
        let mut neg_seqlen = 0usize;

        let mut logits_processor = LogitsProcessor::new(
            42,
            cfg.do_sample.then_some(cfg.temperature),
            None,
        );

        let candidate_ids = [SPEECH_START_ID, SPEECH_END_ID, SPEECH_DIFFUSION_ID, EOS_TOKEN_ID];
        let vocab_size = self.config.decoder_config.vocab_size;

        let mut all_latents: Vec<Tensor> = Vec::new();
        let mut audio_samples: Vec<f32> = Vec::new();
        let mut prev_audio_len = 0usize;

        for _step in 0..cfg.max_new_tokens {
            let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.flatten_all()?;
            let candidate_values: Vec<f32> = candidate_ids
                .iter()
                .map(|&id| last_logits.narrow(0, id as usize, 1)?.to_dtype(DType::F32)?.to_vec1::<f32>().map(|v| v[0]))
                .collect::<candle_core::Result<_>>()?;
            let mut masked = vec![f32::NEG_INFINITY; vocab_size];
            for (id, &v) in candidate_ids.iter().zip(candidate_values.iter()) {
                masked[*id as usize] = v;
            }
            let masked_logits = Tensor::from_vec(masked, vocab_size, &device)?;
            let next_token = logits_processor.sample(&masked_logits)?;

            generated_ids.push(next_token);
            if next_token == EOS_TOKEN_ID || next_token == SPEECH_END_ID {
                break;
            }

            let next_embed = if next_token == SPEECH_DIFFUSION_ID {
                let condition = hidden.narrow(1, hidden.dim(1)? - 1, 1)?.squeeze(1)?; // [1, hidden]

                let speech_latent_scaled = if use_cfg {
                    let neg_hidden = neg_decoder.forward_embeds(&neg_last_embed, neg_seqlen)?;
                    neg_seqlen += 1;
                    let neg_condition = neg_hidden.narrow(1, neg_hidden.dim(1)? - 1, 1)?.squeeze(1)?;
                    self.sample_speech_latents_cfg(&condition, &neg_condition, cfg.cfg_scale)?
                } else {
                    self.sample_speech_latents(&condition)?
                };
                let speech_latent = self.unscale_acoustic_latent(&speech_latent_scaled)?;
                all_latents.push(speech_latent);

                let latents_bct = Tensor::stack(&all_latents, 1)?.transpose(1, 2)?.contiguous()?; // [1, vae_dim, T]
                let full_audio = self.decode_acoustic(&latents_bct)?; // [1, 1, num_samples]
                let num_samples = full_audio.dim(2)?;
                let new_chunk = full_audio.narrow(D::Minus1, prev_audio_len, num_samples - prev_audio_len)?;
                let chunk: Vec<f32> = new_chunk.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
                audio_samples.extend(chunk);
                prev_audio_len = num_samples;

                let full_semantic = self.encode_semantic(&full_audio)?; // [1, sem_dim, T_sem]
                let t_sem = full_semantic.dim(2)?;
                let last_semantic = full_semantic
                    .narrow(D::Minus1, t_sem - 1, 1)?
                    .transpose(1, 2)?
                    .contiguous()?
                    .squeeze(1)?; // [1, sem_dim]

                let acoustic_embed = self.acoustic_connector.forward(&speech_latent_scaled)?;
                let semantic_embed = self.semantic_connector.forward(&last_semantic)?;
                (acoustic_embed + semantic_embed)?.unsqueeze(1)? // [1, 1, hidden]
            } else {
                embed_one(self, next_token)?
            };

            let (h, l) = self.forward(&next_embed, pos_seqlen)?;
            hidden = h;
            logits = l;
            pos_seqlen += 1;

            if use_cfg && next_token == SPEECH_DIFFUSION_ID {
                neg_last_embed = next_embed;
            }
        }

        Ok(KugelAudioGenerationOutput {
            token_ids: generated_ids,
            audio: audio_samples,
        })
    }
}

/// Sampling / stopping-condition knobs for [`KugelAudioModel::generate`].
/// Defaults match `kugelaudio_inference.py`'s `generate()` signature.
#[derive(Debug, Clone)]
pub struct KugelAudioGenerationConfig {
    /// Classifier-free guidance strength. `1.0` disables CFG (single
    /// forward pass per diffusion step, no negative decoder stream).
    pub cfg_scale: f64,
    pub max_new_tokens: usize,
    pub do_sample: bool,
    pub temperature: f64,
}

impl Default for KugelAudioGenerationConfig {
    fn default() -> Self {
        Self {
            cfg_scale: 3.0,
            max_new_tokens: 2048,
            do_sample: false,
            temperature: 1.0,
        }
    }
}

/// Result of [`KugelAudioModel::generate`].
pub struct KugelAudioGenerationOutput {
    /// The generated continuation only (control/diffusion-placeholder
    /// token ids), starting with `speech_start_id` — **not** including the
    /// prompt. Mirrors `KugelAudioGenerationOutput.sequences` in spirit,
    /// though the Python returns the full prompt+continuation.
    pub token_ids: Vec<u32>,
    /// Mono 24kHz waveform, concatenated across every generated frame.
    /// Empty if generation stopped before any diffusion token was produced.
    pub audio: Vec<f32>,
}

#[cfg(test)]
mod tests {
    // Real-checkpoint loading is exercised by
    // `crane-core/tests/kugelaudio_load.rs` (requires a local clone of
    // `kugelaudio/kugelaudio-0-open`, so it's an integration test, not a
    // unit test here — this module's own logic (scaling affine math) is
    // trivial enough that its correctness rides on the sub-module unit
    // tests plus that integration test's shape checks).
}
