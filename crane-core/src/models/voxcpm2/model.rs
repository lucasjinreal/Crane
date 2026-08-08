//! VoxCPM2 orchestration: loads all five sub-networks and implements the
//! generation loop for all four reference-audio-conditioning modes (see
//! [`VoxCpm2Conditioning`]) plus prompt-cache reuse (see
//! [`VoxCpm2PromptCache`]). **Streaming generation is not implemented** —
//! see the crate-level scope note in `mod.rs`. Port of `voxcpm2.py`'s
//! `VoxCPM2Model._generate`/`_generate_with_prompt_cache`/`_inference`.
//!
//! `enc_outputs = fsq_layer(enc_outputs) * audio_mask + enc_outputs *
//! text_mask` (in [`VoxCpm2Model::prefill`]): `lm_hidden` is FSQ-quantized
//! wherever it comes from an audio-masked position (true for `Continuation`/
//! `RefContinuation`'s last prefill position, and for every position from
//! the second generated step onward in every mode) but **not** where it
//! comes from a text-masked position (`ZeroShot`/`Reference`'s last prefill
//! position, and the very first generated step in every mode) — this
//! asymmetry is real, matches the reference exactly, and is *not* a minor
//! detail: [`super::fsq::round_half_to_even`]'s doc comment explains a real
//! bug this session found in exactly this FSQ codepath (Rust's `round()`
//! ties away from zero, `torch.round()` ties to even) that only manifested
//! for the audio-masked/FSQ'd case, since `ZeroShot`/`Reference` mode never
//! hit it before the first (harmless, single) FSQ application.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, linear_no_bias, Activation, Linear, VarBuilder};
use serde::Deserialize;

use super::audio_vae::{AudioVaeDecoder, AudioVaeEncoder};
use super::cfm::UnifiedCfm;
use super::config::load_config;
use super::fsq::ScalarQuantizationLayer;
use super::local_dit::VoxCpmLocDit;
use super::local_encoder::VoxCpmLocEnc;
use super::minicpm4::MiniCpm4Model;
use super::tokenizer::VoxCpm2Tokenizer;

/// Hardcoded in the Python (`VoxCPM2Model.__init__`'s
/// `self.audio_start_token = 101`), not config-derived. `ref_audio_start`/
/// `ref_audio_end` bracket a reference-audio-conditioning prefix segment
/// (`_make_ref_prefix`). `audio_end_token = 102` is set in `__init__` but
/// never actually referenced anywhere in `_generate`/`_inference`/
/// `_make_ref_prefix` in this version of the source — deliberately not
/// wired in here; it would be a phantom requirement.
const AUDIO_START_TOKEN: u32 = 101;
const REF_AUDIO_START_TOKEN: u32 = 103;
const REF_AUDIO_END_TOKEN: u32 = 104;

/// `_inference`'s `streaming_prefix_len` default. Its only live effect in
/// non-streaming generation (this pass's scope) is how many of a
/// continuation prompt's trailing real audio patches get included in the
/// AudioVAE decode call before being trimmed back off — see
/// [`VoxCpm2Model::generate_conditioned_inner`]'s doc comment for why that
/// matters. Hardcoded rather than exposed as a config knob since streaming
/// itself (where the parameter's name suggests its real purpose) is out of
/// scope for this pass.
const STREAMING_PREFIX_LEN: usize = 4;

#[derive(Debug, Clone)]
pub struct VoxCpm2GenerationConfig {
    /// Won't stop before this many patches even if the stop head fires.
    pub min_len: usize,
    /// Hard cap on generated patches.
    pub max_len: usize,
    /// Euler steps per patch in the flow-matching sampler.
    pub inference_timesteps: usize,
    /// Classifier-free guidance strength (`dit_config.cfm_config.inference_cfg_rate`
    /// in the checkpoint; exposed here since callers commonly override it).
    pub cfg_value: f64,
}

impl Default for VoxCpm2GenerationConfig {
    fn default() -> Self {
        Self { min_len: 2, max_len: 2000, inference_timesteps: 10, cfg_value: 2.0 }
    }
}

/// Shape/runtime fields read out of `config.json`'s generic
/// `audio_vae_config` block (kept as `serde_json::Value` in
/// [`VoxCpm2Config`] since the AudioVAE decoder only needs a handful of its
/// fields — see `config.rs`'s module docs).
#[derive(Debug, Clone, Deserialize)]
struct AudioVaeShapeConfig {
    encoder_dim: usize,
    encoder_rates: Vec<usize>,
    latent_dim: usize,
    decoder_dim: usize,
    decoder_rates: Vec<usize>,
    sr_bin_boundaries: Vec<i64>,
    sample_rate: i64,
    out_sample_rate: i64,
}

pub struct VoxCpm2Model {
    tokenizer: VoxCpm2Tokenizer,
    base_lm: MiniCpm4Model,
    residual_lm: MiniCpm4Model,
    feat_encoder: VoxCpmLocEnc,
    feat_decoder: UnifiedCfm,
    fsq_layer: ScalarQuantizationLayer,
    enc_to_lm_proj: Linear,
    lm_to_dit_proj: Linear,
    res_to_dit_proj: Linear,
    fusion_concat_proj: Linear,
    stop_proj: Linear,
    stop_head: Linear,
    audio_vae: AudioVaeDecoder,
    audio_vae_encoder: AudioVaeEncoder,
    patch_size: usize,
    feat_dim: usize,
    /// `prod(encoder_rates)` — the encoder's total downsample factor; a
    /// reference/prompt audio's raw sample count must be padded to a
    /// multiple of `patch_size * encoder_chunk_size` before encoding so the
    /// resulting patch count divides evenly (see `encode_reference_audio`).
    encoder_chunk_size: usize,
    /// `prod(decoder_rates)` — used only for the continuation-mode decode
    /// trim (`generate_conditioned_inner`), not for zero-shot/reference-only.
    decoder_chunk_size: usize,
    lm_use_mup: bool,
    lm_scale_emb: f64,
    device: Device,
    dtype: DType,
    pub sample_rate: u32,
    /// Sample rate `encode_reference_audio` expects its input already
    /// resampled to (the AudioVAE encoder's native rate — distinct from
    /// [`Self::sample_rate`], which is the *decoder's* output rate).
    pub encoder_sample_rate: u32,
}

impl VoxCpm2Model {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let cfg = load_config(&format!("{model_path}/config.json"))
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("load VoxCPM2 config.json")?;

        let tokenizer_path = format!("{model_path}/tokenizer.json");
        let tokenizer = VoxCpm2Tokenizer::from_file(&tokenizer_path)?;

        let weights_path = format!("{model_path}/model.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], *dtype, device) }
            .context("mmap model.safetensors")?;

        let table_len = cfg.max_length;
        let base_lm = MiniCpm4Model::new(&cfg.lm_config, table_len, vb.pp("base_lm"))?;
        let residual_cfg = cfg.lm_config.derive_residual_lm(cfg.residual_lm_num_layers, cfg.residual_lm_no_rope);
        let residual_lm = MiniCpm4Model::new(&residual_cfg, table_len, vb.pp("residual_lm"))?;

        let encoder_cfg = cfg.lm_config.derive(&cfg.encoder_config);
        let feat_encoder = VoxCpmLocEnc::new(&encoder_cfg, cfg.feat_dim, table_len, vb.pp("feat_encoder"))?;

        let dit_cfg = cfg.lm_config.derive(&cfg.dit_config.shape);
        let estimator = VoxCpmLocDit::new(&dit_cfg, cfg.feat_dim, table_len, vb.pp("feat_decoder").pp("estimator"))?;
        let feat_decoder = UnifiedCfm::new(estimator, cfg.dit_config.mean_mode);

        let hidden = cfg.lm_config.hidden_size;
        let fsq_layer = ScalarQuantizationLayer::new(
            hidden,
            hidden,
            cfg.scalar_quantization_latent_dim,
            cfg.scalar_quantization_scale,
            vb.pp("fsq_layer"),
        )?;
        let enc_to_lm_proj = linear(cfg.encoder_config.hidden_dim, hidden, vb.pp("enc_to_lm_proj"))?;
        let lm_to_dit_proj = linear(hidden, cfg.dit_config.shape.hidden_dim, vb.pp("lm_to_dit_proj"))?;
        let res_to_dit_proj = linear(hidden, cfg.dit_config.shape.hidden_dim, vb.pp("res_to_dit_proj"))?;
        let fusion_concat_proj = linear(hidden * 2, hidden, vb.pp("fusion_concat_proj"))?;
        let stop_proj = linear(hidden, hidden, vb.pp("stop_proj"))?;
        let stop_head = linear_no_bias(hidden, 2, vb.pp("stop_head"))?;

        let avc: AudioVaeShapeConfig =
            serde_json::from_value(cfg.audio_vae_config.clone()).context("parse audio_vae_config")?;
        let vae_weights_path = format!("{model_path}/audiovae.safetensors");
        let vae_vb = unsafe { VarBuilder::from_mmaped_safetensors(&[vae_weights_path], DType::F32, device) }
            .context("mmap audiovae.safetensors (run the .pth -> safetensors conversion first)")?;
        let audio_vae = AudioVaeDecoder::new(
            avc.latent_dim,
            avc.decoder_dim,
            &avc.decoder_rates,
            avc.sr_bin_boundaries,
            avc.out_sample_rate,
            vae_vb.pp("decoder"),
        )?;
        let audio_vae_encoder =
            AudioVaeEncoder::new(avc.encoder_dim, avc.latent_dim, &avc.encoder_rates, vae_vb.pp("encoder"))?;

        Ok(Self {
            tokenizer,
            base_lm,
            residual_lm,
            feat_encoder,
            feat_decoder,
            fsq_layer,
            enc_to_lm_proj,
            lm_to_dit_proj,
            res_to_dit_proj,
            fusion_concat_proj,
            stop_proj,
            stop_head,
            audio_vae,
            audio_vae_encoder,
            patch_size: cfg.patch_size,
            feat_dim: cfg.feat_dim,
            encoder_chunk_size: avc.encoder_rates.iter().product(),
            decoder_chunk_size: avc.decoder_rates.iter().product(),
            lm_use_mup: cfg.lm_config.use_mup,
            lm_scale_emb: cfg.lm_config.scale_emb,
            device: device.clone(),
            dtype: *dtype,
            sample_rate: avc.out_sample_rate as u32,
            encoder_sample_rate: avc.sample_rate as u32,
        })
    }

    pub fn clear_kv_cache(&mut self) {
        self.base_lm.clear_kv_cache();
        self.residual_lm.clear_kv_cache();
    }

    /// Zero-shot text-to-speech: `text` in, a `[1, 1, T]` f32 waveform in
    /// `[-1, 1]` at [`Self::sample_rate`] out. Thin wrapper over
    /// [`Self::generate_speech_conditioned`] with [`VoxCpm2Conditioning::ZeroShot`].
    pub fn generate_speech(&mut self, text: &str, cfg: &VoxCpm2GenerationConfig) -> Result<Tensor> {
        self.generate_speech_conditioned(text, &VoxCpm2Conditioning::ZeroShot, cfg)
    }

    /// Text-to-speech with optional reference-audio conditioning / voice
    /// cloning. See [`VoxCpm2Conditioning`] for the four supported modes.
    pub fn generate_speech_conditioned(
        &mut self,
        target_text: &str,
        conditioning: &VoxCpm2Conditioning,
        cfg: &VoxCpm2GenerationConfig,
    ) -> Result<Tensor> {
        Ok(self.generate_conditioned_retrying(target_text, conditioning, cfg)?.0)
    }

    /// CFM/flow-matching sampling can, for a given random noise draw,
    /// occasionally produce a near-silent, content-free patch sequence — a
    /// known failure mode of diffusion-style samplers (found live during
    /// this port's own validation: reproducible with candle's CUDA backend,
    /// whose `Tensor::randn` uses a fixed default seed unless the caller
    /// explicitly reseeds — see `candle_core::Device::set_seed` — so a given
    /// process/call-count combination can deterministically land on the same
    /// bad noise draw every run; not observed against the real Python
    /// reference, which reseeds from OS entropy by default). The reference's
    /// own `_generate` has an optional `retry_badcase` for a related but
    /// narrower symptom (output length far exceeding the input text's, e.g.
    /// babbling) — this instead directly checks for the amplitude-based
    /// symptom actually observed, and retries a bounded number of times with
    /// fresh noise (each `feat_decoder.forward` call draws new randomness
    /// regardless of the fixed base seed, since the underlying generator's
    /// state still advances draw to draw) before giving up and returning the
    /// last attempt as-is rather than erroring.
    fn generate_conditioned_retrying(
        &mut self,
        target_text: &str,
        conditioning: &VoxCpm2Conditioning,
        cfg: &VoxCpm2GenerationConfig,
    ) -> Result<(Tensor, Tensor)> {
        const MAX_ATTEMPTS: usize = 3;
        const SILENCE_MAX_ABS_THRESHOLD: f32 = 0.02;

        let mut last = None;
        for attempt in 0..MAX_ATTEMPTS {
            let (wav, feat) = self.generate_conditioned_inner(target_text, conditioning, cfg)?;
            let samples: Vec<f32> = wav.flatten_all()?.to_vec1()?;
            let max_abs = samples.iter().fold(0f32, |a, &b| a.max(b.abs()));
            if max_abs >= SILENCE_MAX_ABS_THRESHOLD {
                return Ok((wav, feat));
            }
            eprintln!(
                "[voxcpm2] generation attempt {attempt} produced near-silent audio (max_abs={max_abs}); retrying with fresh noise"
            );
            last = Some((wav, feat));
        }
        Ok(last.expect("MAX_ATTEMPTS > 0, loop runs at least once"))
    }

    /// Core generation, shared by [`Self::generate_speech_conditioned`] and
    /// [`Self::generate_with_prompt_cache`]. Returns `(waveform,
    /// newly_generated_patches)` — the second element is only needed by
    /// [`Self::merge_prompt_cache`] (extending a cache with this turn's
    /// output), everything else discards it.
    ///
    /// Port of `_inference`'s **general** form (`_generate`'s zero-shot
    /// branch was previously hardcoded here as a bit-exact special case —
    /// see the module docs' history; this is now the real masked-blend
    /// prefill every mode funnels through, with zero-shot as the degenerate
    /// `audio_mask`-all-zero case).
    ///
    /// **Continuation-mode decode trim, worth explaining since it's easy to
    /// get wrong**: for [`VoxCpm2Conditioning::Continuation`]/
    /// [`VoxCpm2Conditioning::RefContinuation`] (where the conditioning
    /// tensor's last position is real prompt audio, not text), the
    /// AudioVAE's causal-conv decoder produces a warm-up transient at the
    /// very start of whatever it decodes. Decoding *only* the newly
    /// generated patches in isolation would put that transient right at the
    /// start of the returned audio (an audible glitch/discontinuity at the
    /// prompt/generated seam). The reference avoids this by prepending the
    /// prompt's own last `min(STREAMING_PREFIX_LEN - 1, prompt_len)`
    /// patches to the decode call, then trimming the corresponding samples
    /// back off afterward — so the transient lands inside the discarded
    /// region instead of the returned one. `context_len` below is that
    /// patch count (`0` for zero-shot/reference-only, where there's no
    /// prompt-audio seam to protect).
    /// Port of `_inference`'s prefill (`voxcpm2.py:1009-1058`): the masked
    /// blend of text/audio embeddings through `base_lm`, mask-conditional
    /// FSQ, then `residual_lm`. Returns `(lm_hidden, residual_hidden)` — the
    /// last-position slices `generate_conditioned_inner`'s decode loop
    /// actually needs. Split out from `generate_conditioned_inner` so it can
    /// be exercised directly (bypassing tokenization/encoding) by the
    /// `hf_diff` cross-check below, with raw dumped `ids`/`audio_feat`/
    /// masks fed straight in — isolating this method's own math from
    /// whether the tokenizer/encoder happen to agree with Python.
    fn prefill(
        &mut self,
        ids: &[u32],
        audio_feat: &Tensor,
        text_mask: &[u32],
        audio_mask: &[u32],
    ) -> Result<(Tensor, Tensor)> {
        let total_len = ids.len();
        let ids_tensor = Tensor::new(ids, &self.device)?.unsqueeze(0)?; // [1, T]
        let embed_tokens = self.base_lm.embed_tokens.as_ref().context("base_lm has no embed_tokens")?;
        let text_embed = embed_tokens.forward(&ids_tensor)?.to_dtype(self.dtype)?; // [1, T, H]
        let text_embed = if self.lm_use_mup { (text_embed * self.lm_scale_emb)? } else { text_embed };

        let text_mask_t = mask_to_tensor(text_mask, &self.device, self.dtype)?;
        let audio_mask_t = mask_to_tensor(audio_mask, &self.device, self.dtype)?;

        let feat_embed = self.enc_to_lm_proj.forward(&self.feat_encoder.forward(audio_feat)?)?; // [1, T, H]
        let combined_embed =
            (text_embed.broadcast_mul(&text_mask_t)? + feat_embed.broadcast_mul(&audio_mask_t)?)?;

        let enc_outputs_raw = self.base_lm.forward(&combined_embed, true)?; // [1, T, H]
        let fsq_out = self.fsq_layer.forward(&enc_outputs_raw)?;
        let enc_outputs =
            (fsq_out.broadcast_mul(&audio_mask_t)? + enc_outputs_raw.broadcast_mul(&text_mask_t)?)?;
        let lm_hidden = enc_outputs.narrow(1, total_len - 1, 1)?.squeeze(1)?; // [1, H]

        let masked_feat_embed = feat_embed.broadcast_mul(&audio_mask_t)?;
        let residual_in = self.fusion_concat_proj.forward(&Tensor::cat(&[&enc_outputs, &masked_feat_embed], 2)?)?;
        let residual_outputs = self.residual_lm.forward(&residual_in, true)?;
        let residual_hidden = residual_outputs.narrow(1, total_len - 1, 1)?.squeeze(1)?; // [1, H]

        Ok((lm_hidden, residual_hidden))
    }

    fn generate_conditioned_inner(
        &mut self,
        target_text: &str,
        conditioning: &VoxCpm2Conditioning,
        cfg: &VoxCpm2GenerationConfig,
    ) -> Result<(Tensor, Tensor)> {
        self.clear_kv_cache();

        let (ids, audio_feat, text_mask, audio_mask, context_len) =
            self.build_conditioning_tensors(target_text, conditioning)?;
        let total_len = ids.len();
        anyhow::ensure!(total_len > 0, "empty conditioning sequence");

        let (mut lm_hidden, mut residual_hidden) = self.prefill(&ids, &audio_feat, &text_mask, &audio_mask)?;

        // `feat[:, -1, ...]`: the real last position of the full input —
        // an all-zero patch for zero-shot/reference-only (text-terminated
        // prefixes), a real prompt-audio patch for continuation modes. No
        // special-casing needed: this is correct as-is for every mode.
        let mut prefix_feat_cond = audio_feat.narrow(1, total_len - 1, 1)?.squeeze(1)?; // [1, P, D]

        // Seed `generated` with the prompt's own trailing `context_len`
        // patches (continuation modes only) — see this method's doc comment.
        let mut generated: Vec<Tensor> = Vec::new();
        if context_len > 0 {
            let audio_positions: Vec<usize> =
                audio_mask.iter().enumerate().filter(|&(_, &m)| m == 1).map(|(i, _)| i).collect();
            let seed_positions = &audio_positions[audio_positions.len() - context_len..];
            for &pos in seed_positions {
                generated.push(audio_feat.narrow(1, pos, 1)?.squeeze(1)?);
            }
        }

        for step in 0..cfg.max_len {
            let dit_h1 = self.lm_to_dit_proj.forward(&lm_hidden)?;
            let dit_h2 = self.res_to_dit_proj.forward(&residual_hidden)?;
            let dit_hidden = Tensor::cat(&[&dit_h1, &dit_h2], 1)?; // [1, 2*dit_hidden]

            let cond = prefix_feat_cond.transpose(1, 2)?.contiguous()?; // [1, D, P]
            let pred_feat = self.feat_decoder.forward(
                &dit_hidden,
                cfg.inference_timesteps,
                self.patch_size,
                &cond,
                cfg.cfg_value,
                1.0,
                1.0,
                true,
            )?; // [1, D, P]
            let pred_feat = pred_feat.transpose(1, 2)?.contiguous()?; // [1, P, D]

            let curr_embed = self.feat_encoder.forward(&pred_feat.unsqueeze(1)?)?; // [1, 1, H_enc]
            let curr_embed = self.enc_to_lm_proj.forward(&curr_embed)?; // [1, 1, H]

            generated.push(pred_feat.clone());
            prefix_feat_cond = pred_feat;

            let stop_hidden = Activation::Silu.forward(&self.stop_proj.forward(&lm_hidden)?)?;
            let stop_logits = self.stop_head.forward(&stop_hidden)?; // [1, 2]
            let stop_flag = stop_logits.argmax(1)?.reshape(())?.to_scalar::<u32>()?;
            if step > cfg.min_len && stop_flag == 1 {
                break;
            }

            let step_embed = curr_embed.squeeze(1)?; // [1, H]
            let position = total_len + step;
            let next_lm_hidden = self.base_lm.forward_step(&step_embed, position)?; // [1, H]
            lm_hidden = self.fsq_layer.forward(&next_lm_hidden)?; // FSQ applied for every generated step.
            let residual_input = self.fusion_concat_proj.forward(&Tensor::cat(&[&lm_hidden, &step_embed], 1)?)?;
            residual_hidden = self.residual_lm.forward_step(&residual_input, position)?;
        }

        anyhow::ensure!(generated.len() > context_len, "generated zero new audio patches");

        let stack = |patches: &[Tensor]| -> Result<Tensor> {
            // "b t p d -> b d (t p)": stack the per-step patches into a time
            // axis, then permute+flatten so channels lead and (step,
            // within-patch) collapse into one axis, step outer / within-patch inner.
            let terms: Vec<Tensor> = patches.iter().map(|p| p.unsqueeze(1)).collect::<candle_core::Result<_>>()?;
            let stacked = Tensor::cat(&terms, 1)?; // [1, n, P, D]
            let (_b, n, p, d) = stacked.dims4()?;
            Ok(stacked.permute((0, 3, 1, 2))?.contiguous()?.reshape((1, d, n * p))?)
        };

        let latent = stack(&generated)?;
        let wav = self.audio_vae.decode(&latent.to_dtype(DType::F32)?).map_err(|e| anyhow::anyhow!("{e}"))?;
        let wav = if context_len > 0 {
            let trim = self.patch_size * self.decoder_chunk_size * context_len;
            let total = wav.dim(2)?;
            wav.narrow(2, trim, total - trim)?
        } else {
            wav
        };

        // Patch format `[1, n_new, P, D]` — matches `encode_reference_audio`'s
        // output convention (NOT the flattened-latent `[1, D, n*P]` format
        // `stack()`/`audio_vae.decode()` use), so `merge_prompt_cache` can
        // `Tensor::cat` this directly with an existing `prompt_feat`/round-trip
        // it back through `Continuation`/`RefContinuation` conditioning later.
        let new_patches: Vec<Tensor> =
            generated[context_len..].iter().map(|p| p.unsqueeze(1)).collect::<candle_core::Result<_>>()?;
        let generated_feat = Tensor::cat(&new_patches, 1)?;
        Ok((wav, generated_feat))
    }

    /// Port of `_make_ref_prefix`: brackets a reference-audio segment with
    /// `ref_audio_start`/`ref_audio_end` tokens (`text_mask=1`), the audio
    /// patches in between get `audio_mask=1`. Returns
    /// `(tokens, feats, text_mask, audio_mask)`.
    fn make_ref_prefix(&self, ref_feat: &Tensor) -> Result<(Vec<u32>, Tensor, Vec<u32>, Vec<u32>)> {
        let ref_len = ref_feat.dim(1)?;
        let zero_patch = Tensor::zeros((1, 1, self.patch_size, self.feat_dim), self.dtype, &self.device)?;

        let mut tokens = vec![REF_AUDIO_START_TOKEN];
        tokens.extend(std::iter::repeat_n(0u32, ref_len));
        tokens.push(REF_AUDIO_END_TOKEN);

        let feats = Tensor::cat(&[&zero_patch, ref_feat, &zero_patch], 1)?; // [1, ref_len+2, P, D]

        let mut text_mask = vec![1u32];
        text_mask.extend(std::iter::repeat_n(0u32, ref_len));
        text_mask.push(1);

        let mut audio_mask = vec![0u32];
        audio_mask.extend(std::iter::repeat_n(1u32, ref_len));
        audio_mask.push(0);

        Ok((tokens, feats, text_mask, audio_mask))
    }

    /// Builds `(text_token_ids, audio_feat[1,T,P,D], text_mask, audio_mask,
    /// context_len)` for one of the four conditioning modes — port of
    /// `_generate`'s four branches (`_generate_with_prompt_cache`'s branches
    /// are the same shape, driven from cached tensors instead of re-encoded
    /// wav files).
    fn build_conditioning_tensors(
        &self,
        target_text: &str,
        conditioning: &VoxCpm2Conditioning,
    ) -> Result<(Vec<u32>, Tensor, Vec<u32>, Vec<u32>, usize)> {
        let text_suffix_ids = |text: &str| -> Result<Vec<u32>> {
            let mut ids = self.tokenizer.encode(text)?;
            ids.push(AUDIO_START_TOKEN);
            Ok(ids)
        };
        let text_pad_feat = |len: usize| -> Result<Tensor> {
            Ok(Tensor::zeros((1, len, self.patch_size, self.feat_dim), self.dtype, &self.device)?)
        };

        match conditioning {
            VoxCpm2Conditioning::ZeroShot => {
                let ids = text_suffix_ids(target_text)?;
                let text_len = ids.len();
                let audio_feat = text_pad_feat(text_len)?;
                Ok((ids, audio_feat, vec![1u32; text_len], vec![0u32; text_len], 0))
            }
            VoxCpm2Conditioning::Reference(ref_feat) => {
                let (mut tokens, ref_feats, mut text_mask, mut audio_mask) = self.make_ref_prefix(ref_feat)?;
                let ids = text_suffix_ids(target_text)?;
                let text_len = ids.len();
                tokens.extend(ids);
                let feats = Tensor::cat(&[&ref_feats, &text_pad_feat(text_len)?], 1)?;
                text_mask.extend(std::iter::repeat_n(1u32, text_len));
                audio_mask.extend(std::iter::repeat_n(0u32, text_len));
                Ok((tokens, feats, text_mask, audio_mask, 0))
            }
            VoxCpm2Conditioning::Continuation { prompt_text, prompt_feat } => {
                let full_text = format!("{prompt_text}{target_text}");
                let mut ids = text_suffix_ids(&full_text)?;
                let text_len = ids.len();
                let prompt_len = prompt_feat.dim(1)?;
                ids.extend(std::iter::repeat_n(0u32, prompt_len));
                let feats = Tensor::cat(&[&text_pad_feat(text_len)?, prompt_feat], 1)?;
                let mut text_mask = vec![1u32; text_len];
                text_mask.extend(std::iter::repeat_n(0u32, prompt_len));
                let mut audio_mask = vec![0u32; text_len];
                audio_mask.extend(std::iter::repeat_n(1u32, prompt_len));
                let context_len = (STREAMING_PREFIX_LEN - 1).min(prompt_len);
                Ok((ids, feats, text_mask, audio_mask, context_len))
            }
            VoxCpm2Conditioning::RefContinuation { ref_feat, prompt_text, prompt_feat } => {
                let (mut tokens, ref_feats, mut text_mask, mut audio_mask) = self.make_ref_prefix(ref_feat)?;
                let full_text = format!("{prompt_text}{target_text}");
                let mut ids = text_suffix_ids(&full_text)?;
                let text_len = ids.len();
                let prompt_len = prompt_feat.dim(1)?;
                ids.extend(std::iter::repeat_n(0u32, prompt_len));
                tokens.extend(ids);
                let feats = Tensor::cat(&[&ref_feats, &text_pad_feat(text_len)?, prompt_feat], 1)?;
                text_mask.extend(std::iter::repeat_n(1u32, text_len));
                text_mask.extend(std::iter::repeat_n(0u32, prompt_len));
                audio_mask.extend(std::iter::repeat_n(0u32, text_len));
                audio_mask.extend(std::iter::repeat_n(1u32, prompt_len));
                let context_len = (STREAMING_PREFIX_LEN - 1).min(prompt_len);
                Ok((tokens, feats, text_mask, audio_mask, context_len))
            }
        }
    }

    /// Sample rate `samples` must already be resampled to before calling —
    /// see [`Self::encoder_sample_rate`]. `pad_left`: `true` for continuation
    /// prompts (so the tail — the part that matters for a smooth
    /// continuation — stays aligned), `false` for reference clips (matches
    /// `_encode_wav`'s `padding_mode="left"`/`"right"` split between the
    /// two call sites in the reference).
    ///
    /// Returns `[1, T, patch_size, latent_dim]` patches — port of
    /// `_encode_wav`: pad raw samples to a `patch_size * encoder_chunk_size`
    /// multiple, VAE-encode, reshape `[D, T']` into `[T'/patch_size,
    /// patch_size, D]` patches (`.view(D,-1,P).permute(1,2,0)` in the
    /// reference — note the axis order here, batch dim included).
    pub fn encode_reference_audio(&self, samples: &[f32], pad_left: bool) -> Result<Tensor> {
        let patch_len = self.patch_size * self.encoder_chunk_size;
        let raw_len = samples.len();
        let padded_len = raw_len.div_ceil(patch_len).max(1) * patch_len;
        let pad = padded_len - raw_len;

        let mut padded = vec![0f32; padded_len];
        if pad_left {
            padded[pad..].copy_from_slice(samples);
        } else {
            padded[..raw_len].copy_from_slice(samples);
        }

        let audio = Tensor::from_vec(padded, (1, 1, padded_len), &self.device)?;
        let mu = self.audio_vae_encoder.encode(&audio)?; // [1, latent_dim, T'], F32 (the VAE always runs in F32)
        let t_prime = mu.dim(2)?;
        anyhow::ensure!(t_prime % self.patch_size == 0, "encoder output length not a multiple of patch_size");
        let n_patches = t_prime / self.patch_size;
        // [1, D, T'] -> [1, D, n_patches, P] -> [1, n_patches, P, D], cast to
        // the model's runtime dtype so it can be concatenated/blended with
        // text embeddings downstream (`build_conditioning_tensors`).
        mu.reshape((1, self.feat_dim, n_patches, self.patch_size))?
            .permute((0, 2, 3, 1))?
            .contiguous()?
            .to_dtype(self.dtype)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Sample rate [`Self::encode_reference_audio`] expects its input at.
    pub fn encoder_sample_rate(&self) -> u32 {
        self.encoder_sample_rate
    }

    /// Generate using a pre-built [`VoxCpm2PromptCache`] (`None` = zero-shot).
    /// Returns `(waveform, newly_generated_patches)` — pass the second
    /// value to [`Self::merge_prompt_cache`] to extend the cache with this
    /// turn's output for the next call.
    pub fn generate_with_prompt_cache(
        &mut self,
        target_text: &str,
        cache: Option<&VoxCpm2PromptCache>,
        cfg: &VoxCpm2GenerationConfig,
    ) -> Result<(Tensor, Tensor)> {
        let conditioning = cache.map_or(VoxCpm2Conditioning::ZeroShot, VoxCpm2PromptCache::to_conditioning);
        self.generate_conditioned_retrying(target_text, &conditioning, cfg)
    }

    /// Port of `merge_prompt_cache`: fold newly generated text+audio into a
    /// (possibly `None`, i.e. first-turn) existing cache, stabilizing voice
    /// across a multi-turn session by growing the continuation prompt.
    pub fn merge_prompt_cache(
        &self,
        original: Option<&VoxCpm2PromptCache>,
        new_text: &str,
        new_audio_feat: &Tensor,
    ) -> Result<VoxCpm2PromptCache> {
        let Some(original) = original else {
            return Ok(VoxCpm2PromptCache {
                ref_audio_feat: None,
                prompt_text: new_text.to_string(),
                audio_feat: Some(new_audio_feat.clone()),
            });
        };
        let prompt_text = format!("{}{}", original.prompt_text, new_text);
        let audio_feat = match &original.audio_feat {
            Some(old) => Tensor::cat(&[old, new_audio_feat], 1)?,
            None => new_audio_feat.clone(),
        };
        Ok(VoxCpm2PromptCache { ref_audio_feat: original.ref_audio_feat.clone(), prompt_text, audio_feat: Some(audio_feat) })
    }
}

fn mask_to_tensor(mask: &[u32], device: &Device, dtype: DType) -> Result<Tensor> {
    let floats: Vec<f32> = mask.iter().map(|&m| m as f32).collect();
    Ok(Tensor::from_vec(floats, (1, mask.len(), 1), device)?.to_dtype(dtype)?)
}

/// The four reference-audio-conditioning modes `_generate`/
/// `_generate_with_prompt_cache` support, unified behind one entry point
/// ([`VoxCpm2Model::generate_speech_conditioned`]/
/// [`VoxCpm2Model::generate_with_prompt_cache`]). Audio-feat tensors are
/// `[1, T, patch_size, latent_dim]`, produced by
/// [`VoxCpm2Model::encode_reference_audio`].
#[derive(Debug, Clone)]
pub enum VoxCpm2Conditioning {
    /// No reference audio — the original, already-shipped behavior.
    ZeroShot,
    /// "Controllable Cloning": clone a voice from a short reference clip,
    /// no transcript needed. Structurally isolated via
    /// `ref_audio_start`/`ref_audio_end` tokens, so the reference audio's
    /// own content never gets continued/repeated in the output.
    Reference(Tensor),
    /// "Ultimate Cloning": continue a reference clip's audio, given its
    /// transcript (`prompt_text`) — every vocal nuance faithfully
    /// reproduced, per the checkpoint's own docs. This is the mode
    /// `crane::audio::Tts::generate_voice_clone` maps onto (its `ref_audio`+
    /// `ref_text` shape matches exactly).
    Continuation { prompt_text: String, prompt_feat: Tensor },
    /// Both: an isolated reference-audio prefix for timbre, *and* a
    /// continuation suffix for content/prosody.
    RefContinuation { ref_feat: Tensor, prompt_text: String, prompt_feat: Tensor },
}

/// Precomputed reference/prompt audio + transcript, reusable across many
/// [`VoxCpm2Model::generate_with_prompt_cache`] calls instead of
/// re-encoding a wav file every time. Port of the dict `build_prompt_cache`/
/// `merge_prompt_cache`/`_generate_with_prompt_cache` pass around in Python.
#[derive(Debug, Clone)]
pub struct VoxCpm2PromptCache {
    pub ref_audio_feat: Option<Tensor>,
    pub prompt_text: String,
    pub audio_feat: Option<Tensor>,
}

impl VoxCpm2PromptCache {
    /// Port of `build_prompt_cache`. At least one of `ref_audio_feat`/
    /// `prompt` must be given. `prompt` is `(prompt_text, prompt_feat)` —
    /// both required together, matching the Python validation
    /// (`prompt_wav_path`/`prompt_text` must both be provided or both `None`).
    pub fn build(ref_audio_feat: Option<Tensor>, prompt: Option<(String, Tensor)>) -> Result<Self> {
        anyhow::ensure!(
            ref_audio_feat.is_some() || prompt.is_some(),
            "at least one of ref_audio_feat or prompt must be provided"
        );
        let (prompt_text, audio_feat) = match prompt {
            Some((text, feat)) => (text, Some(feat)),
            None => (String::new(), None),
        };
        Ok(Self { ref_audio_feat, prompt_text, audio_feat })
    }

    fn to_conditioning(&self) -> VoxCpm2Conditioning {
        match (&self.ref_audio_feat, &self.audio_feat) {
            (Some(r), Some(p)) => VoxCpm2Conditioning::RefContinuation {
                ref_feat: r.clone(),
                prompt_text: self.prompt_text.clone(),
                prompt_feat: p.clone(),
            },
            (Some(r), None) => VoxCpm2Conditioning::Reference(r.clone()),
            (None, Some(p)) => {
                VoxCpm2Conditioning::Continuation { prompt_text: self.prompt_text.clone(), prompt_feat: p.clone() }
            }
            (None, None) => VoxCpm2Conditioning::ZeroShot,
        }
    }
}

#[cfg(test)]
mod hf_diff {
    use super::*;

    // Cross-checks the generalized `prefill` (the masked-blend prefill
    // every conditioning mode funnels through — see `generate_conditioned_inner`'s
    // doc comment) against a real `_inference` prefill dump for the
    // Continuation-only mode (see voxcpm_prefill_diff.py in this session's
    // job tmp dir — reuses the real model's own bound submodules to build
    // the reference, not a from-scratch reimplementation). Raw dumped
    // `text_token`/`audio_feat`/masks are fed straight into `prefill`,
    // bypassing this crate's own tokenizer/encoder, so this isolates just
    // the prefill math from tokenization/encoding parity.
    //
    // `cargo test --release -p crane-core voxcpm2_prefill_matches_python -- --ignored --nocapture`
    #[test]
    #[ignore = "needs the real VoxCPM2 checkpoint + a matching HF dump"]
    fn voxcpm2_prefill_matches_python() {
        let model_path = "/home/hahihula/mywork/ai/additional_models/VoxCPM2";
        let diff_dir = "/home/hahihula/.claude/jobs/02aa7312/tmp/voxcpm_prefill_diff";
        let device = Device::Cpu;
        let dtype = DType::F32;

        let mut model = VoxCpm2Model::new(model_path, &device, &dtype).expect("load VoxCPM2");

        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{diff_dir}/meta.json")).unwrap()).unwrap();
        let total_len = meta["total_len"].as_u64().unwrap() as usize;
        let hidden_size = meta["hidden_size"].as_u64().unwrap() as usize;
        let patch_size = meta["patch_size"].as_u64().unwrap() as usize;
        let latent_dim = meta["latent_dim"].as_u64().unwrap() as usize;

        let load_f32 = |name: &str, n: usize| -> Vec<f32> {
            let raw = std::fs::read(format!("{diff_dir}/{name}.bin")).unwrap();
            raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).take(n).collect()
        };
        let load_i64_as_u32 = |name: &str, n: usize| -> Vec<u32> {
            let raw = std::fs::read(format!("{diff_dir}/{name}.bin")).unwrap();
            raw.chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as u32)
                .take(n)
                .collect()
        };

        let ids = load_i64_as_u32("text_token", total_len);
        let text_mask = load_i64_as_u32("text_mask", total_len);
        let audio_mask = load_i64_as_u32("audio_mask", total_len);
        let audio_feat_flat = load_f32("audio_feat", total_len * patch_size * latent_dim);
        let audio_feat = Tensor::from_vec(audio_feat_flat, (1, total_len, patch_size, latent_dim), &device).unwrap();

        model.clear_kv_cache();
        let (lm_hidden, residual_hidden) = model.prefill(&ids, &audio_feat, &text_mask, &audio_mask).unwrap();

        let hf_lm_hidden = load_f32("lm_hidden", hidden_size);
        let hf_residual_hidden = load_f32("residual_hidden", hidden_size);

        let cosine = |a: &[f32], b: &[f32]| -> f64 {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
            let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum::<f64>().sqrt();
            dot / (na * nb)
        };

        let rust_lm_hidden: Vec<f32> = lm_hidden.flatten_all().unwrap().to_vec1().unwrap();
        let rust_residual_hidden: Vec<f32> = residual_hidden.flatten_all().unwrap().to_vec1().unwrap();

        let cos_lm = cosine(&rust_lm_hidden, &hf_lm_hidden);
        let cos_residual = cosine(&rust_residual_hidden, &hf_residual_hidden);
        println!("lm_hidden cosine: {cos_lm}");
        println!("residual_hidden cosine: {cos_residual}");

        assert!(cos_lm > 0.999, "lm_hidden diverged from Python: cosine={cos_lm}");
        assert!(cos_residual > 0.999, "residual_hidden diverged from Python: cosine={cos_residual}");
    }


    /// Pure Rust-internal consistency check (no Python involved): does
    /// `base_lm`'s incremental `forward_step` decode, run for *several*
    /// consecutive steps after a large (~60-position) prefill, match a
    /// single bulk causal `forward` over the same full sequence? This is
    /// the exact KV-cache/rotary-position invariant the generation loop
    /// depends on, at the same scale (`total_len` ~60-70) as the real
    /// multi-turn prompt-cache bug — `attention.rs`'s own
    /// `test_incremental_decode_matches_full_prefill` covers this same
    /// property but with `rope=None`/small `seq_len=3`, not exercising
    /// `MiniCpm4Model`'s real `LongRoPE` at this scale or multiple
    /// consecutive `forward_step` calls the way real generation does.
    #[test]
    #[ignore = "needs the real VoxCPM2 checkpoint"]
    fn voxcpm2_base_lm_incremental_matches_full_prefill_at_scale() {
        let model_path = "/home/hahihula/mywork/ai/additional_models/VoxCPM2";
        let device = Device::new_cuda(0).unwrap();
        let dtype = DType::BF16;
        let mut model = VoxCpm2Model::new(model_path, &device, &dtype).expect("load VoxCPM2");

        let hidden = 2048usize;
        let total_positions = 68usize;
        let n_incremental = 5usize;
        let prefix_len = total_positions - n_incremental;

        let full_input = Tensor::randn(0f32, 0.5f32, (1, total_positions, hidden), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        model.clear_kv_cache();
        let full_out = model.base_lm.forward(&full_input, true).unwrap();
        let full_tail = full_out.narrow(1, prefix_len, n_incremental).unwrap(); // [1, n_incremental, H]

        model.clear_kv_cache();
        let prefix_input = full_input.narrow(1, 0, prefix_len).unwrap();
        let _ = model.base_lm.forward(&prefix_input, true).unwrap();

        let mut incr_outputs = Vec::with_capacity(n_incremental);
        for step in 0..n_incremental {
            let pos_input = full_input.narrow(1, prefix_len + step, 1).unwrap().squeeze(1).unwrap(); // [1, H]
            let out = model.base_lm.forward_step(&pos_input, prefix_len + step).unwrap(); // [1, H]
            incr_outputs.push(out);
        }

        let cosine = |a: &[f32], b: &[f32]| -> f64 {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
            let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum::<f64>().sqrt();
            dot / (na * nb)
        };
        let to_f32_vec = |t: &Tensor| -> Vec<f32> { t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap() };

        for step in 0..n_incremental {
            let full_step = full_tail.narrow(1, step, 1).unwrap().squeeze(1).unwrap();
            let cos = cosine(&to_f32_vec(&full_step), &to_f32_vec(&incr_outputs[step]));
            let max_diff = to_f32_vec(&full_step)
                .iter()
                .zip(to_f32_vec(&incr_outputs[step]))
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            println!("position {}: cosine={cos}, max_abs_diff={max_diff}", prefix_len + step);
        }
    }
}
