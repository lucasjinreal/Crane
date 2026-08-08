//! MiniCPM-o-4.5 omni-understanding chat: glues [`super::vision::VisionModel`]
//! + [`super::resampler::Resampler`] (image) and [`super::audio_encoder::AudioEncoder`]
//! + [`super::audio::AudioProjector`] (audio) onto [`super::llm::MiniCpmOLlm`]'s
//! plain Qwen3 decoder.
//!
//! Mirrors `crate::models::minicpm_v::vlm::MinicpmV46VLModel`'s shape, with
//! two real differences: positions are **plain scalar** (Qwen3 has no MRoPE,
//! unlike MiniCPM-V-4.6's Qwen-3.5 backbone — no `[3, S]` position tensor
//! needed), and the vision pipeline is SigLIP + `Resampler` (see
//! `super::vision`/`super::resampler` module docs on the packed-pixel-values
//! convention and the fixed per-image token count) rather than NaViT +
//! window-merger + hierarchical `Merger`.
//!
//! Image and audio **understanding** only — no speech output (see the
//! `minicpmo` module doc for phase status). Single-sequence only, same
//! constraint as `minicpm_v`/`qwen3_5_vl`. Both modalities splice over the
//! same `<unk>` placeholder token (`crate::models::qwen3_5::vlm::splice_image_features`
//! doesn't care which tower an embedding came from) — when both are present
//! in one turn, [`MiniCpmOVlModel::forward`] concatenates vision embeddings
//! before audio embeddings, matching [`MiniCpmOVlModel::render_prompt`]'s
//! placeholder order.

use anyhow::{Context, Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use super::audio::AudioProjector;
use super::audio_encoder::{pooled_output_len, AudioEncoder};
use super::config::{load_config, MiniCpmOConfig};
use super::llm::MiniCpmOLlm;
use super::preprocess::{self, build_placeholder, load_preprocessor_config, PreprocessorConfig, ProcessedImage};
use super::resampler::Resampler;
use super::vision::VisionModel;
use crate::models::qwen3_5::vlm::splice_image_features;
use crate::models::qwen3_asr::feature_extractor::WhisperFeatureExtractor;
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::utils;

#[derive(Debug, Clone)]
pub struct VlGenerationConfig {
    pub max_new_tokens: usize,
}

impl Default for VlGenerationConfig {
    fn default() -> Self {
        Self { max_new_tokens: 512 }
    }
}

pub struct MiniCpmOVlModel {
    pub tokenizer: TokenOutputStream,
    pub device: Device,
    pub dtype: DType,
    pub preprocessor: PreprocessorConfig,
    audio_pool_step: usize,
    unk_token_id: u32,
    audio_start_token: String,
    audio_end_token: String,
    eos_token_ids: Vec<u32>,
    tts_bos_token_id: Option<u32>,
    tts_eos_token_id: Option<u32>,
    vision: VisionModel,
    resampler: Resampler,
    audio_encoder: AudioEncoder,
    audio_projector: AudioProjector,
    mel_extractor: WhisperFeatureExtractor,
    llm: MiniCpmOLlm,
}

impl MiniCpmOVlModel {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let llm = MiniCpmOLlm::new(model_path, device, dtype)?;

        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let config_path = std::path::Path::new(model_path).join("config.json");
        let config: MiniCpmOConfig = load_config(&config_path.to_string_lossy())?;

        let filenames = utils::get_safetensors_files(model_path)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, *dtype, device) }
            .with_context(|| format!("mmap safetensors at {model_path}"))?;

        let vision = VisionModel::new(&config.vision_config, vb.pp("vpm")).context("build vision tower")?;
        let vision_dim = config.vision_config.hidden_size;
        let embed_dim = config.llm.hidden_size;
        let resampler = Resampler::new(config.query_num, embed_dim, embed_dim / 128, vision_dim, vb.pp("resampler"))
            .context("build resampler")?;

        let audio_encoder = AudioEncoder::new(&config.audio_config, vb.pp("apm")).context("build audio encoder")?;
        let audio_output_dim = config.audio_config.encoder_ffn_dim / 4;
        let audio_projector = AudioProjector::new(audio_output_dim, embed_dim, config.audio_pool_step, vb.pp("audio_projection_layer"))
            .context("build audio projector")?;
        let mel_extractor = WhisperFeatureExtractor::new(config.audio_config.num_mel_bins, device, *dtype)?;

        let preprocessor = load_preprocessor_config(model_path)?;

        let vocab = tokenizer.get_vocab(true);
        let unk_token_id = *vocab
            .get(preprocessor.unk.as_str())
            .with_context(|| format!("tokenizer has no {} token", preprocessor.unk))?;
        let eos_token_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| vocab.get(*t).copied())
            .collect();
        if eos_token_ids.is_empty() {
            anyhow::bail!("tokenizer has neither <|im_end|> nor <|endoftext|>");
        }
        let tts_bos_token_id = vocab.get("<|tts_bos|>").copied();
        let tts_eos_token_id = vocab.get("<|tts_eos|>").copied();

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype: *dtype,
            preprocessor,
            audio_pool_step: config.audio_pool_step,
            unk_token_id,
            audio_start_token: "<|audio_start|>".to_string(),
            audio_end_token: "<|audio_end|>".to_string(),
            eos_token_ids,
            tts_bos_token_id,
            tts_eos_token_id,
            vision,
            resampler,
            audio_encoder,
            audio_projector,
            mel_extractor,
            llm,
        })
    }

    /// Run the vision tower + resampler on a packed image batch.
    ///
    /// `pixel_values`: `[1, C, patch_size, total_patches*patch_size]`.
    /// `target_sizes`: per-image/-slice `(h, w)` patch-grid dims.
    ///
    /// Returns `[num_images, query_num, llm_hidden]`.
    pub fn encode_images(&self, pixel_values: &Tensor, target_sizes: &[(usize, usize)]) -> Result<Tensor> {
        // VisionModel expects a real batch dim per image/slice (padded-batch
        // SigLIP, not NaViT-packed-into-one-row like minicpm_v) — reshape
        // the packed `[1, C, patch, total*patch]` into `[num_images, C,
        // patch, max_patches*patch]`, padding each image's own row.
        let (_one, c, patch_size, _total_cols) = pixel_values.dims4()?;
        let max_patches = target_sizes.iter().map(|&(h, w)| h * w).max().unwrap_or(0);
        let device = pixel_values.device();

        let mut rows = Vec::with_capacity(target_sizes.len());
        let mut col_offset = 0usize;
        for &(h, w) in target_sizes {
            let n = h * w;
            let cols = n * patch_size;
            let row = pixel_values.narrow(3, col_offset * patch_size, cols)?; // [1, C, patch, cols]
            col_offset += n;
            let row = if n < max_patches {
                let pad = Tensor::zeros((1, c, patch_size, (max_patches - n) * patch_size), row.dtype(), device)?;
                Tensor::cat(&[&row, &pad], 3)?
            } else {
                row
            };
            rows.push(row);
        }
        let batched = Tensor::cat(&rows, 0)?; // [num_images, C, patch, max_patches*patch]

        let vision_out = self.vision.forward(&batched, target_sizes).context("vision forward")?;
        self.resampler.forward(&vision_out, target_sizes).context("resampler forward")
    }

    /// Run the audio encoder + projector on one mono 16kHz PCM clip.
    ///
    /// Returns `[1, pooled_len, llm_hidden]`.
    pub fn encode_audio(&self, samples: &[f32]) -> Result<Tensor> {
        let features = self.mel_extractor.extract(samples)?;
        let encoder_out = self.audio_encoder.forward(&features.input_features).context("audio encoder forward")?;
        self.audio_projector.forward(&encoder_out).context("audio projector forward")
    }

    /// Forward pass for a prefill with optional image/audio (or text-only,
    /// when both are `None`).
    ///
    /// `input_ids`: `[1, S]`, already containing the expanded placeholder(s)
    /// (see [`Self::render_prompt`]).
    ///
    /// Returns logits `[1, 1, V]` (last position only).
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        target_sizes: Option<&[(usize, usize)]>,
        audio_samples: Option<&[f32]>,
        start_pos: usize,
    ) -> Result<Tensor> {
        let image_embeds = match (pixel_values, target_sizes) {
            (Some(pv), Some(ts)) => Some(self.encode_images(pv, ts)?),
            _ => None,
        };
        let audio_embeds = audio_samples.map(|s| self.encode_audio(s)).transpose()?;

        let mut hidden_states = self.llm.embed_only(input_ids)?;

        let mut placeholder_embeds = Vec::new();
        if let Some(img_emb) = &image_embeds {
            let (n, q, h) = img_emb.dims3()?;
            placeholder_embeds.push(img_emb.reshape((n * q, h))?);
        }
        if let Some(aud_emb) = &audio_embeds {
            let (n, t, h) = aud_emb.dims3()?;
            placeholder_embeds.push(aud_emb.reshape((n * t, h))?);
        }
        if !placeholder_embeds.is_empty() {
            let refs: Vec<&Tensor> = placeholder_embeds.iter().collect();
            let combined = Tensor::cat(&refs, 0)?;
            hidden_states = splice_image_features(input_ids, &hidden_states, &combined, self.unk_token_id)?;
        }

        self.llm.forward_embeds(&hidden_states, start_pos).map_err(anyhow::Error::from)
    }

    pub fn decode_step(&mut self, token: u32, start_pos: usize) -> Result<Tensor> {
        let input = Tensor::from_vec(vec![token], (1usize, 1usize), &self.device)?;
        let hidden = self.llm.embed_only(&input)?;
        self.llm.forward_embeds(&hidden, start_pos).map_err(anyhow::Error::from)
    }

    pub fn clear_kv_cache(&mut self) {
        self.llm.clear_kv_cache();
    }

    /// Build the `<|audio_start|><unk>...<|audio_end|>` placeholder for one
    /// audio clip — direct port of the non-chunked branch of
    /// `get_audio_placeholder` (`chunk_input=False`). Takes the *real*
    /// extracted mel frame count (not an approximation from the raw sample
    /// count) so this always matches [`Self::encode_audio`]'s actual output
    /// length exactly — a mismatch here would desync the placeholder-token
    /// count from the spliced-in embedding count.
    fn audio_placeholder(&self, mel_frames: usize) -> String {
        let output_lens = pooled_output_len(mel_frames, self.audio_pool_step);
        format!("{}{}{}", self.audio_start_token, "<unk>".repeat(output_lens), self.audio_end_token)
    }

    /// Preprocess `image`/`audio` and build the `input_ids` for one user
    /// turn. `audio` is mono 16kHz PCM. Returns `None` for `ProcessedImage`
    /// when `image` is `None` (and no audio placeholder when `audio` is
    /// `None`).
    pub fn render_prompt(
        &self,
        image: Option<&image::DynamicImage>,
        audio: Option<&[f32]>,
        user_text: &str,
    ) -> Result<(Vec<u32>, Option<ProcessedImage>)> {
        let processed = image.map(|img| preprocess::process_image(img, &self.preprocessor, &self.device, self.dtype)).transpose()?;

        let mut user_content = String::new();
        if let Some(p) = &processed {
            user_content.push_str(&build_placeholder(p, &self.preprocessor, 0));
            user_content.push('\n');
        }
        if let Some(samples) = audio {
            let mel_frames = self.mel_extractor.extract(samples)?.real_frame_count;
            user_content.push_str(&self.audio_placeholder(mel_frames));
            user_content.push('\n');
        }
        user_content.push_str(user_text);

        // The real chat template appends a pre-closed empty think block
        // after the generation prompt whenever `enable_thinking=False`
        // (this model's default for plain omni Q&A) — the LLM was trained
        // expecting that boundary already in the prompt, not to produce it
        // itself; omitting it here produced garbled/duplicated `<think>`
        // output in testing.
        let rendered = format!("<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");

        let base = self.tokenizer.tokenizer.encode(rendered, false).map_err(E::msg)?;
        let ids = base.get_ids().to_vec();
        Ok((ids, processed))
    }

    /// Answer `user_text` about `image`/`audio` (pass `None` for either/both
    /// to skip that modality). Resets the KV cache, so each call is an
    /// independent single turn.
    pub fn generate(
        &mut self,
        image: Option<&image::DynamicImage>,
        audio: Option<&[f32]>,
        user_text: &str,
        cfg: &VlGenerationConfig,
        mut on_token: impl FnMut(&str),
    ) -> Result<String> {
        let (input_ids, processed) = self.render_prompt(image, audio, user_text)?;
        let input_tensor = Tensor::from_vec(input_ids.clone(), (1usize, input_ids.len()), &self.device)?;

        let (pixel_values, target_sizes) = match &processed {
            Some(p) => (Some(p.pixel_values.clone().unsqueeze(0)?), Some(p.target_sizes.clone())),
            None => (None, None),
        };

        self.clear_kv_cache();
        let mut logits = self.forward(&input_tensor, pixel_values.as_ref(), target_sizes.as_deref(), audio, 0)?;

        let mut generated: Vec<u32> = Vec::with_capacity(cfg.max_new_tokens);
        let mut cur_pos = input_ids.len();
        for _ in 0..cfg.max_new_tokens {
            let next = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;
            if self.eos_token_ids.contains(&next) {
                break;
            }
            if let Ok(text) = self.tokenizer.tokenizer.decode(&[next], true) {
                on_token(&text);
            }
            generated.push(next);
            logits = self.decode_step(next, cur_pos)?;
            cur_pos += 1;
        }

        self.tokenizer.tokenizer.decode(&generated, true).map_err(E::msg)
    }

    /// Like [`Self::render_prompt`], but appends `<|tts_bos|>` after the
    /// think block — the chat template's `use_tts_template=True` branch,
    /// auto-enabled whenever the turn should produce a spoken response.
    /// The chat LLM is trained expecting this marker already in the prompt
    /// (not to generate it itself), same reasoning as the think-block fix.
    fn render_prompt_for_tts(
        &self,
        image: Option<&image::DynamicImage>,
        audio: Option<&[f32]>,
        user_text: &str,
    ) -> Result<(Vec<u32>, Option<ProcessedImage>)> {
        let tts_bos = self.tts_bos_token_id.context("tokenizer has no <|tts_bos|> token")?;
        let (mut ids, processed) = self.render_prompt(image, audio, user_text)?;
        ids.push(tts_bos);
        Ok((ids, processed))
    }

    /// Generate a spoken chat response: same generation loop as
    /// [`Self::generate`], but also captures each generated position's
    /// final-layer hidden state (needed by [`super::tts_llm::MiniCpmTts`]'s
    /// `build_condition_embeds`) and stops at `<|tts_eos|>` in addition to
    /// the normal end-of-turn tokens — direct port of the `tts_bound`
    /// slicing in `_generate_speech_non_streaming`, see that method's
    /// module-doc reference for why decode-step hidden states alone (no
    /// prefill rows) are exactly what's needed here.
    ///
    /// Returns `(text, response_token_ids, hidden_states [1, response_len, hidden])`.
    pub fn generate_for_tts(
        &mut self,
        image: Option<&image::DynamicImage>,
        audio: Option<&[f32]>,
        user_text: &str,
        cfg: &VlGenerationConfig,
    ) -> Result<(String, Vec<u32>, Tensor)> {
        let (input_ids, processed) = self.render_prompt_for_tts(image, audio, user_text)?;
        let input_tensor = Tensor::from_vec(input_ids.clone(), (1usize, input_ids.len()), &self.device)?;

        let (pixel_values, target_sizes) = match &processed {
            Some(p) => (Some(p.pixel_values.clone().unsqueeze(0)?), Some(p.target_sizes.clone())),
            None => (None, None),
        };

        self.clear_kv_cache();
        let mut logits = self.forward(&input_tensor, pixel_values.as_ref(), target_sizes.as_deref(), audio, 0)?;

        let mut generated: Vec<u32> = Vec::with_capacity(cfg.max_new_tokens);
        let mut hidden_rows: Vec<Tensor> = Vec::with_capacity(cfg.max_new_tokens);
        let mut cur_pos = input_ids.len();
        for _ in 0..cfg.max_new_tokens {
            let next = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;
            if self.eos_token_ids.contains(&next) || Some(next) == self.tts_eos_token_id {
                break;
            }
            generated.push(next);
            logits = self.decode_step(next, cur_pos)?;
            // This step's hidden state corresponds to the token we just fed
            // in (`next`), matching HF's per-generation-step
            // `output_hidden_states` semantics that `tts_bound` indexes into.
            let hidden = self.llm.last_hidden_states().context("no hidden states after forward")?.clone();
            hidden_rows.push(hidden);
            cur_pos += 1;
        }

        let text = self.tokenizer.tokenizer.decode(&generated, true).map_err(E::msg)?;
        let hidden_states = Tensor::cat(&hidden_rows, 1)?;
        Ok((text, generated, hidden_states))
    }
}
