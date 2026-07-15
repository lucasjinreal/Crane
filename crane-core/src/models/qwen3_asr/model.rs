//! High-level Qwen3-ASR model wrapper.
//!
//! Handles model loading from a HuggingFace-style checkpoint directory,
//! audio feature extraction, prompt construction, and the autoregressive
//! decode loop that turns audio into transcribed text.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;

use super::config::Config;
use super::feature_extractor::{
    SAMPLE_RATE, WhisperFeatureExtractor, get_feat_extract_output_lengths,
};
use super::modeling::Qwen3AsrModel;
use crate::autotokenizer::AutoTokenizer;
use crate::generation::TranscribeOptions;
use crate::utils::utils::get_safetensors_files;

/// Builds the fixed Qwen3-ASR chat-template prompt for a single audio-only
/// user turn, with `n_audio_tokens` copies of `<|audio_pad|>` in place of the
/// template's single placeholder (§3 of the design doc: expansion happens as
/// text substitution before tokenization, mirroring
/// `Qwen3ASRProcessor.replace_multimodal_special_tokens`).
fn build_asr_prompt(n_audio_tokens: usize) -> String {
    let mut prompt = String::with_capacity(96 + n_audio_tokens * "<|audio_pad|>".len());
    prompt.push_str("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>");
    for _ in 0..n_audio_tokens {
        prompt.push_str("<|audio_pad|>");
    }
    prompt.push_str("<|audio_end|><|im_end|>\n<|im_start|>assistant\n");
    prompt
}

/// Public Qwen3-ASR model: loads a `-hf` checkpoint and transcribes audio to
/// text.
///
/// Wraps [`Qwen3AsrModel`] (encoder + projector + decoder) with a tokenizer
/// and a [`WhisperFeatureExtractor`], following the same shape as
/// `qwen3_tts::Model`.
pub struct Model {
    inner: Qwen3AsrModel,
    tokenizer: AutoTokenizer,
    feature_extractor: WhisperFeatureExtractor,
    config: Config,
    device: Device,
}

impl Model {
    /// Loads a Qwen3-ASR model from a HuggingFace-style checkpoint
    /// directory.
    ///
    /// Expects `config.json`, `tokenizer_config.json` + `tokenizer.json`,
    /// and `model.safetensors` (or sharded `model-*.safetensors` with a
    /// `model.safetensors.index.json`).
    ///
    /// # Errors
    ///
    /// Returns an error if any required file is missing or malformed, or if
    /// checkpoint weight shapes don't match `config.json`.
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let model_dir = std::path::Path::new(model_path);

        let config_data = std::fs::read(model_dir.join("config.json"))
            .context("reading qwen3_asr config.json")?;
        let config: Config = serde_json::from_slice(&config_data)?;

        let tokenizer = AutoTokenizer::from_pretrained(model_path, None)
            .map_err(|e| anyhow::anyhow!("failed to load qwen3_asr tokenizer: {e}"))?;

        let filenames = get_safetensors_files(model_path)?;
        // Same pattern as every other model loader in this crate (e.g.
        // `qwen3_tts::Model::new`): memory-maps the checkpoint's safetensors
        // files, trusting them to match `config.json`.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, *dtype, device) }?;
        let inner = Qwen3AsrModel::new(&config, vb)?;

        let feature_extractor =
            WhisperFeatureExtractor::new(config.audio_config.num_mel_bins, device, *dtype);

        Ok(Self {
            inner,
            tokenizer,
            feature_extractor,
            config,
            device: device.clone(),
        })
    }

    /// Transcribes `audio` (mono `f32` PCM at [`Self::sample_rate`]) to
    /// text.
    ///
    /// Runs one non-streaming call: extract mel features, prefill the
    /// decoder with the spliced audio/text embeddings, then autoregressively
    /// decode until an EOS token or `opts.max_new_tokens` is reached.
    ///
    /// # Errors
    ///
    /// Returns an error if feature extraction, tokenization, or any model
    /// forward pass fails.
    pub fn transcribe(&mut self, audio: &[f32], opts: &TranscribeOptions) -> Result<String> {
        let features = self.feature_extractor.extract(audio)?;
        let n_audio_tokens = get_feat_extract_output_lengths(features.real_frame_count);

        let prompt = build_asr_prompt(n_audio_tokens);
        // `add_special_tokens = false`: the ChatML markers and `<|audio_*|>`
        // tokens are already literal in `prompt` and exist as added tokens
        // in the Qwen tokenizer vocabulary.
        let input_ids = self
            .tokenizer
            .encode(&prompt, false)
            .map_err(|e| anyhow::anyhow!("qwen3_asr prompt tokenization failed: {e}"))?;
        let input_ids_tensor = Tensor::new(input_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let prefill_len = input_ids_tensor.dim(1)?;

        // Also clears any cache left behind by a prior call that returned
        // early via `?` before reaching the trailing `clear_kv_cache` below.
        self.inner.clear_kv_cache();

        let temperature = (opts.temperature > 0.0).then_some(opts.temperature);
        let mut logits_processor = LogitsProcessor::new(1024, temperature, opts.top_p);
        // Cloned because `self.inner` is borrowed mutably inside the decode
        // loop below.
        let eos_token_id = self.inner.eos_token_id().to_vec();

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut logits = self
            .inner
            .forward(&input_ids_tensor, &features.input_features)?
            .squeeze(0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let mut pos = prefill_len;

        while generated_tokens.len() < opts.max_new_tokens {
            let sampling_logits = if (opts.repetition_penalty - 1.0).abs() < f32::EPSILON {
                logits
            } else {
                let start_at = generated_tokens.len().saturating_sub(opts.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    opts.repetition_penalty,
                    &generated_tokens[start_at..],
                )?
            };
            let next_token = logits_processor.sample(&sampling_logits)?;
            if eos_token_id.contains(&next_token) {
                break;
            }
            generated_tokens.push(next_token);

            let next_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            logits = self
                .inner
                .forward_token(&next_input, pos)?
                .squeeze(0)?
                .squeeze(0)?
                .to_dtype(DType::F32)?;
            pos += 1;
        }

        self.inner.clear_kv_cache();

        self.tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("qwen3_asr transcript decoding failed: {e}"))
    }

    /// Audio sample rate this model's feature extractor expects, in Hz.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        // SAMPLE_RATE is a fixed 16_000, well within u32 range.
        #[allow(clippy::cast_possible_truncation)]
        let sample_rate = SAMPLE_RATE as u32;
        sample_rate
    }

    /// The model's deserialized `config.json`.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};
    use candle_nn::{Activation, VarBuilder, VarMap};

    use super::super::config::{AudioConfig, RopeParameters, TextConfig};
    use super::*;

    #[test]
    fn prompt_construction_single_token() {
        let prompt = build_asr_prompt(1);
        assert_eq!(
            prompt,
            "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n\
             <|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn prompt_construction_zero_tokens() {
        let prompt = build_asr_prompt(0);
        assert!(prompt.contains("<|audio_start|><|audio_end|>"));
        assert_eq!(prompt.matches("<|audio_pad|>").count(), 0);
    }

    #[test]
    fn prompt_pad_count_matches_encoder_formula() {
        for mel_frames in [100, 150, 830] {
            let n = get_feat_extract_output_lengths(mel_frames);
            let prompt = build_asr_prompt(n);
            assert_eq!(
                prompt.matches("<|audio_pad|>").count(),
                n,
                "mismatch for {mel_frames} mel frames"
            );
        }
    }

    const TEST_VOCAB_SIZE: usize = 32;
    const TEST_AUDIO_TOKEN_ID: u32 = 20;

    fn tiny_asr_config() -> Config {
        Config {
            audio_config: AudioConfig {
                d_model: 8,
                encoder_layers: 1,
                encoder_attention_heads: 2,
                num_key_value_heads: 2,
                encoder_ffn_dim: 16,
                output_dim: 8,
                num_mel_bins: 16,
                downsample_hidden_size: 3,
                max_position_embeddings: 13,
                n_window: 50,
                n_window_infer: 800,
                conv_chunksize: 500,
                activation_function: Activation::Gelu,
                scale_embedding: false,
            },
            text_config: TextConfig {
                vocab_size: TEST_VOCAB_SIZE,
                hidden_size: 8,
                intermediate_size: 16,
                num_hidden_layers: 1,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                head_dim: 4,
                max_position_embeddings: 128,
                rms_norm_eps: 1e-5,
                rope_parameters: RopeParameters {
                    rope_theta: 10_000.0,
                },
                attention_bias: false,
                tie_word_embeddings: true,
                sliding_window: None,
                max_window_layers: 0,
                use_sliding_window: false,
            },
            audio_token_id: TEST_AUDIO_TOKEN_ID,
            timestamp_token_id: 21,
            pad_token_id: 0,
            eos_token_id: vec![1, 2],
            tie_word_embeddings: true,
        }
    }

    /// Exercises the same prefill + decode-loop shape `Model::transcribe`
    /// uses, directly against [`Qwen3AsrModel`] (no tokenizer/checkpoint on
    /// disk needed), to prove the loop terminates and produces valid shapes.
    #[test]
    fn generation_loop_runs() {
        let config = tiny_asr_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = Qwen3AsrModel::new(&config, vb).expect("new");

        // 100 mel frames = one full window = 13 encoder output tokens.
        let mel = Tensor::zeros((1, 16, 100), DType::F32, &device).expect("mel");
        let n_audio = get_feat_extract_output_lengths(100);
        assert_eq!(n_audio, 13);

        let mut ids: Vec<u32> = vec![3];
        ids.extend(std::iter::repeat_n(TEST_AUDIO_TOKEN_ID, n_audio));
        ids.push(4);
        let input_ids = Tensor::new(ids.as_slice(), &device)
            .expect("ids")
            .unsqueeze(0)
            .expect("unsqueeze");
        let prefill_len = input_ids.dim(1).expect("dim");

        model.clear_kv_cache();
        let mut logits = model
            .forward(&input_ids, &mel)
            .expect("prefill")
            .squeeze(0)
            .expect("squeeze")
            .squeeze(0)
            .expect("squeeze")
            .to_dtype(DType::F32)
            .expect("dtype");

        let mut logits_processor = LogitsProcessor::new(1024, None, None);
        let eos_token_id = model.eos_token_id().to_vec();
        let mut generated: Vec<u32> = Vec::new();
        let mut pos = prefill_len;

        while generated.len() < 5 {
            let next_token = logits_processor.sample(&logits).expect("sample");
            if eos_token_id.contains(&next_token) {
                break;
            }
            generated.push(next_token);

            let next_input = Tensor::new(&[next_token], &device)
                .expect("next_input")
                .unsqueeze(0)
                .expect("unsqueeze");
            logits = model
                .forward_token(&next_input, pos)
                .expect("decode")
                .squeeze(0)
                .expect("squeeze")
                .squeeze(0)
                .expect("squeeze")
                .to_dtype(DType::F32)
                .expect("dtype");
            pos += 1;
        }

        assert!(!generated.is_empty());
        for &token in &generated {
            assert!((token as usize) < TEST_VOCAB_SIZE);
        }

        model.clear_kv_cache();
    }
}
