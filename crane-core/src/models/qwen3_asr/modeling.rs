//! Multi-modal projector for Qwen3-ASR: maps the audio encoder's `d_model`-
//! wide output to the text decoder's `output_dim`-wide (`hidden_size`)
//! embedding space.

use std::time::Instant;

use candle_core::{Module, Result, Tensor};
use candle_nn::{Activation, VarBuilder};
use ribo::utils::log;

use super::audio_encoder::AudioEncoder;
use super::config::{AudioConfig, Config};
use crate::models::qwen3::modeling::Qwen3Model;
use crate::models::with_tracing;

/// Two-layer MLP projecting audio encoder output to the text decoder's
/// embedding width: `linear_1` -> activation (GELU on released checkpoints)
/// -> `linear_2`, both linear layers with bias.
///
/// The supplied [`VarBuilder`] must be scoped to the checkpoint's
/// `model.multi_modal_projector` prefix (tensors `linear_1.{weight,bias}`,
/// `linear_2.{weight,bias}` sit directly underneath).
#[derive(Debug, Clone)]
pub struct MultiModalProjector {
    linear_1: with_tracing::Linear,
    linear_2: with_tracing::Linear,
    activation: Activation,
}

impl MultiModalProjector {
    /// Builds the projector from an `audio_config` and a `VarBuilder`
    /// already scoped to `model.multi_modal_projector`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if a required weight tensor is missing or has
    /// an unexpected shape.
    // See `AudioEncoderAttention::new`'s comment on `VarBuilder` by-value.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(config: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let linear_1 = with_tracing::linear(config.d_model, config.d_model, vb.pp("linear_1"))?;
        let linear_2 = with_tracing::linear(config.d_model, config.output_dim, vb.pp("linear_2"))?;
        Ok(Self {
            linear_1,
            linear_2,
            activation: config.activation_function,
        })
    }

    /// Projects `[..., d_model]` audio encoder output to `[..., output_dim]`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `xs`'s last dimension doesn't match
    /// `d_model`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.activation.forward(&self.linear_1.forward(xs)?)?;
        self.linear_2.forward(&xs)
    }
}

/// Top-level Qwen3-ASR model: audio encoder ([`AudioEncoder`]) + projector
/// ([`MultiModalProjector`]) + Qwen3 text decoder
/// ([`Qwen3Model`]), wired
/// together via embedding splice — text tokens are embedded as usual, audio
/// is encoded and projected to the decoder's embedding width, then the
/// projected audio embeddings overwrite the `audio_token_id` placeholder
/// positions before the merged sequence is run through the decoder.
///
/// Assumes `batch == 1` (single-utterance inference); no support for
/// batching multiple utterances in one call.
pub struct Qwen3AsrModel {
    encoder: AudioEncoder,
    projector: MultiModalProjector,
    decoder: Qwen3Model,
    audio_token_id: u32,
    eos_token_id: Vec<u32>,
}

impl Qwen3AsrModel {
    /// Builds the full ASR model from checkpoint weights.
    ///
    /// `vb` must be the checkpoint root `VarBuilder`. Weight scoping:
    /// audio encoder under `model.audio_tower`, projector under
    /// `model.multi_modal_projector`, text decoder directly under
    /// `model.language_model.*` (the checkpoint's `language_model` is a bare
    /// `Qwen3Model`, not a `Qwen3ForCausalLM` wrapper, so there's no further
    /// `model.` nesting — hence [`Qwen3Model::new_from_model_vb`] taking
    /// `lang_vb` as-is rather than [`Qwen3Model::new`], which would add one).
    ///
    /// # Errors
    ///
    /// Returns a candle error if a required weight tensor is missing or has
    /// an unexpected shape.
    // See `AudioEncoderAttention::new`'s comment on `VarBuilder` by-value.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let model_vb = vb.pp("model");
        let encoder = AudioEncoder::new(&config.audio_config, model_vb.pp("audio_tower"))?;
        let projector =
            MultiModalProjector::new(&config.audio_config, model_vb.pp("multi_modal_projector"))?;

        let lang_vb = model_vb.pp("language_model");
        let qwen3_config = config.text_config.to_qwen3_config();
        let decoder = Qwen3Model::new_from_model_vb(&qwen3_config, lang_vb, vb)?;

        Ok(Self {
            encoder,
            projector,
            decoder,
            audio_token_id: config.audio_token_id,
            eos_token_id: config.eos_token_id.clone(),
        })
    }

    /// Prefill forward pass: encodes `mel`, splices the projected audio
    /// embeddings into `input_ids`'s embedded sequence at the contiguous run
    /// of `audio_token_id` placeholders, and runs the text decoder.
    ///
    /// `input_ids` must be `[1, seq_len]`, with exactly as many consecutive
    /// `audio_token_id` positions as the audio encoder produces tokens for
    /// `mel` (the caller expands the template's single placeholder to that
    /// count beforehand, via
    /// [`get_feat_extract_output_lengths`](super::feature_extractor::get_feat_extract_output_lengths)).
    /// `mel` is `[1, n_mels, n_frames]`, as produced by
    /// [`WhisperFeatureExtractor::extract`](super::feature_extractor::WhisperFeatureExtractor::extract).
    ///
    /// Returns logits for the last position, shape `[1, 1, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the number of `audio_token_id` positions in
    /// `input_ids` doesn't match the audio encoder's output token count for
    /// `mel`, or if any tensor operation fails.
    pub fn forward(&mut self, input_ids: &Tensor, mel: &Tensor) -> Result<Tensor> {
        let device = mel.device();

        let embed_start = Instant::now();
        let input_embeds = self.decoder.embed_tokens().forward(input_ids)?;
        let (_batch, seq_len, _hidden_dim) = input_embeds.dims3()?;
        device.synchronize()?;
        let embed_elapsed = embed_start.elapsed();

        // Indices must be known on the host to locate the placeholder run;
        // this sync happens once per utterance during prefill, not per
        // decode step.
        let ids: Vec<u32> = input_ids.squeeze(0)?.to_vec1()?;
        let Some(start) = ids.iter().position(|&id| id == self.audio_token_id) else {
            candle_core::bail!(
                "no audio_token_id ({}) found in input_ids",
                self.audio_token_id,
            );
        };
        let end = ids[start..]
            .iter()
            .position(|&id| id != self.audio_token_id)
            .map_or(ids.len(), |offset| start + offset);
        let placeholder_count = end - start;

        let audio_encoder_start = Instant::now();
        let audio_encoder_out = self.encoder.forward(mel)?;
        device.synchronize()?;
        let audio_encoder_elapsed = audio_encoder_start.elapsed();

        let projector_start = Instant::now();
        let audio_embeds = self.projector.forward(&audio_encoder_out)?;
        let audio_embeds = audio_embeds.to_dtype(input_embeds.dtype())?;
        device.synchronize()?;
        let projector_elapsed = projector_start.elapsed();

        let n_audio_tokens = audio_embeds.dim(1)?;
        if placeholder_count != n_audio_tokens {
            candle_core::bail!(
                "audio token count mismatch: input_ids has {placeholder_count} \
                 consecutive audio_token_id ({}) placeholders but the audio \
                 encoder produced {n_audio_tokens} tokens for the given audio",
                self.audio_token_id,
            );
        }

        let splice_start = Instant::now();
        let prefix = input_embeds.narrow(1, 0, start)?;
        let suffix = input_embeds.narrow(1, end, seq_len - end)?;
        let input_embeds = Tensor::cat(&[&prefix, &audio_embeds, &suffix], 1)?;
        device.synchronize()?;
        let splice_elapsed = splice_start.elapsed();

        let decoder_start = Instant::now();
        let logits = self.decoder.forward_embeds(&input_embeds, 0)?;
        device.synchronize()?;
        let decoder_elapsed = decoder_start.elapsed();

        log::info!(
            "qwen3_asr prefill breakdown: embed {:.1}ms, audio_encoder {:.1}ms, \
             projector {:.1}ms, splice {:.1}ms, decoder {:.1}ms",
            embed_elapsed.as_secs_f64() * 1000.0,
            audio_encoder_elapsed.as_secs_f64() * 1000.0,
            projector_elapsed.as_secs_f64() * 1000.0,
            splice_elapsed.as_secs_f64() * 1000.0,
            decoder_elapsed.as_secs_f64() * 1000.0,
        );

        Ok(logits)
    }

    /// Autoregressive decode step: runs a single new token through the text
    /// decoder's `KV`-cached forward pass. No audio processing happens here
    /// — audio embeddings are only spliced in during the prefill call to
    /// [`Self::forward`].
    ///
    /// `input_ids` is `[1, 1]`; `start_pos` is the number of positions
    /// already processed (prefill length plus prior decode steps).
    ///
    /// Returns logits for the new position, shape `[1, 1, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the decoder forward pass fails.
    pub fn forward_token(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        self.decoder.forward(input_ids, start_pos)
    }

    /// Clears the text decoder's `KV` cache. Call between utterances to
    /// reset autoregressive state.
    pub fn clear_kv_cache(&mut self) {
        self.decoder.clear_kv_cache();
    }

    /// End-of-sequence token ids; generation should stop once any of these
    /// is sampled.
    #[must_use]
    pub fn eos_token_id(&self) -> &[u32] {
        &self.eos_token_id
    }

    /// The `<|audio_pad|>` placeholder token id expanded in `input_ids`.
    #[must_use]
    pub fn audio_token_id(&self) -> u32 {
        self.audio_token_id
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    use super::super::config::{RopeParameters, TextConfig};
    use super::*;

    fn small_audio_config(d_model: usize, output_dim: usize) -> AudioConfig {
        AudioConfig {
            d_model,
            encoder_layers: 2,
            encoder_attention_heads: 2,
            num_key_value_heads: 2,
            encoder_ffn_dim: 16,
            output_dim,
            num_mel_bins: 16,
            downsample_hidden_size: 3,
            max_position_embeddings: 13,
            n_window: 50,
            n_window_infer: 800,
            conv_chunksize: 500,
            activation_function: Activation::Gelu,
            scale_embedding: false,
        }
    }

    fn make_projector_vb(config: &AudioConfig) -> VarBuilder<'static> {
        let device = &Device::Cpu;
        let dtype = DType::F32;
        let d = config.d_model;
        let out = config.output_dim;

        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert(
            "linear_1.weight".into(),
            Tensor::zeros((d, d), dtype, device).expect("zeros"),
        );
        t.insert(
            "linear_1.bias".into(),
            Tensor::zeros(d, dtype, device).expect("zeros"),
        );
        t.insert(
            "linear_2.weight".into(),
            Tensor::zeros((out, d), dtype, device).expect("zeros"),
        );
        t.insert(
            "linear_2.bias".into(),
            Tensor::zeros(out, dtype, device).expect("zeros"),
        );

        VarBuilder::from_tensors(t, dtype, device)
    }

    #[test]
    fn projector_output_shape_0_6b() {
        let config = small_audio_config(896, 1024);
        let vb = make_projector_vb(&config);
        let projector = MultiModalProjector::new(&config, vb).expect("new");

        let xs = Tensor::zeros((1, 13, config.d_model), DType::F32, &Device::Cpu).expect("zeros");
        let out = projector.forward(&xs).expect("forward");
        assert_eq!(out.dims(), &[1, 13, config.output_dim]);
    }

    #[test]
    fn projector_output_shape_1_7b() {
        let config = small_audio_config(1024, 2048);
        let vb = make_projector_vb(&config);
        let projector = MultiModalProjector::new(&config, vb).expect("new");

        let xs = Tensor::zeros((1, 13, config.d_model), DType::F32, &Device::Cpu).expect("zeros");
        let out = projector.forward(&xs).expect("forward");
        assert_eq!(out.dims(), &[1, 13, config.output_dim]);
    }

    const TEST_VOCAB_SIZE: usize = 32;
    const TEST_AUDIO_TOKEN_ID: u32 = 20;

    /// A tiny end-to-end `Config` (audio encoder + text decoder) with
    /// matching `output_dim`/`hidden_size` (8), small enough that a
    /// `VarMap`-backed `VarBuilder` can build every sub-module cheaply.
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

    /// Builds `input_ids` of shape `[1, prefix.len() + n_audio + suffix.len()]`
    /// with `n_audio` consecutive copies of [`TEST_AUDIO_TOKEN_ID`] between
    /// `prefix` and `suffix`.
    fn make_input_ids(prefix: &[u32], n_audio: usize, suffix: &[u32], device: &Device) -> Tensor {
        let mut ids = prefix.to_vec();
        ids.extend(std::iter::repeat_n(TEST_AUDIO_TOKEN_ID, n_audio));
        ids.extend_from_slice(suffix);
        Tensor::new(ids, device)
            .expect("input_ids")
            .unsqueeze(0)
            .expect("unsqueeze")
    }

    #[test]
    fn model_constructor() {
        let config = tiny_asr_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        Qwen3AsrModel::new(&config, vb).expect("new");
    }

    #[test]
    fn forward_output_shape() {
        let config = tiny_asr_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = Qwen3AsrModel::new(&config, vb).expect("new");

        // 100 mel frames = one full window = 13 encoder output tokens.
        let mel = Tensor::zeros((1, 16, 100), DType::F32, &device).expect("mel");
        let input_ids = make_input_ids(&[3], 13, &[4], &device);

        let out = model.forward(&input_ids, &mel).expect("forward");
        assert_eq!(out.dims(), &[1, 1, TEST_VOCAB_SIZE]);
    }

    #[test]
    fn forward_token_output_shape() {
        let config = tiny_asr_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = Qwen3AsrModel::new(&config, vb).expect("new");

        let mel = Tensor::zeros((1, 16, 100), DType::F32, &device).expect("mel");
        let input_ids = make_input_ids(&[3], 13, &[4], &device);
        model.forward(&input_ids, &mel).expect("forward");

        let next_token = Tensor::new(&[[5u32]], &device).expect("next_token");
        let out = model
            .forward_token(&next_token, input_ids.dim(1).expect("dim"))
            .expect("forward_token");
        assert_eq!(out.dims(), &[1, 1, TEST_VOCAB_SIZE]);
    }

    #[test]
    fn forward_count_mismatch_errors() {
        let config = tiny_asr_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = Qwen3AsrModel::new(&config, vb).expect("new");

        // 100 mel frames -> encoder produces 13 tokens, but input_ids only
        // has 5 placeholder positions.
        let mel = Tensor::zeros((1, 16, 100), DType::F32, &device).expect("mel");
        let input_ids = make_input_ids(&[3], 5, &[4], &device);

        let err = model.forward(&input_ids, &mel).unwrap_err().to_string();
        assert!(err.contains("mismatch"), "unexpected error: {err}");
    }

    #[test]
    fn forward_no_audio_tokens_errors() {
        let config = tiny_asr_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = Qwen3AsrModel::new(&config, vb).expect("new");

        let mel = Tensor::zeros((1, 16, 100), DType::F32, &device).expect("mel");
        let input_ids = Tensor::new(&[[3u32, 4, 5]], &device).expect("input_ids");

        let err = model.forward(&input_ids, &mel).unwrap_err().to_string();
        assert!(err.contains("no audio_token_id"), "unexpected error: {err}");
    }
}
