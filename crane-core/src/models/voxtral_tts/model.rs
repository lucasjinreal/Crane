//! Voxtral TTS model configuration, tokenizer integration, voice embedding loader,
//! and end-to-end speech generation.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hound::{SampleFormat, WavSpec, WavWriter};
use serde::Deserialize;
use tekken::Tekkenizer;

use super::codec::CodecDecoder;
use super::modeling::{
    rename_voxtral_transformer_keys, AcousticTransformer, AudioCodebookEmbedding, VoxtralLlm,
};
use crate::generation::SpeechOptions;

// ── Token IDs ──────────────────────────────────────────────────────────────

/// BOS token (id=1, `<s>`).
pub const BOS: u32 = 1;
/// Marks the start of an audio section (id=25).
pub const BEGIN_AUDIO: u32 = 25;
/// `[INST]` instruction start marker (id=35).
pub const INST: u32 = 35;
/// `[/INST]` instruction end marker (id=36).
pub const INST_END: u32 = 36;

/// End-of-audio semantic code (special value returned by the semantic head to
/// signal that generation is complete).
pub const END_AUDIO_CODE: u32 = 1;

// ── Config types ───────────────────────────────────────────────────────────

/// Top-level Voxtral TTS configuration parsed from `params.json`.
#[derive(Debug, Deserialize)]
pub struct VoxtralConfig {
    /// Hidden dimension of the LLM backbone (3072).
    pub dim: usize,
    /// Number of transformer layers in the LLM backbone (26).
    pub n_layers: usize,
    /// Attention head dimension (128).
    pub head_dim: usize,
    /// `SwiGLU` FFN intermediate dimension (9216).
    pub hidden_dim: usize,
    /// Number of query heads (32).
    pub n_heads: usize,
    /// Number of key-value heads (GQA, 8).
    pub n_kv_heads: usize,
    /// `RoPE` base frequency for the LLM backbone (1 000 000.0).
    pub rope_theta: f64,
    /// `RMSNorm` epsilon (1e-5).
    pub norm_eps: f64,
    /// Tekken vocabulary size (131 072).
    pub vocab_size: usize,
    /// Maximum sequence length for pre-computed `RoPE` tables (65 536).
    pub max_seq_len: usize,
    /// Whether the token embedding and output projection share weights.
    pub tied_embeddings: bool,
    /// Multimodal (audio) configuration.
    pub multimodal: MultimodalConfig,
}

/// Multimodal section of `params.json`.
#[derive(Debug, Deserialize)]
pub struct MultimodalConfig {
    /// BOS token id (1).
    pub bos_token_id: u32,
    /// Audio model (LLM + acoustic transformer) configuration.
    pub audio_model_args: AudioModelArgs,
    /// Audio codec (encoder/decoder) configuration.
    pub audio_tokenizer_args: AudioTokenizerArgs,
}

/// Configuration for the LLM backbone's audio head and the acoustic transformer.
#[derive(Debug, Deserialize)]
pub struct AudioModelArgs {
    /// Number of VQ codes in the semantic codebook (8192).
    pub semantic_codebook_size: usize,
    /// Number of FSQ levels for acoustic codes (21).
    pub acoustic_codebook_size: usize,
    /// Number of acoustic codebooks (36).
    pub n_acoustic_codebook: usize,
    /// Token id used as the audio placeholder token in the LLM input (24).
    pub audio_token_id: u32,
    /// Token id that marks the beginning of an audio section (25).
    pub begin_audio_token_id: u32,
    /// Flow-matching acoustic transformer configuration.
    pub acoustic_transformer_args: AcousticTransformerArgs,
}

/// Configuration for the 3-layer flow-matching acoustic transformer.
#[derive(Debug, Deserialize)]
pub struct AcousticTransformerArgs {
    /// Hidden dimension (3072).
    pub dim: usize,
    /// Number of transformer layers (3).
    pub n_layers: usize,
    /// Attention head dimension (128).
    pub head_dim: usize,
    /// `SwiGLU` FFN intermediate dimension (9216).
    pub hidden_dim: usize,
    /// Number of query heads (32).
    pub n_heads: usize,
    /// Number of key-value heads (8).
    pub n_kv_heads: usize,
    /// `RoPE` base frequency — present in `params.json` but unused (acoustic
    /// transformer does not use `RoPE`).
    #[serde(default)]
    pub rope_theta: f64,
}

/// Configuration for the audio codec decoder.
#[derive(Debug, Deserialize)]
pub struct AudioTokenizerArgs {
    /// Output sample rate in Hz (24 000).
    pub sampling_rate: usize,
    /// Number of audio samples per codec frame (240).
    pub pretransform_patch_size: usize,
    /// Number of entries in the semantic VQ codebook (8192).
    pub semantic_codebook_size: usize,
    /// Dimension of each semantic codebook embedding (256).
    pub semantic_dim: usize,
    /// Number of FSQ levels for acoustic codes (21).
    pub acoustic_codebook_size: usize,
    /// Number of acoustic FSQ dimensions (36).
    pub acoustic_dim: usize,
    /// Codec transformer hidden dimension (1024).
    pub dim: usize,
    /// Codec `SwiGLU` FFN intermediate dimension (4096).
    pub hidden_dim: usize,
    /// Codec attention head dimension (128).
    pub head_dim: usize,
    /// Number of query heads in the codec transformer (8).
    pub n_heads: usize,
    /// Number of key-value heads in the codec transformer (8).
    pub n_kv_heads: usize,
    /// QK norm epsilon (1e-6).
    pub qk_norm_eps: f64,
    /// `RMSNorm` epsilon for the codec transformer (0.01).
    pub norm_eps: f64,
    /// Initial value for layer scale parameters (0.01).
    pub layer_scale_init: f64,
    /// Transformer layer counts per codec stage (from `"2,2,2,2"` in `params.json`).
    #[serde(
        rename = "decoder_transformer_lengths_str",
        deserialize_with = "deserialize_comma_separated"
    )]
    pub decoder_transformer_lengths: Vec<usize>,
    /// Convolution kernel sizes per codec stage (from `"3,4,4,4"` in `params.json`).
    #[serde(
        rename = "decoder_convs_kernels_str",
        deserialize_with = "deserialize_comma_separated"
    )]
    pub decoder_convs_kernels: Vec<usize>,
    /// Convolution strides per codec stage (from `"1,2,2,2"` in `params.json`).
    #[serde(
        rename = "decoder_convs_strides_str",
        deserialize_with = "deserialize_comma_separated"
    )]
    pub decoder_convs_strides: Vec<usize>,
    /// Map of voice name to integer voice ID.
    pub voice: HashMap<String, usize>,
}

fn deserialize_comma_separated<'de, D>(deserializer: D) -> Result<Vec<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(serde::de::Error::custom)
        })
        .collect()
}

impl VoxtralConfig {
    /// Load configuration from the `params.json` file in `model_dir`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the JSON is malformed.
    pub fn from_model_dir(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("params.json");
        let data =
            std::fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        serde_json::from_slice(&data).with_context(|| format!("failed to parse {}", path.display()))
    }
}

// ── Prompt construction ────────────────────────────────────────────────────

/// A segment in the Voxtral TTS prompt embedding sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptSegment {
    /// A single token to be embedded via the LLM token embedding table.
    Token(u32),
    /// Placeholder for the pre-computed voice embedding tensor `[N, 3072]`.
    VoiceEmbeddings,
}

/// Build the ordered sequence of prompt segments for Voxtral TTS.
///
/// The prompt layout (per the reference implementation) is:
/// ```text
/// [BOS=1] [BEGIN_AUDIO=25] <voice_embeddings> [INST_END=36]
///   text_token_0 ... text_token_M [INST=35] [BEGIN_AUDIO=25]
/// ```
///
/// The returned `Vec` uses `PromptSegment::VoiceEmbeddings` as a placeholder;
/// callers replace it with the actual `[N, 3072]` BF16 tensor in Step 2.
///
/// # Note
///
/// `text_token_ids` must be encoded **without** BOS or EOS; this function
/// inserts its own `BOS` at the start of the sequence.
#[must_use]
pub fn build_prompt_segments(text_token_ids: &[u32]) -> Vec<PromptSegment> {
    let mut segments = Vec::with_capacity(6 + text_token_ids.len());
    segments.push(PromptSegment::Token(BOS));
    segments.push(PromptSegment::Token(BEGIN_AUDIO));
    segments.push(PromptSegment::VoiceEmbeddings);
    segments.push(PromptSegment::Token(INST_END));
    for &id in text_token_ids {
        segments.push(PromptSegment::Token(id));
    }
    segments.push(PromptSegment::Token(INST));
    segments.push(PromptSegment::Token(BEGIN_AUDIO));
    segments
}

// ── Voice embedding loader ─────────────────────────────────────────────────

/// Internal embedding dimension, fixed by the Voxtral architecture.
///
/// Voice embeddings produced by the codec encoder always have this width.
/// Must equal `VoxtralConfig::dim` (3072). Intentionally not read from the
/// config so that `load_voice_embedding` has no config dependency.
const EMBED_DIM: usize = 3072;
const SILENCE_THRESHOLD: f32 = 1e-8;

/// Load a pre-computed voice embedding from a `PyTorch` `.pt` file.
///
/// The file is a `PyTorch` ZIP archive containing a single BF16 tensor of
/// shape `[N, 3072]`, where N is the number of reference audio frames for
/// the chosen voice.  The archive entry is tried at several conventional
/// `PyTorch` paths.  Returns the tensor on `device` in BF16.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, the ZIP structure is invalid,
/// no known tensor entry is found, or tensor construction fails.
///
pub fn load_voice_embedding(path: &Path, device: &Device) -> Result<Tensor> {
    use std::io::Read as _;
    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(std::io::BufReader::new(file))
        .with_context(|| format!("failed to read ZIP in {}", path.display()))?;

    // PyTorch saves the raw tensor data at one of these paths inside the archive.
    let candidates = ["voice_embed/data/0", "archive/data/0", "data/0"];
    let mut bytes = Vec::new();
    let mut found = false;
    for &name in &candidates {
        if let Ok(mut entry) = archive.by_name(name) {
            entry
                .read_to_end(&mut bytes)
                .with_context(|| format!("failed to read '{}' from {}", name, path.display()))?;
            found = true;
            break;
        }
    }
    anyhow::ensure!(
        found,
        "could not find tensor data entry in {}; tried {:?}",
        path.display(),
        candidates
    );

    // Each BF16 element is 2 bytes; the tensor is [N, EMBED_DIM].
    anyhow::ensure!(
        bytes.len() % (EMBED_DIM * 2) == 0,
        "voice embedding size {} is not a multiple of {} (EMBED_DIM * 2 bytes) in {}",
        bytes.len(),
        EMBED_DIM * 2,
        path.display()
    );
    let n_frames = bytes.len() / (EMBED_DIM * 2);

    Tensor::from_raw_buffer(&bytes, DType::BF16, &[n_frames, EMBED_DIM], device).with_context(
        || {
            format!(
                "failed to create voice embedding tensor from {}",
                path.display()
            )
        },
    )
}

// ── Tekken tokenizer ───────────────────────────────────────────────────────

/// Load the Tekken tokenizer from `tekken.json` in `model_dir`.
///
/// # Errors
///
/// Returns an error if the file cannot be read or the tokenizer JSON is invalid.
pub fn load_tokenizer(model_dir: &Path) -> Result<Tekkenizer> {
    let path = model_dir.join("tekken.json");
    Tekkenizer::from_file(&path).map_err(|e| {
        anyhow::anyhow!(
            "failed to load Tekken tokenizer from {}: {}",
            path.display(),
            e
        )
    })
}

// ── End-to-end Model ──────────────────────────────────────────────────────

/// End-to-end Voxtral TTS model.
///
/// Owns the LLM backbone, acoustic transformer, audio codebook embeddings,
/// codec decoder, Tekken tokenizer, and pre-loaded voice embeddings. Call
/// [`Model::generate_speech`] to go from text + voice name to a waveform tensor.
pub struct Model {
    llm: VoxtralLlm,
    acoustic: AcousticTransformer,
    codebook_embed: AudioCodebookEmbedding,
    codec: CodecDecoder,
    tokenizer: Tekkenizer,
    voices: HashMap<String, Tensor>,
    config: VoxtralConfig,
    device: Device,
    dtype: DType,
}

impl Model {
    /// Load the complete Voxtral TTS model from a checkpoint directory.
    ///
    /// Expects:
    /// - `params.json` — model configuration
    /// - `tekken.json` — Tekken tokenizer
    /// - `consolidated.safetensors` — all weights (LLM + acoustic TX + codec)
    /// - `voice_embedding/*.pt` — pre-computed voice embeddings
    ///
    /// # Errors
    ///
    /// Returns an error if any required file is missing, weights are
    /// incompatible, or tensor construction fails.
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let model_dir = Path::new(model_path);

        let config = VoxtralConfig::from_model_dir(model_dir)?;
        let tokenizer = load_tokenizer(model_dir)?;

        let safetensors_path = model_dir.join("consolidated.safetensors");
        anyhow::ensure!(
            safetensors_path.exists(),
            "consolidated.safetensors not found in {}",
            model_dir.display()
        );
        let raw = candle_core::safetensors::load(&safetensors_path, device)
            .with_context(|| format!("failed to load {}", safetensors_path.display()))?;
        let renamed = rename_voxtral_transformer_keys(raw);
        let vb = VarBuilder::from_tensors(renamed, *dtype, device);

        let llm = VoxtralLlm::new(&config, vb.clone(), device)
            .map_err(|e| anyhow::anyhow!("failed to construct VoxtralLlm: {e}"))?;
        let acoustic = AcousticTransformer::new(&config, vb.clone())
            .map_err(|e| anyhow::anyhow!("failed to construct AcousticTransformer: {e}"))?;
        let codebook_embed = AudioCodebookEmbedding::new(&config, vb.clone())
            .map_err(|e| anyhow::anyhow!("failed to construct AudioCodebookEmbedding: {e}"))?;
        let codec = CodecDecoder::new(
            &config.multimodal.audio_tokenizer_args,
            vb.pp("audio_tokenizer"),
        )
        .map_err(|e| anyhow::anyhow!("failed to construct CodecDecoder: {e}"))?;

        let voice_dir = model_dir.join("voice_embedding");
        let mut voices = HashMap::new();
        for name in config.multimodal.audio_tokenizer_args.voice.keys() {
            let pt_path = voice_dir.join(format!("{name}.pt"));
            if pt_path.exists() {
                let emb = load_voice_embedding(&pt_path, device)?;
                voices.insert(name.clone(), emb);
            }
        }
        anyhow::ensure!(
            !voices.is_empty(),
            "no voice embeddings found in {}",
            voice_dir.display()
        );

        Ok(Self {
            llm,
            acoustic,
            codebook_embed,
            codec,
            tokenizer,
            voices,
            config,
            device: device.clone(),
            dtype: *dtype,
        })
    }

    /// Build prompt embeddings from segments and a voice embedding.
    ///
    /// Each `Token` segment is looked up in the LLM token embedding table;
    /// `VoiceEmbeddings` is replaced by `voice_embed`. Returns a tensor of
    /// shape `[1, seq_len, dim]`.
    fn build_prompt_embeds(
        &self,
        segments: &[PromptSegment],
        voice_embed: &Tensor,
    ) -> Result<Tensor> {
        let mut parts: Vec<Tensor> = Vec::with_capacity(segments.len());
        for seg in segments {
            match seg {
                PromptSegment::Token(id) => {
                    let id_tensor = Tensor::new(&[*id], &self.device)?;
                    let emb = self
                        .llm
                        .embed_tokens(&id_tensor)
                        .map_err(|e| anyhow::anyhow!("embed_tokens failed: {e}"))?
                        .to_dtype(self.dtype)?;
                    parts.push(emb);
                }
                PromptSegment::VoiceEmbeddings => {
                    parts.push(voice_embed.clone());
                }
            }
        }
        Ok(Tensor::cat(&parts, 0)?.unsqueeze(0)?) // [1, seq, dim]
    }

    /// Scale `audio` so its peak absolute sample equals `target_peak`.
    ///
    /// The raw codec output is typically very quiet (< 0.15 peak).  This
    /// normalization brings the output to a consistent volume independent of
    /// the specific codes generated.  Silent audio (peak < [`SILENCE_THRESHOLD`])
    /// is returned unchanged to avoid division by zero.
    fn normalize_peak(audio: &Tensor, target_peak: f32) -> candle_core::Result<Tensor> {
        let audio_f32 = audio.to_dtype(DType::F32)?;
        let max_abs: f32 = audio_f32.abs()?.max_all()?.to_scalar()?;
        if max_abs < SILENCE_THRESHOLD {
            return Ok(audio_f32);
        }
        audio_f32.affine(f64::from(target_peak) / f64::from(max_abs), 0.0)
    }

    /// Generate speech from text using a named voice.
    ///
    /// `language`, `temperature`, `top_p`, and `repetition_penalty` are accepted
    /// for API compatibility with the server's TTS interface but are not used —
    /// Voxtral TTS always uses greedy decoding and language is implicit in the
    /// text.
    ///
    /// Returns `(waveform, sample_rate)` where `waveform` has shape `[1, n_samples]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the voice is unknown, tokenization fails, or any
    /// tensor operation fails during generation or decoding.
    ///
    /// # Panics
    ///
    /// Panics if the model was constructed with no voice embeddings, which
    /// cannot happen because [`Model::new`] ensures at least one voice exists.
    #[allow(unused_variables)]
    pub fn generate_speech(
        &mut self,
        text: &str,
        language: &str,
        voice: Option<&str>,
        opts: &SpeechOptions,
    ) -> Result<(Tensor, u32)> {
        self.llm.clear_kv_cache();

        let voice_name = match voice {
            Some(v) => v.to_string(),
            None => self
                .voices
                .keys()
                .next()
                .expect("at least one voice")
                .clone(),
        };
        let voice_embed = self
            .voices
            .get(&voice_name)
            .ok_or_else(|| anyhow::anyhow!("unknown voice '{voice_name}'"))?
            .to_dtype(self.dtype)?;

        let token_ids = self
            .tokenizer
            .encode(text, false, false)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let segments = build_prompt_segments(&token_ids);

        let prompt_embeds = self.build_prompt_embeds(&segments, &voice_embed)?;

        // Prefill: the last hidden state (from [BEGIN_AUDIO]) seeds the first
        // audio frame prediction — exactly as the reference implementation does.
        let prefill_h = self
            .llm
            .forward(&prompt_embeds, 0)
            .map_err(|e| anyhow::anyhow!("LLM prefill failed: {e}"))?;
        let prompt_len = prompt_embeds.dim(1)?;
        let mut h_for_frame = prefill_h.i((.., prompt_len - 1, ..))?.unsqueeze(1)?; // [1, 1, dim]
        drop(prefill_h);

        anyhow::ensure!(
            prompt_len + opts.max_new_tokens <= self.config.max_seq_len,
            "prompt ({prompt_len}) + max_new_tokens ({}) exceeds max_seq_len ({})",
            opts.max_new_tokens,
            self.config.max_seq_len,
        );

        let n_codebooks = 1 + self.config.multimodal.audio_model_args.n_acoustic_codebook;
        let mut all_codes: Vec<Vec<u32>> = Vec::new();
        let mut frame_codes = Vec::with_capacity(n_codebooks);

        for frame_idx in 0..opts.max_new_tokens {
            let h_squeezed = h_for_frame.reshape(self.config.dim)?; // [1,1,dim] -> [dim]

            let semantic_code = self
                .acoustic
                .predict_semantic_code(&h_squeezed)
                .map_err(|e| anyhow::anyhow!("predict_semantic_code failed: {e}"))?;

            if semantic_code == END_AUDIO_CODE {
                break;
            }

            let acoustic_codes = self
                .acoustic
                .flow_match_inference(&h_squeezed)
                .map_err(|e| anyhow::anyhow!("flow_match_inference failed: {e}"))?;
            let acoustic_vec: Vec<u32> = acoustic_codes.to_vec1()?;

            frame_codes.clear();
            frame_codes.push(semantic_code);
            frame_codes.extend_from_slice(&acoustic_vec);

            if frame_idx + 1 < opts.max_new_tokens {
                let codes_tensor = Tensor::new(frame_codes.as_slice(), &self.device)?;
                let summed_embed = self
                    .codebook_embed
                    .forward(&codes_tensor)
                    .map_err(|e| anyhow::anyhow!("codebook embed failed: {e}"))?
                    .to_dtype(self.dtype)?
                    .unsqueeze(0)?
                    .unsqueeze(0)?; // [1, 1, dim]
                let start_pos = prompt_len + frame_idx;
                h_for_frame = self
                    .llm
                    .forward(&summed_embed, start_pos)
                    .map_err(|e| anyhow::anyhow!("LLM decode step failed: {e}"))?;
            }
            all_codes.push(frame_codes.clone());
        }

        anyhow::ensure!(!all_codes.is_empty(), "no speech frames generated");

        let n_frames = all_codes.len();
        #[allow(clippy::cast_precision_loss)]
        // codes are small integers (semantic <= 8191, acoustic <= 20)
        let flat: Vec<f32> = all_codes
            .iter()
            .flat_map(|f| f.iter().copied().map(|c| c as f32))
            .collect();
        let codes_for_codec =
            Tensor::new(flat.as_slice(), &self.device)?.reshape((1, n_frames, n_codebooks))?;

        let waveform = self
            .codec
            .decode(&codes_for_codec)
            .map_err(|e| anyhow::anyhow!("codec decode failed: {e}"))?;

        let waveform = Self::normalize_peak(&waveform, 0.95)
            .map_err(|e| anyhow::anyhow!("peak normalization failed: {e}"))?;

        Ok((waveform, self.sample_rate()))
    }

    /// Generate speech and write to a WAV file.
    ///
    /// Returns the output file path on success.
    ///
    /// # Errors
    ///
    /// Returns an error if generation or WAV writing fails.
    pub fn generate_speech_to_file(
        &mut self,
        text: &str,
        voice: Option<&str>,
        max_new_tokens: usize,
        output_path: &str,
    ) -> Result<String> {
        let opts = SpeechOptions {
            max_new_tokens,
            temperature: 0.0,
            top_p: None,
            repetition_penalty: 0.0,
        };
        let (audio, sr) = self.generate_speech(text, "auto", voice, &opts)?;
        Self::save_wav(&audio, output_path, sr)
    }

    /// Write a waveform tensor to a 16-bit PCM WAV file.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor cannot be converted or the file cannot be
    /// written.
    pub fn save_wav(audio: &Tensor, path: &str, sample_rate: u32) -> Result<String> {
        let audio_f32 = audio.to_dtype(DType::F32)?.flatten_all()?;
        let samples: Vec<f32> = audio_f32.to_vec1()?;

        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(path, spec)?;
        for &s in &samples {
            #[allow(clippy::cast_possible_truncation)] // value is clamped to i16 range
            let s16 = (s * 32767.0).clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
            writer.write_sample(s16)?;
        }
        writer.finalize()?;
        Ok(path.to_string())
    }

    /// List the names of available voices.
    #[must_use]
    pub fn available_voices(&self) -> Vec<&str> {
        self.voices.keys().map(String::as_str).collect()
    }

    /// Output sample rate in Hz (always 24 000 for Voxtral).
    ///
    /// # Panics
    ///
    /// Panics if the configured sample rate exceeds `u32::MAX`, which cannot
    /// happen with any real audio config.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        u32::try_from(self.config.multimodal.audio_tokenizer_args.sampling_rate)
            .expect("sample rate fits u32")
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Synthetic helpers ─────────────────────────────────────────────────

    fn minimal_params_json() -> &'static str {
        // Voice IDs are synthetic; they don't match the real checkpoint order.
        r#"{
            "dim": 3072, "n_layers": 26, "head_dim": 128, "hidden_dim": 9216,
            "n_heads": 32, "n_kv_heads": 8, "rope_theta": 1000000.0,
            "norm_eps": 1e-05, "vocab_size": 131072, "max_seq_len": 65536,
            "tied_embeddings": true,
            "multimodal": {
                "bos_token_id": 1,
                "audio_model_args": {
                    "semantic_codebook_size": 8192, "acoustic_codebook_size": 21,
                    "n_acoustic_codebook": 36, "audio_token_id": 24,
                    "begin_audio_token_id": 25,
                    "acoustic_transformer_args": {
                        "dim": 3072, "n_layers": 3, "head_dim": 128, "hidden_dim": 9216,
                        "n_heads": 32, "n_kv_heads": 8, "rope_theta": 10000.0
                    }
                },
                "audio_tokenizer_args": {
                    "sampling_rate": 24000, "pretransform_patch_size": 240,
                    "semantic_codebook_size": 8192, "semantic_dim": 256,
                    "acoustic_codebook_size": 21, "acoustic_dim": 36,
                    "dim": 1024, "hidden_dim": 4096, "head_dim": 128,
                    "n_heads": 8, "n_kv_heads": 8, "qk_norm_eps": 1e-06,
                    "norm_eps": 0.01, "layer_scale_init": 0.01,
                    "decoder_transformer_lengths_str": "2,2,2,2",
                    "decoder_convs_kernels_str": "3,4,4,4",
                    "decoder_convs_strides_str": "1,2,2,2",
                    "voice": {
                        "neutral_female": 0, "casual_male": 1, "ar_male": 2,
                        "de_male": 3, "de_female": 4, "es_male": 5,
                        "es_female": 6, "fr_male": 7, "fr_female": 8,
                        "it_male": 9, "it_female": 10, "nl_male": 11,
                        "nl_female": 12, "pt_male": 13, "pt_female": 14,
                        "hi_male": 15, "hi_female": 16, "cheerful_female": 17,
                        "casual_female": 18, "neutral_male": 19
                    }
                }
            }
        }"#
    }

    /// Build a minimal in-memory PyTorch `.pt` (ZIP) file containing a BF16
    /// tensor of shape `[n_frames, 3072]` at the path `voice_embed/data/0`.
    fn make_pt_bytes(n_frames: usize) -> Vec<u8> {
        use std::io::Write as _;
        let raw: Vec<u8> = vec![0u8; n_frames * EMBED_DIM * 2]; // all-zero BF16
        let mut buf = Vec::new();
        {
            let mut zip = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
            let opts = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("voice_embed/data/0", opts).unwrap();
            zip.write_all(&raw).unwrap();
            zip.finish().unwrap();
        }
        buf
    }

    // ── Unit tests (no external files) ───────────────────────────────────

    #[test]
    fn test_config_parse() {
        let cfg: VoxtralConfig =
            serde_json::from_str(minimal_params_json()).expect("config should parse");
        assert_eq!(cfg.dim, 3072);
        assert_eq!(cfg.n_layers, 26);
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.n_kv_heads, 8);
        assert_eq!(cfg.vocab_size, 131_072);
        assert_eq!(cfg.max_seq_len, 65_536);
        assert!(cfg.tied_embeddings);
        assert_eq!(cfg.multimodal.audio_model_args.semantic_codebook_size, 8192);
        assert_eq!(cfg.multimodal.audio_model_args.n_acoustic_codebook, 36);
        assert_eq!(cfg.multimodal.audio_tokenizer_args.voice.len(), 20);
        assert_eq!(
            cfg.multimodal
                .audio_tokenizer_args
                .decoder_transformer_lengths,
            vec![2, 2, 2, 2]
        );
        assert_eq!(
            cfg.multimodal.audio_tokenizer_args.decoder_convs_kernels,
            vec![3, 4, 4, 4]
        );
        assert_eq!(
            cfg.multimodal.audio_tokenizer_args.decoder_convs_strides,
            vec![1, 2, 2, 2]
        );
    }

    #[test]
    fn test_voice_embedding_load() {
        let n_frames = 4usize;
        let pt_bytes = make_pt_bytes(n_frames);

        // Write to a temp file so load_voice_embedding can open it.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        use std::io::Write as _;
        tmp.as_file().write_all(&pt_bytes).unwrap();

        let emb =
            load_voice_embedding(tmp.path(), &Device::Cpu).expect("voice embedding should load");
        assert_eq!(emb.dims(), &[n_frames, EMBED_DIM]);
        assert_eq!(emb.dtype(), DType::BF16);
    }

    #[test]
    fn test_prompt_segments() {
        let text_ids = [1000u32, 2000, 3000];
        let segs = build_prompt_segments(&text_ids);

        assert_eq!(segs[0], PromptSegment::Token(BOS));
        assert_eq!(segs[1], PromptSegment::Token(BEGIN_AUDIO));
        assert_eq!(segs[2], PromptSegment::VoiceEmbeddings);
        assert_eq!(segs[3], PromptSegment::Token(INST_END));
        assert_eq!(segs[4], PromptSegment::Token(1000));
        assert_eq!(segs[5], PromptSegment::Token(2000));
        assert_eq!(segs[6], PromptSegment::Token(3000));
        let n = segs.len();
        assert_eq!(segs[n - 2], PromptSegment::Token(INST));
        assert_eq!(segs[n - 1], PromptSegment::Token(BEGIN_AUDIO));
    }

    #[test]
    fn test_prompt_segments_empty() {
        let segs = build_prompt_segments(&[]);
        assert_eq!(segs.len(), 6);
        assert_eq!(segs[0], PromptSegment::Token(BOS));
        assert_eq!(segs[1], PromptSegment::Token(BEGIN_AUDIO));
        assert_eq!(segs[2], PromptSegment::VoiceEmbeddings);
        assert_eq!(segs[3], PromptSegment::Token(INST_END));
        assert_eq!(segs[4], PromptSegment::Token(INST));
        assert_eq!(segs[5], PromptSegment::Token(BEGIN_AUDIO));
    }

    #[test]
    fn test_load_tokenizer_missing_file() {
        let result = load_tokenizer(std::path::Path::new("/nonexistent/voxtral/path"));
        assert!(result.is_err());
    }

    // ── Integration tests (require local checkpoint, skipped in CI) ───────

    fn checkpoint_path() -> Option<std::path::PathBuf> {
        let p = if let Ok(dir) = std::env::var("VOXTRAL_CHECKPOINT_DIR") {
            std::path::PathBuf::from(dir)
        } else {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()?
                .join("checkpoints/Voxtral-4B-TTS-2603")
        };
        p.is_dir().then_some(p)
    }

    #[test]
    #[ignore = "requires checkpoint dir (set VOXTRAL_CHECKPOINT_DIR or place at checkpoints/Voxtral-4B-TTS-2603)"]
    fn test_config_parse_real() {
        let dir = checkpoint_path().expect("checkpoint not found");
        let cfg = VoxtralConfig::from_model_dir(&dir).expect("config should parse");
        assert_eq!(cfg.dim, 3072);
        assert_eq!(cfg.multimodal.audio_tokenizer_args.voice.len(), 20);
    }

    #[test]
    #[ignore = "requires checkpoint dir (set VOXTRAL_CHECKPOINT_DIR or place at checkpoints/Voxtral-4B-TTS-2603)"]
    fn test_tekken_encode_real() {
        let dir = checkpoint_path().expect("checkpoint not found");
        let tok = load_tokenizer(&dir).expect("tokenizer should load");
        let ids = tok
            .encode("Hello, world!", false, false)
            .expect("encode should succeed");
        assert!(!ids.is_empty());
        let vocab = tok.vocab_size();
        for &id in &ids {
            assert!((id as usize) < vocab);
        }
    }

    #[test]
    #[ignore = "requires checkpoint dir (set VOXTRAL_CHECKPOINT_DIR or place at checkpoints/Voxtral-4B-TTS-2603)"]
    fn test_voice_embedding_load_real() {
        let dir = checkpoint_path().expect("checkpoint not found");
        let emb =
            load_voice_embedding(&dir.join("voice_embedding/neutral_female.pt"), &Device::Cpu)
                .expect("voice embedding should load");
        assert_eq!(emb.dims()[1], EMBED_DIM);
        assert_eq!(emb.dtype(), DType::BF16);
    }

    #[test]
    fn test_normalize_peak_scales_to_target() {
        let data: Vec<f32> = vec![0.5, -0.3, 0.1];
        let t = Tensor::new(data.as_slice(), &Device::Cpu).unwrap();
        let out = Model::normalize_peak(&t, 0.95).unwrap();
        let out_data = out.to_vec1::<f32>().unwrap();
        let peak: f32 = out_data.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        assert!((peak - 0.95).abs() < 1e-6, "peak={peak}");
    }

    #[test]
    fn test_normalize_peak_silent_unchanged() {
        let data: Vec<f32> = vec![0.0, 0.0, 0.0];
        let t = Tensor::new(data.as_slice(), &Device::Cpu).unwrap();
        let out = Model::normalize_peak(&t, 0.95).unwrap();
        let out_data = out.to_vec1::<f32>().unwrap();
        assert!(out_data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_model_new_missing_dir() {
        let result = Model::new("/nonexistent/voxtral/path", &Device::Cpu, &DType::F32);
        assert!(result.is_err());
    }

    #[test]
    #[ignore = "requires checkpoint dir (set VOXTRAL_CHECKPOINT_DIR or place at checkpoints/Voxtral-4B-TTS-2603)"]
    fn test_model_load_real() {
        let dir = checkpoint_path().expect("checkpoint not found");
        let model = Model::new(dir.to_str().unwrap(), &Device::Cpu, &DType::BF16)
            .expect("model should load");
        assert!(!model.available_voices().is_empty());
        assert_eq!(model.sample_rate(), 24000);
    }
}
