//! High-level Qwen3-TTS model wrapper.
//!
//! Handles model loading, text tokenization, speech generation, and
//! code → waveform decoding via native Candle speech tokenizer decoder.

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use super::modeling::{Qwen3TTSConfig, Qwen3TTSModel, StreamingState};
use super::speech_tokenizer_v2::NativeSpeechTokenizerDecoder;
use crate::generation::SpeechOptions;
use crate::models::modules::mel::{compute_mel_spectrogram, MelSpectrogramConfig};
use crate::utils::utils;

// ── Speech Tokenizer decoders (codes → waveform) ───────────────────────

/// ONNX-based speech tokenizer decoder. Converts 16-codebook codec tokens
/// into raw audio waveform.
#[cfg(feature = "onnx")]
pub struct SpeechTokenizerDecoder {
    model: candle_onnx::onnx::ModelProto,
    pub sample_rate: u32,
}

enum SpeechDecoderBackend {
    Native(NativeSpeechTokenizerDecoder),
    #[cfg(feature = "onnx")]
    Onnx(SpeechTokenizerDecoder),
}

impl SpeechDecoderBackend {
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        match self {
            Self::Native(m) => m.chunked_decode(codes, 300, 25),
            #[cfg(feature = "onnx")]
            Self::Onnx(m) => m.decode(codes),
        }
    }

    fn sample_rate(&self) -> u32 {
        match self {
            Self::Native(m) => m.sample_rate(),
            #[cfg(feature = "onnx")]
            Self::Onnx(m) => m.sample_rate,
        }
    }

    fn decode_chunk(&self, codes: &Tensor, context_frames: usize) -> Result<Tensor> {
        match self {
            Self::Native(m) => m.decode_chunk(codes, context_frames),
            #[cfg(feature = "onnx")]
            Self::Onnx(m) => {
                let wav = m.decode(codes)?;
                if context_frames == 0 {
                    return Ok(wav);
                }
                // total_upsample=1920 for Qwen3-TTS speech tokenizer
                let trim = context_frames * 1920usize;
                let tw = wav.dim(candle_core::D::Minus1)?;
                Ok(wav.narrow(candle_core::D::Minus1, trim, tw.saturating_sub(trim))?)
            }
        }
    }
}

#[cfg(feature = "onnx")]
impl SpeechTokenizerDecoder {
    /// Load from a pre-exported ONNX file.
    pub fn new(onnx_path: &str, sample_rate: Option<u32>) -> Result<Self> {
        if !std::path::Path::new(onnx_path).exists() {
            anyhow::bail!(
                "Speech tokenizer ONNX not found at {}. \
                 Export it first: python scripts/export_qwen_tts_tokenizer_onnx.py <model_dir> {}",
                onnx_path,
                onnx_path,
            );
        }
        let model = candle_onnx::read_file(onnx_path)?;
        Ok(Self {
            model,
            sample_rate: sample_rate.unwrap_or(24000),
        })
    }

    /// Decode `[batch, num_quantizers, seq_len]` codes → `[batch, 1, samples]`.
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let inputs =
            std::collections::HashMap::from_iter([("codes".to_string(), codes.clone())]);
        let out = candle_onnx::simple_eval(&self.model, inputs)?;
        let out_names = &self.model.graph.as_ref().unwrap().output;
        let audio = out.get(&out_names[0].name).unwrap().clone();
        Ok(audio)
    }
}

// ── Qwen3-TTS Model ────────────────────────────────────────────────────

pub struct Model {
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub dtype: DType,
    pub config: Qwen3TTSConfig,
    inner: Qwen3TTSModel,
    speech_decoder: Option<SpeechDecoderBackend>,
}

impl Model {
    fn tts_debug_enabled() -> bool {
        std::env::var("CRANE_TTS_DEBUG")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
    }

    fn normalize_codec_id(&self, code: u32) -> u32 {
        let codebook_size = self.config.talker_config.code_predictor_config.vocab_size as u32;
        if codebook_size == 0 {
            return code;
        }
        // Codes from the talker are already in [0, codebook_size) range
        // (tokens >= codebook_size are suppressed during generation).
        // Just clamp to be safe.
        code.min(codebook_size.saturating_sub(1))
    }

    fn validate_tts_generation_mode(&self) -> Result<()> {
        match self.config.tts_model_type.as_deref().unwrap_or("base") {
            "custom_voice" => Ok(()),
            "base" => anyhow::bail!(
                "Qwen3-TTS base model needs voice-clone prompt/reference audio. \
                 The current Rust direct-TTS API does not implement base-model prompt flow yet. \
                 Use Qwen3-TTS-12Hz-0.6B-CustomVoice for direct text->speech, or add voice-clone support first."
            ),
            other => anyhow::bail!(
                "Unsupported tts_model_type '{other}' for direct text->speech in current Rust API"
            ),
        }
    }

    /// Load from a HuggingFace-style directory.
    ///
    /// Expects:
    ///   - `config.json` (Qwen3TTSConfig)
    ///   - `tokenizer.json` (or `vocab.json` + `merges.txt`)
    ///   - `model.safetensors` / `model-*.safetensors` (talker weights)
    ///   - `speech_tokenizer/config.json` + `speech_tokenizer/model.safetensors` (preferred)
    ///   - `speech_tokenizer/speech_tokenizer_decoder.onnx` (optional fallback)
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let model_dir = std::path::Path::new(model_path);

        // Config
        let config_data = std::fs::read(model_dir.join("config.json"))?;
        let config: Qwen3TTSConfig = serde_json::from_slice(&config_data)?;

        // Tokenizer (tokenizer.json preferred, fallback to vocab.json + merges.txt)
        let tokenizer = crate::utils::tokenizer_utils::load_tokenizer_from_model_dir(model_dir)?;

        // Safetensors
        let filenames = utils::get_safetensors_files(model_path)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, *dtype, device) }?;
        let inner = Qwen3TTSModel::new(&config, vb)?;

        // Speech tokenizer decoder (native Candle preferred, ONNX fallback)
        let speech_dir = model_dir.join("speech_tokenizer");
        let speech_decoder = if speech_dir.exists() {
            match NativeSpeechTokenizerDecoder::new(speech_dir.to_str().unwrap(), device, *dtype) {
                Ok(native) => Some(SpeechDecoderBackend::Native(native)),
                Err(native_err) => {
                    #[cfg(feature = "onnx")]
                    {
                        let onnx_path = speech_dir.join("speech_tokenizer_decoder.onnx");
                        if onnx_path.exists() {
                            eprintln!(
                                "Warning: native speech tokenizer load failed: {native_err}. Falling back to ONNX at {}",
                                onnx_path.display()
                            );
                            Some(SpeechDecoderBackend::Onnx(SpeechTokenizerDecoder::new(
                                onnx_path.to_str().unwrap(),
                                Some(24000),
                            )?))
                        } else {
                            eprintln!(
                                "Warning: speech tokenizer native load failed and ONNX missing: {native_err}. \
                                 Code-to-waveform decoding will not be available."
                            );
                            None
                        }
                    }
                    #[cfg(not(feature = "onnx"))]
                    {
                        eprintln!(
                            "Warning: native speech tokenizer load failed: {native_err}. \
                             Code-to-waveform decoding will not be available."
                        );
                        None
                    }
                }
            }
        } else {
            eprintln!(
                "Warning: speech_tokenizer directory not found at {}. \
                 Code-to-waveform decoding will not be available.",
                speech_dir.display()
            );
            None
        };

        Ok(Self {
            tokenizer,
            device: device.clone(),
            dtype: *dtype,
            config,
            inner,
            speech_decoder,
        })
    }

    pub fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }

    /// Tokenize text input for TTS.
    ///
    /// Returns raw text tokens (no ChatML wrapping).
    /// The role prefix is added by the talker prefill construction.
    pub fn prepare_tts_input(
        &self,
        text: &str,
    ) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, false).map_err(E::msg)?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Generate speech: text → codec codes → waveform tensor.
    ///
    /// Returns `(audio_tensor, sample_rate)`.  
    /// `audio_tensor` shape: `[1, 1, samples]` (f32).
    pub fn generate_speech(
        &mut self,
        text: &str,
        language: &str,
        speaker: Option<&str>,
        opts: &SpeechOptions,
    ) -> Result<(Tensor, u32)> {
        self.validate_tts_generation_mode()?;
        let input_ids = self.prepare_tts_input(text)?;

        let codes = self.inner.generate_speech_codes(&input_ids, language, speaker, opts)?;

        if codes.is_empty() {
            anyhow::bail!("No speech codes generated");
        }

        let speech_decoder = self
            .speech_decoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Speech tokenizer decoder not loaded; cannot decode to audio"))?;

        // Convert codes: Vec<Vec<u32>> of shape [timesteps, num_code_groups]
        // → Tensor [1, num_code_groups, timesteps]
        let num_steps = codes.len();
        let num_groups = codes[0].len();
        let flat: Vec<i64> = codes
            .iter()
            .flat_map(|frame| frame.iter().map(|&c| self.normalize_codec_id(c) as i64))
            .collect();

        if Self::tts_debug_enabled() {
            let (min_id, max_id) = flat
                .iter()
                .fold((i64::MAX, i64::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));
            eprintln!(
                "[CRANE_TTS_DEBUG] decode input codes shape=[1,{num_groups},{num_steps}] min_id={min_id} max_id={max_id}"
            );
        }

        let codes_tensor = Tensor::new(flat.as_slice(), &self.device)?
            .reshape((num_steps, num_groups))?
            .transpose(0, 1)?
            .unsqueeze(0)?;

        let audio = speech_decoder.decode(&codes_tensor)?;
        Ok((audio, speech_decoder.sample_rate()))
    }

    /// Generate speech as an incremental stream of f32 PCM chunks.
    ///
    /// Returns a [`SpeechStream`] that yields audio chunks as codec frames are
    /// generated. The first chunk arrives after [`STREAM_FIRST_CHUNK_SIZE`] frames
    /// (~0.4 s); subsequent chunks every [`STREAM_CHUNK_SIZE`] frames (~2 s).
    pub fn generate_speech_streaming(
        &mut self,
        text: &str,
        language: &str,
        speaker: Option<&str>,
        opts: &SpeechOptions,
    ) -> Result<SpeechStream<'_>> {
        self.validate_tts_generation_mode()?;
        let input_ids = self.prepare_tts_input(text)?;
        let state = self.inner.prepare_streaming(&input_ids, language, speaker, opts)?;
        Ok(SpeechStream {
            model: self,
            state,
            max_new_tokens: opts.max_new_tokens,
            chunk_size: STREAM_CHUNK_SIZE,
            first_chunk_size: STREAM_FIRST_CHUNK_SIZE,
            left_context: STREAM_LEFT_CONTEXT,
            all_codes: Vec::new(),
            emitted_up_to: 0,
            done: false,
        })
    }

    /// Generate only the codec codes (no waveform decode).
    /// Useful when you have an external vocoder.
    pub fn generate_codes(
        &mut self,
        text: &str,
        language: &str,
        speaker: Option<&str>,
        opts: &SpeechOptions,
    ) -> Result<Vec<Vec<u32>>> {
        self.validate_tts_generation_mode()?;
        let input_ids = self.prepare_tts_input(text)?;
        Ok(self.inner.generate_speech_codes(&input_ids, language, speaker, opts)?)
    }

    /// Convert pre-generated codes to raw audio bytes (PCM 16-bit LE).
    pub fn codes_to_pcm(&self, codes: &[Vec<u32>]) -> Result<Vec<u8>> {
        let speech_decoder = self
            .speech_decoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Speech tokenizer decoder not loaded"))?;

        let num_steps = codes.len();
        let num_groups = codes[0].len();
        let flat: Vec<i64> = codes
            .iter()
            .flat_map(|frame| frame.iter().map(|&c| self.normalize_codec_id(c) as i64))
            .collect();
        let codes_tensor = Tensor::new(flat.as_slice(), &self.device)?
            .reshape((num_steps, num_groups))?
            .transpose(0, 1)?
            .unsqueeze(0)?;

        let audio = speech_decoder.decode(&codes_tensor)?;
        let audio = audio.to_dtype(DType::F32)?.flatten_all()?;

        // Scale to i16 PCM
        let scaled = audio.affine(32767.0, 0.0)?.clamp(-32768.0, 32767.0)?.round()?;
        let samples = scaled.to_dtype(DType::I64)?.to_vec1::<i64>()?;

        let mut pcm_bytes = Vec::with_capacity(samples.len() * 2);
        for s in samples {
            let s16 = s.clamp(i16::MIN as i64, i16::MAX as i64) as i16;
            pcm_bytes.extend_from_slice(&s16.to_le_bytes());
        }
        Ok(pcm_bytes)
    }

    /// Get the sample rate of the speech tokenizer decoder.
    pub fn sample_rate(&self) -> u32 {
        self.speech_decoder
            .as_ref()
            .map(|d| d.sample_rate())
            .unwrap_or(24000)
    }

    /// Get the sample rate expected by the speaker encoder, for use with
    /// [`Model::generate_voice_clone`]'s `ref_samples` argument.
    pub fn speaker_encoder_sample_rate(&self) -> u32 {
        self.config.speaker_encoder_config.sample_rate
    }

    /// Voice-clone: synthesize `text` in the voice of the speaker from `ref_samples`.
    ///
    /// `ref_samples`: reference speaker audio, mono f32 in `[-1, 1]`, already
    /// resampled to the speaker encoder's sample rate
    /// ([`Model::speaker_encoder_sample_rate`]).
    /// `ref_text`: transcript of the reference audio (required for ICL mode).
    /// `language`: target language ("japanese", "chinese", "english", "auto", …).
    ///
    /// Returns `(audio_tensor, sample_rate)`.
    pub fn generate_voice_clone(
        &mut self,
        text: &str,
        language: &str,
        ref_samples_spk: &[f32],
        ref_text: &str,
        opts: &SpeechOptions,
    ) -> Result<(Tensor, u32)> {
        // 1. Validate model type
        match self.config.tts_model_type.as_deref().unwrap_or("base") {
            "base" => {} // OK
            other => anyhow::bail!(
                "generate_voice_clone requires tts_model_type=base, got '{other}'. \
                 Use Qwen3-TTS-12Hz-0.6B-Base."
            ),
        }

        if ref_samples_spk.is_empty() {
            anyhow::bail!("Reference audio is empty");
        }

        // 3. Extract speaker embedding via ECAPA-TDNN
        //    Speaker encoder runs in F32 for precision (matching vendor).
        //    The embedding is later cast to model dtype in build_voice_clone_prefill.
        let spk_embed = {
            let enc = self.inner.speaker_encoder.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Speaker encoder not loaded (base model required)"))?;
            // Speaker encoder mel constants: n_fft=1024, num_mels=128, sr=24000,
            // hop=256, win=1024, fmin=0, fmax=12000 — Hann window, reflect-padded,
            // log-compressed.
            const SPEAKER_MEL_CONFIG: MelSpectrogramConfig = MelSpectrogramConfig {
                n_fft: 1024,
                hop_length: 256,
                win_length: 1024,
                sample_rate: 24000,
                n_mels: 128,
                fmin: 0.0,
                fmax: 12000.0,
            };
            let mels = compute_mel_spectrogram(&SPEAKER_MEL_CONFIG, ref_samples_spk, &self.device, DType::F32)?
                .unsqueeze(0)?; // [1, n_mels, T_frames] — matches speaker encoder input [B, n_mels, T]
            let embed = enc.forward(&mels)?.squeeze(0)?; // [enc_dim], F32
            if Self::tts_debug_enabled() {
                let norm: f32 = embed.sqr()?.sum_all()?.sqrt()?.to_scalar()?;
                eprintln!(
                    "[CRANE_TTS_DEBUG] speaker_embed: dtype={:?}, shape={:?}, norm={:.4}, \
                     mel_shape={:?}, ref_samples={}",
                    embed.dtype(), embed.dims(), norm, mels.dims(), ref_samples_spk.len(),
                );
            }
            embed
        };

        // 4. Encode reference audio to codec codes using speech tokenizer encoder
        //    Load reference audio at codec SR (24kHz) — same SR, reuse ref_samples_spk
        let ref_codes = {
            let speech_dec = self.speech_decoder.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Speech tokenizer not loaded; cannot encode reference audio"))?;
            match speech_dec {
                SpeechDecoderBackend::Native(native) => {
                    // Encode: samples [1, 1, N] → codes [1, T, n_q] → squeeze → [T, n_q]
                    let ref_tensor = Tensor::new(ref_samples_spk, &self.device)?
                        .unsqueeze(0)?.unsqueeze(0)?; // [1, 1, N]
                    let codes = native.encode(&ref_tensor)?.squeeze(0)?; // [T, n_q]
                    if Self::tts_debug_enabled() {
                        eprintln!(
                            "[CRANE_TTS_DEBUG] ref_codes: shape={:?}, dtype={:?}",
                            codes.dims(), codes.dtype(),
                        );
                        if let Ok(v) = codes.to_dtype(DType::U32).and_then(|t| t.to_vec2::<u32>()) {
                            if !v.is_empty() {
                                let head: Vec<Vec<u32>> = v.iter().take(2).cloned().collect();
                                let tail: Vec<Vec<u32>> = v
                                    .iter()
                                    .rev()
                                    .take(2)
                                    .cloned()
                                    .collect::<Vec<_>>()
                                    .into_iter()
                                    .rev()
                                    .collect();
                                eprintln!("[CRANE_TTS_DEBUG] ref_codes first2={head:?}");
                                eprintln!("[CRANE_TTS_DEBUG] ref_codes last2={tail:?}");
                            }
                        }
                    }
                    codes
                }
                #[cfg(feature = "onnx")]
                SpeechDecoderBackend::Onnx(_) => {
                    anyhow::bail!("Voice-clone requires native speech tokenizer (ONNX encoder not supported)")
                }
            }
        };

        // 5. Tokenize target text and reference text
        let input_ids = self.prepare_tts_input(text)?;
        let ref_encoding = self.tokenizer.encode(ref_text, false).map_err(E::msg)?;
        let ref_token_ids = ref_encoding.get_ids().to_vec();

        // 6. Generate codec codes (voice-clone mode)
        let (new_codes, _ref_code_len) = self.inner.generate_voice_clone_codes(
            &input_ids,
            &ref_token_ids,
            &ref_codes,
            &spk_embed,
            language,
            opts,
        )?;

        if new_codes.is_empty() {
            anyhow::bail!("No speech codes generated");
        }

        let speech_decoder = self.speech_decoder.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Speech tokenizer decoder not loaded"))?;

        // 7. Prepend ref_codes to generated codes, decode, then trim ref portion.
        //    This matches the official Python implementation: ref_codes provide
        //    decoder continuity context, and the proportional trim removes
        //    the reference audio from the output, keeping only target speech.
        let num_groups = new_codes[0].len();
        let ref_t = ref_codes.dim(0)?;

        if Self::tts_debug_enabled() {
            eprintln!(
                "[CRANE_TTS_DEBUG] voice_clone decode: generated={}, prepend_ref={}",
                new_codes.len(), ref_t,
            );
        }

        // Build combined codes: [ref_codes; new_codes] → [total_T, num_groups]
        let ref_flat: Vec<i64> = ref_codes.to_dtype(DType::I64)?.to_vec2::<i64>()?
            .into_iter().flatten().collect();
        let new_flat: Vec<i64> = new_codes.iter()
            .flat_map(|frame| frame.iter().map(|&c| self.normalize_codec_id(c) as i64))
            .collect();
        let total_t = ref_t + new_codes.len();
        let mut combined = ref_flat;
        combined.extend(new_flat);

        let codes_tensor = Tensor::new(combined.as_slice(), &self.device)?
            .reshape((total_t, num_groups))?
            .transpose(0, 1)?
            .unsqueeze(0)?; // [1, num_groups, total_T]

        let audio_full = speech_decoder.decode(&codes_tensor)?;

        // Trim the reference portion from decoded audio.
        // Use exact sample count from decoding ref_codes alone; this is more
        // robust than proportional trimming when the decoder introduces a
        // fixed leading latency (which otherwise leaves silence/ref leakage).
        let ref_only_flat: Vec<i64> = ref_codes
            .to_dtype(DType::I64)?
            .to_vec2::<i64>()?
            .into_iter()
            .flatten()
            .collect();
        let ref_only_tensor = Tensor::new(ref_only_flat.as_slice(), &self.device)?
            .reshape((ref_t, num_groups))?
            .transpose(0, 1)?
            .unsqueeze(0)?; // [1, num_groups, ref_T]
        let ref_audio = speech_decoder.decode(&ref_only_tensor)?;

        let total_samples = audio_full.dim(2)?;
        let ref_samples = ref_audio.dim(2)?;
        let mut cut = ref_samples.min(total_samples.saturating_sub(1));
        if cut == 0 {
            let proportional = (ref_t as f64 / total_t.max(1) as f64 * total_samples as f64) as usize;
            cut = proportional.min(total_samples.saturating_sub(1));
        }
        let audio = audio_full.narrow(2, cut, total_samples - cut)?;

        Ok((audio, speech_decoder.sample_rate()))
    }
}

// ── Streaming speech generation ────────────────────────────────────────────

/// Number of codec frames in each chunk after the first.
/// 25 frames × 1920 samples/frame = 48 000 samples ≈ 2 s at 24 kHz.
const STREAM_CHUNK_SIZE: usize = 25;
/// Number of codec frames in the first emitted chunk.
/// 5 frames × 1920 samples/frame = 9 600 samples ≈ 0.4 s — fast first audio.
const STREAM_FIRST_CHUNK_SIZE: usize = 5;
/// Number of overlap frames prepended to each codec decode call.
/// Must exceed the codec's causal receptive field (~25 frames) for exact output.
const STREAM_LEFT_CONTEXT: usize = 25;

/// Incremental speech generator that yields PCM audio chunks on demand.
///
/// Returned by [`Model::generate_speech_streaming`]. Implements
/// `Iterator<Item = anyhow::Result<Tensor>>`, where each `Tensor` is a flat
/// `[n_samples]` f32 PCM slice at 24 kHz.
///
/// The first chunk arrives after [`STREAM_FIRST_CHUNK_SIZE`] codec frames
/// (~0.4 s). Subsequent chunks are emitted every [`STREAM_CHUNK_SIZE`] frames
/// (~2 s). [`STREAM_LEFT_CONTEXT`] frames of overlap between consecutive codec
/// calls ensure the waveform is numerically equivalent to non-streaming output.
pub struct SpeechStream<'a> {
    model: &'a mut Model,
    state: StreamingState,
    max_new_tokens: usize,
    chunk_size: usize,
    first_chunk_size: usize,
    left_context: usize,
    all_codes: Vec<Vec<u32>>,
    emitted_up_to: usize,
    done: bool,
}

impl SpeechStream<'_> {
    fn generate_one_frame(&mut self) -> Result<bool> {
        if self.state.step >= self.max_new_tokens {
            return Ok(true);
        }
        match self.model.inner.generate_one_frame(&mut self.state, &self.all_codes)? {
            None => Ok(true),
            Some(frame) => {
                self.all_codes.push(frame);
                Ok(false)
            }
        }
    }

    fn try_next(&mut self) -> Result<Option<Tensor>> {
        let target = if self.emitted_up_to == 0 {
            self.first_chunk_size
        } else {
            self.chunk_size
        };

        loop {
            let new_frames = self.all_codes.len() - self.emitted_up_to;
            if new_frames >= target {
                break;
            }
            let done = self.generate_one_frame()?;
            if done {
                self.done = true;
                break;
            }
        }

        let new_frames = self.all_codes.len() - self.emitted_up_to;
        if new_frames == 0 {
            return Ok(None);
        }

        let ctx = self.emitted_up_to.min(self.left_context);
        let chunk_start = self.emitted_up_to - ctx;
        let chunk_end = self.all_codes.len();
        let n_chunk_frames = chunk_end - chunk_start;
        let num_groups = self.all_codes[0].len();

        // Normalize and flatten codes into [1, num_groups, n_chunk_frames] i64 tensor.
        let codebook_size = self.model.config.talker_config.code_predictor_config.vocab_size as u32;
        let flat: Vec<i64> = self.all_codes[chunk_start..chunk_end]
            .iter()
            .flat_map(|frame| {
                frame.iter().map(move |&c| {
                    let normalized = if codebook_size == 0 { c } else { c.min(codebook_size - 1) };
                    normalized as i64
                })
            })
            .collect();
        let codes_tensor = Tensor::new(flat.as_slice(), &self.model.device)?
            .reshape((n_chunk_frames, num_groups))?
            .transpose(0, 1)?
            .unsqueeze(0)?;

        let speech_decoder = self.model.speech_decoder.as_ref()
            .ok_or_else(|| anyhow::anyhow!("speech decoder not loaded"))?;
        let audio = speech_decoder.decode_chunk(&codes_tensor, ctx)?.flatten_all()?;

        self.emitted_up_to = chunk_end;

        let keep_from = self.emitted_up_to.saturating_sub(self.left_context);
        if keep_from > 0 {
            self.all_codes.drain(..keep_from);
            self.emitted_up_to -= keep_from;
        }

        Ok(Some(audio))
    }
}

impl Iterator for SpeechStream<'_> {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done && self.emitted_up_to >= self.all_codes.len() {
            return None;
        }
        match self.try_next() {
            Ok(Some(chunk)) => Some(Ok(chunk)),
            Ok(None) => None,
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

