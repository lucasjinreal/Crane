//! High-level Qwen3-TTS model wrapper.
//!
//! Handles model loading, text tokenization, speech generation, and
//! code → waveform decoding via native Candle speech tokenizer decoder.

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hound::{SampleFormat, WavSpec, WavWriter};
use tokenizers::Tokenizer;

use super::modeling::{Qwen3TTSConfig, Qwen3TTSModel};
use super::speech_tokenizer_v2::NativeSpeechTokenizerDecoder;
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

    /// Convenience: decode and write a WAV file.
    pub fn decode_to_wav(&self, codes: &Tensor, filename: &str) -> Result<String> {
        let audio = self.decode(codes)?;
        Self::save_wav(&audio, filename, self.sample_rate)
    }

    pub fn save_wav(audio_values: &Tensor, filename: &str, sample_rate: u32) -> Result<String> {
        let audio = audio_values.to_dtype(DType::F32)?.flatten_all()?;
        let scaled = audio.affine(32767.0, 0.0)?.clamp(-32768.0, 32767.0)?.round()?;
        let audio_i64 = scaled.to_dtype(DType::I64)?;
        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(filename, spec)?;
        for sample in audio_i64.to_vec1::<i64>()? {
            writer.write_sample(sample.clamp(i16::MIN as i64, i16::MAX as i64) as i16)?;
        }
        writer.finalize()?;
        Ok(filename.to_string())
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

        // Qwen3-TTS first-codebook tokens may come from the tail range of the
        // talker vocab; map them back to local codebook ids.
        let vocab_size = self.config.talker_config.vocab_size as u32;
        let tail_base = vocab_size.saturating_sub(codebook_size);

        let mapped = if code >= tail_base && vocab_size >= codebook_size {
            code - tail_base
        } else {
            code
        };

        mapped.min(codebook_size.saturating_sub(1))
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
    /// Wraps with Qwen chat-style tags:
    /// `<|im_start|>system\nYou are Qwen...<|im_end|>\n<|im_start|>user\n<tts>text</tts><|im_end|>\n<|im_start|>assistant\n`
    pub fn prepare_tts_input(
        &self,
        text: &str,
        system_prompt: Option<&str>,
    ) -> Result<Vec<u32>> {
        let system = system_prompt.unwrap_or(
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group. \
             You can perceive speech, text, and other multimodal information, and you can \
             output speech and text information.",
        );
        let chat_text = format!(
            "<|im_start|>system\n{system}<|im_end|>\n\
             <|im_start|>user\n<tts>{text}</tts><|im_end|>\n\
             <|im_start|>assistant\n"
        );
        let encoding = self.tokenizer.encode(chat_text.as_str(), false).map_err(E::msg)?;
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
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        repetition_penalty: f32,
        system_prompt: Option<&str>,
    ) -> Result<(Tensor, u32)> {
        let input_ids = self.prepare_tts_input(text, system_prompt)?;

        let codes = self.inner.generate_speech_codes(
            &input_ids,
            language,
            speaker,
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
        )?;

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

    /// Generate speech and write directly to a WAV file.
    pub fn generate_speech_to_file(
        &mut self,
        text: &str,
        language: &str,
        speaker: Option<&str>,
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        repetition_penalty: f32,
        output_path: &str,
        system_prompt: Option<&str>,
    ) -> Result<String> {
        let (audio, sr) = self.generate_speech(
            text,
            language,
            speaker,
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
            system_prompt,
        )?;
        Self::save_wav(&audio, output_path, sr)
    }

    /// Generate only the codec codes (no waveform decode).
    /// Useful when you have an external vocoder.
    pub fn generate_codes(
        &mut self,
        text: &str,
        language: &str,
        speaker: Option<&str>,
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        repetition_penalty: f32,
        system_prompt: Option<&str>,
    ) -> Result<Vec<Vec<u32>>> {
        let input_ids = self.prepare_tts_input(text, system_prompt)?;
        let codes = self.inner.generate_speech_codes(
            &input_ids,
            language,
            speaker,
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
        )?;
        Ok(codes)
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

    fn save_wav(audio_values: &Tensor, filename: &str, sample_rate: u32) -> Result<String> {
        let audio = audio_values.to_dtype(DType::F32)?.flatten_all()?;
        let scaled = audio.affine(32767.0, 0.0)?.clamp(-32768.0, 32767.0)?.round()?;
        let audio_i64 = scaled.to_dtype(DType::I64)?;
        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut writer = WavWriter::create(filename, spec)?;
        for sample in audio_i64.to_vec1::<i64>()? {
            writer.write_sample(sample.clamp(i16::MIN as i64, i16::MAX as i64) as i16)?;
        }
        writer.finalize()?;
        Ok(filename.to_string())
    }
}
