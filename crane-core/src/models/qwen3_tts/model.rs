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
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        repetition_penalty: f32,
    ) -> Result<(Tensor, u32)> {
        self.validate_tts_generation_mode()?;
        let input_ids = self.prepare_tts_input(text)?;

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
    ) -> Result<String> {
        let (audio, sr) = self.generate_speech(
            text,
            language,
            speaker,
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
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
    ) -> Result<Vec<Vec<u32>>> {
        self.validate_tts_generation_mode()?;
        let input_ids = self.prepare_tts_input(text)?;
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

    /// Compute a mel spectrogram matching the Python reference:
    ///   n_fft=1024, num_mels=128, sr=24000, hop=256, win=1024, fmin=0, fmax=12000
    ///   Hann window, reflect-padded, log-compressed.
    ///
    /// Input: f32 samples in [-1, 1], length N.
    /// Output: Tensor `[1, T_frames, num_mels]` on `device` in `dtype`.
    fn compute_mel_spectrogram(
        samples: &[f32],
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        const N_FFT: usize = 1024;
        const NUM_MELS: usize = 128;
        const SR: usize = 24000;
        const HOP: usize = 256;
        const WIN: usize = 1024;
        const FMIN: f64 = 0.0;
        const FMAX: f64 = 12000.0;

        // Hann window
        let hann: Vec<f32> = (0..WIN)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / WIN as f64).cos()) as f32)
            .collect();

        // Reflect pad: (n_fft - hop) / 2 = (1024 - 256) / 2 = 384
        let pad = (N_FFT - HOP) / 2;
        let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
        // left: mirror samples[1..pad+1] reversed
        for i in (1..=pad.min(samples.len().saturating_sub(1))).rev() {
            padded.push(samples[i]);
        }
        // fill if not enough
        while padded.len() < pad {
            padded.push(0.0);
        }
        padded.extend_from_slice(samples);
        // right: mirror last pad samples reversed
        let n = samples.len();
        for i in (n.saturating_sub(1 + pad)..n.saturating_sub(1)).rev() {
            padded.push(samples[i]);
        }
        while padded.len() < samples.len() + 2 * pad {
            padded.push(0.0);
        }

        // STFT: compute frames
        let n_frames = (padded.len().saturating_sub(WIN)) / HOP + 1;
        let n_bins = N_FFT / 2 + 1; // 513

        // Build mel filterbank (Slaney norm, librosa-compatible)
        let mel_basis = Self::build_mel_filterbank(SR, N_FFT, NUM_MELS, FMIN, FMAX);

        // Compute magnitude spectrogram frame by frame using rustfft (O(N log N))
        let mut mel_frames: Vec<f32> = Vec::with_capacity(n_frames * NUM_MELS);

        use rustfft::{FftPlanner, num_complex::Complex as FftComplex};
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        let mut spec_mag = vec![0f32; n_frames * n_bins];
        for frame_idx in 0..n_frames {
            let start = frame_idx * HOP;
            // Apply Hann window and build complex FFT input
            let mut buffer: Vec<FftComplex<f32>> = (0..N_FFT)
                .map(|i| {
                    let s = if i < WIN && start + i < padded.len() {
                        padded[start + i] * hann[i]
                    } else {
                        0.0
                    };
                    FftComplex::new(s, 0.0)
                })
                .collect();

            fft.process(&mut buffer);

            // Take magnitude of positive frequencies (n_fft/2 + 1 bins)
            for k in 0..n_bins {
                let c = &buffer[k];
                let mag = (c.re * c.re + c.im * c.im + 1e-9).sqrt();
                spec_mag[frame_idx * n_bins + k] = mag;
            }
        }

        // Apply mel filterbank and log compression
        for frame_idx in 0..n_frames {
            for m in 0..NUM_MELS {
                let mut val = 0f32;
                for k in 0..n_bins {
                    val += mel_basis[m * n_bins + k] * spec_mag[frame_idx * n_bins + k];
                }
                // log compression: log(max(val, 1e-5))
                mel_frames.push(val.max(1e-5).ln());
            }
        }

        // Shape: [1, NUM_MELS, T_frames] — matches speaker encoder input [B, n_mels, T]
        let t = Tensor::new(mel_frames.as_slice(), device)?
            .reshape((n_frames, NUM_MELS))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .to_dtype(dtype)?;
        Ok(t)
    }

    /// Build a Slaney-normalized mel filterbank matching librosa (Slaney mel scale).
    ///
    /// Slaney scale: linear below 1000 Hz, logarithmic above.
    /// This matches `librosa.filters.mel(norm="slaney", htk=False)`.
    fn build_mel_filterbank(sr: usize, n_fft: usize, n_mels: usize, fmin: f64, fmax: f64) -> Vec<f32> {
        // Slaney / O'Shaughnessy mel scale
        const F_SP: f64 = 200.0 / 3.0;    // ~66.667 Hz per mel below breakpoint
        const MIN_LOG_HZ: f64 = 1000.0;
        const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP; // 15.0
        // ln(6.4) / 27 ≈ 0.068751739
        const LOG_STEP: f64 = 0.068_751_74;

        fn hz_to_mel(f: f64) -> f64 {
            if f < MIN_LOG_HZ { f / F_SP } else { MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() / LOG_STEP }
        }
        fn mel_to_hz(m: f64) -> f64 {
            if m < MIN_LOG_MEL { m * F_SP } else { MIN_LOG_HZ * ((m - MIN_LOG_MEL) * LOG_STEP).exp() }
        }

        let n_bins = n_fft / 2 + 1;
        let fft_freqs: Vec<f64> = (0..n_bins).map(|k| k as f64 * sr as f64 / n_fft as f64).collect();

        let mel_min = hz_to_mel(fmin);
        let mel_max = hz_to_mel(fmax);
        let mel_points: Vec<f64> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64)
            .collect();
        let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

        let mut filters = vec![0f32; n_mels * n_bins];
        for m in 0..n_mels {
            let f_left = hz_points[m];
            let f_center = hz_points[m + 1];
            let f_right = hz_points[m + 2];
            // Slaney area-normalization: 2 / bandwidth
            let enorm = if f_right > f_left { 2.0 / (f_right - f_left) } else { 0.0 };
            for k in 0..n_bins {
                let f = fft_freqs[k];
                let val = if f >= f_left && f <= f_center && f_center > f_left {
                    (f - f_left) / (f_center - f_left)
                } else if f > f_center && f <= f_right && f_right > f_center {
                    (f_right - f) / (f_right - f_center)
                } else {
                    0.0
                };
                filters[m * n_bins + k] = (val * enorm) as f32;
            }
        }
        filters
    }

    /// Load a WAV file and return f32 samples normalized to [-1, 1].
    /// Resamples to `target_sr` if needed (simple linear interpolation).
    fn load_wav_f32(path: &str, target_sr: u32) -> Result<Vec<f32>> {
        let mut reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let raw_sr = spec.sample_rate;

        let samples_f32: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (SampleFormat::Float, 32) => reader.samples::<f32>().map(|s| s.map_err(|e| anyhow::anyhow!(e))).collect::<Result<Vec<_>>>()?,
            (SampleFormat::Int, 16) => reader.samples::<i16>().map(|s| s.map(|v| v as f32 / 32768.0).map_err(|e| anyhow::anyhow!(e))).collect::<Result<Vec<_>>>()?,
            (SampleFormat::Int, 32) => reader.samples::<i32>().map(|s| s.map(|v| v as f32 / 2147483648.0).map_err(|e| anyhow::anyhow!(e))).collect::<Result<Vec<_>>>()?,
            _ => anyhow::bail!("Unsupported WAV format: {:?} {}bit", spec.sample_format, spec.bits_per_sample),
        };

        // Mix down to mono if stereo
        let mono: Vec<f32> = if spec.channels == 1 {
            samples_f32
        } else {
            let ch = spec.channels as usize;
            samples_f32.chunks(ch).map(|c| c.iter().sum::<f32>() / ch as f32).collect()
        };

        // Resample if needed (linear interpolation)
        if raw_sr == target_sr {
            return Ok(mono);
        }
        let ratio = raw_sr as f64 / target_sr as f64;
        let out_len = (mono.len() as f64 / ratio) as usize;
        let resampled: Vec<f32> = (0..out_len)
            .map(|i| {
                let src = i as f64 * ratio;
                let lo = src.floor() as usize;
                let hi = (lo + 1).min(mono.len() - 1);
                let frac = src - lo as f64;
                mono[lo] * (1.0 - frac as f32) + mono[hi] * frac as f32
            })
            .collect();
        Ok(resampled)
    }

    /// Voice-clone: synthesize `text` in the voice of the speaker from `ref_audio_path`.
    ///
    /// `ref_audio_path`: path to a WAV file of the reference speaker (any SR, mono or stereo).
    /// `ref_text`: transcript of the reference audio (required for ICL mode).
    /// `language`: target language ("japanese", "chinese", "english", "auto", …).
    ///
    /// Returns `(audio_tensor, sample_rate)`.
    pub fn generate_voice_clone(
        &mut self,
        text: &str,
        language: &str,
        ref_audio_path: &str,
        ref_text: &str,
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        repetition_penalty: f32,
    ) -> Result<(Tensor, u32)> {
        // 1. Validate model type
        match self.config.tts_model_type.as_deref().unwrap_or("base") {
            "base" => {} // OK
            other => anyhow::bail!(
                "generate_voice_clone requires tts_model_type=base, got '{other}'. \
                 Use Qwen3-TTS-12Hz-0.6B-Base."
            ),
        }

        let spk_sr = self.config.speaker_encoder_config.sample_rate;

        // 2. Load reference audio at speaker encoder SR (24kHz)
        let ref_samples_spk = Self::load_wav_f32(ref_audio_path, spk_sr)?;
        if ref_samples_spk.is_empty() {
            anyhow::bail!("Reference audio file is empty: {ref_audio_path}");
        }

        // 3. Extract speaker embedding via ECAPA-TDNN
        //    Speaker encoder runs in F32 for precision (matching vendor).
        //    The embedding is later cast to model dtype in build_voice_clone_prefill.
        let spk_embed = {
            let enc = self.inner.speaker_encoder.as_ref()
                .ok_or_else(|| anyhow::anyhow!("Speaker encoder not loaded (base model required)"))?;
            let mels = Self::compute_mel_spectrogram(&ref_samples_spk, &self.device, DType::F32)?;
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
                    let ref_tensor = Tensor::new(ref_samples_spk.as_slice(), &self.device)?
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
        let (new_codes, ref_code_len) = self.inner.generate_voice_clone_codes(
            &input_ids,
            &ref_token_ids,
            &ref_codes,
            &spk_embed,
            language,
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
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

        let _ = ref_code_len;
        Ok((audio, speech_decoder.sample_rate()))
    }

    /// Voice-clone and write to WAV file.
    pub fn generate_voice_clone_to_file(
        &mut self,
        text: &str,
        language: &str,
        ref_audio_path: &str,
        ref_text: &str,
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        repetition_penalty: f32,
        output_path: &str,
    ) -> Result<String> {
        let (audio, sr) = self.generate_voice_clone(
            text, language, ref_audio_path, ref_text,
            max_new_tokens, temperature, top_p, repetition_penalty,
        )?;
        Self::save_wav(&audio, output_path, sr)
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
