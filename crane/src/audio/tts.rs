//! Shared TTS abstraction: trait, stream, and PCM utilities.
//!
//! Consumers (crane-engine, crane-serve, ...) hold a
//! `Box<dyn Tts>` and never import model-specific types.

use std::path::Path;

use anyhow::Result;
use candle_core::Tensor;

use crane_core::generation::SpeechOptions;

/// Audio format metadata returned by TTS models.
///
/// Describes the PCM format of generated audio. All Crane TTS models
/// currently produce mono 16-bit PCM; this struct makes the format
/// explicit so consumers don't have to hard-code assumptions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioInfo {
    /// Sample rate in Hz (e.g. 24000).
    pub sample_rate: u32,
    /// Number of audio channels (1 = mono).
    pub channels: u16,
    /// Bits per sample (16 for i16 PCM).
    pub bits_per_sample: u16,
}

/// Voice information for service discovery.
///
/// Describes one voice that a TTS model can produce. Used by the
/// Wyoming `info` event and the HTTP API voice listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceInfo {
    /// Voice identifier (e.g. "neutral_female", "Chelsie").
    pub name: String,
    /// Language identifiers this voice supports (e.g. `["en", "zh"]`).
    ///
    /// Values are typically BCP-47 codes for Wyoming service discovery but
    /// may be any string the model recognizes. Empty means the voice works
    /// with any language the model supports.
    pub languages: Vec<String>,
}

/// A stream of f32 PCM audio chunks produced incrementally during
/// speech generation.
///
/// Consumers iterate with `while let Some(chunk) = stream.next_chunk()? { ... }`
/// and read the audio format from `stream.audio_info`.
///
/// The f32-to-i16 PCM conversion is NOT done here. Streams yield raw
/// f32 samples; the transport layer (Wyoming or HTTP) converts to the
/// wire format (i16 LE) before sending. This keeps the stream
/// transport-agnostic and avoids double conversion when the consumer
/// needs f32 (e.g. for resampling or normalization).
pub struct TtsStream<'a> {
    /// Audio format for all chunks in this stream.
    pub audio_info: AudioInfo,
    inner: Box<dyn Iterator<Item = Result<Tensor>> + 'a>,
}

impl<'a> TtsStream<'a> {
    /// Create a stream from an audio format and a chunk iterator.
    pub fn new(audio_info: AudioInfo, iter: impl Iterator<Item = Result<Tensor>> + 'a) -> Self {
        Self {
            audio_info,
            inner: Box::new(iter),
        }
    }

    /// Returns the next chunk of f32 PCM audio, or `None` when generation is complete.
    ///
    /// Returns `Err` if the underlying model encounters a generation error.
    pub fn next_chunk(&mut self) -> Result<Option<Tensor>> {
        self.inner.next().transpose()
    }

    /// Create a single-chunk stream from a completed tensor.
    ///
    /// Used by the default [`Tts::generate_speech_stream`] implementation
    /// for models that produce the full waveform before returning.
    pub fn once(audio_info: AudioInfo, tensor: Tensor) -> Self {
        Self::new(audio_info, std::iter::once(Ok(tensor)))
    }
}

/// Unified TTS interface.
///
/// Implemented by every TTS model in Crane. Consumers (crane-engine,
/// crane-serve, crane-wyoming) use `Box<dyn Tts>` and never import
/// model-specific types.
pub trait Tts {
    /// Returns audio format metadata for this model.
    ///
    /// The returned [`AudioInfo`] describes the PCM format that
    /// [`generate_speech`](Tts::generate_speech) and
    /// [`generate_speech_stream`](Tts::generate_speech_stream) produce.
    fn audio_info(&self) -> AudioInfo;

    /// Lists available voices.
    ///
    /// Returns an empty Vec for models that only support voice cloning
    /// (no predefined voices).
    fn voices(&self) -> Vec<VoiceInfo>;

    /// Returns true if this model supports voice cloning from reference audio.
    fn supports_voice_cloning(&self) -> bool {
        false
    }

    /// Generate complete audio for the given text.
    ///
    /// Returns the full waveform as a single tensor of f32 PCM samples.
    /// Use this for the HTTP `/v1/audio/speech` endpoint and anywhere
    /// latency-to-first-byte is not critical.
    ///
    /// # Parameters
    ///
    /// * `text` — The text to synthesize.
    /// * `language` — Target language (e.g. "english", "auto").
    /// * `voice` — Voice name from [`voices`](Tts::voices), or `None` for the model default.
    /// * `opts` — Generation parameters (temperature, max tokens, etc.).
    ///
    /// The tensor shape is model-dependent; callers should use `.flatten_all()` to
    /// obtain a flat sample buffer before passing to [`pcm_f32_to_i16`].
    /// The sample rate is available via [`audio_info`](Tts::audio_info).
    fn generate_speech(
        &mut self,
        text: &str,
        language: &str,
        voice: Option<&str>,
        opts: &SpeechOptions,
    ) -> Result<Tensor>;

    /// Clone a voice from reference audio and synthesize text in that voice.
    ///
    /// # Parameters
    ///
    /// * `text` — The text to synthesize.
    /// * `language` — Target language.
    /// * `ref_audio` — Path to a reference WAV file of the target speaker.
    /// * `ref_text` — Transcript of the reference audio (required by some models).
    /// * `opts` — Generation parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the model does not support voice cloning.
    /// Check [`supports_voice_cloning`](Tts::supports_voice_cloning) first.
    fn generate_voice_clone(
        &mut self,
        _text: &str,
        _language: &str,
        _ref_audio: &Path,
        _ref_text: &str,
        _opts: &SpeechOptions,
    ) -> Result<Tensor> {
        anyhow::bail!("voice cloning not supported by this model")
    }

    /// Generate audio as a stream of f32 PCM chunks.
    ///
    /// Returns a [`TtsStream`] that yields chunks incrementally as the model
    /// generates them. Use this for Wyoming (`audio-chunk` events) and future
    /// HTTP streaming.
    ///
    /// The default implementation calls [`generate_speech`](Tts::generate_speech)
    /// and wraps the result in a single-chunk stream. Models that support true
    /// incremental generation override this method.
    fn generate_speech_stream(
        &mut self,
        text: &str,
        language: &str,
        voice: Option<&str>,
        opts: &SpeechOptions,
    ) -> Result<TtsStream<'_>> {
        let tensor = self.generate_speech(text, language, voice, opts)?;
        Ok(TtsStream::once(self.audio_info(), tensor))
    }
}

/// Convert f32 PCM samples to 16-bit little-endian PCM bytes.
///
/// Each sample is scaled by 32767, clamped to the i16 range
/// `[-32768, 32767]`, and written as two little-endian bytes. The returned
/// Vec has length `samples.len() * 2`.
///
/// # PCM asymmetry
///
/// Positive full-scale (`1.0`) maps to `32767` and negative full-scale
/// (`-1.0`) maps to `-32767`, not `-32768`. This matches the convention
/// used by most audio software (libsndfile, ffmpeg, SoX): symmetric
/// scaling avoids DC offset on clipping. `i16::MIN` (`-32768`) is only
/// produced when a sample exceeds `-1.0` and is clamped.
///
/// # NaN and Infinity
///
/// Non-finite inputs (`NaN`, `Infinity`, `-Infinity`) are mapped to
/// silence (`0`). Passing them through the multiply/clamp path would
/// produce platform-dependent results.
///
/// This is the canonical conversion used by the Wyoming transport
/// (`audio-chunk` payloads) and the HTTP PCM response format.
pub fn pcm_f32_to_i16(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = if s.is_finite() {
            (s * 32767.0).clamp(f32::from(i16::MIN), f32::from(i16::MAX))
        } else {
            0.0
        };
        #[allow(clippy::cast_possible_truncation)]
        let s16 = clamped as i16;
        out.extend_from_slice(&s16.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // --- AudioInfo ---

    #[test]
    fn test_audio_info_construction() {
        let info = AudioInfo {
            sample_rate: 24000,
            channels: 1,
            bits_per_sample: 16,
        };
        assert_eq!(info.sample_rate, 24000);
        assert_eq!(info.channels, 1);
        assert_eq!(info.bits_per_sample, 16);

        let info2 = info;
        assert_eq!(info, info2);
    }

    // --- VoiceInfo ---

    #[test]
    fn test_voice_info_construction() {
        let voice = VoiceInfo {
            name: "neutral_female".to_string(),
            languages: vec!["en".to_string(), "fr".to_string()],
        };
        assert_eq!(voice.name, "neutral_female");
        assert_eq!(voice.languages.len(), 2);

        let empty = VoiceInfo {
            name: "default".to_string(),
            languages: vec![],
        };
        assert!(empty.languages.is_empty());
    }

    // --- pcm_f32_to_i16 ---

    #[test]
    fn test_pcm_silence() {
        let bytes = pcm_f32_to_i16(&[0.0]);
        assert_eq!(bytes, vec![0, 0]);
    }

    #[test]
    fn test_pcm_max_positive() {
        let bytes = pcm_f32_to_i16(&[1.0]);
        assert_eq!(bytes, 32767i16.to_le_bytes().to_vec());
    }

    #[test]
    fn test_pcm_max_negative() {
        let bytes = pcm_f32_to_i16(&[-1.0]);
        assert_eq!(bytes, (-32767i16).to_le_bytes().to_vec());
    }

    #[test]
    fn test_pcm_clamping_high() {
        let bytes = pcm_f32_to_i16(&[2.5]);
        assert_eq!(bytes, 32767i16.to_le_bytes().to_vec());
    }

    #[test]
    fn test_pcm_clamping_low() {
        let bytes = pcm_f32_to_i16(&[-3.0]);
        assert_eq!(bytes, i16::MIN.to_le_bytes().to_vec());
    }

    #[test]
    fn test_pcm_empty() {
        assert!(pcm_f32_to_i16(&[]).is_empty());
    }

    #[test]
    fn test_pcm_output_length() {
        let samples: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        assert_eq!(pcm_f32_to_i16(&samples).len(), samples.len() * 2);
    }

    #[test]
    fn test_pcm_nan_is_silence() {
        assert_eq!(pcm_f32_to_i16(&[f32::NAN]), vec![0, 0]);
    }

    #[test]
    fn test_pcm_positive_infinity_is_silence() {
        assert_eq!(pcm_f32_to_i16(&[f32::INFINITY]), vec![0, 0]);
    }

    #[test]
    fn test_pcm_negative_infinity_is_silence() {
        assert_eq!(pcm_f32_to_i16(&[f32::NEG_INFINITY]), vec![0, 0]);
    }

    // --- TtsStream ---

    #[test]
    fn test_tts_stream_once() -> Result<()> {
        let info = AudioInfo { sample_rate: 24000, channels: 1, bits_per_sample: 16 };
        let tensor = Tensor::new(&[0.1f32, 0.2, 0.3], &Device::Cpu)?;

        let mut stream = TtsStream::once(info, tensor);
        assert_eq!(stream.audio_info, info);

        let chunk = stream.next_chunk()?.expect("should have one chunk");
        assert_eq!(chunk.to_vec1::<f32>()?, vec![0.1, 0.2, 0.3]);

        assert!(stream.next_chunk()?.is_none());
        Ok(())
    }

    #[test]
    fn test_tts_stream_multi_chunk() -> Result<()> {
        let info = AudioInfo { sample_rate: 24000, channels: 1, bits_per_sample: 16 };
        let dev = &Device::Cpu;
        let chunks = vec![
            Ok(Tensor::new(&[1.0f32, 2.0], dev)?),
            Ok(Tensor::new(&[3.0f32, 4.0, 5.0], dev)?),
            Ok(Tensor::new(&[6.0f32], dev)?),
        ];

        let mut stream = TtsStream::new(info, chunks.into_iter());

        let c1 = stream.next_chunk()?.expect("chunk 1");
        assert_eq!(c1.to_vec1::<f32>()?, vec![1.0, 2.0]);

        let c2 = stream.next_chunk()?.expect("chunk 2");
        assert_eq!(c2.to_vec1::<f32>()?, vec![3.0, 4.0, 5.0]);

        let c3 = stream.next_chunk()?.expect("chunk 3");
        assert_eq!(c3.to_vec1::<f32>()?, vec![6.0]);

        assert!(stream.next_chunk()?.is_none());
        Ok(())
    }

    #[test]
    fn test_tts_stream_error_propagation() {
        let info = AudioInfo { sample_rate: 24000, channels: 1, bits_per_sample: 16 };
        let chunks: Vec<Result<Tensor>> = vec![Err(anyhow::anyhow!("generation failed"))];
        let mut stream = TtsStream::new(info, chunks.into_iter());

        let result = stream.next_chunk();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("generation failed"));
    }

    #[test]
    fn test_tts_stream_empty() -> Result<()> {
        let info = AudioInfo { sample_rate: 24000, channels: 1, bits_per_sample: 16 };
        let chunks: Vec<Result<Tensor>> = vec![];
        let mut stream = TtsStream::new(info, chunks.into_iter());
        assert!(stream.next_chunk()?.is_none());
        Ok(())
    }

    // --- Tts trait default methods ---

    struct MockTts;

    impl Tts for MockTts {
        fn audio_info(&self) -> AudioInfo {
            AudioInfo { sample_rate: 24000, channels: 1, bits_per_sample: 16 }
        }

        fn voices(&self) -> Vec<VoiceInfo> {
            vec![VoiceInfo { name: "test".to_string(), languages: vec![] }]
        }

        fn generate_speech(
            &mut self,
            _text: &str,
            _language: &str,
            _voice: Option<&str>,
            _opts: &SpeechOptions,
        ) -> Result<Tensor> {
            Tensor::new(&[0.5f32, -0.5], &Device::Cpu).map_err(Into::into)
        }
    }

    #[test]
    fn test_default_supports_voice_cloning() {
        assert!(!MockTts.supports_voice_cloning());
    }

    #[test]
    fn test_default_generate_voice_clone_errors() {
        let opts = SpeechOptions::default();
        let result = MockTts.generate_voice_clone("hello", "en", Path::new("/ref.wav"), "hello", &opts);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("voice cloning not supported"));
    }

    #[test]
    fn test_default_generate_speech_stream() -> Result<()> {
        let opts = SpeechOptions::default();
        let mut model = MockTts;
        let mut stream = model.generate_speech_stream("hello", "en", None, &opts)?;

        assert_eq!(stream.audio_info.sample_rate, 24000);

        let chunk = stream.next_chunk()?.expect("should have one chunk");
        assert_eq!(chunk.to_vec1::<f32>()?, vec![0.5, -0.5]);

        assert!(stream.next_chunk()?.is_none());
        Ok(())
    }
}
