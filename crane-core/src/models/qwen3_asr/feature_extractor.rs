//! Whisper-compatible log-mel spectrogram feature extractor for Qwen3-ASR.
//!
//! `Qwen3ASRProcessor` uses `transformers`' own `WhisperFeatureExtractor`
//! directly: Hann-windowed STFT (`n_fft=400`, `hop_length=160`) → Slaney
//! mel filterbank (128 bands, `fmax=8000`) → power spectrum → `log10` →
//! per-utterance dynamic-range normalization.
//! This differs from
//! [`crate::models::modules::mel::compute_mel_spectrogram`] (extracted from
//! the `qwen3_tts` speaker encoder) in padding amount, spectrum type, log
//! base, and post-log normalization, so this module implements the STFT/log
//! pipeline directly rather than reusing that helper — only the Slaney mel
//! filterbank builder and the low-level Hann window / reflect-pad helpers
//! are shared.
//!
//! Also verified: `Qwen3ASRProcessorKwargs` passes `padding=True` (a bool),
//! which `SequenceFeatureExtractor._get_padding_strategies` resolves to
//! `PaddingStrategy.LONGEST`, not `"max_length"`. For a single (non-batched)
//! utterance this is a no-op — there is no fixed 480000-sample/30s padding
//! for offline single-utterance transcription, matching the encoder's own
//! `% 100` / `// 100` chunking formula, which only makes sense against the
//! audio's real length.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use rustfft::{Fft, FftPlanner, num_complex::Complex as FftComplex};

use crate::models::modules::mel::{build_mel_filterbank, hann_window, reflect_pad};

/// FFT size for the STFT (`WhisperFeatureExtractor`'s `n_fft`).
const N_FFT: usize = 400;
/// Hop length between STFT frames, in samples — 100Hz frame rate at 16kHz.
const HOP_LENGTH: usize = 160;
/// Audio sample rate the feature extractor expects, in Hz.
pub(crate) const SAMPLE_RATE: usize = 16_000;
/// Shortest input `extract` will process, in samples (0.5s at 16kHz) —
/// `WhisperFeatureExtractor`'s `min_length`; shorter input is zero-padded up
/// to this length before extraction.
const MIN_LENGTH: usize = 8_000;
/// Highest frequency covered by the mel filterbank, in Hz — hardcoded in
/// `WhisperFeatureExtractor` independent of `sampling_rate`.
const FMAX: f64 = 8000.0;
/// Raw mel frames per second (`HOP_LENGTH` at `SAMPLE_RATE`); also the
/// window size (`n_window * 2`) the audio encoder chunks on (§3).
pub(crate) const FRAMES_PER_WINDOW: usize = 100;
/// Encoder output tokens produced by one full `FRAMES_PER_WINDOW`-sized
/// chunk (§3).
pub(crate) const TOKENS_PER_WINDOW: usize = 13;

/// Output of [`WhisperFeatureExtractor::extract`].
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    /// Log-mel spectrogram, shape `[1, n_mels, n_frames]`.
    pub input_features: Tensor,
    /// Number of mel frames produced (100Hz raw frame rate).
    pub real_frame_count: usize,
}

/// Whisper-compatible log-mel spectrogram frontend, matching
/// `WhisperFeatureExtractor`'s parameters (`n_fft=400`, `hop_length=160`,
/// `sampling_rate=16000`) as used by `Qwen3ASRProcessor` (§3).
pub struct WhisperFeatureExtractor {
    /// Slaney mel filterbank, transposed to `[n_bins, n_mels]` and made
    /// contiguous so `extract` can project a whole utterance's power
    /// spectrogram through it with a single `matmul` rather than a
    /// per-frame scalar dot-product loop.
    mel_filters_t: Tensor,
    hann_window: Vec<f32>,
    /// Precomputed forward FFT plan for [`N_FFT`], built once since the
    /// transform size is a fixed constant.
    fft: Arc<dyn Fft<f32>>,
    n_mels: usize,
    device: Device,
    dtype: DType,
}

impl WhisperFeatureExtractor {
    /// Builds a feature extractor producing `n_mels` mel bands
    /// (`AudioConfig::num_mel_bins`).
    ///
    /// # Errors
    ///
    /// Returns a candle error if building the mel filterbank tensor fails.
    pub fn new(n_mels: usize, device: &Device, dtype: DType) -> Result<Self> {
        let n_bins = N_FFT / 2 + 1;
        let mel_filters = build_mel_filterbank(SAMPLE_RATE, N_FFT, n_mels, 0.0, FMAX);
        let mel_filters_t = Tensor::from_vec(mel_filters, (n_mels, n_bins), &Device::Cpu)?
            .transpose(0, 1)?
            .contiguous()?;
        let mut planner = FftPlanner::<f32>::new();
        Ok(Self {
            mel_filters_t,
            hann_window: hann_window(N_FFT),
            fft: planner.plan_fft_forward(N_FFT),
            n_mels,
            device: device.clone(),
            dtype,
        })
    }

    /// Extracts log-mel spectrogram features from mono `f32` PCM samples at
    /// 16kHz (`SAMPLE_RATE`).
    ///
    /// # Errors
    ///
    /// Returns a candle error if tensor allocation or dtype conversion
    /// fails.
    pub fn extract(&self, samples: &[f32]) -> Result<AudioFeatures> {
        let samples = if samples.len() < MIN_LENGTH {
            let mut padded = samples.to_vec();
            padded.resize(MIN_LENGTH, 0.0);
            std::borrow::Cow::Owned(padded)
        } else {
            std::borrow::Cow::Borrowed(samples)
        };

        let n_bins = N_FFT / 2 + 1;
        // Whisper convention: one fewer frame than a centered-STFT frame
        // count would give; the extra frame implied by the reflect-padded
        // length is computed then discarded.
        let n_frames = samples.len() / HOP_LENGTH;
        let padded = reflect_pad(&samples, N_FFT / 2);

        let mut buffer = vec![FftComplex::new(0.0, 0.0); N_FFT];
        let mut all_power = Vec::with_capacity(n_frames * n_bins);

        for frame_idx in 0..n_frames {
            let start = frame_idx * HOP_LENGTH;
            for (i, b) in buffer.iter_mut().enumerate() {
                *b = FftComplex::new(padded[start + i] * self.hann_window[i], 0.0);
            }
            self.fft.process(&mut buffer);

            // Power spectrum (magnitude-squared), not magnitude.
            all_power.extend(buffer[..n_bins].iter().map(|c| c.re * c.re + c.im * c.im));
        }

        // Batched mel projection: [n_frames, n_bins] x [n_bins, n_mels] via
        // a single matmul instead of a per-frame scalar dot-product loop.
        let power_t = Tensor::from_vec(all_power, (n_frames, n_bins), &Device::Cpu)?;
        let mut mel_frames = power_t
            .matmul(&self.mel_filters_t)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for v in &mut mel_frames {
            *v = v.max(1e-10).log10();
        }

        // Per-utterance dynamic-range clamp + rescale, matching
        // `WhisperFeatureExtractor`: clamp to (global max - 8), then
        // (v + 4) / 4.
        let global_max = mel_frames.iter().copied().fold(f32::MIN, f32::max);
        let floor = global_max - 8.0;
        for v in &mut mel_frames {
            *v = (v.max(floor) + 4.0) / 4.0;
        }

        let input_features = Tensor::from_vec(mel_frames, (n_frames, self.n_mels), &self.device)?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .to_dtype(self.dtype)?;

        Ok(AudioFeatures {
            input_features,
            real_frame_count: n_frames,
        })
    }
}

/// Output length of one stride-2, kernel-3, padding-1 `Conv2d` pass.
pub(crate) fn conv_output_len(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    (len - 1) / 2 + 1
}

/// Splits `n_frames` raw mel frames into a whole number of full
/// [`FRAMES_PER_WINDOW`]-sized chunks plus a trailing remainder, returned as
/// `(n_full_chunks, remainder)`. Shared by every call site that must chunk
/// on window boundaries identically (the encoder frontend, the
/// block-diagonal mask, and the output-length formula below) so they can't
/// drift out of sync.
pub(crate) fn chunk_split(n_frames: usize) -> (usize, usize) {
    (n_frames / FRAMES_PER_WINDOW, n_frames % FRAMES_PER_WINDOW)
}

/// Number of audio encoder output tokens for `mel_frames` raw (100Hz) mel
/// frames — mirrors `_get_feat_extract_output_lengths` (§3), used to expand
/// the `<|audio_pad|>` placeholder to the encoder's actual output length.
///
/// Each full [`FRAMES_PER_WINDOW`]-frame chunk contributes exactly
/// [`TOKENS_PER_WINDOW`] tokens; the trailing partial chunk is run through
/// the same three stride-2 conv-output-length steps as the encoder's own
/// `Conv2d` frontend.
#[must_use]
pub fn get_feat_extract_output_lengths(mel_frames: usize) -> usize {
    let (full_windows, remainder) = chunk_split(mel_frames);
    let remainder_tokens = conv_output_len(conv_output_len(conv_output_len(remainder)));
    remainder_tokens + full_windows * TOKENS_PER_WINDOW
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_N_MELS: usize = 128;

    #[test]
    fn output_lengths_match_reference_formula() {
        // Hand-computed against the Python reference:
        //   input_lengths_leave = n % 100
        //   feat = (input_lengths_leave - 1) // 2 + 1
        //   out = ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (n // 100) * 13
        assert_eq!(get_feat_extract_output_lengths(0), 0);
        assert_eq!(get_feat_extract_output_lengths(100), 13); // exactly 1s
        assert_eq!(get_feat_extract_output_lengths(150), 20); // 1.5s
        assert_eq!(get_feat_extract_output_lengths(830), 108); // 8.3s
        assert_eq!(get_feat_extract_output_lengths(99), 13); // partial chunk only
        assert_eq!(get_feat_extract_output_lengths(3000), 390); // 30s
    }

    #[test]
    fn mel_shape_matches_frame_rate() {
        let extractor =
            WhisperFeatureExtractor::new(TEST_N_MELS, &Device::Cpu, DType::F32).expect("new");
        let samples = vec![0.0f32; SAMPLE_RATE]; // 1s of silence
        let features = extractor.extract(&samples).expect("extract");
        assert_eq!(features.real_frame_count, 100);
        assert_eq!(features.input_features.dims(), &[1, 128, 100]);
    }

    #[test]
    fn real_frame_count_scales_with_audio_length() {
        let extractor =
            WhisperFeatureExtractor::new(TEST_N_MELS, &Device::Cpu, DType::F32).expect("new");

        let one_second = vec![0.0f32; SAMPLE_RATE];
        assert_eq!(
            extractor
                .extract(&one_second)
                .expect("extract")
                .real_frame_count,
            100
        );

        let one_and_half_seconds = vec![0.0f32; SAMPLE_RATE + SAMPLE_RATE / 2];
        assert_eq!(
            extractor
                .extract(&one_and_half_seconds)
                .expect("extract")
                .real_frame_count,
            150
        );
    }

    #[test]
    fn silence_is_constant_log_floor() {
        let extractor =
            WhisperFeatureExtractor::new(TEST_N_MELS, &Device::Cpu, DType::F32).expect("new");
        let samples = vec![0.0f32; SAMPLE_RATE];
        let features = extractor.extract(&samples).expect("extract");
        let data = features
            .input_features
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1");

        // All-silent input: every mel bin is the log floor before
        // normalization, so global_max == floor and every output value
        // collapses to the same constant after (v + 4) / 4.
        let expected = (1e-10_f32.log10() + 4.0) / 4.0;
        for (i, &v) in data.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-6,
                "index {i}: got {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn short_audio_is_padded_to_min_length() {
        let extractor =
            WhisperFeatureExtractor::new(TEST_N_MELS, &Device::Cpu, DType::F32).expect("new");

        // Below MIN_LENGTH: zero-padded up to MIN_LENGTH (8000 samples / 50 frames).
        let short = vec![0.0f32; 100];
        let short_features = extractor.extract(&short).expect("extract");
        assert_eq!(short_features.real_frame_count, MIN_LENGTH / HOP_LENGTH);
        assert_eq!(
            short_features.input_features.dims(),
            &[1, TEST_N_MELS, MIN_LENGTH / HOP_LENGTH]
        );

        // Explicit MIN_LENGTH-sample silence should match exactly.
        let min_len_silence = vec![0.0f32; MIN_LENGTH];
        let min_len_features = extractor.extract(&min_len_silence).expect("extract");
        assert_eq!(
            short_features.real_frame_count,
            min_len_features.real_frame_count
        );
    }

    #[test]
    fn test_tone_pinning() {
        // Fixed synthetic signal: 1 second of a 440 Hz sine at 16000 Hz.
        // These values pin the extracted STFT/mel/log/normalization
        // pipeline against this same implementation; a regression in the
        // window, padding, FFT, filterbank, or normalization math will
        // change them.
        let extractor =
            WhisperFeatureExtractor::new(TEST_N_MELS, &Device::Cpu, DType::F32).expect("new");
        let sr = SAMPLE_RATE as f32;
        let samples: Vec<f32> = (0..SAMPLE_RATE)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr).sin())
            .collect();

        let features = extractor.extract(&samples).expect("extract");
        let n_frames = features.real_frame_count;
        let data = features
            .input_features
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1");
        let get = |band: usize, frame: usize| data[band * n_frames + frame];

        let v00 = get(0, 0);
        let v64_0 = get(64, 0);
        let v0_10 = get(0, 10);

        assert!(
            (v00 - 1.058_034_8).abs() < 1e-4,
            "band 0, frame 0: got {v00}"
        );
        assert!(
            (v64_0 - 0.564_188).abs() < 1e-4,
            "band 64, frame 0: got {v64_0}"
        );
        assert!(
            (v0_10 - (-0.364_131)).abs() < 1e-4,
            "band 0, frame 10: got {v0_10}"
        );
    }
}
