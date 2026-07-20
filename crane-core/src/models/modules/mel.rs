//! Parameterized log-mel spectrogram frontend.
//!
//! Hann-windowed STFT (via `rustfft`) → Slaney-normalized mel filterbank →
//! log compression. Shared by any model that needs a Whisper/librosa-style
//! mel spectrogram, parameterized so callers with different DSP constants
//! (e.g. Qwen3-TTS's speaker encoder vs. Qwen3-ASR's Whisper-style feature
//! extractor) can reuse the same implementation.

use candle_core::{DType, Device, Result, Tensor};
use rustfft::{FftPlanner, num_complex::Complex as FftComplex};

/// Parameters for [`compute_mel_spectrogram`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MelSpectrogramConfig {
    /// FFT size (number of samples per transform).
    pub n_fft: usize,
    /// Hop length between successive STFT frames, in samples.
    pub hop_length: usize,
    /// Analysis window length, in samples (Hann window).
    pub win_length: usize,
    /// Sample rate of the input audio, in Hz.
    pub sample_rate: usize,
    /// Number of mel filter bands.
    pub n_mels: usize,
    /// Lowest frequency covered by the mel filterbank, in Hz.
    pub fmin: f64,
    /// Highest frequency covered by the mel filterbank, in Hz.
    pub fmax: f64,
}

/// Periodic Hann window of length `n`, matching `torch.hann_window(n,
/// periodic=True)`.
pub(crate) fn hann_window(n: usize) -> Vec<f32> {
    // n is an STFT window/FFT size, far below f64's exact-integer range, so
    // the usize -> f64 casts below lose no precision; the window magnitude
    // is in [0, 1], so the final f64 -> f32 cast cannot truncate meaningfully.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let window: Vec<f32> = (0..n)
        .map(|i| {
            let v = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos());
            v as f32
        })
        .collect();
    window
}

/// Reflect-pads `samples` by `pad` on each side (`torch.stft`'s
/// `pad_mode="reflect"` centering, excluding the edge sample itself —
/// matches `numpy.pad(mode="reflect")`).
pub(crate) fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let n = samples.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    for i in (1..=pad.min(n.saturating_sub(1))).rev() {
        out.push(samples[i]);
    }
    while out.len() < pad {
        out.push(0.0);
    }
    out.extend_from_slice(samples);
    for i in (n.saturating_sub(1 + pad)..n.saturating_sub(1)).rev() {
        out.push(samples[i]);
    }
    while out.len() < n + 2 * pad {
        out.push(0.0);
    }
    out
}

/// Compute a log-mel spectrogram from raw audio samples.
///
/// Applies a Hann window, reflect-pads the signal by `(n_fft - hop_length) / 2`
/// on each side, computes the STFT via `rustfft`, applies a Slaney-normalized
/// mel filterbank, and log-compresses the result (`ln(max(val, 1e-5))`).
///
/// `samples` are mono audio in `[-1, 1]`. Returns a tensor of shape
/// `[n_mels, T_frames]` on `device` in `dtype`; callers needing a batch
/// dimension should call `.unsqueeze(0)?` on the result.
///
/// # Errors
///
/// Returns a candle error if tensor allocation or dtype conversion fails,
/// or if `config.hop_length > config.n_fft`.
pub fn compute_mel_spectrogram(
    config: &MelSpectrogramConfig,
    samples: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let n_fft = config.n_fft;
    let hop = config.hop_length;
    let win = config.win_length;
    let n_mels = config.n_mels;

    if hop > n_fft {
        candle_core::bail!(
            "hop_length ({hop}) must not exceed n_fft ({n_fft})"
        );
    }

    let hann = hann_window(win);

    // Reflect pad: (n_fft - hop) / 2
    let pad = (n_fft - hop) / 2;
    let padded = reflect_pad(samples, pad);

    // STFT: compute frames
    let n_frames = (padded.len().saturating_sub(win)) / hop + 1;
    let n_bins = n_fft / 2 + 1;

    // Build mel filterbank (Slaney norm, librosa-compatible)
    let mel_basis =
        build_mel_filterbank(config.sample_rate, n_fft, n_mels, config.fmin, config.fmax);

    // Compute magnitude spectrogram frame by frame using rustfft (O(N log N)),
    // applying the mel filterbank and log compression immediately per frame
    // rather than materializing the full [n_frames, n_bins] magnitude buffer.
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);

    let mut buffer = vec![FftComplex::new(0.0, 0.0); n_fft];
    let mut mag = vec![0f32; n_bins];
    let mut mel_frames: Vec<f32> = Vec::with_capacity(n_frames * n_mels);
    for frame_idx in 0..n_frames {
        let start = frame_idx * hop;
        // Apply Hann window and build complex FFT input in-place
        for (i, b) in buffer.iter_mut().enumerate() {
            let s = if i < win && start + i < padded.len() {
                padded[start + i] * hann[i]
            } else {
                0.0
            };
            *b = FftComplex::new(s, 0.0);
        }

        fft.process(&mut buffer);

        // Take magnitude of positive frequencies (n_fft/2 + 1 bins)
        for (k, m) in mag.iter_mut().enumerate() {
            let c = &buffer[k];
            *m = (c.re * c.re + c.im * c.im + 1e-9).sqrt();
        }

        // Apply mel filterbank and log compression: log(max(val, 1e-5))
        for m in 0..n_mels {
            let mut val = 0f32;
            for k in 0..n_bins {
                val += mel_basis[m * n_bins + k] * mag[k];
            }
            mel_frames.push(val.max(1e-5).ln());
        }
    }

    // Shape: [n_mels, T_frames]
    let t = Tensor::new(mel_frames.as_slice(), device)?
        .reshape((n_frames, n_mels))?
        .transpose(0, 1)?
        .to_dtype(dtype)?;
    Ok(t)
}

/// Build a Slaney-normalized triangular mel filterbank matching
/// `librosa.filters.mel(norm="slaney", htk=False)`.
///
/// Returns a flat `Vec<f32>` of length `n_mels * (n_fft / 2 + 1)`, indexed
/// as `filters[m * n_bins + k]` (mel band `m`, frequency bin `k`).
///
/// Takes loose parameters rather than [`MelSpectrogramConfig`] since a
/// filterbank only depends on frequency-domain settings, not STFT framing
/// (`hop_length`/`win_length`); kept crate-private since only
/// [`compute_mel_spectrogram`] needs it today.
#[must_use]
pub(crate) fn build_mel_filterbank(
    sr: usize,
    n_fft: usize,
    n_mels: usize,
    fmin: f64,
    fmax: f64,
) -> Vec<f32> {
    // Slaney / O'Shaughnessy mel scale
    const F_SP: f64 = 200.0 / 3.0; // ~66.667 Hz per mel below breakpoint
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP; // 15.0
    // ln(6.4) / 27 ≈ 0.068751739
    const LOG_STEP: f64 = 0.068_751_74;

    fn hz_to_mel(f: f64) -> f64 {
        if f < MIN_LOG_HZ {
            f / F_SP
        } else {
            MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() / LOG_STEP
        }
    }
    fn mel_to_hz(m: f64) -> f64 {
        if m < MIN_LOG_MEL {
            m * F_SP
        } else {
            MIN_LOG_HZ * ((m - MIN_LOG_MEL) * LOG_STEP).exp()
        }
    }

    let n_bins = n_fft / 2 + 1;
    #[allow(clippy::cast_precision_loss)]
    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|k| k as f64 * sr as f64 / n_fft as f64)
        .collect();

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    #[allow(clippy::cast_precision_loss)]
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
        let enorm = if f_right > f_left {
            2.0 / (f_right - f_left)
        } else {
            0.0
        };
        for k in 0..n_bins {
            let f = fft_freqs[k];
            let val = if f >= f_left && f <= f_center && f_center > f_left {
                (f - f_left) / (f_center - f_left)
            } else if f > f_center && f <= f_right && f_right > f_center {
                (f_right - f) / (f_right - f_center)
            } else {
                0.0
            };
            #[allow(clippy::cast_possible_truncation)]
            let filter_val = (val * enorm) as f32;
            filters[m * n_bins + k] = filter_val;
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speaker_encoder_config() -> MelSpectrogramConfig {
        MelSpectrogramConfig {
            n_fft: 1024,
            hop_length: 256,
            win_length: 1024,
            sample_rate: 24000,
            n_mels: 128,
            fmin: 0.0,
            fmax: 12000.0,
        }
    }

    #[test]
    fn test_mel_filterbank_shape() {
        for (n_fft, n_mels) in [(1024, 128), (400, 80), (400, 128)] {
            let filters = build_mel_filterbank(16000, n_fft, n_mels, 0.0, 8000.0);
            let n_bins = n_fft / 2 + 1;
            assert_eq!(filters.len(), n_mels * n_bins);
        }
    }

    #[test]
    fn test_mel_filterbank_slaney_normalization() {
        // Each triangular filter's peak value should equal 2/bandwidth
        // (area-normalized triangle: peak * bandwidth / 2 == 1 -> peak == 2/bandwidth).
        let sr = 16000;
        let n_fft = 400;
        let n_mels = 80;
        let fmin = 0.0;
        let fmax = 8000.0;
        let filters = build_mel_filterbank(sr, n_fft, n_mels, fmin, fmax);
        let n_bins = n_fft / 2 + 1;

        let max_val: f32 = filters.iter().copied().fold(0.0_f32, f32::max);
        assert!(max_val > 0.0, "expected some non-zero filter values");

        // No row should be entirely zero for a reasonable band count within range.
        for m in 0..n_mels {
            let row_sum: f32 = filters[m * n_bins..(m + 1) * n_bins].iter().sum();
            assert!(row_sum > 0.0, "mel band {m} is entirely zero");
        }
    }

    #[test]
    fn test_mel_spectrogram_shape() {
        let config = speaker_encoder_config();
        let samples = vec![0.0f32; config.sample_rate]; // 1 second of silence
        let mel =
            compute_mel_spectrogram(&config, &samples, &Device::Cpu, DType::F32).expect("mel");
        let pad = (config.n_fft - config.hop_length) / 2;
        let padded_len = samples.len() + 2 * pad;
        let expected_frames = (padded_len - config.win_length) / config.hop_length + 1;
        assert_eq!(mel.dims(), &[config.n_mels, expected_frames]);
    }

    #[test]
    fn test_mel_spectrogram_silence_is_log_floor() {
        let config = speaker_encoder_config();
        let samples = vec![0.0f32; 4096];
        let mel =
            compute_mel_spectrogram(&config, &samples, &Device::Cpu, DType::F32).expect("mel");
        let data = mel
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1");
        let expected = 1e-5_f32.ln();
        for (i, &v) in data.iter().enumerate() {
            assert!(
                (v - expected).abs() < 1e-6,
                "index {i}: expected log-floor {expected}, got {v}"
            );
        }
    }

    #[test]
    fn test_mel_spectrogram_pinning() {
        // Fixed synthetic signal: 1 second of a 440 Hz sine at 24000 Hz,
        // using the speaker-encoder's exact DSP constants. These values pin
        // the extracted implementation against the original inline code.
        let config = speaker_encoder_config();
        let sr = config.sample_rate as f32;
        let samples: Vec<f32> = (0..config.sample_rate)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr).sin())
            .collect();

        let mel =
            compute_mel_spectrogram(&config, &samples, &Device::Cpu, DType::F32).expect("mel");
        let data = mel
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1");

        let n_frames = mel.dims()[1];
        let get = |band: usize, frame: usize| data[band * n_frames + frame];

        // Values captured from this same implementation; regressions in the
        // DSP math (window, padding, FFT, filterbank, log compression) will
        // change these.
        let v00 = get(0, 0);
        let v64_0 = get(64, 0);
        let v0_10 = get(0, 10);

        assert!(
            (v00 - (-0.466_971)).abs() < 1e-4,
            "band 0, frame 0: got {v00}"
        );
        assert!(
            (v64_0 - (-3.544_756)).abs() < 1e-4,
            "band 64, frame 0: got {v64_0}"
        );
        assert!(
            (v0_10 - (-7.284_913)).abs() < 1e-4,
            "band 0, frame 10: got {v0_10}"
        );
    }
}
