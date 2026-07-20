//! Shared PCM/WAV utilities used by both TTS and ASR.
//!
//! These conversions are direction-agnostic: TTS uses them to encode
//! generated audio for transport, ASR uses them to decode incoming audio
//! for transcription.

use std::io::Cursor;

use anyhow::Result;
use candle_core::{DType, Tensor};
use hound::{SampleFormat, WavSpec, WavWriter};

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

/// Convert a single f32 sample to a rounded, clamped i16 PCM sample.
///
/// The sample is scaled by 32767, rounded to the nearest integer, and
/// clamped to the i16 range `[-32768, 32767]`. Non-finite inputs (`NaN`,
/// `Infinity`, `-Infinity`) are mapped to silence (`0`).
fn f32_to_i16_sample(s: f32) -> i16 {
    let clamped = if s.is_finite() {
        (s * 32767.0)
            .round()
            .clamp(f32::from(i16::MIN), f32::from(i16::MAX))
    } else {
        0.0
    };
    #[allow(clippy::cast_possible_truncation)]
    let s16 = clamped as i16;
    s16
}

/// Convert f32 PCM samples to 16-bit little-endian PCM bytes.
///
/// Each sample is scaled by 32767, rounded to the nearest integer, clamped
/// to the i16 range `[-32768, 32767]`, and written as two little-endian
/// bytes. The returned Vec has length `samples.len() * 2`.
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
        out.extend_from_slice(&f32_to_i16_sample(s).to_le_bytes());
    }
    out
}

/// Convert raw 16-bit little-endian PCM bytes to f32 samples normalized to `[-1, 1]`.
///
/// Inverse of [`pcm_f32_to_i16`]. A trailing odd byte (if `bytes.len()` is
/// odd) is ignored, since it cannot form a complete sample.
///
/// This is the canonical conversion for incoming Wyoming `audio-chunk`
/// payloads and other raw PCM input.
pub fn pcm_i16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect()
}

/// Encode f32 PCM samples as an in-memory 16-bit mono WAV buffer.
///
/// Writes a standard RIFF WAV header followed by the PCM data. The
/// returned `Vec<u8>` is a complete, self-contained WAV file and can be
/// sent directly over HTTP or written to disk.
///
/// # Parameters
///
/// * `samples` — f32 audio samples in the range [-1.0, 1.0]. Out-of-range and
///   non-finite values are clamped/silenced.
/// * `info` — Audio format metadata (sample rate, channels, bits per sample).
///   Only 16-bit mono is supported.
///
/// # Errors
///
/// Returns an error if `info` requests a format other than 16-bit mono, or
/// if WAV encoding fails (malformed spec or internal I/O error).
pub fn encode_wav(samples: &[f32], info: &AudioInfo) -> Result<Vec<u8>> {
    anyhow::ensure!(
        info.bits_per_sample == 16,
        "encode_wav only supports 16-bit samples"
    );
    anyhow::ensure!(info.channels == 1, "encode_wav only supports mono audio");
    let mut buf = Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
    let spec = WavSpec {
        channels: info.channels,
        sample_rate: info.sample_rate,
        bits_per_sample: info.bits_per_sample,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::new(&mut buf, spec)?;
    for &s in samples {
        writer.write_sample(f32_to_i16_sample(s))?;
    }
    writer.finalize()?;
    Ok(buf.into_inner())
}

/// Write a waveform tensor to a 16-bit mono PCM WAV file.
///
/// Flattens the tensor (any shape), converts f32 samples, and writes the
/// WAV to `path`. All Crane TTS models produce mono 16-bit audio.
///
/// # Parameters
///
/// * `audio` — Waveform tensor (any shape, any dtype; converted to f32 and
///   flattened).
/// * `path` — Destination file path.
/// * `sample_rate` — Sample rate in Hz (e.g. 24000).
///
/// # Returns
///
/// Returns `path` as a `String` on success.
///
/// # Errors
///
/// Returns an error if tensor conversion, WAV encoding, or file writing fails.
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
        writer.write_sample(f32_to_i16_sample(s))?;
    }
    writer.finalize()?;
    Ok(path.to_string())
}

/// Decode f32 PCM samples (with mono mixdown and resampling) from an open WAV reader.
///
/// Shared by [`load_wav_f32`] (file-backed) and [`decode_wav`] (in-memory).
fn read_wav_f32<R: std::io::Read>(
    mut reader: hound::WavReader<R>,
    target_sr: u32,
) -> Result<Vec<f32>> {
    let spec = reader.spec();
    let raw_sr = spec.sample_rate;

    let samples_f32: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .map(|s| s.map_err(|e| anyhow::anyhow!(e)))
            .collect::<Result<Vec<_>>>()?,
        (SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|s| {
                s.map(|v| v as f32 / 32768.0)
                    .map_err(|e| anyhow::anyhow!(e))
            })
            .collect::<Result<Vec<_>>>()?,
        (SampleFormat::Int, 32) => reader
            .samples::<i32>()
            .map(|s| {
                s.map(|v| v as f32 / 2147483648.0)
                    .map_err(|e| anyhow::anyhow!(e))
            })
            .collect::<Result<Vec<_>>>()?,
        _ => anyhow::bail!(
            "Unsupported WAV format: {:?} {}bit",
            spec.sample_format,
            spec.bits_per_sample
        ),
    };

    // Mix down to mono if stereo
    let mono: Vec<f32> = if spec.channels == 1 {
        samples_f32
    } else {
        let ch = spec.channels as usize;
        samples_f32
            .chunks(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
            .collect()
    };

    // Resample if needed (linear interpolation)
    if raw_sr == target_sr {
        return Ok(mono);
    }

    use audioadapter_buffers::direct::SequentialSliceOfVecs;
    use rubato::{
        Async, FixedAsync, Resampler as RubatoResampler, SincInterpolationParameters,
        SincInterpolationType, WindowFunction, audioadapter::Adapter,
    };

    let ratio = target_sr as f64 / raw_sr as f64;
    let chunk_size = 1024usize;
    let params = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: Some(0.95),
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler =
        Async::<f32>::new_sinc(ratio, 1.0, &params, chunk_size, 1, FixedAsync::Input)?;

    let mut output = Vec::new();
    let mut pos = 0usize;
    while pos < mono.len() {
        let end = (pos + chunk_size).min(mono.len());
        let chunk = &mono[pos..end];
        let data = if chunk.len() < chunk_size {
            let mut padded = chunk.to_vec();
            padded.resize(chunk_size, 0.0);
            padded
        } else {
            chunk.to_vec()
        };

        let input_vecs = vec![data];
        let input = SequentialSliceOfVecs::new(&input_vecs, 1, chunk_size)?;
        let result = resampler.process(&input, None)?;

        let frames = result.frames();
        for i in 0..frames {
            output.push(result.read_sample(0, i).unwrap_or(0.0));
        }
        pos += chunk_size;
    }

    Ok(output)
}

/// Loads a WAV file and returns f32 samples normalized to `[-1, 1]`.
///
/// Mixes down to mono if the file is stereo, and resamples to `target_sr` if
/// needed (high-quality sinc interpolation).
///
/// # Errors
///
/// Returns an error if the file can't be opened, uses an unsupported sample
/// format (only 32-bit float and 16/32-bit integer PCM are supported), or if
/// resampling fails.
pub fn load_wav_f32(path: &str, target_sr: u32) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)?;
    read_wav_f32(reader, target_sr)
}

/// Decodes in-memory WAV bytes and returns f32 samples normalized to `[-1, 1]`.
///
/// Inverse of [`encode_wav`], and the in-memory counterpart of
/// [`load_wav_f32`]. Mixes down to mono if the WAV is stereo, and resamples
/// to `target_sr` if needed (high-quality sinc interpolation).
///
/// # Errors
///
/// Returns an error if `wav_bytes` is not a valid WAV file, uses an
/// unsupported sample format (only 32-bit float and 16/32-bit integer PCM
/// are supported), or if resampling fails.
pub fn decode_wav(wav_bytes: &[u8], target_sr: u32) -> Result<Vec<f32>> {
    let reader = hound::WavReader::new(Cursor::new(wav_bytes))?;
    read_wav_f32(reader, target_sr)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_pcm_rounding() {
        // 0.5 * 32767 = 16383.5, rounds to 16384 rather than truncating to 16383.
        let bytes = pcm_f32_to_i16(&[0.5]);
        assert_eq!(bytes, 16384i16.to_le_bytes().to_vec());
    }

    // --- pcm_i16_to_f32 ---

    #[test]
    fn test_pcm_i16_to_f32_silence() {
        assert_eq!(pcm_i16_to_f32(&[0, 0]), vec![0.0]);
    }

    #[test]
    fn test_pcm_i16_to_f32_max_positive() {
        let bytes = 32767i16.to_le_bytes();
        let samples = pcm_i16_to_f32(&bytes);
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 32767.0 / 32768.0).abs() < 1e-6);
    }

    #[test]
    fn test_pcm_i16_to_f32_max_negative() {
        let bytes = i16::MIN.to_le_bytes();
        let samples = pcm_i16_to_f32(&bytes);
        assert_eq!(samples, vec![-1.0]);
    }

    #[test]
    fn test_pcm_i16_to_f32_empty() {
        assert!(pcm_i16_to_f32(&[]).is_empty());
    }

    #[test]
    fn test_pcm_i16_to_f32_odd_trailing_byte_ignored() {
        // Two full samples (4 bytes) plus one trailing byte that can't form a sample.
        let mut bytes = pcm_f32_to_i16(&[0.0, 0.0]);
        bytes.push(0xFF);
        assert_eq!(pcm_i16_to_f32(&bytes).len(), 2);
    }

    #[test]
    fn test_pcm_i16_to_f32_output_length() {
        let bytes = vec![0u8; 200];
        assert_eq!(pcm_i16_to_f32(&bytes).len(), 100);
    }

    #[test]
    fn test_pcm_roundtrip() {
        let samples = [-1.0f32, -0.5, 0.0, 0.5, 1.0];
        let bytes = pcm_f32_to_i16(&samples);
        let roundtripped = pcm_i16_to_f32(&bytes);
        for (a, b) in samples.iter().zip(roundtripped.iter()) {
            assert!((a - b).abs() < 1e-3, "expected {a}, got {b}");
        }
    }

    // --- encode_wav / save_wav ---

    #[test]
    fn test_encode_wav_roundtrip() -> Result<()> {
        let samples = [0.0f32, 0.5, -0.5, 1.0, -1.0];
        let info = AudioInfo {
            sample_rate: 24000,
            channels: 1,
            bits_per_sample: 16,
        };
        let wav_bytes = encode_wav(&samples, &info)?;

        let mut reader = hound::WavReader::new(Cursor::new(&wav_bytes)).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 24000);
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.bits_per_sample, 16);

        let read_back: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        let expected_bytes = pcm_f32_to_i16(&samples);
        let expected: Vec<i16> = expected_bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(read_back, expected);
        Ok(())
    }

    #[test]
    fn test_encode_wav_rejects_non_16bit() {
        let info = AudioInfo {
            sample_rate: 24000,
            channels: 1,
            bits_per_sample: 24,
        };
        let err = encode_wav(&[0.0], &info).unwrap_err();
        assert!(err.to_string().contains("16-bit"));
    }

    #[test]
    fn test_encode_wav_rejects_non_mono() {
        let info = AudioInfo {
            sample_rate: 24000,
            channels: 2,
            bits_per_sample: 16,
        };
        let err = encode_wav(&[0.0], &info).unwrap_err();
        assert!(err.to_string().contains("mono"));
    }

    #[test]
    fn test_encode_wav_empty() -> Result<()> {
        let info = AudioInfo {
            sample_rate: 44100,
            channels: 1,
            bits_per_sample: 16,
        };
        let wav_bytes = encode_wav(&[], &info)?;

        let mut reader = hound::WavReader::new(Cursor::new(&wav_bytes)).unwrap();
        let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        assert!(samples.is_empty());
        Ok(())
    }

    #[test]
    fn test_encode_wav_matches_pcm_f32_to_i16() -> Result<()> {
        let samples = [-1.0f32, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
        let info = AudioInfo {
            sample_rate: 24000,
            channels: 1,
            bits_per_sample: 16,
        };
        let wav_bytes = encode_wav(&samples, &info)?;

        let mut reader = hound::WavReader::new(Cursor::new(&wav_bytes)).unwrap();
        let from_wav: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();

        let pcm_bytes = pcm_f32_to_i16(&samples);
        let from_pcm: Vec<i16> = pcm_bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        assert_eq!(from_wav, from_pcm);
        Ok(())
    }

    #[test]
    fn test_save_wav_file() -> Result<()> {
        use candle_core::Device;
        let tensor = Tensor::new(&[0.5f32, -0.5, 0.25], &Device::Cpu)?;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap();

        let returned_path = save_wav(&tensor, path, 24000)?;
        assert_eq!(returned_path, path);

        let mut reader = hound::WavReader::open(path).unwrap();
        assert_eq!(reader.spec().sample_rate, 24000);
        let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        assert_eq!(samples.len(), 3);
        Ok(())
    }

    #[test]
    fn test_save_wav_multidim_tensor() -> Result<()> {
        use candle_core::Device;
        // 2D tensor [[0.1, 0.2], [0.3, 0.4]] should flatten to [0.1, 0.2, 0.3, 0.4]
        let tensor = Tensor::new(&[[0.1f32, 0.2], [0.3, 0.4]], &Device::Cpu)?;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap();

        save_wav(&tensor, path, 16000)?;

        let mut reader = hound::WavReader::open(path).unwrap();
        assert_eq!(reader.spec().sample_rate, 16000);
        let samples: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        assert_eq!(samples.len(), 4);
        Ok(())
    }

    // --- decode_wav ---

    #[test]
    fn test_decode_wav_roundtrip_with_encode_wav() -> Result<()> {
        let samples = [0.0f32, 0.5, -0.5, 1.0, -1.0];
        let info = AudioInfo {
            sample_rate: 24000,
            channels: 1,
            bits_per_sample: 16,
        };
        let wav_bytes = encode_wav(&samples, &info)?;

        let decoded = decode_wav(&wav_bytes, 24000)?;
        for (a, b) in samples.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-3, "expected {a}, got {b}");
        }
        Ok(())
    }

    #[test]
    fn test_decode_wav_rejects_invalid_bytes() {
        let err = decode_wav(&[0, 1, 2, 3], 16000);
        assert!(err.is_err());
    }

    #[test]
    fn test_decode_wav_empty() -> Result<()> {
        let info = AudioInfo {
            sample_rate: 16000,
            channels: 1,
            bits_per_sample: 16,
        };
        let wav_bytes = encode_wav(&[], &info)?;
        let decoded = decode_wav(&wav_bytes, 16000)?;
        assert!(decoded.is_empty());
        Ok(())
    }

    #[test]
    fn test_decode_wav_matches_load_wav_f32() -> Result<()> {
        let samples = [0.1f32, -0.2, 0.3, -0.4];
        let info = AudioInfo {
            sample_rate: 16000,
            channels: 1,
            bits_per_sample: 16,
        };
        let wav_bytes = encode_wav(&samples, &info)?;

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &wav_bytes).unwrap();
        let path = tmp.path().to_str().unwrap();

        let from_file = load_wav_f32(path, 16000)?;
        let from_bytes = decode_wav(&wav_bytes, 16000)?;
        assert_eq!(from_file, from_bytes);
        Ok(())
    }
}
