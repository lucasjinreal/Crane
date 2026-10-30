//! Conditioning pipeline for `MuScriptor`.
//!
//! Three parallel conditioners, each of which runs once per 5-second
//! audio chunk:
//!
//! * `MelSpectrogramConditioner` — 16 kHz mono audio → log-magnitude
//!   mel spectrogram (`n_fft`=2048, hop=160 → 100 Hz × 512 bins) →
//!   linear projection to `dim`. Produces ~500 frames per 5-second
//!   chunk (the prefix tokens fed to the transformer).
//! * `ClassConditioner` for `instrument_group` (1000 + 1 pads) —
//!   optional "what instruments are in this recording" signal.
//! * `ClassConditioner` for `dataset_name` (4 + 1 pads) — always
//!   passed as `None` at inference (CFG null condition).
//!
//! The output is a prefix embedding of shape `[B, T, dim]` for each
//! conditioner; the model preconds them to the token sequence before
//! the first transformer call.
//!
//! Mel numerics: uses **magnitude** (not power) spectrum + natural log,
//! matching the upstream's `power=1.0` mel transform. The existing
//! `models::modules::mel::compute_mel_spectrogram` does power spectrum,
//! so this module pulls the same primitives (`build_mel_filterbank`,
//! `hann_window`, `reflect_pad`) but assembles them into a magnitude +
//! log pipeline. Compute stays in fp32; the linear projection output
//! is cast to whatever dtype the transformer wants.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder, embedding, linear};
use rustfft::num_complex::Complex as FftComplex;
use rustfft::{Fft, FftPlanner};

use crate::models::modules::mel::{build_mel_filterbank, hann_window, reflect_pad};

use super::config::{FRAME_RATE_HZ, HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE};

// ── Per-chunk conditioning data ───────────────────────────────────────

#[derive(Debug, Clone)]
pub struct WavCondition {
    /// PCM samples, shape `[B, 1, T]` (mono).
    pub wav: Tensor,
    /// Valid sample count per batch row, shape `[B]`.
    pub length: Tensor,
    /// Sample rates for each batch row (always 16 000 in this model).
    pub sample_rate: Vec<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct ConditioningAttributes {
    /// `(name, wav condition)` pairs in declaration order — the order
    /// matches the order they're prepended to the token sequence.
    pub wav: Vec<(String, WavCondition)>,
    /// `(name, optional space-separated class ids)` pairs.
    pub text: Vec<(String, Option<String>)>,
}

impl ConditioningAttributes {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

// ── MelSpectrogramConditioner ───────────────────────────────────────────

/// 16 kHz mono audio → log-magnitude mel → linear projection to `dim`.
pub struct MelSpectrogramConditioner {
    output_proj: Linear,
    /// Slaney-normalized mel filterbank `[n_mels, n_fft/2 + 1]`,
    /// either loaded from the safetensors checkpoint at
    /// `condition_provider.conditioners.self_wav.mel_spec_transform.mel_scale.fb`
    /// or built deterministically from the Slaney parameters.
    filterbank: Tensor,
    hann: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    device: Device,
    dtype: DType,
    log_eps: f32,
}

impl MelSpectrogramConditioner {
    /// Build from a `VarBuilder` rooted at
    /// `condition_provider.conditioners.self_wav` and a separately
    /// fetched filterbank tensor (the upstream stores it as a
    /// non-parameter buffer at save time, so `VarBuilder::pp(...)`
    /// can't reach it through the `nn.Module` API).
    ///
    /// # Errors
    ///
    /// Returns an error if the `output_proj` weight is missing from `vb` or
    /// doesn't match `(N_MELS, output_dim)`.
    pub fn new(
        output_dim: usize,
        device: &Device,
        dtype: DType,
        vb: &VarBuilder,
        filterbank: Tensor,
    ) -> Result<Self> {
        let output_proj = linear(N_MELS, output_dim, vb.pp("output_proj"))?;
        let hann = hann_window(N_FFT);
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        Ok(Self {
            output_proj,
            filterbank,
            hann,
            fft,
            device: device.clone(),
            dtype,
            log_eps: 1e-6,
        })
    }

    /// Convenience: build the conditioner without a per-checkpoint
    /// filterbank (e.g. in tests), using the deterministic Slaney
    /// construction.
    ///
    /// # Errors
    ///
    /// Returns an error if the synthetic filterbank tensor cannot be built,
    /// or if [`Self::new`] fails to build the `output_proj` weight from `vb`.
    pub fn new_without_filterbank(
        output_dim: usize,
        device: &Device,
        dtype: DType,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let raw = build_mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS, 0.0, 8000.0);
        // `build_mel_filterbank` returns `[n_mels, n_bins]` (row-major,
        // `filters[m * n_bins + k]`), but `mel_for_row`'s matmul — and the
        // real checkpoint's stored `mel_scale.fb` tensor — both expect
        // `[n_bins, n_mels]`. Transpose so this synthetic fallback matches
        // the orientation the real weights load in.
        let fb = Tensor::from_vec(raw, (N_MELS, N_FFT / 2 + 1), &Device::Cpu)?
            .t()?
            .contiguous()?;
        Self::new(output_dim, device, dtype, vb, fb)
    }

    /// Returns `([B, T_mel, dim] embeddings, [B, T_mel] mask)`. Frames
    /// past the chunk's real `length` are zeroed by the post-projection
    /// mask so the prefix conditioner contributes no signal there.
    ///
    /// # Errors
    ///
    /// Returns an error if `cond.wav` isn't a 3-D `[B, 1, T]` tensor, or if
    /// any tensor op in the per-row mel/FFT pipeline fails.
    pub fn forward(&self, cond: &WavCondition) -> Result<(Tensor, Tensor)> {
        let (b, _, _) = cond.wav.dims3()?;
        let wav_cpu = cond.wav.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let wav_cpu = wav_cpu.squeeze(1)?; // [B, T]
        let length_cpu = cond.length.to_device(&Device::Cpu)?;

        // Per-row computation, then stack — different rows can have
        // different `length`s, so per-row loop is unavoidable.
        let mut mel_padded: Vec<Tensor> = Vec::with_capacity(b);
        let mut mask_padded: Vec<Tensor> = Vec::with_capacity(b);
        let mut max_t = 0usize;

        let length_vec: Vec<u32> = length_cpu.to_vec1::<u32>()?;

        for row in 0..b {
            let row_wav = wav_cpu.narrow(0, row, 1)?.squeeze(0)?.to_vec1::<f32>()?;
            let row_len = length_vec[row] as usize;
            let (mel, mask) = self.mel_for_row(&row_wav, row_len)?;
            max_t = max_t.max(mel.dim(0)?);
            mel_padded.push(mel);
            mask_padded.push(mask);
        }

        for (mel, mask) in mel_padded.iter_mut().zip(mask_padded.iter_mut()) {
            let t = mel.dim(0)?;
            if t < max_t {
                let pad = max_t - t;
                let pad_mel = Tensor::zeros((pad, N_MELS), DType::F32, &Device::Cpu)?;
                *mel = Tensor::cat(&[&*mel, &pad_mel], 0)?;
                let pad_mask = Tensor::zeros((pad,), DType::F32, &Device::Cpu)?;
                *mask = Tensor::cat(&[&*mask, &pad_mask], 0)?;
            }
        }

        let mel = Tensor::stack(&mel_padded, 0)?; // [B, T, N_MELS]
        let mask = Tensor::stack(&mask_padded, 0)?; // [B, T]
        let mel = mel.to_device(&self.device)?.to_dtype(self.dtype)?;
        let mask = mask.to_device(&self.device)?.to_dtype(self.dtype)?;

        // `candle_nn::Linear::forward` requires a 2D input — flatten the
        // (B, T) batch dims, then reshape back after the projection.
        let (b, t, _) = mel.dims3()?;
        let mel_flat = mel.reshape((b * t, N_MELS))?;
        let embeds_flat = self.output_proj.forward(&mel_flat)?;
        let embeds = embeds_flat.reshape((b, t, self.output_proj.weight().dim(0)?))?;
        let embeds = embeds.broadcast_mul(&mask.unsqueeze(D::Minus1)?)?;
        Ok((embeds, mask))
    }

    fn mel_for_row(&self, wav: &[f32], length: usize) -> Result<(Tensor, Tensor)> {
        // The upstream stores the filterbank as `[n_bins, n_mels]`
        // (1025 × 512) and multiplies `[T, n_bins] @ [n_bins, n_mels]
        // = [T, n_mels]` directly (no transpose).
        let filterbank = self
            .filterbank
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .contiguous()?;

        let real_len = length.min(wav.len());
        let effective_len = if real_len == 0 { 0 } else { real_len };
        let n_frames = effective_len / HOP_LENGTH;
        let padded = reflect_pad(wav, N_FFT / 2);

        let n_bins = N_FFT / 2 + 1;
        let mut buffer = vec![FftComplex::new(0.0_f32, 0.0_f32); N_FFT];
        let mut all_mag = Vec::with_capacity(n_frames * n_bins);

        for frame_idx in 0..n_frames {
            let start = frame_idx * HOP_LENGTH;
            for (i, b) in buffer.iter_mut().enumerate() {
                let s = *padded.get(start + i).unwrap_or(&0.0);
                *b = FftComplex::new(s * self.hann[i], 0.0);
            }
            self.fft.process(&mut buffer);
            // Magnitude (matches upstream's power=1.0).
            all_mag.extend(
                buffer[..n_bins]
                    .iter()
                    .map(|c| (c.re * c.re + c.im * c.im).sqrt()),
            );
        }

        if n_frames == 0 {
            let empty = Tensor::zeros((0, N_MELS), DType::F32, &Device::Cpu)?;
            let mask = Tensor::zeros((0,), DType::F32, &Device::Cpu)?;
            return Ok((empty, mask));
        }

        let magnitude = Tensor::from_vec(all_mag, (n_frames, n_bins), &Device::Cpu)?;
        let mel = magnitude.matmul(&filterbank)?;
        let eps = Tensor::full(self.log_eps, mel.dims(), &Device::Cpu)?;
        let mel = (mel + eps)?.log()?;

        // Mask: frames beyond `length` are masked. `length` is in
        // samples; convert to frames with `length // HOP_LENGTH`.
        let valid_frames = length / HOP_LENGTH;
        let valid = n_frames.min(valid_frames);
        let mut mask_data = vec![0.0f32; n_frames];
        for v in &mut mask_data[..valid] {
            *v = 1.0;
        }
        let mask = Tensor::from_vec(mask_data, (n_frames,), &Device::Cpu)?;
        Ok((mel, mask))
    }

    #[must_use]
    pub fn frame_rate(&self) -> u32 {
        FRAME_RATE_HZ
    }
}

// ── ClassConditioner ───────────────────────────────────────────────────

/// Class-index conditioner (e.g. instrument group, dataset name). Embeds
/// `(class_id + 1)` so a reserved `pad_idx = 0` slot stays the zero
/// vector under the upstream convention.
pub struct ClassConditioner {
    embed: Embedding,
}

impl ClassConditioner {
    /// # Errors
    ///
    /// Returns an error if the embedding weight is missing from `vb` or
    /// doesn't match `(num_classes + 1, output_dim)`.
    pub fn new(num_classes: usize, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        let embed = embedding(num_classes + 1, output_dim, vb)?;
        Ok(Self { embed })
    }

    /// Build from a directly-supplied embedding weight tensor.
    /// Useful when the weight sits at a path `VarBuilder::pp` can't
    /// reach (e.g. the upstream `MuScriptor` checkpoints store
    /// `condition_provider.conditioners.instrument_group.embed.weight`
    /// as a non-`Parameter` module attribute).
    ///
    /// # Errors
    ///
    /// Returns an error if `weight`'s element count doesn't match
    /// `(num_classes + 1) * output_dim`.
    pub fn from_embedding_tensor(
        num_classes: usize,
        output_dim: usize,
        weight: &Tensor,
    ) -> Result<Self> {
        let embeddings = weight.reshape((num_classes + 1, output_dim))?;
        let embed = Embedding::new(embeddings, output_dim);
        Ok(Self { embed })
    }

    /// `indices` is `[B, L]` of class IDs in `[-1, num_classes]` —
    /// `-1` is the CFG null class and shifts to row 0 internally.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedding lookup fails, e.g. because a
    /// shifted index falls outside `[0, num_classes]`.
    pub fn forward(&self, indices: &Tensor) -> Result<(Tensor, Tensor)> {
        let one = Tensor::ones_like(indices)?;
        let shifted = indices.add(&one)?;
        let embeds = self.embed.forward(&shifted)?;
        // Mask is 1D of length N (number of input tokens); every row
        // is the unconditional "all 1" since CFG doesn't gate class
        // inputs in the upstream.
        let n = embeds.dim(0)?;
        let mask = Tensor::ones((n,), self.embed.embeddings().dtype(), embeds.device())?;
        Ok((embeds, mask))
    }
}

// ── ConditioningProvider ───────────────────────────────────────────────

/// Runs every conditioner and returns a `name → (embed, mask)` map.
pub struct ConditioningProvider {
    mel: HashMap<String, Arc<MelSpectrogramConditioner>>,
    class: HashMap<String, Arc<ClassConditioner>>,
}

impl ConditioningProvider {
    #[must_use]
    pub fn new(
        mel: HashMap<String, Arc<MelSpectrogramConditioner>>,
        class: HashMap<String, Arc<ClassConditioner>>,
    ) -> Self {
        Self { mel, class }
    }

    /// Run all conditioners over a single chunk's attributes.
    ///
    /// # Errors
    ///
    /// Returns an error if any matched conditioner's `forward` fails, or if
    /// parsing a text attribute's class indices fails.
    pub fn forward(
        &self,
        attrs: &ConditioningAttributes,
    ) -> Result<HashMap<String, (Tensor, Tensor)>> {
        let mut out: HashMap<String, (Tensor, Tensor)> = HashMap::new();
        for (name, cond) in &attrs.wav {
            let Some(c) = self.mel.get(name) else {
                continue;
            };
            out.insert(name.clone(), c.forward(cond)?);
        }
        for (name, indices_opt) in &attrs.text {
            let Some(c) = self.class.get(name) else {
                continue;
            };
            let indices_t = tokenize_class_indices(&[indices_opt.clone()])?;
            out.insert(name.clone(), c.forward(&indices_t)?);
        }
        Ok(out)
    }

    /// Force every condition to its null/unconditional value (for CFG
    /// at inference).
    ///
    /// # Errors
    ///
    /// Returns an error if zeroing a wav condition's `wav` or `length`
    /// tensor fails.
    pub fn nullify(&self, attrs: &ConditioningAttributes) -> Result<ConditioningAttributes> {
        let mut out = attrs.clone();
        for (_, cond) in &mut out.wav {
            cond.wav = cond.wav.zeros_like()?;
            cond.length = cond.length.zeros_like()?;
        }
        for (_, txt) in &mut out.text {
            *txt = None;
        }
        Ok(out)
    }
}

/// Convert a list of "space-separated class ids" or `None` to a `[B, L]`
/// tensor. `None` and empty strings both mean "unconditional" → all
/// `-1`s in that row.
fn tokenize_class_indices(values: &[Option<String>]) -> Result<Tensor> {
    let b = values.len();
    if b == 0 {
        return Tensor::zeros((0, 0), DType::I64, &Device::Cpu);
    }
    let mut max_l = 1usize;
    let parsed: Vec<Vec<i64>> = values
        .iter()
        .map(|v| match v {
            None => vec![-1],
            Some(s) => {
                let ids: Vec<i64> = s
                    .split_whitespace()
                    .filter(|t| !t.is_empty())
                    .map(|t| t.parse::<i64>().unwrap_or(-1))
                    .collect();
                if !ids.is_empty() {
                    max_l = max_l.max(ids.len());
                }
                ids
            },
        })
        .collect();
    let mut data = vec![-1i64; b * max_l];
    for (row, vs) in parsed.iter().enumerate() {
        for (col, &v) in vs.iter().enumerate() {
            data[row * max_l + col] = v;
        }
    }
    Tensor::from_vec(data, (b, max_l), &Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn mel_for_silence_has_expected_frame_count() {
        let device = Device::Cpu;
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &device);
        let cond = MelSpectrogramConditioner::new_without_filterbank(64, &device, DType::F32, &vb)
            .unwrap();
        let wav = Tensor::zeros((1, 1, SAMPLE_RATE), DType::F32, &device).unwrap();
        let length = Tensor::from_vec(vec![SAMPLE_RATE as u32], (1,), &device).unwrap();
        let wc = WavCondition {
            wav,
            length,
            sample_rate: vec![SAMPLE_RATE as u32],
        };
        let (emb, mask) = cond.forward(&wc).unwrap();
        let mask_data = mask.to_vec2::<f32>().unwrap();
        // 1s of silence → 100 valid 10ms frames.
        assert_eq!(mask_data[0].iter().filter(|&&v| v > 0.0).count(), 100);
        let emb_dims = emb.dims();
        assert_eq!(emb_dims, &[1, 100, 64]);
    }

    #[test]
    fn class_conditioner_handles_null_class() {
        let device = Device::Cpu;
        let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &device);
        let c = ClassConditioner::new(10, 8, vb).unwrap();
        // -1 maps to the pad row (zero vector).
        let indices = Tensor::from_vec(vec![-1i64, 0, 5], (1, 3), &device).unwrap();
        let (emb, _) = c.forward(&indices).unwrap();
        assert_eq!(emb.dims(), &[1, 3, 8]);
    }
}
