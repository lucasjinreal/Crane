//! CosyVoice2's `HiFTGenerator` — HiFTNet (Neural Source Filter + ISTFTNet):
//! turns [`super::flow::Flow`]'s mel spectrogram into a waveform. Ported
//! from `stepaudio2.flashcosyvoice.modules.hifigan[_components]` (real
//! reference source — see the plan doc). The largest single component in
//! this phase.
//!
//! **Not bit-reproducible against Python**, unlike every other Token2wav
//! stage: `SourceModuleHnNSF2`'s harmonic-sine source synthesis draws real
//! random phase/noise at inference time (its own `torch.rand`/`torch.randn`
//! calls, not the checkpoint's `rand_noise` buffer trick `cfm.rs` deals
//! with). Validated via shape/finite-range/spectral sanity instead of an
//! exact HF-diff match — see the module's test.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{conv1d, linear, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, VarBuilder};
use rustfft::{num_complex::Complex as FftComplex, Fft, FftPlanner};

use crate::models::modules::mel::{hann_window, reflect_pad};
use crate::models::voxtral_tts::codec::reconstruct_weight_norm;

const SAMPLING_RATE: usize = 24_000;
const BASE_CHANNELS: usize = 512;
const IN_CHANNELS: usize = 80; // mel channels
const NB_HARMONICS: usize = 8;
const NSF_ALPHA: f32 = 0.1; // sine amplitude
const NSF_SIGMA: f32 = 0.003; // additive noise std
const NSF_VOICED_THRESHOLD: f32 = 10.0;
const UPSAMPLE_RATES: [usize; 3] = [8, 5, 3];
const UPSAMPLE_KERNELS: [usize; 3] = [16, 11, 7];
const ISTFT_N_FFT: usize = 16;
const ISTFT_HOP: usize = 4;
const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
const RESBLOCK_DILATIONS: [[usize; 3]; 3] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]];
const SOURCE_RESBLOCK_KERNELS: [usize; 3] = [7, 7, 11];
const LRELU_SLOPE: f64 = 0.1;
const AUDIO_LIMIT: f32 = 0.99;

fn get_padding(kernel_size: usize, dilation: usize) -> usize {
    (kernel_size * dilation - dilation) / 2
}

fn weight_norm_conv1d(in_ch: usize, out_ch: usize, kernel: usize, cfg: Conv1dConfig, vb: VarBuilder) -> Result<Conv1d> {
    let g = vb.get((out_ch, 1, 1), "parametrizations.weight.original0")?;
    let v = vb.get((out_ch, in_ch, kernel), "parametrizations.weight.original1")?;
    let weight = reconstruct_weight_norm(&v, &g)?;
    let bias = vb.get(out_ch, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), cfg))
}

fn weight_norm_conv_transpose1d(in_ch: usize, out_ch: usize, kernel: usize, cfg: ConvTranspose1dConfig, vb: VarBuilder) -> Result<ConvTranspose1d> {
    // PyTorch ConvTranspose1d weight shape: [in_ch, out_ch, kernel] — weight_norm dim=0 is still `in_ch` (the first dim).
    let g = vb.get((in_ch, 1, 1), "parametrizations.weight.original0")?;
    let v = vb.get((in_ch, out_ch, kernel), "parametrizations.weight.original1")?;
    let weight = reconstruct_weight_norm(&v, &g)?;
    let bias = vb.get(out_ch, "bias")?;
    Ok(ConvTranspose1d::new(weight, Some(bias), cfg))
}

// ── Snake activation ─────────────────────────────────────────────────────

struct Snake {
    alpha: Tensor,
}

impl Snake {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { alpha: vb.get(channels, "alpha")?.reshape((1, channels, 1))? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let sin = x.broadcast_mul(&self.alpha)?.sin()?;
        let sin2 = sin.sqr()?;
        let recip = (&self.alpha + 1e-9)?.recip()?;
        x + sin2.broadcast_mul(&recip)?
    }
}

// ── ResBlock ──────────────────────────────────────────────────────────────

struct ResBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    act1: Vec<Snake>,
    act2: Vec<Snake>,
}

impl ResBlock {
    fn new(channels: usize, kernel: usize, dilations: &[usize], vb: VarBuilder) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(dilations.len());
        let mut convs2 = Vec::with_capacity(dilations.len());
        let mut act1 = Vec::with_capacity(dilations.len());
        let mut act2 = Vec::with_capacity(dilations.len());
        for (i, &dilation) in dilations.iter().enumerate() {
            let cfg1 = Conv1dConfig { padding: get_padding(kernel, dilation), stride: 1, dilation, ..Default::default() };
            convs1.push(weight_norm_conv1d(channels, channels, kernel, cfg1, vb.pp("convs1").pp(i))?);
            let cfg2 = Conv1dConfig { padding: get_padding(kernel, 1), stride: 1, dilation: 1, ..Default::default() };
            convs2.push(weight_norm_conv1d(channels, channels, kernel, cfg2, vb.pp("convs2").pp(i))?);
            act1.push(Snake::new(channels, vb.pp("activations1").pp(i))?);
            act2.push(Snake::new(channels, vb.pp("activations2").pp(i))?);
        }
        Ok(Self { convs1, convs2, act1, act2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for i in 0..self.convs1.len() {
            let xt = self.act1[i].forward(&x)?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = self.act2[i].forward(&xt)?;
            let xt = self.convs2[i].forward(&xt)?;
            x = (xt + x)?;
        }
        Ok(x)
    }
}

// ── F0 predictor ──────────────────────────────────────────────────────────

struct ConvRNNF0Predictor {
    convs: Vec<Conv1d>,
    classifier: Linear,
}

impl ConvRNNF0Predictor {
    fn new(in_channels: usize, cond_channels: usize, vb: VarBuilder) -> Result<Self> {
        let vb_net = vb.pp("condnet");
        let mut convs = Vec::with_capacity(5);
        let dims = [in_channels, cond_channels, cond_channels, cond_channels, cond_channels, cond_channels];
        for i in 0..5 {
            let cfg = Conv1dConfig { padding: 1, stride: 1, ..Default::default() };
            convs.push(weight_norm_conv1d(dims[i], dims[i + 1], 3, cfg, vb_net.pp(2 * i))?);
        }
        let classifier = linear(cond_channels, 1, vb.pp("classifier"))?;
        Ok(Self { convs, classifier })
    }

    /// `x`: `[1, 80, T]`. Returns `f0`: `[1, T]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut xs = x.clone();
        for conv in &self.convs {
            xs = elu(&conv.forward(&xs)?)?;
        }
        let xs = xs.transpose(1, 2)?.contiguous()?; // [1, T, cond_channels]
        self.classifier.forward(&xs)?.squeeze(D::Minus1)?.abs()
    }
}

/// `ELU(x) = x if x > 0 else exp(x) - 1` (alpha=1.0, `nn.ELU()`'s default).
/// Clamps the `exp` branch's input to `<= 0` first so a large positive `x`
/// can't overflow `exp` into `inf` before being masked out (`inf * 0 = NaN`,
/// which would corrupt the result despite mathematically being 0).
fn elu(x: &Tensor) -> Result<Tensor> {
    let positive = x.relu()?;
    let clamped_negative = (x - &positive)?; // min(x, 0) = x - max(x, 0)
    let negative_mask = x.le(0f64)?.to_dtype(x.dtype())?;
    let negative = ((clamped_negative.exp()? - 1.0)? * negative_mask)?;
    positive + negative
}

// ── Harmonic source (SineGen2 / SourceModuleHnNSF2) ──────────────────────

/// `f0`: `[1, T, 1]` Hz, already upsampled to waveform sample rate.
/// `harmonic_num`: 8. `upsample_scale`: 480 (`prod(upsample_rates)*hop_len`).
///
/// Returns sine waves `[1, T, harmonic_num+1]` (fundamental + harmonics) —
/// direct port of `SineGen2._f02sine`'s interpolate-down/cumsum/
/// interpolate-up trick (keeps phase accumulation at low mel-rate to avoid
/// precision loss over long sequences), **including its real
/// randomness** (`rand_ini`, the additive noise term) — not reproducible
/// against a specific Python run, see the module doc.
fn f0_to_sines(f0: &Tensor, harmonic_num: usize, upsample_scale: usize, sine_amp: f32, noise_std: f32, voiced_threshold: f32) -> Result<Tensor> {
    let device = f0.device();
    let dtype = f0.dtype();
    let (b, t, _one) = f0.dims3()?;
    let dim = harmonic_num + 1;

    let harmonics = Tensor::arange(1u32, (dim + 1) as u32, device)?.to_dtype(dtype)?; // [dim]
    let fn_ = f0.broadcast_mul(&harmonics.reshape((1, 1, dim))?)?; // [b, t, dim]

    let mut rad_values = (fn_ / SAMPLING_RATE as f64)?;
    rad_values = (&rad_values - rad_values.floor()?)?; // mod 1

    let rand_ini = Tensor::rand(0f32, 1f32, (b, dim), device)?.to_dtype(dtype)?;
    let rand_ini = rand_ini.slice_assign(&[0..b, 0..1], &Tensor::zeros((b, 1), dtype, device)?)?;
    let first = (rad_values.narrow(1, 0, 1)? + rand_ini.unsqueeze(1)?)?;
    rad_values = rad_values.slice_assign(&[0..b, 0..1, 0..dim], &first)?;

    // downsample -> cumsum -> upsample
    let rad_t = rad_values.transpose(1, 2)?.contiguous()?; // [b, dim, t]
    let down_len = t / upsample_scale;
    let rad_down = linear_interpolate_1d(&rad_t, down_len)?; // [b, dim, down_len]
    let rad_down = rad_down.transpose(1, 2)?.contiguous()?; // [b, down_len, dim]

    let phase_down = (cumsum_dim1(&rad_down)? * (2.0 * std::f64::consts::PI))?;
    let phase_down_t = phase_down.transpose(1, 2)?.contiguous()?; // [b, dim, down_len]
    let phase_up = linear_interpolate_1d(&phase_down_t, t)?; // [b, dim, t]
    let phase = phase_up.transpose(1, 2)?.contiguous()?; // [b, t, dim]
    let sines = phase.sin()?;

    let uv = f0.gt(f64::from(voiced_threshold))?.to_dtype(dtype)?; // [b, t, 1]
    let sine_waves = (sines * f64::from(sine_amp))?;

    let noise_amp = (uv.affine(f64::from(noise_std) - f64::from(sine_amp) / 3.0, f64::from(sine_amp) / 3.0))?;
    let noise = Tensor::randn(0f32, 1f32, (b, t, dim), device)?.to_dtype(dtype)?.broadcast_mul(&noise_amp)?;

    (sine_waves.broadcast_mul(&uv))? + noise
}

/// Nearest-neighbor-free linear interpolation along the last dim,
/// `align_corners=False` (`F.interpolate(..., mode="linear")`'s default),
/// matching PyTorch's sampling-coordinate convention
/// `src = (dst + 0.5) * (src_len / dst_len) - 0.5`.
fn linear_interpolate_1d(x: &Tensor, out_len: usize) -> Result<Tensor> {
    let (b, c, in_len) = x.dims3()?;
    if out_len == in_len {
        return x.contiguous();
    }
    let scale = in_len as f64 / out_len as f64;
    let flat: Vec<f32> = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let mut out = vec![0f32; b * c * out_len];
    for bi in 0..b {
        for ci in 0..c {
            let src_row = &flat[(bi * c + ci) * in_len..(bi * c + ci + 1) * in_len];
            let dst_row = &mut out[(bi * c + ci) * out_len..(bi * c + ci + 1) * out_len];
            for (o, dst) in dst_row.iter_mut().enumerate() {
                let src = ((o as f64 + 0.5) * scale - 0.5).max(0.0).min((in_len - 1) as f64);
                let i0 = src.floor() as usize;
                let i1 = (i0 + 1).min(in_len - 1);
                let frac = (src - i0 as f64) as f32;
                *dst = src_row[i0] * (1.0 - frac) + src_row[i1] * frac;
            }
        }
    }
    Tensor::from_vec(out, (b, c, out_len), x.device())?.to_dtype(x.dtype())
}

/// Cumulative sum along dim 1 (the time axis of a `[b, t, dim]` tensor).
fn cumsum_dim1(x: &Tensor) -> Result<Tensor> {
    x.cumsum(1)
}

struct SourceModule {
    l_linear: Linear,
    sine_amp: f32,
    noise_std: f32,
    upsample_scale: usize,
}

impl SourceModule {
    fn new(harmonic_num: usize, sine_amp: f32, noise_std: f32, upsample_scale: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { l_linear: linear(harmonic_num + 1, 1, vb.pp("l_linear"))?, sine_amp, noise_std, upsample_scale })
    }

    /// `f0`: `[1, T, 1]` (already upsampled to waveform rate). Returns `[1, T, 1]`.
    fn forward(&self, f0: &Tensor) -> Result<Tensor> {
        let sine_wavs = f0_to_sines(f0, NB_HARMONICS, self.upsample_scale, self.sine_amp, self.noise_std, NSF_VOICED_THRESHOLD)?;
        self.l_linear.forward(&sine_wavs)?.tanh()
    }
}

// ── STFT / ISTFT (n_fft=16, hop=4, Hann, center=True, one-sided) ────────

struct StftIstft {
    window: Vec<f32>,
    fft_fwd: std::sync::Arc<dyn Fft<f32>>,
    fft_inv: std::sync::Arc<dyn Fft<f32>>,
    n_fft: usize,
    hop: usize,
}

impl StftIstft {
    fn new(n_fft: usize, hop: usize) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        Self { window: hann_window(n_fft), fft_fwd: planner.plan_fft_forward(n_fft), fft_inv: planner.plan_fft_inverse(n_fft), n_fft, hop }
    }

    /// `samples`: mono waveform. Returns `(real, imag)`, each `[n_bins, n_frames]`, `n_bins = n_fft/2+1`.
    fn stft(&self, samples: &[f32]) -> (Vec<f32>, Vec<f32>, usize) {
        let n_bins = self.n_fft / 2 + 1;
        let padded = reflect_pad_both(samples, self.n_fft / 2);
        let n_frames = (padded.len() - self.n_fft) / self.hop + 1;
        let mut real = vec![0f32; n_bins * n_frames];
        let mut imag = vec![0f32; n_bins * n_frames];
        let mut buf = vec![FftComplex::new(0.0, 0.0); self.n_fft];
        for frame in 0..n_frames {
            let start = frame * self.hop;
            for i in 0..self.n_fft {
                buf[i] = FftComplex::new(padded[start + i] * self.window[i], 0.0);
            }
            self.fft_fwd.process(&mut buf);
            for bin in 0..n_bins {
                real[bin * n_frames + frame] = buf[bin].re;
                imag[bin * n_frames + frame] = buf[bin].im;
            }
        }
        (real, imag, n_frames)
    }

    /// `magnitude`/`phase`: `[n_bins, n_frames]`. Returns the reconstructed waveform.
    fn istft(&self, magnitude: &[f32], phase: &[f32], n_frames: usize) -> Vec<f32> {
        let n_bins = self.n_fft / 2 + 1;
        let padded_len = (n_frames - 1) * self.hop + self.n_fft;
        let mut out = vec![0f32; padded_len];
        let mut win_sum = vec![0f32; padded_len];
        let mut spectrum = vec![FftComplex::new(0.0, 0.0); self.n_fft];

        for frame in 0..n_frames {
            for bin in 0..n_bins {
                let idx = bin * n_frames + frame;
                let (mag, ph) = (magnitude[idx].min(1e2), phase[idx]);
                spectrum[bin] = FftComplex::new(mag * ph.cos(), mag * ph.sin());
            }
            // Hermitian symmetry to fill the full n_fft-length spectrum.
            for bin in n_bins..self.n_fft {
                spectrum[bin] = spectrum[self.n_fft - bin].conj();
            }
            let mut buf = spectrum.clone();
            self.fft_inv.process(&mut buf);
            let start = frame * self.hop;
            for i in 0..self.n_fft {
                // rustfft's inverse FFT is unnormalized — divide by n_fft.
                out[start + i] += buf[i].re / self.n_fft as f32 * self.window[i];
                win_sum[start + i] += self.window[i] * self.window[i];
            }
        }
        for i in 0..padded_len {
            if win_sum[i] > 1e-11 {
                out[i] /= win_sum[i];
            }
        }
        // center=True: trim n_fft/2 from each end.
        let half = self.n_fft / 2;
        out[half..padded_len - half].to_vec()
    }
}

fn reflect_pad_both(samples: &[f32], pad: usize) -> Vec<f32> {
    let left = reflect_pad(samples, pad);
    let mut right_source = samples.to_vec();
    right_source.reverse();
    let mut out = left;
    // Mirror the tail the same way `reflect_pad` mirrors the head (skip the
    // boundary sample itself, matching `torch.nn.functional.pad`'s "reflect" mode).
    for i in 0..pad {
        out.push(samples[samples.len() - 2 - i]);
    }
    out
}

// ── Full generator ────────────────────────────────────────────────────────

pub struct HiFTGenerator {
    f0_predictor: ConvRNNF0Predictor,
    m_source: SourceModule,
    conv_pre: Conv1d,
    ups: Vec<ConvTranspose1d>,
    source_downs: Vec<Conv1d>,
    source_resblocks: Vec<ResBlock>,
    resblocks: Vec<ResBlock>,
    conv_post: Conv1d,
    stft_istft: StftIstft,
}

impl HiFTGenerator {
    pub fn new(model_path: &str, device: &Device, dtype: DType) -> Result<Self> {
        let path = format!("{model_path}/assets/token2wav/hift.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device) }?;

        let f0_predictor = ConvRNNF0Predictor::new(IN_CHANNELS, BASE_CHANNELS, vb.pp("f0_predictor"))?;
        let upsample_scale = UPSAMPLE_RATES.iter().product::<usize>() * ISTFT_HOP;
        let m_source = SourceModule::new(NB_HARMONICS, NSF_ALPHA, NSF_SIGMA, upsample_scale, vb.pp("m_source"))?;

        let conv_pre_cfg = Conv1dConfig { padding: 3, stride: 1, ..Default::default() };
        let conv_pre = weight_norm_conv1d(IN_CHANNELS, BASE_CHANNELS, 7, conv_pre_cfg, vb.pp("conv_pre"))?;

        let mut ups = Vec::with_capacity(3);
        for (i, (&u, &k)) in UPSAMPLE_RATES.iter().zip(UPSAMPLE_KERNELS.iter()).enumerate() {
            let in_ch = BASE_CHANNELS / 2usize.pow(i as u32);
            let out_ch = BASE_CHANNELS / 2usize.pow(i as u32 + 1);
            let cfg = ConvTranspose1dConfig { padding: (k - u) / 2, stride: u, output_padding: 0, dilation: 1, groups: 1 };
            ups.push(weight_norm_conv_transpose1d(in_ch, out_ch, k, cfg, vb.pp("ups").pp(i))?);
        }

        let downsample_rates: Vec<usize> = {
            let mut v = vec![1usize];
            v.extend(UPSAMPLE_RATES.iter().rev().take(UPSAMPLE_RATES.len() - 1));
            v
        };
        let mut cum = 1usize;
        let downsample_cum_rates: Vec<usize> = downsample_rates.iter().map(|&r| { cum *= r; cum }).collect();
        let downsample_cum_rates_rev: Vec<usize> = downsample_cum_rates.iter().rev().copied().collect();

        let mut source_downs = Vec::with_capacity(3);
        let mut source_resblocks = Vec::with_capacity(3);
        for i in 0..3 {
            let out_ch = BASE_CHANNELS / 2usize.pow(i as u32 + 1);
            let u = downsample_cum_rates_rev[i];
            let conv = if u == 1 {
                let cfg = Conv1dConfig { padding: 0, stride: 1, ..Default::default() };
                conv1d(ISTFT_N_FFT + 2, out_ch, 1, cfg, vb.pp("source_downs").pp(i))?
            } else {
                let cfg = Conv1dConfig { padding: u / 2, stride: u, ..Default::default() };
                conv1d(ISTFT_N_FFT + 2, out_ch, u * 2, cfg, vb.pp("source_downs").pp(i))?
            };
            source_downs.push(conv);
            source_resblocks.push(ResBlock::new(out_ch, SOURCE_RESBLOCK_KERNELS[i], &RESBLOCK_DILATIONS[i], vb.pp("source_resblocks").pp(i))?);
        }

        let mut resblocks = Vec::with_capacity(3 * RESBLOCK_KERNELS.len());
        for i in 0..3 {
            let ch = BASE_CHANNELS / 2usize.pow(i as u32 + 1);
            for (j, &k) in RESBLOCK_KERNELS.iter().enumerate() {
                resblocks.push(ResBlock::new(ch, k, &RESBLOCK_DILATIONS[j], vb.pp("resblocks").pp(i * RESBLOCK_KERNELS.len() + j))?);
            }
        }

        let last_ch = BASE_CHANNELS / 2usize.pow(3);
        let conv_post_cfg = Conv1dConfig { padding: 3, stride: 1, ..Default::default() };
        let conv_post = weight_norm_conv1d(last_ch, ISTFT_N_FFT + 2, 7, conv_post_cfg, vb.pp("conv_post"))?;

        Ok(Self {
            f0_predictor,
            m_source,
            conv_pre,
            ups,
            source_downs,
            source_resblocks,
            resblocks,
            conv_post,
            stft_istft: StftIstft::new(ISTFT_N_FFT, ISTFT_HOP),
        })
    }

    /// `mel`: `[1, 80, T]`. Returns mono waveform samples (`T * 480` long).
    ///
    /// # Errors
    ///
    /// Returns an error if any forward pass fails.
    pub fn forward(&self, mel: &Tensor) -> Result<Vec<f32>> {
        let device = mel.device();
        let dtype = mel.dtype();

        let f0 = self.f0_predictor.forward(mel)?; // [1, T]
        let upsample_scale = UPSAMPLE_RATES.iter().product::<usize>() * ISTFT_HOP;
        let t_mel = f0.dim(1)?;
        let f0_flat: Vec<f32> = f0.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let mut f0_up = vec![0f32; t_mel * upsample_scale];
        for (i, v) in f0_flat.iter().enumerate() {
            f0_up[i * upsample_scale..(i + 1) * upsample_scale].fill(*v);
        }
        let f0_up = Tensor::from_vec(f0_up, (1, t_mel * upsample_scale, 1), device)?.to_dtype(dtype)?;

        let source = self.m_source.forward(&f0_up)?; // [1, T_wave, 1]
        let source_samples: Vec<f32> = source.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        let (s_real, s_imag, n_frames) = self.stft_istft.stft(&source_samples);
        let s_stft = {
            let n_bins = ISTFT_N_FFT / 2 + 1;
            let mut data = vec![0f32; 2 * n_bins * n_frames];
            data[..n_bins * n_frames].copy_from_slice(&s_real);
            data[n_bins * n_frames..].copy_from_slice(&s_imag);
            Tensor::from_vec(data, (1, 2 * n_bins, n_frames), device)?.to_dtype(dtype)?
        };

        let mut x = self.conv_pre.forward(mel)?;
        for i in 0..3 {
            x = candle_nn::ops::leaky_relu(&x, LRELU_SLOPE)?;
            x = self.ups[i].forward(&x)?;
            if i == 2 {
                x = reflection_pad_left1(&x)?;
            }

            let si = self.source_downs[i].forward(&s_stft)?;
            let si = self.source_resblocks[i].forward(&si)?;
            let si = crop_or_pad_time(&si, x.dim(2)?)?;
            x = (x + si)?;

            let mut xs: Option<Tensor> = None;
            for j in 0..RESBLOCK_KERNELS.len() {
                let out = self.resblocks[i * RESBLOCK_KERNELS.len() + j].forward(&x)?;
                xs = Some(match xs {
                    Some(acc) => (acc + out)?,
                    None => out,
                });
            }
            x = (xs.unwrap() / RESBLOCK_KERNELS.len() as f64)?;
        }

        x = candle_nn::ops::leaky_relu(&x, 1.0)?; // default negative_slope=0.01 in torch's F.leaky_relu()
        x = self.conv_post.forward(&x)?;

        let n_bins = ISTFT_N_FFT / 2 + 1;
        let magnitude = x.narrow(1, 0, n_bins)?.exp()?;
        let phase = x.narrow(1, n_bins, n_bins)?.sin()?;
        let (out_frames,) = (magnitude.dim(2)?,);

        let mag_flat: Vec<f32> = magnitude.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let phase_flat: Vec<f32> = phase.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let waveform = self.stft_istft.istft(&mag_flat, &phase_flat, out_frames);

        Ok(waveform.into_iter().map(|v| v.clamp(-AUDIO_LIMIT, AUDIO_LIMIT)).collect())
    }
}

/// `nn.ReflectionPad1d((1, 0))` — pad 1 sample on the left only, mirroring
/// (not duplicating) the second sample.
fn reflection_pad_left1(x: &Tensor) -> Result<Tensor> {
    let t = x.dim(2)?;
    let mirror = x.narrow(2, 1, 1)?;
    Tensor::cat(&[&mirror, &x.narrow(2, 0, t)?], 2)
}

/// The source branch and the main upsampling branch can differ by a
/// handful of frames (STFT framing vs `ConvTranspose1d` framing round
/// differently) — narrow (or zero-pad) the source branch to match exactly,
/// matching PyTorch's implicit broadcasting-by-equal-shape assumption (the
/// reference code assumes they already match; this keeps Rust's stricter
/// shape checking happy without changing the math when they do).
fn crop_or_pad_time(x: &Tensor, target: usize) -> Result<Tensor> {
    let t = x.dim(2)?;
    if t == target {
        x.contiguous()
    } else if t > target {
        x.narrow(2, 0, target)
    } else {
        x.pad_with_zeros(2, 0, target - t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── HF-diff: F0 predictor (deterministic, no randomness) ──────────
    //
    // Run with:
    // `cargo test -p crane-core --release -- --ignored --nocapture minicpmo_f0_predictor_matches_python`
    #[test]
    #[ignore = "needs the real MiniCPM-o-4.5 checkpoint + a matching HF dump"]
    fn minicpmo_f0_predictor_matches_python() {
        let model_path = "/home/hahihula/mywork/ai/additional_models/MiniCPM-o-4_5";
        let diff_dir = "/home/hahihula/.claude/jobs/02aa7312/tmp/minicpmo_diff";
        let device = Device::Cpu;
        let dtype = DType::F32;

        let hift = HiFTGenerator::new(model_path, &device, dtype).expect("load hift");

        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{diff_dir}/hift_meta.json")).unwrap()).unwrap();
        let mel_shape: Vec<usize> = meta["mel_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let mel_bytes = std::fs::read(format!("{diff_dir}/hift_mel_input.bin")).unwrap();
        let mel_flat: Vec<f32> = mel_bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let mel = Tensor::from_vec(mel_flat, mel_shape.as_slice(), &device).unwrap();

        let f0 = hift.f0_predictor.forward(&mel).expect("f0 forward");
        let rust_f0: Vec<f32> = f0.flatten_all().unwrap().to_vec1().unwrap();

        let py_bytes = std::fs::read(format!("{diff_dir}/hift_f0_output.bin")).unwrap();
        let py_f0: Vec<f32> = py_bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        assert_eq!(rust_f0.len(), py_f0.len());

        let max_abs_diff = rust_f0.iter().zip(&py_f0).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let dot: f64 = rust_f0.iter().zip(&py_f0).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
        let norm_a: f64 = rust_f0.iter().map(|a| f64::from(*a) * f64::from(*a)).sum::<f64>().sqrt();
        let norm_b: f64 = py_f0.iter().map(|b| f64::from(*b) * f64::from(*b)).sum::<f64>().sqrt();
        let cosine = dot / (norm_a * norm_b);
        println!("f0 cosine similarity: {cosine}, max abs diff: {max_abs_diff}");
        assert!(cosine > 0.999, "f0 cosine similarity too low: {cosine}");
    }

    // ── Shape/finite/range sanity for the full generator ───────────────
    //
    // `SourceModuleHnNSF2` draws real randomness at inference time (see the
    // module doc) so this can't be an exact HF-diff — checks the waveform
    // is well-formed instead. Uses a real `Flow::inference` output as the
    // mel input for a realistic end-to-end shape check.
    #[test]
    #[ignore = "needs the real MiniCPM-o-4.5 checkpoint + a matching HF dump"]
    fn minicpmo_hifigan_produces_well_formed_waveform() {
        use crate::models::minicpmo::token2wav::{Flow, SystemDefaultPrompt};

        let model_path = "/home/hahihula/mywork/ai/additional_models/MiniCPM-o-4_5";
        let diff_dir = "/home/hahihula/.claude/jobs/02aa7312/tmp/minicpmo_diff";
        let device = Device::Cpu;
        let dtype = DType::F32;

        let flow = Flow::new(model_path, &device, dtype).expect("load flow");
        let hift = HiFTGenerator::new(model_path, &device, dtype).expect("load hift");
        let prompt = SystemDefaultPrompt::load(model_path, &device, dtype).expect("load system default prompt");

        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{diff_dir}/flow_meta.json")).unwrap()).unwrap();
        let gen_tokens: Vec<i64> = meta["gen_tokens"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        let token = Tensor::from_vec(gen_tokens.clone(), (1, gen_tokens.len()), &device).unwrap();

        let noise_shape: Vec<usize> = meta["noise_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let noise_bytes = std::fs::read(format!("{diff_dir}/flow_noise.bin")).unwrap();
        let noise_flat: Vec<f32> = noise_bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let noise = Tensor::from_vec(noise_flat, noise_shape.as_slice(), &device).unwrap();

        let n_timesteps = meta["n_timesteps"].as_u64().unwrap() as usize;
        let feat = flow
            .inference(&token, &prompt.prompt_token, &prompt.prompt_feat, &prompt.spk_emb, &noise, n_timesteps)
            .expect("flow inference");

        let waveform = hift.forward(&feat).expect("hift forward");
        let mel_len = feat.dim(2).unwrap();
        println!("mel frames: {mel_len}, waveform samples: {}", waveform.len());

        assert_eq!(waveform.len(), mel_len * 480, "waveform length should be mel_frames * hop_size(480)");
        assert!(waveform.iter().all(|v| v.is_finite()), "non-finite sample in waveform");
        assert!(waveform.iter().all(|v| (-1.0..=1.0).contains(v)), "sample outside clamp range");
        let max_abs = waveform.iter().fold(0f32, |a, &b| a.max(b.abs()));
        assert!(max_abs > 0.001, "waveform looks like near-silence (max_abs={max_abs})");
    }
}
