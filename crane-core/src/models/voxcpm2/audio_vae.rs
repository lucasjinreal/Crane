//! AudioVAE V2 decoder: a DAC-style causal-conv decoder (Snake activations,
//! weight-normed convs, sample-rate-conditioned super-resolution to 48kHz).
//! **Decode-only** — no encoder, no streaming state — since zero-shot
//! generation never needs `AudioVAE.encode()` (no reference audio) and this
//! pass doesn't implement streaming. Port of `audiovae/audio_vae_v2.py`'s
//! `CausalDecoder`/`CausalDecoderBlock`/`CausalResidualUnit`/
//! `SampleRateConditionLayer`/`Snake1d` — `AudioVAE.decode()`'s call chain,
//! traced precisely against the real checkpoint's tensor names/shapes
//! (`decoder.model.{0..9}`, `decoder.sr_cond_model.{2..7}`) since the
//! `depthwise=True` default changes which convs are grouped in a way that
//! isn't obvious from the config alone (see module-level comments below).
//!
//! Weights ship in `audiovae.pth` (PyTorch pickle) — converted once to
//! `audiovae.safetensors` via a small Python script (`safetensors.torch.save_file`
//! after `torch.load(..., weights_only=True)`); candle has no pickle reader.
//! **Zero-padding**, not reflect-padding: this checkpoint's `CausalConv1d.forward`
//! is `F.pad(x, (left_pad, 0))` with no `mode=` argument, i.e. PyTorch's default
//! `mode="constant", value=0` — different from `voxtral_tts::codec`'s causal
//! convs (which reflect-pad), so this module writes its own causal conv
//! wrappers rather than reusing that struct; only the dtype-independent
//! `reconstruct_weight_norm` math is shared.

use candle_core::{Result, Tensor};
use candle_nn::{embedding, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Embedding, Module, VarBuilder};

use crate::models::voxtral_tts::codec::reconstruct_weight_norm;

// ── Weight-normed causal convs (zero-padding) ───────────────────────────

/// Weight-normed `Conv1d`, manually zero-left-padded by `2*padding -
/// output_padding` before an otherwise-unpadded conv — matches
/// `CausalConv1d.forward`'s `F.pad(x, (padding*2 - output_padding, 0))`.
struct CausalConv1d {
    conv: Conv1d,
    left_pad: usize,
}

impl CausalConv1d {
    #[allow(clippy::too_many_arguments)]
    fn load(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        padding: usize,
        output_padding: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight_v = vb.get((out_ch, in_ch / groups, kernel), "weight_v")?;
        let weight_g = vb.get((out_ch, 1, 1), "weight_g")?;
        let weight = reconstruct_weight_norm(&weight_v, &weight_g)?;
        let bias = vb.get(out_ch, "bias")?;
        let cfg = Conv1dConfig { padding: 0, stride, dilation, groups, cudnn_fwd_algo: None };
        let conv = candle_nn::Conv1d::new(weight, Some(bias), cfg);
        Ok(Self { conv, left_pad: padding * 2 - output_padding })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if self.left_pad > 0 { x.pad_with_zeros(2, self.left_pad, 0)? } else { x.clone() };
        self.conv.forward(&x)
    }
}

/// Weight-normed `ConvTranspose1d` run with `padding=0, output_padding=0`
/// (matching how `CausalTransposeConv1d.__init__` never forwards its own
/// `padding`/`output_padding` args to the underlying `nn.ConvTranspose1d` —
/// they're consumed for the *right-trim* below instead), then right-trimmed
/// by `2*padding - output_padding` — matches `CausalTransposeConv1d.forward`.
struct CausalConvTranspose1d {
    conv: ConvTranspose1d,
    trim: usize,
}

impl CausalConvTranspose1d {
    fn load(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, padding: usize, output_padding: usize, vb: VarBuilder) -> Result<Self> {
        // weight_norm's default dim=0 normalizes over `in_channels` for
        // ConvTranspose1d (weight shape `[in_ch, out_ch/groups, kernel]`,
        // groups=1 always here — see module docs).
        let weight_v = vb.get((in_ch, out_ch, kernel), "weight_v")?;
        let weight_g = vb.get((in_ch, 1, 1), "weight_g")?;
        let weight = reconstruct_weight_norm(&weight_v, &weight_g)?;
        let bias = vb.get(out_ch, "bias")?;
        let cfg = ConvTranspose1dConfig { padding: 0, output_padding: 0, stride, dilation: 1, groups: 1 };
        let conv = candle_nn::ConvTranspose1d::new(weight, Some(bias), cfg);
        Ok(Self { conv, trim: padding * 2 - output_padding })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.conv.forward(x)?;
        let len = out.dim(2)?;
        out.narrow(2, 0, len - self.trim)
    }
}

// ── Snake activation ─────────────────────────────────────────────────────

/// `x + (alpha+eps)^-1 * sin(alpha*x)^2`, `alpha`: `[1, C, 1]` (per-channel,
/// broadcasts over batch/time). Port of `Snake1d`/`snake`.
struct Snake1d {
    alpha: Tensor,
}

impl Snake1d {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { alpha: vb.get((1, channels, 1), "alpha")? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let sin = x.broadcast_mul(&self.alpha)?.sin()?;
        let sin2 = sin.sqr()?;
        let recip = (&self.alpha + 1e-9)?.recip()?;
        x + sin2.broadcast_mul(&recip)?
    }
}

// ── Residual unit ────────────────────────────────────────────────────────

/// Port of `CausalResidualUnit`: `snake -> dilated depthwise k=7 conv ->
/// snake -> pointwise k=1 conv`, residual-added. The depthwise conv gets
/// `groups=dim` when the model is `depthwise` (this checkpoint always is);
/// the pointwise `k=1` conv is **always** `groups=1` regardless — `groups`
/// is only threaded to the first conv in the Python (`CausalResidualUnit.__init__`'s
/// `groups` param is used solely by the dilated conv).
struct CausalResidualUnit {
    snake1: Snake1d,
    conv1: CausalConv1d,
    snake2: Snake1d,
    conv2: CausalConv1d,
}

impl CausalResidualUnit {
    fn load(dim: usize, dilation: usize, groups: usize, vb: VarBuilder) -> Result<Self> {
        // `self.block = nn.Sequential(...)` in Python — an extra "block"
        // path segment wraps the four sub-layers.
        let vb = vb.pp("block");
        let pad = (7 - 1) * dilation / 2;
        Ok(Self {
            snake1: Snake1d::load(dim, vb.pp(0))?,
            conv1: CausalConv1d::load(dim, dim, 7, 1, dilation, groups, pad, 0, vb.pp(1))?,
            snake2: Snake1d::load(dim, vb.pp(2))?,
            conv2: CausalConv1d::load(dim, dim, 1, 1, 1, 1, 0, 0, vb.pp(3))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv1.forward(&self.snake1.forward(x)?)?;
        let y = self.conv2.forward(&self.snake2.forward(&y)?)?;
        x + y
    }
}

// ── Decoder block (upsample stage) ──────────────────────────────────────

/// Port of `CausalDecoderBlock`: `snake -> transpose-conv (upsample by
/// `stride`) -> 3x CausalResidualUnit (dilations 1, 3, 9)`. No `NoiseBlock`
/// (`use_noise_block=False` for this checkpoint).
struct CausalDecoderBlock {
    snake: Snake1d,
    upsample: CausalConvTranspose1d,
    res_units: [CausalResidualUnit; 3],
}

impl CausalDecoderBlock {
    fn load(input_dim: usize, output_dim: usize, stride: usize, depthwise: bool, vb: VarBuilder) -> Result<Self> {
        // `self.block = nn.Sequential(...)` in Python — same extra "block"
        // path segment as `CausalResidualUnit`.
        let vb = vb.pp("block");
        let groups = if depthwise { output_dim } else { 1 };
        let padding = stride.div_ceil(2);
        let output_padding = stride % 2;
        let snake = Snake1d::load(input_dim, vb.pp(0))?;
        let upsample =
            CausalConvTranspose1d::load(input_dim, output_dim, 2 * stride, stride, padding, output_padding, vb.pp(1))?;
        let res_units = [
            CausalResidualUnit::load(output_dim, 1, groups, vb.pp(2))?,
            CausalResidualUnit::load(output_dim, 3, groups, vb.pp(3))?,
            CausalResidualUnit::load(output_dim, 9, groups, vb.pp(4))?,
        ];
        Ok(Self { snake, upsample, res_units })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.upsample.forward(&self.snake.forward(x)?)?;
        for unit in &self.res_units {
            x = unit.forward(&x)?;
        }
        Ok(x)
    }
}

// ── Sample-rate conditioning (FiLM-style) ───────────────────────────────

/// `x * scale_embed[sr_idx] + bias_embed[sr_idx]`, both `[1, C, 1]` after
/// lookup — port of `SampleRateConditionLayer` (`cond_type="scale_bias"`,
/// `out_layer=False` for this checkpoint, so no trailing conv/snake here).
struct SampleRateConditionLayer {
    scale_embed: Embedding,
    bias_embed: Embedding,
}

impl SampleRateConditionLayer {
    fn load(input_dim: usize, sr_bin_buckets: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            scale_embed: embedding(sr_bin_buckets, input_dim, vb.pp("scale_embed"))?,
            bias_embed: embedding(sr_bin_buckets, input_dim, vb.pp("bias_embed"))?,
        })
    }

    fn forward(&self, x: &Tensor, sr_idx: u32) -> Result<Tensor> {
        let device = x.device();
        let idx = Tensor::new(&[sr_idx], device)?;
        let dim = x.dim(1)?;
        let scale = self.scale_embed.forward(&idx)?.reshape((1, dim, 1))?;
        let bias = self.bias_embed.forward(&idx)?.reshape((1, dim, 1))?;
        x.broadcast_mul(&scale)?.broadcast_add(&bias)
    }
}

// ── Top-level decoder ────────────────────────────────────────────────────

/// Port of `CausalDecoder.forward` (`sr_bin_boundaries` branch only — this
/// checkpoint always has one). Channel progression for
/// `decoder_dim=2048, decoder_rates=[8,6,5,2,2,2]`: `64 -[dw+pw]-> 2048 ->
/// 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -[final conv]-> 1`.
pub struct AudioVaeDecoder {
    init_dw: CausalConv1d,
    init_pw: CausalConv1d,
    blocks: Vec<CausalDecoderBlock>,
    sr_cond: Vec<SampleRateConditionLayer>,
    final_snake: Snake1d,
    final_conv: CausalConv1d,
    sr_bin_boundaries: Vec<i64>,
    out_sample_rate: i64,
}

impl AudioVaeDecoder {
    pub fn new(
        latent_dim: usize,
        decoder_dim: usize,
        decoder_rates: &[usize],
        sr_bin_boundaries: Vec<i64>,
        out_sample_rate: i64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let depthwise = true; // this checkpoint's AudioVAEConfig default
        let vb_model = vb.pp("model");

        // `depthwise=True` initial layer is two convs: depthwise k=7 (groups=in)
        // then pointwise k=1 — matches `decoder.model.0`/`decoder.model.1`.
        let init_dw = CausalConv1d::load(latent_dim, latent_dim, 7, 1, 1, latent_dim, 3, 0, vb_model.pp(0))?;
        let init_pw = CausalConv1d::load(latent_dim, decoder_dim, 1, 1, 1, 1, 0, 0, vb_model.pp(1))?;

        let sr_bin_buckets = sr_bin_boundaries.len() + 1;
        let mut blocks = Vec::with_capacity(decoder_rates.len());
        let mut sr_cond = Vec::with_capacity(decoder_rates.len());
        let mut ch = decoder_dim;
        for (i, &stride) in decoder_rates.iter().enumerate() {
            let output_dim = ch / 2;
            // `decoder.model.{2..7}` (indices 0,1 are the initial dw/pw convs).
            blocks.push(CausalDecoderBlock::load(ch, output_dim, stride, depthwise, vb_model.pp(2 + i))?);
            sr_cond.push(SampleRateConditionLayer::load(ch, sr_bin_buckets, vb.pp("sr_cond_model").pp(2 + i))?);
            ch = output_dim;
        }

        let final_idx = 2 + decoder_rates.len();
        let final_snake = Snake1d::load(ch, vb_model.pp(final_idx))?;
        let final_conv = CausalConv1d::load(ch, 1, 7, 1, 1, 1, 3, 0, vb_model.pp(final_idx + 1))?;

        Ok(Self {
            init_dw,
            init_pw,
            blocks,
            sr_cond,
            final_snake,
            final_conv,
            sr_bin_boundaries,
            out_sample_rate,
        })
    }

    /// `torch.bucketize(input, boundaries, right=False)`: smallest `i` with
    /// `input <= boundaries[i]`, or `len(boundaries)` if none — equivalently
    /// the count of boundaries strictly less than `input`.
    fn sr_bucket_idx(&self, sr_hz: i64) -> u32 {
        self.sr_bin_boundaries.iter().filter(|&&b| b < sr_hz).count() as u32
    }

    /// `z`: `[B, latent_dim, T]`. Returns `[B, 1, T']` waveform in `[-1, 1]`
    /// (final `Tanh`). Always decodes at `out_sample_rate` (48kHz) — matches
    /// `AudioVAE.decode(z, sr_cond=None)`'s default.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let sr_idx = self.sr_bucket_idx(self.out_sample_rate);

        let mut x = self.init_pw.forward(&self.init_dw.forward(z)?)?;
        for (block, cond) in self.blocks.iter().zip(&self.sr_cond) {
            x = cond.forward(&x, sr_idx)?;
            x = block.forward(&x)?;
        }
        let x = self.final_conv.forward(&self.final_snake.forward(&x)?)?;
        x.tanh()
    }
}

// ── Encoder block (downsample stage) ────────────────────────────────────

/// Port of `CausalEncoderBlock`: `3x CausalResidualUnit (dilations 1, 3, 9)
/// -> snake -> causal conv (downsample by `stride`)`. Mirror-image of
/// `CausalDecoderBlock` (residual units *before* the resample op instead of
/// after, downsampling conv instead of transpose-conv upsample). The
/// residual units run on `input_dim` (pre-downsample channel count), the
/// final conv projects `input_dim -> output_dim`; unlike the decoder's
/// downsample-mirror, **this conv is always `groups=1`** regardless of
/// `depthwise` — `CausalEncoderBlock.__init__` never threads `groups` into
/// its own `WNCausalConv1d` call, confirmed against the real checkpoint's
/// tensor shapes (`encoder.block.N.block.4.weight_v` has full `input_dim`
/// in its second axis, not `input_dim/groups`).
struct CausalEncoderBlock {
    res_units: [CausalResidualUnit; 3],
    snake: Snake1d,
    downsample: CausalConv1d,
}

impl CausalEncoderBlock {
    fn load(input_dim: usize, output_dim: usize, stride: usize, depthwise: bool, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("block");
        let groups = if depthwise { input_dim } else { 1 };
        let padding = stride.div_ceil(2);
        let output_padding = stride % 2;
        let res_units = [
            CausalResidualUnit::load(input_dim, 1, groups, vb.pp(0))?,
            CausalResidualUnit::load(input_dim, 3, groups, vb.pp(1))?,
            CausalResidualUnit::load(input_dim, 9, groups, vb.pp(2))?,
        ];
        let snake = Snake1d::load(input_dim, vb.pp(3))?;
        let downsample =
            CausalConv1d::load(input_dim, output_dim, 2 * stride, stride, 1, 1, padding, output_padding, vb.pp(4))?;
        Ok(Self { res_units, snake, downsample })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for unit in &self.res_units {
            x = unit.forward(&x)?;
        }
        self.downsample.forward(&self.snake.forward(&x)?)
    }
}

// ── Top-level encoder ────────────────────────────────────────────────────

/// Port of `CausalEncoder.forward`, `mu` output only — no `logvar`/sampling
/// at inference (deterministic, matches `AudioVAE.encode()`'s own posture:
/// `self.encoder(audio_data)["mu"]`, `fc_logvar` is computed but unused
/// there, so this doesn't even load it). Channel progression for
/// `encoder_dim=128, encoder_rates=[2,5,8,8]`: `1 -[init conv]-> 128 -> 256
/// -> 512 -> 1024 -> 2048 -[fc_mu]-> latent_dim(64)`.
pub struct AudioVaeEncoder {
    init_conv: CausalConv1d,
    blocks: Vec<CausalEncoderBlock>,
    fc_mu: CausalConv1d,
    /// `prod(encoder_rates)` — total downsample factor; audio must be
    /// padded to a multiple of this before encoding (`preprocess`).
    hop_length: usize,
}

impl AudioVaeEncoder {
    pub fn new(encoder_dim: usize, latent_dim: usize, encoder_rates: &[usize], vb: VarBuilder) -> Result<Self> {
        let depthwise = true; // this checkpoint's AudioVAEConfig default, same as the decoder
        let vb_block = vb.pp("block");

        let init_conv = CausalConv1d::load(1, encoder_dim, 7, 1, 1, 1, 3, 0, vb_block.pp(0))?;

        let mut blocks = Vec::with_capacity(encoder_rates.len());
        let mut ch = encoder_dim;
        for (i, &stride) in encoder_rates.iter().enumerate() {
            let output_dim = ch * 2;
            // `encoder.block.{1..}` (index 0 is the initial conv above).
            blocks.push(CausalEncoderBlock::load(ch, output_dim, stride, depthwise, vb_block.pp(1 + i))?);
            ch = output_dim;
        }

        let fc_mu = CausalConv1d::load(ch, latent_dim, 3, 1, 1, 1, 1, 0, vb.pp("fc_mu"))?;

        Ok(Self { init_conv, blocks, fc_mu, hop_length: encoder_rates.iter().product() })
    }

    /// `audio`: `[B, 1, T]` raw waveform at the VAE's native (encoder) sample
    /// rate. Right-zero-pads `T` to a multiple of `hop_length` (matching
    /// `AudioVAE.preprocess`), then returns `mu`: `[B, latent_dim, T']`
    /// where `T' = T_padded / hop_length`.
    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        let t = audio.dim(2)?;
        let right_pad = t.div_ceil(self.hop_length) * self.hop_length - t;
        let x = if right_pad > 0 { audio.pad_with_zeros(2, 0, right_pad)? } else { audio.clone() };

        let mut x = self.init_conv.forward(&x)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        self.fc_mu.forward(&x)
    }
}

#[cfg(test)]
mod shape_smoke_test {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    // Opt-in: loads the real converted checkpoint and checks decode() runs
    // and produces a plausibly-shaped, finite, in-range waveform. Not a
    // numeric HF diff (see task #24) — just catches shape/dimension bugs
    // cheaply before building the full generation loop on top.
    #[test]
    #[ignore = "needs the real VoxCPM2 checkpoint (audiovae.safetensors)"]
    fn decode_real_weights_shape_smoke_test() {
        let path = "/home/hahihula/mywork/ai/additional_models/VoxCPM2/audiovae.safetensors";
        let device = Device::Cpu;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device) }.unwrap();

        let decoder = AudioVaeDecoder::new(64, 2048, &[8, 6, 5, 2, 2, 2], vec![20_000, 30_000, 40_000], 48_000, vb.pp("decoder"))
            .expect("build decoder");

        // A handful of latent timesteps is enough to exercise every stride.
        let z = Tensor::randn(0f32, 1f32, (1, 64, 16), &device).unwrap();
        let wav = decoder.decode(&z).expect("decode");
        eprintln!("wav shape: {:?}", wav.dims());
        assert_eq!(wav.dim(0).unwrap(), 1);
        assert_eq!(wav.dim(1).unwrap(), 1);
        // Total upsample factor = product(decoder_rates) = 8*6*5*2*2*2 = 1920.
        assert_eq!(wav.dim(2).unwrap(), 16 * 8 * 6 * 5 * 2 * 2 * 2);

        let flat: Vec<f32> = wav.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()), "non-finite sample in output");
        assert!(flat.iter().all(|v| (-1.0..=1.0).contains(v)), "sample outside tanh range");
    }

    #[test]
    #[ignore = "needs the real VoxCPM2 checkpoint (audiovae.safetensors)"]
    fn encode_real_weights_shape_smoke_test() {
        let path = "/home/hahihula/mywork/ai/additional_models/VoxCPM2/audiovae.safetensors";
        let device = Device::Cpu;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device) }.unwrap();

        let encoder = AudioVaeEncoder::new(128, 64, &[2, 5, 8, 8], vb.pp("encoder")).expect("build encoder");

        // hop_length = 2*5*8*8 = 640. A few multiples of that is enough to
        // exercise every stride and the right-zero-pad path.
        let audio = Tensor::randn(0f32, 1f32, (1, 1, 640 * 5 + 37), &device).unwrap();
        let mu = encoder.encode(&audio).expect("encode");
        eprintln!("mu shape: {:?}", mu.dims());
        assert_eq!(mu.dim(0).unwrap(), 1);
        assert_eq!(mu.dim(1).unwrap(), 64);
        // Padded length is the next multiple of 640 above 640*5+37, i.e. 640*6.
        assert_eq!(mu.dim(2).unwrap(), 6);

        let flat: Vec<f32> = mu.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()), "non-finite value in mu output");
    }

    // Round-trips a real waveform through encode() then decode() and checks
    // the reconstruction is plausible (not a numeric HF diff — that's a
    // separate, `#[ignore]`d cross-check against a real Python dump).
    #[test]
    #[ignore = "needs the real VoxCPM2 checkpoint (audiovae.safetensors)"]
    fn encode_decode_roundtrip_shape_smoke_test() {
        let path = "/home/hahihula/mywork/ai/additional_models/VoxCPM2/audiovae.safetensors";
        let device = Device::Cpu;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, &device) }.unwrap();

        let encoder = AudioVaeEncoder::new(128, 64, &[2, 5, 8, 8], vb.pp("encoder")).expect("build encoder");
        let decoder = AudioVaeDecoder::new(64, 2048, &[8, 6, 5, 2, 2, 2], vec![20_000, 30_000, 40_000], 48_000, vb.pp("decoder"))
            .expect("build decoder");

        let audio = Tensor::randn(0f32, 0.1f32, (1, 1, 640 * 8), &device).unwrap();
        let mu = encoder.encode(&audio).expect("encode");
        let wav = decoder.decode(&mu).expect("decode");
        eprintln!("input len: {}, mu shape: {:?}, output len: {}", 640 * 8, mu.dims(), wav.dim(2).unwrap());

        let flat: Vec<f32> = wav.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()), "non-finite sample in round-tripped output");
        assert!(flat.iter().all(|v| (-1.0..=1.0).contains(v)), "sample outside tanh range");
    }
}

#[cfg(test)]
mod hf_diff {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    // Cross-checks `AudioVaeEncoder::encode` against a real
    // `AudioVAE.encode()` dump (see voxcpm_encode_diff.py in this session's
    // job tmp dir). `cargo test --release -p crane-core
    // encode_matches_python -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs the real VoxCPM2 checkpoint + a matching HF dump"]
    fn encode_matches_python() {
        let model_path = "/home/hahihula/mywork/ai/additional_models/VoxCPM2";
        let diff_dir = "/home/hahihula/.claude/jobs/02aa7312/tmp/voxcpm_encode_diff";
        let device = Device::Cpu;
        let dtype = DType::F32;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[format!("{model_path}/audiovae.safetensors")], dtype, &device) }.unwrap();
        let encoder = AudioVaeEncoder::new(128, 64, &[2, 5, 8, 8], vb.pp("encoder")).unwrap();

        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{diff_dir}/meta.json")).unwrap()).unwrap();
        let input_len = meta["input_len"].as_u64().unwrap() as usize;
        let latent_dim = meta["latent_dim"].as_u64().unwrap() as usize;
        let out_len = meta["out_len"].as_u64().unwrap() as usize;

        let load = |name: &str, shape: (usize, usize, usize)| -> Tensor {
            let raw = std::fs::read(format!("{diff_dir}/{name}.bin")).unwrap();
            let floats: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            Tensor::from_vec(floats, shape, &device).unwrap()
        };

        let input_audio = load("input_audio", (1, 1, input_len));
        let hf_mu = load("mu", (1, latent_dim, out_len));

        let rust_mu = encoder.encode(&input_audio).unwrap();
        println!("rust_mu shape: {:?}, hf_mu shape: {:?}", rust_mu.dims(), hf_mu.dims());
        assert_eq!(rust_mu.dims(), hf_mu.dims());

        let rust_flat: Vec<f32> = rust_mu.flatten_all().unwrap().to_vec1().unwrap();
        let hf_flat: Vec<f32> = hf_mu.flatten_all().unwrap().to_vec1().unwrap();

        let dot: f64 = rust_flat.iter().zip(&hf_flat).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
        let norm_a: f64 = rust_flat.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = hf_flat.iter().map(|b| f64::from(*b).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (norm_a * norm_b);
        let max_abs_diff = rust_flat.iter().zip(&hf_flat).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("cosine: {cosine}, max_abs_diff: {max_abs_diff}");

        assert!(cosine > 0.999, "encode() diverged from Python: cosine={cosine}");
    }
}
