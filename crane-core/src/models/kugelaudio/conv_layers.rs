//! Causal 1D-conv building blocks shared by the acoustic and semantic
//! tokenizers (`tokenizer_vae.rs`).
//!
//! Port of `kugelaudio_open.models.tokenizer` (structurally a renamed copy of
//! `microsoft/VibeVoice`'s `modular_vibevoice_tokenizer.py`) — `SConv1d`,
//! `SConvTranspose1d`, `ConvRMSNorm`, `Block1D`, `TokenizerEncoder`,
//! `TokenizerDecoder`. **Non-streaming only**: the Python source supports an
//! optional streaming-cache path (`VibeVoiceTokenizerStreamingCache`) for
//! chunked online encode/decode; this port always takes the
//! `_forward_non_streaming` branch, since generation always has the full
//! speech-tensor / full generated-latent sequence available (see `model.rs`).
//!
//! Scoped to the reference checkpoint's config values, matching this crate's
//! convention elsewhere (e.g. `voxcpm2::audio_vae`'s decode-only `AudioVAE`
//! port): `conv_norm: "none"` (no weight-norm/spectral-norm reparametrization
//! — [`SConv1d`]/[`SConvTranspose1d`] only implement the unparametrized
//! conv), `causal: true` for every conv (the non-causal symmetric-padding
//! branch is not implemented), `pad_mode: "constant"` (zero-padding; the
//! Python's `reflect` branch is not implemented), and
//! `trim_right_ratio = 1.0` (`SConvTranspose1d`'s Python default — not a
//! `config.json` field, so there is nothing to override here: causal
//! transpose-conv trimming always removes `kernel_size - stride` samples
//! from the *end* only, never the start).

use candle_core::{D, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};

use crate::models::with_tracing::RmsNorm;

use super::config::TokenizerConfig;

/// `ConvRMSNorm`: RMSNorm applied over the channel dimension of a `[B, C, T]`
/// tensor (transpose to `[B, T, C]`, normalize the last axis, transpose
/// back). The Python also supports plain `LayerNorm` (`layernorm: "LN"`);
/// the reference checkpoint always uses `"RMSNorm"`, so that's the only
/// variant implemented here.
#[derive(Debug, Clone)]
struct ConvRmsNorm {
    norm: RmsNorm,
}

impl ConvRmsNorm {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: RmsNorm::new(dim, eps, vb)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.transpose(1, 2)?;
        let x = self.norm.forward(&x)?;
        x.transpose(1, 2)?.contiguous()
    }
}

/// `get_extra_padding_for_conv1d`: extra right-padding so the conv's output
/// length matches PyTorch's `ceil`-based length formula (candle's `Conv1d`
/// follows the `floor` convention like PyTorch's raw op; this recovers the
/// same "round up" framing the Python's manual padding relies on).
fn extra_padding_for_conv1d(length: usize, kernel_size: usize, stride: usize, padding_total: usize) -> usize {
    let n_frames = (length as f64 - kernel_size as f64 + padding_total as f64) / stride as f64 + 1.0;
    let ideal_length = ((n_frames.ceil() as i64 - 1) * stride as i64) + (kernel_size as i64 - padding_total as i64);
    (ideal_length - length as i64).max(0) as usize
}

/// Causal `SConv1d`: manual left/right zero-padding (`pad_mode: "constant"`)
/// sized so output length follows the `ceil` convention, then an
/// unparametrized `Conv1d` with `padding: 0`.
#[derive(Debug, Clone)]
struct SConv1d {
    conv: Conv1d,
    kernel_size: usize,
    stride: usize,
    padding_total: usize,
}

impl SConv1d {
    #[allow(clippy::too_many_arguments)]
    fn load(
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Python: `SConv1d.conv` = `NormConv1d`, whose own `.conv` is the raw
        // `nn.Conv1d` (the norm reparametrization module in between is
        // `nn.Identity()` for `conv_norm == "none"`, so it carries no weights).
        let vb_conv = vb.pp("conv").pp("conv");
        let weight = vb_conv.get((out_ch, in_ch / groups, kernel_size), "weight")?;
        let bias = if bias {
            Some(vb_conv.get(out_ch, "bias")?)
        } else {
            None
        };
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };
        let padding_total = (kernel_size - 1) * dilation - (stride - 1);
        Ok(Self {
            conv: Conv1d::new(weight, bias, cfg),
            kernel_size,
            stride,
            padding_total,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let length = x.dim(D::Minus1)?;
        let extra = extra_padding_for_conv1d(length, self.kernel_size, self.stride, self.padding_total);
        // Causal + `pad_mode == "constant"`: left-pad by `padding_total`,
        // right-pad by `extra` (both zero).
        let x = x.pad_with_zeros(D::Minus1, self.padding_total, extra)?;
        self.conv.forward(&x)
    }
}

/// Causal `SConvTranspose1d`: an unparametrized `ConvTranspose1d` with
/// `padding: 0`, followed by trimming `kernel_size - stride` samples off the
/// **end** only (`trim_right_ratio = 1.0` — see module doc comment).
#[derive(Debug, Clone)]
struct SConvTranspose1d {
    convtr: ConvTranspose1d,
    trim_right: usize,
}

impl SConvTranspose1d {
    fn load(
        in_ch: usize,
        out_ch: usize,
        kernel_size: usize,
        stride: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Python: `SConvTranspose1d.convtr` = `NormConvTranspose1d`, whose
        // `.convtr` is the raw `nn.ConvTranspose1d` (again, no reparametrization
        // weights for `conv_norm == "none"`).
        let vb_conv = vb.pp("convtr").pp("convtr");
        // PyTorch `ConvTranspose1d` weight layout: `[in_channels, out_channels, kernel_size]`.
        let weight = vb_conv.get((in_ch, out_ch, kernel_size), "weight")?;
        let bias = if bias {
            Some(vb_conv.get(out_ch, "bias")?)
        } else {
            None
        };
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation: 1,
            groups: 1,
        };
        Ok(Self {
            convtr: ConvTranspose1d::new(weight, bias, cfg),
            trim_right: kernel_size - stride,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.convtr.forward(x)?;
        if self.trim_right == 0 {
            return Ok(y);
        }
        let len = y.dim(D::Minus1)?;
        y.narrow(D::Minus1, 0, len - self.trim_right)
    }
}

/// `FFN`: `linear1 -> gelu -> linear2`, expansion ratio 4 (Python's
/// `Block1D`'s hardcoded `kwargs.get('ffn_expansion', 4)` default — not a
/// `config.json` field).
#[derive(Debug, Clone)]
struct ConvFfn {
    linear1: candle_nn::Linear,
    linear2: candle_nn::Linear,
}

impl ConvFfn {
    fn load(dim: usize, ffn_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let linear1 = candle_nn::linear_b(dim, ffn_dim, bias, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear_b(ffn_dim, dim, bias, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?.gelu_erf()?;
        self.linear2.forward(&x)
    }
}

/// One ConvNeXt-style residual block: depthwise/grouped conv mixer (kernel
/// size fixed at 7 — `Block1D`'s Python default, not config-driven) plus an
/// FFN, each with an optional learnable `LayerScale` and a residual add.
/// `DropPath` is intentionally not ported: it's `nn.Identity()` at
/// `eval()`/inference regardless of `drop_path_rate`.
#[derive(Debug, Clone)]
struct Block1D {
    norm: ConvRmsNorm,
    mixer: SConv1d,
    ffn_norm: ConvRmsNorm,
    ffn: ConvFfn,
    gamma: Option<Tensor>,
    ffn_gamma: Option<Tensor>,
}

const BLOCK1D_KERNEL_SIZE: usize = 7;
const FFN_EXPANSION: usize = 4;

impl Block1D {
    fn load(dim: usize, cfg: &TokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let norm = ConvRmsNorm::load(dim, cfg.layernorm_eps, vb.pp("norm"))?;
        let ffn_norm = ConvRmsNorm::load(dim, cfg.layernorm_eps, vb.pp("ffn_norm"))?;
        let groups = match cfg.mixer_layer.as_str() {
            "depthwise_conv" => dim,
            "conv" => 1,
            other => candle_core::bail!("kugelaudio: unsupported mixer_layer {other:?}"),
        };
        let mixer = SConv1d::load(
            dim,
            dim,
            BLOCK1D_KERNEL_SIZE,
            1,
            1,
            groups,
            cfg.conv_bias,
            vb.pp("mixer").pp("conv"),
        )?;
        let ffn = ConvFfn::load(dim, FFN_EXPANSION * dim, cfg.conv_bias, vb.pp("ffn"))?;
        let (gamma, ffn_gamma) = if cfg.layer_scale_init_value > 0.0 {
            (
                Some(vb.get(dim, "gamma")?),
                Some(vb.get(dim, "ffn_gamma")?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            norm,
            mixer,
            ffn_norm,
            ffn,
            gamma,
            ffn_gamma,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let y = self.norm.forward(x)?;
        let y = self.mixer.forward(&y)?;
        let y = match &self.gamma {
            Some(g) => y.broadcast_mul(&g.reshape((1, (), 1))?)?,
            None => y,
        };
        let x = (residual + y)?;

        let residual = &x;
        let y = self.ffn_norm.forward(&x)?;
        // FFN operates on the channel-last layout.
        let y = y.transpose(1, 2)?;
        let y = self.ffn.forward(&y)?;
        let y = y.transpose(1, 2)?.contiguous()?;
        let y = match &self.ffn_gamma {
            Some(g) => y.broadcast_mul(&g.reshape((1, (), 1))?)?,
            None => y,
        };
        residual + y
    }
}

/// Downsampling/upsampling causal-conv encoder. Structurally: a stem conv,
/// then `len(depths)` stages of `[downsample conv] -> [Block1D * depth]`,
/// then an optional final norm and a head conv projecting to `vae_dim`.
pub struct TokenizerEncoder {
    /// `downsample_layers[0]` is the stem (channels -> `n_filters`, no
    /// downsampling); `downsample_layers[1..]` halve/downsample by
    /// `ratios[i-1]` and double the channel count.
    downsample_layers: Vec<SConv1d>,
    stages: Vec<Vec<Block1D>>,
    final_norm: Option<ConvRmsNorm>,
    head: SConv1d,
}

const STEM_KERNEL_SIZE: usize = 7;
const HEAD_KERNEL_SIZE: usize = 7;

impl TokenizerEncoder {
    fn load(cfg: &TokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let depths = cfg.encoder_depths_vec();
        // Python reverses `ratios` for the encoder (finest stride last in
        // the config list becomes the first downsample step here).
        let mut ratios = cfg.encoder_ratios.clone();
        ratios.reverse();
        let n_filters = cfg.encoder_n_filters;

        let vb_down = vb.pp("downsample_layers");
        let mut downsample_layers = Vec::with_capacity(ratios.len() + 1);
        downsample_layers.push(SConv1d::load(
            cfg.channels,
            n_filters,
            STEM_KERNEL_SIZE,
            1,
            1,
            1,
            cfg.conv_bias,
            vb_down.pp(0).pp(0),
        )?);
        for (i, &ratio) in ratios.iter().enumerate() {
            let in_ch = n_filters * (1 << i);
            let out_ch = n_filters * (1 << (i + 1));
            downsample_layers.push(SConv1d::load(
                in_ch,
                out_ch,
                ratio * 2,
                ratio,
                1,
                1,
                cfg.conv_bias,
                vb_down.pp(i + 1).pp(0),
            )?);
        }

        let vb_stages = vb.pp("stages");
        let mut stages = Vec::with_capacity(depths.len());
        let mut last_ch = n_filters;
        for (i, &depth) in depths.iter().enumerate() {
            let ch = n_filters * (1 << i);
            last_ch = ch;
            let vb_stage = vb_stages.pp(i);
            let mut blocks = Vec::with_capacity(depth);
            for j in 0..depth {
                blocks.push(Block1D::load(ch, cfg, vb_stage.pp(j))?);
            }
            stages.push(blocks);
        }

        let final_norm = if cfg.disable_last_norm {
            None
        } else {
            Some(ConvRmsNorm::load(last_ch, cfg.layernorm_eps, vb.pp("norm"))?)
        };
        let head = SConv1d::load(
            last_ch,
            cfg.vae_dim,
            HEAD_KERNEL_SIZE,
            1,
            1,
            1,
            cfg.conv_bias,
            vb.pp("head"),
        )?;

        Ok(Self {
            downsample_layers,
            stages,
            final_norm,
            head,
        })
    }

    /// `x`: `[batch, channels, time]` raw waveform (or, for the decoder's
    /// mirror shape, `[batch, vae_dim, time]`). Returns `[batch, vae_dim, T']`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (down, stage) in self.downsample_layers.iter().zip(self.stages.iter()) {
            x = down.forward(&x)?;
            for block in stage {
                x = block.forward(&x)?;
            }
        }
        if let Some(norm) = &self.final_norm {
            x = norm.forward(&x)?;
        }
        self.head.forward(&x)
    }
}

/// Mirror of [`TokenizerEncoder`]: upsampling stages back to `channels`
/// (waveform space). Only built for the acoustic tokenizer (see
/// `tokenizer_vae.rs`) — the semantic tokenizer never decodes.
pub struct TokenizerDecoder {
    upsample_layers: Vec<UpsampleLayer>,
    stages: Vec<Vec<Block1D>>,
    final_norm: Option<ConvRmsNorm>,
    head: SConv1d,
}

enum UpsampleLayer {
    /// The stem: a plain `SConv1d` (no channel-count change from `vae_dim`),
    /// matching Python's `TokenizerDecoder`'s stem being an `SConv1d`, not
    /// an `SConvTranspose1d`.
    Stem(SConv1d),
    Up(SConvTranspose1d),
}

impl UpsampleLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Stem(c) => c.forward(x),
            Self::Up(c) => c.forward(x),
        }
    }
}

impl TokenizerDecoder {
    fn load(cfg: &TokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let depths = cfg.decoder_depths_vec();
        let ratios = cfg
            .decoder_ratios
            .clone()
            .ok_or_else(|| candle_core::Error::Msg("kugelaudio: decoder_ratios missing".into()))?;
        let n_filters = cfg
            .decoder_n_filters
            .ok_or_else(|| candle_core::Error::Msg("kugelaudio: decoder_n_filters missing".into()))?;
        let n_stages = depths.len();

        let vb_up = vb.pp("upsample_layers");
        let stem_ch = n_filters * (1 << (n_stages - 1));
        let mut upsample_layers = Vec::with_capacity(ratios.len() + 1);
        upsample_layers.push(UpsampleLayer::Stem(SConv1d::load(
            cfg.vae_dim,
            stem_ch,
            STEM_KERNEL_SIZE,
            1,
            1,
            1,
            cfg.conv_bias,
            vb_up.pp(0).pp(0),
        )?));
        for (i, &ratio) in ratios.iter().enumerate() {
            let in_ch = n_filters * (1 << (n_stages - 1 - i));
            let out_ch = n_filters * (1 << (n_stages - 2 - i));
            upsample_layers.push(UpsampleLayer::Up(SConvTranspose1d::load(
                in_ch,
                out_ch,
                ratio * 2,
                ratio,
                cfg.conv_bias,
                vb_up.pp(i + 1).pp(0),
            )?));
        }

        let vb_stages = vb.pp("stages");
        let mut stages = Vec::with_capacity(n_stages);
        let mut last_ch = stem_ch;
        for (i, &depth) in depths.iter().enumerate() {
            let ch = n_filters * (1 << (n_stages - 1 - i));
            last_ch = ch;
            let vb_stage = vb_stages.pp(i);
            let mut blocks = Vec::with_capacity(depth);
            for j in 0..depth {
                blocks.push(Block1D::load(ch, cfg, vb_stage.pp(j))?);
            }
            stages.push(blocks);
        }

        let final_norm = if cfg.disable_last_norm {
            None
        } else {
            Some(ConvRmsNorm::load(last_ch, cfg.layernorm_eps, vb.pp("norm"))?)
        };
        let head = SConv1d::load(
            last_ch,
            cfg.channels,
            HEAD_KERNEL_SIZE,
            1,
            1,
            1,
            cfg.conv_bias,
            vb.pp("head"),
        )?;

        Ok(Self {
            upsample_layers,
            stages,
            final_norm,
            head,
        })
    }

    /// `x`: `[batch, vae_dim, T]` latents. Returns `[batch, channels, T']` waveform.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (up, stage) in self.upsample_layers.iter().zip(self.stages.iter()) {
            x = up.forward(&x)?;
            for block in stage {
                x = block.forward(&x)?;
            }
        }
        if let Some(norm) = &self.final_norm {
            x = norm.forward(&x)?;
        }
        self.head.forward(&x)
    }
}

/// Build a `TokenizerEncoder` for a given [`TokenizerConfig`].
pub fn load_encoder(cfg: &TokenizerConfig, vb: VarBuilder) -> Result<TokenizerEncoder> {
    TokenizerEncoder::load(cfg, vb)
}

/// Build a `TokenizerDecoder` for a given [`TokenizerConfig`]. Requires
/// `decoder_n_filters`/`decoder_ratios` to be present (the acoustic
/// tokenizer's config; the semantic tokenizer's config omits them and never
/// calls this).
pub fn load_decoder(cfg: &TokenizerConfig, vb: VarBuilder) -> Result<TokenizerDecoder> {
    TokenizerDecoder::load(cfg, vb)
}

impl TokenizerEncoder {
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

impl TokenizerDecoder {
    pub fn decode(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    fn small_cfg(with_decoder: bool) -> TokenizerConfig {
        TokenizerConfig {
            channels: 1,
            vae_dim: 8,
            encoder_n_filters: 4,
            encoder_ratios: vec![2, 2],
            encoder_depths: "1-1-1".to_string(),
            decoder_n_filters: with_decoder.then_some(4),
            decoder_ratios: with_decoder.then(|| vec![2, 2]),
            decoder_depths: None,
            causal: true,
            conv_bias: true,
            conv_norm: "none".to_string(),
            pad_mode: "constant".to_string(),
            layernorm: "RMSNorm".to_string(),
            layernorm_eps: 1e-5,
            layernorm_elementwise_affine: true,
            mixer_layer: "depthwise_conv".to_string(),
            layer_scale_init_value: 1e-6,
            disable_last_norm: true,
            fix_std: 0.5,
            std_dist_type: "gaussian".to_string(),
        }
    }

    /// Populate every tensor a `TokenizerEncoder`/`TokenizerDecoder` needs
    /// with small nonzero values, keyed by exact HF weight name, so
    /// `VarBuilder::from_tensors` can satisfy every `vb.get`/`vb.pp` call in
    /// `load`. Shapes are derived the same way `TokenizerEncoder::load` /
    /// `TokenizerDecoder::load` derive them.
    fn make_vb(cfg: &TokenizerConfig, device: &Device) -> VarBuilder<'static> {
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let fill = |shape: &[usize]| -> Tensor {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| 0.01 * (i as f32 + 1.0)).collect();
            Tensor::from_vec(data, shape, device).unwrap()
        };
        let ones = |n: usize| Tensor::ones(n, DType::F32, device).unwrap();

        let depths = cfg.encoder_depths_vec();
        let mut ratios = cfg.encoder_ratios.clone();
        ratios.reverse();
        let n_filters = cfg.encoder_n_filters;

        // Encoder stem + downsample convs.
        t.insert(
            "encoder.downsample_layers.0.0.conv.conv.weight".into(),
            fill(&[n_filters, cfg.channels, STEM_KERNEL_SIZE]),
        );
        t.insert(
            "encoder.downsample_layers.0.0.conv.conv.bias".into(),
            fill(&[n_filters]),
        );
        for (i, &ratio) in ratios.iter().enumerate() {
            let in_ch = n_filters * (1 << i);
            let out_ch = n_filters * (1 << (i + 1));
            t.insert(
                format!("encoder.downsample_layers.{}.0.conv.conv.weight", i + 1),
                fill(&[out_ch, in_ch, ratio * 2]),
            );
            t.insert(
                format!("encoder.downsample_layers.{}.0.conv.conv.bias", i + 1),
                fill(&[out_ch]),
            );
        }
        // Encoder stages.
        for (i, &depth) in depths.iter().enumerate() {
            let ch = n_filters * (1 << i);
            for j in 0..depth {
                let p = format!("encoder.stages.{i}.{j}");
                t.insert(format!("{p}.norm.weight"), ones(ch));
                t.insert(format!("{p}.ffn_norm.weight"), ones(ch));
                t.insert(
                    format!("{p}.mixer.conv.conv.conv.weight"),
                    fill(&[ch, 1, BLOCK1D_KERNEL_SIZE]),
                );
                t.insert(format!("{p}.mixer.conv.conv.conv.bias"), fill(&[ch]));
                t.insert(format!("{p}.ffn.linear1.weight"), fill(&[FFN_EXPANSION * ch, ch]));
                t.insert(format!("{p}.ffn.linear1.bias"), fill(&[FFN_EXPANSION * ch]));
                t.insert(format!("{p}.ffn.linear2.weight"), fill(&[ch, FFN_EXPANSION * ch]));
                t.insert(format!("{p}.ffn.linear2.bias"), fill(&[ch]));
                t.insert(format!("{p}.gamma"), ones(ch));
                t.insert(format!("{p}.ffn_gamma"), ones(ch));
            }
        }
        let last_ch = n_filters * (1 << (depths.len() - 1));
        t.insert(
            "encoder.head.conv.conv.weight".into(),
            fill(&[cfg.vae_dim, last_ch, HEAD_KERNEL_SIZE]),
        );
        t.insert("encoder.head.conv.conv.bias".into(), fill(&[cfg.vae_dim]));

        if cfg.decoder_n_filters.is_some() {
            let ddepths = cfg.decoder_depths_vec();
            let dratios = cfg.decoder_ratios.clone().unwrap();
            let dn_filters = cfg.decoder_n_filters.unwrap();
            let n_stages = ddepths.len();
            let stem_ch = dn_filters * (1 << (n_stages - 1));
            t.insert(
                "decoder.upsample_layers.0.0.conv.conv.weight".into(),
                fill(&[stem_ch, cfg.vae_dim, STEM_KERNEL_SIZE]),
            );
            t.insert(
                "decoder.upsample_layers.0.0.conv.conv.bias".into(),
                fill(&[stem_ch]),
            );
            for (i, &ratio) in dratios.iter().enumerate() {
                let in_ch = dn_filters * (1 << (n_stages - 1 - i));
                let out_ch = dn_filters * (1 << (n_stages - 2 - i));
                t.insert(
                    format!("decoder.upsample_layers.{}.0.convtr.convtr.weight", i + 1),
                    fill(&[in_ch, out_ch, ratio * 2]),
                );
                t.insert(
                    format!("decoder.upsample_layers.{}.0.convtr.convtr.bias", i + 1),
                    fill(&[out_ch]),
                );
            }
            for (i, &depth) in ddepths.iter().enumerate() {
                let ch = dn_filters * (1 << (n_stages - 1 - i));
                for j in 0..depth {
                    let p = format!("decoder.stages.{i}.{j}");
                    t.insert(format!("{p}.norm.weight"), ones(ch));
                    t.insert(format!("{p}.ffn_norm.weight"), ones(ch));
                    t.insert(
                        format!("{p}.mixer.conv.conv.conv.weight"),
                        fill(&[ch, 1, BLOCK1D_KERNEL_SIZE]),
                    );
                    t.insert(format!("{p}.mixer.conv.conv.conv.bias"), fill(&[ch]));
                    t.insert(format!("{p}.ffn.linear1.weight"), fill(&[FFN_EXPANSION * ch, ch]));
                    t.insert(format!("{p}.ffn.linear1.bias"), fill(&[FFN_EXPANSION * ch]));
                    t.insert(format!("{p}.ffn.linear2.weight"), fill(&[ch, FFN_EXPANSION * ch]));
                    t.insert(format!("{p}.ffn.linear2.bias"), fill(&[ch]));
                    t.insert(format!("{p}.gamma"), ones(ch));
                    t.insert(format!("{p}.ffn_gamma"), ones(ch));
                }
            }
            let final_ch = dn_filters; // last stage channel count == n_filters * 2^0
            t.insert(
                "decoder.head.conv.conv.weight".into(),
                fill(&[cfg.channels, final_ch, HEAD_KERNEL_SIZE]),
            );
            t.insert("decoder.head.conv.conv.bias".into(), fill(&[cfg.channels]));
        }

        VarBuilder::from_tensors(t, DType::F32, device)
    }

    #[test]
    fn encoder_forward_shape_and_downsample_ratio() {
        let device = Device::Cpu;
        let cfg = small_cfg(false);
        let vb = make_vb(&cfg, &device);
        let encoder = load_encoder(&cfg, vb.pp("encoder")).expect("load encoder");

        let t_in = 64usize;
        let x = Tensor::rand(-1f32, 1f32, (1, cfg.channels, t_in), &device).unwrap();
        let y = encoder.encode(&x).expect("encode");
        assert_eq!(y.dim(0).unwrap(), 1);
        assert_eq!(y.dim(1).unwrap(), cfg.vae_dim);
        // Total downsample ratio is the product of `encoder_ratios` (2*2=4 here).
        let hop: usize = cfg.encoder_ratios.iter().product();
        let expected_t = t_in.div_ceil(hop);
        assert_eq!(y.dim(2).unwrap(), expected_t);
    }

    #[test]
    fn encoder_decoder_roundtrip_shape_matches_upsample_ratio() {
        let device = Device::Cpu;
        let cfg = small_cfg(true);
        let vb = make_vb(&cfg, &device);
        let encoder = load_encoder(&cfg, vb.pp("encoder")).expect("load encoder");
        let decoder = load_decoder(&cfg, vb.pp("decoder")).expect("load decoder");

        let t_in = 64usize;
        let x = Tensor::rand(-1f32, 1f32, (1, cfg.channels, t_in), &device).unwrap();
        let latents = encoder.encode(&x).expect("encode");
        let recon = decoder.decode(&latents).expect("decode");

        assert_eq!(recon.dim(0).unwrap(), 1);
        assert_eq!(recon.dim(1).unwrap(), cfg.channels);
        let hop: usize = cfg.encoder_ratios.iter().product();
        let latent_t = t_in.div_ceil(hop);
        assert_eq!(recon.dim(2).unwrap(), latent_t * hop);
    }

    #[test]
    fn conv_layers_produce_finite_nonzero_output() {
        let device = Device::Cpu;
        let cfg = small_cfg(false);
        let vb = make_vb(&cfg, &device);
        let encoder = load_encoder(&cfg, vb.pp("encoder")).expect("load encoder");
        let x = Tensor::rand(-1f32, 1f32, (1, cfg.channels, 64), &device).unwrap();
        let y = encoder.encode(&x).expect("encode");
        let max_abs: f32 = y.abs().unwrap().max_all().unwrap().to_scalar().unwrap();
        assert!(max_abs.is_finite());
        assert!(max_abs > 0.0);
    }
}
