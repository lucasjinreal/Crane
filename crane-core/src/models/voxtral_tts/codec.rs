//! Audio codec decoder for Voxtral TTS.
//!
//! Converts 37 codes per frame (1 semantic + 36 acoustic) into a 24 kHz
//! waveform.  The decoder pipeline is:
//!
//! ```text
//! codes [n_frames, 37]
//!   → embed (semantic VQ lookup + FSQ decode) → [292, T]
//!   → CausalConv1d(292→1024, k=3, s=1)
//!   → 4 stages × (Transformer layers + optional upsample ConvTranspose1d)
//!   → CausalConv1d(1024→240, k=7, s=1)
//!   → reshape → waveform
//! ```
//!
//! The codec transformer layers use `ALiBi` attention with sliding windows,
//! QK norm, `LayerScale`, and weight-norm convolutions — sufficiently
//! different from the shared [`TransformerBlock`](super::super::modules::transformer::TransformerBlock)
//! that they are kept self-contained here.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{
    linear_no_bias, Activation, Conv1d as CandleConv1d, Conv1dConfig,
    ConvTranspose1d as CandleConvTranspose1d, ConvTranspose1dConfig, Linear, VarBuilder,
};

use crate::models::with_tracing::RmsNorm;

use super::model::AudioTokenizerArgs;

// ── Constants ────────────────────────────────────────────────────────────────

const SPECIAL_TOKEN_OFFSET: f64 = 2.0;

// ── Weight norm reconstruction ───────────────────────────────────────────────

/// Reconstruct effective weight from `PyTorch` weight-norm parameters.
///
/// `weight_v` has shape `[N, ...]` and `weight_g` has shape `[N, 1, 1]`
/// (or any shape that flattens to `[N]`).  For each output channel `n`:
///
/// ```text
/// w[n] = v[n] × (g[n] / ‖v[n]‖₂)
/// ```
///
/// Computation is performed in `f32` and converted back to the original dtype.
fn reconstruct_weight_norm(weight_v: &Tensor, weight_g: &Tensor) -> Result<Tensor> {
    let original_dtype = weight_v.dtype();
    let v = weight_v.to_dtype(DType::F32)?;
    let g = weight_g.to_dtype(DType::F32)?.flatten_all()?;

    let shape = v.dims().to_vec();
    let n = shape[0];
    let fan_in: usize = shape[1..].iter().product();

    let v_flat = v.reshape((n, fan_in))?;
    let norm = ((v_flat.sqr()?.sum_keepdim(1))? + 1e-12)?.sqrt()?;

    let scale = (g.reshape((n, 1))? / norm)?;
    let effective = v_flat.broadcast_mul(&scale)?;
    effective.reshape(shape)?.to_dtype(original_dtype)
}

// ── Reflect padding ──────────────────────────────────────────────────────────

/// Left-side reflect padding for causal convolutions.
///
/// For input `[B, C, T]` and `left_pad = p`, produces `[B, C, T + p]` where
/// positions `[-p, ..., -1]` mirror positions `[p, ..., 1]` from the input.
#[allow(clippy::cast_possible_wrap)]
fn reflect_pad_1d(x: &Tensor, left_pad: usize) -> Result<Tensor> {
    if left_pad == 0 {
        return Ok(x.clone());
    }
    let t = x.dim(2)?;
    let total = left_pad + t;
    let mut indices: Vec<i64> = Vec::with_capacity(total);
    for i in (1..=left_pad).rev() {
        indices.push(i.min(t - 1) as i64);
    }
    for i in 0..t {
        indices.push(i as i64);
    }
    let idx = Tensor::new(indices.as_slice(), x.device())?;
    x.contiguous()?.index_select(&idx, 2)
}

// ── ALiBi helpers ────────────────────────────────────────────────────────────

/// Precompute `ALiBi` slopes: `slopes[h] = 2^(-8/n_heads × (h+1))`.
#[allow(clippy::cast_precision_loss)]
fn compute_alibi_slopes(n_heads: usize) -> Vec<f32> {
    (0..n_heads)
        .map(|h| 2f32.powf(-8.0 / n_heads as f32 * (h + 1) as f32))
        .collect()
}

/// Build a combined causal + sliding-window + `ALiBi` bias mask.
///
/// Shape: `[1, n_heads, seq_len, seq_len]`.  Valid positions get an `ALiBi`
/// bias `slope[h] × (j − i)` (always ≤ 0 for causal); blocked positions
/// get `−∞`.
#[allow(clippy::cast_precision_loss)]
fn build_alibi_sliding_mask(
    seq_len: usize,
    window: usize,
    slopes: &[f32],
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let n_heads = slopes.len();
    let mut data = vec![0f32; n_heads * seq_len * seq_len];
    for (h, &slope) in slopes.iter().enumerate() {
        for i in 0..seq_len {
            let k_start = if window > 0 && i + 1 > window {
                i + 1 - window
            } else {
                0
            };
            for j in 0..seq_len {
                let idx = h * seq_len * seq_len + i * seq_len + j;
                if j > i || j < k_start {
                    data[idx] = f32::NEG_INFINITY;
                } else {
                    data[idx] = slope * (j as f32 - i as f32);
                }
            }
        }
    }
    Tensor::from_vec(data, (1, n_heads, seq_len, seq_len), device)?.to_dtype(dtype)
}

// ── Causal Conv1d ────────────────────────────────────────────────────────────

/// Weight-normed causal `Conv1d` with reflect left-padding.
#[derive(Debug, Clone)]
struct CausalConv1d {
    conv: CandleConv1d,
    left_pad: usize,
}

impl CausalConv1d {
    /// Load a weight-normed causal `Conv1d` from the checkpoint.
    ///
    /// Expects weights at `{vb_prefix}.conv.parametrizations.weight.original0/1`
    /// and an optional bias at `{vb_prefix}.conv.bias`.
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let param_vb = vb.pp("conv").pp("parametrizations").pp("weight");
        let weight_g = param_vb.get((out_ch, 1, 1), "original0")?;
        let weight_v = param_vb.get((out_ch, in_ch, kernel), "original1")?;
        let weight = reconstruct_weight_norm(&weight_v, &weight_g)?;

        let bias = vb.pp("conv").get(out_ch, "bias").ok();

        let config = Conv1dConfig {
            stride,
            ..Default::default()
        };
        let left_pad = kernel - stride;
        Ok(Self {
            conv: CandleConv1d::new(weight, bias, config),
            left_pad,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let padded = reflect_pad_1d(x, self.left_pad)?;
        padded.apply(&self.conv)
    }
}

// ── Causal ConvTranspose1d ───────────────────────────────────────────────────

/// Weight-normed causal `ConvTranspose1d` with right-side trimming.
#[derive(Debug, Clone)]
struct CausalConvTranspose1d {
    conv: CandleConvTranspose1d,
    right_trim: usize,
}

impl CausalConvTranspose1d {
    /// Load a weight-normed causal `ConvTranspose1d`.
    ///
    /// The weight shape for transposed convolutions is `[in_ch, out_ch, kernel]`.
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let param_vb = vb.pp("conv").pp("parametrizations").pp("weight");
        let weight_g = param_vb.get((in_ch, 1, 1), "original0")?;
        let weight_v = param_vb.get((in_ch, out_ch, kernel), "original1")?;
        let weight = reconstruct_weight_norm(&weight_v, &weight_g)?;

        let bias = vb.pp("conv").get(out_ch, "bias").ok();

        let config = ConvTranspose1dConfig {
            stride,
            ..Default::default()
        };
        let right_trim = kernel - stride;
        Ok(Self {
            conv: CandleConvTranspose1d::new(weight, bias, config),
            right_trim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.apply(&self.conv)?;
        if self.right_trim == 0 {
            return Ok(y);
        }
        let t = y.dim(D::Minus1)?;
        y.narrow(D::Minus1, 0, t.saturating_sub(self.right_trim))
    }
}

// ── Codec attention ──────────────────────────────────────────────────────────

/// MHA with QK norm for codec transformer layers.
///
/// Uses `ALiBi` positional bias (passed in as a precomputed mask) instead of
/// `RoPE`.  No KV cache — the codec processes fixed-length sequences.
#[derive(Debug, Clone)]
struct CodecAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl CodecAttention {
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        qk_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attn_vb = vb.pp("attention");
        Ok(Self {
            wq: linear_no_bias(dim, n_heads * head_dim, attn_vb.pp("wq"))?,
            wk: linear_no_bias(dim, n_kv_heads * head_dim, attn_vb.pp("wk"))?,
            wv: linear_no_bias(dim, n_kv_heads * head_dim, attn_vb.pp("wv"))?,
            wo: linear_no_bias(n_heads * head_dim, dim, attn_vb.pp("wo"))?,
            q_norm: RmsNorm::new(n_heads * head_dim, qk_norm_eps, attn_vb.pp("q_norm"))?,
            k_norm: RmsNorm::new(n_kv_heads * head_dim, qk_norm_eps, attn_vb.pp("k_norm"))?,
            n_heads,
            n_kv_heads,
            head_dim,
        })
    }

    /// Forward pass with a precomputed `ALiBi` + sliding-window mask.
    ///
    /// `x` has shape `[B, T, dim]`.  `mask` has shape `[1, n_heads, T, T]`.
    #[allow(clippy::many_single_char_names, clippy::cast_precision_loss)]
    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (batch, seq, _) = x.dims3()?;

        let q = x.apply(&self.wq)?;
        let k = x.apply(&self.wk)?;
        let v = x.apply(&self.wv)?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q = q
            .reshape((batch, seq, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let rep = self.n_heads / self.n_kv_heads;
        let k = repeat_kv(&k, rep)?;
        let v = repeat_kv(&v, rep)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let scores = scores.broadcast_add(mask)?;

        let input_dtype = scores.dtype();
        let attn = candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
            .to_dtype(input_dtype)?;

        let out = attn.matmul(&v.contiguous()?)?.transpose(1, 2)?.reshape((
            batch,
            seq,
            self.n_heads * self.head_dim,
        ))?;
        out.apply(&self.wo)
    }
}

/// Repeat KV heads to match query head count (for GQA/MQA).
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, kv_h, t, d) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, kv_h, n_rep, t, d))?
        .reshape((b, kv_h * n_rep, t, d))
}

// ── LayerScale ───────────────────────────────────────────────────────────────

/// Per-channel multiplicative scaling applied to residual branches.
#[derive(Debug, Clone)]
struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    #[allow(clippy::needless_pass_by_value)]
    fn new(dim: usize, vb: VarBuilder, name: &str) -> Result<Self> {
        let scale = vb.get(dim, name)?;
        Ok(Self { scale })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let c = self.scale.dims1()?;
        x.broadcast_mul(&self.scale.reshape((1, 1, c))?)
    }
}

// ── Codec FFN ────────────────────────────────────────────────────────────────

/// `SwiGLU` FFN using Mistral-style weight names (w1/w2/w3).
#[derive(Debug, Clone)]
struct CodecFfn {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl CodecFfn {
    #[allow(clippy::needless_pass_by_value)]
    fn new(dim: usize, hidden_dim: usize, vb: VarBuilder) -> Result<Self> {
        let ffn_vb = vb.pp("feed_forward");
        Ok(Self {
            w1: linear_no_bias(dim, hidden_dim, ffn_vb.pp("w1"))?,
            w2: linear_no_bias(hidden_dim, dim, ffn_vb.pp("w2"))?,
            w3: linear_no_bias(dim, hidden_dim, ffn_vb.pp("w3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = x.apply(&self.w1)?.apply(&Activation::Silu)?;
        let up = x.apply(&self.w3)?;
        (gate * up)?.apply(&self.w2)
    }
}

// ── Codec transformer layer ──────────────────────────────────────────────────

/// Single codec transformer layer: pre-norm attention + `LayerScale` +
/// pre-norm FFN + `LayerScale`.
#[derive(Debug, Clone)]
struct CodecTransformerLayer {
    attention: CodecAttention,
    ffn: CodecFfn,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    attn_scale: LayerScale,
    ffn_scale: LayerScale,
}

impl CodecTransformerLayer {
    #[allow(clippy::needless_pass_by_value)]
    fn new(cfg: &AudioTokenizerArgs, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attention: CodecAttention::new(
                cfg.dim,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.qk_norm_eps,
                vb.clone(),
            )?,
            ffn: CodecFfn::new(cfg.dim, cfg.hidden_dim, vb.clone())?,
            attention_norm: RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("attention_norm"))?,
            ffn_norm: RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("ffn_norm"))?,
            attn_scale: LayerScale::new(cfg.dim, vb.clone(), "attention_scale")?,
            ffn_scale: LayerScale::new(cfg.dim, vb, "ffn_scale")?,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let h = self.attention_norm.forward(x)?;
        let h = self.attention.forward(&h, mask)?;
        let x = (residual + self.attn_scale.forward(&h)?)?;

        let residual = x.clone();
        let h = self.ffn_norm.forward(&x)?;
        let h = self.ffn.forward(&h)?;
        residual + self.ffn_scale.forward(&h)?
    }
}

// ── Codec stage ──────────────────────────────────────────────────────────────

/// One decoder stage: transformer layers followed by an optional upsample conv.
#[derive(Debug, Clone)]
struct CodecStage {
    layers: Vec<CodecTransformerLayer>,
    upsample: Option<CausalConvTranspose1d>,
    window_size: usize,
}

// ── Top-level codec decoder ──────────────────────────────────────────────────

/// Audio codec decoder that converts frame codes to a 24 kHz waveform.
///
/// # Weight prefix
///
/// All weights are loaded from the `audio_tokenizer` prefix in the checkpoint.
/// The codec uses original Mistral-style weight names (not renamed by
/// [`rename_voxtral_transformer_keys`](super::modeling::rename_voxtral_transformer_keys)).
#[derive(Debug, Clone)]
pub struct CodecDecoder {
    semantic_codebook: Tensor,
    input_conv: CausalConv1d,
    stages: Vec<CodecStage>,
    output_conv: CausalConv1d,
    alibi_slopes: Vec<f32>,
    semantic_dim: usize,
    acoustic_dim: usize,
    fsq_levels: usize,
}

impl CodecDecoder {
    /// Construct the codec decoder from checkpoint weights.
    ///
    /// `vb` should be scoped to the `audio_tokenizer` prefix.
    ///
    /// # Errors
    ///
    /// Returns a candle error if any weight is missing or has an unexpected
    /// shape.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(cfg: &AudioTokenizerArgs, vb: VarBuilder) -> Result<Self> {
        let semantic_codebook = Self::load_semantic_codebook(cfg, vb.clone())?;
        let alibi_slopes = compute_alibi_slopes(cfg.n_heads);

        let embed_dim = cfg.semantic_dim + cfg.acoustic_dim;
        let input_conv = CausalConv1d::new(
            embed_dim,
            cfg.dim,
            cfg.decoder_convs_kernels[0],
            cfg.decoder_convs_strides[0],
            vb.pp("decoder_blocks.0"),
        )?;

        let n_stages = cfg.decoder_transformer_lengths.len();
        if cfg.decoder_convs_kernels.len() < n_stages || cfg.decoder_convs_strides.len() < n_stages
        {
            candle_core::bail!(
                "decoder_convs_kernels/strides must have at least {n_stages} entries, \
                 got {}/{}",
                cfg.decoder_convs_kernels.len(),
                cfg.decoder_convs_strides.len(),
            );
        }
        debug_assert!(n_stages <= 4, "window_sizes formula assumes <= 4 stages");
        let window_sizes: Vec<usize> = (0..n_stages).map(|s| 2 << s).collect();

        let mut stages = Vec::with_capacity(n_stages);
        let mut block_idx = 1usize;

        for (stage_idx, (&n_layers, &window)) in cfg
            .decoder_transformer_lengths
            .iter()
            .zip(window_sizes.iter())
            .enumerate()
        {
            let mut layers = Vec::with_capacity(n_layers);
            for l in 0..n_layers {
                let layer_vb = vb
                    .pp(format!("decoder_blocks.{block_idx}"))
                    .pp("layers")
                    .pp(l);
                layers.push(CodecTransformerLayer::new(cfg, layer_vb)?);
            }
            block_idx += 1;

            let upsample = if stage_idx < n_stages - 1 {
                let conv = CausalConvTranspose1d::new(
                    cfg.dim,
                    cfg.dim,
                    cfg.decoder_convs_kernels[stage_idx + 1],
                    cfg.decoder_convs_strides[stage_idx + 1],
                    vb.pp(format!("decoder_blocks.{block_idx}")),
                )?;
                block_idx += 1;
                Some(conv)
            } else {
                None
            };

            stages.push(CodecStage {
                layers,
                upsample,
                window_size: window,
            });
        }

        let output_conv = CausalConv1d::new(
            cfg.dim,
            cfg.pretransform_patch_size,
            7,
            1,
            vb.pp("output_proj"),
        )?;

        Ok(Self {
            semantic_codebook,
            input_conv,
            stages,
            output_conv,
            alibi_slopes,
            semantic_dim: cfg.semantic_dim,
            acoustic_dim: cfg.acoustic_dim,
            fsq_levels: cfg.acoustic_codebook_size,
        })
    }

    /// Load and precompute the semantic codebook: `embedding_sum / cluster_usage`.
    #[allow(clippy::needless_pass_by_value)]
    fn load_semantic_codebook(cfg: &AudioTokenizerArgs, vb: VarBuilder) -> Result<Tensor> {
        let cb_vb = vb.pp("quantizer").pp("semantic_codebook");
        let embedding_sum = cb_vb
            .get(
                (cfg.semantic_codebook_size, cfg.semantic_dim),
                "embedding_sum",
            )?
            .to_dtype(DType::F32)?;
        let cluster_usage = cb_vb
            .get(cfg.semantic_codebook_size, "cluster_usage")?
            .to_dtype(DType::F32)?;
        let cluster_usage = cluster_usage.clamp(1e-5, f64::MAX)?;
        let codebook = embedding_sum
            .broadcast_div(&cluster_usage.reshape((cfg.semantic_codebook_size, 1))?)?;
        Ok(codebook)
    }

    /// Embed frame codes into the latent space.
    ///
    /// Input `codes` has shape `[B, n_frames, 37]` (f32, with special token
    /// offset 2 already included).  Returns `[B, 292, n_frames]` (channel-first).
    #[allow(clippy::cast_precision_loss)]
    fn embed_codes(&self, codes: &Tensor) -> Result<Tensor> {
        let (b, n_frames, _) = codes.dims3()?;

        let sem_codes = codes.narrow(2, 0, 1)?.squeeze(2)?;
        let sem_indices = (sem_codes - SPECIAL_TOKEN_OFFSET)?
            .clamp(0.0, (self.semantic_codebook.dim(0)? - 1) as f64)?
            .to_dtype(DType::U32)?
            .flatten_all()?;
        let sem_emb = self
            .semantic_codebook
            .index_select(&sem_indices, 0)?
            .reshape((b, n_frames, self.semantic_dim))?;

        let ac_codes = codes.narrow(2, 1, self.acoustic_dim)?;
        let ac_codes =
            (ac_codes - SPECIAL_TOKEN_OFFSET)?.clamp(0.0, (self.fsq_levels - 1) as f64)?;
        let ac_vals = ((ac_codes * (2.0 / (self.fsq_levels - 1) as f64))? - 1.0)?;

        let embedded = Tensor::cat(&[sem_emb, ac_vals], 2)?;
        embedded.transpose(1, 2)
    }

    /// Decode frame codes to a waveform.
    ///
    /// `codes` has shape `[B, n_frames, 37]` — integer codes cast to f32 with
    /// the special token offset (2) included.  Returns `[B, n_samples]` where
    /// `n_samples = n_frames × total_upsample × patch_size`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if tensor operations fail.
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let (b, _n_frames, _) = codes.dims3()?;

        let mut x = self.embed_codes(codes)?;
        x = self.input_conv.forward(&x)?;
        x = x.transpose(1, 2)?;

        for stage in &self.stages {
            let seq_len = x.dim(1)?;
            let mask = build_alibi_sliding_mask(
                seq_len,
                stage.window_size,
                &self.alibi_slopes,
                x.dtype(),
                x.device(),
            )?;

            for layer in &stage.layers {
                x = layer.forward(&x, &mask)?;
            }

            if let Some(upsample) = &stage.upsample {
                x = x.transpose(1, 2)?;
                x = upsample.forward(&x)?;
                x = x.transpose(1, 2)?;
            }
        }

        x = x.transpose(1, 2)?;
        x = self.output_conv.forward(&x)?;

        let patch_size = x.dim(1)?;
        let t_final = x.dim(2)?;
        x.transpose(1, 2)?
            .contiguous()?
            .reshape((b, t_final * patch_size))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_alibi_slopes() {
        let slopes = compute_alibi_slopes(8);
        assert_eq!(slopes.len(), 8);
        for (h, &s) in slopes.iter().enumerate() {
            let expected = 2f32.powf(-8.0 / 8.0 * (h + 1) as f32);
            assert!(
                (s - expected).abs() < 1e-7,
                "slope[{h}]: got {s}, expected {expected}"
            );
        }
        assert!((slopes[0] - 0.5).abs() < 1e-7);
        assert!((slopes[7] - 2f32.powf(-8.0)).abs() < 1e-7);
    }

    #[test]
    fn test_fsq_round_trip() {
        let levels: usize = 21;
        for code in 0..levels {
            let val = code as f64 * 2.0 / (levels - 1) as f64 - 1.0;
            let recovered = ((val + 1.0) / 2.0 * (levels - 1) as f64).round() as usize;
            assert_eq!(
                code, recovered,
                "FSQ round-trip failed for code={code}, val={val}"
            );
        }
    }

    #[test]
    fn test_weight_norm_reconstruction() -> Result<()> {
        let dev = &Device::Cpu;
        let weight_v = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], dev)?.reshape((2, 3))?;
        let weight_g = Tensor::new(&[1.0f32, 2.0], dev)?.reshape((2, 1))?;

        let w = reconstruct_weight_norm(&weight_v, &weight_g)?;
        let w_data: Vec<f32> = w.flatten_all()?.to_vec1()?;

        let norm0 = (1.0f32 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0).sqrt();
        let norm1 = (4.0f32 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0).sqrt();
        let expected = [
            1.0 / norm0,
            2.0 / norm0,
            3.0 / norm0,
            2.0 * 4.0 / norm1,
            2.0 * 5.0 / norm1,
            2.0 * 6.0 / norm1,
        ];
        for (i, (&got, &exp)) in w_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "weight_norm[{i}]: got {got}, expected {exp}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_reflect_pad() -> Result<()> {
        let dev = &Device::Cpu;
        let x = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0, 50.0], dev)?.reshape((1, 1, 5))?;

        let padded = reflect_pad_1d(&x, 2)?;
        let vals: Vec<f32> = padded.flatten_all()?.to_vec1()?;
        assert_eq!(vals, vec![30.0, 20.0, 10.0, 20.0, 30.0, 40.0, 50.0]);
        Ok(())
    }

    #[test]
    fn test_reflect_pad_short_sequence() -> Result<()> {
        let dev = &Device::Cpu;
        let x = Tensor::new(&[42.0f32], dev)?.reshape((1, 1, 1))?;
        let padded = reflect_pad_1d(&x, 2)?;
        let vals: Vec<f32> = padded.flatten_all()?.to_vec1()?;
        assert_eq!(vals, vec![42.0, 42.0, 42.0]);
        Ok(())
    }

    #[test]
    fn test_alibi_sliding_mask() -> Result<()> {
        let dev = &Device::Cpu;
        let slopes = vec![0.5f32, 0.25];
        let mask = build_alibi_sliding_mask(4, 2, &slopes, DType::F32, dev)?;
        assert_eq!(mask.dims(), &[1, 2, 4, 4]);

        let data: Vec<f32> = mask.flatten_all()?.to_vec1()?;
        // Head 0, slope=0.5: check diagonal (i=j) => bias=0
        assert_eq!(data[0 * 16 + 0 * 4 + 0], 0.0); // (0,0)
        assert_eq!(data[0 * 16 + 1 * 4 + 1], 0.0); // (1,1)
                                                   // Head 0: (1,0) => slope * (0-1) = -0.5
        assert!((data[0 * 16 + 1 * 4 + 0] - (-0.5)).abs() < 1e-6);
        // Head 0: (0,1) => blocked (future)
        assert!(data[0 * 16 + 0 * 4 + 1].is_infinite());
        // Window=2: (3,0) should be blocked (outside window)
        assert!(data[0 * 16 + 3 * 4 + 0].is_infinite());
        // Window=2: (3,2) should be valid: slope * (2-3) = -0.5
        assert!((data[0 * 16 + 3 * 4 + 2] - (-0.5)).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_causal_conv_output_shape() -> Result<()> {
        let dev = &Device::Cpu;
        let in_ch = 4;
        let out_ch = 8;
        let kernel = 3;
        let stride = 1;
        let t = 10;

        let weight_v = Tensor::randn(0f32, 1.0, (out_ch, in_ch, kernel), dev)?;
        let weight_g = Tensor::ones((out_ch, 1, 1), DType::F32, dev)?;
        let weight = reconstruct_weight_norm(&weight_v, &weight_g)?;
        let bias = Tensor::zeros(out_ch, DType::F32, dev)?;

        let config = Conv1dConfig {
            stride,
            ..Default::default()
        };
        let conv = CausalConv1d {
            conv: CandleConv1d::new(weight, Some(bias), config),
            left_pad: kernel - stride,
        };

        let x = Tensor::randn(0f32, 1.0, (1, in_ch, t), dev)?;
        let y = conv.forward(&x)?;
        assert_eq!(y.dims(), &[1, out_ch, t]);
        Ok(())
    }

    #[test]
    fn test_conv_transpose_output_shape() -> Result<()> {
        let dev = &Device::Cpu;
        let ch = 8;
        let kernel = 4;
        let stride = 2;
        let t = 10;

        let weight_v = Tensor::randn(0f32, 1.0, (ch, ch, kernel), dev)?;
        let weight_g = Tensor::ones((ch, 1, 1), DType::F32, dev)?;
        let weight = reconstruct_weight_norm(&weight_v, &weight_g)?;
        let bias = Tensor::zeros(ch, DType::F32, dev)?;

        let config = ConvTranspose1dConfig {
            stride,
            ..Default::default()
        };
        let conv = CausalConvTranspose1d {
            conv: CandleConvTranspose1d::new(weight, Some(bias), config),
            right_trim: kernel - stride,
        };

        let x = Tensor::randn(0f32, 1.0, (1, ch, t), dev)?;
        let y = conv.forward(&x)?;
        assert_eq!(y.dims(), &[1, ch, t * stride]);
        Ok(())
    }

    #[test]
    fn test_embed_codes() -> Result<()> {
        let dev = &Device::Cpu;
        let semantic_dim = 256;
        let acoustic_dim = 36;
        let n_frames = 5;
        let codebook_size = 8192;

        let codebook = Tensor::randn(0f32, 1.0, (codebook_size, semantic_dim), dev)?;

        let decoder = CodecDecoder {
            semantic_codebook: codebook,
            input_conv: dummy_causal_conv1d(292, 1024, 3, 1, dev)?,
            stages: vec![],
            output_conv: dummy_causal_conv1d(1024, 240, 7, 1, dev)?,
            alibi_slopes: compute_alibi_slopes(8),
            semantic_dim,
            acoustic_dim,
            fsq_levels: 21,
        };

        let mut code_data = vec![0f32; n_frames * 37];
        for t in 0..n_frames {
            code_data[t * 37] = 100.0;
            for i in 1..37 {
                code_data[t * 37 + i] = 12.0;
            }
        }
        let codes = Tensor::new(code_data.as_slice(), dev)?.reshape((1, n_frames, 37))?;

        let emb = decoder.embed_codes(&codes)?;
        assert_eq!(emb.dims(), &[1, 292, n_frames]);

        let emb_vals: Vec<f32> = emb.flatten_all()?.to_vec1()?;
        let ac_val = (12.0 - 2.0) * 2.0 / 20.0 - 1.0;
        assert!(
            (emb_vals[semantic_dim * n_frames] - ac_val).abs() < 1e-5,
            "FSQ decode mismatch"
        );
        Ok(())
    }

    fn dummy_causal_conv1d(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        dev: &Device,
    ) -> Result<CausalConv1d> {
        let weight = Tensor::randn(0f32, 0.01, (out_ch, in_ch, kernel), dev)?;
        let bias = Tensor::zeros(out_ch, DType::F32, dev)?;
        let config = Conv1dConfig {
            stride,
            ..Default::default()
        };
        Ok(CausalConv1d {
            conv: CandleConv1d::new(weight, Some(bias), config),
            left_pad: kernel - stride,
        })
    }

    #[test]
    #[ignore]
    fn test_codec_with_checkpoint() {
        use candle_core::safetensors;
        use std::path::PathBuf;

        let checkpoint_dir = match std::env::var("VOXTRAL_CHECKPOINT_DIR") {
            Ok(d) => PathBuf::from(d),
            Err(_) => {
                eprintln!("VOXTRAL_CHECKPOINT_DIR not set, skipping");
                return;
            }
        };

        let weights_path = checkpoint_dir.join("consolidated.safetensors");
        let dev = &Device::Cpu;

        let tensors = safetensors::load(&weights_path, dev).expect("load weights");
        let vb = VarBuilder::from_tensors(tensors, DType::F32, dev);

        let params_path = checkpoint_dir.join("params.json");
        let params: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&params_path).expect("read params"))
                .expect("parse params");
        let at_args: AudioTokenizerArgs =
            serde_json::from_value(params["audio_tokenizer_args"].clone())
                .expect("parse audio_tokenizer_args");

        let decoder =
            CodecDecoder::new(&at_args, vb.pp("audio_tokenizer")).expect("construct codec");

        let n_frames = 10;
        let mut code_data = vec![0f32; n_frames * 37];
        for t in 0..n_frames {
            code_data[t * 37] = 100.0;
            for i in 1..37 {
                code_data[t * 37 + i] = 12.0;
            }
        }
        let codes = Tensor::new(code_data.as_slice(), dev)
            .unwrap()
            .reshape((1, n_frames, 37))
            .unwrap();

        let waveform = decoder.decode(&codes).expect("decode");
        let expected_samples = n_frames * 8 * 240;
        assert_eq!(waveform.dims(), &[1, expected_samples]);

        let wav_data: Vec<f32> = waveform.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            wav_data.iter().all(|v| v.is_finite()),
            "waveform contains non-finite values"
        );
        eprintln!(
            "Codec decode OK: {n_frames} frames -> {} samples ({:.2}s)",
            wav_data.len(),
            wav_data.len() as f64 / 24000.0
        );
    }
}
