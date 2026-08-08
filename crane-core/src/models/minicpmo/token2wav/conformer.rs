//! CosyVoice2's `UpsampleConformerEncoderV2` — the first stage of the
//! Token2wav vocoder, turning (prompt + generated) speech-token embeddings
//! into a `mu` conditioning sequence for the DiT/CFM decoder.
//!
//! Ported from `stepaudio2.cosyvoice2.transformer.upsample_encoder_v2` (real
//! reference source, from the `minicpmo-utils` PyPI sdist — see the plan
//! doc). Despite the "conformer" name, these blocks have **no macaron FFN
//! and no convolution module** — just relative-position MHA + a single FFN,
//! pre-norm. Non-streaming (`forward`) only; the `forward_chunk` /
//! cache-based streaming path isn't ported (out of scope, see the
//! `minicpmo` module doc). Single sequence, no padding (`B=1`), so real
//! attention masking is skipped — every position attends to every other.

use candle_core::{Module, Result, Tensor};
use candle_nn::{layer_norm, linear, linear_no_bias, Activation, LayerNorm, Linear, VarBuilder};

// ── Relative positional encoding ────────────────────────────────────────

/// Fixed sinusoidal Transformer-XL-style relative position table (**not**
/// RoPE). Precomputes both positive and negative offsets once up to
/// `max_len`; `position_encoding(size)` slices out the `2*size-1`-wide
/// window centered on offset 0, matching
/// `EspnetRelPositionalEncoding.position_encoding`.
pub struct RelPositionalEncoding {
    /// `[1, 2*max_len-1, d_model]`.
    pe: Tensor,
    xscale: f64,
}

impl RelPositionalEncoding {
    pub fn new(d_model: usize, max_len: usize, device: &candle_core::Device, dtype: candle_core::DType) -> Result<Self> {
        let half = d_model / 2;
        let mut div_term = vec![0f32; half];
        for (i, d) in div_term.iter_mut().enumerate() {
            *d = (-(f64::from((2 * i) as u32)) * (10000f64).ln() / d_model as f64).exp() as f32;
        }

        let mut pe_positive = vec![0f32; max_len * d_model];
        let mut pe_negative = vec![0f32; max_len * d_model];
        for pos in 0..max_len {
            for i in 0..half {
                let angle = pos as f32 * div_term[i];
                pe_positive[pos * d_model + 2 * i] = angle.sin();
                pe_positive[pos * d_model + 2 * i + 1] = angle.cos();
                let neg_angle = -(pos as f32) * div_term[i];
                pe_negative[pos * d_model + 2 * i] = neg_angle.sin();
                pe_negative[pos * d_model + 2 * i + 1] = neg_angle.cos();
            }
        }
        // pe_positive: reversed row order, rows [max_len-1 .. 0].
        // pe_negative: rows [1 .. max_len) (skip row 0, duplicate of positive's last row).
        let mut pe = Vec::with_capacity((2 * max_len - 1) * d_model);
        for pos in (0..max_len).rev() {
            pe.extend_from_slice(&pe_positive[pos * d_model..(pos + 1) * d_model]);
        }
        for pos in 1..max_len {
            pe.extend_from_slice(&pe_negative[pos * d_model..(pos + 1) * d_model]);
        }
        let pe = Tensor::from_vec(pe, (1, 2 * max_len - 1, d_model), device)?.to_dtype(dtype)?;

        Ok(Self { pe, xscale: (d_model as f64).sqrt() })
    }

    /// `x`: `[1, T, d_model]`. Returns `(x*xscale, pos_emb [1, 2*T-1, d_model])`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let t = x.dim(1)?;
        let scaled = (x * self.xscale)?;
        let pos_emb = self.position_encoding(t)?;
        Ok((scaled, pos_emb))
    }

    pub fn position_encoding(&self, size: usize) -> Result<Tensor> {
        let total = self.pe.dim(1)?;
        let center = total / 2;
        let start = center + 1 - size;
        let len = 2 * size - 1;
        self.pe.narrow(1, start, len)
    }
}

// ── Linear (no) subsampling ─────────────────────────────────────────────

pub struct LinearNoSubsampling {
    linear: Linear,
    norm: LayerNorm,
    pos_enc: RelPositionalEncoding,
}

impl LinearNoSubsampling {
    pub fn new(idim: usize, odim: usize, max_len: usize, vb: VarBuilder) -> Result<Self> {
        let linear = linear(idim, odim, vb.pp("out").pp(0))?;
        let norm = layer_norm(odim, 1e-5, vb.pp("out").pp(1))?;
        let pos_enc = RelPositionalEncoding::new(odim, max_len, vb.device(), vb.dtype())?;
        Ok(Self { linear, norm, pos_enc })
    }

    /// `x`: `[1, T, idim]`. Returns `(x [1, T, odim], pos_emb)`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = self.norm.forward(&self.linear.forward(x)?)?;
        self.pos_enc.forward(&x)
    }
}

// ── Relative-position multi-head attention ──────────────────────────────

struct RelPositionMultiHeadedAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl RelPositionMultiHeadedAttention {
    fn new(num_heads: usize, dim: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            linear_q: linear(dim, dim, vb.pp("linear_q"))?,
            linear_k: linear(dim, dim, vb.pp("linear_k"))?,
            linear_v: linear(dim, dim, vb.pp("linear_v"))?,
            linear_out: linear(dim, dim, vb.pp("linear_out"))?,
            linear_pos: linear_no_bias(dim, dim, vb.pp("linear_pos"))?,
            pos_bias_u: vb.get((num_heads, head_dim), "pos_bias_u")?,
            pos_bias_v: vb.get((num_heads, head_dim), "pos_bias_v")?,
            num_heads,
            head_dim,
        })
    }

    /// Transformer-XL rel-shift trick: `[b, h, t1, 2*t1-1] -> [b, h, t1, t1]`.
    fn rel_shift(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, t1, n) = x.dims4()?;
        let zero_pad = Tensor::zeros((b, h, t1, 1), x.dtype(), x.device())?;
        let x_padded = Tensor::cat(&[&zero_pad, x], 3)?; // [b, h, t1, n+1]
        let x_padded = x_padded.reshape((b, h, n + 1, t1))?;
        let x = x_padded.narrow(2, 1, n)?.contiguous()?.reshape((b, h, t1, n))?;
        x.narrow(3, 0, n / 2 + 1)
    }

    /// `x`: `[1, T, dim]`, `pos_emb`: `[1, 2*T-1, dim]`. Returns `[1, T, dim]`.
    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let q = self.linear_q.forward(x)?.reshape((b, t, self.num_heads, self.head_dim))?;
        let k = self.linear_k.forward(x)?.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.linear_v.forward(x)?.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        let (pb, tp, _) = pos_emb.dims3()?;
        let p = self.linear_pos.forward(pos_emb)?.reshape((pb, tp, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        let q_with_u = q.broadcast_add(&self.pos_bias_u)?.transpose(1, 2)?.contiguous()?; // [b,h,t,d]
        let q_with_v = q.broadcast_add(&self.pos_bias_v)?.transpose(1, 2)?.contiguous()?;

        let matrix_ac = q_with_u.matmul(&k.transpose(2, 3)?.contiguous()?)?; // [b,h,t,t]
        let matrix_bd = q_with_v.matmul(&p.transpose(2, 3)?.contiguous()?)?; // [b,h,t,2t-1]
        let matrix_bd = self.rel_shift(&matrix_bd)?; // [b,h,t,t]

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = ((matrix_ac + matrix_bd)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores.to_dtype(candle_core::DType::F32)?)?.to_dtype(scores.dtype())?;
        let out = attn.matmul(&v)?; // [b,h,t,d]
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, self.num_heads * self.head_dim))?;
        self.linear_out.forward(&out)
    }
}

// ── Feed forward ─────────────────────────────────────────────────────────

struct PositionwiseFeedForward {
    w1: Linear,
    w2: Linear,
}

impl PositionwiseFeedForward {
    fn new(dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { w1: linear(dim, hidden, vb.pp("w_1"))?, w2: linear(hidden, dim, vb.pp("w_2"))? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.w2.forward(&Activation::Silu.forward(&self.w1.forward(x)?)?)
    }
}

// ── Encoder layer (no macaron FFN, no conv module) ──────────────────────

struct ConformerEncoderLayer {
    self_attn: RelPositionMultiHeadedAttention,
    feed_forward: PositionwiseFeedForward,
    norm_mha: LayerNorm,
    norm_ff: LayerNorm,
}

impl ConformerEncoderLayer {
    fn new(dim: usize, num_heads: usize, ffn_hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: RelPositionMultiHeadedAttention::new(num_heads, dim, vb.pp("self_attn"))?,
            feed_forward: PositionwiseFeedForward::new(dim, ffn_hidden, vb.pp("feed_forward"))?,
            norm_mha: layer_norm(dim, 1e-12, vb.pp("norm_mha"))?,
            norm_ff: layer_norm(dim, 1e-12, vb.pp("norm_ff"))?,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let residual = x;
        let normed = self.norm_mha.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, pos_emb)?;
        let x = (residual + attn_out)?;

        let residual = &x;
        let normed = self.norm_ff.forward(&x)?;
        let ff_out = self.feed_forward.forward(&normed)?;
        residual + ff_out
    }
}

// ── Pre-lookahead layer ──────────────────────────────────────────────────

struct PreLookaheadLayer {
    conv1: candle_nn::Conv1d,
    conv2: candle_nn::Conv1d,
    pre_lookahead_len: usize,
}

impl PreLookaheadLayer {
    fn new(channels: usize, pre_lookahead_len: usize, vb: VarBuilder) -> Result<Self> {
        let conv1_cfg = candle_nn::Conv1dConfig { padding: 0, stride: 1, ..Default::default() };
        let conv1 = candle_nn::conv1d(channels, channels, pre_lookahead_len + 1, conv1_cfg, vb.pp("conv1"))?;
        let conv2_cfg = candle_nn::Conv1dConfig { padding: 0, stride: 1, ..Default::default() };
        let conv2 = candle_nn::conv1d(channels, channels, 3, conv2_cfg, vb.pp("conv2"))?;
        Ok(Self { conv1, conv2, pre_lookahead_len })
    }

    /// `x`: `[1, T, C]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xt = x.transpose(1, 2)?.contiguous()?; // [1, C, T]
        let xt = xt.pad_with_zeros(2, 0, self.pre_lookahead_len)?;
        let xt = self.conv1.forward(&xt)?;
        let xt = candle_nn::ops::leaky_relu(&xt, 0.01)?;
        let xt = xt.pad_with_zeros(2, 2, 0)?;
        let xt = self.conv2.forward(&xt)?;
        let out = xt.transpose(1, 2)?.contiguous()?; // [1, T, C]
        out + x
    }
}

// ── Upsample1D ────────────────────────────────────────────────────────────

struct Upsample1D {
    conv: candle_nn::Conv1d,
    stride: usize,
}

impl Upsample1D {
    fn new(channels: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv1dConfig { padding: 0, stride: 1, ..Default::default() };
        let conv = candle_nn::conv1d(channels, channels, stride * 2 + 1, conv_cfg, vb.pp("conv"))?;
        Ok(Self { conv, stride })
    }

    /// `x`: `[1, C, T]` -> `[1, C, T*stride]` (nearest 2x upsample, causal pad, conv).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (_b, _c, t) = x.dims3()?;
        // `scale_factor=stride` nearest interpolation: repeat each frame `stride` times.
        let up = x.unsqueeze(3)?.repeat((1, 1, 1, self.stride))?.reshape((x.dim(0)?, x.dim(1)?, t * self.stride))?;
        let up = up.pad_with_zeros(2, self.stride * 2, 0)?;
        self.conv.forward(&up)
    }
}

// ── Full encoder ──────────────────────────────────────────────────────────

pub struct UpsampleConformerEncoderV2 {
    embed: LinearNoSubsampling,
    pre_lookahead_layer: PreLookaheadLayer,
    encoders: Vec<ConformerEncoderLayer>,
    up_layer: Upsample1D,
    up_embed: LinearNoSubsampling,
    up_encoders: Vec<ConformerEncoderLayer>,
    after_norm: LayerNorm,
}

impl UpsampleConformerEncoderV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_size: usize,
        output_size: usize,
        num_blocks: usize,
        num_up_blocks: usize,
        pre_lookahead_len: usize,
        up_stride: usize,
        attention_heads: usize,
        linear_units: usize,
        max_len: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let embed = LinearNoSubsampling::new(input_size, output_size, max_len, vb.pp("embed"))?;
        let pre_lookahead_layer = PreLookaheadLayer::new(output_size, pre_lookahead_len, vb.pp("pre_lookahead_layer"))?;
        let vb_enc = vb.pp("encoders");
        let mut encoders = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            encoders.push(ConformerEncoderLayer::new(output_size, attention_heads, linear_units, vb_enc.pp(i))?);
        }
        let up_layer = Upsample1D::new(output_size, up_stride, vb.pp("up_layer"))?;
        let up_embed = LinearNoSubsampling::new(input_size, output_size, max_len * up_stride, vb.pp("up_embed"))?;
        let vb_up = vb.pp("up_encoders");
        let mut up_encoders = Vec::with_capacity(num_up_blocks);
        for i in 0..num_up_blocks {
            up_encoders.push(ConformerEncoderLayer::new(output_size, attention_heads, linear_units, vb_up.pp(i))?);
        }
        let after_norm = layer_norm(output_size, 1e-5, vb.pp("after_norm"))?;

        Ok(Self { embed, pre_lookahead_layer, encoders, up_layer, up_embed, up_encoders, after_norm })
    }

    /// `xs`: `[1, T, input_size]`. Returns `[1, T*up_stride, output_size]`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (xs, pos_emb) = self.embed.forward(xs)?;
        let mut xs = self.pre_lookahead_layer.forward(&xs)?;
        for layer in &self.encoders {
            xs = layer.forward(&xs, &pos_emb)?;
        }

        let xs_t = xs.transpose(1, 2)?.contiguous()?; // [1, C, T]
        let xs_t = self.up_layer.forward(&xs_t)?;
        let xs = xs_t.transpose(1, 2)?.contiguous()?; // [1, T*stride, C]

        let (mut xs, pos_emb) = self.up_embed.forward(&xs)?;
        for layer in &self.up_encoders {
            xs = layer.forward(&xs, &pos_emb)?;
        }
        self.after_norm.forward(&xs)
    }
}
