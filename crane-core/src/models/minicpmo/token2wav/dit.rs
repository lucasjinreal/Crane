//! CosyVoice2's `DiT` flow-matching velocity estimator ("DiT-v5" per the
//! reference source's own comment — adds a causal-conv branch to each block
//! "to increase high-freq component", on top of a standard AdaLN-Zero DiT).
//!
//! Ported from `stepaudio2.cosyvoice2.flow.decoder_dit` (real reference
//! source — see the plan doc). Conditioning is **channel-concatenation**
//! (`x`/`mu`/broadcast-`spks`/`cond` stacked along the channel dim before
//! `in_proj`), unlike VoxCPM2's DiT which concatenates along the token
//! (sequence) axis instead — genuinely different mechanism, no shared code
//! with `voxcpm2::local_dit`. Single sequence, no padding (`B` here is the
//! doubled cond/uncond CFG batch, not a real padding batch), so attention
//! runs unmasked — see `cfm.rs`.

use candle_core::{Module, Result, Tensor, D};
use candle_nn::{layer_norm, linear, Conv1d, Conv1dConfig, LayerNorm, Linear, VarBuilder};

/// Parameterless LayerNorm (`elementwise_affine=False` — no learned
/// weight/bias, confirmed against the checkpoint: DiT blocks' `norm1`/`norm2`/
/// `norm3`/`final_layer.norm_final` have no corresponding tensors at all).
/// AdaLN's `modulate()` step supplies the scale/shift instead.
fn norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    centered.broadcast_div(&(var + eps)?.sqrt()?)
}

// ── Timestep embedding ───────────────────────────────────────────────────

struct TimestepEmbedder {
    fc1: Linear,
    fc2: Linear,
    freq_dim: usize,
}

impl TimestepEmbedder {
    fn new(hidden_size: usize, freq_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(freq_dim, hidden_size, vb.pp("mlp").pp(0))?,
            fc2: linear(hidden_size, hidden_size, vb.pp("mlp").pp(2))?,
            freq_dim,
        })
    }

    /// `t`: `[B]`, scalar timesteps in `[0, 1]`. Returns `[B, hidden_size]`.
    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let t = (t * 1000.0)?;
        let half = self.freq_dim / 2;
        let device = t.device();
        let dtype = t.dtype();
        let mut freqs = vec![0f32; half];
        for (i, f) in freqs.iter_mut().enumerate() {
            *f = (-(10000f64.ln()) * i as f64 / half as f64).exp() as f32;
        }
        let freqs = Tensor::from_vec(freqs, half, device)?.to_dtype(dtype)?;
        let t = t.to_dtype(dtype)?;
        let args = t.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?; // [B, half]
        let emb = Tensor::cat(&[args.cos()?, args.sin()?], 1)?; // [B, freq_dim]

        let h = self.fc1.forward(&emb)?.silu()?;
        self.fc2.forward(&h)
    }
}

// ── DiT-internal attention (separate from conformer's — qk-norm, no rel-pos) ──

struct DitAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    q_norm: LayerNorm,
    k_norm: LayerNorm,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl DitAttention {
    fn new(dim: usize, num_heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner = num_heads * head_dim;
        Ok(Self {
            to_q: linear(dim, inner, vb.pp("to_q"))?,
            to_k: linear(dim, inner, vb.pp("to_k"))?,
            to_v: linear(dim, inner, vb.pp("to_v"))?,
            q_norm: layer_norm(head_dim, 1e-5, vb.pp("q_norm"))?,
            k_norm: layer_norm(head_dim, 1e-5, vb.pp("k_norm"))?,
            proj: linear(inner, dim, vb.pp("proj"))?,
            num_heads,
            head_dim,
        })
    }

    /// `x`: `[b, t, c]`. Unmasked (B=1-equivalent, no padding — see module doc).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _c) = x.dims3()?;
        let to_heads = |t_: &Tensor| -> Result<Tensor> {
            t_.reshape((b, t, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()
        };
        let q = self.q_norm.forward(&to_heads(&self.to_q.forward(x)?)?)?;
        let k = self.k_norm.forward(&to_heads(&self.to_k.forward(x)?)?)?;
        let v = to_heads(&self.to_v.forward(x)?)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores.to_dtype(candle_core::DType::F32)?)?.to_dtype(scores.dtype())?;
        let out = attn.matmul(&v)?; // [b, h, t, d]
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, self.num_heads * self.head_dim))?;
        self.proj.forward(&out)
    }
}

// ── MLP (GELU-tanh) ───────────────────────────────────────────────────────

struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { fc1: linear(dim, hidden, vb.pp("fc1"))?, fc2: linear(hidden, dim, vb.pp("fc2"))? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu()?)
    }
}

// ── Causal conv block (zero-padded, kernel-1 left pad) ──────────────────

/// `Conv1d` with `kernel-1` zero-padding on the left only (causal). Distinct
/// from `voxcpm2::audio_vae`'s private `CausalConv1d` (same convention,
/// different module — not cross-module reusable — see that module's doc for
/// the sibling `voxtral_tts` variant which uses *reflect* padding instead).
struct CausalConv1d {
    conv: Conv1d,
    left_pad: usize,
}

impl CausalConv1d {
    fn new(in_ch: usize, out_ch: usize, kernel: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig { padding: 0, stride: 1, ..Default::default() };
        Ok(Self { conv: candle_nn::conv1d(in_ch, out_ch, kernel, cfg, vb)?, left_pad: kernel - 1 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(&x.pad_with_zeros(2, self.left_pad, 0)?)
    }
}

struct CausalConvBlock {
    conv1: CausalConv1d,
    norm: LayerNorm,
    conv2: CausalConv1d,
}

impl CausalConvBlock {
    fn new(channels: usize, kernel: usize, vb: VarBuilder) -> Result<Self> {
        let block = vb.pp("block");
        Ok(Self {
            conv1: CausalConv1d::new(channels, channels, kernel, block.pp(1))?,
            norm: layer_norm(channels, 1e-5, block.pp(3))?,
            conv2: CausalConv1d::new(channels, channels, kernel, block.pp(6))?,
        })
    }

    /// `x`: `[b, t, c]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xt = x.transpose(1, 2)?.contiguous()?;
        let xt = self.conv1.forward(&xt)?;
        let xt = xt.transpose(1, 2)?.contiguous()?;
        let xt = self.norm.forward(&xt)?;
        let xt = mish(&xt)?;
        let xt = xt.transpose(1, 2)?.contiguous()?;
        let xt = self.conv2.forward(&xt)?;
        xt.transpose(1, 2)?.contiguous()
    }
}

/// `Mish(x) = x * tanh(softplus(x))`.
fn mish(x: &Tensor) -> Result<Tensor> {
    let softplus = (x.exp()? + 1.0)?.log()?;
    x * softplus.tanh()?
}

// ── DiT block (9-way AdaLN-Zero: msa + mlp + conv, each shift/scale/gate) ──

struct DiTBlock {
    attn: DitAttention,
    mlp: Mlp,
    conv: CausalConvBlock,
    ada_ln: Linear,
    hidden_size: usize,
}

impl DiTBlock {
    fn new(hidden_size: usize, num_heads: usize, head_dim: usize, mlp_ratio: f64, vb: VarBuilder) -> Result<Self> {
        let mlp_hidden = (hidden_size as f64 * mlp_ratio) as usize;
        Ok(Self {
            attn: DitAttention::new(hidden_size, num_heads, head_dim, vb.pp("attn"))?,
            mlp: Mlp::new(hidden_size, mlp_hidden, vb.pp("mlp"))?,
            conv: CausalConvBlock::new(hidden_size, 3, vb.pp("conv"))?,
            ada_ln: linear(hidden_size, 9 * hidden_size, vb.pp("adaLN_modulation").pp(1))?,
            hidden_size,
        })
    }

    /// `x`: `[b, t, c]`. `c_emb`: `[b, 1, c]` (timestep embedding).
    fn forward(&self, x: &Tensor, c_emb: &Tensor) -> Result<Tensor> {
        let c_emb = c_emb.silu()?;
        let modulation = self.ada_ln.forward(&c_emb)?; // [b, 1, 9*c]
        let chunks: Vec<Tensor> = (0..9).map(|i| modulation.narrow(D::Minus1, i * self.hidden_size, self.hidden_size)).collect::<Result<_>>()?;
        let (shift_msa, scale_msa, gate_msa) = (&chunks[0], &chunks[1], &chunks[2]);
        let (shift_mlp, scale_mlp, gate_mlp) = (&chunks[3], &chunks[4], &chunks[5]);
        let (shift_conv, scale_conv, gate_conv) = (&chunks[6], &chunks[7], &chunks[8]);

        let modulate = |normed: &Tensor, shift: &Tensor, scale: &Tensor| -> Result<Tensor> {
            normed.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(shift)
        };

        let attn_in = modulate(&norm_no_affine(x, 1e-6)?, shift_msa, scale_msa)?;
        let x = (x + self.attn.forward(&attn_in)?.broadcast_mul(gate_msa)?)?;

        let conv_in = modulate(&norm_no_affine(&x, 1e-6)?, shift_conv, scale_conv)?;
        let x = (&x + self.conv.forward(&conv_in)?.broadcast_mul(gate_conv)?)?;

        let mlp_in = modulate(&norm_no_affine(&x, 1e-6)?, shift_mlp, scale_mlp)?;
        &x + self.mlp.forward(&mlp_in)?.broadcast_mul(gate_mlp)?
    }
}

struct FinalLayer {
    linear: Linear,
    ada_ln: Linear,
    hidden_size: usize,
}

impl FinalLayer {
    fn new(hidden_size: usize, out_channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear: linear(hidden_size, out_channels, vb.pp("linear"))?,
            ada_ln: linear(hidden_size, 2 * hidden_size, vb.pp("adaLN_modulation").pp(1))?,
            hidden_size,
        })
    }

    fn forward(&self, x: &Tensor, c_emb: &Tensor) -> Result<Tensor> {
        let c_emb = c_emb.silu()?;
        let modulation = self.ada_ln.forward(&c_emb)?;
        let shift = modulation.narrow(D::Minus1, 0, self.hidden_size)?;
        let scale = modulation.narrow(D::Minus1, self.hidden_size, self.hidden_size)?;
        let x = norm_no_affine(x, 1e-6)?.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
        self.linear.forward(&x)
    }
}

// ── Full DiT ──────────────────────────────────────────────────────────────

pub struct DiT {
    t_embedder: TimestepEmbedder,
    in_proj: Linear,
    blocks: Vec<DiTBlock>,
    final_layer: FinalLayer,
}

impl DiT {
    pub fn new(in_channels: usize, out_channels: usize, hidden_size: usize, depth: usize, num_heads: usize, head_dim: usize, mlp_ratio: f64, vb: VarBuilder) -> Result<Self> {
        let t_embedder = TimestepEmbedder::new(hidden_size, 256, vb.pp("t_embedder"))?;
        let in_proj = linear(in_channels, hidden_size, vb.pp("in_proj"))?;
        let vb_blocks = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            blocks.push(DiTBlock::new(hidden_size, num_heads, head_dim, mlp_ratio, vb_blocks.pp(i))?);
        }
        let final_layer = FinalLayer::new(hidden_size, out_channels, vb.pp("final_layer"))?;
        Ok(Self { t_embedder, in_proj, blocks, final_layer })
    }

    /// `x`, `mu`, `cond`: `[b, 80, t]`. `spks`: `[b, 80]`. `t_step`: `[b]`.
    /// Returns `[b, 80, t]`.
    pub fn forward(&self, x: &Tensor, mu: &Tensor, t_step: &Tensor, spks: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let t_emb = self.t_embedder.forward(t_step)?.unsqueeze(1)?; // [b, 1, hidden]

        let (b, _c, t) = x.dims3()?;
        let spks_bcast = spks.unsqueeze(2)?.broadcast_as((b, spks.dim(1)?, t))?;
        let packed = Tensor::cat(&[x, mu, &spks_bcast, cond], 1)?; // [b, 320, t]

        let mut xs = packed.transpose(1, 2)?.contiguous()?; // [b, t, 320]
        xs = self.in_proj.forward(&xs)?;
        for block in &self.blocks {
            xs = block.forward(&xs, &t_emb)?;
        }
        let out = self.final_layer.forward(&xs, &t_emb)?; // [b, t, 80]
        out.transpose(1, 2)?.contiguous()
    }
}
