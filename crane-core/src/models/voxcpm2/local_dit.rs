//! The flow-matching velocity estimator: a DiT whose conditioning (LM
//! context `mu`, timestep `t`, previous-patch `cond`) is concatenated as
//! extra sequence tokens rather than injected via AdaLN. Port of
//! `locdit/local_dit_v2.py`'s `VoxCPMLocDiT` (imported upstream as
//! `VoxCPMLocDiTV2` — `locdit/__init__.py` aliases it on import; the class
//! itself is named identically to the older, structurally different V1 in
//! `local_dit.py`, which this crate does not implement).

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, Activation, Linear, VarBuilder};

use super::config::MiniCpm4Config;
use super::minicpm4::MiniCpm4Model;

/// Functional (no learnable params) sinusoidal timestep embedding. Port of
/// `SinusoidalPosEmb` — output width equals `dim` (the DiT's `hidden_size`).
fn sinusoidal_pos_emb(t: &Tensor, dim: usize, scale: f64) -> Result<Tensor> {
    let device = t.device();
    let half_dim = dim / 2;
    let emb_scale = (10000f64).ln() / (half_dim as f64 - 1.0);
    let freqs: Vec<f32> = (0..half_dim).map(|i| (-(i as f64) * emb_scale).exp() as f32).collect();
    let freqs = Tensor::from_vec(freqs, half_dim, device)?; // [half_dim]

    let t = t.to_dtype(candle_core::DType::F32)?.unsqueeze(1)?; // [N, 1]
    let angles = (t.broadcast_mul(&freqs.unsqueeze(0)?)? * scale)?; // [N, half_dim]
    Tensor::cat(&[angles.sin()?, angles.cos()?], 1) // [N, dim]
}

struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    fn new(in_channels: usize, time_embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: linear(in_channels, time_embed_dim, vb.pp("linear_1"))?,
            linear_2: linear(time_embed_dim, time_embed_dim, vb.pp("linear_2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(x)?;
        let x = Activation::Silu.forward(&x)?;
        self.linear_2.forward(&x)
    }
}

pub struct VoxCpmLocDit {
    in_proj: Linear,
    cond_proj: Linear,
    out_proj: Linear,
    time_mlp: TimestepEmbedding,
    delta_time_mlp: TimestepEmbedding,
    decoder: MiniCpm4Model,
    hidden_size: usize,
    in_channels: usize,
}

impl VoxCpmLocDit {
    pub fn new(cfg: &MiniCpm4Config, in_channels: usize, max_length: usize, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        Ok(Self {
            in_proj: linear(in_channels, hidden_size, vb.pp("in_proj"))?,
            cond_proj: linear(in_channels, hidden_size, vb.pp("cond_proj"))?,
            out_proj: linear(hidden_size, in_channels, vb.pp("out_proj"))?,
            time_mlp: TimestepEmbedding::new(hidden_size, hidden_size, vb.pp("time_mlp"))?,
            delta_time_mlp: TimestepEmbedding::new(hidden_size, hidden_size, vb.pp("delta_time_mlp"))?,
            decoder: MiniCpm4Model::new(cfg, max_length, vb.pp("decoder"))?,
            hidden_size,
            in_channels,
        })
    }

    /// `x`: `[N, C, T]` (noisy patch). `mu`: `[N, 2*hidden_size]` (LM
    /// context, reshaped here into 2 tokens). `t`/`dt`: `[N]`. `cond`:
    /// `[N, C, T']` (previous patch). Returns `[N, C, T]`.
    ///
    /// Stateless / one-shot, same reasoning as [`super::local_encoder::VoxCpmLocEnc`]
    /// — always clears the inner decoder's KV cache first.
    pub fn forward(&mut self, x: &Tensor, mu: &Tensor, t: &Tensor, cond: &Tensor, dt: &Tensor) -> Result<Tensor> {
        self.decoder.clear_kv_cache();

        let n = x.dim(0)?;
        let x_h = self.in_proj.forward(&x.transpose(1, 2)?.contiguous()?)?; // [N, T, H]
        let cond_h = self.cond_proj.forward(&cond.transpose(1, 2)?.contiguous()?)?; // [N, T', H]
        let prefix = cond_h.dim(1)?;

        let t_emb = sinusoidal_pos_emb(t, self.hidden_size, 1000.0)?.to_dtype(x_h.dtype())?;
        let t_emb = self.time_mlp.forward(&t_emb)?;
        let dt_emb = sinusoidal_pos_emb(dt, self.hidden_size, 1000.0)?.to_dtype(x_h.dtype())?;
        let dt_emb = self.delta_time_mlp.forward(&dt_emb)?;
        let t_tok = (t_emb + dt_emb)?.unsqueeze(1)?; // [N, 1, H]

        let mu_toks = mu.reshape((n, mu.dim(1)? / self.hidden_size, self.hidden_size))?; // [N, 2, H]
        let mu_len = mu_toks.dim(1)?;
        let seq = Tensor::cat(&[&mu_toks, &t_tok, &cond_h, &x_h], 1)?;

        let hidden = self.decoder.forward(&seq, false)?;
        let x_len = x_h.dim(1)?;
        let total = hidden.dim(1)?;
        let hidden = hidden.narrow(1, total - x_len, x_len)?;
        debug_assert_eq!(prefix + mu_len + 1 + x_len, total);

        self.out_proj.forward(&hidden)?.transpose(1, 2)?.contiguous()
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }
}
