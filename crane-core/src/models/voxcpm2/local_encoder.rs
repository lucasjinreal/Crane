//! Encodes each 4-frame audio "patch" into one embedding vector, via a
//! prepended learned CLS-like token pooled at position 0. Port of
//! `locenc/local_encoder.py`'s `VoxCPMLocEnc`.
//!
//! Used both for the (all-zero, in zero-shot mode) prefix and — inside the
//! generation loop — to re-encode every newly-generated patch before it
//! feeds `base_lm`'s next step.

use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use super::config::MiniCpm4Config;
use super::minicpm4::MiniCpm4Model;

pub struct VoxCpmLocEnc {
    special_token: Tensor,
    in_proj: Linear,
    encoder: MiniCpm4Model,
}

impl VoxCpmLocEnc {
    pub fn new(cfg: &MiniCpm4Config, input_dim: usize, max_length: usize, vb: VarBuilder) -> Result<Self> {
        let special_token = vb.get((1, 1, 1, cfg.hidden_size), "special_token")?;
        let in_proj = linear(input_dim, cfg.hidden_size, vb.pp("in_proj"))?;
        let encoder = MiniCpm4Model::new(cfg, max_length, vb.pp("encoder"))?;
        Ok(Self { special_token, in_proj, encoder })
    }

    /// `x`: `[B, T, P, D]` (`P` = `patch_size`, `D` = `feat_dim`). Returns
    /// `[B, T, hidden_size]`.
    ///
    /// Stateless / one-shot: always clears the inner encoder's KV cache
    /// first, since this network is called fresh every time (never
    /// incrementally) — a stale cache from a previous call would silently
    /// corrupt the non-causal attention here.
    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        self.encoder.clear_kv_cache();

        let (b, t, p, _d) = x.dims4()?;
        let hidden = self.in_proj.forward(x)?; // [B, T, P, H]
        let h = hidden.dim(3)?;
        let special = self.special_token.broadcast_as((b, t, 1, h))?;
        let with_cls = Tensor::cat(&[&special, &hidden], 2)?; // [B, T, P+1, H]
        let flat = with_cls.reshape((b * t, p + 1, h))?;

        let out = self.encoder.forward(&flat, false)?; // [(B*T), P+1, H]
        let cls = out.narrow(1, 0, 1)?.reshape((b * t, h))?;
        cls.reshape((b, t, h))
    }
}
