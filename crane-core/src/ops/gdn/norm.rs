//! RMSNorm with a SiLU-gated output: `y = (x / rms(x)) * weight * silu(gate)`.
//!
//! This is the output-side normalization used by every Gated Delta Net layer in
//! Qwen 3.5 — the `z` projection (silu-gate) modulates the normalized
//! recurrence output before it goes through `out_proj`.
//!
//! The normalization is computed manually in f32 (rather than via candle's
//! fused `RmsNorm` op) so it is robust across devices and dtypes: the fused
//! CUDA op requires `x` and `weight` to share a dtype, but the gated norm runs
//! in f32 while the model weights may be f16/bf16 on GPU.

use candle_core::{DType, Result, Tensor, D};
use candle_nn::VarBuilder;

/// `RmsNorm(x) * silu(z)` with a learned per-channel weight (plain `weight`,
/// no unit offset — matches HF's `Qwen3_5RMSNormGated`).
pub struct RmsNormGated {
    weight: Tensor,
    eps: f64,
}

impl RmsNormGated {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    /// Construct from an already-loaded weight (e.g. dequantized from GGUF).
    pub fn from_weight(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Forward pass. `x` and `gate` must share shape `[..., size]`.
    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let gate = gate.to_dtype(DType::F32)?;
        let weight = self.weight.to_dtype(DType::F32)?;
        // Norm before gate (HF order): normalize, scale by weight, then * silu(gate).
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let normalized = x_normed.broadcast_mul(&weight)?;
        let silu_gate = candle_nn::ops::silu(&gate)?;
        normalized.broadcast_mul(&silu_gate)?.to_dtype(dtype)
    }

    /// Length of the learned per-channel weight vector. Exposed so that
    /// callers (e.g. `GatedDeltaNet`) can recover the per-head value dim
    /// without a separate config field.
    pub fn weight_len(&self) -> usize {
        self.weight.dim(0).unwrap_or(0)
    }
}
