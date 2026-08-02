//! RMSNorm with a SiLU-gated output: `y = (x / rms(x)) * weight * silu(gate)`.
//!
//! This is the output-side normalization used by every Gated Delta Net layer in
//! Qwen 3.5 — the `z` projection (silu-gate) modulates the normalized
//! recurrence output before it goes through `out_proj`.
//!
//! Normalization goes through candle's fused `rms_norm`, which accumulates in
//! f32 internally whatever the tensor dtype — so the f32 normalization the
//! manual chain spelled out is preserved, in one launch instead of five. The
//! fused op does require `x` and `weight` to share a dtype; casting `weight`
//! to `x`'s is a no-op clone here, since GGUF dequantizes to the model dtype.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// `RmsNorm(x) * silu(z)` with a learned per-channel weight (plain `weight`,
/// no unit offset — matches HF's `Qwen3_5RMSNormGated`).
pub struct RmsNormGated {
    weight: Tensor,
    eps: f32,
}

impl RmsNormGated {
    pub fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self::from_weight(weight, eps))
    }

    /// Construct from an already-loaded weight (e.g. dequantized from GGUF).
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_weight(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps: eps as f32 }
    }

    /// Forward pass. `x` and `gate` must share shape `[..., size]`.
    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        // Norm before gate (HF order): normalize, scale by weight, then * silu(gate).
        let weight = self.weight.to_dtype(x.dtype())?;
        let normalized = candle_nn::ops::rms_norm(&x.contiguous()?, &weight, self.eps)?;
        let silu_gate = candle_nn::ops::silu(&gate.to_dtype(x.dtype())?)?;
        normalized.mul(&silu_gate)
    }

    /// Length of the learned per-channel weight vector. Exposed so that
    /// callers (e.g. `GatedDeltaNet`) can recover the per-head value dim
    /// without a separate config field.
    pub fn weight_len(&self) -> usize {
        self.weight.dim(0).unwrap_or(0)
    }
}
