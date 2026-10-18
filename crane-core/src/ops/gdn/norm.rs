//! `RMSNorm` with a gated output: `y = (x / rms(x)) * weight * act(gate)`.
//!
//! This is the output-side normalization used by every Gated Delta Net layer —
//! the `z` projection gate modulates the normalized recurrence output before it
//! goes through `out_proj`. Qwen 3.5 gates with silu, Qwen4-Exp with sigmoid
//! (see [`GateActivation`]).
//!
//! Normalization goes through candle's fused `rms_norm`, which accumulates in
//! f32 internally whatever the tensor dtype — so the f32 normalization the
//! manual chain spelled out is preserved, in one launch instead of five. The
//! fused op does require `x` and `weight` to share a dtype; casting `weight`
//! to `x`'s is a no-op clone here, since GGUF dequantizes to the model dtype.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// Activation applied to the output gate of [`RmsNormGated`] (HF
/// `output_gate_type`, falling back to `hidden_act`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateActivation {
    /// `silu` / `swish`: Qwen 3.5 and its 3.6 / 3.8 dense successors.
    Silu,
    /// `sigmoid`: Qwen4-Exp.
    Sigmoid,
}

impl GateActivation {
    /// Parse the HF config spelling.
    ///
    /// # Errors
    ///
    /// Returns an error for any activation other than silu/swish or sigmoid.
    pub fn from_name(name: &str) -> Result<Self> {
        match name {
            "silu" | "swish" => Ok(Self::Silu),
            "sigmoid" => Ok(Self::Sigmoid),
            other => candle_core::bail!(
                "unsupported GDN output gate activation {other:?} (expected silu, swish or sigmoid)"
            ),
        }
    }

    fn apply(self, gate: &Tensor) -> Result<Tensor> {
        match self {
            Self::Silu => candle_nn::ops::silu(gate),
            Self::Sigmoid => candle_nn::ops::sigmoid(gate),
        }
    }
}

/// `RmsNorm(x) * act(z)` with a learned per-channel weight (plain `weight`,
/// no unit offset — matches HF's `Qwen3_5RMSNormGated`).
pub struct RmsNormGated {
    weight: Tensor,
    eps: f32,
    activation: GateActivation,
}

impl RmsNormGated {
    /// # Errors
    ///
    /// Returns an error if the `weight` tensor of length `size` is missing
    /// from `vb`.
    pub fn new(size: usize, eps: f64, activation: GateActivation, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self::from_weight(weight, eps, activation))
    }

    /// Construct from an already-loaded weight (e.g. dequantized from GGUF).
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_weight(weight: Tensor, eps: f64, activation: GateActivation) -> Self {
        Self {
            weight,
            eps: eps as f32,
            activation,
        }
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub(super) fn activation(&self) -> GateActivation {
        self.activation
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub(super) fn eps(&self) -> f32 {
        self.eps
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    pub(super) fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Forward pass. `x` and `gate` must share shape `[..., size]`.
    ///
    /// # Errors
    ///
    /// Returns an error if `x` and `gate` have incompatible shapes or
    /// dtypes for the underlying `rms_norm`/activation/multiply ops.
    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        if matches!(
            x.dtype(),
            candle_core::DType::BF16 | candle_core::DType::F16
        ) {
            let dtype = x.dtype();
            let xf = x.to_dtype(candle_core::DType::F32)?.contiguous()?;
            // HF: F32 normalization -> input dtype -> weight multiply in input
            // dtype -> F32 activation/multiply -> final input dtype.
            let normalized = crate::ops::float_math::rms_norm(&xf, None, self.eps)?
                .to_dtype(dtype)?
                .broadcast_mul(&self.weight.to_dtype(dtype)?)?;
            let gate = self
                .activation
                .apply(&gate.to_dtype(candle_core::DType::F32)?)?;
            return normalized
                .to_dtype(candle_core::DType::F32)?
                .mul(&gate)?
                .to_dtype(dtype);
        }
        // Norm before gate (HF order): normalize, scale by weight, then * act(gate).
        let weight = self.weight.to_dtype(x.dtype())?;
        let normalized = candle_nn::ops::rms_norm(&x.contiguous()?, &weight, self.eps)?;
        let gate = self.activation.apply(&gate.to_dtype(x.dtype())?)?;
        normalized.mul(&gate)
    }

    /// Length of the learned per-channel weight vector. Exposed so that
    /// callers (e.g. `GatedDeltaNet`) can recover the per-head value dim
    /// without a separate config field.
    pub fn weight_len(&self) -> usize {
        self.weight.dim(0).unwrap_or(0)
    }
}
