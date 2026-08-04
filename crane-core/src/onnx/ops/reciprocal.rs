// SPDX-License-Identifier: MIT
//! Crane Added 20260804: ONNX `Reciprocal` as a native eval op, needed by a
//! fine-tuned Kokoro backbone's `ISTFTNet` vocoder — its noise-residual
//! block computes `1 / alpha` (Snake activation) via `x.__rdiv__`, which the
//! ONNX exporter lowers to `Reciprocal`.

use candle_core::{Result, Tensor};

/// ONNX `Reciprocal`: <https://onnx.ai/onnx/operators/onnx__Reciprocal.html>.
pub(crate) fn reciprocal(xs: &Tensor) -> Result<Tensor> {
    xs.recip()
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    use super::reciprocal;

    #[test]
    fn reciprocal_inverts_positive_and_negative_values() -> Result<()> {
        // Reciprocal must invert positive and negative inputs, and match
        // real division's IEEE-754 behavior at the x = 0 pole (+inf).
        let x = Tensor::new(&[2.0f32, -4.0, 0.5, 0.0], &Device::Cpu)?;

        let y = reciprocal(&x)?;

        let got = y.to_vec1::<f32>()?;
        for (g, e) in got[..3].iter().zip([0.5f32, -0.25, 2.0].iter()) {
            assert!((g - e).abs() < 1e-6, "{g} vs {e}");
        }
        assert_eq!(got[3], f32::INFINITY);
        Ok(())
    }
}
