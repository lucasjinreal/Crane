// SPDX-License-Identifier: MIT
//! Crane Added 20260827: ONNX `Softplus` as a native eval op, needed by
//! Audio8-TTS's Mamba-2 hybrid backbone (applied to the `dt` time-step
//! pre-activation in every layer).

use candle_core::{Result, Tensor};

/// ONNX `Softplus`: <https://onnx.ai/onnx/operators/onnx__Softplus.html>.
///
/// Computes `log(1 + exp(x))`, elementwise, via the identity
/// `softplus(x) = relu(x) + log(1 + exp(-|x|))`. `exp(-|x|)` is always in
/// `(0, 1]`, so this never overflows regardless of dtype or magnitude of
/// `x` — unlike the direct formula, which overflows `exp(x)` to `inf` for
/// large `x` (the overflow threshold depends on dtype, e.g. `f16` overflows
/// far earlier than `f32`).
pub(crate) fn softplus(xs: &Tensor) -> Result<Tensor> {
    let neg_abs_exp = xs.abs()?.neg()?.exp()?;
    xs.relu()? + (neg_abs_exp + 1.0)?.log()?
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::softplus;

    #[test]
    fn softplus_matches_reference_and_stays_stable_at_extremes() -> Result<()> {
        // Normal-range values must match the closed-form log(1 + exp(x)),
        // and large-magnitude inputs must not overflow to inf/NaN.
        let x = Tensor::new(&[0.0f32, 1.0, -1.0, 100.0, -100.0], &Device::Cpu)?;

        let y = softplus(&x)?;

        let got = y.to_vec1::<f32>()?;
        let expected = [
            (2.0f32).ln(),
            (1.0 + std::f32::consts::E).ln(),
            (1.0 + (-1.0f32).exp()).ln(),
            100.0,
            0.0,
        ];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-4, "{g} vs {e}");
        }
        assert!(got.iter().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn softplus_stays_finite_on_f16_where_exp_overflows_early() -> Result<()> {
        // f16's exp() overflows to inf starting around x ~= 11.09, well
        // below f32's ~88 — a threshold tuned for f32 would misfire here.
        let x = Tensor::new(&[15.0f32, 30.0, -15.0], &Device::Cpu)?.to_dtype(DType::F16)?;

        let y = softplus(&x)?;

        let got = y.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
        // For large positive x, softplus(x) ~= x.
        assert!((got[0] - 15.0).abs() < 0.1, "{got:?}");
        assert!((got[1] - 30.0).abs() < 0.1, "{got:?}");
        Ok(())
    }
}
