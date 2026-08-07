// SPDX-License-Identifier: MIT
//! Crane Added 20260804: ONNX `IsNaN` as a native eval op, needed by a
//! fine-tuned Kokoro backbone's PLBERT attention (SDPA's NaN check).

use candle_core::{Result, Tensor};

/// ONNX `IsNaN`: <https://onnx.ai/onnx/operators/onnx__IsNaN.html>.
///
/// Candle has no dedicated NaN test, so this uses the standard `x != x`
/// trick: NaN is the only value that never compares equal to itself under
/// IEEE 754.
pub(crate) fn is_nan(xs: &Tensor) -> Result<Tensor> {
    xs.ne(xs)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result};

    use super::is_nan;

    #[test]
    fn is_nan_flags_only_nan_elements() -> Result<()> {
        // Verifies IsNaN uses the `x != x` trick correctly: NaN never
        // compares equal to itself, but every other value (including
        // infinities and zero) does.
        let x = candle_core::Tensor::new(
            &[f32::NAN, 0.0, 1.5, f32::INFINITY, f32::NEG_INFINITY],
            &Device::Cpu,
        )?;

        let y = is_nan(&x)?;

        assert_eq!(y.to_vec1::<u8>()?, vec![1, 0, 0, 0, 0]);
        Ok(())
    }
}
