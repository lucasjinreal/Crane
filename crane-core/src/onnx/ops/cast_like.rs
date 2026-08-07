// SPDX-License-Identifier: MIT
//! Crane Added 20260804: ONNX `CastLike` as a native eval op, needed by
//! `optimizer/compat.rs`'s `fix_int_cumsum` to cast a `CumSum` result back
//! to its original dtype without statically knowing that dtype (a
//! real-world export can leave `graph.value_info` completely empty, e.g.
//! `torch.onnx.export(dynamo=False)`, so a rewrite that guesses the dtype
//! from `value_info` silently leaves the wrong dtype behind when the guess
//! fails). `CastLike` reads the target dtype from `target_type`'s actual
//! runtime value instead, so there's nothing to guess.

use candle_core::{Result, Tensor};

/// ONNX `CastLike`: <https://onnx.ai/onnx/operators/onnx__CastLike.html>.
pub(crate) fn cast_like(input: &Tensor, target_type: &Tensor) -> Result<Tensor> {
    input.to_dtype(target_type.dtype())
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::cast_like;

    #[test]
    fn cast_like_matches_target_runtime_dtype() -> Result<()> {
        // CastLike reads the target dtype from `target_type`'s actual
        // value at eval time, not a static attribute — so an F64 input
        // cast against an F32 target must come back as F32.
        let x = Tensor::new(&[1.5f64, 2.5, 3.5], &Device::Cpu)?;
        let target = Tensor::new(&[0f32], &Device::Cpu)?;

        let y = cast_like(&x, &target)?;

        assert_eq!(y.dtype(), DType::F32);
        assert_eq!(y.to_vec1::<f32>()?, vec![1.5, 2.5, 3.5]);
        Ok(())
    }
}
