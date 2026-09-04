// SPDX-License-Identifier: MIT
//! Crane Added 20260827: ONNX `Einsum` as a native eval op, restricted to
//! the outer-product equation `"i,j->ij"` — the only variant Audio8-TTS's
//! codec encoder/decoder graphs use (sinusoidal positional embeddings).

use candle_core::{Result, Tensor, bail};

/// ONNX `Einsum`: <https://onnx.ai/onnx/operators/onnx__Einsum.html>.
pub(crate) fn einsum(equation: &str, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Byte-exact match: the only equation Audio8-TTS's exported graphs use.
    // A semantically-equivalent but differently-formatted string (e.g. extra
    // spaces) would bail here rather than being parsed as equivalent.
    if equation != "i,j->ij" {
        bail!(
            "unsupported Einsum equation \"{equation}\", only \"i,j->ij\" (outer product) is implemented"
        );
    }
    if a.rank() != 1 || b.rank() != 1 {
        bail!(
            "Einsum \"i,j->ij\" requires two 1-D operands, got shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        );
    }
    a.unsqueeze(1)?.broadcast_mul(&b.unsqueeze(0)?)
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    use super::einsum;

    #[test]
    fn einsum_outer_product() -> Result<()> {
        // out[i, j] = a[i] * b[j] for the "i,j->ij" equation.
        let a = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
        let b = Tensor::new(&[10.0f32, 20.0], &Device::Cpu)?;

        let out = einsum("i,j->ij", &a, &b)?;

        assert_eq!(out.dims(), &[3, 2]);
        let got = out.to_vec2::<f32>()?;
        let expected = [[10.0, 20.0], [20.0, 40.0], [30.0, 60.0]];
        for (row_got, row_expected) in got.iter().zip(expected.iter()) {
            for (g, e) in row_got.iter().zip(row_expected.iter()) {
                assert!((g - e).abs() < 1e-6, "{g} vs {e}");
            }
        }
        Ok(())
    }

    #[test]
    fn einsum_unsupported_equation_bails() {
        let a = Tensor::new(&[1.0f32, 2.0], &Device::Cpu).unwrap();
        let b = Tensor::new(&[1.0f32, 2.0], &Device::Cpu).unwrap();

        let err = einsum("ij,jk->ik", &a, &b).unwrap_err();

        assert!(err.to_string().contains("unsupported Einsum equation"));
    }
}
