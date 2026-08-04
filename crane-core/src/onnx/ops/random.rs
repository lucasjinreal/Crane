// SPDX-License-Identifier: MIT
//! Crane Added 20260804: ONNX `RandomUniformLike`/`RandomNormalLike` as
//! native eval ops, plus dtype/seed helpers shared with the pre-existing
//! `RandomUniform`/`RandomNormal` arm in `eval.rs`.

use candle_core::{DType, Result, Tensor, bail};

use crate::onnx::eval::{dtype, get_attr_opt};
use crate::onnx::proto::NodeProto;
use crate::onnx::proto::tensor_proto::DataType;

/// Crane Added 20260806: shared dtype resolver for the Random* family of ops
/// (`RandomUniform`, `RandomNormal`, `RandomUniformLike`,
/// `RandomNormalLike`). `fallback` is used when the 'dtype' attribute is
/// absent, or set to UNDEFINED (0) — the latter is the ONNX sentinel some
/// exporters emit for "same as input", relevant for the *Like variants.
pub(crate) fn parse_random_float_dtype(
    node: &NodeProto,
    op_name: &str,
    fallback: DType,
) -> Result<DType> {
    let Some(&dt) = get_attr_opt::<i64>(node, "dtype")? else {
        return Ok(fallback);
    };
    let Ok(dt_i32) = i32::try_from(dt) else {
        bail!("unsupported 'dtype' value {dt:?} for {op_name} {}", node.name)
    };
    match DataType::try_from(dt_i32) {
        Ok(DataType::Undefined) => Ok(fallback),
        Ok(dt) => match dtype(dt) {
            Some(DType::U8 | DType::U32 | DType::I64) => {
                bail!(
                    "unsupported 'dtype' value {dt:?}, only floats are allowed, for {op_name} {}",
                    node.name
                )
            },
            Some(dt) => Ok(dt),
            None => {
                bail!("unsupported 'dtype' value {dt:?} for {op_name} {}", node.name)
            },
        },
        Err(_) => {
            bail!("unsupported 'dtype' value {dt_i32:?} for {op_name} {}", node.name)
        },
    }
}

/// Crane Added 20260806: shared seed rejection for the Random* family of
/// ops — Crane doesn't support seeded/deterministic RNG in the ONNX
/// evaluator.
pub(crate) fn reject_random_seed(node: &NodeProto, op_name: &str) -> Result<()> {
    let seed: Option<f32> = get_attr_opt(node, "seed")?.copied();
    if seed.is_some() {
        bail!("seed for {op_name} is currently not supported")
    }
    Ok(())
}

/// ONNX `RandomUniformLike`:
/// <https://onnx.ai/onnx/operators/onnx__RandomUniformLike.html>. Needed by
/// a fine-tuned Kokoro backbone's `ISTFTNet` vocoder — its sine-excitation
/// source signal adds uniform noise via `RandomUniformLike` (`_f02sine` in
/// istftnet.py). Output shape and (absent an explicit `dtype`) dtype come
/// from `input`, not a `shape` attribute like `RandomUniform`.
pub(crate) fn random_uniform_like(node: &NodeProto, input: &Tensor) -> Result<Tensor> {
    let dtype = parse_random_float_dtype(node, "RandomUniformLike", input.dtype())?;
    reject_random_seed(node, "RandomUniformLike")?;
    let low: f32 = get_attr_opt(node, "low")?.copied().unwrap_or(0.0);
    let high: f32 = get_attr_opt(node, "high")?.copied().unwrap_or(1.0);
    Tensor::rand(low, high, input.dims(), input.device())?.to_dtype(dtype)
}

/// ONNX `RandomNormalLike`:
/// <https://onnx.ai/onnx/operators/onnx__RandomNormalLike.html>. Needed by a
/// fine-tuned Kokoro backbone's `ISTFTNet` vocoder — its harmonic-plus-noise
/// source module adds Gaussian noise via `RandomNormalLike`
/// (`SourceModuleHnNSF.forward` in istftnet.py). Sources its output shape
/// from `input`, the same as `random_uniform_like`.
pub(crate) fn random_normal_like(node: &NodeProto, input: &Tensor) -> Result<Tensor> {
    let dtype = parse_random_float_dtype(node, "RandomNormalLike", input.dtype())?;
    reject_random_seed(node, "RandomNormalLike")?;
    let mean: f32 = get_attr_opt(node, "mean")?.copied().unwrap_or(0.0);
    let scale: f32 = get_attr_opt(node, "scale")?.copied().unwrap_or(1.0);
    Tensor::randn(mean, scale, input.dims(), input.device())?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::{random_normal_like, random_uniform_like};
    use crate::onnx::proto::attribute_proto::AttributeType;
    use crate::onnx::proto::{AttributeProto, NodeProto};

    fn random_uniform_like_node(attribute: Vec<AttributeProto>) -> NodeProto {
        NodeProto {
            op_type: "RandomUniformLike".to_string(),
            input: vec!["x".to_string()],
            output: vec!["y".to_string()],
            attribute,
            ..Default::default()
        }
    }

    #[test]
    fn random_uniform_like_matches_input_shape_and_bounds() -> Result<()> {
        // Verifies the output shape/dtype come from the input tensor (not a
        // "shape" attribute, unlike RandomUniform) and every generated
        // value falls within [low, high) — the values themselves are
        // random, so bounds and shape are what's actually checkable.
        let node = random_uniform_like_node(vec![
            AttributeProto {
                name: "low".to_string(),
                r#type: AttributeType::Float as i32,
                f: -2.0,
                ..Default::default()
            },
            AttributeProto {
                name: "high".to_string(),
                r#type: AttributeType::Float as i32,
                f: 3.0,
                ..Default::default()
            },
        ]);
        let x = Tensor::new(&[[0f32, 0., 0., 0., 0.], [0., 0., 0., 0., 0.]], &Device::Cpu)?;

        let y = random_uniform_like(&node, &x)?;

        assert_eq!(y.dims(), &[2, 5]);
        assert_eq!(y.dtype(), DType::F32);
        for v in y.flatten_all()?.to_vec1::<f32>()? {
            assert!((-2.0..3.0).contains(&v), "{v} out of [-2, 3)");
        }
        Ok(())
    }

    #[test]
    fn random_uniform_like_inherits_input_dtype_when_unspecified() -> Result<()> {
        // Per the ONNX spec, an absent 'dtype' attribute on RandomUniformLike
        // must fall back to the input tensor's dtype, not always F32.
        let node = random_uniform_like_node(vec![]);
        let x = Tensor::new(&[0f64, 0., 0.], &Device::Cpu)?;

        let y = random_uniform_like(&node, &x)?;

        assert_eq!(y.dtype(), DType::F64);
        Ok(())
    }

    #[test]
    fn random_uniform_like_dtype_zero_means_inherit() -> Result<()> {
        // An explicit 'dtype' of 0 is ONNX's UNDEFINED sentinel, which some
        // exporters emit to mean "same as input" rather than omitting the
        // attribute outright — it must not be treated as an error.
        let node = random_uniform_like_node(vec![AttributeProto {
            name: "dtype".to_string(),
            r#type: AttributeType::Int as i32,
            i: 0,
            ..Default::default()
        }]);
        let x = Tensor::new(&[0f64, 0., 0.], &Device::Cpu)?;

        let y = random_uniform_like(&node, &x)?;

        assert_eq!(y.dtype(), DType::F64);
        Ok(())
    }

    #[test]
    fn random_normal_like_matches_input_shape_and_distribution() -> Result<()> {
        // Same shape-from-input contract as RandomUniformLike, but for a
        // Gaussian: checks the sample mean lands near the requested "mean"
        // (law of large numbers over enough elements) rather than an exact
        // per-element bound, since a Gaussian is unbounded.
        let node = NodeProto {
            op_type: "RandomNormalLike".to_string(),
            input: vec!["x".to_string()],
            output: vec!["y".to_string()],
            attribute: vec![
                AttributeProto {
                    name: "mean".to_string(),
                    r#type: AttributeType::Float as i32,
                    f: 5.0,
                    ..Default::default()
                },
                AttributeProto {
                    name: "scale".to_string(),
                    r#type: AttributeType::Float as i32,
                    f: 0.5,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let x = Tensor::zeros((4096,), DType::F32, &Device::Cpu)?;

        let y = random_normal_like(&node, &x)?;

        assert_eq!(y.dims(), &[4096]);
        assert_eq!(y.dtype(), DType::F32);
        let values = y.to_vec1::<f32>()?;
        let sample_mean = values.iter().sum::<f32>() / values.len() as f32;
        assert!((sample_mean - 5.0).abs() < 0.2, "sample mean {sample_mean} far from 5.0");
        Ok(())
    }
}
