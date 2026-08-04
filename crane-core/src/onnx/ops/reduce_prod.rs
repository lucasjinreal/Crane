// SPDX-License-Identifier: MIT
//! Crane Added 20260804: ONNX `ReduceProd` as a native eval op, needed by
//! `pack_padded_sequence` in a fine-tuned Kokoro backbone's text encoder
//! (`invert_permutation` computes a flat size via `ReduceProd` over a Shape
//! tensor).

use candle_core::{Result, Tensor, bail};

use crate::onnx::eval::get_attr_opt;
use crate::onnx::proto::NodeProto;

/// ONNX `ReduceProd`: <https://onnx.ai/onnx/operators/onnx__ReduceProd.html>.
///
/// Candle has no native product reduction, so this multiplies 1-wide slices
/// along each axis one at a time via `broadcast_mul` — fine since this only
/// ever runs on small shape-derived tensors, never a hot path. `axes` may
/// come from the opset 18+ second input tensor or fall back to the legacy
/// (opset <= 13) `axes` attribute.
pub(crate) fn reduce_prod(node: &NodeProto, input: &Tensor, axes: Option<&Tensor>) -> Result<Tensor> {
    // TODO: Handle empty set
    // Definition: the identity for product is 1, so "reduction over an
    // empty set of values yields 1". For now, bail for consistency with
    // ReduceMax / ReduceMin.
    if input.elem_count() == 0 {
        bail!("reduction over zero-size tensor not supported");
    }
    let keepdims = get_attr_opt::<i64>(node, "keepdims")?.copied().unwrap_or(1);
    let noop_with_empty_axes =
        get_attr_opt::<i64>(node, "noop_with_empty_axes")?.copied().unwrap_or(0);

    let raw_axes: Option<Vec<i64>> = match axes {
        Some(axes) => Some(axes.to_vec1::<i64>()?),
        None => {
            // Backward compatibility with version 13 and below.
            get_attr_opt::<[i64]>(node, "axes")?.map(<[i64]>::to_vec)
        },
    };

    // An explicitly empty axes tensor/attribute is treated the same as no
    // axes given at all, so `noop_with_empty_axes` still applies instead of
    // silently becoming a no-op.
    let mut axes = match raw_axes {
        Some(ref axes) if !axes.is_empty() => {
            axes.iter().map(|&a| input.normalize_axis(a)).collect::<Result<Vec<_>>>()?
        },
        _ if noop_with_empty_axes == 1 => vec![],
        _ => (0..input.rank()).collect(),
    };
    // Reduce from the highest axis down so removing a (non-keepdim) axis
    // doesn't shift the index of an axis still waiting to be processed.
    axes.sort_unstable();
    if axes.windows(2).any(|w| w[0] == w[1]) {
        bail!("Duplicate value in 'axes'");
    }

    let mut output = input.clone();
    for &axis in axes.iter().rev() {
        let n = output.dims()[axis];
        let mut acc = output.narrow(axis, 0, 1)?;
        for i in 1..n {
            acc = acc.broadcast_mul(&output.narrow(axis, i, 1)?)?;
        }
        output = if keepdims == 1 { acc } else { acc.squeeze(axis)? };
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::reduce_prod;
    use crate::onnx::proto::attribute_proto::AttributeType;
    use crate::onnx::proto::{AttributeProto, NodeProto};

    fn reduce_prod_node(attribute: Vec<AttributeProto>) -> NodeProto {
        NodeProto {
            op_type: "ReduceProd".to_string(),
            input: vec!["x".to_string()],
            output: vec!["y".to_string()],
            attribute,
            ..Default::default()
        }
    }

    #[test]
    fn reduce_prod_defaults_to_all_axes() -> Result<()> {
        // Verifies the no-axes-given case (matching the real ONNX export
        // this op was added for): reduces over every axis, keepdims=0
        // squeezes down to a scalar.
        let node = reduce_prod_node(vec![AttributeProto {
            name: "keepdims".to_string(),
            r#type: AttributeType::Int as i32,
            i: 0,
            ..Default::default()
        }]);
        let x = Tensor::new(&[2f32, 3., 4.], &Device::Cpu)?;

        let y = reduce_prod(&node, &x, None)?;

        assert_eq!(y.to_scalar::<f32>()?, 24.);
        Ok(())
    }

    #[test]
    fn reduce_prod_keepdims_along_attribute_axis() -> Result<()> {
        // Verifies the legacy (opset <= 13) `axes`-as-attribute path and
        // keepdims=1: row-wise product along axis 1 of a 2x2 input.
        let node = reduce_prod_node(vec![
            AttributeProto {
                name: "axes".to_string(),
                r#type: AttributeType::Ints as i32,
                ints: vec![1],
                ..Default::default()
            },
            AttributeProto {
                name: "keepdims".to_string(),
                r#type: AttributeType::Int as i32,
                i: 1,
                ..Default::default()
            },
        ]);
        let x = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?;

        let y = reduce_prod(&node, &x, None)?;

        assert_eq!(y.dims(), &[2, 1]);
        assert_eq!(y.to_vec2::<f32>()?, vec![vec![2.], vec![12.]]);
        Ok(())
    }

    #[test]
    fn reduce_prod_multi_axis_no_keepdims() -> Result<()> {
        // Verifies reducing over two non-adjacent axes (0 and 2) at once
        // with keepdims=0: this is the case the highest-axis-first
        // processing order exists for, since squeezing axis 0 first would
        // shift axis 2 down to axis 1 before it's processed.
        let node = reduce_prod_node(vec![
            AttributeProto {
                name: "axes".to_string(),
                r#type: AttributeType::Ints as i32,
                ints: vec![0, 2],
                ..Default::default()
            },
            AttributeProto {
                name: "keepdims".to_string(),
                r#type: AttributeType::Int as i32,
                i: 0,
                ..Default::default()
            },
        ]);
        let x = Tensor::new(
            vec![
                vec![vec![1f32, 2.], vec![3., 4.], vec![5., 6.]],
                vec![vec![7., 8.], vec![9., 10.], vec![11., 12.]],
            ],
            &Device::Cpu,
        )?;

        let y = reduce_prod(&node, &x, None)?;

        assert_eq!(y.dims(), &[3]);
        assert_eq!(y.to_vec1::<f32>()?, vec![112., 1080., 3960.]);
        Ok(())
    }

    #[test]
    fn reduce_prod_negative_axis() -> Result<()> {
        // Verifies that a negative axis attribute is normalized correctly:
        // axis -1 on a rank-2 input is equivalent to axis 1.
        let node = reduce_prod_node(vec![
            AttributeProto {
                name: "axes".to_string(),
                r#type: AttributeType::Ints as i32,
                ints: vec![-1],
                ..Default::default()
            },
            AttributeProto {
                name: "keepdims".to_string(),
                r#type: AttributeType::Int as i32,
                i: 0,
                ..Default::default()
            },
        ]);
        let x = Tensor::new(&[[2f32, 3., 5.], [7., 11., 13.]], &Device::Cpu)?;

        let y = reduce_prod(&node, &x, None)?;

        assert_eq!(y.dims(), &[2]);
        assert_eq!(y.to_vec1::<f32>()?, vec![30., 1001.]);
        Ok(())
    }

    #[test]
    fn reduce_prod_axes_as_tensor_input() -> Result<()> {
        // Verifies the opset 18+ path where `axes` is passed as a second
        // graph input tensor rather than an attribute.
        let node = reduce_prod_node(vec![AttributeProto {
            name: "keepdims".to_string(),
            r#type: AttributeType::Int as i32,
            i: 0,
            ..Default::default()
        }]);
        let x = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?;
        let axes = Tensor::new(&[0i64], &Device::Cpu)?;

        let y = reduce_prod(&node, &x, Some(&axes))?;

        assert_eq!(y.dims(), &[2]);
        assert_eq!(y.to_vec1::<f32>()?, vec![3., 8.]);
        Ok(())
    }

    #[test]
    fn reduce_prod_duplicate_axes_errors() -> Result<()> {
        // Verifies that a duplicate axis (0 and -2, which both resolve to
        // 0 on a rank-2 input) is rejected instead of panicking.
        let node = reduce_prod_node(vec![AttributeProto {
            name: "axes".to_string(),
            r#type: AttributeType::Ints as i32,
            ints: vec![0, -2],
            ..Default::default()
        }]);
        let x = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?;

        assert!(reduce_prod(&node, &x, None).is_err());
        Ok(())
    }

    #[test]
    fn reduce_prod_zero_size_tensor_errors() -> Result<()> {
        // Verifies that reducing a zero-element tensor returns an error
        // rather than panicking inside the narrow()/broadcast_mul() loop.
        let node = reduce_prod_node(vec![]);
        let x = Tensor::zeros((0,), DType::F32, &Device::Cpu)?;

        assert!(reduce_prod(&node, &x, None).is_err());
        Ok(())
    }
}
