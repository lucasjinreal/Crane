// SPDX-License-Identifier: MIT
//! Crane Added 20260804: ONNX `TopK` as a native eval op, needed by
//! `pack_padded_sequence` in a fine-tuned Kokoro backbone's text encoder —
//! exporters use full-width `TopK` as a stand-in for `Sort`.

use candle_core::{DType, Result, Tensor, bail};

use crate::onnx::eval::{get_attr_opt, to_scalar_flexible};
use crate::onnx::proto::NodeProto;

/// ONNX `TopK`: <https://onnx.ai/onnx/operators/onnx__TopK.html>.
///
/// Candle has no native top-k op, but does have `arg_sort_last_dim`, which
/// is enough to build one: move the target axis to the last position, sort
/// it, keep the first `k` indices, then transpose back. Returns
/// `(values, indices)`; the caller is responsible for wiring the second
/// output only if the node declares one.
pub(crate) fn top_k(node: &NodeProto, xs: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
    let k_raw = to_scalar_flexible::<i64>(k)?;
    if k_raw < 0 {
        bail!("TopK: k must be non-negative, got {k_raw}");
    }
    let k = usize::try_from(k_raw).map_err(candle_core::Error::wrap)?;
    let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(-1);
    let axis = xs.normalize_axis(axis)?;
    let largest = get_attr_opt::<i64>(node, "largest")?.copied().unwrap_or(1) != 0;
    // The `sorted` attribute (default 1) is not read — `arg_sort_last_dim`
    // always returns sorted order, so the output is valid for both
    // sorted=0 and sorted=1.

    let last = xs.rank() - 1;
    let dim_size = xs.dims()[axis];
    if k > dim_size {
        bail!("TopK: k ({k}) exceeds axis {axis} dimension ({dim_size})");
    }
    let xs_t = if axis == last {
        xs.contiguous()?
    } else {
        xs.transpose(axis, last)?.contiguous()?
    };

    // `arg_sort_last_dim`'s `asc` sorts ascending; TopK's `largest` (default
    // true) wants descending order.
    let sort_idx = xs_t.arg_sort_last_dim(!largest)?;
    // `narrow` on the last dim of a multi-row tensor leaves the result
    // non-contiguous (row stride still spans the pre-narrow width), but
    // `gather` requires contiguous input.
    let topk_idx = sort_idx.narrow(last, 0, k)?.contiguous()?;
    let topk_values = xs_t.gather(&topk_idx, last)?;

    let (out_values, out_indices) = if axis == last {
        (topk_values, topk_idx)
    } else {
        (
            topk_values.transpose(axis, last)?.contiguous()?,
            topk_idx.transpose(axis, last)?.contiguous()?,
        )
    };

    Ok((out_values, out_indices.to_dtype(DType::I64)?))
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    use super::top_k;
    use crate::onnx::proto::attribute_proto::AttributeType;
    use crate::onnx::proto::{AttributeProto, NodeProto};

    fn topk_node(axis: i64, largest: i64) -> NodeProto {
        NodeProto {
            op_type: "TopK".to_string(),
            input: vec!["x".to_string(), "k".to_string()],
            output: vec!["values".to_string(), "indices".to_string()],
            attribute: vec![
                AttributeProto {
                    name: "axis".to_string(),
                    r#type: AttributeType::Int as i32,
                    i: axis,
                    ..Default::default()
                },
                AttributeProto {
                    name: "largest".to_string(),
                    r#type: AttributeType::Int as i32,
                    i: largest,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn top_k_largest_along_last_axis() -> Result<()> {
        // Verifies the default largest=1 path on the last axis: for each
        // row, the top 2 values (descending) and their original indices.
        let node = topk_node(-1, 1);
        let x = Tensor::new(&[[3f32, 1., 4., 1.], [9., 2., 6., 5.]], &Device::Cpu)?;
        let k = Tensor::new(&[2i64], &Device::Cpu)?;

        let (values, indices) = top_k(&node, &x, &k)?;

        assert_eq!(values.to_vec2::<f32>()?, vec![vec![4., 3.], vec![9., 6.]]);
        assert_eq!(indices.to_vec2::<i64>()?, vec![vec![2, 0], vec![0, 2]]);
        Ok(())
    }

    #[test]
    fn top_k_full_width_acts_as_sort() -> Result<()> {
        // Verifies the k == axis-length case (exporters' Sort stand-in,
        // per the commit motivating this op): the full row comes back
        // sorted descending with matching indices.
        let node = topk_node(-1, 1);
        let x = Tensor::new(&[[3f32, 1., 2.], [6., 5., 4.]], &Device::Cpu)?;
        let k = Tensor::new(&[3i64], &Device::Cpu)?;

        let (values, indices) = top_k(&node, &x, &k)?;

        assert_eq!(values.to_vec2::<f32>()?, vec![vec![3., 2., 1.], vec![6., 5., 4.]]);
        assert_eq!(indices.to_vec2::<i64>()?, vec![vec![0, 2, 1], vec![0, 1, 2]]);
        Ok(())
    }

    #[test]
    fn top_k_defaults_to_last_axis_and_largest() -> Result<()> {
        // Verifies the default attribute values (axis=-1, largest=1) when
        // neither is present on the node, exercising the `unwrap_or`
        // branches directly.
        let node = NodeProto {
            op_type: "TopK".to_string(),
            input: vec!["x".to_string(), "k".to_string()],
            output: vec!["values".to_string(), "indices".to_string()],
            ..Default::default()
        };
        let x = Tensor::new(&[[5f32, 2., 9.], [1., 8., 3.]], &Device::Cpu)?;
        let k = Tensor::new(&[1i64], &Device::Cpu)?;

        let (values, indices) = top_k(&node, &x, &k)?;

        assert_eq!(values.to_vec2::<f32>()?, vec![vec![9.], vec![8.]]);
        assert_eq!(indices.to_vec2::<i64>()?, vec![vec![2], vec![1]]);
        Ok(())
    }

    #[test]
    fn top_k_middle_axis_on_rank3() -> Result<()> {
        // Verifies a middle (non-last, non-zero) axis on a rank-3 tensor,
        // closer to the [batch, seq, hidden] shapes TopK sees in practice.
        let node = topk_node(1, 1);
        let x = Tensor::new(
            &[[[1f32, 2.], [9., 3.], [5., 8.]], [[4., 6.], [7., 1.], [2., 9.]]],
            &Device::Cpu,
        )?;
        let k = Tensor::new(&[2i64], &Device::Cpu)?;

        let (values, indices) = top_k(&node, &x, &k)?;

        assert_eq!(
            values.to_vec3::<f32>()?,
            vec![vec![vec![9., 8.], vec![5., 3.]], vec![vec![7., 9.], vec![4., 6.]]]
        );
        assert_eq!(
            indices.to_vec3::<i64>()?,
            vec![vec![vec![1, 2], vec![2, 1]], vec![vec![1, 2], vec![0, 0]]]
        );
        Ok(())
    }

    #[test]
    fn top_k_smallest_along_axis0() -> Result<()> {
        // Verifies largest=0 (smallest-first) on a non-last axis (0),
        // exercising the transpose-to-last-dim-and-back path.
        let node = topk_node(0, 0);
        let x = Tensor::new(&[[3f32, 8.], [1., 2.], [5., 0.]], &Device::Cpu)?;
        let k = Tensor::new(&[2i64], &Device::Cpu)?;

        let (values, indices) = top_k(&node, &x, &k)?;

        assert_eq!(values.to_vec2::<f32>()?, vec![vec![1., 0.], vec![3., 2.]]);
        assert_eq!(indices.to_vec2::<i64>()?, vec![vec![1, 2], vec![0, 1]]);
        Ok(())
    }
}
