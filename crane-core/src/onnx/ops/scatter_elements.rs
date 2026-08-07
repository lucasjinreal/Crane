// SPDX-License-Identifier: MIT
//! Crane Added 20260804: ONNX `ScatterElements` as a native eval op, needed
//! by `pack_padded_sequence` in a fine-tuned Kokoro backbone's text encoder
//! (`invert_permutation` builds an inverse permutation via a plain scatter).

use candle_core::{DType, Result, Tensor, bail};

use crate::onnx::eval::get_attr_opt;
use crate::onnx::proto::NodeProto;

/// ONNX `ScatterElements`:
/// <https://onnx.ai/onnx/operators/onnx__ScatterElements.html>.
///
/// Candle's native `scatter`/`scatter_add` map directly onto ONNX's
/// "none"/"add" reduction modes; the two ONNX-18 reduction modes with no
/// candle equivalent (`mul`, `min`, `max`) bail with a clear error instead
/// of silently doing the wrong thing.
pub(crate) fn scatter_elements(
    node: &NodeProto,
    data: &Tensor,
    indices: &Tensor,
    updates: &Tensor,
) -> Result<Tensor> {
    let axis = get_attr_opt::<i64>(node, "axis")?.copied().unwrap_or(0);
    let axis = data.normalize_axis(axis)?;
    let reduction = get_attr_opt::<str>(node, "reduction")?.unwrap_or("none");

    // Crane Added 20260805: normalize indices to I64 up front. scatter's
    // CPU backend only accepts U8/U32/I64, but mid-graph indices (e.g. from
    // Cast/Where/mask ops) can be I32, which would otherwise fail with
    // UnsupportedDTypeForOp.
    let indices = &indices.to_dtype(DType::I64)?;

    // Crane Added 20260805: ONNX allows indices in [-s, s-1], but candle's
    // scatter/scatter_add index directly into the buffer, so negative
    // indices must be normalized to positive first.
    let indices = &{
        let zeros = Tensor::zeros(indices.shape(), indices.dtype(), indices.device())?;
        // A tensor dimension size can never approach i64::MAX, so this never wraps.
        #[allow(clippy::cast_possible_wrap)]
        let dim_size = data.dims()[axis] as i64;
        let max = Tensor::new(dim_size, indices.device())?.to_dtype(indices.dtype())?;
        let mask = indices.lt(&zeros)?;
        mask.to_dtype(indices.dtype())?.broadcast_mul(&max)?.add(indices)?
    };

    match reduction {
        "none" => data.scatter(indices, updates, axis),
        "add" => data.scatter_add(indices, updates, axis),
        other => bail!("unsupported ScatterElements reduction {other} for {}", node.name),
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Result, Tensor};

    use super::scatter_elements;
    use crate::onnx::proto::attribute_proto::AttributeType;
    use crate::onnx::proto::{AttributeProto, NodeProto};

    fn scatter_elements_node(attribute: Vec<AttributeProto>) -> NodeProto {
        NodeProto {
            op_type: "ScatterElements".to_string(),
            input: vec!["data".to_string(), "indices".to_string(), "updates".to_string()],
            output: vec!["y".to_string()],
            attribute,
            ..Default::default()
        }
    }

    #[test]
    fn scatter_elements_default_builds_inverse_permutation() -> Result<()> {
        // Verifies the default reduction="none" path (matching the real
        // ONNX export this op was added for): `torch.nn.utils.rnn`'s
        // `invert_permutation` scatters `arange(n)` into a zeroed tensor
        // at `permutation`'s positions to build the inverse permutation.
        // For permutation [2, 0, 3, 1]: inv[2]=0, inv[0]=1, inv[3]=2,
        // inv[1]=3 -> [1, 3, 0, 2].
        let node = scatter_elements_node(vec![AttributeProto {
            name: "axis".to_string(),
            r#type: AttributeType::Int as i32,
            i: 0,
            ..Default::default()
        }]);
        let data = Tensor::new(&[0i64, 0, 0, 0], &Device::Cpu)?;
        let indices = Tensor::new(&[2i64, 0, 3, 1], &Device::Cpu)?;
        let updates = Tensor::new(&[0i64, 1, 2, 3], &Device::Cpu)?;

        let y = scatter_elements(&node, &data, &indices, &updates)?;

        assert_eq!(y.to_vec1::<i64>()?, vec![1, 3, 0, 2]);
        Ok(())
    }

    #[test]
    fn scatter_elements_add_reduction_accumulates() -> Result<()> {
        // Verifies reduction="add": repeated indices accumulate rather
        // than overwrite.
        let node = scatter_elements_node(vec![
            AttributeProto {
                name: "axis".to_string(),
                r#type: AttributeType::Int as i32,
                i: 0,
                ..Default::default()
            },
            AttributeProto {
                name: "reduction".to_string(),
                r#type: AttributeType::String as i32,
                s: b"add".to_vec(),
                ..Default::default()
            },
        ]);
        let data = Tensor::new(&[0i64, 0, 0], &Device::Cpu)?;
        let indices = Tensor::new(&[0i64, 0, 1], &Device::Cpu)?;
        let updates = Tensor::new(&[1i64, 2, 3], &Device::Cpu)?;

        let y = scatter_elements(&node, &data, &indices, &updates)?;

        assert_eq!(y.to_vec1::<i64>()?, vec![3, 3, 0]);
        Ok(())
    }

    #[test]
    fn scatter_elements_negative_indices() -> Result<()> {
        // Verifies negative indices (ONNX allows [-s, s-1]) are
        // normalized to positive before scattering: index -1 on a
        // length-3 axis must land on position 2, not error/wrap.
        let node = scatter_elements_node(vec![AttributeProto {
            name: "axis".to_string(),
            r#type: AttributeType::Int as i32,
            i: 0,
            ..Default::default()
        }]);
        let data = Tensor::new(&[0i64, 0, 0], &Device::Cpu)?;
        let indices = Tensor::new(&[-1i64, 0], &Device::Cpu)?;
        let updates = Tensor::new(&[9i64, 7], &Device::Cpu)?;

        let y = scatter_elements(&node, &data, &indices, &updates)?;

        assert_eq!(y.to_vec1::<i64>()?, vec![7, 0, 9]);
        Ok(())
    }

    #[test]
    fn scatter_elements_2d_axis1() -> Result<()> {
        // Verifies scattering along a non-zero axis on a
        // multi-dimensional tensor (the two tests above only cover
        // 1-D/axis=0, which can't distinguish real axis handling from
        // a hard-coded flatten).
        let node = scatter_elements_node(vec![AttributeProto {
            name: "axis".to_string(),
            r#type: AttributeType::Int as i32,
            i: 1,
            ..Default::default()
        }]);
        let data = Tensor::new(&[[0i64, 0, 0], [0, 0, 0]], &Device::Cpu)?;
        let indices = Tensor::new(&[[2i64], [0]], &Device::Cpu)?;
        let updates = Tensor::new(&[[5i64], [8]], &Device::Cpu)?;

        let y = scatter_elements(&node, &data, &indices, &updates)?;

        assert_eq!(y.to_vec2::<i64>()?, vec![vec![0, 0, 5], vec![8, 0, 0]]);
        Ok(())
    }

    #[test]
    fn scatter_elements_unsupported_reduction_errors() -> Result<()> {
        // Verifies a reduction mode with no candle equivalent (e.g.
        // "mul") is rejected instead of silently falling back to
        // overwrite semantics.
        let node = scatter_elements_node(vec![AttributeProto {
            name: "reduction".to_string(),
            r#type: AttributeType::String as i32,
            s: b"mul".to_vec(),
            ..Default::default()
        }]);
        let data = Tensor::new(&[1i64, 1, 1], &Device::Cpu)?;
        let indices = Tensor::new(&[0i64], &Device::Cpu)?;
        let updates = Tensor::new(&[2i64], &Device::Cpu)?;

        assert!(scatter_elements(&node, &data, &indices, &updates).is_err());
        Ok(())
    }
}
