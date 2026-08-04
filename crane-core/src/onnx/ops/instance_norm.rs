// SPDX-License-Identifier: MIT
//! Crane Added 20260804: ONNX `InstanceNormalization` as a native eval op.

use candle_core::{DType, Result, Tensor, bail};

use crate::onnx::proto::{self, NodeProto};

/// ONNX `InstanceNormalization`: normalizes each `(batch, channel)` slice of
/// `x` independently over its remaining (spatial) dimensions, then scales
/// and shifts per channel.
///
/// `x` has shape `(N, C, D1, ..., Dn)`; `scale`/`bias` have shape `(C,)`.
/// `epsilon` (default `1e-5`) is added to the variance before the square
/// root to avoid division by zero. Unlike
/// [`layer_norm`](super::layer_norm::layer_norm), which reduces over a
/// trailing-axis suffix, this always excludes both the batch axis (0) and
/// the channel axis (1) from the reduction, matching `PyTorch`'s
/// `InstanceNorm*d` (always per-sample, per-channel).
pub(crate) fn instance_normalization(
    node: &NodeProto,
    x: &Tensor,
    scale: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    let epsilon = float_attribute(node, "epsilon", 1e-5)?;

    let rank = x.rank();
    if rank < 3 {
        bail!("InstanceNormalization node '{}' expects rank >= 3, got {rank}", node.name);
    }
    let channels = x.dim(1)?;
    if scale.rank() != 1 || scale.dim(0)? != channels {
        bail!(
            "InstanceNormalization node '{}': scale must be shape ({channels},), got {:?}",
            node.name,
            scale.shape(),
        );
    }
    if bias.rank() != 1 || bias.dim(0)? != channels {
        bail!(
            "InstanceNormalization node '{}': bias must be shape ({channels},), got {:?}",
            node.name,
            bias.shape(),
        );
    }
    let reduce_axes: Vec<usize> = (2..rank).collect();

    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        dtype => dtype,
    };
    let x_internal = x.to_dtype(internal_dtype)?;

    let mean = x_internal.mean_keepdim(reduce_axes.as_slice())?;
    let centered = x_internal.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(reduce_axes.as_slice())?;
    let normalized = centered.broadcast_div(&(variance + epsilon)?.sqrt()?)?;

    // scale/bias are shape (C,); reshape to (1, C, 1, ..., 1) so
    // broadcasting aligns C with axis 1 instead of the trailing axis.
    let mut param_shape = vec![1usize; rank];
    param_shape[1] = channels;
    let scale = scale.reshape(param_shape.as_slice())?;
    let bias = bias.reshape(param_shape)?;

    normalized.to_dtype(x_dtype)?.broadcast_mul(&scale)?.broadcast_add(&bias)
}

fn float_attribute(node: &NodeProto, name: &str, default: f64) -> Result<f64> {
    let Some(attribute) = node.attribute.iter().find(|attribute| attribute.name == name) else {
        return Ok(default);
    };
    if attribute.r#type() != proto::attribute_proto::AttributeType::Float {
        bail!(
            "InstanceNormalization node '{}' has a non-FLOAT '{}' attribute ({:?})",
            node.name,
            name,
            attribute.r#type(),
        );
    }
    Ok(f64::from(attribute.f))
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use crate::onnx::proto::{AttributeProto, NodeProto, attribute_proto::AttributeType};

    use super::instance_normalization;

    fn node_with_epsilon(epsilon: f32) -> NodeProto {
        NodeProto {
            name: "InstanceNormalization.0".to_string(),
            attribute: vec![AttributeProto {
                name: "epsilon".to_string(),
                r#type: AttributeType::Float as i32,
                f: epsilon,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn default_node() -> NodeProto {
        NodeProto {
            name: "InstanceNormalization.0".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn matches_manual_instance_norm_1d() -> Result<()> {
        // A [1, 2, 4] input (batch=1, 2 channels, 4-length spatial dim):
        // each channel is normalized independently over its own 4 values.
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]], &Device::Cpu)?;
        let scale = Tensor::new(&[1.5f32, 0.5], &Device::Cpu)?;
        let bias = Tensor::new(&[0.1f32, -0.2], &Device::Cpu)?;
        let node = node_with_epsilon(1e-5);

        let output = instance_normalization(&node, &x, &scale, &bias)?;

        let mean = 2.5f32;
        let variance = 1.25f32;
        let std_dev = (variance + 1e-5).sqrt();
        let expected_channel0 = [1.0f32, 2.0, 3.0, 4.0]
            .map(|v| (v - mean) / std_dev * 1.5 + 0.1);
        let expected_channel1 = [4.0f32, 3.0, 2.0, 1.0]
            .map(|v| (v - mean) / std_dev * 0.5 - 0.2);

        let got: Vec<Vec<Vec<f32>>> = output.to_vec3()?;
        for (got, expected) in got[0][0].iter().zip(expected_channel0.iter()) {
            assert!((got - expected).abs() < 1e-4, "{got} vs {expected}");
        }
        for (got, expected) in got[0][1].iter().zip(expected_channel1.iter()) {
            assert!((got - expected).abs() < 1e-4, "{got} vs {expected}");
        }
        Ok(())
    }

    #[test]
    fn each_batch_element_normalized_independently() -> Result<()> {
        // Two batch elements with different value ranges must each be
        // normalized against their own mean/variance, not a shared one.
        let x = Tensor::new(
            &[[[1.0f32, 2.0, 3.0]], [[10.0f32, 20.0, 30.0]]],
            &Device::Cpu,
        )?;
        let scale = Tensor::new(&[1.0f32], &Device::Cpu)?;
        let bias = Tensor::new(&[0.0f32], &Device::Cpu)?;
        let node = node_with_epsilon(1e-5);

        let output = instance_normalization(&node, &x, &scale, &bias)?;
        let got: Vec<Vec<Vec<f32>>> = output.to_vec3()?;

        for batch in got {
            for (value, expected) in batch[0].iter().zip([-1.2247449f32, 0.0, 1.2247449]) {
                assert!((value - expected).abs() < 1e-3, "{value} vs {expected}");
            }
        }
        Ok(())
    }

    #[test]
    fn rejects_rank_below_three() -> Result<()> {
        // Instance norm needs at least one spatial dim beyond (N, C); a
        // rank-2 input must be a clear error, not a silently wrong result.
        let x = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu)?;
        let scale = Tensor::new(&[1.0f32, 1.0], &Device::Cpu)?;
        let bias = Tensor::new(&[0.0f32, 0.0], &Device::Cpu)?;
        let node = default_node();

        let error = instance_normalization(&node, &x, &scale, &bias).unwrap_err();
        assert!(error.to_string().contains("expects rank >= 3"), "{error}");
        Ok(())
    }

    #[test]
    fn uses_default_epsilon_when_absent() -> Result<()> {
        // No "epsilon" attribute on the node must fall back to ONNX's 1e-5
        // default rather than erroring or using 0.
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0, 4.0]]], &Device::Cpu)?;
        let scale = Tensor::new(&[1.0f32], &Device::Cpu)?;
        let bias = Tensor::new(&[0.0f32], &Device::Cpu)?;
        let node = default_node();

        let output = instance_normalization(&node, &x, &scale, &bias)?;

        let mean = 2.5f32;
        let variance = 1.25f32;
        let std_dev = (variance + 1e-5).sqrt();
        let expected = [1.0f32, 2.0, 3.0, 4.0].map(|v| (v - mean) / std_dev);
        let got: Vec<Vec<Vec<f32>>> = output.to_vec3()?;
        for (got, expected) in got[0][0].iter().zip(expected.iter()) {
            assert!((got - expected).abs() < 1e-4, "{got} vs {expected}");
        }
        Ok(())
    }

    #[test]
    fn f16_input_round_trips_through_f32() -> Result<()> {
        // F16/BF16 inputs are promoted to F32 for the mean/variance
        // accumulation; the output must still come back as F16.
        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0, 4.0]]], &Device::Cpu)?.to_dtype(DType::F16)?;
        let scale = Tensor::new(&[2.0f32], &Device::Cpu)?.to_dtype(DType::F16)?;
        let bias = Tensor::new(&[0.5f32], &Device::Cpu)?.to_dtype(DType::F16)?;
        let node = node_with_epsilon(1e-5);

        let output = instance_normalization(&node, &x, &scale, &bias)?;
        assert_eq!(output.dtype(), DType::F16);

        let mean = 2.5f32;
        let variance = 1.25f32;
        let std_dev = (variance + 1e-5).sqrt();
        let expected = [1.0f32, 2.0, 3.0, 4.0].map(|v| (v - mean) / std_dev * 2.0 + 0.5);
        let got: Vec<Vec<Vec<f32>>> = output.to_dtype(DType::F32)?.to_vec3()?;
        for (got, expected) in got[0][0].iter().zip(expected.iter()) {
            assert!((got - expected).abs() < 1e-2, "{got} vs {expected}");
        }
        Ok(())
    }
}
