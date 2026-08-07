// SPDX-License-Identifier: MIT
//! Crane Added 20260804: ONNX `Pad` as a native eval op, covering
//! `mode="reflect"` and `mode="edge"` — the latter needed by a fine-tuned
//! Kokoro backbone's `ISTFTNet` vocoder, which pads its overlap-add
//! reconstruction window edges by replicating the border sample instead of
//! reflecting.

use candle_core::{Result, Tensor, bail};

use crate::onnx::proto::NodeProto;

/// Infinite `min..max, max..min` bounce sequence used by `mode="reflect"`'s
/// index construction below.
fn zigzag(min: i64, max: i64) -> impl Iterator<Item = i64> {
    std::iter::repeat((min..max).chain((min + 1..=max).rev())).flatten()
}

/// ONNX `Pad`: <https://onnx.ai/onnx/operators/onnx__Pad.html>.
///
/// Only `mode="reflect"` and `mode="edge"` are implemented; any other mode
/// (including the ONNX default, `"constant"`) is rejected. Negative `pads`
/// values (cropping) are also rejected rather than silently computing a
/// bogus output length.
pub(crate) fn pad(node: &NodeProto, data: &Tensor, pads: &Tensor, mode: &str) -> Result<Tensor> {
    if pads.rank() != 1 {
        bail!("Pad expects 'pads' input to be 1D vector: {pads:?}");
    }
    if pads.dim(0).unwrap() != 2 * data.rank() {
        bail!(
            "Pad expects 'pads' input len to be 2 * rank of 'data' input: pads: {}, data rank: {}",
            pads,
            data.rank()
        );
    }

    let pads = pads.to_vec1::<i64>()?;
    if pads.iter().any(|&p| p < 0) {
        bail!("Pad: negative pad values (cropping) are not supported");
    }
    let (pads_pre, pads_post) = pads.split_at(pads.len() / 2);

    match mode {
        "reflect" => {
            let mut out = data.clone();
            for (i, &dim) in data.dims().iter().enumerate().rev() {
                if pads_pre[i] == 0 && pads_post[i] == 0 {
                    continue;
                }
                // `pads_pre`/`pads_post` were already checked non-negative
                // above, so these i64->usize casts never lose the sign;
                // pad/dimension counts stay far below either type's range.
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let (pre_count, post_count) = (pads_pre[i] as usize, pads_post[i] as usize);
                let idx = if dim > 1 {
                    let cycle_len = dim * 2 - 2;
                    let skip = cycle_len - (pre_count % cycle_len);
                    #[allow(clippy::cast_possible_wrap)]
                    let idx = zigzag(0, (dim - 1) as i64)
                        .skip(skip)
                        .take(pre_count + dim + post_count);
                    Tensor::from_iter(idx, out.device())?
                } else {
                    Tensor::full(0i64, (dim,), out.device())?
                };

                out = out.index_select(&idx, i)?;
            }

            Ok(out)
        },
        // Crane Added 20260804: new mode, needed by a fine-tuned Kokoro
        // backbone's ISTFTNet vocoder — its overlap-add reconstruction
        // pads the STFT window edges by replicating the border sample
        // instead of reflecting.
        "edge" => {
            // pad value at each new position is the nearest border
            // element of `data`, i.e. `clamp(j - pad_pre, 0, dim - 1)`
            // for output position `j` along that axis.
            let mut out = data.clone();
            for (i, &dim) in data.dims().iter().enumerate().rev() {
                if pads_pre[i] == 0 && pads_post[i] == 0 {
                    continue;
                }
                if dim == 0 {
                    bail!(
                        "Pad mode=\"edge\" requires non-empty axis {i} for Pad node {:?}",
                        node.name
                    );
                }
                // A tensor dimension is always far below i64::MAX, so this never wraps.
                #[allow(clippy::cast_possible_wrap)]
                let dim = dim as i64;
                let out_len = pads_pre[i] + dim + pads_post[i];
                let idx = (0..out_len).map(|j| (j - pads_pre[i]).clamp(0, dim - 1));
                let idx = Tensor::from_iter(idx, out.device())?;
                out = out.index_select(&idx, i)?;
            }

            Ok(out)
        },
        _ => bail!("unsupported 'mode' value {mode:?} for Pad node {:?}", node.name),
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::pad;
    use crate::onnx::proto::NodeProto;

    fn pad_node() -> NodeProto {
        NodeProto {
            op_type: "Pad".to_string(),
            input: vec!["x".to_string(), "pads".to_string()],
            output: vec!["y".to_string()],
            ..Default::default()
        }
    }

    #[test]
    fn pad_edge_mode_replicates_border_elements() -> Result<()> {
        // "edge" mode pads with a repeat of the nearest border element,
        // unlike "reflect" (which mirrors) — [1, 2, 3] padded (2, 1) must
        // become [1, 1, 1, 2, 3, 3].
        let node = pad_node();
        let x = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
        let pads = Tensor::new(&[2i64, 1], &Device::Cpu)?;

        let y = pad(&node, &x, &pads, "edge")?;

        assert_eq!(y.to_vec1::<f32>()?, vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0]);
        Ok(())
    }

    #[test]
    fn pad_edge_mode_on_2d_pads_both_axes() -> Result<()> {
        // Verifies "edge" mode pads each axis independently against a
        // rank-2 input, replicating the nearest row/column border.
        let node = pad_node();
        let x = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu)?;
        // pads = [pre_axis0, pre_axis1, post_axis0, post_axis1]
        let pads = Tensor::new(&[1i64, 0, 0, 1], &Device::Cpu)?;

        let y = pad(&node, &x, &pads, "edge")?;

        assert_eq!(y.to_vec2::<f32>()?, vec![
            vec![1.0, 2.0, 2.0],
            vec![1.0, 2.0, 2.0],
            vec![3.0, 4.0, 4.0],
        ]);
        Ok(())
    }

    #[test]
    fn pad_edge_mode_zero_length_axis_errors() -> Result<()> {
        // "edge" mode has no border element to replicate on a zero-length
        // axis; this must return an error rather than panic in `clamp`.
        let node = pad_node();
        let x = Tensor::zeros((0,), DType::F32, &Device::Cpu)?;
        let pads = Tensor::new(&[1i64, 1], &Device::Cpu)?;

        assert!(pad(&node, &x, &pads, "edge").is_err());
        Ok(())
    }

    #[test]
    fn pad_negative_pads_errors() -> Result<()> {
        // ONNX allows negative pads for cropping, but Crane doesn't
        // implement cropping; this must return an error rather than
        // silently computing a bogus output length.
        let node = pad_node();
        let x = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
        let pads = Tensor::new(&[-1i64, 0], &Device::Cpu)?;

        assert!(pad(&node, &x, &pads, "edge").is_err());
        Ok(())
    }
}
