//! Causal Conv1D over the QKV channels, in pure Candle ops.
//!
//! Split out of `backend.rs`, which is at the workspace 400-line cap.
//!
//! The convolution is expressed as `kernel` shifted multiply-accumulates over
//! the whole sequence rather than one windowed reduction per timestep — see
//! [`shift_accumulate`] for why that matters.

use candle_core::{Result, Tensor};

use super::cache::GdnLayerCache;
use super::config::GdnDims;

/// Causal Conv1D over the QKV channels, continuing from the cached state.
///
/// One implementation covers every call shape: the first prefill (the state is
/// zeros, so prepending it *is* the left zero-pad), a mid-prompt prefill chunk,
/// and a single decode token. A separate zero-padding "whole sequence" variant
/// is exactly what makes chunked prefill silently wrong — it discards the
/// history the previous chunk left in `cache.conv_state`, so the first
/// `kernel - 1` tokens of every chunk after the first see zeros instead of
/// their real predecessors.
pub fn causal_conv1d(
    x: &Tensor,
    conv1d_weight: &Tensor,
    dims: &GdnDims,
    cache: &mut GdnLayerCache,
) -> Result<Tensor> {
    let (_, seq_len, _) = x.dims3()?;
    let x_t = x.transpose(1, 2)?.contiguous()?;
    // `conv_state` is shape `[1, conv_dim, kernel_size]` — the kernel-size
    // dim is the LAST (matches mistral.rs).
    let state_len = cache.conv_state.dim(2)?;
    let hidden = Tensor::cat(&[cache.conv_state.clone(), x_t], 2)?;
    let new_len = hidden.dim(2)?;
    // `contiguous()`: the narrow is a view into `hidden`, which spans the whole
    // chunk. Without the copy each GDN layer would pin a
    // `[1, conv_dim, kernel + chunk]` buffer for the rest of the request.
    cache.conv_state = hidden
        .narrow(2, new_len - state_len, state_len)?
        .contiguous()?;

    let weight = conv1d_weight.squeeze(1)?.to_dtype(hidden.dtype())?;
    // Output `t` reads the `kernel`-wide window ending at `t`, i.e. the cached
    // tail followed by this call's tokens up to `t`.
    let offset = (new_len + 1)
        .checked_sub(seq_len + dims.conv_kernel_size)
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "causal_conv1d: conv state of {state_len} is shorter than kernel {}",
                dims.conv_kernel_size
            ))
        })?;
    let out = shift_accumulate(&hidden, &weight, seq_len, offset)?;
    candle_nn::ops::silu(&out)?.transpose(1, 2)
}

/// `out[.., .., t] = sum_j padded[.., .., offset + t + j] * weight[.., j]`.
///
/// The convolution as `kernel` shifted multiply-accumulates over the whole
/// sequence at once, rather than one windowed reduction per timestep. The
/// per-timestep form costs ~4 candle ops per token per layer, so a 1024-token
/// prefill across 18 GDN layers is ~73 000 kernel launches — the same
/// launch-bound shape the fused recurrence kernel exists to avoid, and on a
/// GPU it dominated what was left of prefill once the recurrence was fixed.
///
/// `padded` is `[B, C, offset + seq_len + kernel - 1]` and `weight` is
/// `[C, kernel]`; the result is `[B, C, seq_len]`.
fn shift_accumulate(
    padded: &Tensor,
    weight: &Tensor,
    seq_len: usize,
    offset: usize,
) -> Result<Tensor> {
    let kernel = weight.dim(1)?;
    let mut acc: Option<Tensor> = None;
    for j in 0..kernel {
        let shifted = padded.narrow(2, offset + j, seq_len)?;
        // [C, 1] -> [1, C, 1], broadcast over batch and time.
        let w_j = weight.narrow(1, j, 1)?.unsqueeze(0)?;
        let term = shifted.broadcast_mul(&w_j)?;
        acc = Some(match acc {
            None => term,
            Some(sum) => (sum + term)?,
        });
    }
    acc.ok_or_else(|| candle_core::Error::Msg("causal_conv1d: zero-width kernel".into()))
}

#[cfg(test)]
mod tests {
    use super::{causal_conv1d, shift_accumulate, GdnDims, GdnLayerCache};
    use crate::ops::gdn::config::VHeadOrder;
    use candle_core::{DType, Device, Tensor, D};

    fn dims(conv_dim: usize, kernel: usize) -> GdnDims {
        GdnDims {
            hidden_size: conv_dim,
            num_k_heads: 1,
            num_v_heads: 1,
            head_k_dim: 1,
            head_v_dim: 1,
            conv_kernel_size: kernel,
            key_dim: 1,
            value_dim: conv_dim - 2,
            conv_dim,
            v_per_group: 1,
            v_head_order: VHeadOrder::Interleaved,
        }
    }

    fn empty_cache(conv_dim: usize, kernel: usize, dev: &Device) -> GdnLayerCache {
        GdnLayerCache {
            conv_state: Tensor::zeros((1, conv_dim, kernel), DType::F32, dev).unwrap(),
            recurrent_state: Tensor::zeros((1, 1, 1, 1), DType::F32, dev).unwrap(),
        }
    }

    /// Splitting a sequence into chunks must give bit-identical output to one
    /// pass, including chunks shorter than the kernel and a length-1 remainder.
    /// The old zero-padding prefill path failed this: it ignored `conv_state`,
    /// so every chunk after the first convolved against zeros.
    #[test]
    fn chunked_conv_matches_the_single_pass() {
        let dev = Device::Cpu;
        let (conv_dim, kernel, seq_len) = (6usize, 4usize, 13usize);
        let d = dims(conv_dim, kernel);
        let x = Tensor::rand(-1f32, 1.0, (1, seq_len, conv_dim), &dev).unwrap();
        let w = Tensor::rand(-1f32, 1.0, (conv_dim, 1, kernel), &dev).unwrap();

        let mut whole_cache = empty_cache(conv_dim, kernel, &dev);
        let whole = causal_conv1d(&x, &w, &d, &mut whole_cache).unwrap();

        // 5 + 5 + 2 + 1: a full chunk, a chunk crossing the kernel boundary, a
        // chunk shorter than the kernel, and a length-1 remainder.
        let mut chunk_cache = empty_cache(conv_dim, kernel, &dev);
        let mut pieces = Vec::new();
        let mut off = 0;
        for len in [5usize, 5, 2, 1] {
            let piece = x.narrow(1, off, len).unwrap().contiguous().unwrap();
            pieces.push(causal_conv1d(&piece, &w, &d, &mut chunk_cache).unwrap());
            off += len;
        }
        let chunked = Tensor::cat(&pieces, 1).unwrap();

        let whole = whole.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let chunked = chunked.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(whole.len(), chunked.len());
        for (a, b) in whole.iter().zip(&chunked) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }

        // The trailing conv state must also agree, or the next decode step
        // diverges even though every prefill output matched.
        let ws = whole_cache
            .conv_state
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cs = chunk_cache
            .conv_state
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (a, b) in ws.iter().zip(&cs) {
            assert!((a - b).abs() < 1e-6, "conv state {a} != {b}");
        }
    }

    /// The shifted multiply-accumulate has to agree with the windowed
    /// per-timestep reduction it replaced, offset included — the decode path
    /// reads its window out of the middle of the conv state.
    #[test]
    fn shift_accumulate_matches_the_windowed_reference() {
        let dev = Device::Cpu;
        let (b, c, kernel, seq_len, offset) = (2usize, 3usize, 4usize, 5usize, 2usize);
        let padded =
            Tensor::rand(0f32, 1.0, (b, c, offset + seq_len + kernel - 1), &dev).unwrap();
        let weight = Tensor::rand(0f32, 1.0, (c, kernel), &dev).unwrap();

        let got = shift_accumulate(&padded, &weight, seq_len, offset).unwrap();

        let mut windows = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let window = padded.narrow(2, offset + t, kernel).unwrap();
            windows.push(
                window
                    .broadcast_mul(&weight.unsqueeze(0).unwrap())
                    .unwrap()
                    .sum(D::Minus1)
                    .unwrap(),
            );
        }
        let want = Tensor::stack(&windows, 2).unwrap();

        assert_eq!(got.dims(), want.dims());
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let want = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "{g} != {w}");
        }
    }
}
