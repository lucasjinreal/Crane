//! K/V cache for the full-attention layers of Qwen 3.5.
//!
//! The hybrid model only needs this for the 1-in-4 full-attention blocks; the
//! linear-attention (GDN) blocks carry a constant-size recurrent state instead
//! (see [`crate::ops::gdn::GdnLayerCache`]), so the context-growing part of the
//! cache lives in just these layers. At long context that K/V dominates memory,
//! which is why quantizing it lets a single agent hold much more context
//! locally (e.g. Ornith-9B's full 262K window on a 24 GB GPU).
>>>>>>> 495f6b1 (fixup! Add Qwen 3.5 hybrid GDN/attention support)
//!
//! # The swap seam
//!
//! This is a lossless fp store. It is the single place a quantized K/V backend
//! (e.g. rotation + scalar quant for long-context local setups) would slot in:
//! a future variant only has to honor the same contract —
//!
//! - [`KvCache::append`] takes the new post-RoPE `k`/`v` for the current step
//!   (`[B, num_kv_heads, S, head_dim]`) and returns the *full* `k`/`v` spanning
//!   all cached positions, ready for attention.
//! - [`KvCache::reset`] clears it between unrelated requests.
//!
//! Attention logic never touches the storage representation, so swapping fp for
//! a quantized backend won't reach into the attention math.

use candle_core::{Result, Tensor};

/// Pre-allocated, append-in-place K/V cache for one full-attention layer.
///
/// Mirrors the `qwen3` model's cache: a buffer that may be larger than the
/// filled length, written with `slice_set` (O(new tokens), not `cat`), grown
/// with a small fixed headroom only when it overflows.
#[derive(Debug)]
pub struct KvCache {
    /// `(k, v)` buffers of shape `[B, num_kv_heads, capacity, head_dim]`.
    buffers: Option<(Tensor, Tensor)>,
    /// Number of valid (filled) positions in the buffers.
    seq_len: usize,
}

/// Headroom (in positions) added when (re)allocating, to amortize growth.
const ROOM: usize = 256;

impl Default for KvCache {
    fn default() -> Self {
        Self::new()
    }
}

impl KvCache {
    pub fn new() -> Self {
        Self {
            buffers: None,
            seq_len: 0,
        }
    }

    /// Filled length (number of cached positions).
    pub fn len(&self) -> usize {
        self.seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Drop all cached state (between unrelated requests).
    pub fn reset(&mut self) {
        self.buffers = None;
        self.seq_len = 0;
    }

    /// Append `k`/`v` for the current step and return the full cached `(k, v)`.
    ///
    /// `k`/`v` are `[B, num_kv_heads, S, head_dim]` (post-RoPE, pre-GQA-expand).
    /// The returned tensors span `[B, num_kv_heads, seq_len + S, head_dim]`.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let new_seq_len = k.dim(2)?;
        let filled = self.seq_len;

        match self.buffers.take() {
            None => {
                // First write — allocate with headroom.
                let (b, h, s, d) = k.dims4()?;
                let buf_k = Tensor::zeros((b, h, s + ROOM, d), k.dtype(), k.device())?;
                let buf_v = Tensor::zeros((b, h, s + ROOM, d), v.dtype(), v.device())?;
                buf_k.slice_set(&k, 2, 0)?;
                buf_v.slice_set(&v, 2, 0)?;
                self.buffers = Some((buf_k, buf_v));
                self.seq_len = s;
                Ok((k, v))
            }
            Some((buf_k, buf_v)) => {
                let capacity = buf_k.dim(2)?;
                let new_total = filled + new_seq_len;

                if new_total <= capacity {
                    // Fits — in-place write, O(new_seq_len).
                    buf_k.slice_set(&k, 2, filled)?;
                    buf_v.slice_set(&v, 2, filled)?;
                    let k_view = buf_k.narrow(2, 0, new_total)?;
                    let v_view = buf_v.narrow(2, 0, new_total)?;
                    self.buffers = Some((buf_k, buf_v));
                    self.seq_len = new_total;
                    Ok((k_view, v_view))
                } else {
                    // Overflow — concat then reallocate with headroom.
                    let cur_k = buf_k.narrow(2, 0, filled)?;
                    let cur_v = buf_v.narrow(2, 0, filled)?;
                    let full_k = Tensor::cat(&[&cur_k, &k], 2)?;
                    let full_v = Tensor::cat(&[&cur_v, &v], 2)?;
                    let (b, h, total, d) = full_k.dims4()?;
                    let new_buf_k = Tensor::zeros((b, h, total + ROOM, d), k.dtype(), k.device())?;
                    let new_buf_v = Tensor::zeros((b, h, total + ROOM, d), v.dtype(), v.device())?;
                    new_buf_k.slice_set(&full_k, 2, 0)?;
                    new_buf_v.slice_set(&full_v, 2, 0)?;
                    self.buffers = Some((new_buf_k, new_buf_v));
                    self.seq_len = total;
                    Ok((full_k, full_v))
                }
            }
        }
    }
}
