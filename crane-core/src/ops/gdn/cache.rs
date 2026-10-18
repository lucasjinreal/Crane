//! Per-layer state for a Gated Delta Net: causal `Conv1D` ring buffer plus the
//! recurrent `(K, V)` matrix per head.

use candle_core::{DType, Device, Result, Tensor};

use super::config::GdnDims;

/// State held by one Gated Delta Net layer across decoding steps.
///
/// `conv_state` rolls the most recent `conv_kernel_size - 1` QKV channels so
/// the next decode step can apply the causal `Conv1D` without reprocessing
/// history. `recurrent_state` is the per-head `(K, V)` matrix that carries the
/// linear-attention memory; always f32 regardless of model dtype.
#[derive(Debug)]
pub struct GdnLayerCache {
    pub conv_state: Tensor,
    pub recurrent_state: Tensor,
    /// Speculative decoding: capture a copy of the state after this many
    /// tokens of the next forward (see [`Self::take_checkpoint`]). `None`
    /// (the default) is the ordinary path.
    pub checkpoint_at: Option<usize>,
    /// State captured at `checkpoint_at`: `(conv_state, recurrent_state)`.
    pub checkpoint: Option<(Tensor, Tensor)>,
    /// Run the next forward's recurrence on the serial kernel even for
    /// `seq_len > 1` (short speculative-decoding blocks; see `fused_recurrence`).
    pub prefer_serial: bool,
}

impl GdnLayerCache {
    /// Allocate zero-initialized state for a single sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if allocating either state tensor on `device` fails.
    pub fn new(cfg: &dyn super::config::GdnConfig, dtype: DType, device: &Device) -> Result<Self> {
        let dims = GdnDims::new(cfg);
        // 3D shape `[1, conv_dim, kernel_size]` — the leading dim lets the
        // cache broadcast cleanly against per-batch QKV tensors in
        // `causal_conv1d_full` / `_update`.
        let conv_state = Tensor::zeros((1, dims.conv_dim, dims.conv_kernel_size), dtype, device)?;
        let recurrent_state = Tensor::zeros(
            (1, dims.num_v_heads, dims.head_k_dim, dims.head_v_dim),
            DType::F32,
            device,
        )?;
        Ok(Self {
            conv_state,
            recurrent_state,
            checkpoint_at: None,
            checkpoint: None,
            prefer_serial: false,
        })
    }

    /// Deep-copy the current state as the checkpoint. A real copy, not a
    /// `Tensor::clone`: the fused recurrence kernels may update state in place.
    ///
    /// # Errors
    ///
    /// Returns an error if copying either state tensor fails.
    pub fn save_checkpoint(&mut self) -> Result<()> {
        self.checkpoint = Some((self.conv_state.copy()?, self.recurrent_state.copy()?));
        Ok(())
    }

    /// Replace the live state with the checkpoint (if one was captured) and
    /// clear the speculative-decoding bookkeeping.
    pub fn take_checkpoint(&mut self) {
        if let Some((conv, rec)) = self.checkpoint.take() {
            self.conv_state = conv;
            self.recurrent_state = rec;
        }
        self.checkpoint_at = None;
        self.prefer_serial = false;
    }

    /// Zero out the state for reuse (e.g. between unrelated requests sharing a
    /// pre-allocated layer). Avoids re-allocating the underlying tensors.
    ///
    /// # Errors
    ///
    /// Returns an error if zeroing either state tensor fails.
    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.recurrent_state = self.recurrent_state.zeros_like()?;
        Ok(())
    }
}

impl Clone for GdnLayerCache {
    fn clone(&self) -> Self {
        Self {
            conv_state: self.conv_state.clone(),
            recurrent_state: self.recurrent_state.clone(),
            checkpoint_at: self.checkpoint_at,
            checkpoint: self.checkpoint.clone(),
            prefer_serial: self.prefer_serial,
        }
    }
}
