//! Per-layer state for a Gated Delta Net: causal Conv1D ring buffer plus the
//! recurrent `(K, V)` matrix per head.

use candle_core::{DType, Device, Result, Tensor};

use super::config::GdnDims;

/// State held by one Gated Delta Net layer across decoding steps.
///
/// `conv_state` rolls the most recent `conv_kernel_size - 1` QKV channels so
/// the next decode step can apply the causal Conv1D without reprocessing
/// history. `recurrent_state` is the per-head `(K, V)` matrix that carries the
/// linear-attention memory; always f32 regardless of model dtype.
#[derive(Debug)]
pub struct GdnLayerCache {
    pub conv_state: Tensor,
    pub recurrent_state: Tensor,
}

impl GdnLayerCache {
    /// Allocate zero-initialized state for a single sequence.
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
        })
    }

    /// Zero out the state for reuse (e.g. between unrelated requests sharing a
    /// pre-allocated layer). Avoids re-allocating the underlying tensors.
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
        }
    }
}