//! Gated Delta Net (GDN) implementation for hybrid attention models.
//!
//! Used by Qwen 3.5 (and other hybrid Mamba/Transformer architectures) for the
//! `linear_attention` layers that interleave with standard `full_attention`
//! layers.
//!
//! # Algorithm
//!
//! Given per-timestep Q, K, V, g (decay), β (write strength), and a learned
//! negative-exponent A_log used to derive per-head decay, the recurrence is:
//!
//! ```text
//! S = S * exp(g)                  // decay state by head-dependent factor
//! kv_mem = sum(S * k, d_state)    // retrieve current memory at this k
//! delta = (v - kv_mem) * beta     // delta-rule residual
//! S = S + outer(k, delta)         // write update
//! y = sum(S * q, d_state)         // output via q-projection of state
//! ```
//!
//! State shape: `[batch, num_v_heads, head_k_dim, head_v_dim]`, kept in f32
//! regardless of model dtype (`mamba_ssm_dtype: float32` in the HF config).
//!
//! The QKV input also passes through a causal Conv1D with kernel width 4
//! (`linear_conv_kernel_dim`) before the recurrence; the conv state is
//! maintained alongside the recurrent state for decode-step inference.
//!
//! # Origin
//!
//! Algorithm inspired by [mistral.rs](https://github.com/EricLBuehler/mistral.rs)
//! (MIT licensed); Crane's implementation uses plain `candle_nn::Linear`
//! without that project's tensor-parallel / quantization abstractions.
//!
//! See also the [Gated Delta Net paper](https://arxiv.org/abs/2412.06464)
//! (Yang et al., 2024) for the original formulation.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod backend;
mod cache;
mod config;
#[cfg(feature = "cuda")]
mod cuda_backend;
mod layer;
mod norm;
mod projection;

pub use backend::{
    causal_conv1d, gated_delta_rule_recurrence, l2_norm, softplus, apply_recurrence,
};
pub use cache::GdnLayerCache;
pub use config::{defaults, GdnConfig, GdnDims};
pub use layer::GatedDeltaNet;
pub use norm::RmsNormGated;
pub use projection::{GdnInputProjection, GdnInputProjectionKind, GdnProjection};