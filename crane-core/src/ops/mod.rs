//! Custom CUDA kernels and other core ops for Crane transformer inference.
//!
//! Submodules:
//! - [`fused_ops`] — Fused elementwise/normalisation kernels (silu-mul,
//!   add+rmsnorm, gpu_argmax, top-k, HtoD/DtoH copies).
//! - [`gdn`]       — Gated Delta Net recurrence (linear-attention path used
//!   by Qwen 3.5 hybrid layers), with a fused CUDA recurrence kernel.

pub mod fused_ops;
pub mod gdn;

pub use fused_ops::*;
pub use gdn::*;