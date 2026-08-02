//! Custom GPU kernels and other core ops for Crane transformer inference.
//!
//! Submodules:
//! - [`fused_ops`] — Fused elementwise/normalisation kernels (silu-mul,
//!   add+rmsnorm, gpu_argmax, top-k, HtoD/DtoH copies).
//! - [`gdn`]       — Gated Delta Net recurrence (linear-attention path used
//!   by Qwen 3.5 hybrid layers), with a fused recurrence kernel.
//! - [`linear`]    — Polymorphic linear layer (standard or GGUF-quantized)
//!   plus in-situ quantization helpers.
//! - `rocm`        — shared launch plumbing for the `rocm` builds of the two
//!   kernel modules above.
//!
//! The kernel sources in `crane-core/kernels/` are shared: `build.rs` compiles
//! them to PTX for CUDA, and on ROCm candle hands the same text to `hipcc` at
//! first use.

pub mod fused_ops;
pub mod gdn;
pub mod linear;
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub mod rocm;

pub use fused_ops::*;
pub use gdn::*;
pub use linear::LinearLayer;