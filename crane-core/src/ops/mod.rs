//! Custom GPU kernels and other core ops for Crane transformer inference.
//!
//! Submodules:
//! - [`arg_reduce`] — `argmax`/`argmin` along any dim, correct on Metal.
//! - [`fused_ops`] — Fused elementwise/normalisation kernels (silu-mul,
//!   add+rmsnorm, `gpu_argmax`, top-k, HtoD/DtoH copies).
//! - [`gdn`]       — Gated Delta Net recurrence (linear-attention path used
//!   by Qwen 3.5 hybrid layers), with a fused recurrence kernel.
//! - [`linear`]    — Polymorphic linear layer (standard or GGUF-quantized)
//!   plus in-situ quantization helpers.
//! - `rocm`        — shared launch plumbing for the `rocm` builds of the two
//!   kernel modules above.
//!
//! The kernel sources in `crane-core/kernels/` are shared: `build.rs` compiles
//! them to PTX for CUDA, and on `ROCm` candle hands the same text to `hipcc` at
//! first use.

pub mod arg_reduce;
pub mod biased_linear;
#[cfg(any(feature = "flash-attn", feature = "flash-inference"))]
pub mod flash_split;
pub mod float_math;
pub mod fused_ops;
pub mod gdn;
pub mod hyper_connection;
pub mod linear;
pub mod linear_no_bias;
#[cfg(feature = "metal")]
pub mod metal_util;
#[cfg(feature = "cuda")]
pub mod moe_grouped;
#[cfg(feature = "cudnn")]
pub mod patch_conv3d;
pub mod quant_iq;
pub mod quant_ternary;
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub mod rocm;

pub use fused_ops::*;
pub use gdn::*;
pub use linear::LinearLayer;

#[cfg(feature = "cuda")]
pub mod cuda_graph;
