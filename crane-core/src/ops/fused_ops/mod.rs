//! Fused GPU kernels for Crane transformer inference.
//!
//! On a GPU build this module provides:
//! - `fused_silu_mul` — Fused SiLU(gate) * up in one pass
//! - `gpu_argmax` — GPU-side argmax for greedy sampling
//! - `topk_indices` — GPU top-k on 1D f32 tensors
//! - `copy_from_slice_u32` — HtoD: create a new U32 tensor from a host slice
//! - `copy_from_tensor_f32` — contiguous copy of a device f32 tensor
//!
//! Each operation eliminates multiple kernel launches and intermediate
//! GMEM round-trips compared to the equivalent candle op chain.
//!
//! Four implementations, picked at compile time from the backend features:
//! `cuda_impl` (PTX built by `build.rs`), `rocm_impl` (the same `.cu` sources,
//! compiled by `hipcc` on first use), `sycl_impl` (a crate-native `.cpp`,
//! built by `build.rs` with `icpx` into the same `.so` as the GDN kernel —
//! only `fused_silu_mul` has a kernel there so far, everything else falls
//! back to `portable`) and [`portable`], which needs no kernels at all.
//! `cuda`, `rocm` and `sycl` are mutually exclusive in a working build —
//! candle-core cannot link more than one GPU backend — so `cuda` wins if more
//! than one is on.
//!
//! Reusable elementwise ops with their own CPU/CUDA dispatch (not gated by
//! the `cuda` feature at the module level — each op's `CustomOp2` handles
//! backend dispatch internally):
//! - [`snake`] — fused Snake periodic activation
//! - [`atan2`] — fused two-argument arctangent
//! - [`topk_moe`] — fused top-K `MoE` routing (softmax + top-K + normalize)
//! - [`swiglu`] — fused `SwiGLU` activation (silu(gate) * up)

pub mod atan2;
pub mod snake;
pub mod swiglu;
pub mod topk_moe;

#[cfg(feature = "cuda")]
mod cuda_impl;

// Top-k has its own module on both GPU backends, matching its own kernel
// source: it shares nothing with the fused elementwise kernels.
#[cfg(feature = "cuda")]
mod cuda_topk;

#[cfg(feature = "cuda")]
pub use cuda_impl::*;

#[cfg(feature = "cuda")]
pub use cuda_topk::topk_indices;

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
mod rocm_impl;

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub use rocm_impl::*;

#[cfg(all(feature = "sycl", not(any(feature = "cuda", feature = "rocm"))))]
mod sycl_impl;

#[cfg(all(feature = "sycl", not(any(feature = "cuda", feature = "rocm"))))]
pub use sycl_impl::*;

// Always compiled: it needs no backend, the GPU entry points fall back to it
// (off a GPU device, and above the top-k kernel's maximum `k`), and the kernel
// cross-check tests compare against it.
pub mod portable;

#[cfg(not(any(feature = "cuda", feature = "rocm", feature = "sycl")))]
pub use portable::*;
