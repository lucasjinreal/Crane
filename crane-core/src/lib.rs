//! # crane-core
//!
//! Core inference library for Crane — a high-performance LLM inference engine
//! written in Rust on top of the [Candle](https://github.com/huggingface/candle)
//! framework.
//!
//! ## Modules
//!
//! | Module | Purpose |
//! |---|---|
//! | [`ops`] | Custom CUDA kernels and other core ops (fused ops, Gated Delta Net) |
//! | [`models`] | Transformer model implementations (Qwen3, `HunyuanDense`, Qwen2.5, multimodal) |
//! | [`generation`] | Token generation utilities (sampling, stopping criteria, logit processors) |
//! | [`autotokenizer`] | HuggingFace-compatible tokenizer loader |
//! | [`chat`] | Chat-template rendering and message formatting |
//! | [`bins`] | Model loading helpers used by CLI entry-points |
//! | [`quantized`] | Shared GGUF file loading (mmap, typed tensor access) |
//! | [`utils`] | Misc tensor / device utilities |
//! | [`device`] | `DeviceAssignment`, bundling a model's main and MoE-expert devices |
//!
//! ## Feature flags
//!
//! | Flag | Effect |
//! |---|
//! | `metal` | Enable Metal GPU acceleration on macOS |
//! | `cuda` | Enable CUDA device and custom PTX kernels (requires CUDA toolkit) |
//! | `accelerate` | Link against Apple Accelerate for CPU BLAS |
//! | `mkl` | Link against Intel MKL for CPU BLAS |
//! | `sycl` | Enable the Intel oneAPI / SYCL backend (POC; needs a candle fork, see README) |
//! | `flash-attn-intree` | Compile the in-tree flash-attention kernels on CUDA (`ROCm` always has them) |

include!("candle_backend.rs");

// The two CUDA flash-attention backends are alternatives. Building both
// doubles the kernel build cost, so only the A/B test feature may do it.
#[cfg(all(
    feature = "flash-attn-intree",
    any(feature = "flash-attn", feature = "flash-inference"),
    not(feature = "flash-attn-ab")
))]
compile_error!(
    "`flash-attn-intree` and `flash-attn`/`flash-inference` are mutually exclusive. \
     Pick one CUDA flash-attention backend, or enable `flash-attn-ab` to build both for A/B testing."
);

pub mod device;
pub mod generation;
pub mod models;
pub mod ops;
pub mod quantized;
pub mod utils;

#[cfg(feature = "onnx")]
#[path = "onnx.rs"]
pub mod onnx;

pub mod autotokenizer;
pub mod bins;
pub mod chat;

#[doc(hidden)]
pub mod test_data;
