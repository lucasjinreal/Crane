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
//! | [`models`] | Transformer model implementations (Qwen3, HunyuanDense, Qwen2.5, multimodal) |
//! | [`generation`] | Token generation utilities (sampling, stopping criteria, logit processors) |
//! | [`autotokenizer`] | HuggingFace-compatible tokenizer loader |
//! | [`chat`] | Chat-template rendering and message formatting |
//! | [`bins`] | Model loading helpers used by CLI entry-points |
//! | [`utils`] | Misc tensor / device utilities |
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

pub mod generation;
pub mod models;
pub mod ops;
pub mod utils;

#[cfg(feature = "onnx")]
#[path = "onnx.rs"]
pub mod onnx;

pub mod autotokenizer;
pub mod bins;
pub mod chat;

#[doc(hidden)]
pub mod test_data;
