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
//! | [`fused_ops`] | Custom CUDA kernels (silu-mul, argmax, top-k, HtoD/DtoD copies) |
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
//! |---|---|
//! | `cuda` | Enable CUDA device and custom PTX kernels (requires CUDA toolkit) |
//! | `accelerate` | Link against Apple Accelerate for CPU BLAS |
//! | `mkl` | Link against Intel MKL for CPU BLAS |

pub mod fused_ops;
pub mod generation;
pub mod models;
pub mod utils;

pub mod autotokenizer;
pub mod bins;
pub mod chat;
