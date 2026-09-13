// SPDX-License-Identifier: MIT
//! Audio8-TTS 0.1B ONNX INT8 model wrapper: `ChatML` prompt construction, the
//! `DualAR` (slow AR semantic tokens + fast AR codec codebooks) generation
//! loop with persistent KV-cache/Mamba-state buffers, lazy
//! Repetition-Avoidance Sampling, and codec decoding.

mod config;
mod model;
mod prompt;
mod sampling;

pub use model::Model;
