//! Qwen 3.5 dense text-only model.
//!
//! Hybrid Mamba/Transformer: every 4th layer is full (softmax) attention,
//! the other 3 are linear attention via [`crate::gdn::GatedDeltaNet`].
//!
//! See [`config::TextConfig`] for the HF schema mapping and
//! [`model::Qwen3_5TextModel`] for the high-level entry point.
//!
//! Only dense text-only checkpoints are supported; the
//! `Qwen3_5ForConditionalGeneration` multimodal class is recognized for
//! weight loading (the `model.language_model.*` prefix is handled), but
//! vision weights are ignored.

mod config;
mod kv_cache;
mod model;
mod modeling;

pub use config::{load_config, Config, LayerType, TextConfig};
pub use kv_cache::KvCache;
pub use model::{Model, ModelFormat, Qwen3_5TextModel};
pub use modeling::{apply_mrope, DecoderLayer, FullAttention, Mlp, MRotaryEmbedding, Qwen35RmsNorm};