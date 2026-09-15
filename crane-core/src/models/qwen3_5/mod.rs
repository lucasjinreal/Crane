//! Qwen 3.5 dense text-only + multimodal model.
//!
//! Hybrid Mamba/Transformer: every 4th layer is full (softmax) attention,
//! the other 3 are linear attention via [`crate::ops::gdn::GatedDeltaNet`].
//!
//! See [`config::TextConfig`] for the HF schema mapping and
//! [`model::Qwen3_5TextModel`] for the high-level entry point.
//!
//! Multimodal checkpoints (`Qwen3_5ForConditionalGeneration`) carry a vision
//! tower under `model.visual.*` and use the same text weights as the dense
//! class — the [`vision::Qwen3_5VisionModel`] module loads the `ViT`, and the
//! text model recognizes the `model.language_model.*` weight prefix.

mod config;
mod conv3d_temporal_2;
mod model;
mod modeling;
mod prefill;
pub mod processor;
pub mod vision;
pub mod vlm;

pub use crate::models::modules::quant_kv_cache::{KvCache, KvCacheBackend, KvCacheKind};
pub use config::{Config, LayerType, TextConfig, VisionConfig, load_config};
pub use model::{Model, ModelFormat, Qwen3_5TextModel};
pub use modeling::{
    DecoderLayer, FullAttention, MRotaryEmbedding, Mlp, Qwen35RmsNorm, RopeSlice, apply_mrope,
};
pub use prefill::{DEFAULT_CHUNK as DEFAULT_PREFILL_CHUNK, chunk_size as prefill_chunk_size};
pub use processor::{PreprocessorConfig, ProcessedImage, load_preprocessor_config};
pub use vision::Qwen3_5VisionModel;
pub use vlm::{Qwen3_5VLModel, VlGenerationConfig};
