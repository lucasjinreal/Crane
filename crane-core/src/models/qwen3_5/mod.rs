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
//! class — the [`vision::Qwen3_5VisionModel`] module loads the ViT, and the
//! text model recognizes the `model.language_model.*` weight prefix.

mod config;
mod conv3d_temporal_2;
mod kv_cache;
mod model;
mod modeling;
mod prefill;
pub mod processor;
pub mod vision;
pub mod vlm;

pub use config::{load_config, Config, LayerType, TextConfig, VisionConfig};
pub use kv_cache::{KvCache, KvCacheBackend, KvCacheKind};
pub use model::{Model, ModelFormat, Qwen3_5TextModel};
pub use modeling::{
    apply_mrope, DecoderLayer, FullAttention, Mlp, MRotaryEmbedding, Qwen35RmsNorm, RopeSlice,
};
pub use prefill::{chunk_size as prefill_chunk_size, DEFAULT_CHUNK as DEFAULT_PREFILL_CHUNK};
pub use processor::{load_preprocessor_config, PreprocessorConfig, ProcessedImage};
pub use vision::Qwen3_5VisionModel;
pub use vlm::{Qwen3_5VLModel, VlGenerationConfig};
