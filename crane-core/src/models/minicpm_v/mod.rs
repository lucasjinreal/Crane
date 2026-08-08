//! MiniCPM-V-4.6 vision-language model.
//!
//! Reuses [`crate::models::qwen3_5::Qwen3_5TextModel`] unmodified for the
//! text backbone (a field-for-field-compatible `qwen3_5_text` config); the
//! vision tower (NaViT SigLIP-style ViT + mid-stack window-attention merger
//! + hierarchical downsample merger) is new. See `config.rs` for the full
//! rationale and `vlm.rs` for the top-level entry point.

pub mod config;
pub mod merger;
pub mod preprocess;
pub mod vision;
pub mod vlm;

pub use config::{load_config, Config, VisionConfig};
pub use preprocess::{load_preprocessor_config, PreprocessorConfig};
pub use vlm::{MinicpmV46VLModel, VlGenerationConfig};
