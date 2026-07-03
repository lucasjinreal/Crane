//! # Crane AI SDK
//!
//! A comprehensive SDK for building AI applications with various capabilities
//! including chat, vision, audio, and multimodal processing.

pub mod audio;
pub mod chat;
pub mod common;
pub mod engine;
pub mod llm;
pub mod multimodal;
pub mod utils;
pub mod vision;

pub mod prelude {
    //! Common imports for the Crane SDK

    pub use crate::audio::TtsClient;
    pub use crate::chat::{ChatClient, ChatConfig, ChatMessage, ChatRole};
    pub use crate::common::{CraneError, CraneResult};
    pub use crate::vision::{OcrClient, VisionClient};

    #[cfg(feature = "onnx")]
    pub use crate::audio::AsrClient;
}
