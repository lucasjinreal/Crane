//! Audio functionality for the Crane SDK

#[cfg(feature = "onnx")]
mod asr;
pub mod tts;
mod tts_client;
mod tts_qwen3;

#[cfg(feature = "onnx")]
pub use asr::AsrClient;
pub use tts::{AudioInfo, Tts, TtsStream, VoiceInfo, pcm_f32_to_i16};
pub use tts_client::TtsClient;
