//! Audio functionality for the Crane SDK

#[cfg(feature = "onnx")]
mod asr;
mod tts;

#[cfg(feature = "onnx")]
pub use asr::AsrClient;
pub use tts::TtsClient;
