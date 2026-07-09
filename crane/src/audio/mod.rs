//! Audio functionality for the Crane SDK

#[cfg(feature = "onnx")]
mod asr_client;
pub mod tts;
mod tts_client;
mod tts_qwen3;
mod tts_voxtral;

#[cfg(feature = "onnx")]
pub use asr_client::AsrClient;
pub use tts::{AudioInfo, Tts, TtsStream, VoiceInfo, encode_wav, load_wav_f32, pcm_f32_to_i16, save_wav};
pub use tts_client::TtsClient;
