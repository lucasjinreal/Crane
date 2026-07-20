//! Audio functionality for the Crane SDK

pub mod asr;
#[cfg(feature = "onnx")]
mod asr_client;
mod asr_qwen3;
mod pcm;
pub mod tts;
mod tts_client;
mod tts_qwen3;
mod tts_voxtral;

pub use asr::{Asr, AsrStream, TranscribeOptions, Transcript};
#[cfg(feature = "onnx")]
pub use asr_client::AsrClient;
pub use pcm::{AudioInfo, encode_wav, load_wav_f32, pcm_f32_to_i16, save_wav};
pub use tts::{Tts, TtsStream, VoiceInfo};
pub use tts_client::TtsClient;
