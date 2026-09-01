#[cfg(feature = "onnx")]
pub mod g2p;
#[cfg(feature = "onnx")]
pub mod kokoro_tts;
pub mod modules;
#[cfg(feature = "onnx")]
pub mod moonshine_asr;
pub mod muscriptor;
pub mod orpheus;
#[cfg(feature = "onnx")]
pub mod paddleocr_v6;
pub mod paddleocr_vl;
pub mod qwen25;
pub mod qwen25_vit;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_asr;
pub mod qwen3_tts;
// pub mod qwen3_vl;
pub mod gemma4;
pub mod hunyuan_dense;
pub mod minicpm5;
pub mod minicpm_v;
pub mod minicpmo;
pub mod voxcpm2;
pub mod voxtral_tts;

#[cfg(feature = "onnx")]
pub mod silero_vad;
#[cfg(feature = "onnx")]
pub mod snac_onnx;

pub use candle_core;
pub use candle_core::Tensor;
pub use candle_core::{DType, Device};

pub mod utils;
pub mod with_tracing;
