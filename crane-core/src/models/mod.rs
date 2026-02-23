pub mod modules;
#[cfg(feature = "onnx")]
pub mod moonshine_asr;
pub mod orpheus;
pub mod paddleocr_vl;
pub mod qwen25;
pub mod qwen25_vit;
pub mod qwen3;
pub mod qwen3_tts;
// pub mod qwen3_vl;
pub mod hunyuan_dense;

#[cfg(feature = "onnx")]
pub mod silero_vad;
#[cfg(feature = "onnx")]
pub mod snac_onnx;

pub use candle_core;
pub use candle_core::Tensor;
pub use candle_core::{DType, Device};
