//! Request types for VLM (Vision-Language Model) dedicated threads.
//!
//! These types are transport-agnostic -- they carry no HTTP-specific fields
//! and are sent over `tokio::sync::mpsc` channels to a model's dedicated
//! thread, matching the pattern used for TTS (see [`crate::engine::runtime`]).

use std::path::PathBuf;

use anyhow::Result;
use crane_core::models::paddleocr_vl::OcrTask;
use tokio::sync::{mpsc, oneshot};

/// A request to the PaddleOCR-VL dedicated thread.
pub enum VlmRequest {
    /// Non-streaming recognition request.
    Recognize {
        /// Path to the image on disk.
        img_path: PathBuf,
        /// OCR task to perform (plain OCR, table, formula, chart).
        task: OcrTask,
        /// Maximum tokens to generate.
        max_tokens: usize,
        /// Channel to send back the recognized text.
        tx: oneshot::Sender<Result<String>>,
    },
    /// Streaming recognition request.
    RecognizeStream {
        /// Path to the image on disk.
        img_path: PathBuf,
        /// OCR task to perform (plain OCR, table, formula, chart).
        task: OcrTask,
        /// Maximum tokens to generate.
        max_tokens: usize,
        /// Channel to stream recognized text tokens as they're generated.
        token_tx: mpsc::UnboundedSender<String>,
        /// Channel to signal completion or error.
        done_tx: oneshot::Sender<Result<()>>,
    },
}

/// A request to the Gemma4 VLM dedicated thread.
pub struct Gemma4VlmRequest {
    /// Path to the image on disk.
    pub img_path: PathBuf,
    /// Text prompt accompanying the image.
    pub text_prompt: String,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Channel to send back the generated text.
    pub tx: oneshot::Sender<Result<String>>,
}
