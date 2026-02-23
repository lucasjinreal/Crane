//! TTS (Text-to-Speech) handlers for Qwen3-TTS.
//!
//! These handlers bypass the continuous-batching engine and use Qwen3-TTS directly
//! on a dedicated thread. The model generates speech from text input and returns
//! audio bytes (WAV or raw PCM) to the client.

use std::sync::Arc;

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};

use crate::openai_api::*;
use crate::{make_error, AppState};

// ─────────────────────────────────────────────────────────────
//  TTS Request Channel Structure
// ─────────────────────────────────────────────────────────────

pub struct TtsGenerateRequest {
    pub input: String,
    pub voice: Option<String>,
    pub language: String,
    pub instructions: Option<String>,
    pub response_format: AudioResponseFormat,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub repetition_penalty: f32,
    pub max_tokens: usize,
    /// Reference audio path for voice cloning (Base model only).
    pub reference_audio: Option<String>,
    /// Transcript of the reference audio.
    pub reference_text: Option<String>,
    /// Channel to send back the result.
    pub tx: tokio::sync::oneshot::Sender<Result<TtsResult, String>>,
}

pub struct TtsResult {
    pub audio_bytes: Vec<u8>,
    pub content_type: &'static str,
    pub file_name: String,
    pub sample_rate: u32,
}

// ─────────────────────────────────────────────────────────────
//  Handler: POST /v1/audio/speech
// ─────────────────────────────────────────────────────────────

pub async fn speech(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    // Validate that TTS is available.
    let tts_tx = match &state.tts_tx {
        Some(tx) => tx,
        None => {
            let (status, json) = make_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "TTS model not loaded. Start the server with a Qwen3-TTS model to enable /v1/audio/speech.",
            );
            return (status, json).into_response();
        }
    };

    if req.input.trim().is_empty() {
        let (status, json) = make_error(StatusCode::BAD_REQUEST, "Input text cannot be empty.");
        return (status, json).into_response();
    }

    match req.response_format {
        AudioResponseFormat::Wav | AudioResponseFormat::Pcm => {}
        _ => {
            let (status, json) = make_error(
                StatusCode::BAD_REQUEST,
                "Unsupported response_format. Currently supported: wav, pcm.",
            );
            return (status, json).into_response();
        }
    }

    let temperature = req.temperature.unwrap_or(0.9);
    let repetition_penalty = req.repetition_penalty.unwrap_or(1.05);
    let language = req.language.clone().unwrap_or_else(|| "auto".to_string());

    let (tx, rx) = tokio::sync::oneshot::channel();

    let tts_req = TtsGenerateRequest {
        input: req.input,
        voice: req.voice,
        language,
        instructions: req.instructions,
        response_format: req.response_format,
        temperature,
        top_p: req.top_p,
        repetition_penalty,
        max_tokens: req.max_tokens,
        reference_audio: req.reference_audio,
        reference_text: req.reference_text,
        tx,
    };

    if tts_tx.send(tts_req).is_err() {
        let (status, json) = make_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "TTS engine thread has stopped.",
        );
        return (status, json).into_response();
    }

    // Wait for TTS result
    match rx.await {
        Ok(Ok(result)) => {
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, result.content_type)
                .header(
                    header::CONTENT_DISPOSITION,
                    format!("attachment; filename=\"{}\"", result.file_name),
                )
                .body(axum::body::Body::from(result.audio_bytes))
                .unwrap_or_else(|_| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Failed to build response",
                    )
                        .into_response()
                })
        }
        Ok(Err(err)) => {
            let (status, json) =
                make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("TTS generation failed: {err}"));
            (status, json).into_response()
        }
        Err(_) => {
            let (status, json) = make_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "TTS engine did not respond.",
            );
            (status, json).into_response()
        }
    }
}
