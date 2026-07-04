//! TTS (Text-to-Speech) handlers.
//!
//! These handlers bypass the continuous-batching engine and use TTS models
//! (Qwen3-TTS, Voxtral TTS) directly on a dedicated thread. The model generates
//! speech from text input and returns audio bytes (WAV or raw PCM) to the client.

use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};

use crane::engine::TtsGenerateRequest;
use crane_core::generation::SpeechOptions;

use crate::openai_api::*;
use crate::{encode_tts_audio, make_error, AppState};

// ─────────────────────────────────────────────────────────────
//  HTTP-encoded TTS result
// ─────────────────────────────────────────────────────────────

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
    // Both return Some/None together -- register_tts sets them atomically.
    let (tts_name, tts_handle) = match state
        .runtime
        .default_tts_name()
        .zip(state.runtime.default_tts_handle())
    {
        Some(pair) => pair,
        None => {
            let (status, json) = make_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "TTS model not loaded. Start the server with a TTS model to enable /v1/audio/speech.",
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

    let opts = SpeechOptions {
        max_new_tokens: req.max_tokens,
        temperature: req.temperature.unwrap_or(0.9),
        top_p: req.top_p,
        repetition_penalty: req.repetition_penalty.unwrap_or(1.05),
    };
    let language = req.language.clone().unwrap_or_else(|| "auto".to_string());
    let sample_rate = tts_handle.audio_info().sample_rate;

    let (tx, rx) = tokio::sync::oneshot::channel();

    let tts_req = TtsGenerateRequest {
        text: req.input,
        language,
        voice: req.voice,
        opts,
        reference_audio: req.reference_audio.map(PathBuf::from),
        reference_text: req.reference_text,
        response_tx: tx,
    };

    if let Err(e) = state.runtime.generate_speech(tts_name, tts_req) {
        let (status, json) =
            make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("TTS engine unavailable: {e}"));
        return (status, json).into_response();
    }

    // Wait for TTS result
    match rx.await {
        Ok(Ok(audio_tensor)) => match encode_tts_audio(&audio_tensor, sample_rate, &req.response_format) {
            Ok(result) => Response::builder()
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
                }),
            Err(err) => {
                let (status, json) =
                    make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("Audio encoding failed: {err}"));
                (status, json).into_response()
            }
        },
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
