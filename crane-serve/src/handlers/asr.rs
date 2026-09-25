//! ASR (Automatic Speech Recognition) handlers.
//!
//! These handlers bypass the continuous-batching engine and use ASR models
//! (Qwen3-ASR) directly on a dedicated thread. The model transcribes audio to
//! text and returns a JSON transcript to the client.

use std::sync::Arc;

use axum::{
    Json,
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};

use tracing::info;

use crate::openai_api::TranscriptionResponse;
use crate::{AppState, make_error};
use crane_core::device::format_budget;

// ─────────────────────────────────────────────────────────────
//  ASR Request Channel Structure
// ─────────────────────────────────────────────────────────────

/// A transcription request sent from the HTTP handler to the ASR engine thread.
pub struct AsrTranscribeRequest {
    /// Raw bytes of the uploaded audio file (e.g. WAV).
    pub audio_bytes: Vec<u8>,
    /// Optional language hint for the model.
    pub language: Option<String>,
    /// Optional sampling temperature override.
    pub temperature: Option<f64>,
    /// Channel to send back the result.
    pub tx: tokio::sync::oneshot::Sender<Result<String, String>>,
}

// ─────────────────────────────────────────────────────────────
//  Handler: POST /v1/audio/transcriptions
// ─────────────────────────────────────────────────────────────

/// Handles `POST /v1/audio/transcriptions`: parses the multipart audio upload,
/// forwards it to the ASR engine thread, and returns the transcript as JSON.
pub async fn transcriptions(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Response {
    let started = std::time::Instant::now();
    // Validate that ASR is available.
    let Some(asr_tx) = &state.asr_tx else {
        let (status, json) = make_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "ASR model not loaded. Start the server with an ASR model to enable /v1/audio/transcriptions.",
        );
        return (status, json).into_response();
    };

    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut language: Option<String> = None;
    let mut temperature: Option<f64> = None;

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(e) => {
                let (status, json) = make_error(
                    StatusCode::BAD_REQUEST,
                    &format!("Failed to parse multipart request: {e}"),
                );
                return (status, json).into_response();
            },
        };

        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => match field.bytes().await {
                Ok(bytes) => audio_bytes = Some(bytes.to_vec()),
                Err(e) => {
                    let (status, json) = make_error(
                        StatusCode::BAD_REQUEST,
                        &format!("Failed to read 'file' field: {e}"),
                    );
                    return (status, json).into_response();
                },
            },
            "language" => {
                if let Ok(text) = field.text().await
                    && !text.is_empty()
                {
                    language = Some(text);
                }
            },
            "temperature" => {
                if let Ok(text) = field.text().await {
                    match text.parse::<f64>() {
                        Ok(value) => temperature = Some(value),
                        Err(_) => {
                            let (status, json) = make_error(
                                StatusCode::BAD_REQUEST,
                                &format!(
                                    "Invalid 'temperature' value: {text:?} is not a valid number."
                                ),
                            );
                            return (status, json).into_response();
                        },
                    }
                }
            },
            "response_format" => {
                if let Ok(text) = field.text().await
                    && text != "json"
                {
                    let (status, json) = make_error(
                        StatusCode::BAD_REQUEST,
                        &format!(
                            "Unsupported 'response_format': {text:?}. Only 'json' is supported."
                        ),
                    );
                    return (status, json).into_response();
                }
            },
            _ => {
                // Ignore other fields (e.g. "model").
            },
        }
    }

    let Some(audio_bytes) = audio_bytes else {
        let (status, json) = make_error(
            StatusCode::BAD_REQUEST,
            "Missing 'file' field with audio data.",
        );
        return (status, json).into_response();
    };

    info!(
        audio_bytes = %format_budget(audio_bytes.len() as u64),
        language = ?language,
        "ASR request accepted",
    );

    let (tx, rx) = tokio::sync::oneshot::channel();

    let asr_req = AsrTranscribeRequest {
        audio_bytes,
        language,
        temperature,
        tx,
    };

    if asr_tx.send(asr_req).is_err() {
        let (status, json) = make_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "ASR engine thread has stopped.",
        );
        return (status, json).into_response();
    }

    match rx.await {
        Ok(Ok(text)) => {
            info!(
                transcript_len = text.chars().count(),
                total_time = format!("{:.0} ms", started.elapsed().as_secs_f64() * 1000.0),
                "ASR request completed",
            );
            Json(TranscriptionResponse { text }).into_response()
        },
        Ok(Err(err)) => {
            let (status, json) = make_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("Transcription failed: {err}"),
            );
            (status, json).into_response()
        },
        Err(_) => {
            let (status, json) = make_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "ASR engine did not respond.",
            );
            (status, json).into_response()
        },
    }
}
