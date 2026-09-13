//! TTS (Text-to-Speech) handlers.
//!
//! These handlers bypass the continuous-batching engine and use TTS models
//! (Qwen3-TTS, Voxtral TTS) directly on a dedicated thread. The model generates
//! speech from text input and returns audio bytes (WAV or raw PCM) to the client.

use std::io::Write;
use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};

use crate::openai_api::*;
use crate::{AppState, make_error};

// ─────────────────────────────────────────────────────────────
//  TTS Request Channel Structure
// ─────────────────────────────────────────────────────────────

pub struct TtsGenerateRequest {
    pub input: String,
    pub voice: Option<String>,
    pub language: String,
    pub instructions: Option<String>,
    pub response_format: AudioResponseFormat,
    /// `None` lets the model apply its own default.
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub repetition_penalty: f32,
    pub max_tokens: usize,
    /// VoxCPM2 only: flow-matching sampler steps per frame (`None` = model default).
    pub cfm_steps: Option<usize>,
    /// VoxCPM2 only: classifier-free guidance strength (`None` = model default).
    pub cfg_scale: Option<f64>,
    /// Reference audio path for voice cloning (Base model only).
    pub reference_audio: Option<String>,
    /// Transcript of the reference audio.
    pub reference_text: Option<String>,
    /// How the engine thread returns audio. Always `Some` on the wire; the
    /// engine `take()`s it so the rest of the struct stays borrowable.
    pub responder: Option<TtsResponder>,
}

/// Where the TTS engine thread sends generated audio for one request.
pub enum TtsResponder {
    /// Non-streaming: the whole encoded clip (WAV or PCM) in one message.
    Whole(tokio::sync::oneshot::Sender<Result<TtsResult, String>>),
    /// Streaming: `meta` fires once (sample rate on success, or a setup
    /// error) so the HTTP handler can pick a status code and headers before
    /// committing to a 200 body; `chunks` then carries raw PCM16-LE frames
    /// until the sender is dropped. A `chunks` send failure means the client
    /// disconnected and generation should stop.
    Stream {
        meta: tokio::sync::oneshot::Sender<Result<u32, String>>,
        chunks: tokio::sync::mpsc::UnboundedSender<Result<Vec<u8>, String>>,
    },
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
                "TTS model not loaded. Start the server with a TTS model to enable /v1/audio/speech.",
            );
            return (status, json).into_response();
        },
    };

    if req.input.trim().is_empty() {
        let (status, json) = make_error(StatusCode::BAD_REQUEST, "Input text cannot be empty.");
        return (status, json).into_response();
    }

    match req.response_format {
        AudioResponseFormat::Wav | AudioResponseFormat::Pcm => {},
        _ => {
            let (status, json) = make_error(
                StatusCode::BAD_REQUEST,
                "Unsupported response_format. Currently supported: wav, pcm.",
            );
            return (status, json).into_response();
        },
    }

    let repetition_penalty = req.repetition_penalty.unwrap_or(1.05);
    let language = req.language.clone().unwrap_or_else(|| "auto".to_string());

    // Browser clients cannot expose a server-local file path. Accept a
    // base64 data URI for reference audio and keep the temporary WAV alive
    // until the TTS worker has consumed it.
    let (reference_audio, reference_audio_temp) = match req.reference_audio.as_deref() {
        Some(data_uri) if data_uri.starts_with("data:") => {
            let Some((_, payload)) = data_uri.split_once(',') else {
                let (status, json) = make_error(
                    StatusCode::BAD_REQUEST,
                    "Malformed reference_audio data URI",
                );
                return (status, json).into_response();
            };
            use base64::Engine;
            let bytes = match base64::engine::general_purpose::STANDARD.decode(payload) {
                Ok(bytes) => bytes,
                Err(_) => {
                    let (status, json) =
                        make_error(StatusCode::BAD_REQUEST, "Invalid base64 reference audio");
                    return (status, json).into_response();
                },
            };
            let mut file = match tempfile::NamedTempFile::with_suffix(".wav") {
                Ok(file) => file,
                Err(_) => {
                    let (status, json) = make_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "Failed to create reference audio file",
                    );
                    return (status, json).into_response();
                },
            };
            if file.write_all(&bytes).is_err() {
                let (status, json) = make_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to save reference audio",
                );
                return (status, json).into_response();
            }
            (Some(file.path().to_string_lossy().to_string()), Some(file))
        },
        Some(path) => (Some(path.to_string()), None),
        None => (None, None),
    };

    if req.stream {
        if !matches!(req.response_format, AudioResponseFormat::Pcm) {
            let (status, json) = make_error(
                StatusCode::BAD_REQUEST,
                "Streaming requires response_format=pcm.",
            );
            return (status, json).into_response();
        }
        if reference_audio.is_some() {
            let (status, json) = make_error(
                StatusCode::BAD_REQUEST,
                "Streaming is not supported with voice cloning.",
            );
            return (status, json).into_response();
        }
        drop(reference_audio_temp); // never populated on this path

        let (meta_tx, meta_rx) = tokio::sync::oneshot::channel::<Result<u32, String>>();
        let (chunk_tx, mut chunk_rx) =
            tokio::sync::mpsc::unbounded_channel::<Result<Vec<u8>, String>>();

        let tts_req = TtsGenerateRequest {
            input: req.input,
            voice: req.voice,
            language,
            instructions: req.instructions,
            response_format: req.response_format,
            temperature: req.temperature,
            top_p: req.top_p,
            repetition_penalty,
            max_tokens: req.max_tokens,
            cfm_steps: req.cfm_steps,
            cfg_scale: req.cfg_scale,
            reference_audio: None,
            reference_text: None,
            responder: Some(TtsResponder::Stream {
                meta: meta_tx,
                chunks: chunk_tx,
            }),
        };

        if tts_tx.send(tts_req).is_err() {
            let (status, json) = make_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "TTS engine thread has stopped.",
            );
            return (status, json).into_response();
        }

        let sample_rate = match meta_rx.await {
            Ok(Ok(sr)) => sr,
            Ok(Err(err)) => {
                let (status, json) = make_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("TTS generation failed: {err}"),
                );
                return (status, json).into_response();
            },
            Err(_) => {
                let (status, json) = make_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "TTS engine did not respond.",
                );
                return (status, json).into_response();
            },
        };

        let body_stream = async_stream::stream! {
            while let Some(item) = chunk_rx.recv().await {
                match item {
                    Ok(bytes) => yield Ok::<Vec<u8>, std::io::Error>(bytes),
                    Err(err) => {
                        tracing::error!("TTS stream aborted mid-generation: {err}");
                        break;
                    },
                }
            }
        };

        return Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "audio/pcm")
            .header("x-sample-rate", sample_rate.to_string())
            .header(
                header::CONTENT_DISPOSITION,
                "attachment; filename=\"speech.pcm\"",
            )
            .body(axum::body::Body::from_stream(body_stream))
            .unwrap_or_else(|_| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to build response",
                )
                    .into_response()
            });
    }

    let (tx, rx) = tokio::sync::oneshot::channel();

    let tts_req = TtsGenerateRequest {
        input: req.input,
        voice: req.voice,
        language,
        instructions: req.instructions,
        response_format: req.response_format,
        temperature: req.temperature,
        top_p: req.top_p,
        repetition_penalty,
        max_tokens: req.max_tokens,
        cfm_steps: req.cfm_steps,
        cfg_scale: req.cfg_scale,
        reference_audio,
        reference_text: req.reference_text,
        responder: Some(TtsResponder::Whole(tx)),
    };

    if tts_tx.send(tts_req).is_err() {
        let (status, json) = make_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "TTS engine thread has stopped.",
        );
        return (status, json).into_response();
    }

    // Wait for TTS result. Keep the temporary file in scope across this await.
    let outcome = rx.await;
    drop(reference_audio_temp);
    match outcome {
        Ok(Ok(result)) => Response::builder()
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
        Ok(Err(err)) => {
            let (status, json) = make_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("TTS generation failed: {err}"),
            );
            (status, json).into_response()
        },
        Err(_) => {
            let (status, json) = make_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "TTS engine did not respond.",
            );
            (status, json).into_response()
        },
    }
}
