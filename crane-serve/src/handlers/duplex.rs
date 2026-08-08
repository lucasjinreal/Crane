//! Full-duplex live audio chat over WebSocket (`/v1/audio/duplex`) —
//! MiniCPM-o-4.5 only so far (`ModelType::MiniCpmODuplex`). Bypasses the
//! continuous-batching engine and the other handlers' one-shot
//! request/response pattern: a `crane_core::models::minicpmo::DuplexSession`
//! is a genuinely **stateful** conversation, so this handler owns a
//! WebSocket connection for the conversation's whole lifetime instead of a
//! single request.
//!
//! Wire protocol (deliberately simple — JSON + base64 rather than a custom
//! binary framing, so both this server and the demo client can use their
//! ordinary JSON tooling):
//! - **Inbound**: first message is JSON `{"system_prompt": "..."}` (prompt
//!   optional) to start the session; every message after that is a
//!   `Binary` WebSocket frame — one audio chunk, 16-bit little-endian PCM,
//!   16kHz mono (the sample rate `crane_core`'s Whisper-style mel frontend
//!   expects), one `streaming_prefill`/`streaming_generate` cycle per
//!   frame (matching upstream's own "called once per second" chunking, but
//!   the client controls the actual chunk size/cadence).
//! - **Outbound**: one JSON `DuplexChunkEvent` per inbound chunk
//!   (`is_listen`/`text`/`end_of_turn`/`audio_base64`), plus an initial
//!   `{"ready":true}` once `Prepare` succeeds.
//!
//! Only **one** live session is supported at a time — a real session needs
//! all six MiniCPM-o towers resident simultaneously (~18-19GB), so this
//! server has no way to run two concurrently on typical hardware; a second
//! connection while one is active is rejected outright (`state.duplex_lock`)
//! rather than silently corrupting both conversations' state.

use std::sync::Arc;

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use crate::{make_error, AppState};

/// One request to the duplex engine thread — mirrors the
/// request-struct-with-a-reply-channel pattern used by every other model
/// kind in this crate (`TtsGenerateRequest`, `AsrTranscribeRequest`), just
/// with two request shapes instead of one since a duplex session has two
/// distinct operations (start a conversation, feed one audio chunk) rather
/// than a single one-shot call.
pub enum DuplexRequest {
    /// (Re)start a fresh conversation: resets the session and prefills a
    /// system prompt.
    Prepare {
        system_prompt: Option<String>,
        tx: tokio::sync::oneshot::Sender<Result<(), String>>,
    },
    /// One already-decoded audio chunk (16kHz mono `[-1, 1]` f32 PCM) —
    /// runs one `streaming_prefill` + `streaming_generate` cycle.
    Chunk {
        samples: Vec<f32>,
        tx: tokio::sync::oneshot::Sender<Result<DuplexChunkEvent, String>>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct DuplexChunkEvent {
    pub is_listen: bool,
    pub text: String,
    pub end_of_turn: bool,
    /// 16-bit little-endian PCM, base64-encoded — present only when a turn
    /// just completed and produced audio (mirrors `ChunkOutput::audio_waveform`).
    pub audio_base64: Option<String>,
    /// Always 24000 when `audio_base64` is present (`Token2Wav::sample_rate()`
    /// is fixed) — included so clients don't need to hardcode it.
    pub audio_sample_rate: Option<u32>,
}

#[derive(Debug, Deserialize, Default)]
struct PrepareMessage {
    system_prompt: Option<String>,
}

#[derive(Debug, Serialize)]
struct WsErrorMessage {
    error: String,
}

fn error_message(msg: &str) -> Message {
    let json = serde_json::to_string(&WsErrorMessage { error: msg.to_string() }).unwrap_or_else(|_| "{\"error\":\"internal error\"}".to_string());
    Message::Text(json.into())
}

/// `GET /v1/audio/duplex` — upgrades to a WebSocket, or fails immediately
/// (before upgrading) if no duplex model is loaded or a session is already
/// active.
pub async fn duplex_ws(State(state): State<Arc<AppState>>, ws: WebSocketUpgrade) -> Response {
    let Some(duplex_tx) = state.duplex_tx.clone() else {
        let (status, json) = make_error(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "MiniCPM-o duplex model not loaded. Start the server with a MiniCPM-o-4.5 checkpoint to enable /v1/audio/duplex.",
        );
        return (status, json).into_response();
    };

    // Exclusivity: at most one live session (see the module doc). Held for
    // the connection's whole lifetime and released automatically on drop
    // (connection close, in any way — clean disconnect, error, or panic).
    let Ok(permit) = state.duplex_lock.clone().try_lock_owned() else {
        let (status, json) = make_error(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "A MiniCPM-o duplex session is already active. Only one live session is supported at a time.",
        );
        return (status, json).into_response();
    };

    ws.on_upgrade(move |socket| handle_duplex_socket(socket, duplex_tx, permit))
}

async fn handle_duplex_socket(
    mut socket: WebSocket,
    duplex_tx: tokio::sync::mpsc::UnboundedSender<DuplexRequest>,
    _permit: tokio::sync::OwnedMutexGuard<()>,
) {
    // First message: optional JSON {"system_prompt": "..."} to prepare the
    // session. Any non-text/unparseable first message just prepares with
    // the default system prompt instead of erroring — keeps trivial
    // clients (that only ever send audio) working with no boilerplate.
    let system_prompt = match socket.recv().await {
        Some(Ok(Message::Text(text))) => serde_json::from_str::<PrepareMessage>(&text).unwrap_or_default().system_prompt,
        Some(Ok(Message::Close(_))) | None => return,
        _ => None,
    };

    let (tx, rx) = tokio::sync::oneshot::channel();
    if duplex_tx.send(DuplexRequest::Prepare { system_prompt, tx }).is_err() {
        let _ = socket.send(error_message("duplex engine thread has stopped")).await;
        return;
    }
    match rx.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            let _ = socket.send(error_message(&format!("failed to prepare duplex session: {e}"))).await;
            return;
        }
        Err(_) => {
            let _ = socket.send(error_message("duplex engine did not respond to Prepare")).await;
            return;
        }
    }
    if socket.send(Message::Text("{\"ready\":true}".into())).await.is_err() {
        return;
    }

    while let Some(msg) = socket.recv().await {
        let Ok(msg) = msg else { break };
        match msg {
            Message::Binary(bytes) => {
                // 16-bit little-endian PCM -> f32 [-1, 1], matching this
                // codebase's established normalization convention
                // (2^(bits-1), not a raw i16 cast) elsewhere for audio input.
                let samples: Vec<f32> = bytes.chunks_exact(2).map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0).collect();

                let (tx, rx) = tokio::sync::oneshot::channel();
                if duplex_tx.send(DuplexRequest::Chunk { samples, tx }).is_err() {
                    let _ = socket.send(error_message("duplex engine thread has stopped")).await;
                    break;
                }
                match rx.await {
                    Ok(Ok(event)) => {
                        let json = serde_json::to_string(&event).unwrap_or_else(|_| "{\"error\":\"failed to encode event\"}".to_string());
                        if socket.send(Message::Text(json.into())).await.is_err() {
                            break;
                        }
                    }
                    Ok(Err(e)) => {
                        if socket.send(error_message(&e)).await.is_err() {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            Message::Close(_) => break,
            // Ping/Pong are handled by axum's WebSocket implementation
            // automatically; Text messages after the initial Prepare are
            // ignored (no other client->server control messages defined yet).
            _ => {}
        }
    }
}
