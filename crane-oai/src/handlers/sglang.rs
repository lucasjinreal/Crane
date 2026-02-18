//! SGLang-compatible native API handlers.
//!
//! Endpoints:
//! - `POST /generate`        — native text generation
//! - `GET  /model_info`      — model metadata
//! - `GET  /server_info`     — server configuration + live stats
//! - `GET  /health_generate` — deep health check (1-token probe)
//! - `POST /flush_cache`     — flush model KV caches (no-op informational)
//! - `POST /abort_request`   — abort in-flight request (informational)

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
};
use serde_json::json;

use crate::engine::EngineResponse;
use crate::openai_api::ErrorResponse;
use crate::sglang_api::*;
use crate::{make_error, AppState};

use super::sse;

// ─────────────────────────────────────────────────────────────
//  /generate
// ─────────────────────────────────────────────────────────────

/// `POST /generate` — SGLang-style native generation.
///
/// Accepts either `text` (prompt string) or `input_ids` (pre-tokenized).
/// Returns generated text + meta_info, or SSE stream if `stream: true`.
pub async fn generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // Resolve input tokens.
    let input_ids = if let Some(ids) = req.input_ids {
        ids
    } else if let Some(text) = &req.text {
        state
            .tokenizer
            .encode(text.as_str(), true)
            .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Tokenize failed: {e}")))?
            .get_ids()
            .to_vec()
    } else {
        return Err(make_error(
            StatusCode::BAD_REQUEST,
            "Either 'text' or 'input_ids' must be provided",
        ));
    };

    let sp = &req.sampling_params;
    let request_id = req
        .rid
        .unwrap_or_else(|| format!("gen-{}", uuid::Uuid::new_v4()));

    let response_rx = state
        .engine
        .submit(
            request_id.clone(),
            input_ids,
            sp.max_new_tokens,
            sp.temperature.or(Some(0.8)),
            sp.top_p.or(Some(0.95)),
            sp.top_k.or(Some(20)),
            sp.repetition_penalty,
            state.eos_token_id.clone(),
        )
        .map_err(|e| make_error(StatusCode::SERVICE_UNAVAILABLE, &e.to_string()))?;

    if req.stream {
        let stream = sse::make_generate_sse_stream(request_id, response_rx);
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Collect full response.
        let mut full_text = String::new();
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        let mut finish_reason = "length".to_string();

        let mut response_rx = response_rx;
        while let Some(resp) = response_rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => full_text.push_str(&text),
                EngineResponse::Finished {
                    full_text: ft,
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    finish_reason: fr,
                } => {
                    full_text = ft;
                    prompt_tokens = pt;
                    completion_tokens = ct;
                    finish_reason = fr;
                    break;
                }
                EngineResponse::Error(e) => {
                    return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, &e));
                }
            }
        }

        let response = GenerateResponse {
            text: full_text,
            meta_info: GenerateMetaInfo {
                id: request_id,
                prompt_tokens,
                completion_tokens,
                finish_reason,
            },
        };

        Ok(Json(response).into_response())
    }
}

// ─────────────────────────────────────────────────────────────
//  /model_info
// ─────────────────────────────────────────────────────────────

/// `GET /model_info` — model metadata.
pub async fn model_info(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(ModelInfoResponse {
        model_path: state.model_path.clone(),
        model_type: state.model_type_name.clone(),
        is_generation: true,
        dtype: Some(state.dtype_name.clone()),
        device: Some(state.device_name.clone()),
        max_model_len: None,
    })
}

// ─────────────────────────────────────────────────────────────
//  /server_info
// ─────────────────────────────────────────────────────────────

/// `GET /server_info` — server configuration + live stats.
pub async fn server_info(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let stats = state.engine.stats.snapshot();
    Json(ServerInfoResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
        model_path: state.model_path.clone(),
        model_type: state.model_type_name.clone(),
        host: state.host.clone(),
        port: state.port,
        max_concurrent: state.max_concurrent,
        decode_tokens_per_seq: state.decode_tokens_per_seq,
        stats,
    })
}

// ─────────────────────────────────────────────────────────────
//  /health_generate
// ─────────────────────────────────────────────────────────────

/// `GET /health_generate` — deep health check.
///
/// Runs a tiny 1-token generation through the full pipeline.
/// Returns 200 on success, 503 on failure/timeout.
pub async fn health_generate(
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let probe_tokens = state.eos_token_id.clone(); // minimal input (already a Vec)
    let request_id = format!("health-{}", uuid::Uuid::new_v4());

    let response_rx = state
        .engine
        .submit(
            request_id,
            probe_tokens,
            1, // generate just 1 token
            Some(0.0), // greedy
            None,
            None,
            1.0,
            state.eos_token_id.clone(),
        )
        .map_err(|e| {
            make_error(
                StatusCode::SERVICE_UNAVAILABLE,
                &format!("Health probe failed to submit: {e}"),
            )
        })?;

    // Wait with timeout.
    let mut response_rx = response_rx;
    let result = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        while let Some(resp) = response_rx.recv().await {
            match resp {
                EngineResponse::Token { .. } => continue,
                EngineResponse::Finished { .. } => return Ok(()),
                EngineResponse::Error(e) => return Err(e),
            }
        }
        Err("No response received".to_string())
    })
    .await;

    match result {
        Ok(Ok(())) => Ok(Json(json!({"status": "ok"}))),
        Ok(Err(e)) => Err(make_error(
            StatusCode::SERVICE_UNAVAILABLE,
            &format!("Health probe generation failed: {e}"),
        )),
        Err(_) => Err(make_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Health probe timed out (30s)",
        )),
    }
}

// ─────────────────────────────────────────────────────────────
//  /flush_cache
// ─────────────────────────────────────────────────────────────

/// `POST|GET /flush_cache` — informational endpoint.
///
/// In the current architecture, KV caches are per-sequence and automatically
/// freed on completion. This endpoint is provided for API compatibility.
pub async fn flush_cache() -> impl IntoResponse {
    Json(FlushCacheResponse {
        success: true,
        message: "KV caches are managed per-sequence and auto-freed on completion.".into(),
    })
}

// ─────────────────────────────────────────────────────────────
//  /abort_request
// ─────────────────────────────────────────────────────────────

/// `POST /abort_request` — request cancellation.
///
/// Currently, clients can cancel by dropping their SSE connection (which the
/// engine detects automatically). This endpoint is provided for API compatibility;
/// explicit abort via engine control channel is a future enhancement.
pub async fn abort_request(
    Json(req): Json<AbortRequest>,
) -> impl IntoResponse {
    // TODO: Add explicit abort via engine control channel.
    Json(AbortResponse {
        success: true,
        message: format!(
            "Request '{}' marked for abort. \
             Note: clients can also cancel by closing the SSE connection.",
            req.rid
        ),
    })
}
