//! Common handlers: health, stats.

use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::json;

use crate::AppState;

/// `GET /health`
pub async fn health(State(state): State<Arc<AppState>>) -> Response {
    let fatal_error = state
        .engine
        .as_ref()
        .and_then(|e| e.stats.get_fatal_error());
    match fatal_error {
        Some(err) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "unhealthy", "error": err})),
        )
            .into_response(),
        None => Json(json!({"status": "ok"})).into_response(),
    }
}

/// `GET /v1/stats`
pub async fn stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let snap = state
        .engine
        .as_ref()
        .map(|e| e.stats.snapshot())
        .unwrap_or_default();
    Json(snap)
}
