//! VLM (Vision-Language Model) handlers for PaddleOCR-VL.
//!
//! These handlers bypass the text-only engine and use PaddleOcrVL directly
//! for image+text inference. The model processes images and generates OCR
//! results as streaming or non-streaming text.

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
};

use crane_core::models::paddleocr_vl::OcrTask;

use crate::openai_api::*;
use crate::sglang_api::*;
use crate::{make_error, now_epoch, AppState};

// ─────────────────────────────────────────────────────────────
//  VLM Request Channel Structure
// ─────────────────────────────────────────────────────────────

pub enum VlmRequest {
    /// Non-streaming request
    Recognize {
        img_path: std::path::PathBuf,
        task: OcrTask,
        max_tokens: usize,
        tx: tokio::sync::oneshot::Sender<Result<String, String>>,
    },
    /// Streaming request
    RecognizeStream {
        img_path: std::path::PathBuf,
        task: OcrTask,
        max_tokens: usize,
        token_tx: tokio::sync::mpsc::UnboundedSender<String>,
        done_tx: tokio::sync::oneshot::Sender<Result<(), String>>,
    },
}

// ─────────────────────────────────────────────────────────────
//  Image downloading
// ─────────────────────────────────────────────────────────────

/// Download an image from a URL to a temporary file.
/// Returns the path to the temp file (the file persists until the TempDir is dropped).
async fn download_image(url: &str) -> Result<(tempfile::TempDir, std::path::PathBuf), String> {
    let dir = tempfile::TempDir::new()
        .map_err(|e| format!("Failed to create temp dir: {e}"))?;

    let resp = reqwest::get(url)
        .await
        .map_err(|e| format!("Failed to download image from '{}': {e}", url))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Image download failed (HTTP {}): {}",
            resp.status(),
            url
        ));
    }

    // Determine extension from content-type or URL.
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "".to_string());

    let bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("Failed to read image bytes: {e}"))?;

    let ext = if content_type.contains("image/png") {
        "png"
    } else if content_type.contains("image/webp") {
        "webp"
    } else if content_type.contains("image/jpeg") || content_type.contains("image/jpg") {
        "jpg"
    } else {
        // Fallback to URL extension
        let url_lower = url.to_lowercase();
        if url_lower.contains(".png") {
            "png"
        } else if url_lower.contains(".webp") {
            "webp"
        } else {
            "jpg" // safe default
        }
    };

    let img_path = dir.path().join(format!("image.{ext}"));

    std::fs::write(&img_path, &bytes)
        .map_err(|e| format!("Failed to write image to temp file: {e}"))?;

    Ok((dir, img_path))
}

/// Determine the OCR task from the text prompt.
fn detect_ocr_task(text: &str) -> OcrTask {
    let text_lower = text.to_lowercase();
    if text_lower.contains("table") {
        OcrTask::Table
    } else if text_lower.contains("formula") {
        OcrTask::Formula
    } else if text_lower.contains("chart") {
        OcrTask::Chart
    } else {
        OcrTask::Ocr
    }
}

// ─────────────────────────────────────────────────────────────
//  Chat Completions (VLM)
// ─────────────────────────────────────────────────────────────

/// VLM-aware chat completions handler.
///
/// Extracts image URLs and text from multimodal messages, downloads
/// images, and runs PaddleOCR-VL inference.
pub async fn vlm_chat_completions(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let vlm_tx = state.vlm_tx.as_ref().ok_or_else(|| {
        make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM model not loaded")
    })?;

    // Extract image URLs and text from messages.
    let mut image_urls = Vec::new();
    let mut text_prompt = String::new();

    for msg in &req.messages {
        if msg.role == "user" {
            let urls = msg.image_urls();
            image_urls.extend(urls);
            let text = msg.text_content();
            if !text.is_empty() {
                text_prompt = text;
            }
        }
    }

    if image_urls.is_empty() {
        return Err(make_error(
            StatusCode::BAD_REQUEST,
            "No image_url found in messages. PaddleOCR-VL requires at least one image.",
        ));
    }

    // Use the first image URL.
    let image_url = &image_urls[0];

    // Download image.
    let (temp_dir, img_path) = download_image(image_url)
        .await
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &e))?;

    let task = detect_ocr_task(&text_prompt);
    let max_tokens = req.max_tokens;
    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    if req.stream {
        // Streaming mode
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

        if vlm_tx.send(VlmRequest::RecognizeStream {
            img_path,
            task,
            max_tokens,
            token_tx: tx,
            done_tx,
        }).is_err() {
            return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM engine thread crashed"));
        }

        let model_name = state.model_name.clone();
        let created = now_epoch();

        let stream = async_stream::stream! {
            // Role announcement chunk.
            let first_chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".into(),
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: Some("assistant".into()),
                        content: None,
                    },
                    finish_reason: None,
                }],
                usage: None,
            };
            yield Ok::<_, std::convert::Infallible>(Event::default().json_data(&first_chunk).unwrap());

            let mut completion_tokens = 0usize;

            // Stream tokens.
            while let Some(text) = rx.recv().await {
                completion_tokens += 1;
                let chunk = ChatCompletionChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".into(),
                    created,
                    model: model_name.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: None,
                            content: Some(text),
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                };
                yield Ok(Event::default().json_data(&chunk).unwrap());
            }

            // Finish chunk.
            let finish_chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".into(),
                created,
                model: model_name.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: ChunkDelta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".into()),
                }],
                usage: None,
            };
            yield Ok(Event::default().json_data(&finish_chunk).unwrap());
            yield Ok(Event::default().data("[DONE]"));
        };

        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        // Non-streaming mode
        let (tx, rx) = tokio::sync::oneshot::channel();
        if vlm_tx.send(VlmRequest::Recognize {
            img_path,
            task,
            max_tokens,
            tx,
        }).is_err() {
            return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM engine thread crashed"));
        }

        let result = rx.await
            .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("VLM task dropped: {e}")))?
            .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("VLM inference failed: {e}")))?;

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".into(),
            created: now_epoch(),
            model: state.model_name.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: ChatMessageContent::Text(result),
                },
                finish_reason: Some("stop".into()),
            }],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        };
        Ok(Json(response).into_response())
    }
}

// ─────────────────────────────────────────────────────────────
//  /generate (VLM)
// ─────────────────────────────────────────────────────────────

/// VLM-aware generate handler for SGLang-style `/generate`.
pub async fn vlm_generate(
    state: Arc<AppState>,
    req: GenerateRequest,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let vlm_tx = state.vlm_tx.as_ref().ok_or_else(|| {
        make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM model not loaded")
    })?;

    let image_url = req.image_url.as_deref().ok_or_else(|| {
        make_error(
            StatusCode::BAD_REQUEST,
            "PaddleOCR-VL requires 'image_url' in the generate request",
        )
    })?;

    // Download image.
    let (temp_dir, img_path) = download_image(image_url)
        .await
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &e))?;

    let text_prompt = req.text.as_deref().unwrap_or("OCR:");
    let task = detect_ocr_task(text_prompt);
    let max_tokens = req.sampling_params.max_new_tokens;
    let request_id = req
        .rid
        .unwrap_or_else(|| format!("gen-{}", uuid::Uuid::new_v4()));

    if req.stream {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();

        if vlm_tx.send(VlmRequest::RecognizeStream {
            img_path,
            task,
            max_tokens,
            token_tx: tx,
            done_tx,
        }).is_err() {
            return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM engine thread crashed"));
        }

        let rid = request_id.clone();
        let stream = async_stream::stream! {
            while let Some(text) = rx.recv().await {
                let chunk = GenerateStreamChunk {
                    text,
                    meta_info: None,
                };
                yield Ok::<_, std::convert::Infallible>(Event::default().json_data(&chunk).unwrap());
            }

            // Final chunk with meta.
            let final_chunk = GenerateStreamChunk {
                text: String::new(),
                meta_info: Some(GenerateMetaInfo {
                    id: rid,
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    finish_reason: "stop".into(),
                }),
            };
            yield Ok(Event::default().json_data(&final_chunk).unwrap());
            yield Ok(Event::default().data("[DONE]"));
        };

        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let (tx, rx) = tokio::sync::oneshot::channel();
        if vlm_tx.send(VlmRequest::Recognize {
            img_path,
            task,
            max_tokens,
            tx,
        }).is_err() {
            return Err(make_error(StatusCode::INTERNAL_SERVER_ERROR, "VLM engine thread crashed"));
        }

        let result = rx.await
            .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("VLM task dropped: {e}")))?
            .map_err(|e| make_error(StatusCode::INTERNAL_SERVER_ERROR, &format!("VLM inference failed: {e}")))?;

        let response = GenerateResponse {
            text: result,
            meta_info: GenerateMetaInfo {
                id: request_id,
                prompt_tokens: 0,
                completion_tokens: 0,
                finish_reason: "stop".into(),
            },
        };

        Ok(Json(response).into_response())
    }
}
