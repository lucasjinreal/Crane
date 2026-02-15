mod engine;
mod openai_api;

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
    routing::{get, post},
    Router,
};
use clap::Parser;
use futures::stream::Stream;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::info;

use crane_core::models::hunyuan_dense::Model;

use engine::{EngineHandle, EngineResponse, InferenceEngine, StatsSnapshot};
use openai_api::*;

// ── CLI ──

#[derive(Parser, Debug)]
#[command(
    name = "crane-oai",
    about = "OpenAI-compatible API server with continuous batching"
)]
struct Args {
    /// Path to model directory
    #[arg(long, default_value = "model/Hunyuan-0.5B-Instruct")]
    model_path: String,

    /// Model name to report in API responses
    #[arg(long, default_value = "hunyuan")]
    model_name: String,

    /// Host to bind
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Use CPU even if CUDA is available
    #[arg(long)]
    cpu: bool,

    /// Max concurrent sequences in decode phase
    #[arg(long, default_value_t = 8)]
    max_concurrent: usize,

    /// Tokens to decode per sequence before switching (higher = fewer KV swaps)
    #[arg(long, default_value_t = 1)]
    decode_tokens_per_seq: usize,
}

// ── App state ──

struct AppState {
    engine: EngineHandle,
    model_name: String,
    /// Shared tokenizer for request pre-processing (encode only — no model state needed).
    tokenizer: tokenizers::Tokenizer,
}

// ── Chat template ──

fn apply_chat_template(messages: &[ChatMessage]) -> String {
    const BOS: &str = "<\u{ff5c}hy_begin\u{2581}of\u{2581}sentence\u{ff5c}>";
    const USER: &str = "<\u{ff5c}hy_User\u{ff5c}>";
    const ASSISTANT: &str = "<\u{ff5c}hy_Assistant\u{ff5c}>";
    const EOS: &str = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}2\u{ff5c}>";
    const SEP: &str = "<\u{ff5c}hy_place\u{2581}holder\u{2581}no\u{2581}3\u{ff5c}>";

    let mut result = String::new();
    result.push_str(BOS);

    let (system_msg, loop_messages) = if !messages.is_empty() && messages[0].role == "system" {
        (Some(&messages[0].content), &messages[1..])
    } else {
        (None, &messages[..])
    };

    if let Some(sys) = system_msg {
        result.push_str(sys);
        result.push_str(SEP);
    }

    for msg in loop_messages {
        match msg.role.as_str() {
            "user" => {
                result.push_str(USER);
                result.push_str(&msg.content);
            }
            "assistant" => {
                result.push_str(ASSISTANT);
                result.push_str(&msg.content);
                result.push_str(EOS);
            }
            _ => {}
        }
    }

    result.push_str(ASSISTANT);
    result
}

fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── Handlers ──

async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

async fn stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let snap = state.engine.stats.snapshot();
    Json(snap)
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(ModelList {
        object: "list".into(),
        data: vec![ModelInfo {
            id: state.model_name.clone(),
            object: "model".into(),
            created: now_epoch(),
            owned_by: "crane".into(),
        }],
    })
}

/// Handles both streaming and non-streaming chat completions.
async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // Tokenize.
    let formatted = apply_chat_template(&req.messages);
    let input_ids = state
        .tokenizer
        .encode(formatted.as_str(), true)
        .map_err(|e| make_error(StatusCode::BAD_REQUEST, &format!("Tokenize failed: {e}")))?
        .get_ids()
        .to_vec();

    let request_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    // Submit to engine.
    let mut response_rx = state
        .engine
        .submit(
            request_id.clone(),
            input_ids,
            req.max_tokens,
            req.temperature.or(Some(0.7)),
            req.top_p.or(Some(0.8)),
            req.repetition_penalty.unwrap_or(1.05),
            120020, // EOS token
        )
        .map_err(|e| make_error(StatusCode::SERVICE_UNAVAILABLE, &e.to_string()))?;

    if req.stream {
        // ── SSE streaming ──
        let model_name = state.model_name.clone();
        let stream = make_sse_stream(request_id, model_name, response_rx);
        Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
    } else {
        // ── Non-streaming: collect all tokens ──
        let mut full_text = String::new();
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        let mut finish_reason = "length".to_string();

        while let Some(resp) = response_rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => {
                    full_text.push_str(&text);
                }
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

        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".into(),
            created: now_epoch(),
            model: state.model_name.clone(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: full_text,
                },
                finish_reason: Some(finish_reason),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        };

        Ok(Json(response).into_response())
    }
}

/// Build an SSE stream from the engine response channel.
fn make_sse_stream(
    request_id: String,
    model_name: String,
    mut rx: mpsc::UnboundedReceiver<EngineResponse>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let created = now_epoch();

    async_stream::stream! {
        // First chunk: role announcement.
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
        };
        yield Ok(Event::default().json_data(&first_chunk).unwrap());

        // Token chunks.
        while let Some(resp) = rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => {
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
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());
                }
                EngineResponse::Finished { finish_reason, .. } => {
                    let chunk = ChatCompletionChunk {
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
                            finish_reason: Some(finish_reason),
                        }],
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());
                    // [DONE] sentinel.
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                EngineResponse::Error(e) => {
                    yield Ok(Event::default().data(format!("error: {e}")));
                    break;
                }
            }
        }
    }
}

fn make_error(status: StatusCode, msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: ErrorDetail {
                message: msg.to_string(),
                r#type: "invalid_request_error".into(),
                code: None,
            },
        }),
    )
}

// ── Main ──

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    info!("Loading model from: {}", args.model_path);

    // ── Device / dtype selection ──

    let device = if args.cpu {
        crane_core::models::Device::Cpu
    } else {
        #[cfg(feature = "cuda")]
        {
            crane_core::models::Device::cuda_if_available(0)?
        }
        #[cfg(not(feature = "cuda"))]
        {
            #[cfg(target_os = "macos")]
            {
                crane_core::models::Device::new_metal(0)
                    .unwrap_or(crane_core::models::Device::Cpu)
            }
            #[cfg(not(target_os = "macos"))]
            {
                crane_core::models::Device::Cpu
            }
        }
    };

    #[cfg(feature = "cuda")]
    let dtype = if args.cpu {
        crane_core::models::DType::F32
    } else {
        crane_core::models::DType::BF16
    };
    #[cfg(not(feature = "cuda"))]
    let dtype = crane_core::models::DType::F32;

    info!("Device: {:?}, dtype: {:?}", device, dtype);

    // ── Load model ──

    let mut model = Model::new(&args.model_path, &device, &dtype)?;
    info!("Model loaded successfully");

    model.warmup();
    info!("Model warmed up");

    // Clone tokenizer for request pre-processing (runs on async side).
    let tokenizer = model.tokenizer.tokenizer.clone();

    // ── Start engine on dedicated thread ──

    let (engine, handle) = InferenceEngine::new(
        model,
        args.max_concurrent,
        args.decode_tokens_per_seq,
    );
    std::thread::Builder::new()
        .name("inference-engine".into())
        .spawn(move || engine.run())
        .expect("Failed to spawn engine thread");
    info!(
        "Inference engine started (max_concurrent={}, decode_tokens_per_seq={})",
        args.max_concurrent, args.decode_tokens_per_seq,
    );

    // ── Build router ──

    let state = Arc::new(AppState {
        engine: handle,
        model_name: args.model_name.clone(),
        tokenizer,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/stats", get(stats))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    info!("Starting server on {addr}");
    info!("  POST http://{addr}/v1/chat/completions");
    info!("  GET  http://{addr}/v1/models");
    info!("  GET  http://{addr}/v1/stats");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
