mod chat_template;
mod engine;
mod handlers;
mod openai_api;
mod sglang_api;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::{
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use clap::Parser;
use tracing::info;

use chat_template::ChatTemplateProcessor;
use engine::model_factory::{ModelFormat, ModelType};
use engine::{EngineHandle, InferenceEngine, MemoryConfig};
use openai_api::ErrorResponse;

// ═════════════════════════════════════════════════════════════
//  CLI
// ═════════════════════════════════════════════════════════════

#[derive(Parser, Debug)]
#[command(
    name = "crane-oai",
    about = "OpenAI & SGLang compatible API server with continuous batching"
)]
struct Args {
    /// Path to model directory or GGUF file
    #[arg(long)]
    model_path: String,

    /// Model architecture: auto, hunyuan, qwen25, qwen3
    #[arg(long, default_value = "auto")]
    model_type: String,

    /// Model name to report in API responses (defaults to directory name)
    #[arg(long)]
    model_name: Option<String>,

    /// Host to bind
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Use CPU even if GPU is available
    #[arg(long)]
    cpu: bool,

    /// Max concurrent sequences in decode phase
    #[arg(long, default_value_t = 32)]
    max_concurrent: usize,

    /// Tokens to decode per sequence before switching (higher = fewer KV swaps)
    #[arg(long, default_value_t = 16)]
    decode_tokens_per_seq: usize,

    /// Model weight format: auto, safetensors, or gguf
    #[arg(long, default_value = "auto")]
    format: String,

    /// Maximum sequence length (prompt + completion tokens).
    /// Limits KV cache growth per sequence. 0 = unlimited (model default).
    #[arg(long, default_value_t = 0)]
    max_seq_len: usize,

    /// GPU memory limit. Accepts:
    ///   - Absolute size: "5G", "8G", "5120M", "5368709120" (bytes)
    ///   - Utilization fraction: "0.7" (70% of total GPU memory)
    /// When the limit is reached, the engine stops admitting new sequences
    /// until existing ones complete and free memory.
    #[arg(long)]
    gpu_memory_limit: Option<String>,
}

// ═════════════════════════════════════════════════════════════
//  App state (shared across handlers)
// ═════════════════════════════════════════════════════════════

pub struct AppState {
    pub engine: EngineHandle,
    pub model_name: String,
    /// Shared tokenizer for request pre-processing.
    pub tokenizer: tokenizers::Tokenizer,
    /// Chat template processor (model-specific).
    pub chat_template: Box<dyn ChatTemplateProcessor>,
    /// Default EOS token ID(s) for this model.
    pub eos_token_id: Vec<u32>,
    /// Server start time (epoch seconds).
    pub server_start_time: u64,
    // ── Fields for /model_info and /server_info ──
    pub model_path: String,
    pub model_type_name: String,
    pub dtype_name: String,
    pub device_name: String,
    pub host: String,
    pub port: u16,
    pub max_concurrent: usize,
    pub decode_tokens_per_seq: usize,
    pub max_seq_len: usize,
    pub gpu_memory_limit: String,
}

// ═════════════════════════════════════════════════════════════
//  Shared helpers
// ═════════════════════════════════════════════════════════════

pub fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Format a byte count as a human-readable string.
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1}G", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.0}M", bytes as f64 / (1u64 << 20) as f64)
    } else {
        format!("{}B", bytes)
    }
}

pub fn make_error(
    status: StatusCode,
    msg: &str,
) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: openai_api::ErrorDetail {
                message: msg.to_string(),
                r#type: "invalid_request_error".into(),
                code: None,
            },
        }),
    )
}

// ═════════════════════════════════════════════════════════════
//  Main
// ═════════════════════════════════════════════════════════════

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

    let device_name = format!("{:?}", device);
    let dtype_name = format!("{:?}", dtype);

    info!("Device: {}, dtype: {}", device_name, dtype_name);

    // ── Resolve model type ──

    let model_type = ModelType::from_str(&args.model_type);
    let format = ModelFormat::from_str(&args.format);

    // ── Load model via factory ──

    let mut backend =
        engine::model_factory::create_backend(model_type, &args.model_path, &device, &dtype, format)?;

    let resolved_type = if model_type == ModelType::Auto {
        engine::model_factory::detect_model_type(&args.model_path)
    } else {
        model_type
    };

    info!(
        "Model loaded successfully (type: {:?}, format: {:?})",
        resolved_type, format,
    );

    backend.warmup();
    info!("Model warmed up");

    // Clone tokenizer and get EOS token before moving backend into engine.
    let tokenizer = backend.tokenizer().clone();
    let eos_token_id = backend.eos_token_id();

    // ── Chat template ──

    let chat_template =
        engine::model_factory::create_chat_template(model_type, &args.model_path);

    // ── Model name for API responses ──

    let model_name = args.model_name.unwrap_or_else(|| {
        std::path::Path::new(&args.model_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| resolved_type.display_name().to_string())
    });

    // ── Parse memory config ──

    let memory_config = MemoryConfig::parse(
        args.max_seq_len,
        args.gpu_memory_limit.as_deref(),
        &device,
    );
    info!(
        "Memory config: max_seq_len={}, gpu_memory_limit_bytes={}",
        if memory_config.max_seq_len == 0 { "unlimited".to_string() } else { memory_config.max_seq_len.to_string() },
        if memory_config.gpu_memory_limit_bytes == 0 { "unlimited".to_string() } else { format_bytes(memory_config.gpu_memory_limit_bytes) },
    );
    let gpu_memory_limit_display = args.gpu_memory_limit.clone().unwrap_or_else(|| "unlimited".to_string());
    let gpu_limit_bytes = memory_config.gpu_memory_limit_bytes;

    // ── Start engine on dedicated thread ──

    let (engine, handle) =
        InferenceEngine::new(backend, args.max_concurrent, args.decode_tokens_per_seq, memory_config);

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
        model_name: model_name.clone(),
        tokenizer,
        chat_template,
        eos_token_id,
        server_start_time: now_epoch(),
        model_path: args.model_path.clone(),
        model_type_name: resolved_type.display_name().to_string(),
        dtype_name,
        device_name,
        host: args.host.clone(),
        port: args.port,
        max_concurrent: args.max_concurrent,
        decode_tokens_per_seq: args.decode_tokens_per_seq,
        max_seq_len: args.max_seq_len,
        gpu_memory_limit: gpu_memory_limit_display,
    });

    let app = build_router(state.clone());

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    let local_addr = listener.local_addr()?;

    // ── Startup banner (printed after successful bind) ──

    let sep  = "═".repeat(60);
    let sep2 = "─".repeat(60);
    println!("\n  {sep}");
    println!("  🚀  crane-oai  v{}  ready", env!("CARGO_PKG_VERSION"));
    println!("  {sep}");
    println!("  Model   : {} ({})", model_name, resolved_type.display_name());
    println!("  Device  : {}  │  dtype: {}", state.device_name, state.dtype_name);
    println!("  Listen  : http://{local_addr}");
    if args.max_seq_len > 0 || gpu_limit_bytes > 0 {
        let seq_str = if args.max_seq_len == 0 { "unlimited".to_string() } else { args.max_seq_len.to_string() };
        let mem_str = if gpu_limit_bytes == 0 { "unlimited".to_string() } else { format_bytes(gpu_limit_bytes) };
        println!("  Memory  : seq_len={seq_str}  gpu_limit={mem_str}");
    }
    println!("  Batch   : max_concurrent={}  decode_tokens_per_seq={}", args.max_concurrent, args.decode_tokens_per_seq);
    println!("  {sep2}");
    println!("  OpenAI-compatible API");
    println!("    POST  http://{local_addr}/v1/chat/completions");
    println!("    POST  http://{local_addr}/v1/completions");
    println!("    GET   http://{local_addr}/v1/models");
    println!("    POST  http://{local_addr}/v1/tokenize");
    println!("    POST  http://{local_addr}/v1/detokenize");
    println!("  {sep2}");
    println!("  SGLang-compatible API");
    println!("    POST  http://{local_addr}/generate");
    println!("    GET   http://{local_addr}/model_info");
    println!("    GET   http://{local_addr}/server_info");
    println!("    GET   http://{local_addr}/health_generate");
    println!("    POST  http://{local_addr}/flush_cache");
    println!("    POST  http://{local_addr}/abort_request");
    println!("  {sep2}");
    println!("  Management");
    println!("    GET   http://{local_addr}/health");
    println!("    GET   http://{local_addr}/v1/stats");
    println!("  {sep}\n");

    axum::serve(listener, app).await?;

    Ok(())
}

/// Build the Axum router with all endpoint families.
fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        // ── Health & management ──
        .route("/health", get(handlers::common::health))
        .route("/v1/stats", get(handlers::common::stats))
        // ── OpenAI-compatible ──
        .route("/v1/chat/completions", post(handlers::openai::chat_completions))
        .route("/v1/completions", post(handlers::openai::completions))
        .route("/v1/models", get(handlers::openai::list_models))
        .route("/v1/models/{model_id}", get(handlers::openai::retrieve_model))
        .route("/v1/tokenize", post(handlers::openai::tokenize))
        .route("/v1/detokenize", post(handlers::openai::detokenize))
        // ── Convenience aliases (SGLang-style) ──
        .route("/tokenize", post(handlers::openai::tokenize))
        .route("/detokenize", post(handlers::openai::detokenize))
        // ── SGLang-compatible native API ──
        .route("/generate", post(handlers::sglang::generate))
        .route("/model_info", get(handlers::sglang::model_info))
        .route("/server_info", get(handlers::sglang::server_info))
        .route("/health_generate", get(handlers::sglang::health_generate))
        .route("/flush_cache", get(handlers::sglang::flush_cache).post(handlers::sglang::flush_cache))
        .route("/abort_request", post(handlers::sglang::abort_request))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_epoch_is_reasonable() {
        let ts = now_epoch();
        // Should be after 2020-01-01 (1577836800).
        assert!(ts > 1_577_836_800);
    }

    #[test]
    fn make_error_returns_correct_status() {
        let (status, Json(body)) = make_error(StatusCode::BAD_REQUEST, "test error");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body.error.message, "test error");
        assert_eq!(body.error.r#type, "invalid_request_error");
        assert!(body.error.code.is_none());
    }

    #[test]
    fn make_error_internal() {
        let (status, Json(body)) = make_error(StatusCode::INTERNAL_SERVER_ERROR, "boom");
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body.error.message, "boom");
    }

    #[test]
    fn make_error_service_unavailable() {
        let (status, _) = make_error(StatusCode::SERVICE_UNAVAILABLE, "overloaded");
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }
}
