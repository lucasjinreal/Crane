pub mod chat_template;
pub use crane::engine;
pub mod handlers;
pub mod openai_api;
pub mod sglang_api;

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

use chat_template::{AutoChatTemplate, ChatTemplateProcessor, HunyuanChatTemplate};
use crane::engine::model_factory::{ModelFormat, ModelType};
use crane::engine::{Gemma4VlmRequest, MemoryConfig, ModelRuntime, VlmRequest};
use openai_api::ErrorResponse;

#[derive(Parser, Debug, Clone)]
#[command(about = "OpenAI & SGLang compatible API server with continuous batching")]
pub struct Args {
    #[arg(short = 'm', long)]
    pub model_path: String,
    #[arg(long, default_value = "auto")]
    pub model_type: String,
    #[arg(long)]
    pub model_name: Option<String>,
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,
    #[arg(short = 'p', long, default_value_t = 8080)]
    pub port: u16,
    #[arg(long)]
    pub cpu: bool,
    #[arg(short = 'c', long, default_value_t = 16)]
    pub max_concurrent: usize,
    #[arg(long, default_value_t = 16)]
    pub decode_tokens_per_seq: usize,
    #[arg(long, default_value = "auto")]
    pub format: String,
    #[arg(long, default_value_t = 0)]
    pub max_seq_len: usize,
    #[arg(long)]
    pub gpu_memory_limit: Option<String>,
}

/// HTTP application state.
///
/// Wraps the protocol-independent [`ModelRuntime`] (LLM engine, TTS models,
/// VLM channels, tokenizer) plus HTTP-only fields that have no equivalent in
/// crane-wyoming.
pub struct AppState {
    pub runtime: Arc<ModelRuntime>,
    pub chat_template: Box<dyn ChatTemplateProcessor>,
    pub model_path: String,
    pub server_start_time: u64,
    pub host: String,
    pub port: u16,
    pub max_concurrent: usize,
    pub decode_tokens_per_seq: usize,
    pub max_seq_len: usize,
    pub gpu_memory_limit: String,
}

pub fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1}G", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.0}M", bytes as f64 / (1u64 << 20) as f64)
    } else {
        format!("{}B", bytes)
    }
}

pub fn make_error(status: StatusCode, msg: &str) -> (StatusCode, Json<ErrorResponse>) {
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

/// Create a chat template processor for the given model.
///
/// Lives in crane-serve (not crane::engine) because it depends on the
/// chat_template module, which uses `openai_api::ChatMessage`.
fn create_chat_template(model_type: ModelType, model_path: &str) -> Box<dyn ChatTemplateProcessor> {
    let model_type = engine::model_factory::resolve(model_type, model_path);

    match model_type {
        ModelType::HunyuanDense => {
            // Prefer jinja template from tokenizer_config.json if available.
            match AutoChatTemplate::new(model_path) {
                Ok(t) => Box::new(t),
                Err(_) => Box::new(HunyuanChatTemplate),
            }
        }
        _ => match AutoChatTemplate::new(model_path) {
            Ok(t) => Box::new(t),
            Err(e) => {
                tracing::warn!("Failed to load chat template: {e}; using Hunyuan fallback");
                Box::new(HunyuanChatTemplate)
            }
        },
    }
}

pub fn init_logging() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_file(false)
        .with_line_number(false)
        .compact()
        .init();
}

pub async fn cli_main() -> Result<()> {
    init_logging();
    run(Args::parse()).await
}

/// Encode a raw f32 PCM tensor into the requested HTTP audio format.
pub(crate) fn encode_tts_audio(
    audio: &candle_core::Tensor,
    sample_rate: u32,
    format: &openai_api::AudioResponseFormat,
) -> Result<handlers::tts::TtsResult, String> {
    let audio_f32 = audio
        .to_dtype(candle_core::DType::F32)
        .map_err(|e| e.to_string())?
        .flatten_all()
        .map_err(|e| e.to_string())?;
    let samples = audio_f32.to_vec1::<f32>().map_err(|e| e.to_string())?;
    tracing::info!("TTS writing {} samples", samples.len());
    match format {
        openai_api::AudioResponseFormat::Wav => {
            let info = crane::audio::tts::AudioInfo {
                sample_rate,
                channels: 1,
                bits_per_sample: 16,
            };
            let audio_bytes = crane::audio::tts::encode_wav(&samples, &info).map_err(|e| e.to_string())?;
            Ok(handlers::tts::TtsResult {
                audio_bytes,
                content_type: "audio/wav",
                file_name: "speech.wav".to_string(),
                sample_rate,
            })
        }
        openai_api::AudioResponseFormat::Pcm => {
            let audio_bytes = crane::audio::tts::pcm_f32_to_i16(&samples);
            Ok(handlers::tts::TtsResult {
                audio_bytes,
                content_type: "audio/pcm",
                file_name: "speech.pcm".to_string(),
                sample_rate,
            })
        }
        other => Err(format!(
            "Unsupported response_format '{other:?}'. Supported: wav, pcm"
        )),
    }
}

/// Load the tokenizer for a TTS-only model, falling back to a stub tokenizer
/// (TTS models don't need one for generation, only for chat-template plumbing
/// shared with LLM handlers).
fn load_tts_tokenizer(model_path: &str) -> tokenizers::Tokenizer {
    crane_core::utils::tokenizer_utils::load_tokenizer_from_model_dir(model_path).unwrap_or_else(|e| {
        tracing::warn!("Failed to load HF tokenizer: {e}; creating stub for TTS-only mode");
        tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default())
    })
}

fn detect_tts_eos(tokenizer: &tokenizers::Tokenizer) -> u32 {
    tokenizer
        .token_to_id("<|im_end|>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .unwrap_or(2)
}

fn load_vlm_tokenizer(model_path: &str) -> Result<tokenizers::Tokenizer> {
    let tok_path = std::path::Path::new(model_path).join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))
}

fn detect_vlm_eos(tokenizer: &tokenizers::Tokenizer) -> u32 {
    tokenizer
        .token_to_id("</s>")
        .or_else(|| tokenizer.token_to_id("<end_of_turn>"))
        .or_else(|| tokenizer.token_to_id("<|end_of_sentence|>"))
        .unwrap_or(1)
}

// Gemma4 VL chat-template token IDs, used to hand-assemble the prompt below.
const GEMMA4_BOS: u32 = 2;
const GEMMA4_EOS: u32 = 1;
const GEMMA4_START_OF_TURN: u32 = 105;
const GEMMA4_END_OF_TURN: u32 = 106;
const GEMMA4_NEWLINE: u32 = 107;
const GEMMA4_USER: u32 = 2364;
const GEMMA4_MODEL: u32 = 4368;
const GEMMA4_IMAGE_START: u32 = 255_999;
const GEMMA4_IMAGE_TOKEN: u32 = 258_880;
const GEMMA4_IMAGE_END: u32 = 258_882;

/// Spawn the dedicated thread(s) for a VLM model and register its request
/// channel on `runtime`.
///
/// VLM thread spawning stays in crane-serve for now (unlike TTS, which is
/// fully owned by `ModelRuntime`) because the Gemma4VL branch inlines a
/// model-specific token-by-token generation loop that would need a larger
/// refactor to relocate cleanly.
fn load_vlm_into_runtime(
    runtime: &mut ModelRuntime,
    args: &Args,
    resolved_type: ModelType,
    device: &crane_core::models::Device,
    dtype: crane_core::models::DType,
) {
    let use_cpu = args.cpu || {
        #[cfg(feature = "cuda")]
        {
            !candle_core::utils::cuda_is_available()
        }
        #[cfg(not(feature = "cuda"))]
        {
            true
        }
    };
    #[cfg(feature = "cuda")]
    let use_bf16 = !use_cpu;
    #[cfg(not(feature = "cuda"))]
    let use_bf16 = false;

    if resolved_type == ModelType::Gemma4VL {
        info!("Loading Gemma4 VLM model from: {}", args.model_path);
        let model_path_clone = args.model_path.clone();
        let device_clone = device.clone();
        let dtype_clone = dtype;
        let (g4vlm_tx, mut g4vlm_rx) = tokio::sync::mpsc::unbounded_channel::<Gemma4VlmRequest>();
        std::thread::Builder::new()
            .name("gemma4-vlm-engine".into())
            .spawn(move || {
                use crane_core::models::gemma4::vision::{load_and_preprocess_image, ImagePreprocessConfig};
                use crane_core::models::gemma4::vlm::Gemma4VLModel;
                let mut vlm = match Gemma4VLModel::new(&model_path_clone, &device_clone, &dtype_clone) {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::error!("Failed to load Gemma4 VLM model: {e}");
                        return;
                    }
                };
                info!("Gemma4 VLM engine thread started");
                let preprocess_config = ImagePreprocessConfig::default();
                while let Some(req) = g4vlm_rx.blocking_recv() {
                    let Gemma4VlmRequest { img_path, text_prompt, max_tokens, tx } = req;
                    let res = (|| -> anyhow::Result<String> {
                        let preprocessed = load_and_preprocess_image(&img_path, &preprocess_config, &device_clone)?;
                        let image_embeds = vlm.encode_image(
                            &preprocessed.pixel_values,
                            &preprocessed.pixel_position_ids,
                            &preprocessed.padding_positions,
                        )?;
                        let mut prompt_ids: Vec<u32> = vec![
                            GEMMA4_BOS,
                            GEMMA4_START_OF_TURN,
                            GEMMA4_USER,
                            GEMMA4_NEWLINE,
                            GEMMA4_IMAGE_START,
                        ];
                        for _ in 0..preprocessed.num_image_tokens {
                            prompt_ids.push(GEMMA4_IMAGE_TOKEN);
                        }
                        prompt_ids.push(GEMMA4_IMAGE_END);
                        if !text_prompt.is_empty() {
                            let text_ids = vlm
                                .tokenizer
                                .tokenizer
                                .encode(text_prompt.as_str(), false)
                                .map_err(|e| anyhow::anyhow!("{e}"))?
                                .get_ids()
                                .to_vec();
                            prompt_ids.extend(text_ids);
                        }
                        prompt_ids.extend_from_slice(&[
                            GEMMA4_END_OF_TURN,
                            GEMMA4_NEWLINE,
                            GEMMA4_START_OF_TURN,
                            GEMMA4_MODEL,
                            GEMMA4_NEWLINE,
                        ]);
                        vlm.clear_kv_cache();
                        let input_tensor = candle_core::Tensor::new(prompt_ids.as_slice(), &device_clone)?.unsqueeze(0)?;
                        let logits = vlm
                            .forward(&input_tensor, Some(&image_embeds), 0)?
                            .squeeze(0)?
                            .squeeze(0)?
                            .to_dtype(candle_core::DType::F32)?;
                        let mut tokens = prompt_ids.clone();
                        let mut generated = Vec::new();
                        let mut next_token = candle_nn::ops::softmax_last_dim(&logits)?
                            .argmax(candle_core::D::Minus1)?
                            .to_scalar::<u32>()?;
                        generated.push(next_token);
                        tokens.push(next_token);
                        for _ in 1..max_tokens {
                            if next_token == GEMMA4_EOS || next_token == GEMMA4_END_OF_TURN {
                                break;
                            }
                            let input = candle_core::Tensor::new(&[next_token], &device_clone)?.unsqueeze(0)?;
                            let logits = vlm
                                .forward(&input, None, tokens.len() - 1)?
                                .squeeze(0)?
                                .squeeze(0)?
                                .to_dtype(candle_core::DType::F32)?;
                            next_token = candle_nn::ops::softmax_last_dim(&logits)?
                                .argmax(candle_core::D::Minus1)?
                                .to_scalar::<u32>()?;
                            generated.push(next_token);
                            tokens.push(next_token);
                        }
                        Ok(vlm.tokenizer.tokenizer.decode(&generated, true).unwrap_or_default())
                    })();
                    let _ = tx.send(res);
                }
            })
            .expect("Failed to spawn Gemma4 VLM thread");
        runtime.set_gemma4_vlm_tx(g4vlm_tx);
    } else {
        info!("Loading VLM model (PaddleOCR-VL) from: {}", args.model_path);
        let model_path_clone = args.model_path.clone();
        let (vlm_tx, mut vlm_rx) = tokio::sync::mpsc::unbounded_channel::<VlmRequest>();
        std::thread::Builder::new()
            .name("vlm-engine".into())
            .spawn(move || {
                let mut vlm = match engine::model_factory::create_vlm_model(&model_path_clone, use_cpu, use_bf16) {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::error!("Failed to load VLM model: {e}");
                        return;
                    }
                };
                info!("VLM engine thread started");
                while let Some(req) = vlm_rx.blocking_recv() {
                    match req {
                        VlmRequest::Recognize { img_path, task, max_tokens, tx } => {
                            let res = vlm.recognize(&img_path, task, max_tokens).map(|r| r.text);
                            if let Err(ref e) = res {
                                tracing::error!("VLM Recognize failed: {:?}", e);
                            }
                            let _ = tx.send(res);
                        }
                        VlmRequest::RecognizeStream { img_path, task, max_tokens, token_tx, done_tx } => {
                            let res = vlm.recognize_stream(&img_path, task, max_tokens, |token_text: &str| {
                                let _ = token_tx.send(token_text.to_string());
                            });
                            if let Err(ref e) = res {
                                tracing::error!("VLM RecognizeStream failed: {:?}", e);
                            }
                            let _ = done_tx.send(res.map(|_| ()));
                        }
                    }
                }
            })
            .expect("Failed to spawn VLM thread");
        runtime.set_vlm_tx(vlm_tx);
    }
    info!("VLM model routing established (type: {:?})", resolved_type);
}

pub async fn run(args: Args) -> Result<()> {
    info!("Loading model from: {}", args.model_path);

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
                crane_core::models::Device::new_metal(0).unwrap_or(crane_core::models::Device::Cpu)
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

    let model_type = ModelType::from_str(&args.model_type);
    let format = ModelFormat::from_str(&args.format);

    let resolved_type = if model_type == ModelType::Auto {
        engine::model_factory::detect_model_type(&args.model_path)
    } else if model_type == ModelType::Gemma4 {
        let detected = engine::model_factory::detect_model_type(&args.model_path);
        if detected == ModelType::Gemma4VL {
            detected
        } else {
            model_type
        }
    } else {
        model_type
    };

    let is_vlm = resolved_type.is_vlm();
    let is_tts = resolved_type.is_tts();

    let model_name = args.model_name.clone().unwrap_or_else(|| {
        std::path::Path::new(&args.model_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| resolved_type.display_name().to_string())
    });

    let runtime = if is_tts {
        info!("Loading TTS model ({:?}) from: {}", resolved_type, args.model_path);
        let tokenizer = load_tts_tokenizer(&args.model_path);
        let eos_id = detect_tts_eos(&tokenizer);
        let mut runtime = ModelRuntime::new(
            model_name.clone(),
            resolved_type,
            dtype_name.clone(),
            device_name.clone(),
            tokenizer,
            vec![eos_id],
        );
        runtime.load_tts(&args.model_path, &device, &dtype)?;
        runtime
    } else if is_vlm {
        let tokenizer = load_vlm_tokenizer(&args.model_path)?;
        let eos_id = detect_vlm_eos(&tokenizer);
        let mut runtime = ModelRuntime::new(
            model_name.clone(),
            resolved_type,
            dtype_name.clone(),
            device_name.clone(),
            tokenizer,
            vec![eos_id],
        );
        load_vlm_into_runtime(&mut runtime, &args, resolved_type, &device, dtype);
        runtime
    } else {
        let mut backend = engine::model_factory::create_backend(model_type, &args.model_path, &device, &dtype, format)?;
        info!("Model loaded successfully (type: {:?}, format: {:?})", resolved_type, format);
        backend.warmup();
        info!("Model warmed up");
        let tokenizer = backend.tokenizer().clone();
        let eos_token_id = backend.eos_token_id();
        let mut memory_config = MemoryConfig::parse(args.max_seq_len, args.gpu_memory_limit.as_deref(), &device);
        memory_config.record_baseline(&device);
        let baseline_gpu = memory_config.baseline_gpu_bytes;
        info!(
            "Memory config: max_seq_len={}, gpu_limit={}, baseline_gpu={}",
            if memory_config.max_seq_len == 0 { "unlimited".to_string() } else { memory_config.max_seq_len.to_string() },
            if memory_config.gpu_memory_limit_bytes == 0 { "unlimited".to_string() } else { format_bytes(memory_config.gpu_memory_limit_bytes) },
            format_bytes(baseline_gpu)
        );
        let mut runtime = ModelRuntime::new(
            model_name.clone(),
            resolved_type,
            dtype_name.clone(),
            device_name.clone(),
            tokenizer,
            eos_token_id,
        );
        runtime.load_llm_engine(backend, args.max_concurrent, args.decode_tokens_per_seq, memory_config);
        info!(
            "Inference engine started (max_concurrent={}, decode_tokens_per_seq={})",
            args.max_concurrent, args.decode_tokens_per_seq
        );
        runtime
    };

    let runtime = Arc::new(runtime);

    let chat_template = create_chat_template(model_type, &args.model_path);
    let gpu_memory_limit_display = args.gpu_memory_limit.clone().unwrap_or_else(|| "unlimited".to_string());
    let state = Arc::new(AppState {
        runtime: Arc::clone(&runtime),
        chat_template,
        model_path: args.model_path.clone(),
        server_start_time: now_epoch(),
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
    info!(version = env!("CARGO_PKG_VERSION"), listen = %format!("http://{local_addr}"), "crane-serve ready");
    info!(model = %model_name, model_type = %resolved_type.display_name(), device = %device_name, dtype = %dtype_name, "model loaded");
    if is_vlm {
        info!("mode: vlm");
    } else if is_tts {
        info!("mode: tts");
    } else {
        info!(max_concurrent = args.max_concurrent, decode_tokens_per_seq = args.decode_tokens_per_seq, "scheduler configured");
        if args.max_seq_len > 0 || state.gpu_memory_limit != "unlimited" {
            let seq_str = if args.max_seq_len == 0 { "unlimited".to_string() } else { args.max_seq_len.to_string() };
            info!(max_seq_len = %seq_str, gpu_memory_limit = %state.gpu_memory_limit, "memory limits configured");
        }
    }
    info!(chat_completions = %format!("http://{local_addr}/v1/chat/completions"), models = %format!("http://{local_addr}/v1/models"), health = %format!("http://{local_addr}/health"), "api endpoints");
    axum::serve(listener, app).await?;
    Ok(())
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(handlers::common::health))
        .route("/v1/stats", get(handlers::common::stats))
        .route("/v1/chat/completions", post(handlers::openai::chat_completions))
        .route("/v1/completions", post(handlers::openai::completions))
        .route("/v1/audio/speech", post(handlers::tts::speech))
        .route("/v1/models", get(handlers::openai::list_models))
        .route("/v1/models/{model_id}", get(handlers::openai::retrieve_model))
        .route("/v1/tokenize", post(handlers::openai::tokenize))
        .route("/v1/detokenize", post(handlers::openai::detokenize))
        .route("/tokenize", post(handlers::openai::tokenize))
        .route("/detokenize", post(handlers::openai::detokenize))
        .route("/generate", post(handlers::sglang::generate))
        .route("/model_info", get(handlers::sglang::model_info))
        .route("/server_info", get(handlers::sglang::server_info))
        .route("/health_generate", get(handlers::sglang::health_generate))
        .route("/flush_cache", get(handlers::sglang::flush_cache).post(handlers::sglang::flush_cache))
        .route("/abort_request", post(handlers::sglang::abort_request))
        .with_state(state)
}
