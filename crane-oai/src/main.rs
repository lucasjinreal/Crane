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
use handlers::tts::TtsGenerateRequest;
use handlers::vlm::{Gemma4VlmRequest, VlmRequest};
use openai_api::ErrorResponse;
use crane_core::models::paddleocr_vl::PaddleOcrVL;

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

    /// Model architecture: auto, hunyuan, qwen25, qwen3, qwen3_tts, paddleocr_vl
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
    #[arg(long, default_value_t = 16)]
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
    pub engine: Option<EngineHandle>,
    pub model_name: String,
    /// Shared tokenizer for request pre-processing.
    pub tokenizer: tokenizers::Tokenizer,
    /// Chat template processor (model-specific).
    pub chat_template: Box<dyn ChatTemplateProcessor>,
    /// Default EOS token ID(s) for this model.
    pub eos_token_id: Vec<u32>,
    /// Server start time (epoch seconds).
    pub server_start_time: u64,
    /// VLM model (PaddleOCR-VL) — present only for PaddleOCR VLM.
    pub vlm_tx: Option<tokio::sync::mpsc::UnboundedSender<VlmRequest>>,
    /// VLM model (Gemma4-VL) — present only for Gemma4 VLM.
    pub gemma4_vlm_tx: Option<tokio::sync::mpsc::UnboundedSender<Gemma4VlmRequest>>,
    /// TTS model (Qwen3-TTS) — present only for TTS model types.
    pub tts_tx: Option<tokio::sync::mpsc::UnboundedSender<TtsGenerateRequest>>,
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

    // Resolve auto-detection early so we know if this is VLM.
    let resolved_type = if model_type == ModelType::Auto {
        engine::model_factory::detect_model_type(&args.model_path)
    } else if model_type == ModelType::Gemma4 {
        // For Gemma4, auto-detect VL variant from vision_config presence
        let detected = engine::model_factory::detect_model_type(&args.model_path);
        if detected == ModelType::Gemma4VL { detected } else { model_type }
    } else {
        model_type
    };

    let is_vlm = resolved_type.is_vlm();
    let is_tts = resolved_type.is_tts();

    // ── Branch: VLM model vs TTS model vs standard LLM ──

    let (engine_handle, tokenizer, eos_token_id, chat_template, vlm_tx_opt, gemma4_vlm_tx_opt, tts_tx_opt):
        (Option<EngineHandle>, tokenizers::Tokenizer, Vec<u32>, Box<dyn ChatTemplateProcessor>, Option<tokio::sync::mpsc::UnboundedSender<VlmRequest>>, Option<tokio::sync::mpsc::UnboundedSender<Gemma4VlmRequest>>, Option<tokio::sync::mpsc::UnboundedSender<TtsGenerateRequest>>) = if is_tts {
        // TTS path: create Qwen3-TTS on a dedicated thread.
        info!("Loading TTS model (Qwen3-TTS) from: {}", args.model_path);
        let model_path_clone = args.model_path.clone();

        let use_cpu = args.cpu || {
            #[cfg(feature = "cuda")]
            { !candle_core::utils::cuda_is_available() }
            #[cfg(not(feature = "cuda"))]
            { true }
        };

        let tts_device = if use_cpu {
            crane_core::models::Device::Cpu
        } else {
            device.clone()
        };
        let tts_dtype = dtype;

        let (tts_tx, mut tts_rx) = tokio::sync::mpsc::unbounded_channel::<TtsGenerateRequest>();

        std::thread::Builder::new()
            .name("tts-engine".into())
            .spawn(move || {
                let mut tts = match engine::model_factory::create_tts_model(
                    &model_path_clone,
                    &tts_device,
                    &tts_dtype,
                ) {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::error!("Failed to load TTS model: {e}");
                        return;
                    }
                };
                info!("TTS engine thread started");

                while let Some(req) = tts_rx.blocking_recv() {
                    let result = (|| -> Result<handlers::tts::TtsResult, String> {
                        // Choose voice-clone vs normal TTS based on reference_audio field
                        let (audio, sr) = if let Some(ref ref_audio_path) = req.reference_audio {
                            let ref_text = req.reference_text.as_deref().unwrap_or("");
                            tracing::info!(
                                "TTS voice-clone mode: ref_audio={}, ref_text_len={}",
                                ref_audio_path,
                                ref_text.len()
                            );
                            tts.generate_voice_clone(
                                &req.input,
                                &req.language,
                                ref_audio_path,
                                ref_text,
                                req.max_tokens,
                                req.temperature,
                                req.top_p,
                                req.repetition_penalty,
                            )
                            .map_err(|e| e.to_string())?
                        } else {
                            tts.generate_speech(
                                &req.input,
                                &req.language,
                                req.voice.as_deref(),
                                req.max_tokens,
                                req.temperature,
                                req.top_p,
                                req.repetition_penalty,
                            )
                            .map_err(|e| e.to_string())?
                        };

                        tracing::info!("TTS audio tensor shape: {:?}, sr={sr}", audio.dims());
                        let audio_f32 = audio
                            .to_dtype(candle_core::DType::F32)
                            .map_err(|e| e.to_string())?
                            .flatten_all()
                            .map_err(|e| e.to_string())?;
                        let samples = audio_f32
                            .to_vec1::<f32>()
                            .map_err(|e| e.to_string())?;
                        tracing::info!("TTS writing {} samples", samples.len());

                        match req.response_format {
                            openai_api::AudioResponseFormat::Wav => {
                                let mut wav_buf = std::io::Cursor::new(Vec::new());
                                {
                                    let spec = hound::WavSpec {
                                        channels: 1,
                                        sample_rate: sr,
                                        bits_per_sample: 16,
                                        sample_format: hound::SampleFormat::Int,
                                    };
                                    let mut writer = hound::WavWriter::new(&mut wav_buf, spec)
                                        .map_err(|e| e.to_string())?;
                                    for &s in &samples {
                                        let s16 = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
                                        writer.write_sample(s16).map_err(|e| e.to_string())?;
                                    }
                                    writer.finalize().map_err(|e| e.to_string())?;
                                }

                                Ok(handlers::tts::TtsResult {
                                    audio_bytes: wav_buf.into_inner(),
                                    content_type: "audio/wav",
                                    file_name: "speech.wav".to_string(),
                                    sample_rate: sr,
                                })
                            }
                            openai_api::AudioResponseFormat::Pcm => {
                                let mut pcm = Vec::with_capacity(samples.len() * 2);
                                for &s in &samples {
                                    let s16 = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
                                    pcm.extend_from_slice(&s16.to_le_bytes());
                                }
                                Ok(handlers::tts::TtsResult {
                                    audio_bytes: pcm,
                                    content_type: "audio/pcm",
                                    file_name: "speech.pcm".to_string(),
                                    sample_rate: sr,
                                })
                            }
                            other => Err(format!(
                                "Unsupported response_format '{other:?}'. Supported: wav, pcm"
                            )),
                        }
                    })();

                    if let Err(ref e) = result {
                        tracing::error!(
                            "TTS generation failed: {e} (language={}, voice={:?}, input_len={})",
                            req.language,
                            req.voice,
                            req.input.chars().count()
                        );
                    }

                    let _ = req.tx.send(result);
                }
            })
            .expect("Failed to spawn TTS thread");

        info!("TTS model routing established (type: {:?})", resolved_type);

        // Use tokenizer from the TTS model for API compatibility.
        let tokenizer = crane_core::utils::tokenizer_utils::load_tokenizer_from_model_dir(&args.model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        let eos_id = tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(151645);

        let chat_template = engine::model_factory::create_chat_template(model_type, &args.model_path);

        (None, tokenizer, vec![eos_id], chat_template, None, None, Some(tts_tx))
    } else if is_vlm {
        // VLM path: create model on a dedicated thread to avoid Send/Sync issues.

        let use_cpu = args.cpu || {
            #[cfg(feature = "cuda")]
            { !candle_core::utils::cuda_is_available() }
            #[cfg(not(feature = "cuda"))]
            { true }
        };

        #[cfg(feature = "cuda")]
        let use_bf16 = !use_cpu;
        #[cfg(not(feature = "cuda"))]
        let use_bf16 = false;

        let tok_path = std::path::Path::new(&args.model_path).join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        let chat_template = engine::model_factory::create_chat_template(model_type, &args.model_path);

        let mut vlm_tx_opt_inner: Option<tokio::sync::mpsc::UnboundedSender<VlmRequest>> = None;
        let mut gemma4_vlm_tx_opt_inner: Option<tokio::sync::mpsc::UnboundedSender<Gemma4VlmRequest>> = None;

        if resolved_type == engine::model_factory::ModelType::Gemma4VL {
            // Gemma4 VLM path
            info!("Loading Gemma4 VLM model from: {}", args.model_path);

            let model_path_clone = args.model_path.clone();
            let device_clone = device.clone();
            let dtype_clone = dtype;
            let (g4vlm_tx, mut g4vlm_rx) = tokio::sync::mpsc::unbounded_channel::<Gemma4VlmRequest>();

            std::thread::Builder::new()
                .name("gemma4-vlm-engine".into())
                .spawn(move || {
                    use crane_core::models::gemma4::vlm::Gemma4VLModel;
                    use crane_core::models::gemma4::vision::{load_and_preprocess_image, ImagePreprocessConfig};

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
                        match req {
                            Gemma4VlmRequest::Generate { img_path, text_prompt, max_tokens, tx } => {
                                let res = (|| -> anyhow::Result<String> {
                                    // Preprocess image
                                    let preprocessed = load_and_preprocess_image(
                                        &img_path, &preprocess_config, &device_clone,
                                    )?;

                                    // Encode image through vision tower
                                    let image_embeds = vlm.encode_image(
                                        &preprocessed.pixel_values,
                                        &preprocessed.pixel_position_ids,
                                        &preprocessed.padding_positions,
                                    )?;

                                    // Build prompt in Gemma4-it chat format:
                                    // <bos><|turn>user\n<|image|>text<turn|>\n<|turn>model\n
                                    // where <|image|> (258880) expands to 280 vision tokens
                                    let image_token_id = 258880u32;
                                    let mut prompt_ids: Vec<u32> = vec![
                                        2,     // <bos>
                                        105,   // <|turn>
                                        2364,  // "user"
                                        107,   // \n
                                    ];
                                    // Image placeholder tokens (will be replaced with vision embeddings)
                                    for _ in 0..preprocessed.num_image_tokens {
                                        prompt_ids.push(image_token_id);
                                    }
                                    // Text prompt tokens
                                    if !text_prompt.is_empty() {
                                        let text_ids = vlm.tokenizer.tokenizer
                                            .encode(text_prompt.as_str(), false)
                                            .map_err(|e| anyhow::anyhow!("{e}"))?
                                            .get_ids()
                                            .to_vec();
                                        prompt_ids.extend(text_ids);
                                    }
                                    // End user turn, start model turn
                                    prompt_ids.extend_from_slice(&[
                                        106,   // <turn|>
                                        107,   // \n
                                        105,   // <|turn>
                                        4368,  // "model"
                                        107,   // \n
                                    ]);

                                    // Forward pass
                                    vlm.clear_kv_cache();
                                    let input_tensor = candle_core::Tensor::new(
                                        prompt_ids.as_slice(), &device_clone,
                                    )?.unsqueeze(0)?;

                                    let logits = vlm.forward(&input_tensor, Some(&image_embeds), 0)?;
                                    let logits = logits.squeeze(0)?.squeeze(0)?
                                        .to_dtype(candle_core::DType::F32)?;

                                    // Simple greedy generation loop
                                    let mut tokens = prompt_ids.clone();
                                    let mut generated = Vec::new();
                                    let mut next_token = {
                                        let probs = candle_nn::ops::softmax_last_dim(&logits)?;
                                        let next = probs.argmax(candle_core::D::Minus1)?
                                            .to_scalar::<u32>()?;
                                        next
                                    };
                                    generated.push(next_token);
                                    tokens.push(next_token);

                                    for _ in 1..max_tokens {
                                        if next_token == 1 || next_token == 106 { break; } // EOS(1) or <turn|>(106)
                                        let input = candle_core::Tensor::new(
                                            &[next_token], &device_clone,
                                        )?.unsqueeze(0)?;
                                        let logits = vlm.forward(&input, None, tokens.len() - 1)?;
                                        let logits = logits.squeeze(0)?.squeeze(0)?
                                            .to_dtype(candle_core::DType::F32)?;
                                        next_token = candle_nn::ops::softmax_last_dim(&logits)?
                                            .argmax(candle_core::D::Minus1)?
                                            .to_scalar::<u32>()?;
                                        generated.push(next_token);
                                        tokens.push(next_token);
                                    }

                                    let text = vlm.tokenizer.tokenizer
                                        .decode(&generated, true)
                                        .unwrap_or_default();
                                    Ok(text)
                                })();

                                let _ = tx.send(res.map_err(|e| e.to_string()));
                            }
                        }
                    }
                })
                .expect("Failed to spawn Gemma4 VLM thread");

            gemma4_vlm_tx_opt_inner = Some(g4vlm_tx);
        } else {
            // PaddleOCR-VL path
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
                                let _ = tx.send(res.map_err(|e| e.to_string()));
                            }
                            VlmRequest::RecognizeStream { img_path, task, max_tokens, token_tx, done_tx } => {
                                let res = vlm.recognize_stream(
                                    &img_path,
                                    task,
                                    max_tokens,
                                    |token_text: &str| {
                                        let _ = token_tx.send(token_text.to_string());
                                    }
                                );
                                if let Err(ref e) = res {
                                    tracing::error!("VLM RecognizeStream failed: {:?}", e);
                                }
                                let _ = done_tx.send(res.map(|_| ()).map_err(|e| e.to_string()));
                            }
                        }
                    }
                })
                .expect("Failed to spawn VLM thread");

            vlm_tx_opt_inner = Some(vlm_tx);
        }

        info!("VLM model routing established (type: {:?})", resolved_type);

        let eos_id = tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<end_of_turn>"))
            .or_else(|| tokenizer.token_to_id("<|end_of_sentence|>"))
            .unwrap_or(1);

        (None, tokenizer, vec![eos_id], chat_template, vlm_tx_opt_inner, gemma4_vlm_tx_opt_inner, None)
    } else {
        // Standard LLM path.
        let mut backend = engine::model_factory::create_backend(
            model_type, &args.model_path, &device, &dtype, format,
        )?;

        info!(
            "Model loaded successfully (type: {:?}, format: {:?})",
            resolved_type, format,
        );

        backend.warmup();
        info!("Model warmed up");

        // Clone tokenizer and get EOS token before moving backend into engine.
        let tokenizer = backend.tokenizer().clone();
        let eos_token_id = backend.eos_token_id();

        let chat_template = engine::model_factory::create_chat_template(model_type, &args.model_path);

        // ── Parse memory config ──
        let mut memory_config = MemoryConfig::parse(
            args.max_seq_len,
            args.gpu_memory_limit.as_deref(),
            &device,
        );
        memory_config.record_baseline(&device);
        let baseline_gpu = memory_config.baseline_gpu_bytes;
        info!(
            "Memory config: max_seq_len={}, gpu_limit={}, baseline_gpu={}",
            if memory_config.max_seq_len == 0 { "unlimited".to_string() } else { memory_config.max_seq_len.to_string() },
            if memory_config.gpu_memory_limit_bytes == 0 { "unlimited".to_string() } else { format_bytes(memory_config.gpu_memory_limit_bytes) },
            format_bytes(baseline_gpu),
        );

        // ── Start engine on dedicated thread ──
        let (engine, handle) = InferenceEngine::new(
            backend, args.max_concurrent, args.decode_tokens_per_seq, memory_config,
        );

        std::thread::Builder::new()
            .name("inference-engine".into())
            .spawn(move || engine.run())
            .expect("Failed to spawn engine thread");
        info!(
            "Inference engine started (max_concurrent={}, decode_tokens_per_seq={})",
            args.max_concurrent, args.decode_tokens_per_seq,
        );

        (Some(handle), tokenizer, eos_token_id, chat_template, None, None, None)
    };

    // ── Model name for API responses ──

    let model_name = args.model_name.unwrap_or_else(|| {
        std::path::Path::new(&args.model_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| resolved_type.display_name().to_string())
    });

    let gpu_memory_limit_display = args.gpu_memory_limit.clone().unwrap_or_else(|| "unlimited".to_string());

    // ── Build router ──

    let state = Arc::new(AppState {
        engine: engine_handle,
        model_name: model_name.clone(),
        tokenizer,
        chat_template,
        eos_token_id,
        server_start_time: now_epoch(),
        vlm_tx: vlm_tx_opt,
        gemma4_vlm_tx: gemma4_vlm_tx_opt,
        tts_tx: tts_tx_opt,
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
    if is_vlm {
        println!("  Mode    : VLM (vision-language model) — engine bypassed");
    } else if is_tts {
        println!("  Mode    : TTS (text-to-speech) — engine bypassed");
    }
    println!("  Listen  : http://{local_addr}");
    if !is_vlm {
        if args.max_seq_len > 0 || state.gpu_memory_limit != "unlimited" {
            let seq_str = if args.max_seq_len == 0 { "unlimited".to_string() } else { args.max_seq_len.to_string() };
            let mem_str = state.gpu_memory_limit.clone();
            println!("  Memory  : seq_len={seq_str}  gpu_limit={mem_str}");
        }
        println!("  Batch   : max_concurrent={}  decode_tokens_per_seq={}", args.max_concurrent, args.decode_tokens_per_seq);
    }
    println!("  {sep2}");
    println!("  OpenAI-compatible API");
    println!("    POST  http://{local_addr}/v1/chat/completions");
    println!("    POST  http://{local_addr}/v1/completions");
    println!("    POST  http://{local_addr}/v1/audio/speech");
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
        .route("/v1/audio/speech", post(handlers::tts::speech))
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
