pub mod chat_template;
pub mod engine;
pub mod handlers;
pub mod openai_api;
pub mod sglang_api;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::{
    extract::DefaultBodyLimit,
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
use handlers::asr::AsrTranscribeRequest;
use handlers::tts::TtsGenerateRequest;
use handlers::vlm::{Gemma4VlmRequest, VlmRequest};
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
    /// In-situ quantization level for safetensors checkpoints (e.g. q4k,
    /// q8_0). Currently supported for qwen3_5 only. Overrides `CRANE_ISQ`.
    #[arg(long)]
    pub quant: Option<String>,
    /// Compute dtype: f16, bf16 or f32. Defaults per device: BF16 on CUDA,
    /// F32 on CPU; on Metal F32, except model families validated in F16
    /// (currently qwen3_5) which default to F16.
    #[arg(long)]
    pub dtype: Option<String>,
    #[arg(long, default_value_t = 0)]
    pub max_seq_len: usize,
    #[arg(long)]
    pub gpu_memory_limit: Option<String>,
}

pub struct AppState {
    pub engine: Option<EngineHandle>,
    pub model_name: String,
    pub tokenizer: tokenizers::Tokenizer,
    pub chat_template: Box<dyn ChatTemplateProcessor>,
    pub eos_token_id: Vec<u32>,
    pub server_start_time: u64,
    pub vlm_tx: Option<tokio::sync::mpsc::UnboundedSender<VlmRequest>>,
    pub gemma4_vlm_tx: Option<tokio::sync::mpsc::UnboundedSender<Gemma4VlmRequest>>,
    pub tts_tx: Option<tokio::sync::mpsc::UnboundedSender<TtsGenerateRequest>>,
    /// Channel to the ASR engine thread; `None` unless an ASR model is loaded.
    pub asr_tx: Option<tokio::sync::mpsc::UnboundedSender<AsrTranscribeRequest>>,
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

fn encode_tts_audio(
    audio: &candle_core::Tensor,
    audio_info: &crane::audio::AudioInfo,
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
            let wav_bytes = crane::audio::encode_wav(&samples, audio_info).map_err(|e| e.to_string())?;
            Ok(handlers::tts::TtsResult {
                audio_bytes: wav_bytes,
                content_type: "audio/wav",
                file_name: "speech.wav".to_string(),
                sample_rate: audio_info.sample_rate,
            })
        }
        openai_api::AudioResponseFormat::Pcm => {
            let pcm = crane::audio::pcm_f32_to_i16(&samples);
            Ok(handlers::tts::TtsResult {
                audio_bytes: pcm,
                content_type: "audio/pcm",
                file_name: "speech.pcm".to_string(),
                sample_rate: audio_info.sample_rate,
            })
        }
        other => Err(format!(
            "Unsupported response_format '{other:?}'. Supported: wav, pcm"
        )),
    }
}

fn generate_audio(
    tts: &mut dyn crane::audio::Tts,
    model_name: &str,
    req: &TtsGenerateRequest,
) -> Result<candle_core::Tensor, String> {
    let opts = crane_core::generation::SpeechOptions {
        max_new_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        repetition_penalty: req.repetition_penalty,
    };
    if let Some(ref ref_audio_path) = req.reference_audio {
        if !tts.supports_voice_cloning() {
            return Err(format!("{model_name} does not support voice cloning"));
        }
        let ref_text = req.reference_text.as_deref().unwrap_or("");
        tracing::info!(
            "TTS voice-clone mode: ref_audio={}, ref_text_len={}",
            ref_audio_path,
            ref_text.len()
        );
        tts.generate_voice_clone(&req.input, &req.language, ref_audio_path, ref_text, &opts)
            .map_err(|e| e.to_string())
    } else {
        tts.generate_speech(&req.input, &req.language, req.voice.as_deref(), &opts)
            .map_err(|e| e.to_string())
    }
}

fn run_tts_loop(
    mut tts_rx: tokio::sync::mpsc::UnboundedReceiver<TtsGenerateRequest>,
    model_name: &str,
    tts: &mut dyn crane::audio::Tts,
) {
    info!("{model_name} engine thread started");
    let audio_info = tts.audio_info();
    while let Some(req) = tts_rx.blocking_recv() {
        let result = generate_audio(tts, model_name, &req)
            .and_then(|audio| encode_tts_audio(&audio, &audio_info, &req.response_format));
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
}

fn transcribe_audio(asr: &mut dyn crane::audio::Asr, req: &AsrTranscribeRequest) -> Result<String, String> {
    let tmp_file = tempfile::NamedTempFile::new().map_err(|e| e.to_string())?;
    std::fs::write(tmp_file.path(), &req.audio_bytes).map_err(|e| e.to_string())?;
    let tmp_path = tmp_file
        .path()
        .to_str()
        .ok_or_else(|| "temp audio file path is not valid UTF-8".to_string())?;

    let sample_rate = asr.input_sample_rate();
    let samples = crane::audio::load_wav_f32(tmp_path, sample_rate).map_err(|e| e.to_string())?;

    let defaults = crane_core::generation::TranscribeOptions::default();
    let opts = crane_core::generation::TranscribeOptions {
        temperature: req.temperature.unwrap_or(defaults.temperature),
        language: req.language.clone(),
        ..defaults
    };
    asr.transcribe(&samples, &opts).map(|t| t.text).map_err(|e| e.to_string())
}

fn run_asr_loop(
    mut asr_rx: tokio::sync::mpsc::UnboundedReceiver<AsrTranscribeRequest>,
    model_name: &str,
    asr: &mut dyn crane::audio::Asr,
) {
    info!("{model_name} engine thread started");
    while let Some(req) = asr_rx.blocking_recv() {
        let result = transcribe_audio(asr, &req);
        if let Err(ref e) = result {
            tracing::error!("ASR transcription failed: {e}");
        }
        let _ = req.tx.send(result);
    }
}

/// Resolve the compute dtype. An explicit `--dtype` always wins; otherwise
/// BF16 on CUDA and F32 elsewhere — except model families validated in F16 on
/// Metal (currently qwen3_5), which default to F16 there (halves weight,
/// activation and fp-KV memory vs F32).
///
/// F16 stays opt-in for the other families: it has less range than the BF16
/// most checkpoints are trained in, and models with large intermediate
/// activations (e.g. Gemma) can overflow to inf/NaN — flip a family's default
/// only after verifying its output quality in F16.
fn resolve_dtype(
    flag: Option<&str>,
    device: &crane_core::models::Device,
    model_type: ModelType,
) -> Result<crane_core::models::DType> {
    use crane_core::models::DType;
    if let Some(name) = flag {
        return match name.to_lowercase().as_str() {
            "f16" | "fp16" | "half" => Ok(DType::F16),
            "bf16" => Ok(DType::BF16),
            "f32" | "fp32" => Ok(DType::F32),
            other => anyhow::bail!("unsupported --dtype '{other}' (expected f16, bf16 or f32)"),
        };
    }
    if device.is_cuda() {
        return Ok(DType::BF16);
    }
    if device.is_metal() && model_type == ModelType::Qwen3_5 {
        return Ok(DType::F16);
    }
    Ok(DType::F32)
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

    let dtype = resolve_dtype(args.dtype.as_deref(), &device, resolved_type)?;

    let device_name = format!("{:?}", device);
    let dtype_name = format!("{:?}", dtype);

    info!("Device: {}, dtype: {}", device_name, dtype_name);

    let is_vlm = resolved_type.is_vlm();
    let is_tts = resolved_type.is_tts();
    let is_asr = resolved_type.is_asr();

    let (
        engine_handle,
        tokenizer,
        eos_token_id,
        chat_template,
        vlm_tx_opt,
        gemma4_vlm_tx_opt,
        tts_tx_opt,
        asr_tx_opt,
    ): (
        Option<EngineHandle>,
        tokenizers::Tokenizer,
        Vec<u32>,
        Box<dyn ChatTemplateProcessor>,
        Option<tokio::sync::mpsc::UnboundedSender<VlmRequest>>,
        Option<tokio::sync::mpsc::UnboundedSender<Gemma4VlmRequest>>,
        Option<tokio::sync::mpsc::UnboundedSender<TtsGenerateRequest>>,
        Option<tokio::sync::mpsc::UnboundedSender<AsrTranscribeRequest>>,
    ) = if is_tts {
        info!("Loading TTS model ({:?}) from: {}", resolved_type, args.model_path);
        let model_path_clone = args.model_path.clone();
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
        let tts_device = if use_cpu { crane_core::models::Device::Cpu } else { device.clone() };
        let tts_dtype = dtype;
        let (tts_tx, tts_rx) = tokio::sync::mpsc::unbounded_channel::<TtsGenerateRequest>();
        let resolved_name = resolved_type.display_name().to_string();
        std::thread::Builder::new()
            .name("tts-engine".into())
            .spawn(move || {
                let mut tts = match engine::model_factory::create_tts(
                    resolved_type,
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
                // Install candle's affinity-pinned rayon pool for this thread's lifetime.
                tts_device.with_context(|| {
                    run_tts_loop(tts_rx, &resolved_name, tts.as_mut());
                });
            })
            .expect("Failed to spawn TTS thread");
        info!("TTS model routing established (type: {:?})", resolved_type);
        let tokenizer = crane_core::utils::tokenizer_utils::load_tokenizer_from_model_dir(&args.model_path)
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to load HF tokenizer: {e}; creating stub for TTS-only mode");
                tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default())
            });
        let eos_id = tokenizer.token_to_id("<|im_end|>").or_else(|| tokenizer.token_to_id("<|endoftext|>")).unwrap_or(2);
        let chat_template = engine::model_factory::create_chat_template(model_type, &args.model_path);
        (None, tokenizer, vec![eos_id], chat_template, None, None, Some(tts_tx), None)
    } else if is_asr {
        info!("Loading ASR model ({:?}) from: {}", resolved_type, args.model_path);
        let model_path_clone = args.model_path.clone();
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
        let asr_device = if use_cpu { crane_core::models::Device::Cpu } else { device.clone() };
        let asr_dtype = dtype;
        let (asr_tx, asr_rx) = tokio::sync::mpsc::unbounded_channel::<AsrTranscribeRequest>();
        let resolved_name = resolved_type.display_name().to_string();
        std::thread::Builder::new()
            .name("asr-engine".into())
            .spawn(move || {
                let mut asr = match engine::model_factory::create_asr(
                    resolved_type,
                    &model_path_clone,
                    &asr_device,
                    &asr_dtype,
                ) {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::error!("Failed to load ASR model: {e}");
                        return;
                    }
                };
                // Install candle's affinity-pinned rayon pool for this thread's lifetime.
                asr_device.with_context(|| {
                    run_asr_loop(asr_rx, &resolved_name, asr.as_mut());
                });
            })
            .expect("Failed to spawn ASR thread");
        info!("ASR model routing established (type: {:?})", resolved_type);
        let tokenizer = crane_core::utils::tokenizer_utils::load_tokenizer_from_model_dir(&args.model_path)
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to load HF tokenizer: {e}; creating stub for ASR-only mode");
                tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default())
            });
        let eos_id = tokenizer.token_to_id("<|im_end|>").or_else(|| tokenizer.token_to_id("<|endoftext|>")).unwrap_or(2);
        let chat_template = engine::model_factory::create_chat_template(model_type, &args.model_path);
        (None, tokenizer, vec![eos_id], chat_template, None, None, None, Some(asr_tx))
    } else if is_vlm {
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
        let tok_path = std::path::Path::new(&args.model_path).join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        let chat_template = engine::model_factory::create_chat_template(model_type, &args.model_path);
        let mut vlm_tx_opt_inner: Option<tokio::sync::mpsc::UnboundedSender<VlmRequest>> = None;
        let mut gemma4_vlm_tx_opt_inner: Option<tokio::sync::mpsc::UnboundedSender<Gemma4VlmRequest>> = None;
        if resolved_type == engine::model_factory::ModelType::Gemma4VL {
            info!("Loading Gemma4 VLM model from: {}", args.model_path);
            let model_path_clone = args.model_path.clone();
            let device_clone = device.clone();
            let dtype_clone = dtype;
            let (g4vlm_tx, mut g4vlm_rx) = tokio::sync::mpsc::unbounded_channel::<Gemma4VlmRequest>();
            std::thread::Builder::new().name("gemma4-vlm-engine".into()).spawn(move || {
                use crane_core::models::gemma4::vision::{ImagePreprocessConfig, load_and_preprocess_image};
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
                // Install candle's affinity-pinned rayon pool for this thread's lifetime.
                device_clone.with_context(|| {
                    while let Some(req) = g4vlm_rx.blocking_recv() {
                        let Gemma4VlmRequest { img_path, text_prompt, max_tokens, tx } = req;
                        let res = (|| -> anyhow::Result<String> {
                            let preprocessed = load_and_preprocess_image(&img_path, &preprocess_config, &device_clone)?;
                            let image_embeds = vlm.encode_image(&preprocessed.pixel_values, &preprocessed.pixel_position_ids, &preprocessed.padding_positions)?;
                            let image_token_id = 258880u32;
                            let mut prompt_ids: Vec<u32> = vec![2, 105, 2364, 107, 255999];
                            for _ in 0..preprocessed.num_image_tokens { prompt_ids.push(image_token_id); }
                            prompt_ids.push(258882);
                            if !text_prompt.is_empty() {
                                let text_ids = vlm.tokenizer.tokenizer.encode(text_prompt.as_str(), false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec();
                                prompt_ids.extend(text_ids);
                            }
                            prompt_ids.extend_from_slice(&[106, 107, 105, 4368, 107]);
                            vlm.clear_kv_cache();
                            let input_tensor = candle_core::Tensor::new(prompt_ids.as_slice(), &device_clone)?.unsqueeze(0)?;
                            let logits = vlm.forward(&input_tensor, Some(&image_embeds), 0)?.squeeze(0)?.squeeze(0)?.to_dtype(candle_core::DType::F32)?;
                            let mut tokens = prompt_ids.clone();
                            let mut generated = Vec::new();
                            let mut next_token = candle_nn::ops::softmax_last_dim(&logits)?.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;
                            generated.push(next_token);
                            tokens.push(next_token);
                            for _ in 1..max_tokens {
                                if next_token == 1 || next_token == 106 { break; }
                                let input = candle_core::Tensor::new(&[next_token], &device_clone)?.unsqueeze(0)?;
                                let logits = vlm.forward(&input, None, tokens.len() - 1)?.squeeze(0)?.squeeze(0)?.to_dtype(candle_core::DType::F32)?;
                                next_token = candle_nn::ops::softmax_last_dim(&logits)?.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;
                                generated.push(next_token);
                                tokens.push(next_token);
                            }
                            Ok(vlm.tokenizer.tokenizer.decode(&generated, true).unwrap_or_default())
                        })();
                        let _ = tx.send(res.map_err(|e| e.to_string()));
                    }
                });
            }).expect("Failed to spawn Gemma4 VLM thread");
            gemma4_vlm_tx_opt_inner = Some(g4vlm_tx);
        } else {
            info!("Loading VLM model (PaddleOCR-VL) from: {}", args.model_path);
            let model_path_clone = args.model_path.clone();
            let (vlm_tx, mut vlm_rx) = tokio::sync::mpsc::unbounded_channel::<VlmRequest>();
            std::thread::Builder::new().name("vlm-engine".into()).spawn(move || {
                let mut vlm = match engine::model_factory::create_vlm_model(&model_path_clone, use_cpu, use_bf16) {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::error!("Failed to load VLM model: {e}");
                        return;
                    }
                };
                info!("VLM engine thread started");
                // Clone device: with_context borrows &self, which would overlap the &mut vlm borrows below.
                let vlm_device = vlm.device.clone();
                // Install candle's affinity-pinned rayon pool for this thread's lifetime.
                vlm_device.with_context(|| {
                    while let Some(req) = vlm_rx.blocking_recv() {
                        match req {
                            VlmRequest::Recognize { img_path, task, max_tokens, tx } => {
                                let res = vlm.recognize(&img_path, task, max_tokens).map(|r| r.text);
                                if let Err(ref e) = res { tracing::error!("VLM Recognize failed: {:?}", e); }
                                let _ = tx.send(res.map_err(|e| e.to_string()));
                            }
                            VlmRequest::RecognizeStream { img_path, task, max_tokens, token_tx, done_tx } => {
                                let res = vlm.recognize_stream(&img_path, task, max_tokens, |token_text: &str| {
                                    let _ = token_tx.send(token_text.to_string());
                                });
                                if let Err(ref e) = res { tracing::error!("VLM RecognizeStream failed: {:?}", e); }
                                let _ = done_tx.send(res.map(|_| ()).map_err(|e| e.to_string()));
                            }
                        }
                    }
                });
            }).expect("Failed to spawn VLM thread");
            vlm_tx_opt_inner = Some(vlm_tx);
        }
        info!("VLM model routing established (type: {:?})", resolved_type);
        let eos_id = tokenizer.token_to_id("</s>").or_else(|| tokenizer.token_to_id("<end_of_turn>")) .or_else(|| tokenizer.token_to_id("<|end_of_sentence|>")) .unwrap_or(1);
        (None, tokenizer, vec![eos_id], chat_template, vlm_tx_opt_inner, gemma4_vlm_tx_opt_inner, None, None)
    } else {
        // Only one of the TTS/ASR/VLM/LLM branches runs per process, so each is
        // the sole long-lived consumer of candle's process-wide rayon pool.
        let mut backend = engine::model_factory::create_backend(model_type, &args.model_path, &device, &dtype, format, args.quant.as_deref())?;
        info!("Model loaded successfully (type: {:?}, format: {:?})", resolved_type, format);
        // Install candle's affinity-pinned rayon pool so warmup's forward passes run on warm threads.
        device.with_context(|| backend.warmup());
        info!("Model warmed up");
        let tokenizer = backend.tokenizer().clone();
        let eos_token_id = backend.eos_token_id();
        let chat_template = engine::model_factory::create_chat_template(model_type, &args.model_path);
        let mut memory_config = MemoryConfig::parse(args.max_seq_len, args.gpu_memory_limit.as_deref(), &device);
        memory_config.record_baseline(&device);
        let baseline_gpu = memory_config.baseline_gpu_bytes;
        info!("Memory config: max_seq_len={}, gpu_limit={}, baseline_gpu={}", if memory_config.max_seq_len == 0 { "unlimited".to_string() } else { memory_config.max_seq_len.to_string() }, if memory_config.gpu_memory_limit_bytes == 0 { "unlimited".to_string() } else { format_bytes(memory_config.gpu_memory_limit_bytes) }, format_bytes(baseline_gpu));
        let (engine, handle) = InferenceEngine::new(backend, args.max_concurrent, args.decode_tokens_per_seq, memory_config);
        std::thread::Builder::new().name("inference-engine".into()).spawn(move || engine.run()).expect("Failed to spawn engine thread");
        info!("Inference engine started (max_concurrent={}, decode_tokens_per_seq={})", args.max_concurrent, args.decode_tokens_per_seq);
        (Some(handle), tokenizer, eos_token_id, chat_template, None, None, None, None)
    };

    let model_name = args.model_name.clone().unwrap_or_else(|| {
        std::path::Path::new(&args.model_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| resolved_type.display_name().to_string())
    });
    let gpu_memory_limit_display = args.gpu_memory_limit.clone().unwrap_or_else(|| "unlimited".to_string());
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
        asr_tx: asr_tx_opt,
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
    info!(version = env!("CARGO_PKG_VERSION"), listen = %format!("http://{local_addr}"), "crane-serve ready");
    info!(model = %model_name, model_type = %resolved_type.display_name(), device = %state.device_name, dtype = %state.dtype_name, "model loaded");
    if is_vlm {
        info!("mode: vlm");
    } else if is_tts {
        info!("mode: tts");
    } else if is_asr {
        info!("mode: asr");
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

/// Maximum accepted size for `/v1/audio/transcriptions` uploads, matching the
/// OpenAI transcription API's limit. Axum's default body limit (2 MiB) is far
/// too small for real audio files.
const MAX_TRANSCRIPTION_UPLOAD_BYTES: usize = 25 * 1024 * 1024;

pub fn build_router(state: Arc<AppState>) -> Router {
    let transcriptions_router = Router::new()
        .route("/v1/audio/transcriptions", post(handlers::asr::transcriptions))
        .layer(DefaultBodyLimit::max(MAX_TRANSCRIPTION_UPLOAD_BYTES));

    Router::new()
        .route("/health", get(handlers::common::health))
        .route("/v1/stats", get(handlers::common::stats))
        .route("/v1/chat/completions", post(handlers::openai::chat_completions))
        .route("/v1/completions", post(handlers::openai::completions))
        .route("/v1/audio/speech", post(handlers::tts::speech))
        .merge(transcriptions_router)
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

#[cfg(test)]
mod dtype_tests {
    use super::*;
    use crane_core::models::{DType, Device};

    #[test]
    fn explicit_flag_wins() {
        let d = Device::Cpu;
        assert_eq!(resolve_dtype(Some("f16"), &d, ModelType::Qwen3).unwrap(), DType::F16);
        assert_eq!(resolve_dtype(Some("BF16"), &d, ModelType::Qwen3_5).unwrap(), DType::BF16);
        assert_eq!(resolve_dtype(Some("fp32"), &d, ModelType::Qwen3_5).unwrap(), DType::F32);
        assert!(resolve_dtype(Some("int8"), &d, ModelType::Qwen3).is_err());
    }

    #[test]
    fn cpu_defaults_to_f32() {
        let d = Device::Cpu;
        assert_eq!(resolve_dtype(None, &d, ModelType::Qwen3_5).unwrap(), DType::F32);
        assert_eq!(resolve_dtype(None, &d, ModelType::Qwen3).unwrap(), DType::F32);
    }

    #[test]
    fn metal_defaults_f16_only_for_qwen3_5() {
        let Ok(d) = Device::new_metal(0) else {
            return; // no Metal on this machine/CI
        };
        assert_eq!(resolve_dtype(None, &d, ModelType::Qwen3_5).unwrap(), DType::F16);
        assert_eq!(resolve_dtype(None, &d, ModelType::Qwen3).unwrap(), DType::F32);
        assert_eq!(resolve_dtype(None, &d, ModelType::Gemma4).unwrap(), DType::F32);
    }
}
