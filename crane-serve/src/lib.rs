pub mod chat_template;
pub mod engine;
pub mod handlers;
pub mod openai_api;
pub mod reasoning;
pub mod sglang_api;
pub mod tools;
pub mod ui;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::{
    Router,
    extract::DefaultBodyLimit,
    http::StatusCode,
    response::Json,
    routing::{get, post},
};
use clap::Parser;
use crane_core::utils::DeviceExt;
use tracing::info;

use chat_template::ChatTemplateProcessor;
use engine::model_factory::{ModelFormat, ModelType};
use engine::{EngineHandle, InferenceEngine, MemoryConfig};
use handlers::asr::AsrTranscribeRequest;
use handlers::tts::TtsGenerateRequest;
use handlers::vlm::{Gemma4VlmRequest, MinicpmVVlmRequest, Qwen3_5VlmRequest, VlmRequest};
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
    /// Serve Crane's built-in browser UI at `/`. Disabled by default.
    #[arg(long)]
    pub ui: bool,
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
    /// F16 on ROCm and Metal, and F32 on CPU.
    #[arg(long)]
    pub dtype: Option<String>,
    #[arg(long, default_value_t = 0)]
    pub max_seq_len: usize,
    /// GPU memory budget: either a fraction of total VRAM (`0.9`), an absolute
    /// size (`8G`, `8GB`, `8GiB`, `5120M`, `5120MiB` — all binary units), or a
    /// plain byte count. Unset or `0` means unlimited. Only enforced for LLM
    /// engine mode (not TTS/ASR/VLM/duplex).
    #[arg(long)]
    pub gpu_memory_limit: Option<String>,
    /// MiniCPM-o duplex only: load the LLM tower from a standalone
    /// quantized GGUF file (e.g. a llama.cpp-style Qwen3 conversion like
    /// `MiniCPM-o-4_5-Q8_0.gguf`) instead of the checkpoint's own bf16
    /// safetensors weights, cutting the LLM's VRAM footprint roughly in
    /// half — the other five towers still load from `-m`'s checkpoint
    /// directory as usual. `-m` must still point at a real checkpoint
    /// directory (tokenizer/config and the other towers are read from
    /// there regardless).
    #[arg(long)]
    pub llm_gguf: Option<String>,
    /// Qwen 3.5-VL / Ornith only: load the checkpoint as a plain text model
    /// instead of a VLM, even though `config.json` declares a `vision_config`
    /// (and `--model-type` is `auto` or `qwen3_5_vl`). The vision tower's
    /// weights are simply never read — same checkpoint directory, no extra
    /// VRAM for the ~600M-param ViT — and this path also unlocks `--quant`,
    /// which the VLM load path does not support. Models are vision-capable by
    /// default; this is an opt-out, not the default.
    #[arg(long)]
    pub text_only: bool,
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
    pub qwen3_5_vlm_tx: Option<tokio::sync::mpsc::UnboundedSender<Qwen3_5VlmRequest>>,
    pub minicpm_v_vlm_tx: Option<tokio::sync::mpsc::UnboundedSender<MinicpmVVlmRequest>>,
    pub tts_tx: Option<tokio::sync::mpsc::UnboundedSender<TtsGenerateRequest>>,
    /// Channel to the ASR engine thread; `None` unless an ASR model is loaded.
    pub asr_tx: Option<tokio::sync::mpsc::UnboundedSender<AsrTranscribeRequest>>,
    /// Channel to the duplex engine thread; `None` unless a MiniCPM-o
    /// duplex model is loaded.
    pub duplex_tx: Option<tokio::sync::mpsc::UnboundedSender<handlers::duplex::DuplexRequest>>,
    /// Exclusivity guard for `/v1/audio/duplex` — only one live session at
    /// a time (see `handlers::duplex`'s module doc). Always constructed
    /// (even when no duplex model is loaded), just never contended in that
    /// case.
    pub duplex_lock: Arc<tokio::sync::Mutex<()>>,
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
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_file(false)
        .with_line_number(false)
        // Allows RUST_LOG=crane_core=trace,crane_serve=info
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
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
    tracing::debug!(
        "TTS encode: converting output tensor {:?} to f32",
        audio.dims()
    );
    let audio_f32 = audio
        .to_dtype(candle_core::DType::F32)
        .map_err(|e| e.to_string())?
        .flatten_all()
        .map_err(|e| e.to_string())?;
    tracing::debug!(
        "TTS encode: copying {} samples out of the tensor",
        audio_f32.elem_count()
    );
    let samples = audio_f32.to_vec1::<f32>().map_err(|e| e.to_string())?;
    tracing::info!("TTS writing {} samples", samples.len());
    match format {
        openai_api::AudioResponseFormat::Wav => {
            tracing::debug!("TTS encode: building WAV container");
            let wav_bytes =
                crane::audio::encode_wav(&samples, audio_info).map_err(|e| e.to_string())?;
            tracing::debug!(
                "TTS encode: WAV container built ({} bytes)",
                wav_bytes.len()
            );
            Ok(handlers::tts::TtsResult {
                audio_bytes: wav_bytes,
                content_type: "audio/wav",
                file_name: "speech.wav".to_string(),
                sample_rate: audio_info.sample_rate,
            })
        },
        openai_api::AudioResponseFormat::Pcm => {
            tracing::debug!("TTS encode: converting samples to PCM16");
            let pcm = crane::audio::pcm_f32_to_i16(&samples);
            tracing::debug!("TTS encode: PCM16 built ({} bytes)", pcm.len());
            Ok(handlers::tts::TtsResult {
                audio_bytes: pcm,
                content_type: "audio/pcm",
                file_name: "speech.pcm".to_string(),
                sample_rate: audio_info.sample_rate,
            })
        },
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
        cfm_steps: req.cfm_steps,
        cfg_scale: req.cfg_scale,
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
        let started = std::time::Instant::now();
        let result = tts
            .generate_voice_clone(&req.input, &req.language, ref_audio_path, ref_text, &opts)
            .map_err(|e| format!("{e:#}"));
        log_generate_result(&result, started.elapsed());
        result
    } else {
        let started = std::time::Instant::now();
        let result = tts
            .generate_speech(&req.input, &req.language, req.voice.as_deref(), &opts)
            .map_err(|e| format!("{e:#}"));
        log_generate_result(&result, started.elapsed());
        result
    }
}

/// Logs the shape/dtype of a successful [`generate_audio`] result (or the
/// fact that it errored) plus how long the model call took, so a crash that
/// aborts the process before a response is ever sent still leaves a trace of
/// whether the model call itself returned before things went wrong.
fn log_generate_result(result: &Result<candle_core::Tensor, String>, elapsed: std::time::Duration) {
    match result {
        Ok(tensor) => tracing::debug!(
            "TTS model call returned {:?} ({:?}) in {elapsed:?}",
            tensor.dims(),
            tensor.dtype(),
        ),
        Err(e) => tracing::debug!("TTS model call failed in {elapsed:?}: {e}"),
    }
}

fn run_tts_loop(
    mut tts_rx: tokio::sync::mpsc::UnboundedReceiver<TtsGenerateRequest>,
    model_name: &str,
    tts: &mut dyn crane::audio::Tts,
) {
    info!("{model_name} engine thread started");
    let audio_info = tts.audio_info();
    while let Some(mut req) = tts_rx.blocking_recv() {
        tracing::debug!(
            "TTS request received: language={}, voice={:?}, input_len={}, stream={}",
            req.language,
            req.voice,
            req.input.chars().count(),
            matches!(
                req.responder,
                Some(handlers::tts::TtsResponder::Stream { .. })
            ),
        );
        match req.responder.take().expect("responder set on the wire") {
            handlers::tts::TtsResponder::Whole(tx) => {
                let result = generate_audio(tts, model_name, &req).and_then(|audio| {
                    let encoded = encode_tts_audio(&audio, &audio_info, &req.response_format);
                    tracing::debug!("TTS: dropping output tensor {:?}", audio.dims());
                    drop(audio);
                    tracing::debug!("TTS: output tensor dropped");
                    encoded
                });
                tracing::debug!("TTS: result ready ({}), sending to client", result.is_ok());
                if let Err(ref e) = result {
                    tracing::error!(
                        "TTS generation failed: {e} (language={}, voice={:?}, input_len={})",
                        req.language,
                        req.voice,
                        req.input.chars().count()
                    );
                }
                let _ = tx.send(result);
            },
            handlers::tts::TtsResponder::Stream { meta, chunks } => {
                stream_tts(tts, model_name, &req, meta, chunks);
            },
        }
        tracing::debug!("TTS: request handled, waiting for next request");
    }
}

/// Drive [`crane::audio::Tts::generate_speech_stream`] for one request,
/// pushing PCM16-LE frames onto `chunks` as the model produces them. `meta`
/// fires exactly once — the sample rate on success, or a setup error — so the
/// HTTP handler can choose a status code before the 200 body starts.
fn stream_tts(
    tts: &mut dyn crane::audio::Tts,
    model_name: &str,
    req: &TtsGenerateRequest,
    meta: tokio::sync::oneshot::Sender<Result<u32, String>>,
    chunks: tokio::sync::mpsc::UnboundedSender<Result<Vec<u8>, String>>,
) {
    let opts = crane_core::generation::SpeechOptions {
        max_new_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        repetition_penalty: req.repetition_penalty,
        cfm_steps: req.cfm_steps,
        cfg_scale: req.cfg_scale,
    };
    let started = std::time::Instant::now();
    let mut stream =
        match tts.generate_speech_stream(&req.input, &req.language, req.voice.as_deref(), &opts) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("TTS stream setup failed: {e:#}");
                let _ = meta.send(Err(format!("{e:#}")));
                return;
            },
        };
    let sample_rate = stream.audio_info.sample_rate;
    if meta.send(Ok(sample_rate)).is_err() {
        return; // client already gone
    }

    let mut n_chunks = 0usize;
    loop {
        match stream.next_chunk() {
            Ok(Some(tensor)) => {
                let samples: Vec<f32> = match tensor.flatten_all().and_then(|t| t.to_vec1()) {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = chunks.send(Err(e.to_string()));
                        break;
                    },
                };
                let pcm = crane::audio::pcm_f32_to_i16(&samples);
                n_chunks += 1;
                if chunks.send(Ok(pcm)).is_err() {
                    tracing::debug!("{model_name} TTS stream: client disconnected, stopping");
                    break;
                }
            },
            Ok(None) => break,
            Err(e) => {
                tracing::error!("{model_name} TTS stream failed after {n_chunks} chunks: {e:#}");
                let _ = chunks.send(Err(format!("{e:#}")));
                break;
            },
        }
    }
    tracing::debug!(
        "{model_name} TTS stream done: {n_chunks} chunks in {:?}",
        started.elapsed()
    );
}

fn transcribe_audio(
    asr: &mut dyn crane::audio::Asr,
    req: &AsrTranscribeRequest,
) -> Result<String, String> {
    let sample_rate = asr.input_sample_rate();
    let samples =
        crane::audio::decode_wav(&req.audio_bytes, sample_rate).map_err(|e| e.to_string())?;

    let defaults = crane_core::generation::TranscribeOptions::default();
    let opts = crane_core::generation::TranscribeOptions {
        temperature: req.temperature.unwrap_or(defaults.temperature),
        language: req.language.clone(),
        ..defaults
    };
    asr.transcribe(&samples, &opts)
        .map(|t| t.text)
        .map_err(|e| e.to_string())
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

/// Base64 of 16-bit little-endian PCM, matching this crate's other audio
/// encodings (`crane::audio::pcm_f32_to_i16`) and the wire format
/// documented in `handlers::duplex`'s module doc.
fn encode_pcm16_base64(samples: &[f32]) -> String {
    use base64::Engine as _;
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        bytes.extend_from_slice(&(clamped * i16::MAX as f32).to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn run_duplex_loop(
    mut duplex_rx: tokio::sync::mpsc::UnboundedReceiver<handlers::duplex::DuplexRequest>,
    session: &mut crane_core::models::minicpmo::DuplexSession,
) {
    use handlers::duplex::{DuplexChunkEvent, DuplexRequest};

    info!("MiniCPM-o duplex engine thread started");
    let mut chunk_seed: u64 = 0;
    while let Some(req) = duplex_rx.blocking_recv() {
        match req {
            DuplexRequest::Prepare { system_prompt, tx } => {
                let result = session
                    .prepare(system_prompt.as_deref())
                    .map_err(|e| e.to_string());
                if let Err(ref e) = result {
                    tracing::error!("Duplex prepare failed: {e}");
                }
                let _ = tx.send(result);
            },
            DuplexRequest::Chunk { samples, tx } => {
                let result = (|| -> Result<DuplexChunkEvent, String> {
                    session
                        .streaming_prefill(&samples)
                        .map_err(|e| e.to_string())?;
                    chunk_seed = chunk_seed.wrapping_add(1);
                    let out = session
                        .streaming_generate(chunk_seed)
                        .map_err(|e| e.to_string())?;
                    let (audio_base64, audio_sample_rate) = match out.audio_waveform {
                        Some(waveform) => (Some(encode_pcm16_base64(&waveform)), Some(24_000)),
                        None => (None, None),
                    };
                    Ok(DuplexChunkEvent {
                        is_listen: out.is_listen,
                        text: out.text,
                        end_of_turn: out.end_of_turn,
                        audio_base64,
                        audio_sample_rate,
                    })
                })();
                if let Err(ref e) = result {
                    tracing::error!("Duplex chunk processing failed: {e}");
                }
                let _ = tx.send(result);
            },
        }
    }
}

/// Returns `true` if `device` is a real GPU backend (CUDA or ROCm) rather
/// than CPU or a backend (e.g. Metal) without fused-kernel/device-side
/// sampling support.
///
/// Centralizes the `is_cuda() || is_rocm()` check so call sites don't
/// hand-roll a `#[cfg(feature = "cuda")]`-only version of it, which
/// silently forces CPU on ROCm builds (the bug this replaced in the
/// TTS/ASR/duplex/VLM device-selection code below).
pub(crate) fn is_gpu_device(device: &crane_core::models::Device) -> bool {
    device.is_cuda() || device.is_rocm() || device.is_sycl()
}

/// Resolve the compute dtype. An explicit `--dtype` always wins; otherwise
/// BF16 on CUDA, F16 on ROCm and Metal, and F32 on CPU. Metal's F16 path
/// substantially reduces model and KV-cache memory use, including for
/// Qwen3-ASR; pass `--dtype f32` to explicitly prefer full precision.
fn resolve_dtype(
    flag: Option<&str>,
    device: &crane_core::models::Device,
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
    // ROCm backend is experimental: F16 has the broadest kernel coverage on candle's
    // rocm path today, whereas BF16 support is still incomplete. Default there.
    if device.is_rocm() {
        return Ok(DType::F16);
    }
    // SYCL backend is a proof-of-concept: F16/BF16/F32 all work; default to F16
    // to match the ROCm/Metal memory trade-off.
    if device.is_sycl() {
        return Ok(DType::F16);
    }
    if device.is_metal() {
        return Ok(DType::F16);
    }
    Ok(DType::F32)
}

fn apply_text_only_override(
    text_only: bool,
    model_type: ModelType,
    resolved_type: ModelType,
) -> (ModelType, ModelType) {
    if text_only && resolved_type == ModelType::Qwen3_5VL {
        (ModelType::Qwen3_5, ModelType::Qwen3_5)
    } else {
        (model_type, resolved_type)
    }
}

pub async fn run(args: Args) -> Result<()> {
    info!("Loading model from: {}", args.model_path);

    let device = if args.cpu {
        crane_core::models::Device::Cpu
    } else {
        // Exactly one backend is selected, in priority order: cuda → rocm → metal → cpu.
        // cuda and rocm are mutually exclusive builds; the cfg gates below never overlap.
        #[cfg(feature = "cuda")]
        {
            crane_core::models::Device::cuda_if_available(0)?
        }
        #[cfg(all(not(feature = "cuda"), feature = "rocm"))]
        {
            // Fall back to CPU when no AMD GPU is present, mirroring the metal idiom.
            crane_core::models::Device::new_rocm(0).unwrap_or(crane_core::models::Device::Cpu)
        }
        #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), feature = "sycl"))]
        {
            // Intel oneAPI / SYCL (proof-of-concept). Fall back to CPU when no
            // usable SYCL device is found.
            crane_core::models::Device::new_sycl(0).unwrap_or(crane_core::models::Device::Cpu)
        }
        #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), not(feature = "sycl")))]
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

    if args.text_only && resolved_type == ModelType::Qwen3_5VL {
        info!(
            "--text-only: loading Qwen 3.5-VL checkpoint as text-only (vision tower weights not read)"
        );
    }
    let (model_type, resolved_type) =
        apply_text_only_override(args.text_only, model_type, resolved_type);

    let mut dtype = resolve_dtype(args.dtype.as_deref(), &device)?;

    let is_vlm = resolved_type.is_vlm();
    let is_tts = resolved_type.is_tts();
    let is_asr = resolved_type.is_asr();
    let is_duplex = resolved_type.is_duplex();

    // Qwen3-TTS's autoregressive sampler is numerically unstable in F16 on
    // Metal: a non-finite logit reaches WeightedIndex and surfaces as
    // "A weight is negative, too large or not a valid number". Keep an
    // explicit user choice intact, but make the safe precision the default.
    if is_tts && args.dtype.is_none() && device.is_metal() {
        dtype = candle_core::DType::F32;
        info!("TTS on Metal: using F32 for numerically stable sampling");
    }

    let device_name = format!("{:?}", device);
    let dtype_name = format!("{:?}", dtype);
    info!("Device: {}, dtype: {}", device_name, dtype_name);

    // The memory gate lives in the LLM engine's scheduler; the one-shot
    // TTS/ASR/VLM/duplex paths have no admission point to enforce it at yet.
    // Warn instead of silently accepting the flag.
    if (is_tts || is_asr || is_vlm || is_duplex) && args.gpu_memory_limit.is_some() {
        tracing::warn!(
            "--gpu-memory-limit is only enforced for LLM engine mode; ignored for this model type"
        );
    }

    let (
        engine_handle,
        tokenizer,
        eos_token_id,
        chat_template,
        vlm_tx_opt,
        gemma4_vlm_tx_opt,
        qwen3_5_vlm_tx_opt,
        minicpm_v_vlm_tx_opt,
        tts_tx_opt,
        asr_tx_opt,
        duplex_tx_opt,
    ): (
        Option<EngineHandle>,
        tokenizers::Tokenizer,
        Vec<u32>,
        Box<dyn ChatTemplateProcessor>,
        Option<tokio::sync::mpsc::UnboundedSender<VlmRequest>>,
        Option<tokio::sync::mpsc::UnboundedSender<Gemma4VlmRequest>>,
        Option<tokio::sync::mpsc::UnboundedSender<Qwen3_5VlmRequest>>,
        Option<tokio::sync::mpsc::UnboundedSender<MinicpmVVlmRequest>>,
        Option<tokio::sync::mpsc::UnboundedSender<TtsGenerateRequest>>,
        Option<tokio::sync::mpsc::UnboundedSender<AsrTranscribeRequest>>,
        Option<tokio::sync::mpsc::UnboundedSender<handlers::duplex::DuplexRequest>>,
    ) = if is_tts {
        info!(
            "Loading TTS model ({:?}) from: {}",
            resolved_type, args.model_path
        );
        let model_path_clone = args.model_path.clone();
        // `device` already resolves cuda -> rocm -> metal -> cpu (and honors
        // `args.cpu`); re-deriving CPU-vs-GPU here only checked the `cuda`
        // feature, so it silently forced CPU on rocm/metal builds.
        let tts_device = device.clone();
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
                    },
                };
                // Install candle's affinity-pinned rayon pool for this thread's lifetime.
                tts_device.with_context(|| {
                    run_tts_loop(tts_rx, &resolved_name, tts.as_mut());
                });
            })
            .expect("Failed to spawn TTS thread");
        info!("TTS model routing established (type: {:?})", resolved_type);
        let tokenizer =
            crane_core::utils::tokenizer_utils::load_tokenizer_from_model_dir(&args.model_path)
                .unwrap_or_else(|e| {
                    tracing::warn!(
                        "Failed to load HF tokenizer: {e}; creating stub for TTS-only mode"
                    );
                    tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default())
                });
        let eos_id = tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(2);
        let chat_template =
            engine::model_factory::create_chat_template(model_type, &args.model_path);
        (
            None,
            tokenizer,
            vec![eos_id],
            chat_template,
            None,
            None,
            None,
            None,
            Some(tts_tx),
            None,
            None,
        )
    } else if is_asr {
        info!(
            "Loading ASR model ({:?}) from: {}",
            resolved_type, args.model_path
        );
        let model_path_clone = args.model_path.clone();
        // See the `tts_device` comment above: reuse the already-resolved
        // top-level device instead of a CUDA-only re-check.
        let asr_device = device.clone();
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
                    },
                };
                // Install candle's affinity-pinned rayon pool for this thread's lifetime.
                asr_device.with_context(|| {
                    run_asr_loop(asr_rx, &resolved_name, asr.as_mut());
                });
            })
            .expect("Failed to spawn ASR thread");
        info!("ASR model routing established (type: {:?})", resolved_type);
        let tokenizer =
            crane_core::utils::tokenizer_utils::load_tokenizer_from_model_dir(&args.model_path)
                .unwrap_or_else(|e| {
                    tracing::warn!(
                        "Failed to load HF tokenizer: {e}; creating stub for ASR-only mode"
                    );
                    tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default())
                });
        let eos_id = tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(2);
        let chat_template =
            engine::model_factory::create_chat_template(model_type, &args.model_path);
        (
            None,
            tokenizer,
            vec![eos_id],
            chat_template,
            None,
            None,
            None,
            None,
            None,
            Some(asr_tx),
            None,
        )
    } else if is_vlm {
        // Only the PaddleOCR-VL fallback branch below actually consumes
        // `use_cpu`/`use_bf16` (the other VLM variants use `device.clone()`
        // directly); `PaddleOcrVL::from_local` re-derives a CUDA-only
        // device internally, so `use_cpu` here doesn't yet get ROCm/Metal
        // GPU acceleration, but it at least reflects the real device
        // instead of unconditionally forcing CPU on ROCm builds.
        let use_cpu = args.cpu || !is_gpu_device(&device);
        let use_bf16 = device.is_cuda();
        let tok_path = std::path::Path::new(&args.model_path).join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;
        let chat_template =
            engine::model_factory::create_chat_template(model_type, &args.model_path);
        let mut vlm_tx_opt_inner: Option<tokio::sync::mpsc::UnboundedSender<VlmRequest>> = None;
        let mut gemma4_vlm_tx_opt_inner: Option<
            tokio::sync::mpsc::UnboundedSender<Gemma4VlmRequest>,
        > = None;
        let mut qwen3_5_vlm_tx_opt_inner: Option<
            tokio::sync::mpsc::UnboundedSender<Qwen3_5VlmRequest>,
        > = None;
        let mut minicpm_v_vlm_tx_opt_inner: Option<
            tokio::sync::mpsc::UnboundedSender<MinicpmVVlmRequest>,
        > = None;
        if resolved_type == engine::model_factory::ModelType::MinicpmV46 {
            info!("Loading MiniCPM-V-4.6 model from: {}", args.model_path);
            let model_path_clone = args.model_path.clone();
            let device_clone = device.clone();
            let dtype_clone = dtype;
            let (mcpv_tx, mut mcpv_rx) =
                tokio::sync::mpsc::unbounded_channel::<MinicpmVVlmRequest>();
            std::thread::Builder::new()
                .name("minicpm-v-vlm-engine".into())
                .spawn(move || {
                    use crane_core::models::minicpm_v::{MinicpmV46VLModel, VlGenerationConfig};
                    let mut vlm = match MinicpmV46VLModel::new(
                        &model_path_clone,
                        &device_clone,
                        &dtype_clone,
                    ) {
                        Ok(m) => m,
                        Err(e) => {
                            tracing::error!("Failed to load MiniCPM-V-4.6 model: {e}");
                            return;
                        },
                    };
                    info!("MiniCPM-V-4.6 engine thread started");
                    while let Some(req) = mcpv_rx.blocking_recv() {
                        let MinicpmVVlmRequest {
                            img_path,
                            text_prompt,
                            max_tokens,
                            token_tx,
                            tx,
                        } = req;
                        let res = (|| -> anyhow::Result<String> {
                            let cfg = VlGenerationConfig {
                                max_new_tokens: max_tokens,
                                ..Default::default()
                            };
                            let started = std::time::Instant::now();
                            let img = img_path.as_ref().map(image::open).transpose()?;
                            let out = vlm.generate(img.as_ref(), &text_prompt, &cfg, |token| {
                                if let Some(tx) = &token_tx {
                                    let _ = tx.send(token.to_string());
                                }
                            })?;
                            tracing::info!(
                                "MiniCPM-V-4.6 request completed in {:?}",
                                started.elapsed()
                            );
                            Ok(out)
                        })();
                        if let Err(ref e) = res {
                            tracing::error!("MiniCPM-V-4.6 request failed: {e}");
                        }
                        let _ = tx.send(res.map_err(|e| e.to_string()));
                    }
                })
                .expect("Failed to spawn MiniCPM-V-4.6 thread");
            minicpm_v_vlm_tx_opt_inner = Some(mcpv_tx);
        } else if resolved_type == engine::model_factory::ModelType::Qwen3_5VL {
            info!("Loading Qwen 3.5 VL model from: {}", args.model_path);
            let model_path_clone = args.model_path.clone();
            let device_clone = device.clone();
            let dtype_clone = dtype;
            let (q35vlm_tx, mut q35vlm_rx) =
                tokio::sync::mpsc::unbounded_channel::<Qwen3_5VlmRequest>();
            std::thread::Builder::new()
                .name("qwen3_5-vlm-engine".into())
                .spawn(move || {
                    use crane_core::models::qwen3_5::{Qwen3_5VLModel, VlGenerationConfig};
                    let mut vlm =
                        match Qwen3_5VLModel::new(&model_path_clone, &device_clone, &dtype_clone) {
                            Ok(m) => m,
                            Err(e) => {
                                tracing::error!("Failed to load Qwen 3.5 VL model: {e}");
                                return;
                            },
                        };
                    info!("Qwen 3.5 VL engine thread started");
                    while let Some(req) = q35vlm_rx.blocking_recv() {
                        let Qwen3_5VlmRequest {
                            img_path,
                            text_prompt,
                            max_tokens,
                            token_tx,
                            tx,
                        } = req;
                        let res = (|| -> anyhow::Result<String> {
                            let cfg = VlGenerationConfig {
                                max_new_tokens: max_tokens,
                                ..Default::default()
                            };
                            let started = std::time::Instant::now();
                            let img = img_path.as_ref().map(image::open).transpose()?;
                            let out = vlm.generate(img.as_ref(), &text_prompt, &cfg, |token| {
                                if let Some(tx) = &token_tx {
                                    let _ = tx.send(token.to_string());
                                }
                            })?;
                            tracing::info!(
                                "Qwen 3.5 VL request completed in {:?}",
                                started.elapsed()
                            );
                            Ok(out)
                        })();
                        if let Err(ref e) = res {
                            tracing::error!("Qwen 3.5 VL request failed: {e}");
                        }
                        let _ = tx.send(res.map_err(|e| e.to_string()));
                    }
                })
                .expect("Failed to spawn Qwen 3.5 VL thread");
            qwen3_5_vlm_tx_opt_inner = Some(q35vlm_tx);
        } else if resolved_type == engine::model_factory::ModelType::Gemma4VL {
            info!("Loading Gemma4 VLM model from: {}", args.model_path);
            let model_path_clone = args.model_path.clone();
            let device_clone = device.clone();
            let dtype_clone = dtype;
            let (g4vlm_tx, mut g4vlm_rx) =
                tokio::sync::mpsc::unbounded_channel::<Gemma4VlmRequest>();
            std::thread::Builder::new()
                .name("gemma4-vlm-engine".into())
                .spawn(move || {
                    use crane_core::models::gemma4::vision::{
                        ImagePreprocessConfig, load_and_preprocess_image,
                    };
                    use crane_core::models::gemma4::vlm::Gemma4VLModel;
                    let mut vlm =
                        match Gemma4VLModel::new(&model_path_clone, &device_clone, &dtype_clone) {
                            Ok(m) => m,
                            Err(e) => {
                                tracing::error!("Failed to load Gemma4 VLM model: {e}");
                                return;
                            },
                        };
                    info!("Gemma4 VLM engine thread started");
                    let preprocess_config = ImagePreprocessConfig::default();
                    // Install candle's affinity-pinned rayon pool for this thread's lifetime.
                    device_clone.with_context(|| {
                        while let Some(req) = g4vlm_rx.blocking_recv() {
                            let Gemma4VlmRequest {
                                img_path,
                                text_prompt,
                                max_tokens,
                                tx,
                            } = req;
                            let res = (|| -> anyhow::Result<String> {
                                let preprocessed = load_and_preprocess_image(
                                    &img_path,
                                    &preprocess_config,
                                    &device_clone,
                                )?;
                                let image_embeds = vlm.encode_image(
                                    &preprocessed.pixel_values,
                                    &preprocessed.pixel_position_ids,
                                    &preprocessed.padding_positions,
                                )?;
                                let image_token_id = 258880u32;
                                let mut prompt_ids: Vec<u32> = vec![2, 105, 2364, 107, 255999];
                                for _ in 0..preprocessed.num_image_tokens {
                                    prompt_ids.push(image_token_id);
                                }
                                prompt_ids.push(258882);
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
                                prompt_ids.extend_from_slice(&[106, 107, 105, 4368, 107]);
                                vlm.clear_kv_cache();
                                let input_tensor =
                                    candle_core::Tensor::new(prompt_ids.as_slice(), &device_clone)?
                                        .unsqueeze(0)?;
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
                                    if next_token == 1 || next_token == 106 {
                                        break;
                                    }
                                    let input =
                                        candle_core::Tensor::new(&[next_token], &device_clone)?
                                            .unsqueeze(0)?;
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
                                Ok(vlm
                                    .tokenizer
                                    .tokenizer
                                    .decode(&generated, true)
                                    .unwrap_or_default())
                            })();
                            let _ = tx.send(res.map_err(|e| e.to_string()));
                        }
                    });
                })
                .expect("Failed to spawn Gemma4 VLM thread");
            gemma4_vlm_tx_opt_inner = Some(g4vlm_tx);
        } else {
            info!("Loading VLM model (PaddleOCR-VL) from: {}", args.model_path);
            let model_path_clone = args.model_path.clone();
            let (vlm_tx, mut vlm_rx) = tokio::sync::mpsc::unbounded_channel::<VlmRequest>();
            std::thread::Builder::new()
                .name("vlm-engine".into())
                .spawn(move || {
                    let mut vlm = match engine::model_factory::create_vlm_model(
                        &model_path_clone,
                        use_cpu,
                        use_bf16,
                    ) {
                        Ok(m) => m,
                        Err(e) => {
                            tracing::error!("Failed to load VLM model: {e}");
                            return;
                        },
                    };
                    info!("VLM engine thread started");
                    // Clone device: with_context borrows &self, which would overlap the &mut vlm borrows below.
                    let vlm_device = vlm.device.clone();
                    // Install candle's affinity-pinned rayon pool for this thread's lifetime.
                    vlm_device.with_context(|| {
                        while let Some(req) = vlm_rx.blocking_recv() {
                            match req {
                                VlmRequest::Recognize {
                                    img_path,
                                    task,
                                    max_tokens,
                                    tx,
                                } => {
                                    let res =
                                        vlm.recognize(&img_path, task, max_tokens).map(|r| r.text);
                                    if let Err(ref e) = res {
                                        tracing::error!("VLM Recognize failed: {:?}", e);
                                    }
                                    let _ = tx.send(res.map_err(|e| e.to_string()));
                                },
                                VlmRequest::RecognizeStream {
                                    img_path,
                                    task,
                                    max_tokens,
                                    token_tx,
                                    done_tx,
                                } => {
                                    let res = vlm.recognize_stream(
                                        &img_path,
                                        task,
                                        max_tokens,
                                        |token_text: &str| {
                                            let _ = token_tx.send(token_text.to_string());
                                        },
                                    );
                                    if let Err(ref e) = res {
                                        tracing::error!("VLM RecognizeStream failed: {:?}", e);
                                    }
                                    let _ =
                                        done_tx.send(res.map(|_| ()).map_err(|e| e.to_string()));
                                },
                            }
                        }
                    });
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
        (
            None,
            tokenizer,
            vec![eos_id],
            chat_template,
            vlm_tx_opt_inner,
            gemma4_vlm_tx_opt_inner,
            qwen3_5_vlm_tx_opt_inner,
            minicpm_v_vlm_tx_opt_inner,
            None,
            None,
            None,
        )
    } else if is_duplex {
        info!("Loading MiniCPM-o duplex model from: {}", args.model_path);
        let model_path_clone = args.model_path.clone();
        // See the `tts_device` comment above: reuse the already-resolved
        // top-level device instead of a CUDA-only re-check.
        let duplex_device = device.clone();
        let duplex_dtype = dtype;
        let llm_gguf_clone = args.llm_gguf.clone();
        if let Some(ref gguf) = llm_gguf_clone {
            info!("MiniCPM-o duplex: loading LLM tower from GGUF: {gguf}");
        }
        let (duplex_tx, duplex_rx) =
            tokio::sync::mpsc::unbounded_channel::<handlers::duplex::DuplexRequest>();
        std::thread::Builder::new()
            .name("duplex-engine".into())
            .spawn(move || {
                let session_result = if let Some(gguf) = llm_gguf_clone {
                    crane_core::models::minicpmo::DuplexSession::new_with_llm_gguf(
                        &model_path_clone,
                        &gguf,
                        &duplex_device,
                        duplex_dtype,
                        crane_core::models::minicpmo::DuplexConfig::default(),
                    )
                } else {
                    crane_core::models::minicpmo::DuplexSession::new(
                        &model_path_clone,
                        &duplex_device,
                        duplex_dtype,
                        crane_core::models::minicpmo::DuplexConfig::default(),
                    )
                };
                let mut session = match session_result {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::error!("Failed to load MiniCPM-o duplex session: {e}");
                        return;
                    },
                };
                duplex_device.with_context(|| {
                    run_duplex_loop(duplex_rx, &mut session);
                });
            })
            .expect("Failed to spawn duplex thread");
        info!("MiniCPM-o duplex model routing established");
        let tokenizer =
            crane_core::utils::tokenizer_utils::load_tokenizer_from_model_dir(&args.model_path)
                .unwrap_or_else(|e| {
                    tracing::warn!(
                        "Failed to load HF tokenizer: {e}; creating stub for duplex-only mode"
                    );
                    tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default())
                });
        let eos_id = tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(2);
        let chat_template =
            engine::model_factory::create_chat_template(model_type, &args.model_path);
        (
            None,
            tokenizer,
            vec![eos_id],
            chat_template,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(duplex_tx),
        )
    } else {
        // Only one of the TTS/ASR/VLM/LLM branches runs per process, so each is
        // the sole long-lived consumer of candle's process-wide rayon pool.
        let mut backend = engine::model_factory::create_backend(
            model_type,
            &args.model_path,
            &device,
            &dtype,
            format,
            args.quant.as_deref(),
        )?;
        info!(
            "Model loaded successfully (type: {:?}, format: {:?})",
            resolved_type, format
        );
        // Install candle's affinity-pinned rayon pool so warmup's forward passes run on warm threads.
        device.with_context(|| backend.warmup());
        info!("Model warmed up");
        let tokenizer = backend.tokenizer().clone();
        let eos_token_id = backend.eos_token_id();
        let chat_template =
            engine::model_factory::create_chat_template(model_type, &args.model_path);
        let mut memory_config =
            MemoryConfig::parse(args.max_seq_len, args.gpu_memory_limit.as_deref(), &device);
        memory_config.record_baseline(&device);
        let baseline_gpu = memory_config.baseline_gpu_bytes;
        info!(
            "Memory config: max_seq_len={}, gpu_limit={}, baseline_gpu={}",
            if memory_config.max_seq_len == 0 {
                "unlimited".to_string()
            } else {
                memory_config.max_seq_len.to_string()
            },
            if memory_config.gpu_memory_limit_bytes == 0 {
                "unlimited".to_string()
            } else {
                format_bytes(memory_config.gpu_memory_limit_bytes)
            },
            format_bytes(baseline_gpu)
        );
        let (engine, handle) = InferenceEngine::new(
            backend,
            args.max_concurrent,
            args.decode_tokens_per_seq,
            memory_config,
        );
        std::thread::Builder::new()
            .name("inference-engine".into())
            .spawn(move || engine.run())
            .expect("Failed to spawn engine thread");
        info!(
            "Inference engine started (max_concurrent={}, decode_tokens_per_seq={})",
            args.max_concurrent, args.decode_tokens_per_seq
        );
        (
            Some(handle),
            tokenizer,
            eos_token_id,
            chat_template,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
    };

    let model_name = args.model_name.clone().unwrap_or_else(|| {
        std::path::Path::new(&args.model_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| resolved_type.display_name().to_string())
    });
    let gpu_memory_limit_display = args
        .gpu_memory_limit
        .clone()
        .unwrap_or_else(|| "unlimited".to_string());
    let state = Arc::new(AppState {
        engine: engine_handle,
        model_name: model_name.clone(),
        tokenizer,
        chat_template,
        eos_token_id,
        server_start_time: now_epoch(),
        vlm_tx: vlm_tx_opt,
        gemma4_vlm_tx: gemma4_vlm_tx_opt,
        qwen3_5_vlm_tx: qwen3_5_vlm_tx_opt,
        minicpm_v_vlm_tx: minicpm_v_vlm_tx_opt,
        tts_tx: tts_tx_opt,
        asr_tx: asr_tx_opt,
        duplex_tx: duplex_tx_opt,
        duplex_lock: Arc::new(tokio::sync::Mutex::new(())),
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
    let app = build_router_with_ui(state.clone(), args.ui);
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
    } else if is_duplex {
        info!("mode: duplex");
        info!(duplex_ws = %format!("ws://{local_addr}/v1/audio/duplex"), "duplex endpoint");
    } else {
        info!(
            max_concurrent = args.max_concurrent,
            decode_tokens_per_seq = args.decode_tokens_per_seq,
            "scheduler configured"
        );
        if args.max_seq_len > 0 || state.gpu_memory_limit != "unlimited" {
            let seq_str = if args.max_seq_len == 0 {
                "unlimited".to_string()
            } else {
                args.max_seq_len.to_string()
            };
            info!(max_seq_len = %seq_str, gpu_memory_limit = %state.gpu_memory_limit, "memory limits configured");
        }
    }
    info!(chat_completions = %format!("http://{local_addr}/v1/chat/completions"), models = %format!("http://{local_addr}/v1/models"), health = %format!("http://{local_addr}/health"), "api endpoints");
    if args.ui {
        info!(ui = %format!("http://{local_addr}/"), "browser UI enabled");
    }
    axum::serve(listener, app).await?;
    Ok(())
}

/// Maximum accepted size for `/v1/audio/transcriptions` uploads, matching the
/// OpenAI transcription API's limit. Axum's default body limit (2 MiB) is far
/// too small for real audio files.
const MAX_TRANSCRIPTION_UPLOAD_BYTES: usize = 25 * 1024 * 1024;

/// Build the API-only router, retaining the pre-UI behavior for library users.
pub fn build_router(state: Arc<AppState>) -> Router {
    build_router_with_ui(state, false)
}

/// Build the API router, optionally adding the browser UI routes.
pub fn build_router_with_ui(state: Arc<AppState>, ui_enabled: bool) -> Router {
    let transcriptions_router = Router::new()
        .route(
            "/v1/audio/transcriptions",
            post(handlers::asr::transcriptions),
        )
        .layer(DefaultBodyLimit::max(MAX_TRANSCRIPTION_UPLOAD_BYTES));

    let app = Router::new()
        .route("/health", get(handlers::common::health))
        .route("/v1/stats", get(handlers::common::stats))
        .route(
            "/v1/chat/completions",
            post(handlers::openai::chat_completions),
        )
        .route("/v1/completions", post(handlers::openai::completions))
        .route("/v1/audio/speech", post(handlers::tts::speech))
        .route("/v1/audio/duplex", get(handlers::duplex::duplex_ws))
        .merge(transcriptions_router)
        .route("/v1/models", get(handlers::openai::list_models))
        .route(
            "/v1/models/{model_id}",
            get(handlers::openai::retrieve_model),
        )
        .route("/v1/tokenize", post(handlers::openai::tokenize))
        .route("/v1/detokenize", post(handlers::openai::detokenize))
        .route("/tokenize", post(handlers::openai::tokenize))
        .route("/detokenize", post(handlers::openai::detokenize))
        .route("/generate", post(handlers::sglang::generate))
        .route("/model_info", get(handlers::sglang::model_info))
        .route("/server_info", get(handlers::sglang::server_info))
        .route("/health_generate", get(handlers::sglang::health_generate))
        .route(
            "/flush_cache",
            get(handlers::sglang::flush_cache).post(handlers::sglang::flush_cache),
        )
        .route("/abort_request", post(handlers::sglang::abort_request));

    let app = if ui_enabled {
        app.route("/", get(ui::index))
            .route("/ui/config", get(ui::config))
            .route("/ui/assets/{*path}", get(ui::asset))
    } else {
        app
    };

    app.with_state(state)
}

#[cfg(test)]
mod dtype_tests {
    use super::*;
    use crane_core::models::{DType, Device};

    #[test]
    fn explicit_flag_wins() {
        let d = Device::Cpu;
        assert_eq!(resolve_dtype(Some("f16"), &d).unwrap(), DType::F16);
        assert_eq!(resolve_dtype(Some("BF16"), &d).unwrap(), DType::BF16);
        assert_eq!(resolve_dtype(Some("fp32"), &d).unwrap(), DType::F32);
        assert!(resolve_dtype(Some("int8"), &d).is_err());
    }

    #[test]
    fn cpu_defaults_to_f32() {
        let d = Device::Cpu;
        assert_eq!(resolve_dtype(None, &d).unwrap(), DType::F32);
    }

    #[test]
    fn metal_defaults_to_f16() {
        let Ok(Ok(d)) = std::panic::catch_unwind(|| Device::new_metal(0)) else {
            return; // no usable Metal device in this process/CI
        };
        assert_eq!(resolve_dtype(None, &d).unwrap(), DType::F16);
    }

    // ── --text-only override ──

    #[test]
    fn text_only_downgrades_qwen3_5_vl_to_text() {
        let (mt, rt) = apply_text_only_override(true, ModelType::Qwen3_5VL, ModelType::Qwen3_5VL);
        assert_eq!(mt, ModelType::Qwen3_5);
        assert_eq!(rt, ModelType::Qwen3_5);
    }

    #[test]
    fn text_only_downgrades_auto_resolved_qwen3_5_vl() {
        // Mirrors the auto-detect case: model_type is still Auto (never
        // re-detected), resolved_type is what detect_model_type() returned.
        let (mt, rt) = apply_text_only_override(true, ModelType::Auto, ModelType::Qwen3_5VL);
        assert_eq!(mt, ModelType::Qwen3_5);
        assert_eq!(rt, ModelType::Qwen3_5);
    }

    #[test]
    fn text_only_is_noop_without_the_flag() {
        let (mt, rt) = apply_text_only_override(false, ModelType::Qwen3_5VL, ModelType::Qwen3_5VL);
        assert_eq!(mt, ModelType::Qwen3_5VL);
        assert_eq!(rt, ModelType::Qwen3_5VL);
    }

    #[test]
    fn text_only_is_noop_for_other_vlm_types() {
        // Scoped to Qwen 3.5-VL / Ornith only — other VLM families aren't
        // known to share a checkpoint layout with a text-only sibling.
        let (mt, rt) = apply_text_only_override(true, ModelType::MinicpmV46, ModelType::MinicpmV46);
        assert_eq!(mt, ModelType::MinicpmV46);
        assert_eq!(rt, ModelType::MinicpmV46);
    }
}
