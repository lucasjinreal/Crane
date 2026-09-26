pub mod auth;
use crane_core::candle_core;
use crane_core::{D, DType, Tensor, softmax_last_dim};

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

use anyhow::{Context, Result};
use axum::{
    Router,
    extract::DefaultBodyLimit,
    http::StatusCode,
    middleware,
    response::Json,
    routing::{get, post},
};
use clap::Parser;
use crane_core::utils::DeviceExt;
use tracing::{info, warn};

use chat_template::ChatTemplateProcessor;
use crane_core::device::DeviceAssignment;
use engine::backend::ExpertPromotionPolicy;
use engine::model_factory::{ModelFormat, ModelType};
use engine::{EngineHandle, InferenceEngine, KV_GPU_OVERHEAD_FACTOR, MemoryConfig};
use handlers::asr::AsrTranscribeRequest;
use handlers::tts::TtsGenerateRequest;
use handlers::vlm::{Gemma4VlmRequest, MinicpmVVlmRequest, Qwen3_5VlmRequest, VlmRequest};
use openai_api::ErrorResponse;

#[derive(Parser, Debug, Clone)]
#[command(
    about = "OpenAI & SGLang compatible API server with continuous batching",
    version
)]
pub struct Args {
    /// Path to a downloaded model directory or a single `.gguf` file.
    #[arg(short = 'm', long, help_heading = "Model")]
    pub model_path: String,
    /// Model architecture. Usually auto-detected from the model's
    /// `config.json`; set this only if auto-detection picks the wrong one.
    /// Known values: `auto`, `gemma4`, `gemma4_vl`, `hunyuan`, `minicpm5`,
    /// `minicpmv46`, `minicpmo`, `qwen25`, `qwen3`, `qwen3_5`, `qwen3_5_vl`,
    /// `qwen3_tts`, `voxtral_tts`, `kokoro`, `voxcpm2`, `paddleocr_vl`,
    /// `qwen3_asr`.
    #[arg(long, default_value = "auto", help_heading = "Model")]
    pub model_type: String,
    /// Display name reported by `/v1/models`. Defaults to the directory name
    /// of `--model-path` when unset.
    #[arg(long, help_heading = "Model")]
    pub model_name: Option<String>,
    /// Directory containing built-in VoxCPM2 reference audio files. Each
    /// filename stem (without `.wav`, `.mp3`, etc.) becomes a `voice` value.
    /// Embeddings are cached in this directory after the first startup.
    #[arg(long, default_value = "data/voices", help_heading = "Model")]
    pub voice_dir: std::path::PathBuf,
    /// Listen address. Default `0.0.0.0` (all interfaces). Use `127.0.0.1`
    /// to restrict to localhost.
    #[arg(long, default_value = "0.0.0.0", help_heading = "Server")]
    pub host: String,
    /// Listen port. Default `8080`.
    #[arg(short = 'p', long, default_value_t = 8080, help_heading = "Server")]
    pub port: u16,
    /// Serve HTTP/1.1 over a Unix domain socket at this path instead of TCP.
    /// A stale socket file left by a crashed run is removed before binding,
    /// and the new socket is created with 0600 permissions. Unix only.
    #[cfg(unix)]
    #[arg(long, help_heading = "Server")]
    pub unix_socket: Option<std::path::PathBuf>,
    /// Serve Crane's built-in browser UI at `/`. Disabled by default.
    #[arg(long, help_heading = "Server")]
    pub ui: bool,
    /// Log verbosity filter. Accepts a bare level (`debug`, `info`, `warn`)
    /// or comma-separated per-target filters (`info,crane_core=debug`).
    /// Crate names use underscores, not hyphens. Overrides `RUST_LOG` when
    /// both are set. Default: `RUST_LOG`, or `info` if that is also unset.
    #[arg(long, help_heading = "Server")]
    pub log_level: Option<String>,
    /// Force CPU-only inference, ignoring any available GPU.
    #[arg(long, help_heading = "Memory")]
    pub cpu: bool,
    /// Maximum number of requests that can be processed at the same time.
    /// Default `16`. Increase for higher throughput; decrease if you are
    /// running out of memory.
    #[arg(short = 'c', long, default_value_t = 16, help_heading = "Scheduler")]
    pub max_concurrent: usize,
    /// Tokens generated per request before switching to the next one.
    /// Default `16`. Higher values speed up individual requests but slow
    /// down others running at the same time.
    #[arg(long, default_value_t = 16, help_heading = "Scheduler")]
    pub decode_tokens_per_seq: usize,
    /// Model file format: `auto`, `safetensors`, or `gguf`. Default `auto`
    /// (detected from files in the model directory).
    #[arg(long, default_value = "auto", help_heading = "Model")]
    pub format: String,
    /// Quantize the model on load to reduce memory usage (e.g. `q4k`,
    /// `q8_0`). Only supported for Qwen 3.5 models with safetensors
    /// weights. Overrides `CRANE_ISQ`.
    #[arg(long, help_heading = "Model")]
    pub quant: Option<String>,
    /// Floating-point precision for inference: `f16` (half), `bf16`
    /// (bfloat16), or `f32` (full). Lower precision uses less memory and
    /// is faster. Default: `bf16` on NVIDIA GPUs, `f16` on AMD/Apple GPUs
    /// (except `Qwen3-ASR`, which defaults to `f32` on AMD), `f32` on CPU.
    #[arg(long, help_heading = "Model")]
    pub dtype: Option<String>,
    /// Maximum sequence length in tokens. Default `0` (unlimited). See
    /// `--context` for a human-readable alternative (e.g. `128K`).
    #[arg(long, default_value_t = 0, help_heading = "Scheduler")]
    pub max_seq_len: usize,
    /// Maximum context length as a human-readable token count. Accepts K
    /// (x1024) and M (x1024^2) suffixes, e.g. `128K` = 131072 tokens.
    /// Mutually exclusive with `--max-seq-len`.
    #[arg(long, conflicts_with = "max_seq_len", help_heading = "Scheduler")]
    pub context: Option<String>,
    /// GPU memory budget: either a fraction of total VRAM (`0.9`), an absolute
    /// size (`8G`, `8GB`, `8GiB`, `5120M`, `5120MiB` — all binary units), or a
    /// plain byte count. Unset or `0` means unlimited. Only enforced for LLM
    /// engine mode (not TTS/ASR/VLM/duplex).
    #[arg(long, help_heading = "Memory")]
    pub gpu_memory_limit: Option<String>,
    /// Force all `MoE` expert weights to CPU regardless of
    /// `--gpu-memory-limit`. Only affects Qwen3 `MoE` checkpoints.
    #[arg(long, help_heading = "Memory")]
    pub offload_experts: bool,
    /// MiniCPM-o duplex only: load the LLM tower from a standalone
    /// quantized GGUF file (e.g. a llama.cpp-style Qwen3 conversion like
    /// `MiniCPM-o-4_5-Q8_0.gguf`) instead of the checkpoint's own bf16
    /// safetensors weights, cutting the LLM's VRAM footprint roughly in
    /// half — the other five towers still load from `-m`'s checkpoint
    /// directory as usual. `-m` must still point at a real checkpoint
    /// directory (tokenizer/config and the other towers are read from
    /// there regardless).
    #[arg(long, help_heading = "Model")]
    pub llm_gguf: Option<String>,
    /// Qwen 3.5-VL / Ornith only: disable the vision component and load
    /// only the text model. Saves ~600M parameters worth of GPU memory and
    /// enables `--quant`, which the vision-capable path does not support.
    #[arg(long, help_heading = "Model")]
    pub text_only: bool,
    /// API key required to access non-exempt endpoints (`/health`,
    /// `/v1/stats`, `/`, and `/ui/*` excluded). Repeatable to configure
    /// multiple valid keys. Pass with no value (`--api-key`) to generate a
    /// random key printed to stdout at startup. The `CRANE_API_KEY` env var
    /// always requires a value — key generation is CLI-only. Unset means
    /// open access (default).
    #[arg(
        long,
        env = "CRANE_API_KEY",
        num_args = 0..=1,
        default_missing_value = "",
        action = clap::ArgAction::Append,
        help_heading = "Authentication"
    )]
    pub api_key: Vec<String>,
    /// File with one API key per line; `#`-prefixed lines are comments.
    /// Combines with `--api-key`.
    #[arg(long, env = "CRANE_API_KEY_FILE", help_heading = "Authentication")]
    pub api_key_file: Option<String>,
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
    /// Active serving mode: `"llm"`, `"vlm"`, `"tts"`, `"asr"`, or `"duplex"`.
    /// Used to phrase error messages for endpoints unavailable in the
    /// current mode.
    pub mode: &'static str,
    pub dtype_name: String,
    pub device_name: String,
    pub host: String,
    pub port: u16,
    pub max_concurrent: usize,
    pub decode_tokens_per_seq: usize,
    pub max_seq_len: usize,
    pub gpu_memory_limit: String,
    /// Valid API keys. Empty means auth is disabled (open access).
    pub api_keys: Vec<String>,
}

pub fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Human-readable device label (e.g. `cuda:0`, `metal:0`, `cpu`) instead of
/// the Rust `Debug` repr of the underlying candle device handle.
fn format_device_name(device: &candle_core::Device) -> String {
    match device.location() {
        candle_core::DeviceLocation::Cpu => "cpu".to_string(),
        candle_core::DeviceLocation::Cuda { gpu_id } => format!("cuda:{gpu_id}"),
        candle_core::DeviceLocation::Metal { gpu_id } => format!("metal:{gpu_id}"),
        #[cfg(feature = "rocm")]
        candle_core::DeviceLocation::Rocm { gpu_id } => format!("rocm:{gpu_id}"),
        #[cfg(feature = "sycl")]
        candle_core::DeviceLocation::Sycl { gpu_id } => format!("sycl:{gpu_id}"),
    }
}

/// Best-effort CPU model name and physical/logical core counts, read from
/// `/proc/cpuinfo` on Linux. Physical cores are counted as the number of
/// unique (physical id, core id) pairs; platforms without that file (e.g.
/// macOS) or without those fields (some VMs) fall back to the logical count
/// for both, and "unknown" for the name.
fn cpu_info() -> (String, usize, usize) {
    let logical = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);

    let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") else {
        return ("unknown".to_string(), logical, logical);
    };

    let mut name = None;
    let mut cores: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    let mut cur_physical_id = None;
    let mut cur_core_id = None;
    for line in cpuinfo.lines() {
        match line.split_once(':') {
            Some((key, value)) => match key.trim() {
                "model name" if name.is_none() => name = Some(value.trim().to_string()),
                "physical id" => cur_physical_id = Some(value.trim().to_string()),
                "core id" => cur_core_id = Some(value.trim().to_string()),
                _ => {},
            },
            None if line.trim().is_empty() => {
                if let (Some(p), Some(c)) = (cur_physical_id.take(), cur_core_id.take()) {
                    cores.insert((p, c));
                }
            },
            None => {},
        }
    }
    if let (Some(p), Some(c)) = (cur_physical_id, cur_core_id) {
        cores.insert((p, c));
    }

    let physical = if cores.is_empty() {
        logical
    } else {
        cores.len()
    };
    (
        name.unwrap_or_else(|| "unknown".to_string()),
        physical,
        logical,
    )
}

/// Logs CPU (model name, physical/logical core count, candle thread pool
/// size) and, when running on a GPU device, total VRAM once at startup.
fn log_hardware_info(device: &candle_core::Device, device_name: &str) {
    let (cpu_name, physical_cores, logical_cores) = cpu_info();
    info!(
        cpu = %cpu_name,
        physical_cores,
        logical_cores,
        thread_pool_size = candle_core::utils::get_num_threads(),
        "hardware: cpu"
    );

    if !matches!(device.location(), candle_core::DeviceLocation::Cpu) {
        let (_, vram_total) = engine::memory::query_gpu_memory_usage(device);
        if vram_total > 0 {
            info!(
                device = %device_name,
                vram_total = %engine::memory::format_bytes_engine(vram_total),
                "hardware: gpu"
            );
        } else {
            info!(device = %device_name, "hardware: gpu (VRAM query unavailable)");
        }
    }
}

pub fn make_error(status: StatusCode, msg: &str) -> (StatusCode, Json<ErrorResponse>) {
    let error_type = if status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    (
        status,
        Json(ErrorResponse {
            error: openai_api::ErrorDetail {
                message: msg.to_string(),
                r#type: error_type.into(),
                code: None,
            },
        }),
    )
}

/// Generate a random API key. Uses a UUID v4, which packs 122 bits of
/// randomness — far beyond what's brute-forceable — into a copy-pasteable
/// string.
fn generate_api_key() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Merge CLI keys and key-file keys into a deduplicated list, preserving
/// first-seen order. Empty strings (from a bare `--api-key`) generate a
/// random key, printed to stdout so the operator can copy it.
fn load_api_keys(cli_keys: &[String], key_file: Option<&str>) -> Result<Vec<String>> {
    let mut keys: Vec<String> = Vec::new();

    for key in cli_keys {
        let resolved = if key.is_empty() {
            let generated = generate_api_key();
            // Deliberately bypasses `tracing`/`RUST_LOG`: this is the one
            // chance the operator has to capture the generated key, so it
            // must print unconditionally rather than depend on the active
            // log level.
            println!("Generated API key: {generated}");
            generated
        } else {
            key.clone()
        };
        if !keys.iter().any(|k| k == &resolved) {
            keys.push(resolved);
        }
    }

    if let Some(path) = key_file {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("failed to read API key file {path}: {e}"))?;
        for line in contents.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !keys.iter().any(|k| k == trimmed)
            {
                keys.push(trimmed.to_string());
            }
        }
    }

    Ok(keys)
}

/// Mask all but the last 4 characters of an API key for safe logging, e.g.
/// `sk-mysecretkey` becomes `****tkey`. Keys of 4 chars or fewer are fully
/// masked.
fn mask_api_key(key: &str) -> String {
    let len = key.chars().count();
    if len <= 4 {
        "*".repeat(len)
    } else {
        let visible: String = key.chars().skip(len - 4).collect();
        format!("****{visible}")
    }
}

pub fn init_logging(log_level: Option<&str>) -> Result<()> {
    let filter = match log_level {
        Some(level) => tracing_subscriber::EnvFilter::try_new(level)
            .with_context(|| format!("invalid --log-level filter: {level}"))?,
        None => tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
    };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_file(false)
        .with_line_number(false)
        // Allows RUST_LOG=crane_core=trace,crane_serve=info
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .compact()
        .init();
    Ok(())
}

pub async fn cli_main() -> Result<()> {
    // Must run before anything touches SYCL; a no-op off `--features sycl`.
    crane_core::utils::sycl_env::ensure_sycl_runtime_env();

    let args = Args::parse();
    init_logging(args.log_level.as_deref())?;
    run(args).await
}

fn encode_tts_audio(
    audio: &Tensor,
    audio_info: &crane::audio::AudioInfo,
    format: &openai_api::AudioResponseFormat,
) -> Result<handlers::tts::TtsResult, String> {
    tracing::debug!(
        "TTS encode: converting output tensor {:?} to f32",
        audio.dims()
    );
    let audio_f32 = audio
        .to_dtype(DType::F32)
        .map_err(|e| e.to_string())?
        .flatten_all()
        .map_err(|e| e.to_string())?;
    tracing::debug!(
        "TTS encode: copying {} samples out of the tensor",
        audio_f32.elem_count()
    );
    let samples = audio_f32.to_vec1::<f32>().map_err(|e| e.to_string())?;
    tracing::debug!("TTS writing {} samples", samples.len());
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
) -> Result<Tensor, String> {
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
fn log_generate_result(result: &Result<Tensor, String>, elapsed: std::time::Duration) {
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
        info!(
            language = %req.language,
            voice = ?req.voice,
            input_len = req.input.chars().count(),
            stream = matches!(
                req.responder,
                Some(handlers::tts::TtsResponder::Stream { .. })
            ),
            "TTS request received",
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
/// substantially reduces model and KV-cache memory use; pass `--dtype f32`
/// to explicitly prefer full precision.
///
/// ROCm excludes Qwen3-ASR from its F16 default: the audio encoder's
/// intermediate activations overflow F16's smaller range (vs. the BF16 the
/// checkpoint was trained in), producing NaN/garbage logits that never
/// sample EOS and run decode out to `max_new_tokens` every time. Metal has
/// not been verified against this same failure mode and still defaults
/// Qwen3-ASR to F16.
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
    // ROCm backend is experimental: F16 has the broadest kernel coverage on candle's
    // rocm path today, whereas BF16 support is still incomplete. Default there.
    //
    // Qwen3-ASR is excluded: its audio encoder's intermediate activations
    // overflow F16's much smaller range (vs. the BF16 the checkpoint was
    // trained in), producing NaN/garbage logits that never sample EOS and
    // run decode out to `max_new_tokens` every time. F32 is the verified-safe
    // default for this family until it's been checked against F16 output
    // quality on this backend.
    if device.is_rocm() && model_type != ModelType::Qwen3ASR {
        return Ok(DType::F16);
    }
    // F16/BF16/F32 all work on SYCL; default to F16, as on ROCm/Metal.
    if device.is_sycl() {
        return Ok(DType::F16);
    }
    // TODO: Qwen3-ASR hasn't been verified on Metal; it may hit the same
    // F16 overflow as on ROCm and need the same exclusion here.
    if device.is_metal() {
        return Ok(DType::F16);
    }
    Ok(DType::F32)
}

/// Resolve `--gpu-memory-limit` / `--offload-experts` into an initial `MoE`
/// expert `Device` (`devices.expert`) and, when the real GPU/CPU split must
/// be decided after the model exists, an [`ExpertPromotionPolicy`] for the
/// caller to hand to [`crate::engine::backend::Qwen3Backend::new`].
///
/// - No GPU / `--cpu`, or `--offload-experts`: experts start (and stay) on
///   CPU, no promotion.
/// - `--gpu-memory-limit` set: experts start on CPU; the returned policy
///   tells the caller to attempt promoting them to GPU once the model
///   exists and real VRAM headroom is known.
/// - No limit, GPU present: experts load directly to the main GPU device,
///   no promotion needed.
fn resolve_expert_placement(
    gpu_memory_limit: Option<&str>,
    offload_experts: bool,
    device: &crane_core::models::Device,
    max_concurrent: usize,
    max_seq_len: usize,
) -> (DeviceAssignment, Option<ExpertPromotionPolicy>) {
    if !is_gpu_device(device) || offload_experts {
        return (
            DeviceAssignment {
                main: device.clone(),
                expert: crane_core::models::Device::Cpu,
            },
            None,
        );
    }
    let raw_limit = gpu_memory_limit.map_or(0, |s| MemoryConfig::parse_memory_limit(s, device));
    if raw_limit == 0 {
        return (DeviceAssignment::uniform(device), None);
    }
    (
        DeviceAssignment {
            main: device.clone(),
            expert: crane_core::models::Device::Cpu,
        },
        Some(ExpertPromotionPolicy {
            vram_ceiling_bytes: raw_limit,
            max_concurrent: Some(max_concurrent),
            max_seq_len: Some(max_seq_len),
        }),
    )
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

/// Parse a human-readable context size into a raw token count.
///
/// Accepts an optional `K` (x1024) or `M` (x1024^2) suffix (case-insensitive).
/// A plain integer is passed through unchanged. No fractional suffixes
/// (`1.5M`) — token counts are integers.
fn parse_context_size(s: &str) -> Result<usize> {
    let s = s.trim();
    anyhow::ensure!(!s.is_empty(), "context size must not be empty");

    let upper = s.to_ascii_uppercase();
    let (digits, multiplier) = if let Some(d) = upper.strip_suffix('M') {
        (d, 1024 * 1024)
    } else if let Some(d) = upper.strip_suffix('K') {
        (d, 1024)
    } else {
        (upper.as_str(), 1)
    };

    let n: usize = digits
        .trim()
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid context size '{s}': {e}"))?;
    anyhow::ensure!(n > 0, "context size must be greater than zero, got '{s}'");
    n.checked_mul(multiplier)
        .ok_or_else(|| anyhow::anyhow!("context size '{s}' overflows"))
}

/// Maximum time to wait for in-flight connections (including long-lived
/// WebSocket/SSE sessions) to drain after the first shutdown signal before
/// forcing exit.
const SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Waits for Ctrl-C or (on Unix) `SIGTERM`, then logs and returns so
/// `axum::serve`'s graceful shutdown can drain in-flight requests before exit.
/// Once the first signal fires, a repeat signal or `SHUTDOWN_TIMEOUT` elapsing
/// forces an immediate exit so a stuck connection can't hang the process.
async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            warn!("failed to install Ctrl-C handler: {e}");
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            },
            Err(e) => {
                warn!("failed to install SIGTERM handler: {e}");
                std::future::pending::<()>().await;
            },
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }

    info!(
        "Shutting down (repeat signal or wait {}s to force)...",
        SHUTDOWN_TIMEOUT.as_secs()
    );

    tokio::spawn(async {
        let ctrl_c = async {
            let _ = tokio::signal::ctrl_c().await;
        };

        #[cfg(unix)]
        let terminate = async {
            if let Ok(mut sig) =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            {
                sig.recv().await;
            } else {
                std::future::pending::<()>().await;
            }
        };
        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            () = ctrl_c => {
                info!("Second signal received, forcing exit");
            },
            () = terminate => {
                info!("Second signal received, forcing exit");
            },
            () = tokio::time::sleep(SHUTDOWN_TIMEOUT) => {
                warn!(
                    "Graceful shutdown timed out after {}s, forcing exit",
                    SHUTDOWN_TIMEOUT.as_secs()
                );
            },
        }
        std::process::exit(1);
    });
}

/// Auto-derives a safe `--max-seq-len` when the caller left it at `0`
/// (unlimited) while `--gpu-memory-limit` is set. Without this, a single
/// long-running session's own KV cache can grow past physical VRAM with
/// no runtime protection: eviction (`InferenceEngine::is_over_kv_budget`)
/// only preempts *other* competing sequences when a new prefill is
/// scheduled, never a lone session's own growth. The safe headroom is
/// divided evenly across `max_concurrent`, since every concurrent slot
/// could independently grow to the derived cap. Returns `None` when
/// derivation isn't applicable (an explicit `--max-seq-len` or no
/// `--gpu-memory-limit`) or isn't safely computable (VRAM query
/// unsupported for this device, or zero headroom).
fn derive_safe_max_seq_len(
    memory_config: &MemoryConfig,
    physical_total_bytes: u64,
    kv_bytes_per_token: u64,
    max_concurrent: usize,
) -> Option<usize> {
    if memory_config.max_seq_len != 0 || memory_config.gpu_memory_limit_bytes == 0 {
        return None;
    }
    if physical_total_bytes == 0 || kv_bytes_per_token == 0 {
        return None;
    }
    let ceiling = memory_config
        .gpu_memory_limit_bytes
        .min(physical_total_bytes);
    let headroom = ceiling.saturating_sub(memory_config.baseline_gpu_bytes);
    // Raw KV-tensor bytes are only ~15-20% of real GPU growth (padded
    // batch-decode copies, allocator retention, forward-pass intermediates
    // like chunked-prefill attention scores that scale with chunk_size ×
    // kv_len) — see `KV_GPU_OVERHEAD_FACTOR`'s doc comment. Reuse the same
    // empirically-measured factor here instead of a separate guess: an
    // earlier version of this function used an arbitrary 80% margin, which
    // was ~6x too generous and let a single session's own prefill blow
    // past physical VRAM in production.
    let safe_headroom = headroom / KV_GPU_OVERHEAD_FACTOR;
    let per_seq_budget = safe_headroom / max_concurrent.max(1) as u64;
    let derived = per_seq_budget / kv_bytes_per_token;
    if derived == 0 {
        return None;
    }
    // Token counts are bounded by realistic VRAM sizes divided by
    // per-token byte cost — always comfortably within usize range.
    #[allow(clippy::cast_possible_truncation)]
    let derived = derived as usize;
    Some(derived)
}

pub async fn run(mut args: Args) -> Result<()> {
    let model_path = std::path::Path::new(&args.model_path);
    if args.format.to_lowercase() == "gguf" || args.model_path.ends_with(".gguf") {
        anyhow::ensure!(
            model_path.is_file(),
            "--model-path '{}' is not a file. GGUF format requires a path to a .gguf file.",
            args.model_path
        );
    } else {
        anyhow::ensure!(
            model_path.is_dir(),
            "--model-path '{}' is not a directory. Provide the path to a model directory \
             (or use --format gguf for a single .gguf file).",
            args.model_path
        );
    }

    info!("Loading model from: {}", args.model_path);

    if let Some(ref ctx) = args.context {
        args.max_seq_len = parse_context_size(ctx)?;
        info!(
            "--context {ctx} resolved to max_seq_len={}",
            args.max_seq_len
        );
    }

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
    if model_type == ModelType::Auto && args.model_type.to_lowercase() != "auto" {
        warn!(
            input = %args.model_type,
            "Unrecognized --model-type, falling back to auto-detect. Known values: auto, \
             gemma4, gemma4_vl, hunyuan, minicpm5, minicpmv46, minicpmo, qwen25, qwen3, \
             qwen3_5, qwen3_5_vl, qwen3_tts, voxtral_tts, kokoro, voxcpm2, paddleocr_vl, \
             qwen3_asr"
        );
    }
    let format = ModelFormat::from_str(&args.format);
    if format == ModelFormat::Auto && args.format.to_lowercase() != "auto" {
        warn!(
            input = %args.format,
            "Unrecognized --format, falling back to auto-detect. Known values: auto, \
             safetensors, gguf"
        );
    }

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

    let mut dtype = resolve_dtype(args.dtype.as_deref(), &device, resolved_type)?;

    let is_vlm = resolved_type.is_vlm();
    let is_tts = resolved_type.is_tts();
    let is_asr = resolved_type.is_asr();
    let is_duplex = resolved_type.is_duplex();

    // Qwen3-TTS's autoregressive sampler is numerically unstable in F16 on
    // Metal: a non-finite logit reaches WeightedIndex and surfaces as
    // "A weight is negative, too large or not a valid number". Keep an
    // explicit user choice intact, but make the safe precision the default.
    if is_tts && args.dtype.is_none() && device.is_metal() {
        dtype = DType::F32;
        info!("TTS on Metal: using F32 for numerically stable sampling");
    }

    let device_name = format_device_name(&device);
    let dtype_name = format!("{:?}", dtype);
    log_hardware_info(&device, &device_name);

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
        let voice_dir_clone = args.voice_dir.clone();
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
                    Some(&voice_dir_clone),
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
                                    Tensor::new(prompt_ids.as_slice(), &device_clone)?
                                        .unsqueeze(0)?;
                                let logits = vlm
                                    .forward(&input_tensor, Some(&image_embeds), 0)?
                                    .squeeze(0)?
                                    .squeeze(0)?
                                    .to_dtype(DType::F32)?;
                                let mut tokens = prompt_ids.clone();
                                let mut generated = Vec::new();
                                let mut next_token = softmax_last_dim(&logits)?
                                    .argmax(D::Minus1)?
                                    .to_scalar::<u32>()?;
                                generated.push(next_token);
                                tokens.push(next_token);
                                for _ in 1..max_tokens {
                                    if next_token == 1 || next_token == 106 {
                                        break;
                                    }
                                    let input =
                                        Tensor::new(&[next_token], &device_clone)?.unsqueeze(0)?;
                                    let logits = vlm
                                        .forward(&input, None, tokens.len() - 1)?
                                        .squeeze(0)?
                                        .squeeze(0)?
                                        .to_dtype(DType::F32)?;
                                    next_token = softmax_last_dim(&logits)?
                                        .argmax(D::Minus1)?
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
        let (devices, promotion) = resolve_expert_placement(
            args.gpu_memory_limit.as_deref(),
            args.offload_experts,
            &device,
            args.max_concurrent,
            args.max_seq_len,
        );
        let mut backend = engine::model_factory::create_backend(
            model_type,
            &args.model_path,
            &devices,
            &dtype,
            format,
            args.quant.as_deref(),
            promotion.as_ref(),
        )?;
        info!(
            "Model loaded successfully (type: {}, format: {:?})",
            resolved_type.display_name(),
            format
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
        // This guard duplicates derive_safe_max_seq_len's own first-line check
        // intentionally, so that function stays independently callable (and
        // testable) without relying on the caller to have already checked.
        if memory_config.max_seq_len == 0 && memory_config.gpu_memory_limit_bytes > 0 {
            let physical_total = MemoryConfig::query_total_gpu_memory(&device);
            // The engine caps max_concurrent to 1 when the backend doesn't
            // support KV-cache swapping (see InferenceEngine::new). Use the
            // same effective value here so the derivation divides the VRAM
            // budget by the concurrency the engine will actually allow,
            // rather than the (potentially higher) CLI value.
            let effective_concurrent = if backend.supports_kv_swap() {
                args.max_concurrent
            } else {
                1
            };
            match backend.kv_bytes_per_token().and_then(|kv_bpt| {
                derive_safe_max_seq_len(
                    &memory_config,
                    physical_total,
                    kv_bpt,
                    effective_concurrent,
                )
            }) {
                Some(derived) => {
                    info!(
                        "max_seq_len unset with gpu_memory_limit set; auto-derived {derived} \
                         tokens from physical_vram={}, baseline={}",
                        engine::memory::format_bytes_engine(physical_total),
                        engine::memory::format_bytes_engine(memory_config.baseline_gpu_bytes),
                    );
                    memory_config.max_seq_len = derived;
                    args.max_seq_len = derived;
                    // The first expert-promotion pass (in create_backend) used
                    // max_seq_len=0, which fell back to max_position_embeddings
                    // for KV reservation — over-reserving VRAM and
                    // under-promoting experts. Re-run promotion with the
                    // tighter derived cap so the model can reclaim that
                    // headroom and promote additional MoE layers.
                    if let Some(ref promo) = promotion {
                        let updated = ExpertPromotionPolicy {
                            vram_ceiling_bytes: promo.vram_ceiling_bytes,
                            max_concurrent: Some(effective_concurrent),
                            max_seq_len: Some(derived),
                        };
                        backend.re_promote_experts(&updated)?;
                        // Baseline must be re-recorded: promoting more experts
                        // to GPU increases VRAM usage, and the engine's
                        // kv_budget_bytes() subtracts baseline from the
                        // limit. A stale (lower) baseline would over-estimate
                        // the KV budget, risking OOM.
                        memory_config.record_baseline(&device);
                    }
                },
                None => {
                    warn!(
                        "max_seq_len is unlimited (0) with gpu_memory_limit set; a single \
                         long-running session's KV cache can grow past VRAM with no runtime \
                         eviction protection (eviction only guards against multiple competing \
                         sequences). Could not auto-derive a safe cap for this model/device — \
                         set --max-seq-len explicitly."
                    );
                },
            }
        }
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
                engine::memory::format_bytes_engine(memory_config.gpu_memory_limit_bytes)
            },
            engine::memory::format_bytes_engine(baseline_gpu)
        );
        let (engine, handle) = InferenceEngine::new(
            backend,
            args.max_concurrent,
            args.decode_tokens_per_seq,
            memory_config,
            engine::model_factory::uses_xml_tool_format(&args.model_path),
        );
        std::thread::Builder::new()
            .name("inference-engine".into())
            .spawn(move || engine.run())
            .expect("Failed to spawn engine thread");
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
    let api_keys = load_api_keys(&args.api_key, args.api_key_file.as_deref())?;
    let mode = if is_vlm {
        "vlm"
    } else if is_tts {
        "tts"
    } else if is_asr {
        "asr"
    } else if is_duplex {
        "duplex"
    } else {
        "llm"
    };
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
        mode,
        dtype_name,
        device_name,
        host: args.host.clone(),
        port: args.port,
        max_concurrent: args.max_concurrent,
        decode_tokens_per_seq: args.decode_tokens_per_seq,
        max_seq_len: args.max_seq_len,
        gpu_memory_limit: gpu_memory_limit_display,
        api_keys: api_keys.clone(),
    });
    let app = build_router_with_ui(state.clone(), args.ui);
    let addr = format!("{}:{}", args.host, args.port);
    // The TCP and UDS listeners are distinct types behind axum's
    // `serve::Listener` trait, so the serve future is boxed to pick one.
    // `local_addr` keeps an authority/path-only form so the endpoint log
    // lines below stay unchanged for both modes.
    #[cfg(unix)]
    let (local_addr, listen_display, serve): (
        String,
        String,
        std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send>>,
    ) = match &args.unix_socket {
        Some(path) => {
            let _ = std::fs::remove_file(path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let listener = tokio::net::UnixListener::bind(path)?;
            std::fs::set_permissions(
                path,
                <std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o600),
            )
            .map_err(|e| anyhow::anyhow!("chmod 0600 {}: {e}", path.display()))?;
            (
                path.display().to_string(),
                format!("unix://{}", path.display()),
                Box::pin(async move {
                    axum::serve(listener, app)
                        .with_graceful_shutdown(shutdown_signal())
                        .await
                        .map_err(anyhow::Error::from)
                }),
            )
        },
        None => {
            let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
                if e.kind() == std::io::ErrorKind::AddrInUse {
                    anyhow::anyhow!(
                        "Failed to bind {addr}: port already in use. Try --port {}.",
                        args.port.wrapping_add(1)
                    )
                } else {
                    anyhow::anyhow!("Failed to bind {addr}: {e}")
                }
            })?;
            let local_addr = listener.local_addr()?;
            (
                local_addr.to_string(),
                format!("http://{local_addr}"),
                Box::pin(async move {
                    axum::serve(listener, app)
                        .with_graceful_shutdown(shutdown_signal())
                        .await
                        .map_err(anyhow::Error::from)
                }),
            )
        },
    };
    #[cfg(not(unix))]
    let (local_addr, listen_display, serve): (
        String,
        String,
        std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send>>,
    ) = {
        let listener = tokio::net::TcpListener::bind(&addr).await.map_err(|e| {
            if e.kind() == std::io::ErrorKind::AddrInUse {
                anyhow::anyhow!(
                    "Failed to bind {addr}: port already in use. Try --port {}.",
                    args.port.wrapping_add(1)
                )
            } else {
                anyhow::anyhow!("Failed to bind {addr}: {e}")
            }
        })?;
        let local_addr = listener.local_addr()?;
        (
            local_addr.to_string(),
            format!("http://{local_addr}"),
            Box::pin(async move {
                axum::serve(listener, app)
                    .with_graceful_shutdown(shutdown_signal())
                    .await
                    .map_err(anyhow::Error::from)
            }),
        )
    };
    info!(version = env!("CARGO_PKG_VERSION"), listen = %listen_display, "crane-serve ready");
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
    match api_keys.as_slice() {
        [] => {},
        [key] => info!(key = %mask_api_key(key), "API key authentication enabled"),
        keys => info!(count = keys.len(), "API key authentication enabled"),
    }
    if !api_keys.is_empty() && args.host == "0.0.0.0" {
        tracing::warn!(
            "API key configured but server binds 0.0.0.0 without TLS — keys are sent in cleartext over the network"
        );
    }
    if !api_keys.is_empty() && args.ui {
        tracing::warn!(
            "API key configured with --ui enabled — the built-in browser UI does not send an API key, so its requests will be rejected with 401"
        );
    }
    serve.await?;
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

    app.layer(middleware::from_fn_with_state(
        state.clone(),
        auth::require_api_key,
    ))
    .with_state(state)
}

#[cfg(test)]
mod auth_config_tests {
    use super::*;

    // Verifies CLI keys and key-file keys are merged, comments/blank lines
    // in the file are skipped, and duplicates across both sources are
    // removed while preserving first-seen order.
    #[test]
    fn load_api_keys_merges_and_dedups() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.txt");
        std::fs::write(&path, "# comment\n\nkey2\nkey3\nkey1\n").unwrap();

        let cli_keys = vec!["key1".to_string(), "key1".to_string()];
        let keys = load_api_keys(&cli_keys, Some(path.to_str().unwrap())).unwrap();

        assert_eq!(keys, vec!["key1", "key2", "key3"]);
    }

    // Verifies a bare `--api-key` (empty string) generates a random key
    // instead of being silently dropped.
    #[test]
    fn load_api_keys_generates_for_empty_string() {
        let keys = load_api_keys(&[String::new()], None).unwrap();
        assert_eq!(keys.len(), 1);
        assert!(uuid::Uuid::parse_str(&keys[0]).is_ok());
    }

    #[test]
    fn generate_api_key_is_a_valid_uuid() {
        let key = generate_api_key();
        assert!(uuid::Uuid::parse_str(&key).is_ok());
    }

    #[test]
    fn generate_api_key_is_unique_per_call() {
        assert_ne!(generate_api_key(), generate_api_key());
    }

    // Verifies an empty CLI key list with no key file yields no keys (open access).
    #[test]
    fn load_api_keys_empty_when_unconfigured() {
        let keys = load_api_keys(&[], None).unwrap();
        assert!(keys.is_empty());
    }

    // Verifies a missing key file surfaces an error instead of silently
    // producing no keys.
    #[test]
    fn load_api_keys_errors_on_missing_file() {
        assert!(load_api_keys(&[], Some("/nonexistent/path/keys.txt")).is_err());
    }

    #[test]
    fn mask_api_key_keeps_last_four_chars() {
        assert_eq!(mask_api_key("sk-mysecretkey"), "****tkey");
    }

    #[test]
    fn mask_api_key_fully_masks_short_keys() {
        assert_eq!(mask_api_key("abc"), "***");
        assert_eq!(mask_api_key(""), "");
    }

    // `--api-key` combines `num_args = 0..=1` with `ArgAction::Append`, an
    // unusual pairing — this pins down that repeated flags accumulate, a
    // bare flag yields one empty string (the generation trigger), and an
    // absent flag yields an empty vec (open access).
    #[test]
    fn api_key_flag_parses_repeat_bare_and_absent_forms() {
        let repeated = Args::parse_from([
            "crane-serve",
            "-m",
            "x",
            "--api-key",
            "k1",
            "--api-key",
            "k2",
        ]);
        assert_eq!(repeated.api_key, vec!["k1", "k2"]);

        let bare = Args::parse_from(["crane-serve", "-m", "x", "--api-key"]);
        assert_eq!(bare.api_key, vec![""]);

        let absent = Args::parse_from(["crane-serve", "-m", "x"]);
        assert!(absent.api_key.is_empty());
    }
}

/// Exercises `auth::require_api_key` wired into a real router via
/// [`build_router_with_ui`]'s own layering pattern, rather than just the
/// pure helper functions in `auth.rs` — this is what would catch a
/// regression like the layer being applied before routes are merged, or a
/// route being added without updating `auth::EXEMPT_PATHS`.
#[cfg(test)]
mod auth_middleware_tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    struct StubChatTemplate;

    impl ChatTemplateProcessor for StubChatTemplate {
        fn apply(&self, _messages: &[openai_api::ChatMessage]) -> Result<String, String> {
            Ok(String::new())
        }
    }

    fn test_app_state(api_keys: Vec<String>) -> Arc<AppState> {
        Arc::new(AppState {
            engine: None,
            model_name: "test-model".to_string(),
            tokenizer: tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            chat_template: Box::new(StubChatTemplate),
            eos_token_id: vec![0],
            server_start_time: now_epoch(),
            vlm_tx: None,
            gemma4_vlm_tx: None,
            qwen3_5_vlm_tx: None,
            minicpm_v_vlm_tx: None,
            tts_tx: None,
            asr_tx: None,
            duplex_tx: None,
            duplex_lock: Arc::new(tokio::sync::Mutex::new(())),
            model_path: "test".to_string(),
            model_type_name: "test".to_string(),
            mode: "llm",
            dtype_name: "f32".to_string(),
            device_name: "cpu".to_string(),
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_concurrent: 1,
            decode_tokens_per_seq: 1,
            max_seq_len: 0,
            gpu_memory_limit: "unlimited".to_string(),
            api_keys,
        })
    }

    async fn ok() -> StatusCode {
        StatusCode::OK
    }

    // Mirrors `build_router_with_ui`'s own `.layer(...).with_state(...)`
    // pattern with a handful of representative routes, instead of the full
    // production router, to keep the fixture independent of every handler's
    // dependencies.
    fn test_router(state: Arc<AppState>) -> Router {
        Router::new()
            .route("/health", get(ok))
            .route("/v1/stats", get(ok))
            .route("/v1/chat/completions", post(ok))
            .layer(middleware::from_fn_with_state(
                state.clone(),
                auth::require_api_key,
            ))
            .with_state(state)
    }

    async fn send(
        router: Router,
        method: &str,
        path: &str,
        auth_header: Option<(&str, &str)>,
    ) -> StatusCode {
        let mut builder = Request::builder().method(method).uri(path);
        if let Some((name, value)) = auth_header {
            builder = builder.header(name, value);
        }
        let request = builder.body(Body::empty()).unwrap();
        router.oneshot(request).await.unwrap().status()
    }

    #[tokio::test]
    async fn middleware_allows_exempt_paths_with_keys_configured() {
        let router = test_router(test_app_state(vec!["secret".to_string()]));
        assert_eq!(
            send(router.clone(), "GET", "/health", None).await,
            StatusCode::OK
        );
        assert_eq!(send(router, "GET", "/v1/stats", None).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn middleware_rejects_non_exempt_without_key() {
        let router = test_router(test_app_state(vec!["secret".to_string()]));
        assert_eq!(
            send(router, "POST", "/v1/chat/completions", None).await,
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test]
    async fn middleware_accepts_valid_bearer() {
        let router = test_router(test_app_state(vec!["secret".to_string()]));
        let status = send(
            router,
            "POST",
            "/v1/chat/completions",
            Some(("authorization", "Bearer secret")),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn middleware_passes_through_when_no_keys_configured() {
        let router = test_router(test_app_state(vec![]));
        assert_eq!(
            send(router, "POST", "/v1/chat/completions", None).await,
            StatusCode::OK
        );
    }
}

#[cfg(test)]
mod dtype_tests {
    use super::*;
    use crane_core::models::{DType, Device};

    #[test]
    fn explicit_flag_wins() {
        let d = Device::Cpu;
        assert_eq!(
            resolve_dtype(Some("f16"), &d, ModelType::Qwen3).unwrap(),
            DType::F16
        );
        assert_eq!(
            resolve_dtype(Some("BF16"), &d, ModelType::Qwen3).unwrap(),
            DType::BF16
        );
        assert_eq!(
            resolve_dtype(Some("fp32"), &d, ModelType::Qwen3).unwrap(),
            DType::F32
        );
        assert!(resolve_dtype(Some("int8"), &d, ModelType::Qwen3).is_err());
    }

    #[test]
    fn cpu_defaults_to_f32() {
        let d = Device::Cpu;
        assert_eq!(
            resolve_dtype(None, &d, ModelType::Qwen3).unwrap(),
            DType::F32
        );
    }

    #[test]
    fn metal_defaults_to_f16() {
        let Ok(Ok(d)) = std::panic::catch_unwind(|| Device::new_metal(0)) else {
            return; // no usable Metal device in this process/CI
        };
        assert_eq!(
            resolve_dtype(None, &d, ModelType::Qwen3).unwrap(),
            DType::F16
        );
    }

    #[test]
    #[cfg(feature = "rocm")]
    fn rocm_defaults_f32_for_qwen3_asr_only() {
        // Qwen3-ASR's audio encoder overflows F16's range (the checkpoint is
        // trained in BF16), so it must be excluded from ROCm's blanket F16
        // default while every other family keeps defaulting to F16 there.
        let Ok(d) = Device::new_rocm(0) else {
            return; // no ROCm device on this machine/CI
        };
        assert_eq!(
            resolve_dtype(None, &d, ModelType::Qwen3ASR).unwrap(),
            DType::F32
        );
        assert_eq!(
            resolve_dtype(None, &d, ModelType::Qwen3).unwrap(),
            DType::F16
        );
        assert_eq!(
            resolve_dtype(None, &d, ModelType::Qwen3_5).unwrap(),
            DType::F16
        );
        // An explicit --dtype flag must still override the exclusion.
        assert_eq!(
            resolve_dtype(Some("f16"), &d, ModelType::Qwen3ASR).unwrap(),
            DType::F16
        );
    }

    // ── resolve_expert_placement ──

    #[test]
    fn resolve_expert_placement_cpu_device_stays_cpu_no_promotion() {
        let d = Device::Cpu;
        let (devices, promotion) = resolve_expert_placement(Some("8G"), false, &d, 16, 4096);
        assert!(devices.expert.is_cpu());
        assert!(promotion.is_none());
    }

    #[test]
    fn resolve_expert_placement_offload_experts_forces_cpu_no_promotion() {
        let d = Device::Cpu;
        // A CPU device already forces CPU on its own; this asserts
        // --offload-experts is also checked independently of device kind
        // (relevant once a real GPU device reaches this branch).
        let (devices, promotion) = resolve_expert_placement(None, true, &d, 16, 4096);
        assert!(devices.expert.is_cpu());
        assert!(promotion.is_none());
    }

    #[test]
    fn resolve_expert_placement_no_limit_on_cpu_is_uniform_cpu() {
        let d = Device::Cpu;
        let (devices, promotion) = resolve_expert_placement(None, false, &d, 16, 4096);
        assert!(devices.main.is_cpu());
        assert!(devices.expert.is_cpu());
        assert!(promotion.is_none());
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

    // ── parse_context_size ──

    #[test]
    fn context_parses_bare_number() {
        assert_eq!(parse_context_size("4096").unwrap(), 4096);
    }

    #[test]
    fn context_parses_k_suffix() {
        assert_eq!(parse_context_size("128K").unwrap(), 131_072);
    }

    #[test]
    fn context_parses_m_suffix() {
        assert_eq!(parse_context_size("1M").unwrap(), 1_048_576);
    }

    #[test]
    fn context_parses_case_insensitively() {
        assert_eq!(parse_context_size("128k").unwrap(), 131_072);
        assert_eq!(parse_context_size("1m").unwrap(), 1_048_576);
    }

    #[test]
    fn context_rejects_empty_string() {
        assert!(parse_context_size("").is_err());
    }

    #[test]
    fn context_rejects_invalid_suffix() {
        assert!(parse_context_size("128G").is_err());
    }

    #[test]
    fn context_rejects_non_numeric() {
        assert!(parse_context_size("abc").is_err());
    }

    #[test]
    fn context_trims_whitespace() {
        assert_eq!(parse_context_size("  128K  ").unwrap(), 131_072);
    }

    #[test]
    fn context_rejects_zero() {
        assert!(parse_context_size("0").is_err());
        assert!(parse_context_size("0K").is_err());
    }

    #[test]
    fn context_rejects_overflow() {
        assert!(parse_context_size("99999999999999999999M").is_err());
    }

    #[test]
    fn context_rejects_negative() {
        assert!(parse_context_size("-128K").is_err());
    }

    #[test]
    fn context_cli_rejects_context_and_max_seq_len_together() {
        let result = Args::try_parse_from([
            "crane-serve",
            "-m",
            "/tmp/model",
            "--context",
            "128K",
            "--max-seq-len",
            "4096",
        ]);
        assert!(result.is_err());
    }

    // ── derive_safe_max_seq_len ──

    fn memory_config_for_test(
        max_seq_len: usize,
        limit_bytes: u64,
        baseline_bytes: u64,
    ) -> MemoryConfig {
        MemoryConfig {
            max_seq_len,
            gpu_memory_limit_bytes: limit_bytes,
            baseline_gpu_bytes: baseline_bytes,
        }
    }

    #[test]
    fn derive_max_seq_len_none_when_already_set() {
        // An explicit --max-seq-len must never be silently overridden.
        let cfg = memory_config_for_test(4096, 10 << 30, 1 << 30);
        assert!(derive_safe_max_seq_len(&cfg, 16 << 30, 1024, 1).is_none());
    }

    #[test]
    fn derive_max_seq_len_none_when_no_gpu_limit() {
        // No --gpu-memory-limit means genuinely unlimited — nothing to derive.
        let cfg = memory_config_for_test(0, 0, 1 << 30);
        assert!(derive_safe_max_seq_len(&cfg, 16 << 30, 1024, 1).is_none());
    }

    #[test]
    fn derive_max_seq_len_none_when_vram_query_unsupported() {
        // physical_total_bytes == 0 signals an unsupported device query
        // (e.g. Metal) — can't safely compute a bound.
        let cfg = memory_config_for_test(0, 10 << 30, 1 << 30);
        assert!(derive_safe_max_seq_len(&cfg, 0, 1024, 1).is_none());
    }

    #[test]
    fn derive_max_seq_len_none_when_backend_has_no_kv_cost() {
        let cfg = memory_config_for_test(0, 10 << 30, 1 << 30);
        assert!(derive_safe_max_seq_len(&cfg, 16 << 30, 0, 1).is_none());
    }

    #[test]
    fn derive_max_seq_len_none_when_no_headroom() {
        // Baseline already consumes the entire limit — zero room for KV.
        let cfg = memory_config_for_test(0, 10 << 30, 10 << 30);
        assert!(derive_safe_max_seq_len(&cfg, 16 << 30, 1024, 1).is_none());
    }

    #[test]
    fn derive_max_seq_len_computes_expected_value() {
        // Mirrors this session's real numbers: 16 GiB card, 10 GiB limit,
        // 4 GiB baseline, 96 KiB/token, max_concurrent=1,
        // KV_GPU_OVERHEAD_FACTOR=6. headroom=6 GiB, /6 -> 1 GiB safe budget
        // -> 1 GiB / 96 KiB = 10922 tokens (integer division).
        //
        // Notably this is well *below* the 47080-token real workload this
        // session debugged — with the correct overhead factor applied,
        // that workload genuinely does not fit safely in a 10G limit on
        // this card, it isn't just a smaller-than-expected safe cap. An
        // earlier, buggy version of this function used an 80% margin
        // instead of dividing by `KV_GPU_OVERHEAD_FACTOR`, which computed
        // 49152 — 4.5x too generous — and that value OOM'd in production.
        let cfg = memory_config_for_test(0, 10 << 30, 4 << 30);
        let kv_bytes_per_token = 96 * 1024;
        let derived =
            derive_safe_max_seq_len(&cfg, 16 << 30, kv_bytes_per_token, 1).expect("derived");
        assert_eq!(derived, 10_922);
    }

    #[test]
    fn derive_max_seq_len_divides_budget_across_max_concurrent() {
        // Non-power-of-two kv_bytes_per_token exercises floor division, and
        // absolute golden values catch regressions that a relational
        // assertion (single/4 == quad) cannot — that identity is a
        // mathematical tautology for integer division regardless of the
        // implementation, since floor(floor(a/b)/c) == floor(a/(b*c)) for
        // all positive integers.
        let cfg = memory_config_for_test(0, 10 << 30, 1 << 30);
        let single = derive_safe_max_seq_len(&cfg, 16 << 30, 100_000, 1).expect("derived");
        let quad = derive_safe_max_seq_len(&cfg, 16 << 30, 100_000, 4).expect("derived");
        assert_eq!(single, 16_106);
        assert_eq!(quad, 4_026);
    }

    #[test]
    fn derive_max_seq_len_clamps_to_physical_vram() {
        // A --gpu-memory-limit larger than the card's actual physical VRAM
        // (typo, or a shared/overcommitted device) must not let the
        // headroom calculation use the inflated configured limit.
        let cfg_inflated = memory_config_for_test(0, 20 << 30, 1 << 30);
        let cfg_matching = memory_config_for_test(0, 10 << 30, 1 << 30);
        let clamped = derive_safe_max_seq_len(&cfg_inflated, 10 << 30, 1024, 1).expect("derived");
        let unclamped = derive_safe_max_seq_len(&cfg_matching, 10 << 30, 1024, 1).expect("derived");
        assert_eq!(clamped, unclamped);
    }
}
