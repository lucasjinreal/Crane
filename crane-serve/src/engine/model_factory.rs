//! Model factory for automatic model type detection and backend creation.
//!
//! Supports auto-detection from `config.json`'s `model_type` / `architectures`
//! fields, or explicit model type specification via CLI.

#[cfg(feature = "onnx")]
use anyhow::Context;
use anyhow::Result;
use candle_core::{DType, Device};
use serde::Deserialize;
use std::path::Path;

use super::backend::{
    Gemma4Backend, HunyuanBackend, Minicpm5Backend, ModelBackend, Qwen25Backend, Qwen3Backend,
    Qwen3_5Backend,
};
use crate::chat_template::{AutoChatTemplate, ChatTemplateProcessor, HunyuanChatTemplate};

// ─────────────────────────────────────────────────────────────
//  Enums
// ─────────────────────────────────────────────────────────────

/// Supported model architectures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    Auto,
    Gemma4,
    Gemma4VL,
    HunyuanDense,
    Minicpm5,
    MinicpmV46,
    MiniCpmODuplex,
    Qwen25,
    Qwen3,
    Qwen3_5,
    Qwen3_5VL,
    Qwen3TTS,
    VoxtralTTS,
    Kokoro,
    VoxCpm2,
    PaddleOcrVl,
    Qwen3ASR,
}

impl ModelType {
    // Infallible convenience constructor, not the fallible std::str::FromStr
    // trait (unknown strings fall back to `Auto` instead of erroring).
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "gemma4" | "gemma-4" | "gemma4_e2b" => Self::Gemma4,
            "gemma4_vl" | "gemma4-vl" | "gemma4vl" => Self::Gemma4VL,
            "hunyuan" | "hunyuan_dense" | "hunyuandense" => Self::HunyuanDense,
            // Bare "minicpm" is aliased to MiniCPM5 for now, since it's the
            // only OpenBMB family member Crane supports; re-scope this once
            // MiniCPM-o/MiniCPM-V land as their own `ModelType`s.
            "minicpmv46" | "minicpmv4.6" | "minicpmv4_6" | "minicpm-v4.6" | "minicpm-v-4.6" | "minicpmv" => Self::MinicpmV46,
            "minicpmo" | "minicpm-o" | "minicpm_o" | "minicpmoduplex" | "minicpm-o-duplex" | "minicpm_o_duplex" => Self::MiniCpmODuplex,
            "minicpm5" | "minicpm-5" | "minicpm_5" | "minicpm" => Self::Minicpm5,
            "qwen25" | "qwen2.5" | "qwen2" => Self::Qwen25,
            "qwen3" => Self::Qwen3,
            "qwen3_5" | "qwen3.5" | "qwen35" | "qwen3_5_dense" => Self::Qwen3_5,
            "qwen3_5_vl" | "qwen3.5_vl" | "qwen3_5-vl" | "qwen3_5vl" | "qwen35_vl" => {
                Self::Qwen3_5VL
            }
            "qwen3_tts" | "qwen3tts" | "qwen3-tts" | "tts" => Self::Qwen3TTS,
            "voxtral_tts" | "voxtral-tts" | "voxtral" | "voxtral_4b" => Self::VoxtralTTS,
            "kokoro" | "kokoro_tts" | "kokoro-tts" | "kokoro-82m" => Self::Kokoro,
            "voxcpm2" | "voxcpm-2" | "voxcpm_2" | "voxcpm" => Self::VoxCpm2,
            "paddleocr_vl" | "paddleocrv" | "paddleocr" | "paddle_ocr_vl" | "paddleocrvl" => Self::PaddleOcrVl,
            "qwen3_asr" | "qwen3asr" | "qwen3-asr" | "asr" => Self::Qwen3ASR,
            _ => Self::Auto,
        }
    }

    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Gemma4 => "gemma4",
            Self::Gemma4VL => "gemma4_vl",
            Self::HunyuanDense => "hunyuan",
            Self::Minicpm5 => "minicpm5",
            Self::MinicpmV46 => "minicpmv4_6",
            Self::MiniCpmODuplex => "minicpmo_duplex",
            Self::Qwen25 => "qwen25",
            Self::Qwen3 => "qwen3",
            Self::Qwen3_5 => "qwen3_5",
            Self::Qwen3_5VL => "qwen3_5_vl",
            Self::Qwen3TTS => "qwen3_tts",
            Self::VoxtralTTS => "voxtral_tts",
            Self::Kokoro => "kokoro_tts",
            Self::VoxCpm2 => "voxcpm2",
            Self::PaddleOcrVl => "paddleocr_vl",
            Self::Qwen3ASR => "qwen3_asr",
        }
    }

    /// Whether this model type is a vision-language model.
    #[must_use]
    pub fn is_vlm(&self) -> bool {
        matches!(self, Self::PaddleOcrVl | Self::Gemma4VL | Self::Qwen3_5VL | Self::MinicpmV46)
    }

    /// Whether this model type is a TTS model.
    #[must_use]
    pub fn is_tts(&self) -> bool {
        matches!(self, Self::Qwen3TTS | Self::VoxtralTTS | Self::Kokoro | Self::VoxCpm2)
    }

    /// Whether this model type is an ASR model.
    #[must_use]
    pub fn is_asr(&self) -> bool {
        matches!(self, Self::Qwen3ASR)
    }

    /// Whether this model type is a full-duplex live audio session
    /// (served over a WebSocket, not a one-shot HTTP request/response).
    #[must_use]
    pub fn is_duplex(&self) -> bool {
        matches!(self, Self::MiniCpmODuplex)
    }
}

/// Model weight format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelFormat {
    Auto,
    Safetensors,
    Gguf,
}

impl ModelFormat {
    // Infallible convenience constructor, not the fallible std::str::FromStr
    // trait (unknown strings fall back to `Auto` instead of erroring).
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "safetensors" => Self::Safetensors,
            "gguf" => Self::Gguf,
            _ => Self::Auto,
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Detection
// ─────────────────────────────────────────────────────────────

/// Minimal subset of `HuggingFace` `config.json` for architecture detection.
#[derive(Deserialize, Default)]
struct HfConfig {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
    vision_config: Option<serde_json::Value>,
    /// VoxCPM2's `config.json` uses a **singular** `"architecture"` string
    /// field (not the plural HF-style `"architectures"` list) — genuinely
    /// distinctive, checked separately in `detect_model_type`.
    architecture: Option<String>,
}

/// Minimal subset of Mistral `params.json` for architecture detection.
#[derive(Deserialize, Default)]
struct MistralConfig {
    model_type: Option<String>,
}

/// Auto-detect the model type from `config.json` in the model directory.
pub fn detect_model_type(model_path: &str) -> ModelType {
    let path = Path::new(model_path);

    // Locate config.json (same dir for dir paths; parent dir for GGUF files).
    let config_path = if path.is_file() {
        path.parent().map(|p| p.join("config.json"))
    } else {
        Some(path.join("config.json"))
    };

    if let Some(config_path) = config_path
        && let Ok(data) = std::fs::read(&config_path)
        && let Ok(config) = serde_json::from_slice::<HfConfig>(&data)
    {
        // 0. Check the singular `architecture` field (VoxCPM2's schema —
        // no `model_type`, no plural `architectures`, so this must be
        // checked separately or it's silently invisible to the branches
        // below).
        if config.architecture.as_deref().map(str::to_lowercase).as_deref() == Some("voxcpm2") {
            return ModelType::VoxCpm2;
        }

        // 1. Check `model_type` field
        if let Some(ref mt) = config.model_type {
            match mt.to_lowercase().as_str() {
                "gemma4" => {
                    return if config.vision_config.is_some() {
                        ModelType::Gemma4VL
                    } else {
                        ModelType::Gemma4
                    };
                }
                "qwen2" | "qwen2.5" => return ModelType::Qwen25,
                "qwen3" => return ModelType::Qwen3,
                "qwen3_5" | "qwen3.5" => {
                    return if config.vision_config.is_some() {
                        ModelType::Qwen3_5VL
                    } else {
                        ModelType::Qwen3_5
                    };
                }
                "minicpmv4_6" | "minicpmv4.6" => return ModelType::MinicpmV46,
                "minicpmo" => return ModelType::MiniCpmODuplex,
                "qwen3_tts" | "qwen3tts" => return ModelType::Qwen3TTS,
                "qwen3_asr" | "qwen3asr" => return ModelType::Qwen3ASR,
                "style_text_to_speech_2" => return ModelType::Kokoro,
                m if m.contains("hunyuan") => return ModelType::HunyuanDense,
                m if m.contains("paddleocr") => return ModelType::PaddleOcrVl,
                _ => {}
            }
        }

        // 2. Check `architectures` field
        if let Some(ref archs) = config.architectures {
            for arch in archs {
                let a = arch.to_lowercase();
                if a.contains("paddleocr") {
                    return ModelType::PaddleOcrVl;
                }
                if a.contains("hunyuan") {
                    return ModelType::HunyuanDense;
                }
                if a.contains("gemma4") {
                    return ModelType::Gemma4;
                }
                if a.contains("minicpmv4_6") {
                    return ModelType::MinicpmV46;
                }
                // Checked before any bare "minicpm" fallback would exist —
                // "MiniCPMO" is architecturally distinctive (real HF
                // architectures value on the checkpoint).
                if a.contains("minicpmo") {
                    return ModelType::MiniCpmODuplex;
                }
                if a.contains("qwen3ttsforconditional") || a.contains("qwen3_tts") {
                    return ModelType::Qwen3TTS;
                }
                if a.contains("qwen3asrforconditional") || a.contains("qwen3_asr") {
                    return ModelType::Qwen3ASR;
                }
                // Qwen3_5ForConditionalGeneration is the multimodal class;
                // Qwen3_5ForCausalLM (or any other Qwen3_5*) is text-only.
                if a.contains("qwen3_5forconditional") || a.contains("qwen3.5forconditional") {
                    return ModelType::Qwen3_5VL;
                }
                if a.contains("qwen3_5") || a.contains("qwen3.5") {
                    return ModelType::Qwen3_5;
                }
                if a.contains("qwen3") {
                    return ModelType::Qwen3;
                }
                if a.contains("qwen2") {
                    return ModelType::Qwen25;
                }
            }
        }
    }

    // 3. Check params.json (Mistral-style config, e.g. Voxtral)
    let params_path = if path.is_file() {
        path.parent().map(|p| p.join("params.json"))
    } else {
        Some(path.join("params.json"))
    };
    if let Some(params_path) = params_path
        && let Ok(data) = std::fs::read(&params_path)
        && let Ok(config) = serde_json::from_slice::<MistralConfig>(&data)
        && let Some(ref mt) = config.model_type
        && mt == "voxtral_tts"
    {
        return ModelType::VoxtralTTS;
    }

    // 4. GGUF files: the architecture is recorded in the header — far more
    // reliable than the path name.
    if let Some(mt) = detect_from_gguf_header(path) {
        return mt;
    }

    // 5. Heuristic: check the model path name
    let path_lower = model_path.to_lowercase();
    if path_lower.contains("voxtral") {
        ModelType::VoxtralTTS
    } else if path_lower.contains("kokoro") {
        ModelType::Kokoro
    } else if path_lower.contains("voxcpm") {
        ModelType::VoxCpm2
    } else if path_lower.contains("paddleocr") {
        ModelType::PaddleOcrVl
    } else if path_lower.contains("gemma4") || path_lower.contains("gemma-4") {
        ModelType::Gemma4
    } else if path_lower.contains("hunyuan") {
        ModelType::HunyuanDense
    } else if path_lower.contains("minicpm-v") || path_lower.contains("minicpmv") {
        // Checked before the bare "minicpm" branch below — "MiniCPM-V-4.6"
        // contains "minicpm" too, and would otherwise be mis-claimed by the
        // Minicpm5 fallback.
        ModelType::MinicpmV46
    } else if path_lower.contains("minicpm-o") || path_lower.contains("minicpmo") {
        // Same reasoning as MiniCPM-V-4.6 above — checked before the bare
        // "minicpm" fallback.
        ModelType::MiniCpmODuplex
    } else if path_lower.contains("minicpm") {
        // MiniCPM5's config.json is a plain `LlamaForCausalLM`
        // (model_type/architectures give no distinctive signal), so this
        // path-name heuristic is the only auto-detect route; use
        // `--model-type minicpm5` explicitly for renamed directories.
        ModelType::Minicpm5
    } else if path_lower.contains("qwen3-tts") || path_lower.contains("qwen3_tts") || path_lower.contains("qwen3tts") {
        ModelType::Qwen3TTS
    } else if path_lower.contains("qwen3-asr") || path_lower.contains("qwen3_asr") || path_lower.contains("qwen3asr") {
        ModelType::Qwen3ASR
    } else if path_lower.contains("qwen3.5") || path_lower.contains("qwen3_5") || path_lower.contains("qwen35") {
        ModelType::Qwen3_5
    } else if path_lower.contains("qwen3_5_vl") || path_lower.contains("qwen3.5_vl") || path_lower.contains("qwen3.5-vl") || path_lower.contains("qwen35_vl") {
        ModelType::Qwen3_5VL
    } else if path_lower.contains("qwen3") {
        ModelType::Qwen3
    } else if path_lower.contains("qwen2") || path_lower.contains("qwen25") {
        ModelType::Qwen25
    } else {
        tracing::warn!(
            "Could not auto-detect model type from '{model_path}', defaulting to Qwen25"
        );
        ModelType::Qwen25
    }
}

/// Read `general.architecture` from a `.gguf` file's header.
fn detect_from_gguf_header(path: &Path) -> Option<ModelType> {
    let is_gguf = path.is_file()
        && path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
    if !is_gguf {
        return None;
    }
    let mut file = std::fs::File::open(path).ok()?;
    let ct = candle_core::quantized::gguf_file::Content::read(&mut file).ok()?;
    let arch = ct
        .metadata
        .get("general.architecture")?
        .to_string()
        .ok()?
        .to_lowercase();
    match arch.as_str() {
        "qwen35" | "qwen3_5" | "qwen3.5" => Some(ModelType::Qwen3_5),
        "qwen3" | "qwen3moe" => Some(ModelType::Qwen3),
        "qwen2" => Some(ModelType::Qwen25),
        a if a.starts_with("hunyuan") => Some(ModelType::HunyuanDense),
        // Deliberately specific (not bare "llama") — MiniCPM5 GGUF
        // conversions may use "llama" as `general.architecture` since the
        // checkpoint itself is architecturally plain Llama, in which case
        // this won't fire and `--model-type minicpm5` is required instead.
        a if a.starts_with("minicpm") => Some(ModelType::Minicpm5),
        a if a.starts_with("gemma") => Some(ModelType::Gemma4),
        other => {
            tracing::warn!("Unrecognized GGUF architecture '{other}'");
            None
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Factory
// ─────────────────────────────────────────────────────────────

/// Resolve `ModelType::Auto` to a concrete type.
fn resolve(model_type: ModelType, model_path: &str) -> ModelType {
    if model_type == ModelType::Auto {
        detect_model_type(model_path)
    } else {
        model_type
    }
}

/// Create a model backend.
///
/// `quant` requests in-situ quantization of a safetensors checkpoint (e.g.
/// `q4k`, `q8_0`); only backends that support it accept the flag.
pub fn create_backend(
    model_type: ModelType,
    model_path: &str,
    device: &Device,
    dtype: &DType,
    format: ModelFormat,
    quant: Option<&str>,
) -> Result<Box<dyn ModelBackend>> {
    let model_type = resolve(model_type, model_path);
    tracing::info!("Creating backend: {:?}", model_type);

    if quant.is_some() && model_type != ModelType::Qwen3_5 {
        anyhow::bail!(
            "--quant (in-situ quantization) is currently only supported for qwen3_5 models; \
             for other models use a GGUF checkpoint with --format gguf"
        );
    }

    match model_type {
        ModelType::HunyuanDense => {
            let hy_fmt = match format {
                ModelFormat::Safetensors => crane_core::models::hunyuan_dense::ModelFormat::Safetensors,
                ModelFormat::Gguf => crane_core::models::hunyuan_dense::ModelFormat::Gguf,
                ModelFormat::Auto => crane_core::models::hunyuan_dense::ModelFormat::Auto,
            };
            Ok(Box::new(HunyuanBackend::new(model_path, device, dtype, hy_fmt)?))
        }
        ModelType::Gemma4 => {
            let g4_fmt = match format {
                ModelFormat::Safetensors => crane_core::models::gemma4::ModelFormat::Safetensors,
                ModelFormat::Gguf => crane_core::models::gemma4::ModelFormat::Gguf,
                ModelFormat::Auto => crane_core::models::gemma4::ModelFormat::Auto,
            };
            Ok(Box::new(Gemma4Backend::new(model_path, device, dtype, g4_fmt)?))
        }
        ModelType::Minicpm5 => {
            let mc_fmt = match format {
                ModelFormat::Safetensors => crane_core::models::minicpm5::ModelFormat::Safetensors,
                ModelFormat::Gguf => crane_core::models::minicpm5::ModelFormat::Gguf,
                ModelFormat::Auto => crane_core::models::minicpm5::ModelFormat::Auto,
            };
            Ok(Box::new(Minicpm5Backend::new(model_path, device, dtype, mc_fmt)?))
        }
        ModelType::Qwen25 => Ok(Box::new(Qwen25Backend::new(model_path, device, dtype)?)),
        ModelType::Qwen3 => Ok(Box::new(Qwen3Backend::new(model_path, device, dtype)?)),
        ModelType::Qwen3_5 => {
            let quant = quant
                .map(crane_core::ops::linear::parse_ggml_dtype)
                .transpose()?;
            let q35_fmt = match format {
                ModelFormat::Safetensors => crane_core::models::qwen3_5::ModelFormat::Safetensors,
                ModelFormat::Gguf => crane_core::models::qwen3_5::ModelFormat::Gguf,
                ModelFormat::Auto => crane_core::models::qwen3_5::ModelFormat::Auto,
            };
            Ok(Box::new(Qwen3_5Backend::new_with_options(
                model_path, device, dtype, q35_fmt, quant,
            )?))
        }
        ModelType::PaddleOcrVl => {
            anyhow::bail!("PaddleOCR-VL is a VLM model — use create_vlm_model() instead of create_backend()")
        }
        ModelType::Gemma4VL => {
            anyhow::bail!("Gemma4-VL is a VLM model — use the VLM endpoint instead of create_backend()")
        }
        ModelType::Qwen3_5VL => {
            anyhow::bail!("Qwen3_5-VL is a VLM model — use the Qwen3_5-VL endpoint instead of create_backend()")
        }
        ModelType::MinicpmV46 => {
            anyhow::bail!("MiniCPM-V-4.6 is a VLM model — use the MiniCPM-V-4.6 endpoint instead of create_backend()")
        }
        ModelType::Qwen3TTS => {
            anyhow::bail!("Qwen3-TTS is a TTS model — use create_tts() instead of create_backend()")
        }
        ModelType::VoxtralTTS => {
            anyhow::bail!("Voxtral-TTS is a TTS model — use create_tts() instead of create_backend()")
        }
        ModelType::Kokoro => {
            anyhow::bail!("Kokoro is a TTS model — use create_tts() instead of create_backend()")
        }
        ModelType::VoxCpm2 => {
            anyhow::bail!("VoxCPM2 is a TTS model — use create_tts() instead of create_backend()")
        }
        ModelType::Qwen3ASR => {
            anyhow::bail!("Qwen3-ASR is an ASR model — use create_asr() instead of create_backend()")
        }
        ModelType::MiniCpmODuplex => {
            anyhow::bail!("MiniCPM-o is a full-duplex model — use the duplex WebSocket endpoint instead of create_backend()")
        }
        ModelType::Auto => unreachable!(),
    }
}

/// Create a chat template processor for the given model.
pub fn create_chat_template(
    model_type: ModelType,
    model_path: &str,
) -> Box<dyn ChatTemplateProcessor> {
    let model_type = resolve(model_type, model_path);

    // `.gguf` files embed their own chat_template; pass the file path itself
    // so AutoTokenizer::from_pretrained dispatches to its GGUF reader. For
    // directory-style HF layouts (and for non-GGUF single files like a
    // standalone `tokenizer_config.json`) pass the path as-is so the parent
    // resolution can locate the template.
    let path = Path::new(model_path);
    let is_gguf = path.is_file()
        && path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
    let template_target = if path.is_file() && !is_gguf {
        // Non-GGUF file: assume it's a `tokenizer_config.json` (or sibling
        // templates are alongside — nothing to resolve).
        model_path.to_string()
    } else if is_gguf {
        model_path.to_string()
    } else {
        // Directory: tokenizer_config.json + tokenizer.json live inside.
        model_path.to_string()
    };
    let model_path = template_target.as_str();

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

/// Create a PaddleOCR-VL model for VLM inference.
///
/// # Errors
///
/// Returns an error if the model fails to load from `model_path`.
pub fn create_vlm_model(
    model_path: &str,
    use_cpu: bool,
    use_bf16: bool,
) -> Result<crane_core::models::paddleocr_vl::PaddleOcrVL> {
    tracing::info!("Creating PaddleOCR-VL model from: {}", model_path);
    crane_core::models::paddleocr_vl::PaddleOcrVL::from_local(model_path, use_cpu, use_bf16)
}

/// Create a TTS model as a trait object.
///
/// Unified entrypoint for all TTS model types; the returned `Box<dyn Tts + Send>`
/// can be moved into a dedicated thread without model-specific branching.
///
/// # Errors
///
/// Returns an error if `model_type` does not resolve to a TTS variant or the
/// model fails to load from `model_path`.
pub fn create_tts(
    model_type: ModelType,
    model_path: &str,
    device: &Device,
    dtype: &DType,
) -> Result<Box<dyn crane::audio::Tts + Send>> {
    tracing::info!("Creating {} model from: {}", model_type.display_name(), model_path);
    match model_type {
        ModelType::Qwen3TTS => {
            let model = crane_core::models::qwen3_tts::Model::new(model_path, device, dtype)?;
            Ok(Box::new(model))
        }
        ModelType::VoxtralTTS => {
            let model = crane_core::models::voxtral_tts::Model::new(model_path, device, dtype)?;
            Ok(Box::new(model))
        }
        #[cfg(feature = "onnx")]
        ModelType::Kokoro => create_kokoro_tts(model_path, device, dtype),
        #[cfg(not(feature = "onnx"))]
        ModelType::Kokoro => anyhow::bail!("Kokoro TTS requires the 'onnx' feature"),
        ModelType::VoxCpm2 => {
            let model = crane_core::models::voxcpm2::VoxCpm2Model::new(model_path, device, dtype)?;
            Ok(Box::new(model))
        }
        other => anyhow::bail!("{other:?} is not a TTS model type"),
    }
}

/// Builds a [`crane::audio::KokoroTts`] from `model_path`.
///
/// `--model-path` points at the Kokoro model directory itself (`config.json`,
/// `tokenizer.json`, `onnx/`, `voices/`); G2P assets are located at
/// `{model_path}/g2p/`, following Moonshine's per-language directory layout
/// (see [`crane_core::models::g2p::MoonshineG2p::from_g2p_dir`]). A real
/// deployment symlinks or copies its G2P data there, e.g.:
///
/// ```text
/// ln -s /path/to/moonshine-g2p <model_path>/g2p
/// ```
///
/// # Errors
///
/// Returns an error if the Kokoro model fails to load, or if
/// `{model_path}/g2p/en_us/dict_filtered_heteronyms.tsv` is missing or
/// malformed.
#[cfg(feature = "onnx")]
fn create_kokoro_tts(
    model_path: &str,
    device: &Device,
    dtype: &DType,
) -> Result<Box<dyn crane::audio::Tts + Send>> {
    let model = crane_core::models::kokoro_tts::Model::new(model_path, device, dtype)?;

    let g2p_dir = Path::new(model_path).join("g2p");
    let phonemizer = crane_core::models::g2p::MoonshineG2p::from_g2p_dir(&g2p_dir)
        .with_context(|| format!("loading Kokoro G2P assets from {}", g2p_dir.display()))?;

    Ok(Box::new(crane::audio::KokoroTts::new(model, Box::new(phonemizer))))
}

/// Create an ASR model as a trait object.
///
/// Unified entrypoint for all ASR model types; the returned `Box<dyn Asr + Send>`
/// can be moved into a dedicated thread without model-specific branching.
///
/// # Errors
///
/// Returns an error if `model_type` does not resolve to an ASR variant or the
/// model fails to load from `model_path`.
pub fn create_asr(
    model_type: ModelType,
    model_path: &str,
    device: &Device,
    dtype: &DType,
) -> Result<Box<dyn crane::audio::Asr + Send>> {
    tracing::info!("Creating {} model from: {}", model_type.display_name(), model_path);
    match model_type {
        ModelType::Qwen3ASR => {
            let model = crane_core::models::qwen3_asr::Model::new(model_path, device, dtype)?;
            Ok(Box::new(model))
        }
        other => anyhow::bail!("{other:?} is not an ASR model type"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ModelType::from_str ──

    #[test]
    fn model_type_from_str_hunyuan_variants() {
        assert_eq!(ModelType::from_str("hunyuan"), ModelType::HunyuanDense);
        assert_eq!(ModelType::from_str("hunyuan_dense"), ModelType::HunyuanDense);
        assert_eq!(ModelType::from_str("hunyuandense"), ModelType::HunyuanDense);
        assert_eq!(ModelType::from_str("HUNYUAN"), ModelType::HunyuanDense);
    }

    #[test]
    fn model_type_from_str_qwen_variants() {
        assert_eq!(ModelType::from_str("qwen25"), ModelType::Qwen25);
        assert_eq!(ModelType::from_str("qwen2.5"), ModelType::Qwen25);
        assert_eq!(ModelType::from_str("qwen2"), ModelType::Qwen25);
        assert_eq!(ModelType::from_str("QWEN2"), ModelType::Qwen25);
        assert_eq!(ModelType::from_str("qwen3"), ModelType::Qwen3);
        assert_eq!(ModelType::from_str("QWEN3"), ModelType::Qwen3);
    }

    #[test]
    fn model_type_from_str_minicpm5_variants() {
        assert_eq!(ModelType::from_str("minicpm5"), ModelType::Minicpm5);
        assert_eq!(ModelType::from_str("minicpm-5"), ModelType::Minicpm5);
        assert_eq!(ModelType::from_str("minicpm_5"), ModelType::Minicpm5);
        assert_eq!(ModelType::from_str("minicpm"), ModelType::Minicpm5);
        assert_eq!(ModelType::from_str("MINICPM5"), ModelType::Minicpm5);
    }

    #[test]
    fn model_type_from_str_minicpm_v46_variants() {
        assert_eq!(ModelType::from_str("minicpmv46"), ModelType::MinicpmV46);
        assert_eq!(ModelType::from_str("minicpmv4.6"), ModelType::MinicpmV46);
        assert_eq!(ModelType::from_str("minicpmv4_6"), ModelType::MinicpmV46);
        assert_eq!(ModelType::from_str("minicpm-v4.6"), ModelType::MinicpmV46);
        assert_eq!(ModelType::from_str("MINICPMV4_6"), ModelType::MinicpmV46);
    }

    #[test]
    fn model_type_is_vlm_includes_minicpm_v46() {
        assert!(ModelType::MinicpmV46.is_vlm());
        assert!(!ModelType::Minicpm5.is_vlm());
    }

    #[test]
    fn detect_from_config_json_model_type_minicpmv4_6() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(&config, r#"{"model_type": "minicpmv4_6"}"#).unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::MinicpmV46);
    }

    #[test]
    fn detect_from_config_json_architectures_minicpmv4_6() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(&config, r#"{"architectures": ["MiniCPMV4_6ForConditionalGeneration"]}"#).unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::MinicpmV46);
    }

    #[test]
    fn detect_path_heuristic_minicpm_v46_not_claimed_by_minicpm5() {
        let result = detect_model_type("/models/MiniCPM-V-4.6");
        assert_eq!(result, ModelType::MinicpmV46);
    }

    #[test]
    fn model_type_from_str_minicpmo_variants() {
        assert_eq!(ModelType::from_str("minicpmo"), ModelType::MiniCpmODuplex);
        assert_eq!(ModelType::from_str("minicpm-o"), ModelType::MiniCpmODuplex);
        assert_eq!(ModelType::from_str("minicpm_o_duplex"), ModelType::MiniCpmODuplex);
        assert_eq!(ModelType::from_str("MINICPMO"), ModelType::MiniCpmODuplex);
    }

    #[test]
    fn model_type_is_duplex_includes_minicpmo() {
        assert!(ModelType::MiniCpmODuplex.is_duplex());
        assert!(!ModelType::MinicpmV46.is_duplex());
        assert!(!ModelType::Minicpm5.is_duplex());
    }

    #[test]
    fn detect_from_config_json_model_type_minicpmo() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(&config, r#"{"model_type": "minicpmo"}"#).unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::MiniCpmODuplex);
    }

    #[test]
    fn detect_from_config_json_architectures_minicpmo() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(&config, r#"{"architectures": ["MiniCPMO"]}"#).unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::MiniCpmODuplex);
    }

    #[test]
    fn detect_path_heuristic_minicpmo_not_claimed_by_minicpm5() {
        let result = detect_model_type("/models/MiniCPM-o-4_5");
        assert_eq!(result, ModelType::MiniCpmODuplex);
    }

    #[test]
    fn model_type_from_str_voxtral_variants() {
        assert_eq!(ModelType::from_str("voxtral_tts"), ModelType::VoxtralTTS);
        assert_eq!(ModelType::from_str("voxtral-tts"), ModelType::VoxtralTTS);
        assert_eq!(ModelType::from_str("voxtral"), ModelType::VoxtralTTS);
        assert_eq!(ModelType::from_str("voxtral_4b"), ModelType::VoxtralTTS);
        assert_eq!(ModelType::from_str("VOXTRAL"), ModelType::VoxtralTTS);
    }

    #[test]
    fn model_type_from_str_kokoro_variants() {
        assert_eq!(ModelType::from_str("kokoro"), ModelType::Kokoro);
        assert_eq!(ModelType::from_str("kokoro_tts"), ModelType::Kokoro);
        assert_eq!(ModelType::from_str("kokoro-tts"), ModelType::Kokoro);
        assert_eq!(ModelType::from_str("kokoro-82m"), ModelType::Kokoro);
        assert_eq!(ModelType::from_str("KOKORO"), ModelType::Kokoro);
    }

    #[test]
    fn model_type_from_str_voxcpm2_variants() {
        assert_eq!(ModelType::from_str("voxcpm2"), ModelType::VoxCpm2);
        assert_eq!(ModelType::from_str("voxcpm-2"), ModelType::VoxCpm2);
        assert_eq!(ModelType::from_str("voxcpm_2"), ModelType::VoxCpm2);
        assert_eq!(ModelType::from_str("voxcpm"), ModelType::VoxCpm2);
        assert_eq!(ModelType::from_str("VOXCPM2"), ModelType::VoxCpm2);
    }

    #[test]
    fn model_type_is_tts_includes_voxcpm2() {
        assert!(ModelType::VoxCpm2.is_tts());
    }

    #[test]
    fn detect_from_config_json_singular_architecture_voxcpm2() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(&config, r#"{"architecture": "voxcpm2"}"#).unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::VoxCpm2);
    }

    #[test]
    fn detect_path_heuristic_voxcpm2() {
        let result = detect_model_type("/models/VoxCPM2");
        assert_eq!(result, ModelType::VoxCpm2);
    }

    #[test]
    fn model_type_from_str_auto_fallback() {
        assert_eq!(ModelType::from_str("auto"), ModelType::Auto);
        assert_eq!(ModelType::from_str("unknown"), ModelType::Auto);
        assert_eq!(ModelType::from_str(""), ModelType::Auto);
    }

    #[test]
    fn model_type_is_tts() {
        assert!(ModelType::Qwen3TTS.is_tts());
        assert!(ModelType::VoxtralTTS.is_tts());
        assert!(ModelType::Kokoro.is_tts());
        assert!(!ModelType::Qwen3.is_tts());
    }

    #[test]
    fn model_type_from_str_asr_variants() {
        assert_eq!(ModelType::from_str("qwen3_asr"), ModelType::Qwen3ASR);
        assert_eq!(ModelType::from_str("qwen3asr"), ModelType::Qwen3ASR);
        assert_eq!(ModelType::from_str("qwen3-asr"), ModelType::Qwen3ASR);
        assert_eq!(ModelType::from_str("asr"), ModelType::Qwen3ASR);
        assert_eq!(ModelType::from_str("ASR"), ModelType::Qwen3ASR);
    }

    #[test]
    fn model_type_is_asr() {
        assert!(ModelType::Qwen3ASR.is_asr());
        assert!(!ModelType::Qwen3.is_asr());
        assert!(!ModelType::Qwen3TTS.is_asr());
    }

    #[test]
    fn model_type_display_name() {
        assert_eq!(ModelType::Auto.display_name(), "auto");
        assert_eq!(ModelType::HunyuanDense.display_name(), "hunyuan");
        assert_eq!(ModelType::Minicpm5.display_name(), "minicpm5");
        assert_eq!(ModelType::MinicpmV46.display_name(), "minicpmv4_6");
        assert_eq!(ModelType::MiniCpmODuplex.display_name(), "minicpmo_duplex");
        assert_eq!(ModelType::Qwen25.display_name(), "qwen25");
        assert_eq!(ModelType::Qwen3.display_name(), "qwen3");
        assert_eq!(ModelType::Qwen3ASR.display_name(), "qwen3_asr");
    }

    // ── ModelFormat::from_str ──

    #[test]
    fn model_format_from_str() {
        assert_eq!(ModelFormat::from_str("safetensors"), ModelFormat::Safetensors);
        assert_eq!(ModelFormat::from_str("SAFETENSORS"), ModelFormat::Safetensors);
        assert_eq!(ModelFormat::from_str("gguf"), ModelFormat::Gguf);
        assert_eq!(ModelFormat::from_str("auto"), ModelFormat::Auto);
        assert_eq!(ModelFormat::from_str("unknown"), ModelFormat::Auto);
    }

    // ── detect_model_type with temp files ──

    #[test]
    fn detect_from_config_json_model_type_qwen2() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(&config, r#"{"model_type": "qwen2"}"#).unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::Qwen25);
    }

    #[test]
    fn detect_from_config_json_model_type_qwen3() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(&config, r#"{"model_type": "qwen3"}"#).unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::Qwen3);
    }

    #[test]
    fn detect_from_config_json_model_type_qwen3_asr() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(&config, r#"{"model_type": "qwen3_asr"}"#).unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::Qwen3ASR);
    }

    #[test]
    fn detect_from_config_json_architectures_qwen3_asr() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(
            &config,
            r#"{"architectures": ["Qwen3ASRForConditionalGeneration"]}"#,
        )
        .unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::Qwen3ASR);
    }

    #[test]
    fn detect_path_heuristic_qwen3_asr() {
        let result = detect_model_type("/models/Qwen3-ASR-0.6B-hf");
        assert_eq!(result, ModelType::Qwen3ASR);
    }

    #[test]
    fn detect_from_config_json_architectures() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(
            &config,
            r#"{"architectures": ["HunyuanForCausalLM"]}"#,
        )
        .unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::HunyuanDense);
    }

    #[test]
    fn detect_from_config_json_architectures_qwen2() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(
            &config,
            r#"{"architectures": ["Qwen2ForCausalLM"]}"#,
        )
        .unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::Qwen25);
    }

    #[test]
    fn detect_from_config_json_architectures_qwen3() {
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(
            &config,
            r#"{"architectures": ["Qwen3ForCausalLM"]}"#,
        )
        .unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::Qwen3);
    }

    #[test]
    fn detect_path_heuristic_hunyuan() {
        let result = detect_model_type("/models/Hunyuan-Dense-7B");
        assert_eq!(result, ModelType::HunyuanDense);
    }

    #[test]
    fn detect_path_heuristic_minicpm5() {
        let result = detect_model_type("/models/MiniCPM5-1B");
        assert_eq!(result, ModelType::Minicpm5);
    }

    #[test]
    fn detect_from_config_json_llama_architecture_is_not_minicpm5() {
        // MiniCPM5's own config.json is a plain LlamaForCausalLM — verify
        // that alone does NOT get claimed by Minicpm5 (would misfire on
        // real Llama checkpoints). Falls through to the Qwen25 default
        // since there's no path-name signal in this temp dir.
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(
            &config,
            r#"{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}"#,
        )
        .unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_ne!(result, ModelType::Minicpm5);
    }

    #[test]
    fn detect_path_heuristic_qwen3() {
        let result = detect_model_type("/models/Qwen3-8B");
        assert_eq!(result, ModelType::Qwen3);
    }

    #[test]
    fn detect_path_heuristic_qwen2() {
        let result = detect_model_type("/models/Qwen2.5-7B-Instruct");
        assert_eq!(result, ModelType::Qwen25);
    }

    #[test]
    fn detect_from_params_json_voxtral() {
        let dir = tempfile::tempdir().unwrap();
        let params = dir.path().join("params.json");
        std::fs::write(&params, r#"{"model_type": "voxtral_tts"}"#).unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::VoxtralTTS);
    }

    #[test]
    fn detect_path_heuristic_voxtral() {
        let result = detect_model_type("/models/Voxtral-4B-TTS-2603");
        assert_eq!(result, ModelType::VoxtralTTS);
    }

    #[test]
    fn detect_from_config_json_model_type_kokoro() {
        // Real onnx-community Kokoro exports carry only this one key — see
        // the "Local asset locations" note in the G2P design doc.
        let dir = tempfile::tempdir().unwrap();
        let config = dir.path().join("config.json");
        std::fs::write(&config, r#"{"model_type": "style_text_to_speech_2"}"#).unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::Kokoro);
    }

    #[test]
    fn detect_path_heuristic_kokoro() {
        let result = detect_model_type("/models/Kokoro-82M-v1.0-ONNX");
        assert_eq!(result, ModelType::Kokoro);
    }

    #[test]
    fn detect_fallback_unknown_defaults_to_qwen25() {
        // Temp dir with no config.json and no heuristic match.
        let dir = tempfile::tempdir().unwrap();
        let result = detect_model_type(dir.path().to_str().unwrap());
        assert_eq!(result, ModelType::Qwen25);
    }

    // ── resolve ──

    #[test]
    fn resolve_auto_delegates_to_detect() {
        let result = resolve(ModelType::Auto, "/models/Qwen3-8B");
        assert_eq!(result, ModelType::Qwen3);
    }

    #[test]
    fn resolve_explicit_type_is_passthrough() {
        let result = resolve(ModelType::HunyuanDense, "/models/whatever");
        assert_eq!(result, ModelType::HunyuanDense);
    }
}
