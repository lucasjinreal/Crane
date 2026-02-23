//! Model factory for automatic model type detection and backend creation.
//!
//! Supports auto-detection from `config.json`'s `model_type` / `architectures`
//! fields, or explicit model type specification via CLI.

use anyhow::Result;
use candle_core::{DType, Device};
use serde::Deserialize;
use std::path::Path;

use super::backend::{HunyuanBackend, ModelBackend, Qwen25Backend, Qwen3Backend};
use crate::chat_template::{AutoChatTemplate, ChatTemplateProcessor, HunyuanChatTemplate};

// ─────────────────────────────────────────────────────────────
//  Enums
// ─────────────────────────────────────────────────────────────

/// Supported model architectures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    Auto,
    HunyuanDense,
    Qwen25,
    Qwen3,
    Qwen3TTS,
    PaddleOcrVl,
}

impl ModelType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "hunyuan" | "hunyuan_dense" | "hunyuandense" => Self::HunyuanDense,
            "qwen25" | "qwen2.5" | "qwen2" => Self::Qwen25,
            "qwen3" => Self::Qwen3,
            "qwen3_tts" | "qwen3tts" | "qwen3-tts" | "tts" => Self::Qwen3TTS,
            "paddleocr_vl" | "paddleocrv" | "paddleocr" | "paddle_ocr_vl" | "paddleocrvl" => Self::PaddleOcrVl,
            _ => Self::Auto,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::HunyuanDense => "hunyuan",
            Self::Qwen25 => "qwen25",
            Self::Qwen3 => "qwen3",
            Self::Qwen3TTS => "qwen3_tts",
            Self::PaddleOcrVl => "paddleocr_vl",
        }
    }

    /// Whether this model type is a vision-language model.
    pub fn is_vlm(&self) -> bool {
        matches!(self, Self::PaddleOcrVl)
    }

    /// Whether this model type is a TTS model.
    pub fn is_tts(&self) -> bool {
        matches!(self, Self::Qwen3TTS)
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

/// Minimal subset of HuggingFace `config.json` for architecture detection.
#[derive(Deserialize, Default)]
struct HfConfig {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
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

    if let Some(config_path) = config_path {
        if let Ok(data) = std::fs::read(&config_path) {
            if let Ok(config) = serde_json::from_slice::<HfConfig>(&data) {
                // 1. Check `model_type` field
                if let Some(ref mt) = config.model_type {
                    match mt.to_lowercase().as_str() {
                        "qwen2" | "qwen2.5" => return ModelType::Qwen25,
                        "qwen3" => return ModelType::Qwen3,
                        "qwen3_tts" | "qwen3tts" => return ModelType::Qwen3TTS,
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
                        if a.contains("qwen3ttsforconditional") || a.contains("qwen3_tts") {
                            return ModelType::Qwen3TTS;
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
        }
    }

    // 3. Heuristic: check the model path name
    let path_lower = model_path.to_lowercase();
    if path_lower.contains("paddleocr") {
        ModelType::PaddleOcrVl
    } else if path_lower.contains("hunyuan") {
        ModelType::HunyuanDense
    } else if path_lower.contains("qwen3-tts") || path_lower.contains("qwen3_tts") || path_lower.contains("qwen3tts") {
        ModelType::Qwen3TTS
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
pub fn create_backend(
    model_type: ModelType,
    model_path: &str,
    device: &Device,
    dtype: &DType,
    format: ModelFormat,
) -> Result<Box<dyn ModelBackend>> {
    let model_type = resolve(model_type, model_path);
    tracing::info!("Creating backend: {:?}", model_type);

    match model_type {
        ModelType::HunyuanDense => {
            let hy_fmt = match format {
                ModelFormat::Safetensors => crane_core::models::hunyuan_dense::ModelFormat::Safetensors,
                ModelFormat::Gguf => crane_core::models::hunyuan_dense::ModelFormat::Gguf,
                ModelFormat::Auto => crane_core::models::hunyuan_dense::ModelFormat::Auto,
            };
            Ok(Box::new(HunyuanBackend::new(model_path, device, dtype, hy_fmt)?))
        }
        ModelType::Qwen25 => Ok(Box::new(Qwen25Backend::new(model_path, device, dtype)?)),
        ModelType::Qwen3 => Ok(Box::new(Qwen3Backend::new(model_path, device, dtype)?)),
        ModelType::PaddleOcrVl => {
            anyhow::bail!("PaddleOCR-VL is a VLM model — use create_vlm_model() instead of create_backend()")
        }
        ModelType::Qwen3TTS => {
            anyhow::bail!("Qwen3-TTS is a TTS model — use create_tts_model() instead of create_backend()")
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
pub fn create_vlm_model(
    model_path: &str,
    use_cpu: bool,
    use_bf16: bool,
) -> Result<crane_core::models::paddleocr_vl::PaddleOcrVL> {
    tracing::info!("Creating PaddleOCR-VL model from: {}", model_path);
    crane_core::models::paddleocr_vl::PaddleOcrVL::from_local(model_path, use_cpu, use_bf16)
}

/// Create a Qwen3-TTS model for TTS inference.
pub fn create_tts_model(
    model_path: &str,
    device: &Device,
    dtype: &DType,
) -> Result<crane_core::models::qwen3_tts::Model> {
    tracing::info!("Creating Qwen3-TTS model from: {}", model_path);
    crane_core::models::qwen3_tts::Model::new(model_path, device, dtype)
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
    fn model_type_from_str_auto_fallback() {
        assert_eq!(ModelType::from_str("auto"), ModelType::Auto);
        assert_eq!(ModelType::from_str("unknown"), ModelType::Auto);
        assert_eq!(ModelType::from_str(""), ModelType::Auto);
    }

    #[test]
    fn model_type_display_name() {
        assert_eq!(ModelType::Auto.display_name(), "auto");
        assert_eq!(ModelType::HunyuanDense.display_name(), "hunyuan");
        assert_eq!(ModelType::Qwen25.display_name(), "qwen25");
        assert_eq!(ModelType::Qwen3.display_name(), "qwen3");
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
