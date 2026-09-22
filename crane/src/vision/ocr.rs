use crate::common::{
    CraneError, CraneResult,
    config::{CommonConfig, DataType, DeviceConfig},
};
use crate::llm::LlmModelType;
use crane_core::models::paddleocr_vl::{OcrTask, PaddleOcrVL};
use std::path::Path;

#[cfg(feature = "onnx")]
use crane_core::models::paddleocr_v6::PaddleOcrV6Pipeline;
#[cfg(feature = "onnx")]
pub use crane_core::models::paddleocr_v6::{OcrDocument, OcrRegion};

#[cfg(not(feature = "onnx"))]
#[derive(Debug, Clone)]
pub struct OcrRegion {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
    pub text: String,
    pub confidence: f32,
}

#[cfg(not(feature = "onnx"))]
#[derive(Debug, Clone)]
pub struct OcrDocument {
    pub text: String,
    pub regions: Vec<OcrRegion>,
}

enum OcrBackend {
    Vl(PaddleOcrVL),
    #[cfg(feature = "onnx")]
    V6(PaddleOcrV6Pipeline),
}

/// Unified OCR client. Model-specific processing is delegated to its backend.
pub struct OcrClient {
    backend: OcrBackend,
}

impl OcrClient {
    pub fn new(config: CommonConfig) -> CraneResult<Self> {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .try_init();
        let backend = match config.model_type {
            LlmModelType::PaddleOcrV6 => {
                #[cfg(feature = "onnx")]
                {
                    OcrBackend::V6(
                        PaddleOcrV6Pipeline::from_dir(&config.model_path)
                            .map_err(|e| CraneError::ModelError(e.to_string()))?,
                    )
                }
                #[cfg(not(feature = "onnx"))]
                {
                    return Err(CraneError::ConfigError(
                        "PaddleOCR v6 requires the `onnx` feature".into(),
                    ));
                }
            },
            _ => {
                let use_cpu = matches!(config.device, DeviceConfig::Cpu);
                let use_bf16 = matches!(config.dtype, DataType::BF16) && !use_cpu;
                let model = PaddleOcrVL::from_local(&config.model_path, use_cpu, use_bf16)
                    .map_err(|e| {
                        CraneError::ModelError(format!("failed to load PaddleOCR-VL: {e}"))
                    })?;
                OcrBackend::Vl(model)
            },
        };
        Ok(Self { backend })
    }

    pub fn extract_text_from_image<P: AsRef<Path>>(
        &mut self,
        image_path: P,
    ) -> CraneResult<String> {
        Ok(self.extract_text_with_locations(image_path)?.text)
    }

    pub fn extract_text_with_locations<P: AsRef<Path>>(
        &mut self,
        image_path: P,
    ) -> CraneResult<OcrDocument> {
        match &mut self.backend {
            OcrBackend::Vl(model) => {
                let result = model
                    .recognize(image_path.as_ref(), OcrTask::Ocr, 896)
                    .map_err(|e| CraneError::ModelError(format!("PaddleOCR-VL failed: {e}")))?;
                Ok(OcrDocument {
                    text: result.text.trim().to_owned(),
                    regions: Vec::new(),
                })
            },
            #[cfg(feature = "onnx")]
            OcrBackend::V6(pipeline) => pipeline
                .recognize(image_path)
                .map_err(|e| CraneError::ModelError(e.to_string())),
        }
    }

    pub fn extract_text_from_image_stream<P: AsRef<Path>>(
        &mut self,
        image_path: P,
    ) -> CraneResult<String> {
        self.extract_with_task_stream(image_path, OcrTask::Ocr, 896, |token| print!("{token}"))
    }

    pub fn extract_with_task<P: AsRef<Path>>(
        &mut self,
        image_path: P,
        task: OcrTask,
        max_new_tokens: usize,
    ) -> CraneResult<String> {
        match &mut self.backend {
            OcrBackend::Vl(model) => Ok(model
                .recognize(image_path.as_ref(), task, max_new_tokens)
                .map_err(|e| CraneError::ModelError(format!("PaddleOCR-VL failed: {e}")))?
                .text
                .trim()
                .to_owned()),
            #[cfg(feature = "onnx")]
            OcrBackend::V6(pipeline) => Ok(pipeline
                .recognize(image_path)
                .map_err(|e| CraneError::ModelError(e.to_string()))?
                .text),
        }
    }

    pub fn extract_with_task_stream<P: AsRef<Path>, F>(
        &mut self,
        image_path: P,
        task: OcrTask,
        max_new_tokens: usize,
        callback: F,
    ) -> CraneResult<String>
    where
        F: Fn(&str),
    {
        match &mut self.backend {
            OcrBackend::Vl(model) => Ok(model
                .recognize_stream(image_path.as_ref(), task, max_new_tokens, callback)
                .map_err(|e| CraneError::ModelError(format!("PaddleOCR-VL failed: {e}")))?
                .text
                .trim()
                .to_owned()),
            #[cfg(feature = "onnx")]
            OcrBackend::V6(pipeline) => {
                let text = pipeline
                    .recognize(image_path)
                    .map_err(|e| CraneError::ModelError(e.to_string()))?
                    .text;
                callback(&text);
                Ok(text)
            },
        }
    }

    pub fn paddleocr_vl_model_mut(&mut self) -> Option<&mut PaddleOcrVL> {
        match &mut self.backend {
            OcrBackend::Vl(model) => Some(model),
            #[cfg(feature = "onnx")]
            OcrBackend::V6(_) => None,
        }
    }
}
