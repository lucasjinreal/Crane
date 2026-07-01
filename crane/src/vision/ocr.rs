use crate::common::{
    config::{CommonConfig, DataType, DeviceConfig},
    CraneError, CraneResult,
};
use crane_core::models::paddleocr_vl::{OcrTask, PaddleOcrVL};
use ribo::utils::log::info;
use std::path::Path;

pub struct OcrClient {
    config: CommonConfig,
    model: PaddleOcrVL,
}

impl OcrClient {
    pub fn new(config: CommonConfig) -> CraneResult<Self> {
        let mut use_cpu = matches!(config.device, DeviceConfig::Cpu);
        let mut use_bf16 = matches!(config.dtype, DataType::BF16) && !use_cpu;

        ribo::utils::log::init_log(ribo::utils::log::LogLevel::INFO);

        if crane_core::utils::cuda_is_available() && use_cpu {
            info!("Warning: CUDA is available but CPU device is selected.");
        }
        if !crane_core::utils::cuda_is_available() && !use_cpu {
            info!(
                "Warning: CUDA is not available but a GPU device is selected. Falling back to CPU."
            );
            use_cpu = true;
            use_bf16 = false;
        }

        info!(
            "cuda available: {}, use_cpu: {}, use_bf16: {}",
            crane_core::utils::cuda_is_available(),
            use_cpu,
            use_bf16
        );
        info!("device: {}, dtype: {}", config.device, config.dtype);

        if std::path::Path::new(&config.model_path).exists() == false {
            return Err(CraneError::ConfigError(format!(
                "Model path does not exist: {}",
                config.model_path
            )));
        }

        let model = PaddleOcrVL::from_local(&config.model_path, use_cpu, use_bf16)
            .map_err(|e| CraneError::ModelError(format!("Failed to load model: {}", e)))?;

        info!("Loaded PaddleOCR-VL model from {}", config.model_path);
        info!("model device: {:?}", model.device);

        #[cfg(target_os = "macos")]
        if let DeviceConfig::Metal = config.device {
            if !model.device.is_metal() {
                return Err(CraneError::ConfigError(
                    "Metal requested but not available".to_string(),
                ));
            }
        }

        Ok(Self { config, model })
    }

    pub fn extract_text_from_image<P: AsRef<Path>>(
        &mut self,
        image_path: P,
    ) -> CraneResult<String> {
        self.extract_with_task(image_path, OcrTask::Ocr, 896)
    }

    pub fn extract_text_from_image_stream<P: AsRef<Path>>(
        &mut self,
        image_path: P,
    ) -> CraneResult<String> {
        self.extract_with_task_stream(image_path, OcrTask::Ocr, 896, |token| {
            print!("{}", token);
        })
    }

    pub fn extract_with_task<P: AsRef<Path>>(
        &mut self,
        image_path: P,
        task: OcrTask,
        max_new_tokens: usize,
    ) -> CraneResult<String> {
        let result = self
            .model
            .recognize(image_path.as_ref(), task, max_new_tokens)
            .map_err(|e| CraneError::ModelError(format!("PaddleOCR-VL failed: {}", e)))?;

        Ok(result.text.trim().to_string())
    }

    pub fn extract_with_task_stream<P: AsRef<Path>, F>(
        &mut self,
        image_path: P,
        task: OcrTask,
        max_new_tokens: usize,
        callback: F,
    ) -> CraneResult<String>
    where
        P: AsRef<Path>,
        F: Fn(&str),
    {
        let result = self
            .model
            .recognize_stream(image_path.as_ref(), task, max_new_tokens, callback)
            .map_err(|e| CraneError::ModelError(format!("PaddleOCR-VL failed: {}", e)))?;

        Ok(result.text.trim().to_string())
    }

    // pub fn extract_text_from_data(&mut self, image_data: &[u8]) -> CraneResult<String> {
    //     use std::io::Write;
    //     use tempfile::NamedTempFile;

    //     let mut tmp = NamedTempFile::new().map_err(|e| CraneError::IoError(e.to_string()))?;

    //     tmp.write_all(image_data)
    //         .map_err(|e| CraneError::IoError(e.to_string()))?;

    //     self.extract_text_from_image(tmp.path())
    // }

    pub fn model_mut(&mut self) -> &mut PaddleOcrVL {
        &mut self.model
    }
}
