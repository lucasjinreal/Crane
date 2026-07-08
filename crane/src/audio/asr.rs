use crate::common::{
    config::{CommonConfig, DataType, DeviceConfig},
    CraneError, CraneResult,
};
use std::path::Path;

/// Automatic Speech Recognition client
pub struct AsrClient {
    config: CommonConfig,
    // Store the actual ASR model here
}

impl AsrClient {
    /// Create a new ASR client with the given configuration
    pub fn new(config: CommonConfig) -> CraneResult<Self> {
        Ok(Self { config })
    }

    /// Transcribe audio from a file
    pub fn transcribe_from_file<P: AsRef<Path>>(&self, audio_file: P) -> CraneResult<String> {
        let device = match &self.config.device {
            DeviceConfig::Cpu => crane_core::models::Device::Cpu,
            DeviceConfig::Cuda(gpu_id) => {
                crane_core::models::Device::cuda_if_available(*gpu_id as usize)
                    .map_err(|e| CraneError::ModelError(e.to_string()))?
            }
            DeviceConfig::Metal => {
                #[cfg(target_os = "macos")]
                {
                    crane_core::models::Device::new_metal(0)
                        .map_err(|e| CraneError::ModelError(e.to_string()))?
                }
                #[cfg(not(target_os = "macos"))]
                {
                    return Err(CraneError::ConfigError(
                        "Metal device not available on this platform".to_string(),
                    ));
                }
            }
        };

        let model_dir = &self.config.model_path;
        let model_name = if model_dir.contains("base") {
            "base"
        } else {
            "tiny"
        };

        let model = crane_core::models::moonshine_asr::MoonshineASR::new(
            model_dir, model_name, None, &device,
        )
        .map_err(|e| CraneError::ModelError(e.to_string()))?;

        let audio = load_wav_16k_mono(audio_file.as_ref())?;

        let token_ids = model
            .generate(&audio, None)
            .map_err(|e| CraneError::ModelError(e.to_string()))?;

        Ok(format!("{token_ids:?}"))
    }

    /// Transcribe audio from raw audio data (placeholder implementation)
    pub fn transcribe_from_data(&self, _audio_data: &[u8]) -> CraneResult<String> {
        // This would be implemented based on the specific ASR model requirements
        Err(CraneError::Other(
            "Audio data input not implemented yet".to_string(),
        ))
    }
}

/// Loads a 16kHz mono 16-bit PCM WAV file as f32 samples in `[-1, 1]`.
///
/// Moonshine ASR requires this exact format.
fn load_wav_16k_mono(path: &Path) -> CraneResult<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| CraneError::Other(format!("failed to open WAV file: {e}")))?;
    let spec = reader.spec();
    if spec.sample_rate != 16000 || spec.channels != 1 || spec.bits_per_sample != 16 {
        return Err(CraneError::Other(
            "unsupported audio format: expected 16kHz mono 16-bit".to_string(),
        ));
    }
    reader
        .samples::<i16>()
        .map(|s| s.map(|v| f32::from(v) / 32768.0))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| CraneError::Other(format!("failed to read WAV samples: {e}")))
}
