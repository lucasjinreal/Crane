use serde::{Deserialize, Serialize};
use crate::llm::LlmModelType;

/// Common configuration for Crane SDK
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonConfig {
    /// Path to model checkpoints
    pub model_path: String,

    /// Which model architecture to load for this checkpoint
    pub model_type: LlmModelType,

    /// Device to run models on (CPU/GPU)
    pub device: DeviceConfig,

    /// Data type for computations
    pub dtype: DataType,

    /// Maximum memory usage
    pub max_memory: Option<usize>,
}

impl Default for CommonConfig {
    fn default() -> Self {
        Self {
            model_path: "checkpoints".to_string(),
            model_type: LlmModelType::Qwen25,
            device: DeviceConfig::Cpu,
            dtype: DataType::F16,
            max_memory: None,
        }
    }
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceConfig {
    Cpu,
    Cuda(u32), // GPU ID
    Metal,
}

impl std::fmt::Display for DeviceConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceConfig::Cpu => write!(f, "Cpu"),
            DeviceConfig::Cuda(gpu_id) => write!(f, "Cuda({})", gpu_id),
            DeviceConfig::Metal => write!(f, "Metal"),
        }
    }
}

/// Data type configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataType {
    F16,
    F32,
    BF16,
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::F16 => write!(f, "F16"),
            DataType::F32 => write!(f, "F32"),
            DataType::BF16 => write!(f, "BF16"),
        }
    }
}
