use serde::{Deserialize, Serialize};

/// Generation configuration for LLMs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub repetition_penalty: f32,
    pub repeat_last_n: usize,
    pub do_sample: bool,
    pub pad_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub report_speed: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 245,
            temperature: Some(0.67),
            top_p: Some(1.0),
            repetition_penalty: 1.0,
            repeat_last_n: 5,
            do_sample: false,
            pad_token_id: None,
            eos_token_id: None,
            report_speed: false,
        }
    }
}

impl GenerationConfig {
    pub fn with_max_tokens(max: usize) -> Self {
        Self {
            max_new_tokens: max,
            ..Default::default()
        }
    }
}

/// Type of model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LlmModelType {
    Qwen25,
    Qwen3,
    Qwen3VL,
    DeepSeek,
    HunyuanDense,
    PaddleOcrVl,
    MoonshineAsr,
    Vision,
    // Add more as needed
}
