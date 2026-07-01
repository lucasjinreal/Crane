pub mod based;
pub mod streamer;

/// Generation options for TTS (text-to-speech) models.
#[derive(Clone, Debug)]
pub struct SpeechOptions {
    /// Maximum number of codec frames to generate before stopping.
    pub max_new_tokens: usize,
    /// Sampling temperature; higher values increase randomness.
    pub temperature: f64,
    /// Nucleus sampling threshold; `None` disables top-p filtering.
    pub top_p: Option<f64>,
    /// Repetition penalty applied to previously generated tokens; `1.0` means no penalty.
    pub repetition_penalty: f32,
}

impl Default for SpeechOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 8192,
            temperature: 0.9,
            top_p: None,
            repetition_penalty: 1.05,
        }
    }
}

#[derive(Clone, Debug)]
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
