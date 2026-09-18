pub mod based;
pub mod streamer;

/// Generation options for TTS (text-to-speech) models.
#[derive(Clone, Debug)]
pub struct SpeechOptions {
    /// Maximum number of codec frames to generate before stopping.
    pub max_new_tokens: usize,
    /// Sampling temperature; higher values increase randomness. `None` lets
    /// each model apply its own default.
    pub temperature: Option<f64>,
    /// Nucleus sampling threshold; `None` disables top-p filtering.
    pub top_p: Option<f64>,
    /// Repetition penalty applied to previously generated tokens; `1.0` means no penalty.
    pub repetition_penalty: f32,
    /// Flow-matching / diffusion sampler steps per frame. Only consulted by
    /// models with an iterative sampler (VoxCPM2's CFM decoder); `None` keeps
    /// the model's own default. Lower values trade some quality for a roughly
    /// linear speed-up, since the sampler dominates VoxCPM2 generation cost.
    pub cfm_steps: Option<usize>,
    /// Classifier-free guidance strength for the flow-matching sampler
    /// (VoxCPM2 only). `None` keeps the model's own default.
    pub cfg_scale: Option<f64>,
}

impl Default for SpeechOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 8192,
            temperature: None,
            top_p: None,
            repetition_penalty: 1.05,
            cfm_steps: None,
            cfg_scale: None,
        }
    }
}

/// Generation options for ASR (automatic speech recognition) models.
#[derive(Clone, Debug)]
pub struct TranscribeOptions {
    /// Maximum number of tokens to generate before stopping.
    pub max_new_tokens: usize,
    /// Sampling temperature; higher values increase randomness. `0.0`
    /// (the default) selects greedy (argmax) decoding.
    pub temperature: f64,
    /// Nucleus sampling threshold; `None` disables top-p filtering.
    pub top_p: Option<f64>,
    /// Repetition penalty applied to previously generated tokens; `1.0` means no penalty.
    pub repetition_penalty: f32,
    /// Number of most recent tokens considered for the repetition penalty.
    pub repeat_last_n: usize,
    /// Optional language hint (`"auto"` for detection, else e.g. `"en"`/`"zh"`).
    pub language: Option<String>,
}

impl Default for TranscribeOptions {
    /// Defaults to greedy decoding (`temperature: 0.0`): unlike TTS, ASR has
    /// one correct transcription per utterance, so stochastic sampling only
    /// adds transcription errors — and does so more on lower-confidence
    /// (e.g. smaller) models. Matches reference implementations, which
    /// decode ASR with `do_sample=False`.
    fn default() -> Self {
        Self {
            max_new_tokens: 8192,
            temperature: 0.0,
            top_p: None,
            repetition_penalty: 1.05,
            repeat_last_n: 64,
            language: None,
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
    /// Reasoning-mode switch handed to the chat template as `enable_thinking`.
    ///
    /// `None` leaves it undefined so the template's own default wins — which
    /// differs between checkpoints of the same model (official Qwen 3.5
    /// safetensors default to reasoning ON, unsloth's GGUF template to OFF).
    /// Set it explicitly to get the same behaviour from both.
    pub enable_thinking: Option<bool>,
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
            enable_thinking: None,
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
