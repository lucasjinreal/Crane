//! HF-compatible config types for Qwen 3.5 text-only inference.
//!
//! Qwen 3.5 ships as `Qwen3_5ForConditionalGeneration` (multimodal class) but
//! the dense text-only checkpoints have no vision weights — we deserialize the
//! nested `text_config` and ignore the vision block when loading weights.

use candle_core::Result;
use serde::Deserialize;

use crate::ops::gdn::{defaults, GdnConfig};

/// Whether a transformer block at layer index `i` is full (softmax) attention
/// or linear (Gated Delta Net) attention.
///
/// Layer indices run 0..num_hidden_layers. With `full_attention_interval = 4`
/// the layout is `[linear, linear, linear, full, linear, linear, linear, full, …]`,
/// so `full_attention_interval - 1` linear layers precede every full layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

/// `RopeParameters` block nested under `text_config.rope_parameters` in HF
/// `config.json`. `mrope_interleaved: true` is the Qwen 3.5 variant (vs the
/// non-interleaved MRoPE used by Qwen 3 VL).
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "defaults::rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default = "defaults::partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    /// Default false (i.e. standard MRoPE). Qwen 3.5 sets this to `true`.
    #[serde(default)]
    pub mrope_interleaved: bool,
}

/// Text-only model config. Mirrors HF's `text_config` block under
/// `Qwen3_5ForConditionalGeneration`.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub head_dim: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: HiddenAct,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,

    // Hybrid-attention parameters.
    #[serde(default = "defaults::full_attention_interval")]
    pub full_attention_interval: usize,
    #[serde(default = "defaults::conv_kernel")]
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    // Qwen 3.5 attention uses gated output (sigmoid gate on softmax attention).
    #[serde(default = "default_true")]
    pub attn_output_gate: bool,
}

fn default_true() -> bool {
    true
}

/// Top-level config. `text_config` holds the language model; `vision_config`
/// exists for the multimodal class but is ignored by the text-only path.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    #[serde(default)]
    pub vision_config: Option<serde_json::Value>,
    #[serde(default)]
    pub image_token_id: Option<u32>,
    #[serde(default)]
    pub video_token_id: Option<u32>,
    #[serde(default)]
    pub vision_start_token_id: Option<u32>,
    #[serde(default)]
    pub vision_end_token_id: Option<u32>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn text(&self) -> &TextConfig {
        &self.text_config
    }
}

/// Hidden activation. Qwen 3.5 uses `silu` for the MLP.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    Silu,
    #[serde(other)]
    Other,
}

impl TextConfig {
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.rope_theta
    }

    pub fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters.partial_rotary_factor
    }

    pub fn mrope_section(&self) -> &[usize] {
        &self.rope_parameters.mrope_section
    }

    pub fn mrope_interleaved(&self) -> bool {
        self.rope_parameters.mrope_interleaved
    }

    /// `head_dim * partial_rotary_factor` — the slice of the head dim that
    /// actually receives rotary embeddings.
    pub fn rot_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor()) as usize
    }

    /// Layer-type sequence, indexed by transformer-block position.
    pub fn layer_types(&self) -> Vec<LayerType> {
        (0..self.num_hidden_layers)
            .map(|i| {
                if (i + 1) % self.full_attention_interval == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect()
    }

    pub fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn linear_conv_dim(&self) -> usize {
        2 * self.linear_key_dim() + self.linear_value_dim()
    }
}

impl GdnConfig for TextConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }
    fn linear_conv_kernel_dim(&self) -> usize {
        self.linear_conv_kernel_dim
    }
    fn linear_key_head_dim(&self) -> usize {
        self.linear_key_head_dim
    }
    fn linear_value_head_dim(&self) -> usize {
        self.linear_value_head_dim
    }
    fn linear_num_key_heads(&self) -> usize {
        self.linear_num_key_heads
    }
    fn linear_num_value_heads(&self) -> usize {
        self.linear_num_value_heads
    }
}

/// Load a HF `config.json` for a Qwen 3.5 (text-only) checkpoint.
///
/// Vision config is deserialized into a generic JSON value since the text path
/// never reads it; the caller's responsibility is to gate vision processing.
pub fn load_config(path: &str) -> Result<Config> {
    let data = std::fs::read(path)
        .map_err(|e| candle_core::Error::Msg(format!("read config {path}: {e}")))?;
    let cfg: Config = serde_json::from_slice(&data)
        .map_err(|e| candle_core::Error::Msg(format!("parse config {path}: {e}")))?;
    Ok(cfg)
}