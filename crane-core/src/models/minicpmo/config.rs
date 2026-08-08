//! MiniCPM-o-4.5 top-level config.
//!
//! `MiniCPMOConfig` in the reference implementation (`configuration_minicpmo.py`)
//! subclasses `Qwen3Config` directly — the LLM tower's hyperparameters live at
//! the JSON root alongside the nested `vision_config`/`audio_config`/`tts_config`
//! sub-model configs, rather than under their own `llm_config` key. `#[serde(flatten)]`
//! mirrors that: `llm` picks up the root-level fields `qwen3::modeling::Config`
//! expects, ignoring the sibling keys it doesn't know about.
//!
//! The nested sub-model configs are kept as untyped JSON for now and will gain
//! typed structs as the vision/audio/tts phases are implemented (see the
//! `minicpmo` module doc for phase status).

use serde::Deserialize;

use crate::models::minicpm_v::config::VisionConfig;
use crate::models::qwen3::modeling::Config as Qwen3Config;

/// Whisper-medium encoder config (`config.json`'s `audio_config`). Mirrors
/// HF's `WhisperConfig`, only the fields the encoder (not decoder) needs.
#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    pub d_model: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub encoder_layers: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
}

/// `MiniCPMTTS`'s Llama backbone config (`config.json`'s `tts_config`).
/// Mirrors `MiniCPMTTSConfig`'s fields that actually get consumed for
/// zero-shot (no reference audio) non-streaming generation — `backbone_model`
/// is always `"llama"` for real checkpoints, so a `LlamaConfig` is built
/// directly from these; RoPE theta/eps aren't in `MiniCPMTTSConfig` at all,
/// so HF's `LlamaConfig` defaults (`rope_theta=10000.0`, `rms_norm_eps=1e-6`)
/// apply — see `tts_llm.rs`.
#[derive(Debug, Clone, Deserialize)]
pub struct TtsConfig {
    pub llm_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub num_audio_tokens: usize,
    pub num_text_tokens: usize,
    pub num_vq: usize,
    pub audio_bos_token_id: u32,
    pub text_eos_token_id: u32,
    #[serde(default)]
    pub normalize_projected_hidden: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MiniCpmOConfig {
    #[serde(flatten)]
    pub llm: Qwen3Config,

    pub vision_config: VisionConfig,
    pub audio_config: AudioConfig,
    pub tts_config: TtsConfig,

    /// Perceiver `Resampler` query count — the fixed number of vision
    /// tokens each image is compressed to.
    #[serde(default)]
    pub query_num: usize,
    /// `AvgPool1d` stride applied to projected audio embeddings.
    #[serde(default)]
    pub audio_pool_step: usize,

    #[serde(default)]
    pub init_vision: bool,
    #[serde(default)]
    pub init_audio: bool,
    #[serde(default)]
    pub init_tts: bool,
}

/// Load and parse `config.json` from a MiniCPM-o-4.5 checkpoint directory.
///
/// # Errors
///
/// Returns an error if the file can't be read or doesn't match the expected schema.
pub fn load_config(config_path: &str) -> anyhow::Result<MiniCpmOConfig> {
    let data = std::fs::read(config_path)?;
    Ok(serde_json::from_slice(&data)?)
}
