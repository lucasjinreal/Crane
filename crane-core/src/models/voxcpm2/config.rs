//! HF-compatible config types for VoxCPM2.
//!
//! Ported from `voxcpm/modules/minicpm4/config.py` (`MiniCPM4Config`,
//! `RopeScalingConfig`) and `voxcpm/model/voxcpm2.py` (`VoxCPMConfig`,
//! `VoxCPMEncoderConfig`, `VoxCPMDitConfig`).
//!
//! Four of VoxCPM2's five sub-networks (`base_lm`, `residual_lm`,
//! `feat_encoder`, `feat_decoder.estimator`) are the *same* decoder-block
//! class (see `minicpm4.rs`) with different [`MiniCpm4Config`] values — this
//! module's job is just deserializing those four variants plus the
//! non-transformer pieces (FSQ, DiT wrapper, CFM sampler params).
//!
//! `dim_model_base` is deserialized (present in every real `config.json`)
//! but deliberately unused — the upstream Python (`minicpm4/model.py`)
//! never reads it either, only `scale_emb`/`scale_depth`/`use_mup`.

use serde::Deserialize;

/// LongRoPE scaling block. Two regimes (`short_factor` below
/// `original_max_position_embeddings`, `long_factor` above) — see
/// [`super::minicpm4::LongRoPE`] for the frequency-scaling math.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScalingConfig {
    #[serde(rename = "type")]
    pub rope_type: String,
    pub long_factor: Vec<f64>,
    pub short_factor: Vec<f64>,
    pub original_max_position_embeddings: usize,
}

/// Config for one instance of the shared `MiniCPMModel` decoder-block stack
/// (`minicpm4.rs`). `base_lm`/`residual_lm` deserialize this directly from
/// `config.json`'s `lm_config`; `feat_encoder`/`feat_decoder` are built by
/// copying `lm_config` and overriding `hidden_size`/`intermediate_size`/
/// `num_attention_heads`/`num_hidden_layers`/`kv_channels`/`vocab_size` from
/// `encoder_config`/`dit_config` (mirrors `VoxCPM2Model.__init__`'s
/// `model_copy(deep=True)` + field overrides) — see [`Self::derive`].
#[derive(Debug, Clone, Deserialize)]
pub struct MiniCpm4Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: RopeScalingConfig,
    pub rope_theta: f64,
    pub vocab_size: usize,
    #[serde(default = "default_true")]
    pub use_mup: bool,
    pub scale_emb: f64,
    pub scale_depth: f64,
    #[serde(default)]
    pub kv_channels: Option<usize>,
    #[serde(default)]
    pub no_rope: bool,
}

fn default_true() -> bool {
    true
}

impl MiniCpm4Config {
    pub fn head_dim(&self) -> usize {
        self.kv_channels.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Build a derived config for `feat_encoder`/`feat_decoder`: same
    /// `lm_config` base, sized/reshaped per `encoder_config`/`dit_config`,
    /// `vocab_size = 0` (no token embedding — these consume precomputed
    /// hidden vectors, matching HF's `nn.Identity()` embed_tokens when
    /// `vocab_size == 0`).
    pub fn derive(&self, sub: &SubNetworkConfig) -> Self {
        Self {
            hidden_size: sub.hidden_dim,
            intermediate_size: sub.ffn_dim,
            num_attention_heads: sub.num_heads,
            num_hidden_layers: sub.num_layers,
            kv_channels: sub.kv_channels,
            vocab_size: 0,
            ..self.clone()
        }
    }

    /// Build a derived config for `residual_lm`: same `lm_config` base,
    /// `num_hidden_layers` overridden, `vocab_size = 0`, `no_rope` set from
    /// the top-level `residual_lm_no_rope` flag.
    pub fn derive_residual_lm(&self, num_hidden_layers: usize, no_rope: bool) -> Self {
        Self { num_hidden_layers, vocab_size: 0, no_rope, ..self.clone() }
    }
}

/// Shape override for `feat_encoder` (`encoder_config`) or `feat_decoder`
/// (`dit_config`) relative to the shared `lm_config` base.
#[derive(Debug, Clone, Deserialize)]
pub struct SubNetworkConfig {
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    #[serde(default)]
    pub kv_channels: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CfmConfig {
    #[serde(default = "default_sigma_min")]
    pub sigma_min: f64,
    #[serde(default = "default_solver")]
    pub solver: String,
    #[serde(default = "default_t_scheduler")]
    pub t_scheduler: String,
    #[serde(default = "default_inference_cfg_rate")]
    pub inference_cfg_rate: f64,
}

fn default_sigma_min() -> f64 {
    1e-6
}
fn default_solver() -> String {
    "euler".to_string()
}
fn default_t_scheduler() -> String {
    "log-norm".to_string()
}
fn default_inference_cfg_rate() -> f64 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct DitConfig {
    #[serde(flatten)]
    pub shape: SubNetworkConfig,
    #[serde(default)]
    pub mean_mode: bool,
    pub cfm_config: CfmConfig,
}

/// Top-level VoxCPM2 config (`config.json`). `audio_vae_config` is
/// deserialized generically ([`serde_json::Value`]) — the AudioVAE decoder
/// (`audio_vae.rs`) reads the handful of fields it needs (`decoder_rates`,
/// `sr_bin_boundaries`, `latent_dim`, `decoder_dim`, `out_sample_rate`)
/// directly rather than duplicating the full schema here.
#[derive(Debug, Clone, Deserialize)]
pub struct VoxCpm2Config {
    pub lm_config: MiniCpm4Config,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_feat_dim")]
    pub feat_dim: usize,
    #[serde(default = "default_residual_lm_num_layers")]
    pub residual_lm_num_layers: usize,
    #[serde(default)]
    pub residual_lm_no_rope: bool,
    #[serde(default = "default_sq_latent_dim")]
    pub scalar_quantization_latent_dim: usize,
    #[serde(default = "default_sq_scale")]
    pub scalar_quantization_scale: usize,
    pub encoder_config: SubNetworkConfig,
    pub dit_config: DitConfig,
    pub audio_vae_config: serde_json::Value,
    #[serde(default = "default_max_length")]
    pub max_length: usize,
}

fn default_patch_size() -> usize {
    4
}
fn default_feat_dim() -> usize {
    64
}
fn default_residual_lm_num_layers() -> usize {
    8
}
fn default_sq_latent_dim() -> usize {
    512
}
fn default_sq_scale() -> usize {
    9
}
fn default_max_length() -> usize {
    8192
}

/// Load `config.json` for a VoxCPM2 checkpoint.
pub fn load_config(path: &str) -> candle_core::Result<VoxCpm2Config> {
    let data = std::fs::read(path)
        .map_err(|e| candle_core::Error::Msg(format!("read config {path}: {e}")))?;
    let cfg: VoxCpm2Config = serde_json::from_slice(&data)
        .map_err(|e| candle_core::Error::Msg(format!("parse config {path}: {e}")))?;
    Ok(cfg)
}
