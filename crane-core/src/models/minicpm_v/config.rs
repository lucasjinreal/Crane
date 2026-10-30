//! HF-compatible config types for MiniCPM-V-4.6.
//!
//! The text backbone (`text_config`, `"model_type": "qwen3_5_text"`) is a
//! field-for-field match with [`crate::models::qwen3_5::config::TextConfig`]
//! — MiniCPM-V-4.6 reuses Qwen 3.5's hybrid GDN/full-attention decoder
//! verbatim — so it's reused directly instead of duplicated.
//!
//! The vision tower (`vision_config`) is architecturally unrelated to Qwen's
//! own `ViT` (Conv3d + `PatchMerger`) and to MiniCPM-o's classic Perceiver
//! `Resampler`; see `vision.rs` / `merger.rs` for the actual (`NaViT` + mid-stack
//! window-attention merger + hierarchical downsample) design.

use serde::Deserialize;

use crate::models::qwen3_5::TextConfig;

/// Vision tower config. Mirrors HF's `MiniCPMV4_6VisionConfig`.
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    /// Native position-embedding grid resolution (980 -> 70x70 = 4900
    /// learned positions). NOT the runtime image resolution — actual images
    /// are resized per `preprocessor_config.json`'s `scale_resolution` and
    /// their positions bucketized against this native grid (see
    /// `vision::VisionEmbeddings`).
    pub image_size: usize,
    pub patch_size: usize,
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub attention_dropout: f64,
}

impl VisionConfig {
    #[must_use]
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    #[must_use]
    pub fn num_patches(&self) -> usize {
        self.num_patches_per_side() * self.num_patches_per_side()
    }

    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

fn default_insert_layer_id() -> usize {
    6
}

fn default_window_kernel_size() -> (usize, usize) {
    (2, 2)
}

fn default_merge_kernel_size() -> (usize, usize) {
    (2, 2)
}

fn default_merger_times() -> usize {
    1
}

fn default_downsample_mode() -> String {
    "16x".to_string()
}

fn default_true() -> bool {
    true
}

/// Top-level MiniCPM-V-4.6 config. Mirrors HF's `MiniCPMV4_6Config`.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    #[serde(default)]
    pub video_token_id: Option<u32>,
    /// Vision encoder layer index (0-based) after which the window-attention
    /// merger runs. HF's `MiniCPMV4_6Config.__post_init__` copies this same
    /// top-level value into `vision_config.insert_layer_id` — the JSON's
    /// `vision_config` block doesn't carry its own copy, so we read it once
    /// here rather than expecting it inside `vision_config`.
    #[serde(default = "default_insert_layer_id")]
    pub insert_layer_id: usize,
    /// Window size `(h, w)` for the intermediate window-attention merger.
    /// Not present in the checkpoint's `vision_config` JSON (relies on the
    /// HF dataclass default); kept here since it drives the same tower.
    #[serde(default = "default_window_kernel_size")]
    pub window_kernel_size: (usize, usize),
    /// Kernel size `(h, w)` for the hierarchical `Merger`'s spatial-concat MLP.
    #[serde(default = "default_merge_kernel_size")]
    pub merge_kernel_size: (usize, usize),
    /// Number of iterative merge rounds in the `Merger`.
    #[serde(default = "default_merger_times")]
    pub merger_times: usize,
    /// `"16x"` (default, full merge pipeline) or `"4x"` (skip the `vit_merger`,
    /// keep 4x more visual tokens). Only `"16x"` is exercised by this MVP.
    #[serde(default = "default_downsample_mode")]
    pub downsample_mode: String,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    /// 4.7+: `"canvas"` selects canvas M-RoPE ([`super::canvas_rope`]); absent
    /// (4.6) means plain sequential positions.
    #[serde(default)]
    pub mrope_mode: Option<String>,
    /// 4.7+ marker token ids the canvas layout is anchored on.
    #[serde(default)]
    pub image_start_id: Option<u32>,
    #[serde(default)]
    pub image_end_id: Option<u32>,
    #[serde(default)]
    pub slice_start_id: Option<u32>,
    #[serde(default)]
    pub slice_end_id: Option<u32>,
    #[serde(default)]
    pub newline_id: Option<u32>,
}

impl Config {
    /// Marker ids for canvas M-RoPE, or `None` when this checkpoint uses plain
    /// sequential positions (4.6) — or declares `canvas` without all ids.
    #[must_use]
    pub fn canvas_token_ids(&self) -> Option<super::canvas_rope::CanvasTokenIds> {
        if self.mrope_mode.as_deref() != Some("canvas") {
            return None;
        }
        Some(super::canvas_rope::CanvasTokenIds {
            image_start: self.image_start_id?,
            image_end: self.image_end_id?,
            slice_start: self.slice_start_id?,
            slice_end: self.slice_end_id?,
            newline: self.newline_id?,
        })
    }
}

/// Load a HF `config.json` for a MiniCPM-V-4.6/4.7 checkpoint.
///
/// # Errors
///
/// Returns an error if `path` cannot be read or its contents aren't valid
/// JSON matching [`Config`]'s schema.
pub fn load_config(path: &str) -> candle_core::Result<Config> {
    let data = std::fs::read(path)
        .map_err(|e| candle_core::Error::Msg(format!("read config {path}: {e}")))?;
    let cfg: Config = serde_json::from_slice(&data)
        .map_err(|e| candle_core::Error::Msg(format!("parse config {path}: {e}")))?;
    Ok(cfg)
}
