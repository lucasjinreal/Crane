//! HF-compatible config types for Qwen 3.5 text-only inference.
//!
//! Qwen 3.5 ships as `Qwen3_5ForConditionalGeneration` (multimodal class) but
//! the dense text-only checkpoints have no vision weights — we deserialize the
//! nested `text_config` and ignore the vision block when loading weights.
//!
//! The Qwen 3.6 / 3.8 27B dense checkpoints declare the *same*
//! `model_type: "qwen3_5"` and are the same architecture scaled up (64 layers,
//! `hidden_size` 5120, 48 GDN value heads, untied `lm_head`), so they load
//! through these same types — every difference is a config value.

use candle_core::Result;
use serde::Deserialize;

use crate::models::modules::quant_kv_cache;
use crate::ops::gdn::{GdnConfig, defaults};

/// Whether a transformer block at layer index `i` is full (softmax) attention
/// or linear (Gated Delta Net) attention.
///
/// Layer indices run `0..num_hidden_layers`. With `full_attention_interval = 4`
/// the layout is `[linear, linear, linear, full, linear, linear, linear, full, …]`,
/// so `full_attention_interval - 1` linear layers precede every full layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

/// `RopeParameters` block nested under `text_config.rope_parameters` in HF
/// `config.json`. `mrope_interleaved: true` is the Qwen 3.5 variant (vs the
/// non-interleaved `MRoPE` used by Qwen 3 VL).
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "defaults::rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default = "defaults::partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    /// Default false (i.e. standard `MRoPE`). Qwen 3.5 sets this to `true`.
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

    /// Activation of the **Gated Delta Net** output gate (HF `RMSNormGated`),
    /// *not* the softmax-attention gate above — that one is always sigmoid.
    ///
    /// Absent in Qwen 3.5 (implicitly swish); the Qwen 3.6/3.8 27B configs
    /// spell it out as `"swish"`. Only swish/silu is supported, which is what
    /// [`crate::ops::gdn::RmsNormGated`] already computes, so a present-and-
    /// swish value is a no-op. Validated by [`TextConfig::validate`].
    #[serde(default)]
    pub output_gate_type: Option<String>,
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
    pub vision_config: Option<VisionConfig>,
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

/// Qwen 3.5 vision tower config. The architecture is identical to the
/// Qwen2.5-VL / Qwen3-VL `ViT`: Conv3d patch embed (temporal×spatial×spatial)
/// with bias, 12-layer transformer with per-block `LayerNorm`, fast (flash)
/// attention, and a `PatchMerger` MLP that 2×2-spacially-merges the patch
/// grid and projects to `out_hidden_size` (== text hidden size for Qwen 3.5).
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub out_hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    #[serde(default)]
    pub num_position_embeddings: usize,
    /// `ViT` MLP activation (Qwen 3.5 ships `gelu_pytorch_tanh` — exact GELU with
    /// the tanh approximation, as used by every Qwen2.5/3 `ViT`).
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: VisionHiddenAct,
    /// Qwen 3.5 always sets this to `[]` (no deepstack injection).
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
}

fn default_vision_hidden_act() -> VisionHiddenAct {
    VisionHiddenAct::GeluPytorchTanh
}

/// Vision MLP activation function. Only `gelu_pytorch_tanh` is observed in
/// the wild for the Qwen `ViT` family, but we keep the enum open.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionHiddenAct {
    Gelu,
    GeluPytorchTanh,
    Relu,
    Silu,
}

impl VisionConfig {
    #[must_use]
    pub fn merged_hidden_size(&self) -> usize {
        self.hidden_size * self.spatial_merge_size.pow(2)
    }
}

impl VisionHiddenAct {
    /// Map to `candle_nn::Activation`.
    pub fn to_activation(self) -> candle_nn::Activation {
        use candle_nn::Activation;
        match self {
            // `gelu_pytorch_tanh`: exact GELU with the tanh approximation,
            // matching nn.GELU(approximate='tanh') used by the Qwen ViT MLP.
            VisionHiddenAct::Gelu | VisionHiddenAct::GeluPytorchTanh => Activation::Gelu,
            VisionHiddenAct::Relu => Activation::Relu,
            VisionHiddenAct::Silu => Activation::Silu,
        }
    }
}

impl Config {
    #[must_use]
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
    /// Reject configs whose semantics this implementation does not match.
    ///
    /// The only such knob today is [`Self::output_gate_type`]: the GDN gate is
    /// hardwired to swish, so any other activation would silently produce
    /// wrong activations rather than fail loudly.
    ///
    /// # Errors
    ///
    /// Returns an error if `output_gate_type` is set to an unsupported activation.
    pub fn validate(&self) -> Result<()> {
        if let Some(gate) = &self.output_gate_type
            && !matches!(gate.as_str(), "swish" | "silu")
        {
            candle_core::bail!(
                "[qwen3_5] unsupported output_gate_type {gate:?}: the GDN output gate is \
                 implemented as swish (== silu) only"
            );
        }
        Ok(())
    }

    #[must_use]
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.rope_theta
    }

    #[must_use]
    pub fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters.partial_rotary_factor
    }

    #[must_use]
    pub fn mrope_section(&self) -> &[usize] {
        &self.rope_parameters.mrope_section
    }

    #[must_use]
    pub fn mrope_interleaved(&self) -> bool {
        self.rope_parameters.mrope_interleaved
    }

    /// `head_dim * partial_rotary_factor` — the slice of the head dim that
    /// actually receives rotary embeddings.
    #[must_use]
    pub fn rot_dim(&self) -> usize {
        // head_dim is a small model dimension (e.g. <=512); the f64 round-trip
        // through partial_rotary_factor cannot lose precision or sign here.
        #[allow(clippy::cast_precision_loss)]
        let head_dim = self.head_dim as f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let rot_dim = (head_dim * self.partial_rotary_factor()) as usize;
        rot_dim
    }

    /// Layer-type sequence, indexed by transformer-block position.
    #[must_use]
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

    /// Number of layers that carry a full-attention K/V cache — every
    /// `full_attention_interval`-th layer (see [`Self::layer_types`]). The
    /// rest are `GDN` linear-attention layers with recurrent state instead
    /// of a growing K/V cache.
    #[must_use]
    pub fn num_full_attention_layers(&self) -> usize {
        self.num_hidden_layers / self.full_attention_interval
    }

    /// Bytes of K/V cache one sequence consumes per generated token, at the
    /// compute dtype. Only counts full-attention layers ([`Self::num_full_attention_layers`]) —
    /// `GDN` layers don't grow a per-token K/V cache.
    #[must_use]
    pub fn kv_bytes_per_token(&self, dtype_bytes: usize) -> u64 {
        2 * self.num_full_attention_layers() as u64
            * self.num_key_value_heads as u64
            * self.head_dim as u64
            * dtype_bytes as u64
    }

    /// Bytes of K/V cache one sequence consumes per generated token when K/V
    /// are stored as `bits`-wide (4 or 8) quantized codes plus a per-token
    /// f32 scale per head, instead of the compute dtype (see
    /// [`crate::models::modules::quant_kv_cache::QuantKvCache`]). Only an
    /// accurate estimate when the fused dequantize-in-attention kernel
    /// (`crate::ops::fused_ops::quant_attn`) covers the sequence's entire
    /// lifetime — see [`Self::kv_bytes_per_token`]'s doc for why plain
    /// compute-dtype pricing is required otherwise.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is neither 4 nor 8.
    #[must_use]
    pub fn quantized_kv_bytes_per_token(&self, bits: u32) -> u64 {
        quant_kv_cache::quantized_kv_bytes_per_token(
            bits,
            self.num_full_attention_layers(),
            self.num_key_value_heads,
            self.head_dim,
        )
    }

    #[must_use]
    pub fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    #[must_use]
    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    #[must_use]
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
///
/// # Errors
///
/// Returns an error if the file cannot be read or does not parse as a valid config.
pub fn load_config(path: &str) -> Result<Config> {
    let data = std::fs::read(path)
        .map_err(|e| candle_core::Error::Msg(format!("read config {path}: {e}")))?;
    let cfg: Config = serde_json::from_slice(&data)
        .map_err(|e| candle_core::Error::Msg(format!("parse config {path}: {e}")))?;
    cfg.text_config.validate()?;
    Ok(cfg)
}
#[cfg(test)]
mod tests {
    use super::*;

    /// The Qwen 3.8-27B (and, identically, Qwen 3.6-27B) `text_config`, minus
    /// the keys this implementation ignores.
    const QWEN3_8_27B_TEXT: &str = r#"{
        "head_dim": 256,
        "vocab_size": 248320,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "hidden_act": "silu",
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-06,
        "attn_output_gate": true,
        "output_gate_type": "swish",
        "full_attention_interval": 4,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "tie_word_embeddings": false,
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000
        }
    }"#;

    fn qwen3_8() -> TextConfig {
        serde_json::from_str(QWEN3_8_27B_TEXT).expect("parse Qwen3.8-27B text_config")
    }

    /// Qwen 3.8 parses through the Qwen 3.5 types unchanged, and the derived
    /// dimensions come out at the shapes the checkpoint actually ships.
    #[test]
    fn qwen3_8_27b_config_derives_checkpoint_shapes() {
        let cfg = qwen3_8();
        cfg.validate().unwrap();

        // `q_proj` is [12288, 5120] on disk: 2 x heads x head_dim (gated).
        assert!(cfg.attn_output_gate);
        assert_eq!(2 * cfg.num_attention_heads * cfg.head_dim, 12288);
        // `in_proj_qkv` is [10240, 5120], `in_proj_z` is [6144, 5120].
        assert_eq!(cfg.linear_conv_dim(), 10240);
        assert_eq!(cfg.linear_value_dim(), 6144);
        assert_eq!(cfg.linear_key_dim(), 2048);
        // rope.dimension_count 64 = head_dim * partial_rotary_factor, and the
        // mrope sections tile half of it.
        assert_eq!(cfg.rot_dim(), 64);
        assert_eq!(cfg.mrope_section().iter().sum::<usize>(), cfg.rot_dim() / 2);
        assert!(cfg.mrope_interleaved());
        // 16 of 64 layers are full attention, the last of every group of 4.
        let types = cfg.layer_types();
        assert_eq!(types.len(), 64);
        assert_eq!(
            types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count(),
            16
        );
        assert_eq!(types[3], LayerType::FullAttention);
        assert_eq!(types[2], LayerType::LinearAttention);
        // Three value heads per key head — the case that never arises on the
        // Qwen 3.5 0.8B (1) or 4B (2).
        assert_eq!(cfg.linear_num_value_heads / cfg.linear_num_key_heads, 3);
    }

    /// `output_gate_type` names the GDN gate, which is swish-only here. Qwen
    /// 3.5 omits the key entirely.
    #[test]
    fn output_gate_type_accepts_swish_and_absent() {
        assert_eq!(qwen3_8().output_gate_type.as_deref(), Some("swish"));
        qwen3_8().validate().unwrap();

        let mut cfg = qwen3_8();
        cfg.output_gate_type = Some("silu".into());
        cfg.validate().unwrap();

        cfg.output_gate_type = None;
        cfg.validate().unwrap();
    }

    #[test]
    fn output_gate_type_rejects_unimplemented_activation() {
        let mut cfg = qwen3_8();
        cfg.output_gate_type = Some("sigmoid".into());
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("output_gate_type"), "unexpected error: {err}");
    }

    // 16 of 64 layers are full attention (matches layer_types()'s count above).
    #[test]
    fn num_full_attention_layers_counts_only_full_attention() {
        assert_eq!(qwen3_8().num_full_attention_layers(), 16);
    }

    // f16: 2 * 16 full-attn layers * 4 kv_heads * 256 head_dim * 2 bytes =
    // 65,536 bytes/token — must not scale with the 64 total (incl. GDN) layers.
    #[test]
    fn kv_bytes_per_token_counts_only_full_attention_layers() {
        assert_eq!(qwen3_8().kv_bytes_per_token(2), 2 * 16 * 4 * 256 * 2);
    }

    // int8: 1 byte/code + 4 byte f32 scale per head -> 2*16*4*(256+4) bytes/token.
    #[test]
    fn quantized_kv_bytes_per_token_int8_matches_qwen3_8_27b() {
        assert_eq!(
            qwen3_8().quantized_kv_bytes_per_token(8),
            2 * 16 * 4 * (256 + 4)
        );
    }

    // int4: nibble-packed codes (head_dim/2 bytes) + 4 byte f32 scale per head.
    #[test]
    fn quantized_kv_bytes_per_token_int4_matches_qwen3_8_27b() {
        assert_eq!(
            qwen3_8().quantized_kv_bytes_per_token(4),
            2 * 16 * 4 * (128 + 4)
        );
    }

    // Both quantized bit widths must cost strictly less than compute-dtype
    // (f16) pricing for the same geometry.
    #[test]
    fn quantized_kv_bytes_per_token_smaller_than_fp16() {
        let cfg = qwen3_8();
        let fp16_bytes = cfg.kv_bytes_per_token(2);
        assert!(cfg.quantized_kv_bytes_per_token(8) < fp16_bytes);
        assert!(cfg.quantized_kv_bytes_per_token(4) < fp16_bytes);
    }

    #[test]
    #[should_panic(expected = "bits must be 4 or 8")]
    fn quantized_kv_bytes_per_token_rejects_invalid_bits() {
        let _ = qwen3_8().quantized_kv_bytes_per_token(16);
    }
}
