//! Config structs for Qwen3-ASR's `-hf` checkpoint layout (flat
//! `audio_config`/`text_config` under a `qwen3_asr`-typed root config).

use candle_nn::Activation;
use serde::Deserialize;

/// Rope parameters nested under `text_config.rope_parameters`.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    /// `RoPE` base frequency.
    pub rope_theta: f64,
}

/// Audio encoder (`AuT`) config, `config.json`'s `audio_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    /// Encoder hidden size.
    pub d_model: usize,
    /// Number of transformer encoder layers.
    pub encoder_layers: usize,
    /// Number of attention heads.
    pub encoder_attention_heads: usize,
    /// Number of key/value heads; equal to `encoder_attention_heads` in both
    /// released sizes (plain multi-head attention, not GQA).
    pub num_key_value_heads: usize,
    /// Feed-forward intermediate size.
    pub encoder_ffn_dim: usize,
    /// Output width after the multi-modal projector; matches
    /// `text_config.hidden_size`.
    pub output_dim: usize,
    /// Number of mel filterbank bins.
    pub num_mel_bins: usize,
    /// Channel count of the 3x stride-2 `Conv2d` frontend.
    pub downsample_hidden_size: usize,
    /// Length of the fixed sinusoidal positional table (one window's
    /// post-conv frame count).
    pub max_position_embeddings: usize,
    /// Pre-conv chunk size in raw (100Hz) mel frames; `n_window * 2` raw
    /// frames form one conv/position-embedding chunk.
    pub n_window: usize,
    /// Attention window size in raw (100Hz) mel frames; block-diagonal
    /// attention spans `n_window_infer / (n_window * 2)` conv chunks.
    pub n_window_infer: usize,
    /// Max number of `n_window`-sized chunks batched through the conv
    /// frontend in one call (memory bound only, not an attention window).
    pub conv_chunksize: usize,
    /// Activation function for the encoder MLP and projector; always
    /// `gelu` on released checkpoints.
    pub activation_function: Activation,
    /// Whether to scale embeddings; always `false` on released checkpoints.
    pub scale_embedding: bool,
}

/// Text decoder config, `config.json`'s `text_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Decoder hidden size.
    pub hidden_size: usize,
    /// Feed-forward intermediate size.
    pub intermediate_size: usize,
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (GQA).
    pub num_key_value_heads: usize,
    /// Per-head dimension; fixed independent of `hidden_size`/
    /// `num_attention_heads`, per Qwen3 convention.
    pub head_dim: usize,
    /// Max position count the `RoPE` table is built for.
    pub max_position_embeddings: usize,
    /// `RMSNorm` epsilon.
    pub rms_norm_eps: f64,
    /// Nested `RoPE` parameters.
    pub rope_parameters: RopeParameters,
    /// Whether attention projections carry a bias; always `false`.
    pub attention_bias: bool,
    /// Whether input/output embeddings are tied.
    pub tie_word_embeddings: bool,
    /// Sliding window size, if any.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Number of leading layers using the sliding window (when enabled).
    #[serde(default)]
    pub max_window_layers: usize,
    /// Whether sliding-window attention is enabled.
    #[serde(default)]
    pub use_sliding_window: bool,
}

impl TextConfig {
    /// Converts this checkpoint's `text_config` into the existing
    /// `crane-core::models::qwen3` decoder config, since the ASR text
    /// decoder is tensor-name-identical to standalone Qwen3.
    #[must_use]
    pub fn to_qwen3_config(&self) -> crate::models::qwen3::modeling::Config {
        crate::models::qwen3::modeling::Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: Some(self.head_dim),
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_parameters.rope_theta,
            attention_bias: self.attention_bias,
            use_qk_norm: true,
            tie_word_embeddings: self.tie_word_embeddings,
            sliding_window: self.sliding_window,
            max_window_layers: self.max_window_layers,
            use_sliding_window: self.use_sliding_window,
            // EOS handling lives at the top-level `Config` (`Vec<u32>`, since
            // ASR checkpoints have multiple EOS ids), not per-decoder.
            eos_token_id: None,
        }
    }
}

/// Top-level Qwen3-ASR config, matching the flat `-hf` checkpoint
/// `config.json` layout (`audio_config`/`text_config` at the root, no
/// `thinker_config` wrapper).
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Audio encoder (`AuT`) config.
    pub audio_config: AudioConfig,
    /// Text decoder config.
    pub text_config: TextConfig,
    /// Token id substituted for each expanded `<|audio_pad|>` placeholder.
    pub audio_token_id: u32,
    /// Token id reserved for timestamp prediction (unused in v1, §9).
    pub timestamp_token_id: u32,
    /// Padding token id.
    pub pad_token_id: u32,
    /// End-of-sequence token ids (there can be more than one).
    pub eos_token_id: Vec<u32>,
    /// Whether input/output embeddings are tied at the top level.
    pub tie_word_embeddings: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG_0_6B: &str = r#"{
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "audio_config": {
            "activation_dropout": 0,
            "activation_function": "gelu",
            "attention_dropout": 0,
            "conv_chunksize": 500,
            "d_model": 896,
            "downsample_hidden_size": 480,
            "dropout": 0,
            "encoder_attention_heads": 14,
            "encoder_ffn_dim": 3584,
            "encoder_layers": 18,
            "initializer_range": 0.02,
            "max_position_embeddings": 13,
            "model_type": "qwen3_asr_encoder",
            "n_window": 50,
            "n_window_infer": 800,
            "num_key_value_heads": 14,
            "num_mel_bins": 128,
            "output_dim": 1024,
            "scale_embedding": false
        },
        "audio_token_id": 151676,
        "dtype": "bfloat16",
        "eos_token_id": [151643, 151645],
        "initializer_range": 0.02,
        "model_type": "qwen3_asr",
        "pad_token_id": 151645,
        "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": null,
            "eos_token_id": null,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 1024,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 65536,
            "max_window_layers": 28,
            "model_type": "qwen3",
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "pad_token_id": null,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "rope_theta": 1000000,
                "rope_type": "default"
            },
            "sliding_window": null,
            "tie_word_embeddings": true,
            "use_cache": true,
            "use_sliding_window": false,
            "vocab_size": 151936
        },
        "tie_word_embeddings": true,
        "timestamp_token_id": 151705,
        "token_classification_bias": false,
        "transformers_version": "5.13.0.dev0"
    }"#;

    const CONFIG_1_7B: &str = r#"{
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "audio_config": {
            "activation_dropout": 0,
            "activation_function": "gelu",
            "attention_dropout": 0,
            "conv_chunksize": 500,
            "d_model": 1024,
            "downsample_hidden_size": 480,
            "dropout": 0,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "encoder_layers": 24,
            "initializer_range": 0.02,
            "max_position_embeddings": 13,
            "model_type": "qwen3_asr_encoder",
            "n_window": 50,
            "n_window_infer": 800,
            "num_key_value_heads": 16,
            "num_mel_bins": 128,
            "output_dim": 2048,
            "scale_embedding": false
        },
        "audio_token_id": 151676,
        "dtype": "bfloat16",
        "eos_token_id": [151643, 151645],
        "initializer_range": 0.02,
        "model_type": "qwen3_asr",
        "pad_token_id": 151645,
        "text_config": {
            "attention_bias": false,
            "attention_dropout": 0.0,
            "bos_token_id": null,
            "eos_token_id": null,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2048,
            "initializer_range": 0.02,
            "intermediate_size": 6144,
            "max_position_embeddings": 65536,
            "max_window_layers": 28,
            "model_type": "qwen3",
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 8,
            "pad_token_id": null,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "rope_theta": 1000000,
                "rope_type": "default"
            },
            "sliding_window": null,
            "tie_word_embeddings": true,
            "use_cache": true,
            "use_sliding_window": false,
            "vocab_size": 151936
        },
        "tie_word_embeddings": true,
        "timestamp_token_id": 151705,
        "token_classification_bias": false,
        "transformers_version": "5.13.0.dev0"
    }"#;

    #[test]
    fn deserialize_0_6b_config() {
        let config: Config = serde_json::from_str(CONFIG_0_6B).unwrap();

        assert_eq!(config.audio_token_id, 151_676);
        assert_eq!(config.timestamp_token_id, 151_705);
        assert_eq!(config.pad_token_id, 151_645);
        assert_eq!(config.eos_token_id, vec![151_643, 151_645]);
        assert!(config.tie_word_embeddings);

        let audio = &config.audio_config;
        assert_eq!(audio.d_model, 896);
        assert_eq!(audio.encoder_layers, 18);
        assert_eq!(audio.encoder_attention_heads, 14);
        assert_eq!(audio.num_key_value_heads, 14);
        assert_eq!(audio.encoder_ffn_dim, 3584);
        assert_eq!(audio.output_dim, 1024);
        assert_eq!(audio.num_mel_bins, 128);
        assert_eq!(audio.downsample_hidden_size, 480);
        assert_eq!(audio.max_position_embeddings, 13);
        assert_eq!(audio.n_window, 50);
        assert_eq!(audio.n_window_infer, 800);
        assert_eq!(audio.conv_chunksize, 500);
        assert_eq!(audio.activation_function, Activation::Gelu);
        assert!(!audio.scale_embedding);

        let text = &config.text_config;
        assert_eq!(text.vocab_size, 151_936);
        assert_eq!(text.hidden_size, 1024);
        assert_eq!(text.intermediate_size, 3072);
        assert_eq!(text.num_hidden_layers, 28);
        assert_eq!(text.num_attention_heads, 16);
        assert_eq!(text.num_key_value_heads, 8);
        assert_eq!(text.head_dim, 128);
        assert_eq!(text.max_position_embeddings, 65536);
        assert!((text.rms_norm_eps - 1e-6).abs() < f64::EPSILON);
        assert!((text.rope_parameters.rope_theta - 1_000_000.0).abs() < f64::EPSILON);
        assert!(!text.attention_bias);
        assert!(text.tie_word_embeddings);
        assert_eq!(text.sliding_window, None);
        assert_eq!(text.max_window_layers, 28);
        assert!(!text.use_sliding_window);
    }

    #[test]
    fn deserialize_1_7b_config() {
        let config: Config = serde_json::from_str(CONFIG_1_7B).unwrap();

        let audio = &config.audio_config;
        assert_eq!(audio.d_model, 1024);
        assert_eq!(audio.encoder_layers, 24);
        assert_eq!(audio.encoder_attention_heads, 16);
        assert_eq!(audio.num_key_value_heads, 16);
        assert_eq!(audio.encoder_ffn_dim, 4096);
        assert_eq!(audio.output_dim, 2048);

        let text = &config.text_config;
        assert_eq!(text.hidden_size, 2048);
        assert_eq!(text.intermediate_size, 6144);
        assert_eq!(text.num_hidden_layers, 28);
        assert_eq!(text.num_attention_heads, 16);
        assert_eq!(text.num_key_value_heads, 8);

        // output_dim must match text hidden_size for the projector's output
        // to feed the decoder directly.
        assert_eq!(audio.output_dim, text.hidden_size);
    }

    #[test]
    fn text_config_to_qwen3_config() {
        let config: Config = serde_json::from_str(CONFIG_1_7B).unwrap();
        let qwen3_config = config.text_config.to_qwen3_config();

        assert_eq!(qwen3_config.vocab_size, 151_936);
        assert_eq!(qwen3_config.hidden_size, 2048);
        assert_eq!(qwen3_config.intermediate_size, 6144);
        assert_eq!(qwen3_config.num_hidden_layers, 28);
        assert_eq!(qwen3_config.num_attention_heads, 16);
        assert_eq!(qwen3_config.num_key_value_heads, 8);
        assert_eq!(qwen3_config.head_dim, Some(128));
        assert_eq!(qwen3_config.max_position_embeddings, 65536);
        assert!((qwen3_config.rope_theta - 1_000_000.0).abs() < f64::EPSILON);
        assert!(!qwen3_config.attention_bias);
        assert!(qwen3_config.use_qk_norm);
        assert!(qwen3_config.tie_word_embeddings);
        assert_eq!(qwen3_config.sliding_window, None);
        assert!(!qwen3_config.use_sliding_window);
        assert_eq!(qwen3_config.eos_token_id, None);
    }
}
