//! HF-compatible config types for KugelAudio (`kugelaudio/kugelaudio-0-open`).
//!
//! KugelAudio is a post-trained fine-tune of Microsoft's VibeVoice
//! architecture (`model_type: "kugelaudio"`, `KugelAudioForConditionalGeneration`
//! in the checkpoint's `config.json`) — a dense Qwen2 decoder backbone plus two
//! causal-conv VAE "tokenizers" (acoustic + semantic) and a small adaLN
//! diffusion head that predicts continuous speech latents autoregressively.
//! See `kugelaudio/mod.rs` for the module map.
//!
//! Ported against `kugelaudio-0-open`'s own bundled source (structurally a
//! renamed copy of `microsoft/VibeVoice`'s `modular_vibevoice_{tokenizer,
//! diffusion_head}.py` and a vendored copy of `diffusers`'
//! `DPMSolverMultistepScheduler`) rather than the base VibeVoice repo, since
//! this checkpoint's `config.json` is what actually ships.

use serde::Deserialize;

/// `decoder_config` — a stock Qwen2 decoder (28 layers / 3584 hidden /
/// GQA 28:4 heads for the reference 7B checkpoint). Field set matches
/// `qwen25::qwen2::Config` but is kept separate: this checkpoint's JSON
/// additionally carries `use_mrope`/`use_sliding_window`/`max_window_layers`
/// keys that Qwen2.5's config doesn't, and `sliding_window`/`rope_scaling`
/// are `null` here (Qwen2, not Qwen2.5), which the shared struct's
/// non-`Option` `sliding_window: usize` field can't represent.
#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub hidden_act: candle_nn::Activation,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl DecoderConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Shared shape for `acoustic_tokenizer_config`/`semantic_tokenizer_config`.
/// Both are causal-conv VAE encoder/decoder stacks (ConvNeXt-style residual
/// blocks over 1D convs) — see `conv_layers.rs`. The semantic tokenizer only
/// ever uses the encoder half (no `decoder_depths`, `fix_std`, or
/// `std_dist_type` reconstruction is meaningful for it), but both configs
/// share this struct since the checkpoint's JSON gives them the same shape
/// with a few fields the semantic side leaves at their inert defaults
/// (`fix_std: 0`, `std_dist_type: "none"`).
#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerConfig {
    pub channels: usize,
    /// `vae_dim` — the tokenizer's latent dimension (64 for acoustic, 128 for
    /// semantic in the reference checkpoint). Called `dimension` in the
    /// Python `TokenizerEncoder`/`TokenizerDecoder` (derived from this field).
    pub vae_dim: usize,
    pub encoder_n_filters: usize,
    pub encoder_ratios: Vec<usize>,
    /// `"3-3-3-3-3-3-8"` — dash-separated stage depths, parsed by
    /// [`Self::encoder_depths_vec`].
    pub encoder_depths: String,
    #[serde(default)]
    pub decoder_n_filters: Option<usize>,
    #[serde(default)]
    pub decoder_ratios: Option<Vec<usize>>,
    /// `null` in the checkpoint → decoder depths default to
    /// `reversed(encoder_depths)` (mirrors `VibeVoiceAcousticTokenizerModel.__init__`).
    #[serde(default)]
    pub decoder_depths: Option<String>,
    pub causal: bool,
    pub conv_bias: bool,
    /// `"none" | "weight_norm" | "spectral_norm" | "layer_norm" | "time_group_norm"`.
    /// The reference checkpoint uses `"none"` for both tokenizers, so
    /// `conv_layers.rs` only implements the no-reparametrization path and
    /// bails on anything else rather than porting `weight_norm`/
    /// `spectral_norm` for a path that's never hit.
    pub conv_norm: String,
    /// `"constant" | "reflect"`. Reference checkpoint: `"constant"` (zero-pad).
    pub pad_mode: String,
    /// `"LN" | "RMSNorm"`. Reference checkpoint: `"RMSNorm"`.
    pub layernorm: String,
    pub layernorm_eps: f64,
    pub layernorm_elementwise_affine: bool,
    /// `"conv" | "depthwise_conv"`. Reference checkpoint: `"depthwise_conv"`
    /// (groups = channel count).
    pub mixer_layer: String,
    /// Nonzero enables the per-block learnable `LayerScale` (`gamma`/`ffn_gamma`).
    pub layer_scale_init_value: f64,
    pub disable_last_norm: bool,
    /// Fixed reconstruction std for the acoustic VAE's Gaussian sampling
    /// (`std_dist_type == "fix"` path). Unused by the semantic tokenizer.
    #[serde(default)]
    pub fix_std: f64,
    /// `"fix" | "gaussian" | "none"`. Unused at inference here — sampling
    /// (`VibeVoiceTokenizerEncoderOutput::sample`) is a training-time detail;
    /// this port's `encode()` returns the distribution mean directly (`mode()`
    /// in the Python), matching how voice-prompt conditioning is actually
    /// consumed in `kugelaudio_inference.py`.
    #[serde(default)]
    pub std_dist_type: String,
}

impl TokenizerConfig {
    pub fn encoder_depths_vec(&self) -> Vec<usize> {
        parse_dash_depths(&self.encoder_depths)
    }

    /// Decoder depths: `decoder_depths` if present, else
    /// `reversed(encoder_depths)` (`VibeVoiceAcousticTokenizerModel.__init__`'s
    /// fallback).
    pub fn decoder_depths_vec(&self) -> Vec<usize> {
        match &self.decoder_depths {
            Some(s) => parse_dash_depths(s),
            None => {
                let mut d = self.encoder_depths_vec();
                d.reverse();
                d
            },
        }
    }
}

fn parse_dash_depths(s: &str) -> Vec<usize> {
    s.split('-').filter_map(|d| d.parse().ok()).collect()
}

/// `diffusion_head_config` — the small adaLN-modulated FFN stack that
/// predicts the next acoustic latent's noise/velocity conditioned on the
/// decoder's hidden state. See `diffusion_head.rs`.
#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionHeadConfig {
    pub hidden_size: usize,
    pub latent_size: usize,
    pub head_layers: usize,
    pub head_ffn_ratio: f64,
    pub rms_norm_eps: f64,
    /// `"epsilon" | "v_prediction"`. Reference checkpoint: `"v_prediction"`.
    pub prediction_type: String,
    /// `"cosine"` is the only schedule `dpm_solver.rs` implements (matches
    /// the reference checkpoint — see that module's doc comment).
    pub ddpm_beta_schedule: String,
    pub ddpm_num_steps: usize,
    pub ddpm_num_inference_steps: usize,
    /// `"sde-dpmsolver++"` is the only algorithm `dpm_solver.rs` implements.
    pub ddpm_algorithm_type: String,
}

/// Top-level KugelAudio `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct KugelAudioConfig {
    pub decoder_config: DecoderConfig,
    pub acoustic_tokenizer_config: TokenizerConfig,
    pub semantic_tokenizer_config: TokenizerConfig,
    pub diffusion_head_config: DiffusionHeadConfig,
    pub acoustic_vae_dim: usize,
    pub semantic_vae_dim: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

/// Load `config.json` for a KugelAudio checkpoint.
pub fn load_config(path: &str) -> candle_core::Result<KugelAudioConfig> {
    let data = std::fs::read(path)
        .map_err(|e| candle_core::Error::Msg(format!("read config {path}: {e}")))?;
    let cfg: KugelAudioConfig = serde_json::from_slice(&data)
        .map_err(|e| candle_core::Error::Msg(format!("parse config {path}: {e}")))?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact `config.json` published for `kugelaudio/kugelaudio-0-open`
    /// (fetched from the Hub during scoping this port) — a regression test
    /// against silent schema drift, and documentation of the real shape.
    const REAL_CONFIG_JSON: &str = r#"{
      "acostic_vae_dim": 64,
      "acoustic_tokenizer_config": {
        "causal": true, "channels": 1, "conv_bias": true, "conv_norm": "none",
        "corpus_normalize": 0.0, "decoder_depths": null, "decoder_n_filters": 32,
        "decoder_ratios": [8, 5, 5, 4, 2, 2], "disable_last_norm": true,
        "encoder_depths": "3-3-3-3-3-3-8", "encoder_n_filters": 32,
        "encoder_ratios": [8, 5, 5, 4, 2, 2], "fix_std": 0.5,
        "layer_scale_init_value": 1e-06, "layernorm": "RMSNorm",
        "layernorm_elementwise_affine": true, "layernorm_eps": 1e-05,
        "mixer_layer": "depthwise_conv", "model_type": "kugelaudio_acoustic_tokenizer",
        "pad_mode": "constant", "std_dist_type": "gaussian", "torch_dtype": "bfloat16",
        "vae_dim": 64, "weight_init_value": 0.01
      },
      "acoustic_vae_dim": 64,
      "architectures": ["KugelAudioForConditionalGeneration"],
      "decoder_config": {
        "attention_dropout": 0.0, "hidden_act": "silu", "hidden_size": 3584,
        "initializer_range": 0.02, "intermediate_size": 18944,
        "max_position_embeddings": 32768, "max_window_layers": 28,
        "model_type": "qwen2", "num_attention_heads": 28, "num_hidden_layers": 28,
        "num_key_value_heads": 4, "rms_norm_eps": 1e-06, "rope_scaling": null,
        "rope_theta": 1000000.0, "sliding_window": null, "torch_dtype": "bfloat16",
        "use_cache": true, "use_mrope": false, "use_sliding_window": false,
        "vocab_size": 152064
      },
      "diffusion_head_config": {
        "ddpm_algorithm_type": "sde-dpmsolver++", "ddpm_batch_mul": 4,
        "ddpm_beta_schedule": "cosine", "ddpm_num_inference_steps": 20,
        "ddpm_num_steps": 1000, "diffusion_type": "ddpm", "head_ffn_ratio": 3.0,
        "head_layers": 4, "hidden_size": 3584, "latent_size": 64,
        "model_type": "kugelaudio_diffusion_head", "prediction_type": "v_prediction",
        "rms_norm_eps": 1e-05, "speech_vae_dim": 64, "torch_dtype": "bfloat16"
      },
      "model_type": "kugelaudio",
      "semantic_tokenizer_config": {
        "causal": true, "channels": 1, "conv_bias": true, "conv_norm": "none",
        "corpus_normalize": 0.0, "disable_last_norm": true,
        "encoder_depths": "3-3-3-3-3-3-8", "encoder_n_filters": 32,
        "encoder_ratios": [8, 5, 5, 4, 2, 2], "fix_std": 0,
        "layer_scale_init_value": 1e-06, "layernorm": "RMSNorm",
        "layernorm_elementwise_affine": true, "layernorm_eps": 1e-05,
        "mixer_layer": "depthwise_conv", "model_type": "kugelaudio_semantic_tokenizer",
        "pad_mode": "constant", "std_dist_type": "none", "torch_dtype": "bfloat16",
        "vae_dim": 128, "weight_init_value": 0.01
      },
      "semantic_vae_dim": 128,
      "tie_word_embeddings": false,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.52.0.dev0",
      "ddpm_inference_steps": 20
    }"#;

    #[test]
    fn parses_real_checkpoint_config() {
        let cfg: KugelAudioConfig = serde_json::from_str(REAL_CONFIG_JSON).expect("parse");
        assert_eq!(cfg.decoder_config.hidden_size, 3584);
        assert_eq!(cfg.decoder_config.num_hidden_layers, 28);
        assert_eq!(cfg.decoder_config.num_attention_heads, 28);
        assert_eq!(cfg.decoder_config.num_key_value_heads, 4);
        assert_eq!(cfg.decoder_config.head_dim(), 128);
        assert_eq!(cfg.acoustic_vae_dim, 64);
        assert_eq!(cfg.semantic_vae_dim, 128);
        assert_eq!(
            cfg.acoustic_tokenizer_config.encoder_depths_vec(),
            vec![3, 3, 3, 3, 3, 3, 8]
        );
        assert_eq!(
            cfg.acoustic_tokenizer_config.decoder_depths_vec(),
            vec![8, 3, 3, 3, 3, 3, 3]
        );
        assert_eq!(cfg.acoustic_tokenizer_config.vae_dim, 64);
        assert_eq!(cfg.semantic_tokenizer_config.vae_dim, 128);
        assert_eq!(cfg.diffusion_head_config.latent_size, 64);
        assert_eq!(cfg.diffusion_head_config.head_layers, 4);
        assert!((cfg.diffusion_head_config.head_ffn_ratio - 3.0).abs() < 1e-9);
        assert_eq!(cfg.diffusion_head_config.prediction_type, "v_prediction");
        assert_eq!(cfg.diffusion_head_config.ddpm_algorithm_type, "sde-dpmsolver++");
    }
}
