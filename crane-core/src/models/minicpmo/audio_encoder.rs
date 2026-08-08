//! MiniCPM-o-4.5 audio understanding tower: a standard Whisper-medium
//! encoder (`apm.*`) — 2×Conv1d frontend, learned/precomputed sinusoidal
//! position embeddings (loaded straight from the checkpoint, not
//! recomputed), 24 bidirectional pre-norm transformer layers, final
//! `LayerNorm`. Ported from `modeling_minicpmo.py`'s `MiniCPMWhisperEncoder`
//! (a thin `transformers.WhisperEncoder` subclass — the streaming-only
//! `use_extra_context` codepath isn't ported here, see the `minicpmo` module
//! doc for phase status).
//!
//! Mel feature extraction reuses `qwen3_asr::feature_extractor::WhisperFeatureExtractor`
//! unmodified (parametrized by `n_mels`, same `n_fft=400`/`hop_length=160`
//! pipeline `MiniCPMAAudioProcessor` inherits from HF's `WhisperFeatureExtractor`).

use std::io::{Read, Seek};

use candle_core::{Module, Result, Tensor};
use candle_nn::{conv1d, embedding, layer_norm, linear, linear_no_bias, Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, VarBuilder};

use super::config::AudioConfig;
use crate::models::hunyuan_dense::modeling::Gguf;

const LAYER_NORM_EPS: f64 = 1e-5;

struct WhisperAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl WhisperAttention {
    fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            q_proj: linear(dim, dim, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(dim, dim, vb.pp("k_proj"))?,
            v_proj: linear(dim, dim, vb.pp("v_proj"))?,
            out_proj: linear(dim, dim, vb.pp("out_proj"))?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// GGUF equivalent of [`Self::new`] — `prefix` is this attention
    /// block's tensor-name prefix (e.g. `encoder.blocks.0.attn`). Mirrors
    /// the real Whisper `MultiHeadAttention`'s own quirk of `key` having no
    /// bias while `query`/`value`/`out` do (already correctly ported in
    /// [`Self::new`]'s safetensors path too — this just matches it).
    fn from_gguf<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str, dim: usize, num_heads: usize) -> Result<Self> {
        let head_dim = dim / num_heads;
        let q_proj = Linear::new(gg.dequant_tensor(&format!("{prefix}.query.weight"))?, Some(gg.dequant_tensor(&format!("{prefix}.query.bias"))?.flatten_all()?));
        let k_proj = Linear::new(gg.dequant_tensor(&format!("{prefix}.key.weight"))?, None);
        let v_proj = Linear::new(gg.dequant_tensor(&format!("{prefix}.value.weight"))?, Some(gg.dequant_tensor(&format!("{prefix}.value.bias"))?.flatten_all()?));
        let out_proj = Linear::new(gg.dequant_tensor(&format!("{prefix}.out.weight"))?, Some(gg.dequant_tensor(&format!("{prefix}.out.bias"))?.flatten_all()?));
        Ok(Self { q_proj, k_proj, v_proj, out_proj, num_heads, head_dim, scale: 1.0 / (head_dim as f64).sqrt() })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq, _dim) = xs.dims3()?;
        let q = (self.q_proj.forward(xs)? * self.scale)?.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.k_proj.forward(xs)?.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.v_proj.forward(xs)?.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        let attn_weights = q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?; // [b, heads, seq, head_dim]

        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((b, seq, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&attn_output)
    }
}

struct EncoderLayer {
    self_attn_layer_norm: LayerNorm,
    self_attn: WhisperAttention,
    final_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl EncoderLayer {
    fn new(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn_layer_norm: layer_norm(cfg.d_model, LAYER_NORM_EPS, vb.pp("self_attn_layer_norm"))?,
            self_attn: WhisperAttention::new(cfg.d_model, cfg.encoder_attention_heads, vb.pp("self_attn"))?,
            final_layer_norm: layer_norm(cfg.d_model, LAYER_NORM_EPS, vb.pp("final_layer_norm"))?,
            fc1: linear(cfg.d_model, cfg.encoder_ffn_dim, vb.pp("fc1"))?,
            fc2: linear(cfg.encoder_ffn_dim, cfg.d_model, vb.pp("fc2"))?,
        })
    }

    /// GGUF equivalent of [`Self::new`] — `layer_idx`'s tensor-name prefix
    /// is `encoder.blocks.{layer_idx}` (real reference: the checkpoint's
    /// own GGUF export, `openbmb/MiniCPM-o-4_5-gguf`'s
    /// `MiniCPM-o-4_5-audio-F16.gguf` — near-PyTorch-original Whisper
    /// naming, not llama.cpp's usual compact `blk.N.*` scheme).
    fn from_gguf<R: Read + Seek>(gg: &mut Gguf<R>, cfg: &AudioConfig, layer_idx: usize) -> Result<Self> {
        let prefix = format!("encoder.blocks.{layer_idx}");
        let self_attn_layer_norm = LayerNorm::new(gg.dequant_tensor(&format!("{prefix}.attn_ln.weight"))?, gg.dequant_tensor(&format!("{prefix}.attn_ln.bias"))?, LAYER_NORM_EPS);
        let self_attn = WhisperAttention::from_gguf(gg, &format!("{prefix}.attn"), cfg.d_model, cfg.encoder_attention_heads)?;
        let final_layer_norm = LayerNorm::new(gg.dequant_tensor(&format!("{prefix}.mlp_ln.weight"))?, gg.dequant_tensor(&format!("{prefix}.mlp_ln.bias"))?, LAYER_NORM_EPS);
        let fc1 = Linear::new(gg.dequant_tensor(&format!("{prefix}.mlp.0.weight"))?, Some(gg.dequant_tensor(&format!("{prefix}.mlp.0.bias"))?.flatten_all()?));
        let fc2 = Linear::new(gg.dequant_tensor(&format!("{prefix}.mlp.2.weight"))?, Some(gg.dequant_tensor(&format!("{prefix}.mlp.2.bias"))?.flatten_all()?));
        Ok(Self { self_attn_layer_norm, self_attn, final_layer_norm, fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let hidden = self.self_attn_layer_norm.forward(xs)?;
        let hidden = self.self_attn.forward(&hidden)?;
        let hidden = (residual + hidden)?;

        let residual = &hidden;
        let mlp_in = self.final_layer_norm.forward(&hidden)?;
        let mlp = self.fc1.forward(&mlp_in)?;
        let mlp = candle_nn::Activation::Gelu.forward(&mlp)?;
        let mlp = self.fc2.forward(&mlp)?;
        residual + mlp
    }
}

pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    embed_positions: Embedding,
    layers: Vec<EncoderLayer>,
    layer_norm: LayerNorm,
}

impl AudioEncoder {
    pub fn new(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let conv1_cfg = Conv1dConfig { padding: 1, stride: 1, ..Default::default() };
        let conv1 = conv1d(cfg.num_mel_bins, cfg.d_model, 3, conv1_cfg, vb.pp("conv1"))?;
        let conv2_cfg = Conv1dConfig { padding: 1, stride: 2, ..Default::default() };
        let conv2 = conv1d(cfg.d_model, cfg.d_model, 3, conv2_cfg, vb.pp("conv2"))?;

        let embed_positions = embedding(cfg.max_source_positions, cfg.d_model, vb.pp("embed_positions"))?;

        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            layers.push(EncoderLayer::new(cfg, vb_layers.pp(i))?);
        }

        let layer_norm = layer_norm(cfg.d_model, LAYER_NORM_EPS, vb.pp("layer_norm"))?;

        Ok(Self { conv1, conv2, embed_positions, layers, layer_norm })
    }

    /// GGUF equivalent of [`Self::new`] — loads from a standalone
    /// `MiniCPM-o-4_5-audio-F16.gguf`-style file (`general.architecture =
    /// whisper`) instead of the checkpoint's own safetensors weights.
    /// `cfg` still comes from the checkpoint's `config.json` as usual (see
    /// `MiniCpmOLlm::from_gguf`'s doc for why config/tokenizer stay
    /// safetensors-checkpoint-sourced even when weights are GGUF-sourced).
    /// Note this GGUF file's tensors are F16/F32 only — no actual
    /// quantization — so this delivers no VRAM savings over the
    /// safetensors path; it exists for pure-GGUF deployment convenience.
    ///
    /// # Errors
    ///
    /// Returns an error if a required tensor is missing or has an
    /// unexpected shape.
    pub fn from_gguf<R: Read + Seek>(gg: &mut Gguf<R>, cfg: &AudioConfig) -> Result<Self> {
        let conv1_cfg = Conv1dConfig { padding: 1, stride: 1, ..Default::default() };
        let conv1 = Conv1d::new(gg.dequant_tensor("encoder.conv1.weight")?, Some(gg.dequant_tensor("encoder.conv1.bias")?.flatten_all()?), conv1_cfg);

        let conv2_cfg = Conv1dConfig { padding: 1, stride: 2, ..Default::default() };
        let conv2 = Conv1d::new(gg.dequant_tensor("encoder.conv2.weight")?, Some(gg.dequant_tensor("encoder.conv2.bias")?.flatten_all()?), conv2_cfg);

        let embed_positions = Embedding::new(gg.dequant_tensor("encoder.positional_embedding")?, cfg.d_model);

        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            layers.push(EncoderLayer::from_gguf(gg, cfg, i)?);
        }

        let layer_norm = LayerNorm::new(gg.dequant_tensor("encoder.ln_post.weight")?, gg.dequant_tensor("encoder.ln_post.bias")?, LAYER_NORM_EPS);

        Ok(Self { conv1, conv2, embed_positions, layers, layer_norm })
    }

    /// `input_features`: log-mel spectrogram `[B, num_mel_bins, mel_frames]`
    /// (from `WhisperFeatureExtractor::extract`). No attention masking —
    /// single, unpadded clip per batch item (B=1 is the common case; see the
    /// module doc for what padded/masked multi-clip batching would need).
    ///
    /// Returns `[B, encoder_frames, d_model]`.
    pub fn forward(&self, input_features: &Tensor) -> Result<Tensor> {
        let xs = self.conv1.forward(input_features)?;
        let xs = candle_nn::Activation::Gelu.forward(&xs)?;
        let xs = self.conv2.forward(&xs)?;
        let xs = candle_nn::Activation::Gelu.forward(&xs)?; // [B, d_model, encoder_frames]

        let xs = xs.transpose(1, 2)?.contiguous()?; // [B, encoder_frames, d_model]
        let (_b, seq_len, _d) = xs.dims3()?;

        let pos_ids = Tensor::arange(0u32, seq_len as u32, xs.device())?;
        let embed_pos = self.embed_positions.forward(&pos_ids)?.to_dtype(xs.dtype())?; // [seq_len, d_model]

        let mut hidden = xs.broadcast_add(&embed_pos)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        self.layer_norm.forward(&hidden)
    }
}

/// Output length of one stride-2, kernel-3, padding-1 `Conv1d` pass.
fn conv_output_len(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    (len - 1) / 2 + 1
}

/// Encoder output frame count for `mel_frames` raw mel frames — direct port
/// of `_get_feat_extract_output_lengths`'s `input_lengths_after_cnn`
/// (`conv1` is stride 1, so only `conv2`'s stride-2 halving applies).
#[must_use]
pub fn encoder_output_len(mel_frames: usize) -> usize {
    conv_output_len(mel_frames)
}

/// Post-`audio_avg_pooler` token count — direct port of
/// `_get_feat_extract_output_lengths`'s `input_lengths_after_pooling`.
#[must_use]
pub fn pooled_output_len(mel_frames: usize, audio_pool_step: usize) -> usize {
    let after_cnn = encoder_output_len(mel_frames);
    (after_cnn.saturating_sub(audio_pool_step)) / audio_pool_step + 1
}
