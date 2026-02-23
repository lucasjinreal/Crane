//! Qwen3-TTS transformer modeling.
//!
//! Implements the two-level architecture:
//! 1. **Talker** — main transformer with MRoPE (3D position encoding) that
//!    generates the first-codebook token at each step.
//! 2. **CodePredictor** — small transformer that autoregressively predicts
//!    the remaining `num_code_groups - 1` codebook tokens given the talker
//!    hidden state.
//!
//! Weight layout follows the HuggingFace `Qwen3TTSForConditionalGeneration`
//! checkpoint:
//!   - `talker.model.*`       — talker backbone
//!   - `talker.codec_head.*`  — linear head for first-codebook logits
//!   - `talker.text_projection.*` — MLP that projects text embeddings into talker dim
//!   - `talker.code_predictor.*`  — code predictor sub-model

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Embedding, Linear, RmsNorm, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;

// ── Config ──────────────────────────────────────────────────────────────

fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_num_code_groups() -> usize {
    16
}
fn default_head_dim() -> usize {
    128
}
fn default_max_position_embeddings() -> usize {
    32768
}

/// M-RoPE section configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub interleaved: bool,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default)]
    pub rope_type: Option<String>,
}

/// Code predictor config (sub-talker).
#[derive(Debug, Clone, Deserialize)]
pub struct CodePredictorConfig {
    #[serde(default = "default_vocab_size_2048")]
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_num_code_groups")]
    pub num_code_groups: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
}
fn default_vocab_size_2048() -> usize {
    2048
}

/// Talker config.
#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_num_code_groups")]
    pub num_code_groups: usize,
    #[serde(default = "default_text_hidden_size")]
    pub text_hidden_size: usize,
    #[serde(default = "default_text_vocab_size")]
    pub text_vocab_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    pub rope_scaling: Option<RopeScaling>,
    pub code_predictor_config: CodePredictorConfig,

    // Special token IDs
    #[serde(default)]
    pub codec_eos_token_id: usize,
    #[serde(default)]
    pub codec_think_id: usize,
    #[serde(default)]
    pub codec_nothink_id: usize,
    #[serde(default)]
    pub codec_think_bos_id: usize,
    #[serde(default)]
    pub codec_think_eos_id: usize,
    #[serde(default)]
    pub codec_pad_id: usize,
    #[serde(default)]
    pub codec_bos_id: usize,

    /// Language name → codec token id
    #[serde(default)]
    pub codec_language_id: HashMap<String, usize>,

    /// Speaker name → codec token id
    #[serde(default)]
    pub spk_id: HashMap<String, usize>,

    /// Speaker name → dialect string or false
    #[serde(default)]
    pub spk_is_dialect: HashMap<String, serde_json::Value>,
}

fn default_text_hidden_size() -> usize {
    2048
}
fn default_text_vocab_size() -> usize {
    151936
}

/// Speaker encoder config (ECAPA-TDNN).
#[derive(Debug, Clone, Deserialize)]
pub struct SpeakerEncoderConfig {
    #[serde(default = "default_mel_dim")]
    pub mel_dim: usize,
    #[serde(default = "default_enc_dim")]
    pub enc_dim: usize,
    #[serde(default = "default_enc_channels")]
    pub enc_channels: Vec<usize>,
    #[serde(default = "default_enc_kernel_sizes")]
    pub enc_kernel_sizes: Vec<usize>,
    #[serde(default = "default_enc_dilations")]
    pub enc_dilations: Vec<usize>,
    #[serde(default = "default_enc_attention_channels")]
    pub enc_attention_channels: usize,
    #[serde(default = "default_enc_res2net_scale")]
    pub enc_res2net_scale: usize,
    #[serde(default = "default_enc_se_channels")]
    pub enc_se_channels: usize,
    #[serde(default = "default_speaker_sample_rate")]
    pub sample_rate: u32,
}
fn default_mel_dim() -> usize { 128 }
fn default_enc_dim() -> usize { 1024 }
fn default_enc_channels() -> Vec<usize> { vec![512, 512, 512, 512, 1536] }
fn default_enc_kernel_sizes() -> Vec<usize> { vec![5, 3, 3, 3, 1] }
fn default_enc_dilations() -> Vec<usize> { vec![1, 2, 3, 4, 1] }
fn default_enc_attention_channels() -> usize { 128 }
fn default_enc_res2net_scale() -> usize { 8 }
fn default_enc_se_channels() -> usize { 128 }
fn default_speaker_sample_rate() -> u32 { 24000 }

impl Default for SpeakerEncoderConfig {
    fn default() -> Self {
        Self {
            mel_dim: default_mel_dim(),
            enc_dim: default_enc_dim(),
            enc_channels: default_enc_channels(),
            enc_kernel_sizes: default_enc_kernel_sizes(),
            enc_dilations: default_enc_dilations(),
            enc_attention_channels: default_enc_attention_channels(),
            enc_res2net_scale: default_enc_res2net_scale(),
            enc_se_channels: default_enc_se_channels(),
            sample_rate: default_speaker_sample_rate(),
        }
    }
}

/// Top-level Qwen3-TTS config (loaded from `config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TTSConfig {
    pub talker_config: TalkerConfig,
    #[serde(default)]
    pub speaker_encoder_config: SpeakerEncoderConfig,
    #[serde(default)]
    pub tts_model_type: Option<String>,
    #[serde(default)]
    pub tts_model_size: Option<String>,
    #[serde(default)]
    pub tokenizer_type: Option<String>,
    #[serde(default = "default_tts_bos")]
    pub tts_bos_token_id: u32,
    #[serde(default = "default_tts_eos")]
    pub tts_eos_token_id: u32,
    #[serde(default = "default_tts_pad")]
    pub tts_pad_token_id: u32,
}

fn default_tts_bos() -> u32 {
    151672
}
fn default_tts_eos() -> u32 {
    151673
}
fn default_tts_pad() -> u32 {
    151671
}

// ── Rotary Embedding (standard, 1D) ────────────────────────────────────

struct RotaryEmbedding {
    cos_table: Tensor,
    sin_table: Tensor,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_pos: usize, theta: f64, device: &Device) -> Result<Self> {
        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv.as_slice(), device)?;
        let positions: Vec<f32> = (0..max_pos).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let cos_table = freqs.cos()?.contiguous()?;
        let sin_table = freqs.sin()?.contiguous()?;
        Ok(Self {
            cos_table,
            sin_table,
        })
    }

    fn forward(&self, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos_table.narrow(0, 0, seq_len)?;
        let sin = self.sin_table.narrow(0, 0, seq_len)?;
        Ok((cos, sin))
    }
}

// ── RoPE application helpers ────────────────────────────────────────────

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half_dim = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // cos, sin: [seq_len, head_dim/2] → broadcast to [1, 1, seq_len, head_dim]
    let target_dtype = q.dtype();
    let cos_full = Tensor::cat(&[cos, cos], D::Minus1)?
        .to_dtype(target_dtype)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    let sin_full = Tensor::cat(&[sin, sin], D::Minus1)?
        .to_dtype(target_dtype)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    let q_embed = (q.broadcast_mul(&cos_full)? + rotate_half(q)?.broadcast_mul(&sin_full)?)?;
    let k_embed = (k.broadcast_mul(&cos_full)? + rotate_half(k)?.broadcast_mul(&sin_full)?)?;
    Ok((q_embed, k_embed))
}

/// Apply M-RoPE (multi-modal rotary position embedding) as used by the Talker.
/// Position IDs is [3, batch, seq_len], cos/sin from 3D rotary embedding.
fn apply_mrope(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
    _interleaved: bool,
) -> Result<(Tensor, Tensor)> {
    // For simplicity, use the non-interleaved path (concatenated sections):
    // Split head_dim into 3 sections by mrope_section, apply different position
    // embeddings from the 3 modality dims, then concatenate back.
    // cos/sin shape: [3, 1, seq_len, head_dim/2]
    // We duplicate sections * 2 and split along last dim.
    let sections: Vec<usize> = mrope_section.iter().map(|&s| s * 2).collect();

    let cos_parts: Vec<Tensor> = cos.chunk(3, 0)?;
    let sin_parts: Vec<Tensor> = sin.chunk(3, 0)?;

    // Build per-head-dim cos/sin by interleaving sections from each modality
    let mut cos_chunks = Vec::new();
    let mut sin_chunks = Vec::new();
    let mut offset = 0;
    for (i, &sec) in sections.iter().enumerate() {
        let mod_idx = i % 3;
        let c = cos_parts[mod_idx].squeeze(0)?.narrow(D::Minus1, offset, sec)?;
        let s = sin_parts[mod_idx].squeeze(0)?.narrow(D::Minus1, offset, sec)?;
        cos_chunks.push(c);
        sin_chunks.push(s);
        offset += sec;
    }
    let target_dtype = q.dtype();
    let cos_cat = Tensor::cat(&cos_chunks, D::Minus1)?.to_dtype(target_dtype)?;
    let sin_cat = Tensor::cat(&sin_chunks, D::Minus1)?.to_dtype(target_dtype)?;

    let q_embed = (q.broadcast_mul(&cos_cat)? + rotate_half(q)?.broadcast_mul(&sin_cat)?)?;
    let k_embed = (k.broadcast_mul(&cos_cat)? + rotate_half(k)?.broadcast_mul(&sin_cat)?)?;
    Ok((q_embed, k_embed))
}

// ── Attention ───────────────────────────────────────────────────────────

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let make = |in_d, out_d, name: &str| -> Result<Linear> {
            if bias {
                linear(in_d, out_d, vb.pp(name))
            } else {
                linear_no_bias(in_d, out_d, vb.pp(name))
            }
        };
        Ok(Self {
            q_proj: make(hidden_size, num_heads * head_dim, "q_proj")?,
            k_proj: make(hidden_size, num_kv_heads * head_dim, "k_proj")?,
            v_proj: make(hidden_size, num_kv_heads * head_dim, "v_proj")?,
            o_proj: make(num_heads * head_dim, hidden_size, "o_proj")?,
            q_norm: candle_nn::rms_norm(head_dim, rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: candle_nn::rms_norm(head_dim, rms_norm_eps, vb.pp("k_norm"))?,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        use_mrope: bool,
        mrope_section: &[usize],
        mrope_interleaved: bool,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden_states.dims3()?;

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // QK norm
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // RoPE
        let (q, k) = if use_mrope && !mrope_section.is_empty() {
            apply_mrope(&q, &k, cos, sin, mrope_section, mrope_interleaved)?
        } else {
            apply_rotary_pos_emb(&q, &k, cos, sin)?
        };

        // KV cache
        let (k, v) = match self.kv_cache.take() {
            Some((ck, cv)) => {
                let k = Tensor::cat(&[&ck, &k], 2)?;
                let v = Tensor::cat(&[&cv, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // GQA expand
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = if n_rep > 1 {
            let (b, kv_h, s, d) = k.dims4()?;
            k.unsqueeze(2)?
                .expand((b, kv_h, n_rep, s, d))?
                .reshape((b, kv_h * n_rep, s, d))?
        } else {
            k
        };
        let v = if n_rep > 1 {
            let (b, kv_h, s, d) = v.dims4()?;
            v.unsqueeze(2)?
                .expand((b, kv_h, n_rep, s, d))?
                .reshape((b, kv_h * n_rep, s, d))?
        } else {
            v
        };

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };
        // Compute softmax in F32 for numeric stability, matching PyTorch's
        // `F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)`
        let input_dtype = attn_weights.dtype();
        let attn_weights = candle_nn::ops::softmax_last_dim(
            &attn_weights.to_dtype(DType::F32)?,
        )?
        .to_dtype(input_dtype)?;
        let attn_output = attn_weights.matmul(&v)?;

        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_sz, seq_len, ()))?;
        self.o_proj.forward(&attn_output)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

// ── MLP ─────────────────────────────────────────────────────────────────

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ── Resize MLP (text_projection) ────────────────────────────────────────

struct ResizeMlp {
    fc1: Linear,
    fc2: Linear,
}

impl ResizeMlp {
    fn new(
        input_size: usize,
        intermediate_size: usize,
        output_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            fc1: linear(input_size, intermediate_size, vb.pp("linear_fc1"))?,
            fc2: linear(intermediate_size, output_size, vb.pp("linear_fc2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = candle_nn::Activation::Silu.forward(&self.fc1.forward(x)?)?;
        self.fc2.forward(&x)
    }
}

// ── Decoder Layer ───────────────────────────────────────────────────────

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                bias,
                vb.pp("self_attn"),
            )?,
            mlp: Mlp::new(hidden_size, intermediate_size, vb.pp("mlp"))?,
            input_layernorm: candle_nn::rms_norm(hidden_size, rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: candle_nn::rms_norm(
                hidden_size,
                rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        use_mrope: bool,
        mrope_section: &[usize],
        mrope_interleaved: bool,
    ) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(
            &hidden_states,
            cos,
            sin,
            attention_mask,
            use_mrope,
            mrope_section,
            mrope_interleaved,
        )?;
        let hidden_states = (residual + hidden_states)?;

        let residual = &hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        residual + hidden_states
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Code Predictor (sub-talker)
// ═══════════════════════════════════════════════════════════════════════

/// The code predictor predicts codebook groups 1..N-1 given the talker hidden
/// state and previously predicted codes.
pub struct CodePredictor {
    /// Embeddings per code group: codec_embedding[i] for group i+1
    codec_embeddings: Vec<Embedding>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    /// Linear heads per code group: lm_head[i] for group i+1
    lm_heads: Vec<Linear>,
    /// Projection from talker hidden → code predictor hidden (if sizes differ)
    small_to_mtp_projection: Option<Linear>,
    rotary_emb: RotaryEmbedding,
    config: CodePredictorConfig,
    num_code_groups: usize,
}

impl CodePredictor {
    pub fn new(
        config: &CodePredictorConfig,
        talker_hidden_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let n = config.num_code_groups - 1;

        let mut codec_embeddings = Vec::with_capacity(n);
        let model_vb = vb.pp("model");
        for i in 0..n {
            codec_embeddings.push(candle_nn::embedding(
                config.vocab_size,
                talker_hidden_size,
                model_vb.pp("codec_embedding").pp(i),
            )?);
        }

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps,
                config.attention_bias,
                model_vb.pp("layers").pp(i),
            )?);
        }
        let norm = candle_nn::rms_norm(config.hidden_size, config.rms_norm_eps, model_vb.pp("norm"))?;

        let mut lm_heads = Vec::with_capacity(n);
        for i in 0..n {
            lm_heads.push(linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                vb.pp("lm_head").pp(i),
            )?);
        }

        let small_to_mtp_projection = if talker_hidden_size != config.hidden_size {
            Some(linear(
                talker_hidden_size,
                config.hidden_size,
                vb.pp("small_to_mtp_projection"),
            )?)
        } else {
            None
        };

        let rotary_emb = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            model_vb.device(),
        )?;

        Ok(Self {
            codec_embeddings,
            layers,
            norm,
            lm_heads,
            small_to_mtp_projection,
            rotary_emb,
            config: config.clone(),
            num_code_groups: config.num_code_groups,
        })
    }

    /// Given the talker hidden state and the first codebook token, predict the
    /// remaining `num_code_groups - 1` codebook tokens using argmax (greedy).
    pub fn predict(
        &mut self,
        talker_hidden: &Tensor,
        first_code: u32,
        codec_embedding: &Embedding,
        device: &Device,
        temperature: f64,
        top_p: Option<f64>,
    ) -> Result<Vec<u32>> {
        fn as_btd(x: &Tensor) -> Result<Tensor> {
            match x.dims().len() {
                1 => x.unsqueeze(0)?.unsqueeze(0),
                2 => x.unsqueeze(1),
                3 => Ok(x.clone()),
                n => candle_core::bail!("Unexpected hidden rank in code predictor: {n} ({:?})", x.dims()),
            }
        }

        self.clear_kv_cache();
        let n_groups = self.num_code_groups - 1;
        let mut codes = Vec::with_capacity(n_groups);

        // Sampling for code predictor (subtalker) — matches Python defaults:
        //   subtalker_dosample=True, subtalker_temperature=0.9,
        //   subtalker_top_k=50, subtalker_top_p=1.0
        let mut logits_processor =
            candle_transformers::generation::LogitsProcessor::from_sampling(
                42,
                candle_transformers::generation::Sampling::TopKThenTopP {
                    k: 50,
                    p: top_p.unwrap_or(1.0),
                    temperature,
                },
            );

        // Build initial input: [talker_hidden, embed(first_code)]
        let first_embed = as_btd(&codec_embedding.forward(&Tensor::new(&[first_code], device)?)?)?;
        let hidden = as_btd(talker_hidden)?;
        let inputs_embeds = Tensor::cat(&[&hidden, &first_embed], 1)?; // [1, 2, D]

        // Project if needed
        let inputs_embeds = match &self.small_to_mtp_projection {
            Some(proj) => proj.forward(&inputs_embeds)?,
            None => inputs_embeds,
        };

        let seq_len = inputs_embeds.dim(1)?;
        let (cos, sin) = self.rotary_emb.forward(seq_len)?;

        // Build causal mask for the 2-token prefill
        let causal_mask = if seq_len > 1 {
            let mut mask_data = vec![0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
            let mask = Tensor::new(mask_data.as_slice(), device)?
                .reshape((1, 1, seq_len, seq_len))?
                .to_dtype(inputs_embeds.dtype())?;
            Some(mask)
        } else {
            None
        };

        // Forward through layers
        let mut hidden_states = inputs_embeds;
        for layer in &mut self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                &cos,
                &sin,
                causal_mask.as_ref(),
                false,
                &[],
                false,
            )?;
        }
        hidden_states = self.norm.forward(&hidden_states)?;

        // First head prediction (sampling)
        let logits = self.lm_heads[0].forward(
            &hidden_states.narrow(1, seq_len - 1, 1)?,
        )?;
        let logits_f32 = logits.flatten_all()?.to_dtype(candle_core::DType::F32)?;
        let next_token = logits_processor.sample(&logits_f32)?;
        codes.push(next_token);

        // Remaining groups (sampling)
        for g in 1..n_groups {
            let embed = as_btd(&self.codec_embeddings[g - 1].forward(
                &Tensor::new(&[codes[g - 1]], device)?,
            )?)?;
            let embed = match &self.small_to_mtp_projection {
                Some(proj) => proj.forward(&embed)?,
                None => embed,
            };
            let total = seq_len + g;
            let (cos, sin) = self.rotary_emb.forward(total)?;
            let cos = cos.narrow(0, total - 1, 1)?;
            let sin = sin.narrow(0, total - 1, 1)?;

            let mut hs = embed;
            for layer in &mut self.layers {
                hs = layer.forward(&hs, &cos, &sin, None, false, &[], false)?;
            }
            hs = self.norm.forward(&hs)?;
            let logits = self.lm_heads[g].forward(&hs)?;
            let logits_f32 = logits.flatten_all()?.to_dtype(candle_core::DType::F32)?;
            let next_token = logits_processor.sample(&logits_f32)?;
            codes.push(next_token);
        }

        self.clear_kv_cache();
        Ok(codes)
    }

    fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Talker Model
// ═══════════════════════════════════════════════════════════════════════

/// The main talker transformer that generates speech codec tokens.
pub struct TalkerModel {
    /// Codec token embedding (all codebook groups share this for first-token input)
    codec_embedding: Embedding,
    /// Text token embedding (shared Qwen text vocab)
    text_embedding: Embedding,
    /// Text → talker hidden projection
    text_projection: ResizeMlp,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    /// Head that predicts first-codebook tokens
    codec_head: Linear,
    /// Sub-talker for remaining codebook groups
    pub code_predictor: CodePredictor,
    /// M-RoPE rotary embedding (3D)
    rotary_emb: RotaryEmbedding,
    config: TalkerConfig,
    /// M-RoPE section sizes
    mrope_section: Vec<usize>,
    mrope_interleaved: bool,
}

impl TalkerModel {
    pub fn new(config: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let model_vb = vb.pp("model");

        let codec_embedding = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            model_vb.pp("codec_embedding"),
        )?;
        let text_embedding = candle_nn::embedding(
            config.text_vocab_size,
            config.text_hidden_size,
            model_vb.pp("text_embedding"),
        )?;
        let text_projection = ResizeMlp::new(
            config.text_hidden_size,
            config.text_hidden_size,
            config.hidden_size,
            vb.pp("text_projection"),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps,
                config.attention_bias,
                model_vb.pp("layers").pp(i),
            )?);
        }
        let norm = candle_nn::rms_norm(config.hidden_size, config.rms_norm_eps, model_vb.pp("norm"))?;

        let codec_head = linear_no_bias(
            config.hidden_size,
            config.vocab_size,
            vb.pp("codec_head"),
        )?;

        let code_predictor = CodePredictor::new(
            &config.code_predictor_config,
            config.hidden_size,
            vb.pp("code_predictor"),
        )?;

        // M-RoPE
        let (mrope_section, mrope_interleaved) = if let Some(ref rs) = config.rope_scaling {
            (rs.mrope_section.clone(), rs.interleaved)
        } else {
            (vec![], false)
        };

        // Total head_dim for rotary = sum(mrope_section) or head_dim/2
        let rotary_dim = if mrope_section.is_empty() {
            config.head_dim
        } else {
            // The full head_dim is used; mrope_section specifies per-section allocation
            config.head_dim
        };
        let rotary_emb = RotaryEmbedding::new(
            rotary_dim,
            config.max_position_embeddings,
            config.rope_theta,
            model_vb.device(),
        )?;

        Ok(Self {
            codec_embedding,
            text_embedding,
            text_projection,
            layers,
            norm,
            codec_head,
            code_predictor,
            rotary_emb,
            config: config.clone(),
            mrope_section,
            mrope_interleaved,
        })
    }

    /// Build the input embeddings for the talker prefill (streaming mode).
    ///
    /// Matches the vendor reference implementation:
    ///
    /// `text_token_ids` = raw text tokens (no ChatML wrapping).
    ///
    /// Prefill construction:
    ///   [role_prefix(3)]                          — text_proj([im_start, assistant, newline])
    ///   + [tts_pad/bos overlay + codec_prefix]    — tts_pad×(N-2)+tts_bos overlaid on codec[:-1]
    ///   + [first_text + codec_bos]                — text_proj(text[0]) + codec_embed(bos)
    ///
    /// trailing_text_hidden = text_proj(text[1:]) + tts_eos — fed step-by-step during generation
    ///
    /// Returns (input_embeds, trailing_text_hidden, tts_pad_embed)
    pub fn build_prefill_embeds(
        &self,
        text_token_ids: &[u32],
        language: &str,
        speaker: Option<&str>,
        tts_bos_token_id: u32,
        tts_eos_token_id: u32,
        tts_pad_token_id: u32,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let cfg = &self.config;

        // ── 1. Role prefix: text_proj([im_start, assistant, newline]) ──
        let role_ids = Tensor::new(&[151644u32, 77091, 198], device)?;
        let role_embed = self.text_embedding.forward(&role_ids)?.unsqueeze(0)?;
        let role_embed = self.text_projection.forward(&role_embed)?; // [1, 3, D]

        // ── 2. TTS special token embeddings ────────────────────────────
        let tts_ids = Tensor::new(
            &[[tts_pad_token_id, tts_bos_token_id, tts_eos_token_id]],
            device,
        )?;
        let tts_embeds = self.text_embedding.forward(&tts_ids)?;
        let tts_projected = self.text_projection.forward(&tts_embeds)?;
        let tts_pad_embed = tts_projected.narrow(1, 0, 1)?; // [1, 1, D]
        let tts_bos_embed = tts_projected.narrow(1, 1, 1)?; // [1, 1, D]
        let tts_eos_embed = tts_projected.narrow(1, 2, 1)?; // [1, 1, D]

        // ── 3. Language & speaker IDs ──────────────────────────────────
        let language_id = if language.to_lowercase() == "auto" {
            None
        } else {
            cfg.codec_language_id.get(&language.to_lowercase()).copied()
        };

        let speaker_id = speaker.and_then(|s| cfg.spk_id.get(&s.to_lowercase()).copied());

        // Check dialect override
        let final_language_id = if let Some(spk_name) = speaker {
            if language.to_lowercase() == "chinese" || language.to_lowercase() == "auto" {
                if let Some(dialect_val) = cfg.spk_is_dialect.get(&spk_name.to_lowercase()) {
                    if let Some(dialect_str) = dialect_val.as_str() {
                        cfg.codec_language_id.get(dialect_str).copied()
                    } else {
                        language_id
                    }
                } else {
                    language_id
                }
            } else {
                language_id
            }
        } else {
            language_id
        };

        // ── 4. Build codec sequence ────────────────────────────────────
        // [think/nothink, think_bos, (lang,) think_eos, (speaker,) pad, bos]
        let mut codec_ids: Vec<u32> = Vec::new();
        if let Some(lid) = final_language_id {
            codec_ids.extend_from_slice(&[
                cfg.codec_think_id as u32,
                cfg.codec_think_bos_id as u32,
                lid as u32,
                cfg.codec_think_eos_id as u32,
            ]);
        } else {
            codec_ids.extend_from_slice(&[
                cfg.codec_nothink_id as u32,
                cfg.codec_think_bos_id as u32,
                cfg.codec_think_eos_id as u32,
            ]);
        }
        if let Some(sid) = speaker_id {
            codec_ids.push(sid as u32);
        }
        codec_ids.push(cfg.codec_pad_id as u32);
        codec_ids.push(cfg.codec_bos_id as u32);

        let codec_tensor = Tensor::new(codec_ids.as_slice(), device)?;
        let codec_embed = self.codec_embedding.forward(&codec_tensor)?.unsqueeze(0)?; // [1, N, D]
        let codec_len = codec_embed.dim(1)?;

        // ── 5. Overlay tts_pad/tts_bos on first (N-1) codec tokens ─────
        // Last one (codec_bos) is combined with first text token
        let n_overlay = codec_len - 1;
        let n_pad = n_overlay - 1;
        let tts_overlay = Tensor::cat(
            &[
                &tts_pad_embed.expand((1, n_pad, tts_pad_embed.dim(2)?))?,
                &tts_bos_embed,
            ],
            1,
        )?; // [1, n_overlay, D]
        let codec_hidden = (tts_overlay + codec_embed.narrow(1, 0, n_overlay)?)?;

        // ── 6. First text token + codec_bos ────────────────────────────
        let codec_bos_embed = codec_embed.narrow(1, codec_len - 1, 1)?; // [1, 1, D]
        let first_text_and_bos = if !text_token_ids.is_empty() {
            let first_id = Tensor::new(&[text_token_ids[0]], device)?;
            let first_embed = self.text_embedding.forward(&first_id)?.unsqueeze(0)?;
            let first_proj = self.text_projection.forward(&first_embed)?;
            (first_proj + codec_bos_embed)?
        } else {
            (tts_pad_embed.clone() + codec_bos_embed)?
        };

        // ── 7. Assemble prefill ────────────────────────────────────────
        let talker_input_embed = Tensor::cat(
            &[&role_embed, &codec_hidden, &first_text_and_bos],
            1,
        )?;

        // ── 8. Trailing text: text_proj(remaining) + tts_eos ───────────
        // In streaming mode, remaining text tokens are fed step-by-step
        // during generation, providing text guidance to the model.
        let trailing_text_hidden = if text_token_ids.len() > 1 {
            let remaining = &text_token_ids[1..];
            let remaining_ids = Tensor::new(remaining, device)?.unsqueeze(0)?;
            let remaining_embed = self.text_embedding.forward(&remaining_ids)?;
            let remaining_proj = self.text_projection.forward(&remaining_embed)?;
            Tensor::cat(&[&remaining_proj, &tts_eos_embed], 1)?
        } else {
            tts_eos_embed.clone()
        };

        let talker_input_embed = talker_input_embed.to_dtype(dtype)?;
        let trailing_text_hidden = trailing_text_hidden.to_dtype(dtype)?;
        let tts_pad_embed = tts_pad_embed.to_dtype(dtype)?;

        Ok((talker_input_embed, trailing_text_hidden, tts_pad_embed))
    }

    /// Run the talker transformer forward on input embeddings.
    ///
    /// `seq_offset`: the position offset for RoPE. During prefill this is 0;
    /// during autoregressive generation step N this is `prefill_len + N`.
    pub fn forward_embeds(
        &mut self,
        inputs_embeds: &Tensor,
        attention_mask: Option<&Tensor>,
        seq_offset: usize,
    ) -> Result<Tensor> {
        let seq_len = inputs_embeds.dim(1)?;
        // Compute cos/sin for positions [seq_offset .. seq_offset + seq_len]
        let total_pos = seq_offset + seq_len;
        let (cos_full, sin_full) = self.rotary_emb.forward(total_pos)?;
        let cos = cos_full.narrow(0, seq_offset, seq_len)?;
        let sin = sin_full.narrow(0, seq_offset, seq_len)?;

        // For TTS all 3 MRoPE position dimensions are identical,
        // so standard 1D RoPE is equivalent.
        let mut hidden_states = inputs_embeds.clone();
        for layer in &mut self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                &cos,
                &sin,
                attention_mask,
                false,
                &self.mrope_section,
                self.mrope_interleaved,
            )?;
        }
        self.norm.forward(&hidden_states)
    }

    /// Generate one codec token (first codebook) from the last hidden state.
    pub fn predict_first_code(&self, hidden_state: &Tensor) -> Result<Tensor> {
        self.codec_head.forward(hidden_state)
    }

    /// Build ICL voice-clone base prefill (9 positions, no first_text + codec_bos).
    ///
    /// Matches the vendor `prefill_voice_clone(icl_mode=true)`:
    ///   [role_prefix(3)] + [tts_pad/bos overlay + codec[think..pad]](6) = 9 positions
    ///
    /// The codec_bos is NOT included here — it starts the ICL prompt instead.
    ///
    /// Returns `(prefill_embeds, tts_pad_embed)`.
    pub fn build_voice_clone_prefill(
        &self,
        spk_embed: &Tensor,           // [enc_dim] speaker x-vector
        language: &str,
        tts_bos_token_id: u32,
        tts_pad_token_id: u32,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let cfg = &self.config;

        // ── Role prefix: text_proj([im_start, assistant, newline]) ──
        let role_ids = Tensor::new(&[151644u32, 77091, 198], device)?;
        let role_embed = self.text_embedding.forward(&role_ids)?.unsqueeze(0)?;
        let role_embed = self.text_projection.forward(&role_embed)?; // [1, 3, D]

        // ── TTS special tokens ──
        let tts_ids = Tensor::new(&[[tts_pad_token_id, tts_bos_token_id]], device)?;
        let tts_embeds = self.text_embedding.forward(&tts_ids)?;
        let tts_proj = self.text_projection.forward(&tts_embeds)?;
        let tts_pad_embed = tts_proj.narrow(1, 0, 1)?; // [1, 1, D]
        let tts_bos_embed = tts_proj.narrow(1, 1, 1)?; // [1, 1, D]

        // ── Language ID ──
        let language_id = if language.to_lowercase() == "auto" {
            None
        } else {
            cfg.codec_language_id.get(&language.to_lowercase()).copied()
        };

        // ── Codec: [think/nothink, think_bos, (lang,) think_eos] + [spk] + [pad, bos] ──
        let mut codec_prefix_ids: Vec<u32> = Vec::new();
        if let Some(lid) = language_id {
            codec_prefix_ids.extend_from_slice(&[
                cfg.codec_think_id as u32,
                cfg.codec_think_bos_id as u32,
                lid as u32,
                cfg.codec_think_eos_id as u32,
            ]);
        } else {
            codec_prefix_ids.extend_from_slice(&[
                cfg.codec_nothink_id as u32,
                cfg.codec_think_bos_id as u32,
                cfg.codec_think_eos_id as u32,
            ]);
        }

        let codec_prefix_tensor = Tensor::new(codec_prefix_ids.as_slice(), device)?;
        let codec_prefix_embed = self.codec_embedding.forward(&codec_prefix_tensor)?.unsqueeze(0)?;

        let speaker = spk_embed.reshape((1, 1, spk_embed.elem_count()))?
            .to_dtype(codec_prefix_embed.dtype())?;

        let codec_suffix_ids = Tensor::new(&[cfg.codec_pad_id as u32, cfg.codec_bos_id as u32], device)?;
        let codec_suffix_embed = self.codec_embedding.forward(&codec_suffix_ids)?.unsqueeze(0)?;

        // Full codec: [prefix, speaker, pad, bos]
        let codec_full = Tensor::cat(&[&codec_prefix_embed, &speaker, &codec_suffix_embed], 1)?;
        let codec_total = codec_full.dim(1)?;

        // Overlay first (N-1) with tts_pad/tts_bos (skip bos — it goes in ICL prompt)
        let n_overlay = codec_total - 1;
        let n_pad = n_overlay - 1;
        let tts_overlay = Tensor::cat(
            &[
                &tts_pad_embed.expand((1, n_pad, tts_pad_embed.dim(2)?))?,
                &tts_bos_embed,
            ],
            1,
        )?; // [1, n_overlay, D]
        let codec_hidden = (tts_overlay + codec_full.narrow(1, 0, n_overlay)?)?;

        // Assemble: [role(3), codec_hidden(n_overlay)]
        let prefill = Tensor::cat(&[&role_embed, &codec_hidden], 1)?;
        let prefill = prefill.to_dtype(dtype)?;
        let tts_pad_embed = tts_pad_embed.to_dtype(dtype)?;

        Ok((prefill, tts_pad_embed))
    }

    /// Build ICL (in-context learning) prompt for voice cloning (streaming mode).
    ///
    /// Matches the official Python `generate_icl_prompt(non_streaming_mode=False)`
    /// and the vendor Rust-3 `build_icl_prompt(non_streaming=false)`:
    ///
    /// Text = text_proj([ref_text, target_text, tts_eos])
    /// Codec = codec_embed([codec_bos, ref_codec_embeds])
    ///
    /// Streaming overlay:
    ///   If text > codec → overlay first n_codec text with codec, remaining text as trailing
    ///   If text ≤ codec → pad text with tts_pad, overlay all, trailing = tts_pad
    ///
    /// Returns `(icl_embed, trailing_text_hidden)`.
    pub fn build_icl_prompt(
        &self,
        target_text_ids: &[u32],   // raw target text tokens
        ref_text_ids: &[u32],      // raw reference text tokens
        ref_codec_embeds: &Tensor, // [1, T_ref, hidden] — summed codec embeddings
        tts_eos_token_id: u32,
        tts_pad_token_id: u32,
        codec_pad_token_id: u32,
        non_streaming_mode: bool,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let cfg = &self.config;

        // Text: [ref_text, target_text, tts_eos] projected
        let mut all_text_ids: Vec<u32> = Vec::new();
        all_text_ids.extend_from_slice(ref_text_ids);
        all_text_ids.extend_from_slice(target_text_ids);
        all_text_ids.push(tts_eos_token_id);

        let text_ids_tensor = Tensor::new(all_text_ids.as_slice(), device)?.unsqueeze(0)?;
        let text_embed = self.text_embedding.forward(&text_ids_tensor)?;
        let text_embed = self.text_projection.forward(&text_embed)?; // [1, N_text, D]
        let n_text = text_embed.dim(1)?;

        // Codec: [codec_bos, ref_codec_embeds]
        let bos_id = Tensor::new(&[cfg.codec_bos_id as u32], device)?;
        let bos_embed = self.codec_embedding.forward(&bos_id)?.unsqueeze(0)?; // [1, 1, D]
        let codec_embed = Tensor::cat(&[&bos_embed, ref_codec_embeds], 1)?; // [1, n_codec, D]
        let n_codec = codec_embed.dim(1)?;

        // TTS pad embed
        let tts_pad_id = Tensor::new(&[[tts_pad_token_id]], device)?;
        let tts_pad_raw = self.text_embedding.forward(&tts_pad_id)?;
        let tts_pad_embed = self.text_projection.forward(&tts_pad_raw)?; // [1, 1, D]

        let d = tts_pad_embed.dim(2)?;
        let (icl_embed, trailing) = if non_streaming_mode {
            // Non-streaming ICL (vendor rs-2 / mlx-style):
            // [text + codec_pad] || [codec + tts_pad], trailing = tts_pad
            let codec_pad_id = Tensor::new(&[codec_pad_token_id], device)?;
            let codec_pad_embed = self.codec_embedding.forward(&codec_pad_id)?.unsqueeze(0)?; // [1, 1, D]
            let codec_pad_broadcast = codec_pad_embed.expand((1, n_text, d))?;
            let text_with_codec_pad = (&text_embed + &codec_pad_broadcast)?;

            let tts_pad_broadcast = tts_pad_embed.expand((1, n_codec, d))?;
            let codec_with_tts_pad = (&codec_embed + &tts_pad_broadcast)?;

            let icl = Tensor::cat(&[&text_with_codec_pad, &codec_with_tts_pad], 1)?;
            (icl, tts_pad_embed.clone())
        } else {
            // Streaming overlay (matching Python and vendor Rust-3)
            if n_text > n_codec {
                // Text longer: first n_codec text overlaid with codec, rest as trailing
                let text_head = text_embed.narrow(1, 0, n_codec)?;
                let icl = (&text_head + &codec_embed)?;
                let text_tail = text_embed.narrow(1, n_codec, n_text - n_codec)?;
                (icl, text_tail)
            } else {
                // Codec longer or equal: pad text with tts_pad, overlay, trailing=tts_pad
                let padded_text = if n_codec > n_text {
                    let pad = tts_pad_embed.expand((1, n_codec - n_text, d))?;
                    Tensor::cat(&[&text_embed, &pad], 1)?
                } else {
                    text_embed
                };
                let icl = (&padded_text + &codec_embed)?;
                (icl, tts_pad_embed.clone())
            }
        };

        let icl_embed = icl_embed.to_dtype(dtype)?;
        let trailing = trailing.to_dtype(dtype)?;

        Ok((icl_embed, trailing))
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
        self.code_predictor.clear_kv_cache();
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Speaker Encoder (ECAPA-TDNN)
// ═══════════════════════════════════════════════════════════════════════

use candle_nn::{Conv1d, Conv1dConfig};

/// Apply 1D reflect padding to a `[B, C, T]` tensor along the time dimension.
/// Uses index_select to avoid contiguity issues with narrow+flip.
fn reflect_pad_1d(x: &Tensor, pad_left: usize, pad_right: usize) -> Result<Tensor> {
    if pad_left == 0 && pad_right == 0 {
        return Ok(x.clone());
    }
    let x = &x.contiguous()?;
    let (_b, _c, t) = x.dims3()?;
    let mut indices = Vec::with_capacity(pad_left + t + pad_right);
    for i in (1..=pad_left).rev() {
        indices.push(i.min(t - 1) as i64);
    }
    for i in 0..t {
        indices.push(i as i64);
    }
    for i in 0..pad_right {
        indices.push((t.saturating_sub(2).saturating_sub(i)).max(0) as i64);
    }
    let idx = Tensor::new(indices.as_slice(), x.device())?;
    x.index_select(&idx, 2)
}

/// 1D convolution with reflect padding (matches Python `padding="same", padding_mode="reflect"`).
struct ReflectConv1d {
    conv: Conv1d,
    pad_left: usize,
    pad_right: usize,
}

impl ReflectConv1d {
    fn new(in_c: usize, out_c: usize, kernel: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let total_pad = dilation * (kernel - 1);
        let pad_left = total_pad / 2;
        let pad_right = total_pad - pad_left;
        let cfg = Conv1dConfig { padding: 0, dilation, ..Default::default() };
        let conv = candle_nn::conv1d(in_c, out_c, kernel, cfg, vb)?;
        Ok(Self { conv, pad_left, pad_right })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let padded = reflect_pad_1d(x, self.pad_left, self.pad_right)?;
        self.conv.forward(&padded)
    }
}

/// TimeDelayNetBlock: reflect-padded Conv1d + ReLU.
struct TdnnBlock {
    conv: ReflectConv1d,
}

impl TdnnBlock {
    fn new(in_c: usize, out_c: usize, kernel: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self { conv: ReflectConv1d::new(in_c, out_c, kernel, dilation, vb.pp("conv"))? })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv.forward(x)?.relu()
    }
}

/// Res2Net block (scale=8 chunks, residual accumulation).
struct Res2NetBlock {
    blocks: Vec<TdnnBlock>,
    scale: usize,
}

impl Res2NetBlock {
    fn new(in_c: usize, out_c: usize, scale: usize, kernel: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let ch = in_c / scale;
        let och = out_c / scale;
        let mut blocks = Vec::with_capacity(scale - 1);
        for i in 0..(scale - 1) {
            blocks.push(TdnnBlock::new(ch, och, kernel, dilation, vb.pp("blocks").pp(i))?);
        }
        Ok(Self { blocks, scale })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let chunks = x.chunk(self.scale, 1)?;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(self.scale);
        let mut prev: Option<Tensor> = None;
        for (i, chunk) in chunks.iter().enumerate() {
            let out = if i == 0 {
                chunk.clone()
            } else {
                let inp = match &prev {
                    Some(p) => (chunk + p)?,
                    None => chunk.clone(),
                };
                let out = self.blocks[i - 1].forward(&inp)?;
                prev = Some(out.clone());
                out
            };
            if i > 0 { prev = Some(out.clone()); }
            outputs.push(out);
        }
        Tensor::cat(&outputs, 1)
    }
}

/// Squeeze-Excitation block (uses plain Conv1d since kernel=1).
struct SeBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl SeBlock {
    fn new(in_c: usize, se_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv1dConfig::default();
        Ok(Self {
            conv1: candle_nn::conv1d(in_c, se_c, 1, cfg, vb.pp("conv1"))?,
            conv2: candle_nn::conv1d(se_c, out_c, 1, cfg, vb.pp("conv2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean(2)?.unsqueeze(2)?;
        let h = self.conv1.forward(&mean)?.relu()?;
        let scale = candle_nn::ops::sigmoid(&self.conv2.forward(&h)?)?;
        x.broadcast_mul(&scale)
    }
}

/// SqueezeExcitationRes2NetBlock: TDNN1 → Res2Net → TDNN2 → SE + residual.
struct SeRes2NetBlock {
    tdnn1: TdnnBlock,
    res2net: Res2NetBlock,
    tdnn2: TdnnBlock,
    se: SeBlock,
}

impl SeRes2NetBlock {
    fn new(in_c: usize, out_c: usize, scale: usize, se_c: usize, kernel: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            tdnn1: TdnnBlock::new(in_c, out_c, 1, 1, vb.pp("tdnn1"))?,
            res2net: Res2NetBlock::new(out_c, out_c, scale, kernel, dilation, vb.pp("res2net_block"))?,
            tdnn2: TdnnBlock::new(out_c, out_c, 1, 1, vb.pp("tdnn2"))?,
            se: SeBlock::new(out_c, se_c, out_c, vb.pp("se_block"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.tdnn1.forward(x)?;
        let h = self.res2net.forward(&h)?;
        let h = self.tdnn2.forward(&h)?;
        let h = self.se.forward(&h)?;
        (h + x)
    }
}

/// Attentive Statistics Pooling.
struct AttentiveStatisticsPooling {
    tdnn: TdnnBlock,
    conv: Conv1d,
}

impl AttentiveStatisticsPooling {
    fn new(channels: usize, attn_channels: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            tdnn: TdnnBlock::new(channels * 3, attn_channels, 1, 1, vb.pp("tdnn"))?,
            conv: candle_nn::conv1d(attn_channels, channels, 1, Conv1dConfig::default(), vb.pp("conv"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t) = x.dims3()?;
        // Global statistics
        let mean = x.mean(2)?.unsqueeze(2)?;        // [B, C, 1]
        let diff = x.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean(2)?.unsqueeze(2)?; // [B, C, 1]
        let std = (var + 1e-5)?.sqrt()?;

        let mean_exp = mean.broadcast_as((b, c, t))?;
        let std_exp = std.broadcast_as((b, c, t))?;

        // [x, mean, std] → [B, 3C, T]
        let attn_in = Tensor::cat(&[x, &mean_exp, &std_exp], 1)?;

        // Attention: TDNN(3C→attn_ch, +ReLU) → Tanh → Conv(attn_ch→C) → Softmax
        let attn = self.tdnn.forward(&attn_in)?;
        let attn = attn.tanh()?;
        let attn = self.conv.forward(&attn)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?; // softmax over T

        // Weighted mean
        let w_mean = x.broadcast_mul(&attn)?.sum(2)?.unsqueeze(2)?; // [B, C, 1]
        // Weighted std
        let w_diff = x.broadcast_sub(&w_mean)?;
        let w_var = w_diff.sqr()?.broadcast_mul(&attn)?.sum(2)?.unsqueeze(2)?;
        let w_std = (w_var + 1e-5)?.sqrt()?;

        // Output: [B, 2C, 1]
        Ok(Tensor::cat(&[&w_mean, &w_std], 1)?)
    }
}

/// ECAPA-TDNN speaker encoder.
pub struct SpeakerEncoder {
    blocks: Vec<EcapaBlock>,
    mfa: TdnnBlock,
    asp: AttentiveStatisticsPooling,
    fc: Conv1d,
}

enum EcapaBlock {
    Tdnn(TdnnBlock),
    SeRes2Net(SeRes2NetBlock),
}

impl EcapaBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Tdnn(b) => b.forward(x),
            Self::SeRes2Net(b) => b.forward(x),
        }
    }
}

impl SpeakerEncoder {
    pub fn new(cfg: &SpeakerEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let n = cfg.enc_channels.len();
        let mut blocks: Vec<EcapaBlock> = Vec::with_capacity(n - 1);

        // Block 0: initial TDNN
        blocks.push(EcapaBlock::Tdnn(TdnnBlock::new(
            cfg.mel_dim,
            cfg.enc_channels[0],
            cfg.enc_kernel_sizes[0],
            cfg.enc_dilations[0],
            vb.pp("blocks").pp(0),
        )?));

        // Blocks 1..n-2: SE-Res2Net
        for i in 1..(n - 1) {
            blocks.push(EcapaBlock::SeRes2Net(SeRes2NetBlock::new(
                cfg.enc_channels[i - 1],
                cfg.enc_channels[i],
                cfg.enc_res2net_scale,
                cfg.enc_se_channels,
                cfg.enc_kernel_sizes[i],
                cfg.enc_dilations[i],
                vb.pp("blocks").pp(i),
            )?));
        }

        // MFA: uses last enc_channels entry
        let mfa_in = cfg.enc_channels[1..(n - 1)].iter().sum::<usize>();
        let mfa = TdnnBlock::new(
            mfa_in,
            cfg.enc_channels[n - 1],
            cfg.enc_kernel_sizes[n - 1],
            cfg.enc_dilations[n - 1],
            vb.pp("mfa"),
        )?;

        let asp = AttentiveStatisticsPooling::new(cfg.enc_channels[n - 1], cfg.enc_attention_channels, vb.pp("asp"))?;
        let fc = candle_nn::conv1d(cfg.enc_channels[n - 1] * 2, cfg.enc_dim, 1, Conv1dConfig::default(), vb.pp("fc"))?;

        Ok(Self { blocks, mfa, asp, fc })
    }

    /// Forward: input mel `[B, n_mels, T]` → speaker embedding `[B, enc_dim]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // blocks[0]: initial TDNN
        let mut x = self.blocks[0].forward(mel)?;

        // blocks[1..]: SE-Res2Net, collect outputs for MFA
        let mut se_outputs = Vec::with_capacity(self.blocks.len() - 1);
        for block in &self.blocks[1..] {
            x = block.forward(&x)?;
            se_outputs.push(x.clone());
        }

        // MFA: concatenate SE-Res2Net outputs along channel dim
        let mfa_in = Tensor::cat(&se_outputs, 1)?;
        let h = self.mfa.forward(&mfa_in)?;

        // ASP: attentive statistics pooling → [B, 2C, 1]
        let pooled = self.asp.forward(&h)?;

        // FC: project to embedding dimension → [B, enc_dim, 1]
        let embed = self.fc.forward(&pooled)?;
        embed.squeeze(2) // [B, enc_dim]
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Full Qwen3-TTS Model
// ═══════════════════════════════════════════════════════════════════════

/// Top-level Qwen3-TTS model — owns the talker and generates speech codec tokens.
/// Code-to-waveform conversion is handled by the speech tokenizer decoder backend.
pub struct Qwen3TTSModel {
    pub talker: TalkerModel,
    pub speaker_encoder: Option<SpeakerEncoder>,
    pub config: Qwen3TTSConfig,
    pub device: Device,
    pub dtype: DType,
}

impl Qwen3TTSModel {
    pub fn new(config: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let talker = TalkerModel::new(&config.talker_config, vb.pp("talker"))?;
        let speaker_encoder = if config.tts_model_type.as_deref() == Some("base") {
            // Speaker encoder must run in F32 for accurate x-vector extraction.
            // BF16 loses too much precision in the ECAPA-TDNN and produces
            // wrong speaker embeddings (voice not cloned). Matches vendor.
            let se_vb = vb.pp("speaker_encoder").set_dtype(DType::F32);
            match SpeakerEncoder::new(&config.speaker_encoder_config, se_vb) {
                Ok(enc) => Some(enc),
                Err(e) => {
                    eprintln!("Warning: failed to load speaker_encoder weights: {e}");
                    None
                }
            }
        } else {
            None
        };
        Ok(Self {
            talker,
            speaker_encoder,
            config: config.clone(),
            device,
            dtype,
        })
    }

    pub fn clear_kv_cache(&mut self) {
        self.talker.clear_kv_cache();
    }

    /// Build a lower-triangular causal attention mask.
    /// Shape: `[1, 1, seq_len, seq_len]`, with 0.0 for allowed and -inf for blocked.
    fn build_causal_mask(seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let mut mask_data = vec![0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::new(mask_data.as_slice(), device)?
            .reshape((1, 1, seq_len, seq_len))?
            .to_dtype(dtype)?;
        Ok(mask)
    }

    /// Build a causal mask for new tokens attending to previous KV cache + themselves.
    /// Shape: `[1, 1, seq_len, offset + seq_len]`.
    /// New token i can attend to all positions 0..(offset + i), but not future positions.
    fn build_causal_mask_with_offset(seq_len: usize, offset: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let full_len = offset + seq_len;
        let mut mask_data = vec![0f32; seq_len * full_len];
        for i in 0..seq_len {
            for j in (offset + i + 1)..full_len {
                mask_data[i * full_len + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::new(mask_data.as_slice(), device)?
            .reshape((1, 1, seq_len, full_len))?
            .to_dtype(dtype)?;
        Ok(mask)
    }

    /// Generate speech codec tokens from text.
    ///
    /// Returns a Vec of (num_code_groups) tokens per time step.
    pub fn generate_speech_codes(
        &mut self,
        text_token_ids: &[u32],
        language: &str,
        speaker: Option<&str>,
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        repetition_penalty: f32,
    ) -> Result<Vec<Vec<u32>>> {
        self.clear_kv_cache();

        let (prefill_embeds, trailing_text_hidden, tts_pad_embed) =
            self.talker.build_prefill_embeds(
                text_token_ids,
                language,
                speaker,
                self.config.tts_bos_token_id,
                self.config.tts_eos_token_id,
                self.config.tts_pad_token_id,
                &self.device,
                self.dtype,
            )?;

        // Build causal attention mask for prefill
        let prefill_len = prefill_embeds.dim(1)?;
        let causal_mask = Self::build_causal_mask(prefill_len, &self.device, self.dtype)?;

        // Prefill with causal mask, positions start at 0
        let hidden_states =
            self.talker
                .forward_embeds(&prefill_embeds, Some(&causal_mask), 0)?;

        let eos_token_id = self.config.talker_config.codec_eos_token_id as u32;
        let mut all_codes = Vec::new();
        let mut logits_processor =
            candle_transformers::generation::LogitsProcessor::from_sampling(
                42,
                candle_transformers::generation::Sampling::TopKThenTopP {
                    k: 50,
                    p: top_p.unwrap_or(1.0),
                    temperature,
                },
            );

        // Build suppress_tokens mask as a GPU tensor: suppress [vocab_size-1024, vocab_size) except EOS.
        // 0.0 for allowed, -inf for suppressed.
        let vocab_size = self.config.talker_config.vocab_size;
        let suppress_start = vocab_size.saturating_sub(1024);
        let mut suppress_mask_data = vec![0f32; vocab_size];
        for i in suppress_start..vocab_size {
            if i as u32 != eos_token_id {
                suppress_mask_data[i] = f32::NEG_INFINITY;
            }
        }
        let suppress_mask = Tensor::new(suppress_mask_data.as_slice(), &self.device)?;
        // EOS-suppress mask for min_new_tokens=2
        let mut eos_suppress_data = vec![0f32; vocab_size];
        eos_suppress_data[eos_token_id as usize] = f32::NEG_INFINITY;
        let eos_suppress_mask = Tensor::new(eos_suppress_data.as_slice(), &self.device)?;

        // Last hidden state from prefill
        let seq_len = hidden_states.dim(1)?;
        let mut past_hidden = hidden_states.narrow(1, seq_len - 1, 1)?;

        let trailing_len = trailing_text_hidden.dim(1)?;

        for step in 0..max_new_tokens {
            // Predict first codebook token
            let logits = self.talker.predict_first_code(&past_hidden)?
                .squeeze(0)?.squeeze(0)?
                .to_dtype(DType::F32)?;

            // Apply repetition penalty
            let logits = if repetition_penalty != 1.0 && !all_codes.is_empty() {
                let recent: Vec<u32> = all_codes.iter().map(|c: &Vec<u32>| c[0]).collect();
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    repetition_penalty,
                    &recent,
                )?
            } else {
                logits
            };

            // Suppress special tokens (GPU-side) + EOS for first 2 steps (min_new_tokens=2)
            let logits = if step < 2 {
                (logits + &suppress_mask + &eos_suppress_mask)?
            } else {
                (logits + &suppress_mask)?
            };

            let first_code = logits_processor.sample(&logits)?;

            if first_code == eos_token_id {
                break;
            }

            // Predict remaining codebooks using code predictor (with sampling)
            let talker_hidden_1d = past_hidden.squeeze(0)?.squeeze(0)?; // [D]
            let remaining_codes = self.talker.code_predictor.predict(
                &talker_hidden_1d,
                first_code,
                &self.talker.codec_embedding,
                &self.device,
                temperature,
                top_p,
            )?;

            let mut frame_codes = vec![first_code];
            frame_codes.extend(remaining_codes);
            all_codes.push(frame_codes.clone());

            // Build next step input embedding
            // Sum all codebook embeddings for this frame
            let mut sum_embed = self.talker.codec_embedding.forward(
                &Tensor::new(&[first_code], &self.device)?,
            )?;
            for (i, &code) in frame_codes[1..].iter().enumerate() {
                let embed = self.talker.code_predictor.codec_embeddings[i].forward(
                    &Tensor::new(&[code], &self.device)?,
                )?;
                sum_embed = (sum_embed + embed)?;
            }

            // Add trailing text embedding if available
            let text_contrib = if step < trailing_len {
                trailing_text_hidden.squeeze(0)?.narrow(0, step, 1)?
            } else {
                tts_pad_embed.squeeze(0)?
            };
            let next_input = (sum_embed + text_contrib)?.unsqueeze(0)?;

            // Forward one step with correct position offset (no mask needed for single token with KV cache)
            let pos_offset = prefill_len + step;
            let hs = self.talker.forward_embeds(&next_input, None, pos_offset)?;
            past_hidden = hs.narrow(1, hs.dim(1)? - 1, 1)?;
        }

        if std::env::var("CRANE_TTS_DEBUG").map(|v| v == "1" || v == "true").unwrap_or(false) {
            eprintln!(
                "[CRANE_TTS_DEBUG] generate_speech_codes: generated {} frames, max_new_tokens={}, \
                 trailing_text_len={}, top_p={}, temperature={}, repetition_penalty={}",
                all_codes.len(),
                max_new_tokens,
                trailing_len,
                top_p.unwrap_or(1.0),
                temperature,
                repetition_penalty,
            );
            if let Some(first_frame) = all_codes.first() {
                eprintln!(
                    "[CRANE_TTS_DEBUG]   first frame codes: {:?}",
                    first_frame
                );
            }
            if let Some(last_frame) = all_codes.last() {
                eprintln!(
                    "[CRANE_TTS_DEBUG]   last frame codes: {:?}",
                    last_frame
                );
            }
        }

        self.clear_kv_cache();
        Ok(all_codes)
    }

    /// Generate speech codec tokens using voice-clone (ICL) mode.
    ///
    /// Two-phase approach matching the vendor reference:
    ///   Phase 1: Voice-clone base prefill (9 positions) → populate KV cache
    ///   Phase 2: ICL prompt extension (streaming overlay) → extend KV cache
    ///   Phase 3: Autoregressive generation with trailing text guidance
    ///
    /// `ref_codes`: `[T_ref, num_code_groups]` — codec codes from the reference audio.
    /// `spk_embed`: `[enc_dim]` — speaker x-vector from the speaker encoder.
    /// `ref_token_ids`: raw tokenized reference text.
    ///
    /// Returns `(new_codes, ref_code_len)` where `new_codes` are the generated frames
    /// and `ref_code_len` is the number of reference frames prepended for decoding.
    pub fn generate_voice_clone_codes(
        &mut self,
        text_token_ids: &[u32],
        ref_token_ids: &[u32],
        ref_codes: &Tensor,
        spk_embed: &Tensor,
        language: &str,
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        repetition_penalty: f32,
    ) -> Result<(Vec<Vec<u32>>, usize)> {
        self.clear_kv_cache();

        // ICL guardrails (CPU/F32 safety):
        //   - Keep caller repetition penalty (official default 1.05)
        //   - Max new tokens cap: min(caller, max(75, text_tokens * 6))
        // The length cap prevents runaway loops while avoiding excessive
        // repetition penalty that can destabilize codec quality.
        const ICL_MIN_REP_PENALTY: f32 = 1.05;
        const ICL_MIN_FRAMES: usize = 75;
        const ICL_FRAMES_PER_TOKEN: usize = 6;
        let repetition_penalty = repetition_penalty.max(ICL_MIN_REP_PENALTY);
        let max_new_tokens = max_new_tokens
            .min(ICL_MIN_FRAMES.max(text_token_ids.len() * ICL_FRAMES_PER_TOKEN));

        // ── Phase 1: Build prefill (9 positions) ───────────────────────
        let (prefill_embeds, tts_pad_embed) =
            self.talker.build_voice_clone_prefill(
                spk_embed,
                language,
                self.config.tts_bos_token_id,
                self.config.tts_pad_token_id,
                &self.device,
                self.dtype,
            )?;

        // ── Phase 2: Build ICL prompt ──────────────────────────────────
        // Sum reference codec embeddings across all codebook groups
        let ref_codes_u32 = ref_codes.to_dtype(DType::U32)?;
        let num_groups = ref_codes.dim(1)?;
        let mut ref_codec_sum = self.talker.codec_embedding.forward(
            &ref_codes_u32.narrow(1, 0, 1)?.squeeze(1)?.contiguous()?
        )?.unsqueeze(0)?; // [1, T_ref, D]
        for g in 1..num_groups {
            let g_codes = ref_codes_u32.narrow(1, g, 1)?.squeeze(1)?.contiguous()?;
            let g_embed = if g - 1 < self.talker.code_predictor.codec_embeddings.len() {
                self.talker.code_predictor.codec_embeddings[g - 1].forward(&g_codes)?
            } else {
                self.talker.codec_embedding.forward(&g_codes)?
            };
            ref_codec_sum = (ref_codec_sum + g_embed.unsqueeze(0)?)?;
        }

        // Build ICL prompt.
        // Default to streaming mode (official Python behavior).
        // Set CRANE_TTS_ICL_NON_STREAMING=1 to opt into non-streaming mode.
        let use_non_streaming_icl = std::env::var("CRANE_TTS_ICL_NON_STREAMING")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let (icl_embed, trailing_text_hidden) = self.talker.build_icl_prompt(
            text_token_ids,
            ref_token_ids,
            &ref_codec_sum,
            self.config.tts_eos_token_id,
            self.config.tts_pad_token_id,
            self.config.talker_config.codec_pad_id as u32,
            use_non_streaming_icl,
            &self.device,
            self.dtype,
        )?;

        // ── Combined prefill: run [prefill + ICL] in a single forward pass ──
        // This matches the Python implementation which concatenates prefill and
        // ICL embeddings before calling model forward, ensuring a single unified
        // KV cache population and attention computation.
        let combined_embeds = Tensor::cat(&[&prefill_embeds, &icl_embed], 1)?;
        let combined_len = combined_embeds.dim(1)?;
        let causal_mask = Self::build_causal_mask(combined_len, &self.device, self.dtype)?;
        let hidden_states = self.talker.forward_embeds(&combined_embeds, Some(&causal_mask), 0)?;
        let offset = combined_len;
        let mut past_hidden = hidden_states.narrow(1, hidden_states.dim(1)? - 1, 1)?;

        if std::env::var("CRANE_TTS_DEBUG").map(|v| v == "1" || v == "true").unwrap_or(false) {
            let h_norm: f32 = past_hidden.to_dtype(DType::F32)?.sqr()?.sum_all()?.sqrt()?.to_scalar()?;
            eprintln!(
                "[CRANE_TTS_DEBUG] combined prefill: len={}, last_hidden_norm={:.4}",
                combined_len, h_norm,
            );
        }

        // ── Phase 3: Autoregressive generation ─────────────────────────
        let eos_token_id = self.config.talker_config.codec_eos_token_id as u32;
        let mut all_codes = Vec::new();
        let mut logits_processor =
            candle_transformers::generation::LogitsProcessor::from_sampling(
                42,
                candle_transformers::generation::Sampling::TopKThenTopP {
                    k: 50,
                    p: top_p.unwrap_or(1.0),
                    temperature,
                },
            );

        let vocab_size = self.config.talker_config.vocab_size;
        let suppress_start = vocab_size.saturating_sub(1024);
        let mut suppress_mask_data = vec![0f32; vocab_size];
        for i in suppress_start..vocab_size {
            if i as u32 != eos_token_id {
                suppress_mask_data[i] = f32::NEG_INFINITY;
            }
        }
        let suppress_mask = Tensor::new(suppress_mask_data.as_slice(), &self.device)?;
        let mut eos_suppress_data = vec![0f32; vocab_size];
        eos_suppress_data[eos_token_id as usize] = f32::NEG_INFINITY;
        let eos_suppress_mask = Tensor::new(eos_suppress_data.as_slice(), &self.device)?;

        // GPU-side repetition penalty mask: tracks which tokens have been seen.
        // Avoids expensive GPU→CPU roundtrip per step (matching vendor approach).
        let mut penalty_seen = vec![false; vocab_size];

        let trailing_len = trailing_text_hidden.dim(1)?;
        let tts_debug = std::env::var("CRANE_TTS_DEBUG").map(|v| v == "1" || v == "true").unwrap_or(false);

        if tts_debug {
            eprintln!(
                "[CRANE_TTS_DEBUG] voice_clone gen starting: eos_token_id={}, vocab_size={}, \
                 suppress_start={}, offset={}, trailing_len={}, max_new_tokens={}, rep_penalty={:.2}, icl_mode={}",
                eos_token_id, vocab_size, suppress_start, offset, trailing_len, max_new_tokens, repetition_penalty,
                if use_non_streaming_icl { "non_streaming" } else { "streaming" },
            );
        }

        for step in 0..max_new_tokens {
            let logits = self.talker.predict_first_code(&past_hidden)?
                .squeeze(0)?.squeeze(0)?
                .to_dtype(DType::F32)?;

            // Apply repetition penalty (GPU-side, matching HuggingFace/vendor semantics):
            // positive logits → divide by penalty, negative logits → multiply by penalty
            let logits = if repetition_penalty != 1.0 && !all_codes.is_empty() {
                let logits_vec: Vec<f32> = logits.to_vec1()?;
                let mut penalized = logits_vec;
                for (idx, seen) in penalty_seen.iter().enumerate() {
                    if *seen {
                        if penalized[idx] >= 0.0 {
                            penalized[idx] /= repetition_penalty;
                        } else {
                            penalized[idx] *= repetition_penalty;
                        }
                    }
                }
                Tensor::new(penalized.as_slice(), &self.device)?
            } else {
                logits
            };

            // Suppress special tokens + EOS for first 2 steps (min_new_tokens=2)
            let logits = if step < 2 {
                (logits + &suppress_mask + &eos_suppress_mask)?
            } else {
                (logits + &suppress_mask)?
            };

            // Diagnostic: log EOS logit and top-5 values for select steps
            if tts_debug && (step < 5 || step % 500 == 0 || step == max_new_tokens - 1) {
                let logits_vec: Vec<f32> = logits.to_vec1()?;
                let eos_logit = logits_vec.get(eos_token_id as usize).copied().unwrap_or(f32::NAN);
                let mut indexed: Vec<(usize, f32)> = logits_vec.iter().copied().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top5: Vec<(usize, f32)> = indexed.iter().take(5).copied().collect();
                let eos_rank = indexed.iter().position(|(i, _)| *i == eos_token_id as usize).unwrap_or(9999);
                eprintln!(
                    "[CRANE_TTS_DEBUG] step={step}: eos_logit={eos_logit:.3} eos_rank={eos_rank} \
                     top5={:?} trailing={}", top5,
                    if step < trailing_len { "text" } else { "pad" },
                );
            }

            let first_code = logits_processor.sample(&logits)?;
            if first_code == eos_token_id {
                break;
            }

            // Track seen tokens for repetition penalty
            if (first_code as usize) < vocab_size {
                penalty_seen[first_code as usize] = true;
            }

            let talker_hidden_1d = past_hidden.squeeze(0)?.squeeze(0)?;
            let remaining_codes = self.talker.code_predictor.predict(
                &talker_hidden_1d,
                first_code,
                &self.talker.codec_embedding,
                &self.device,
                temperature,
                top_p,
            )?;

            let mut frame_codes = vec![first_code];
            frame_codes.extend(remaining_codes);
            all_codes.push(frame_codes.clone());

            let mut sum_embed = self.talker.codec_embedding.forward(
                &Tensor::new(&[first_code], &self.device)?,
            )?;
            for (i, &code) in frame_codes[1..].iter().enumerate() {
                let embed = self.talker.code_predictor.codec_embeddings[i].forward(
                    &Tensor::new(&[code], &self.device)?,
                )?;
                sum_embed = (sum_embed + embed)?;
            }

            let text_contrib = if step < trailing_len {
                trailing_text_hidden.squeeze(0)?.narrow(0, step, 1)?
            } else {
                tts_pad_embed.squeeze(0)?
            };
            let next_input = (sum_embed + text_contrib)?.unsqueeze(0)?;

            let pos_offset = offset + step;
            let hs = self.talker.forward_embeds(&next_input, None, pos_offset)?;
            past_hidden = hs.narrow(1, hs.dim(1)? - 1, 1)?;

            // Diagnostic: hidden state norm at select steps
            if tts_debug && (step < 5 || step % 100 == 0 || step == max_new_tokens - 1) {
                let h_norm: f32 = past_hidden.to_dtype(DType::F32)?.sqr()?.sum_all()?.sqrt()?.to_scalar()?;
                let inp_norm: f32 = next_input.to_dtype(DType::F32)?.sqr()?.sum_all()?.sqrt()?.to_scalar()?;
                eprintln!(
                    "[CRANE_TTS_DEBUG] step={step}: hidden_norm={h_norm:.4} input_norm={inp_norm:.4} code={first_code}",
                );
            }
        }

        self.clear_kv_cache();

        if tts_debug {
            eprintln!(
                "[CRANE_TTS_DEBUG] generate_voice_clone_codes: generated {} frames, \
                 max_new_tokens={}, trailing_text_len={}, rep_penalty={:.2}, \
                 ref_code_len={}, text_tokens={}, ref_text_tokens={}",
                all_codes.len(),
                max_new_tokens,
                trailing_len,
                repetition_penalty,
                ref_codes.dim(0).unwrap_or(0),
                text_token_ids.len(),
                ref_token_ids.len(),
            );
            if let Some(first_frame) = all_codes.first() {
                eprintln!("[CRANE_TTS_DEBUG]   first frame codes: {:?}", first_frame);
            }
            if let Some(last_frame) = all_codes.last() {
                eprintln!("[CRANE_TTS_DEBUG]   last frame codes: {:?}", last_frame);
            }
        }

        // ref_code_len: number of reference frames in ref_codes (for trimming after decode)
        let ref_code_len = ref_codes.dim(0)?;
        Ok((all_codes, ref_code_len))
    }
}
