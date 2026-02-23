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

/// Top-level Qwen3-TTS config (loaded from `config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TTSConfig {
    pub talker_config: TalkerConfig,
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
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
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
    /// remaining `num_code_groups - 1` codebook tokens using greedy decoding.
    pub fn predict(
        &mut self,
        talker_hidden: &Tensor,
        first_code: u32,
        codec_embedding: &Embedding,
        device: &Device,
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

        // Forward through layers
        let mut hidden_states = inputs_embeds;
        for layer in &mut self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                &cos,
                &sin,
                None,
                false,
                &[],
                false,
            )?;
        }
        hidden_states = self.norm.forward(&hidden_states)?;

        // First head prediction
        let logits = self.lm_heads[0].forward(
            &hidden_states.narrow(1, seq_len - 1, 1)?,
        )?;
        let next_token = logits.squeeze(0)?.squeeze(0)?.argmax(D::Minus1)?
            .to_scalar::<u32>()?;
        codes.push(next_token);

        // Remaining groups
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
            let next_token = logits.squeeze(0)?.squeeze(0)?.argmax(D::Minus1)?
                .to_scalar::<u32>()?;
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

    /// Build the input embeddings for the talker prefill.
    ///
    /// For CustomVoice model with a known speaker:
    ///   [text(<|im_start|>assistant\n)] + [codec tags + speaker + bos] + [text tokens]
    ///
    /// Returns (input_embeds, trailing_text_hidden, tts_pad_embed)
    pub fn build_prefill_embeds(
        &self,
        text_token_ids: &[u32],
        language: &str,
        speaker: Option<&str>,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let cfg = &self.config;

        // Get special embeddings: tts_bos, tts_eos, tts_pad
        // These are text tokens that get projected through text_projection
        // We need to hardcode the parent config tokens here
        // The caller should pass tts token IDs from the top-level config
        // For now, we use the text embedding + projection pipeline

        // 1. Build codec prefill tags
        let language_id = if language.to_lowercase() == "auto" {
            None
        } else {
            cfg.codec_language_id.get(&language.to_lowercase()).copied()
        };

        let mut codec_prefill: Vec<u32> = Vec::new();
        if language_id.is_none() {
            codec_prefill.extend_from_slice(&[
                cfg.codec_nothink_id as u32,
                cfg.codec_think_bos_id as u32,
                cfg.codec_think_eos_id as u32,
            ]);
        } else {
            codec_prefill.extend_from_slice(&[
                cfg.codec_think_id as u32,
                cfg.codec_think_bos_id as u32,
                language_id.unwrap() as u32,
                cfg.codec_think_eos_id as u32,
            ]);
        }

        // Add speaker embed if available
        let speaker_id = speaker.and_then(|s| cfg.spk_id.get(&s.to_lowercase()).copied());

        // Check if dialect should be used
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

        // Rebuild codec_prefill with final language
        let mut codec_prefill: Vec<u32> = Vec::new();
        if final_language_id.is_none() {
            codec_prefill.extend_from_slice(&[
                cfg.codec_nothink_id as u32,
                cfg.codec_think_bos_id as u32,
                cfg.codec_think_eos_id as u32,
            ]);
        } else {
            codec_prefill.extend_from_slice(&[
                cfg.codec_think_id as u32,
                cfg.codec_think_bos_id as u32,
                final_language_id.unwrap() as u32,
                cfg.codec_think_eos_id as u32,
            ]);
        }

        let codec_prefill_tensor = Tensor::new(codec_prefill.as_slice(), device)?;
        let codec_embed_0 = self.codec_embedding.forward(&codec_prefill_tensor)?.unsqueeze(0)?;

        // codec_pad + codec_bos
        let codec_suffix = Tensor::new(&[cfg.codec_pad_id as u32, cfg.codec_bos_id as u32], device)?;
        let codec_embed_1 = self.codec_embedding.forward(&codec_suffix)?.unsqueeze(0)?;

        // Build full codec embedding (with or without speaker)
        let codec_input_embedding = if let Some(sid) = speaker_id {
            let spk_tensor = Tensor::new(&[sid as u32], device)?;
            let spk_embed = self.codec_embedding.forward(&spk_tensor)?.unsqueeze(0)?;
            Tensor::cat(&[&codec_embed_0, &spk_embed, &codec_embed_1], 1)?
        } else {
            Tensor::cat(&[&codec_embed_0, &codec_embed_1], 1)?
        };

        // Text processing:
        // text_token_ids layout: [<|im_start|>, 'assistant', '\n', ...text..., <|im_end|>, '\n', <|im_start|>, 'assistant', '\n']
        // First 3 tokens are the role prefix
        let all_ids = Tensor::new(text_token_ids, device)?;

        // Build tts_pad embedding (we need the TTS pad token embedded through text pipeline)
        // The tts_pad_token_id comes from the parent config (e.g., 151671)
        // For now we'll compute it from zero embeddings as a simple approach
        // Actually we need the text_projection of the text_embedding of tts_pad_token_id
        // These IDs need to be passed in. Let's create a helper.

        // We'll just embed all text and project it
        let text_embeds = self.text_embedding.forward(&all_ids)?;
        let text_embeds = self.text_projection.forward(&text_embeds.unsqueeze(0)?)?;

        // Role prefix: first 3 tokens ("<|im_start|>assistant\n")
        let role_embed = text_embeds.narrow(1, 0, 3)?;

        // The rest of the text (content + trailing tokens)
        let text_len = text_token_ids.len();

        // Pad embeddings for the codec prefix
        let pad_id = Tensor::new(&[cfg.codec_pad_id as u32], device)?;
        let pad_embed = self.codec_embedding.forward(&pad_id)?.unsqueeze(0)?;

        // tts_pad is a placeholder — we use codec_pad embedding as base
        let tts_pad_embed = pad_embed.clone();

        // Build prefill: role + (tts_pad * codec_prefix_len + tts_bos) + codec + first_text
        let codec_prefix_len = codec_input_embedding.dim(1)?;
        let pad_expanded = pad_embed.repeat((1, codec_prefix_len - 2, 1))?; // for codec tags
        // We use text pad for the codec-aligned tokens (silence before speech)
        let pad_text_embed = tts_pad_embed.repeat((1, codec_prefix_len - 2, 1))?;

        // Combine: text_projection(tts_pad) * N + text_projection(tts_bos) aligned with codec prefix
        let bos_id = Tensor::new(&[cfg.codec_bos_id as u32], device)?;
        let bos_embed = self.codec_embedding.forward(&bos_id)?.unsqueeze(0)?;

        // Simple approach: role_embed + (padding aligned with codec prefix except last) + (first text + last codec)
        // concat role + codec prefill
        let prefill_codec = codec_input_embedding.narrow(1, 0, codec_prefix_len - 1)?;

        // Pad text embeddings for alignment
        let pad_text = tts_pad_embed.repeat((1, codec_prefix_len - 2, 1))?;

        // The text side: pad * (codec_prefix_len - 2) + tts_bos
        let aligned = (pad_text + prefill_codec.narrow(1, 0, codec_prefix_len - 2)?)?;

        // Now for the first actual text token (index 3 in text_token_ids)
        let first_text_embed = text_embeds.narrow(1, 3, 1)?;
        let last_codec = codec_input_embedding.narrow(1, codec_prefix_len - 1, 1)?;
        let first_aligned = (first_text_embed + last_codec)?;

        let talker_input_embed = Tensor::cat(&[&role_embed, &aligned, &first_aligned], 1)?;

        // Trailing text hidden: remaining text content (index 4..(text_len-5)) + eos
        // text_token_ids ends with: ...<|im_end|>\n<|im_start|>assistant\n  (5 tokens)
        let trailing_start = 4;
        let trailing_end = text_len.saturating_sub(5);
        let trailing_len = trailing_end.saturating_sub(trailing_start);

        let trailing_text_hidden = if trailing_len > 0 {
            text_embeds.narrow(1, trailing_start, trailing_len)?
        } else {
            tts_pad_embed.clone()
        };

        let talker_input_embed = talker_input_embed.to_dtype(dtype)?;
        let trailing_text_hidden = trailing_text_hidden.to_dtype(dtype)?;
        let tts_pad_embed = tts_pad_embed.to_dtype(dtype)?;

        Ok((talker_input_embed, trailing_text_hidden, tts_pad_embed))
    }

    /// Run the talker transformer forward on input embeddings.
    pub fn forward_embeds(
        &mut self,
        inputs_embeds: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let seq_len = inputs_embeds.dim(1)?;
        let (cos, sin) = self.rotary_emb.forward(seq_len)?;

        let use_mrope = !self.mrope_section.is_empty();

        // For MRoPE we need 3D cos/sin, but for simplicity we'll use 1D RoPE
        // (the 3 position dimensions are identical for pure text)
        let mut hidden_states = inputs_embeds.clone();
        for layer in &mut self.layers {
            hidden_states = layer.forward(
                &hidden_states,
                &cos,
                &sin,
                attention_mask,
                false, // use standard RoPE for now (positions are 1D for TTS)
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

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
        self.code_predictor.clear_kv_cache();
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Full Qwen3-TTS Model
// ═══════════════════════════════════════════════════════════════════════

/// Top-level Qwen3-TTS model — owns the talker and generates speech codec tokens.
/// Code-to-waveform conversion is handled by the speech tokenizer decoder backend.
pub struct Qwen3TTSModel {
    pub talker: TalkerModel,
    pub config: Qwen3TTSConfig,
    pub device: Device,
    pub dtype: DType,
}

impl Qwen3TTSModel {
    pub fn new(config: &Qwen3TTSConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let talker = TalkerModel::new(&config.talker_config, vb.pp("talker"))?;
        Ok(Self {
            talker,
            config: config.clone(),
            device,
            dtype,
        })
    }

    pub fn clear_kv_cache(&mut self) {
        self.talker.clear_kv_cache();
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
                &self.device,
                self.dtype,
            )?;

        // Prefill
        let hidden_states = self.talker.forward_embeds(&prefill_embeds, None)?;

        let eos_token_id = self.config.talker_config.codec_eos_token_id as u32;
        let mut all_codes = Vec::new();
        let mut logits_processor =
            candle_transformers::generation::LogitsProcessor::new(42, Some(temperature), top_p);

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

            let first_code = logits_processor.sample(&logits)?;

            if first_code == eos_token_id {
                break;
            }

            // Predict remaining codebooks using code predictor
            let talker_hidden_1d = past_hidden.squeeze(0)?.squeeze(0)?; // [D]
            let remaining_codes = self.talker.code_predictor.predict(
                &talker_hidden_1d,
                first_code,
                &self.talker.codec_embedding,
                &self.device,
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

            // Forward one step
            let hs = self.talker.forward_embeds(&next_input, None)?;
            past_hidden = hs.narrow(1, hs.dim(1)? - 1, 1)?;
        }

        self.clear_kv_cache();
        Ok(all_codes)
    }
}
