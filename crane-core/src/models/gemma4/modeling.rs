//! Gemma 4 text decoder implementation.
//!
//! Implements Google's Gemma 4 E2B architecture with:
//!
//! 1. **Hybrid attention** — sliding window (512 tokens) + full causal attention
//!    per a repeating `layer_types` pattern.
//! 2. **Dual RoPE** — sliding: theta=10K, full rotation over head_dim=256.
//!    Full: theta=1M, partial rotation (25%) over global_head_dim=512.
//! 3. **Per-Layer Embeddings (PLE)** — gated per-layer token embeddings applied
//!    after attention+MLP as a final residual step in each decoder layer.
//! 4. **KV cache sharing** — layers 15-34 share K/V from prior non-shared layers
//!    of the same type (sliding→layer 13, full→layer 14).
//! 5. **GELU-tanh activation** — MLP uses `gelu_pytorch_tanh` instead of SiLU.
//! 6. **Logit softcapping** — `tanh(logits / cap) * cap` before output.
//! 7. **Pre-allocated KV cache** with in-place `slice_set` writes.
//! 8. **GQA-grouped SDPA** for decode (seq_len=1).
//! 9. **GGUF quantization** via the polymorphic `LinearLayer` enum.
//! 10. **Pre+post norm pairs** — 4 norms per layer (Gemma-style).
//! 11. **QK norms** — per-head RMSNorm applied before RoPE.
//! 12. **Layer scalar** — learnable per-layer scaling factor.

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::rotary_emb::rope;
use candle_nn::{linear_no_bias, Embedding, Linear, RmsNorm, VarBuilder};
use serde::Deserialize;
use std::collections::HashMap;
use std::io::{Read, Seek};

// Reuse the polymorphic linear layer and GGUF loader from the shared Hunyuan module.
pub use crate::models::hunyuan_dense::modeling::{Gguf, LinearLayer};

/// Create an RmsNorm with Gemma-style weight shift.
///
/// Gemma's RMSNorm uses `output = x * (1 + weight)` instead of `output = x * weight`.
/// Weights in HF checkpoints are stored as the raw trained values (initialized near 0),
/// so we add 1.0 to convert them for candle's standard `x * weight` formula.
fn gemma_rms_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<RmsNorm> {
    let weight = vb.get(size, "weight")?;
    let shifted = (weight + 1.0)?;
    Ok(RmsNorm::new(shifted, eps))
}

/// Same shift for GGUF norms: Gemma4 stores raw weights (norm_shift=0),
/// so we must add 1.0 for the `x * weight` formula.
fn gemma_rms_norm_gguf<R: Read + Seek>(
    gg: &mut Gguf<R>,
    name: &str,
    eps: f64,
) -> Result<RmsNorm> {
    let norm = gg.rms_norm(name, eps)?;
    let weight = norm.into_inner().weight().clone();
    let shifted = (weight + 1.0)?;
    Ok(RmsNorm::new(shifted, eps))
}

// ── Event-tracking RAII guard ────────────────────────────────────────────

#[cfg(feature = "cuda")]
struct EventTrackingGuard;

#[cfg(feature = "cuda")]
impl EventTrackingGuard {
    fn disable(device: &candle_core::Device) -> Self {
        if let candle_core::Device::Cuda(ref dev) = device {
            if dev.is_event_tracking() {
                unsafe { dev.disable_event_tracking() };
            }
        }
        Self
    }
}

// ── Config ──────────────────────────────────────────────────────────────

/// Outer config.json wrapper — Gemma 4 nests text config under `text_config`.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4Config {
    pub text_config: Gemma4TextConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeConfig {
    pub rope_theta: f64,
    #[serde(default)]
    pub rope_type: String,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub full_attention: RopeConfig,
    pub sliding_attention: RopeConfig,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default)]
    pub global_head_dim: Option<usize>,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub hidden_activation: Option<String>,
    #[serde(default)]
    pub use_double_wide_mlp: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub hidden_size_per_layer_input: Option<usize>,
    #[serde(default)]
    pub vocab_size_per_layer_input: Option<usize>,
    #[serde(default)]
    pub num_kv_shared_layers: Option<usize>,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

impl Gemma4TextConfig {
    pub fn global_head_dim(&self) -> usize {
        self.global_head_dim.unwrap_or(self.head_dim)
    }

    pub fn parsed_layer_types(&self) -> Vec<LayerType> {
        self.layer_types
            .iter()
            .map(|s| match s.as_str() {
                "sliding_attention" => LayerType::SlidingAttention,
                "full_attention" => LayerType::FullAttention,
                _ => LayerType::SlidingAttention,
            })
            .collect()
    }

    pub fn first_kv_shared_layer(&self) -> usize {
        let n_shared = self.num_kv_shared_layers.unwrap_or(0);
        self.num_hidden_layers.saturating_sub(n_shared)
    }

    pub fn kv_sharing_map(&self) -> Vec<Option<usize>> {
        let layer_types = self.parsed_layer_types();
        let first_shared = self.first_kv_shared_layer();
        let non_shared_types = &layer_types[..first_shared];

        layer_types
            .iter()
            .enumerate()
            .map(|(i, lt)| {
                if i < first_shared {
                    None
                } else {
                    non_shared_types
                        .iter()
                        .enumerate()
                        .rev()
                        .find(|(_, nst)| *nst == lt)
                        .map(|(idx, _)| idx)
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    SlidingAttention,
    FullAttention,
}

// ── Rotary Embedding ────────────────────────────────────────────────────

struct RotaryEmbedding {
    cos_table: Tensor,
    sin_table: Tensor,
}

impl RotaryEmbedding {
    fn new(theta: f64, dim: usize, max_pos: usize, device: &Device) -> Result<Self> {
        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv.as_slice(), device)?;

        let positions: Vec<f32> = (0..max_pos).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;
        let freqs = positions
            .unsqueeze(1)?
            .matmul(&inv_freq.unsqueeze(0)?)?;

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

fn apply_partial_rope(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rotated_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let full_dim = q.dim(D::Minus1)?;
    let pass_dim = full_dim - rotated_dim;

    let q_rot = q.narrow(D::Minus1, 0, rotated_dim)?;
    let q_pass = q.narrow(D::Minus1, rotated_dim, pass_dim)?;
    let k_rot = k.narrow(D::Minus1, 0, rotated_dim)?;
    let k_pass = k.narrow(D::Minus1, rotated_dim, pass_dim)?;

    let q_rot = rope(&q_rot.contiguous()?, cos, sin)?;
    let k_rot = rope(&k_rot.contiguous()?, cos, sin)?;

    Ok((
        Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?,
        Tensor::cat(&[&k_rot, &k_pass], D::Minus1)?,
    ))
}

// ── Attention ───────────────────────────────────────────────────────────

struct Attention {
    q_proj: LinearLayer,
    k_proj: Option<LinearLayer>,
    v_proj: Option<LinearLayer>,
    o_proj: LinearLayer,
    q_norm: RmsNorm,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    is_shared: bool,
    kv_shared_layer_index: Option<usize>,
    kv_cache: Option<(Tensor, Tensor)>,
    cache_seq_len: usize,
}

impl Attention {
    fn new(
        config: &Gemma4TextConfig,
        layer_type: LayerType,
        is_shared: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = match layer_type {
            LayerType::SlidingAttention => config.head_dim,
            LayerType::FullAttention => config.global_head_dim(),
        };
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;

        let q_proj = LinearLayer::Standard(linear_no_bias(
            config.hidden_size,
            num_heads * head_dim,
            vb.pp("q_proj"),
        )?);

        // QK norms: q_norm always, k_norm only for non-shared layers
        let q_norm = gemma_rms_norm(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?;

        let (k_proj, v_proj, k_norm) = if is_shared {
            (None, None, None)
        } else {
            let k = LinearLayer::Standard(linear_no_bias(
                config.hidden_size,
                num_kv_heads * head_dim,
                vb.pp("k_proj"),
            )?);
            let v = LinearLayer::Standard(linear_no_bias(
                config.hidden_size,
                num_kv_heads * head_dim,
                vb.pp("v_proj"),
            )?);
            let kn = gemma_rms_norm(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?;
            (Some(k), Some(v), Some(kn))
        };

        let o_proj = LinearLayer::Standard(linear_no_bias(
            num_heads * head_dim,
            config.hidden_size,
            vb.pp("o_proj"),
        )?);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            is_shared,
            kv_shared_layer_index: None,
            kv_cache: None,
            cache_seq_len: 0,
        })
    }

    fn new_from_gguf<R: Read + Seek>(
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        is_shared: bool,
        rms_norm_eps: f64,
        gg: &mut Gguf<R>,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let q_proj = gg.linear(&format!("{prefix}.attn_q.weight"))?;
        let q_norm = gemma_rms_norm_gguf(gg, &format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;

        let (k_proj, v_proj, k_norm) = if is_shared {
            (None, None, None)
        } else {
            let k = gg.linear(&format!("{prefix}.attn_k.weight"))?;
            let v = gg.linear(&format!("{prefix}.attn_v.weight"))?;
            let kn = gemma_rms_norm_gguf(gg, &format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;
            (Some(k), Some(v), Some(kn))
        };

        let o_proj = gg.linear(&format!("{prefix}.attn_output.weight"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            is_shared,
            kv_shared_layer_index: None,
            kv_cache: None,
            cache_seq_len: 0,
        })
    }

    fn update_kv_cache(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let new_seq_len = k.dim(2)?;
        let cache_seq_len = self.cache_seq_len;

        match self.kv_cache.take() {
            Some((buf_k, buf_v)) => {
                let buf_len = buf_k.dim(2)?;
                let new_total = cache_seq_len + new_seq_len;

                if new_total <= buf_len {
                    buf_k.slice_set(&k, 2, cache_seq_len)?;
                    buf_v.slice_set(&v, 2, cache_seq_len)?;
                    let k_view = buf_k.narrow(2, 0, new_total)?;
                    let v_view = buf_v.narrow(2, 0, new_total)?;
                    self.kv_cache = Some((buf_k, buf_v));
                    self.cache_seq_len = new_total;
                    Ok((k_view, v_view))
                } else {
                    let cur_k = buf_k.narrow(2, 0, cache_seq_len)?;
                    let cur_v = buf_v.narrow(2, 0, cache_seq_len)?;
                    drop(buf_k);
                    drop(buf_v);
                    let full_k = Tensor::cat(&[&cur_k, &k], 2)?;
                    let full_v = Tensor::cat(&[&cur_v, &v], 2)?;
                    drop(cur_k);
                    drop(cur_v);
                    let total = full_k.dim(2)?;
                    let room = 256;
                    let (b, h, _, d) = full_k.dims4()?;
                    let new_buf_k =
                        Tensor::zeros((b, h, total + room, d), k.dtype(), k.device())?;
                    let new_buf_v =
                        Tensor::zeros((b, h, total + room, d), v.dtype(), v.device())?;
                    new_buf_k.slice_set(&full_k, 2, 0)?;
                    new_buf_v.slice_set(&full_v, 2, 0)?;
                    self.kv_cache = Some((new_buf_k, new_buf_v));
                    self.cache_seq_len = total;
                    Ok((full_k, full_v))
                }
            }
            None => {
                let (b, h, s, d) = k.dims4()?;
                let room = 256;
                let buf_k = Tensor::zeros((b, h, s + room, d), k.dtype(), k.device())?;
                let buf_v = Tensor::zeros((b, h, s + room, d), v.dtype(), v.device())?;
                buf_k.slice_set(&k, 2, 0)?;
                buf_v.slice_set(&v, 2, 0)?;
                self.kv_cache = Some((buf_k, buf_v));
                self.cache_seq_len = s;
                Ok((k, v))
            }
        }
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        layer_type: LayerType,
        rotated_dim: Option<usize>,
        shared_kv: Option<&(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden_states.dims3()?;

        let q = self.q_proj.forward(hidden_states)?;
        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        // QK norm: q_norm applied before RoPE
        let q = self.q_norm.forward(&q)?;

        let (k, v) = if self.is_shared {
            let (sk, sv) = shared_kv.expect("shared layer must have shared KV state");
            (sk.clone(), sv.clone())
        } else {
            let k = self.k_proj.as_ref().unwrap().forward(hidden_states)?;
            let v = self.v_proj.as_ref().unwrap().forward(hidden_states)?;
            let k = k
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            // k_norm applied before RoPE
            let k = self.k_norm.as_ref().unwrap().forward(&k)?;
            (k, v)
        };

        // Apply RoPE to Q (always) and K (only for non-shared layers)
        let (q, k) = match layer_type {
            LayerType::SlidingAttention => {
                let q = rope(&q.contiguous()?, cos, sin)?;
                if self.is_shared {
                    (q, k)
                } else {
                    let k = rope(&k.contiguous()?, cos, sin)?;
                    (q, k)
                }
            }
            LayerType::FullAttention => {
                let rdim = rotated_dim.unwrap_or(self.head_dim);
                if self.is_shared {
                    let full_dim = q.dim(D::Minus1)?;
                    let pass_dim = full_dim - rdim;
                    let q_rot = q.narrow(D::Minus1, 0, rdim)?;
                    let q_pass = q.narrow(D::Minus1, rdim, pass_dim)?;
                    let q_rot = rope(&q_rot.contiguous()?, cos, sin)?;
                    let q = Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?;
                    (q, k)
                } else {
                    apply_partial_rope(&q, &k, cos, sin, rdim)?
                }
            }
        };

        // Update KV cache (non-shared layers only)
        let (k, v) = if self.is_shared {
            (k, v)
        } else {
            self.update_kv_cache(k, v)?
        };

        // ── SDPA ──
        let n_rep = self.num_heads / self.num_kv_heads;

        if n_rep > 1 && seq_len == 1 {
            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let q_g =
                (q.reshape((b_sz, self.num_kv_heads, n_rep, self.head_dim))? * scale)?;
            let k_t = k.transpose(2, 3)?;
            let attn_weights = q_g.matmul(&k_t)?;

            let attn_weights = match attention_mask {
                Some(mask) => attn_weights.broadcast_add(mask)?,
                None => attn_weights,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_output = attn_weights.matmul(&v)?;

            let attn_output = attn_output
                .reshape((b_sz, self.num_heads, self.head_dim))?
                .reshape((b_sz, 1, self.num_heads * self.head_dim))?;
            return self.o_proj.forward(&attn_output);
        }

        let k = if n_rep > 1 {
            let (b, kv_heads, s, d) = k.dims4()?;
            k.unsqueeze(2)?
                .expand((b, kv_heads, n_rep, s, d))?
                .reshape((b, kv_heads * n_rep, s, d))?
        } else {
            k
        };
        let v = if n_rep > 1 {
            let (b, kv_heads, s, d) = v.dims4()?;
            v.unsqueeze(2)?
                .expand((b, kv_heads, n_rep, s, d))?
                .reshape((b, kv_heads * n_rep, s, d))?
        } else {
            v
        };

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
        self.cache_seq_len = 0;
    }
}

// ── MLP ─────────────────────────────────────────────────────────────────

struct Mlp {
    gate_proj: LinearLayer,
    up_proj: LinearLayer,
    down_proj: LinearLayer,
}

impl Mlp {
    fn new(config: &Gemma4TextConfig, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = LinearLayer::Standard(linear_no_bias(
            config.hidden_size,
            intermediate_size,
            vb.pp("gate_proj"),
        )?);
        let up_proj = LinearLayer::Standard(linear_no_bias(
            config.hidden_size,
            intermediate_size,
            vb.pp("up_proj"),
        )?);
        let down_proj = LinearLayer::Standard(linear_no_bias(
            intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?);

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn new_from_gguf<R: Read + Seek>(gg: &mut Gguf<R>, layer_idx: usize) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let gate_proj = gg.linear(&format!("{prefix}.ffn_gate.weight"))?;
        let up_proj = gg.linear(&format!("{prefix}.ffn_up.weight"))?;
        let down_proj = gg.linear(&format!("{prefix}.ffn_down.weight"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let gate = candle_nn::Activation::GeluPytorchTanh.forward(&gate)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ── Decoder Layer ───────────────────────────────────────────────────────

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    // Gemma-style pre+post norm pairs (4 norms total)
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
    // Per-layer scalar
    layer_scalar: Tensor,
    // PLE components (per-layer)
    per_layer_input_gate: LinearLayer,
    per_layer_projection: LinearLayer,
    post_per_layer_input_norm: RmsNorm,
    layer_type: LayerType,
}

impl DecoderLayer {
    fn new(
        config: &Gemma4TextConfig,
        layer_type: LayerType,
        is_shared: bool,
        intermediate_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(config, layer_type, is_shared, vb.pp("self_attn"))?;
        let mlp = Mlp::new(config, intermediate_size, vb.pp("mlp"))?;

        let input_layernorm = gemma_rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = gemma_rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let pre_feedforward_layernorm = gemma_rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = gemma_rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;

        // Layer scalar: loaded from checkpoint, shape [1]
        let layer_scalar = vb
            .get(1, "layer_scalar")
            .unwrap_or_else(|_| Tensor::ones(1, DType::F32, vb.device()).unwrap());

        let ple_dim = config.hidden_size_per_layer_input.unwrap_or(256);

        let per_layer_input_gate = LinearLayer::Standard(linear_no_bias(
            config.hidden_size,
            ple_dim,
            vb.pp("per_layer_input_gate"),
        )?);
        let per_layer_projection = LinearLayer::Standard(linear_no_bias(
            ple_dim,
            config.hidden_size,
            vb.pp("per_layer_projection"),
        )?);
        let post_per_layer_input_norm =
            gemma_rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("post_per_layer_input_norm"))?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            layer_scalar,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            layer_type,
        })
    }

    fn new_from_gguf<R: Read + Seek>(
        config: &Gemma4TextConfig,
        layer_type: LayerType,
        is_shared: bool,
        head_dim: usize,
        gg: &mut Gguf<R>,
        layer_idx: usize,
    ) -> Result<Self> {
        let self_attn = Attention::new_from_gguf(
            config.num_attention_heads,
            config.num_key_value_heads,
            head_dim,
            is_shared,
            config.rms_norm_eps,
            gg,
            layer_idx,
        )?;
        let mlp = Mlp::new_from_gguf(gg, layer_idx)?;
        let prefix = format!("blk.{layer_idx}");

        let input_layernorm =
            gemma_rms_norm_gguf(gg, &format!("{prefix}.attn_norm.weight"), config.rms_norm_eps)?;
        let post_attention_layernorm =
            gemma_rms_norm_gguf(gg, &format!("{prefix}.attn_post_norm.weight"), config.rms_norm_eps)?;
        let pre_feedforward_layernorm =
            gemma_rms_norm_gguf(gg, &format!("{prefix}.ffn_norm.weight"), config.rms_norm_eps)?;
        let post_feedforward_layernorm =
            gemma_rms_norm_gguf(gg, &format!("{prefix}.ffn_post_norm.weight"), config.rms_norm_eps)?;

        // Layer scalar: GGUF stores it as a tensor
        let layer_scalar = match gg.tensor(&format!("{prefix}.layer_output_scale")) {
            Ok(qt) => qt.dequantize(&Device::Cpu).unwrap_or_else(|_| {
                Tensor::ones(1, DType::F32, &Device::Cpu).unwrap()
            }),
            Err(_) => Tensor::ones(1, DType::F32, &Device::Cpu).unwrap(),
        };

        // PLE components
        let per_layer_input_gate = gg.linear(&format!("{prefix}.inp_gate.weight"))?;
        let per_layer_projection = gg.linear(&format!("{prefix}.proj.weight"))?;
        let post_per_layer_input_norm =
            gemma_rms_norm_gguf(gg, &format!("{prefix}.post_norm.weight"), config.rms_norm_eps)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            layer_scalar,
            per_layer_input_gate,
            per_layer_projection,
            post_per_layer_input_norm,
            layer_type,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        per_layer_input: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        rotated_dim: Option<usize>,
        shared_kv: Option<&(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        // Pre-norm → attention → post-norm → residual
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(
            &hidden_states,
            cos,
            sin,
            attention_mask,
            self.layer_type,
            rotated_dim,
            shared_kv,
        )?;
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = (residual + hidden_states)?;

        // Pre-norm → MLP → post-norm → residual
        let residual = &hidden_states;
        let hidden_states = self.pre_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        let hidden_states = self.post_feedforward_layernorm.forward(&hidden_states)?;
        let hidden_states = (residual + hidden_states)?;

        // PLE: gated per-layer embedding → residual
        let hidden_states = self.apply_ple(&hidden_states, per_layer_input)?;

        // Layer scalar
        let hidden_states = hidden_states.broadcast_mul(
            &self.layer_scalar.to_dtype(hidden_states.dtype())?,
        )?;

        Ok(hidden_states)
    }

    fn apply_ple(&self, hidden_states: &Tensor, per_layer_input: &Tensor) -> Result<Tensor> {
        let gate = self.per_layer_input_gate.forward(hidden_states)?;
        let gate = candle_nn::Activation::GeluPytorchTanh.forward(&gate)?;
        let gated = (gate * per_layer_input)?;
        let projected = self.per_layer_projection.forward(&gated)?;
        let normed = self.post_per_layer_input_norm.forward(&projected)?;
        hidden_states + normed
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ── Full Model ──────────────────────────────────────────────────────────

pub struct Gemma4Model {
    embed_tokens: Embedding,
    embed_tokens_per_layer: Embedding,
    per_layer_model_projection: LinearLayer,
    per_layer_projection_norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: LinearLayer,
    rotary_sliding: RotaryEmbedding,
    rotary_full: RotaryEmbedding,
    layer_types: Vec<LayerType>,
    kv_sharing_map: Vec<Option<usize>>,
    config: Gemma4TextConfig,
    dtype: DType,
    embed_scale: f64,
    ple_embed_scale: f64,
    ple_projection_scale: f64,
    ple_input_scale: f64,
    sliding_window: usize,
    full_rotated_dim: usize,
}

impl Gemma4Model {
    /// Construct from safetensors / HuggingFace checkpoint.
    /// `is_multimodal` controls whether tensors are under `model.language_model.` or `model.`.
    pub fn new(config: &Gemma4TextConfig, vb: VarBuilder, is_multimodal: bool) -> Result<Self> {
        let dtype = vb.dtype();
        let model_vb = if is_multimodal {
            vb.pp("model").pp("language_model")
        } else {
            vb.pp("model")
        };

        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            model_vb.pp("embed_tokens"),
        )?;

        let ple_vocab = config.vocab_size_per_layer_input.unwrap_or(config.vocab_size);
        let ple_dim = config.hidden_size_per_layer_input.unwrap_or(256);
        let ple_total_dim = config.num_hidden_layers * ple_dim;
        let embed_tokens_per_layer = candle_nn::embedding(
            ple_vocab,
            ple_total_dim,
            model_vb.pp("embed_tokens_per_layer"),
        )?;

        // Model-level PLE projection
        let per_layer_model_projection = LinearLayer::Standard(linear_no_bias(
            config.hidden_size,
            ple_total_dim,
            model_vb.pp("per_layer_model_projection"),
        )?);
        let per_layer_projection_norm = gemma_rms_norm(
            ple_dim,
            config.rms_norm_eps,
            model_vb.pp("per_layer_projection_norm"),
        )?;

        let layer_types = config.parsed_layer_types();
        let kv_sharing_map = config.kv_sharing_map();
        let first_shared = config.first_kv_shared_layer();

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let layers_vb = model_vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            let lt = layer_types[i];
            let is_shared = i >= first_shared;
            // Shared layers use double-wide MLP when use_double_wide_mlp is set
            let intermediate_size = if is_shared && config.use_double_wide_mlp {
                config.intermediate_size * 2
            } else {
                config.intermediate_size
            };
            layers.push(DecoderLayer::new(config, lt, is_shared, intermediate_size, layers_vb.pp(i))?);
            if let Some(source) = kv_sharing_map[i] {
                layers[i].self_attn.kv_shared_layer_index = Some(source);
            }
        }

        let norm = gemma_rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            model_vb.pp("norm"),
        )?;

        let lm_head = if config.tie_word_embeddings {
            LinearLayer::Standard(Linear::new(embed_tokens.embeddings().clone(), None))
        } else {
            LinearLayer::Standard(linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                vb.pp("lm_head"),
            )?)
        };

        let (rotary_sliding, rotary_full, full_rotated_dim) =
            Self::build_rotary_embeddings(config, vb.device())?;

        let embed_scale = (config.hidden_size as f64).sqrt();
        let ple_embed_scale = (ple_dim as f64).sqrt();
        let ple_projection_scale = (config.hidden_size as f64).powf(-0.5);
        let ple_input_scale = 2.0_f64.powf(-0.5);
        let sliding_window = config.sliding_window.unwrap_or(512);

        Ok(Self {
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm,
            lm_head,
            rotary_sliding,
            rotary_full,
            layer_types,
            kv_sharing_map,
            config: config.clone(),
            dtype,
            embed_scale,
            ple_embed_scale,
            ple_projection_scale,
            ple_input_scale,
            sliding_window,
            full_rotated_dim,
        })
    }

    /// Construct from a GGUF file.
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let mut gg = Gguf::new(ct, reader, device.clone(), dtype);
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle_core::bail!("cannot find {s} in GGUF metadata"),
            Some(v) => Ok(v.clone()),
        };

        let arch = gg
            .metadata()
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.clone())
            .unwrap_or_else(|| "gemma4".to_string());

        let num_attention_heads =
            md_get(&format!("{arch}.attention.head_count"))?.to_u32()? as usize;
        let num_kv_heads =
            md_get(&format!("{arch}.attention.head_count_kv"))?.to_u32()? as usize;
        let num_hidden_layers = md_get(&format!("{arch}.block_count"))?.to_u32()? as usize;
        let hidden_size = md_get(&format!("{arch}.embedding_length"))?.to_u32()? as usize;
        let intermediate_size =
            md_get(&format!("{arch}.feed_forward_length"))?.to_u32()? as usize;
        let max_position_embeddings = gg
            .metadata()
            .get(&format!("{arch}.context_length"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(131072) as usize;
        let rms_norm_eps = gg
            .metadata()
            .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1e-6) as f64;
        let rope_theta = gg
            .metadata()
            .get(&format!("{arch}.rope.freq_base"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1_000_000.0) as f64;

        let global_head_dim = gg
            .metadata()
            .get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(512) as usize;
        let swa_head_dim = gg
            .metadata()
            .get(&format!("{arch}.attention.key_length_swa"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(256) as usize;

        let num_kv_shared_layers = gg
            .metadata()
            .get(&format!("{arch}.attention.shared_kv_layers"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(0) as usize;

        let ple_dim = gg
            .metadata()
            .get(&format!("{arch}.embedding_length_per_layer_input"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(256) as usize;

        let sliding_window = gg
            .metadata()
            .get(&format!("{arch}.attention.sliding_window"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(512) as usize;

        let final_logit_softcapping = gg
            .metadata()
            .get(&format!("{arch}.final_logit_softcapping"))
            .and_then(|v| v.to_f32().ok());

        let layer_types_str: Vec<String> = if let Some(gguf_file::Value::Array(arr)) = gg
            .metadata()
            .get(&format!("{arch}.attention.sliding_window_pattern"))
        {
            arr.iter()
                .map(|v| {
                    let is_swa = match v {
                        gguf_file::Value::Bool(b) => *b,
                        gguf_file::Value::U8(n) => *n != 0,
                        gguf_file::Value::I8(n) => *n != 0,
                        _ => true,
                    };
                    if is_swa {
                        "sliding_attention".to_string()
                    } else {
                        "full_attention".to_string()
                    }
                })
                .collect()
        } else {
            (0..num_hidden_layers)
                .map(|i| {
                    if i % 5 == 4 {
                        "full_attention".to_string()
                    } else {
                        "sliding_attention".to_string()
                    }
                })
                .collect()
        };

        let tie_word_embeddings = !gg.ct.tensor_infos.contains_key("output.weight");

        let config = Gemma4TextConfig {
            vocab_size: 0,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: num_kv_heads,
            head_dim: swa_head_dim,
            global_head_dim: Some(global_head_dim),
            max_position_embeddings,
            rms_norm_eps,
            hidden_activation: Some("gelu_pytorch_tanh".to_string()),
            use_double_wide_mlp: false,
            tie_word_embeddings,
            sliding_window: Some(sliding_window),
            final_logit_softcapping,
            hidden_size_per_layer_input: Some(ple_dim),
            vocab_size_per_layer_input: None,
            num_kv_shared_layers: Some(num_kv_shared_layers),
            layer_types: layer_types_str,
            rope_parameters: None,
            eos_token_id: None,
        };

        let embed_tokens = gg.embedding("token_embd.weight", hidden_size)?;
        let actual_vocab_size = embed_tokens.embeddings().dim(0)?;

        let ple_total_dim = num_hidden_layers * ple_dim;
        let embed_tokens_per_layer = gg.embedding("per_layer_token_embd.weight", ple_total_dim)?;
        let ple_vocab = embed_tokens_per_layer.embeddings().dim(0)?;

        // Model-level PLE projection
        let per_layer_model_projection = gg.linear("per_layer_model_proj.weight")?;
        let per_layer_projection_norm =
            gemma_rms_norm_gguf(&mut gg, "per_layer_proj_norm.weight", rms_norm_eps)?;

        let config = Gemma4TextConfig {
            vocab_size: actual_vocab_size,
            vocab_size_per_layer_input: Some(ple_vocab),
            ..config
        };

        let layer_types = config.parsed_layer_types();
        let kv_sharing_map = config.kv_sharing_map();
        let first_shared = config.first_kv_shared_layer();

        let mut layers = Vec::with_capacity(num_hidden_layers);
        for i in 0..num_hidden_layers {
            let lt = layer_types[i];
            let is_shared = i >= first_shared;
            let head_dim = match lt {
                LayerType::SlidingAttention => swa_head_dim,
                LayerType::FullAttention => global_head_dim,
            };
            layers.push(DecoderLayer::new_from_gguf(
                &config, lt, is_shared, head_dim, &mut gg, i,
            )?);
            if let Some(source) = kv_sharing_map[i] {
                layers[i].self_attn.kv_shared_layer_index = Some(source);
            }
        }

        let norm = gemma_rms_norm_gguf(&mut gg, "output_norm.weight", rms_norm_eps)?;

        let lm_head = if tie_word_embeddings {
            LinearLayer::Standard(Linear::new(embed_tokens.embeddings().clone(), None))
        } else {
            gg.linear("output.weight")?
        };

        let sliding_theta = 10_000.0;
        let full_theta = rope_theta;

        let full_rotated_dim = gg
            .metadata()
            .get(&format!("{arch}.rope.dimension_count"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or((global_head_dim / 4) as u32) as usize;

        let rotary_sliding = RotaryEmbedding::new(
            sliding_theta,
            swa_head_dim,
            max_position_embeddings,
            device,
        )?;
        let rotary_full = RotaryEmbedding::new(
            full_theta,
            full_rotated_dim,
            max_position_embeddings,
            device,
        )?;

        let embed_scale = (hidden_size as f64).sqrt();
        let ple_embed_scale = (ple_dim as f64).sqrt();
        let ple_projection_scale = (hidden_size as f64).powf(-0.5);
        let ple_input_scale = 2.0_f64.powf(-0.5);

        Ok(Self {
            embed_tokens,
            embed_tokens_per_layer,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm,
            lm_head,
            rotary_sliding,
            rotary_full,
            layer_types,
            kv_sharing_map,
            config,
            dtype,
            embed_scale,
            ple_embed_scale,
            ple_projection_scale,
            ple_input_scale,
            sliding_window,
            full_rotated_dim,
        })
    }

    fn build_rotary_embeddings(
        config: &Gemma4TextConfig,
        device: &Device,
    ) -> Result<(RotaryEmbedding, RotaryEmbedding, usize)> {
        let sliding_theta = config
            .rope_parameters
            .as_ref()
            .map(|rp| rp.sliding_attention.rope_theta)
            .unwrap_or(10_000.0);
        let full_theta = config
            .rope_parameters
            .as_ref()
            .map(|rp| rp.full_attention.rope_theta)
            .unwrap_or(1_000_000.0);
        let partial_factor = config
            .rope_parameters
            .as_ref()
            .and_then(|rp| rp.full_attention.partial_rotary_factor)
            .unwrap_or(0.25);

        let global_hd = config.global_head_dim();
        let full_rotated_dim = (global_hd as f64 * partial_factor) as usize;

        let rotary_sliding = RotaryEmbedding::new(
            sliding_theta,
            config.head_dim,
            config.max_position_embeddings,
            device,
        )?;
        let rotary_full = RotaryEmbedding::new(
            full_theta,
            full_rotated_dim,
            config.max_position_embeddings,
            device,
        )?;

        Ok((rotary_sliding, rotary_full, full_rotated_dim))
    }

    // ── Forward ─────────────────────────────────────────────────────────

    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;

        #[cfg(feature = "cuda")]
        let _event_guard = EventTrackingGuard::disable(input_ids.device());

        // Embed + scale by sqrt(hidden_size)
        let hidden_states = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;
        let hidden_states = (hidden_states * self.embed_scale)?;

        // PLE preparation: combine token embeddings + model projection
        let ple_dim = self.config.hidden_size_per_layer_input.unwrap_or(256);
        let num_layers = self.config.num_hidden_layers;

        // 1. Token-based PLE: embed_tokens_per_layer(input_ids) * sqrt(ple_dim)
        let per_layer_token_embeds = self
            .embed_tokens_per_layer
            .forward(input_ids)?
            .to_dtype(self.dtype)?;
        let per_layer_token_embeds = (per_layer_token_embeds * self.ple_embed_scale)?;
        // Shape: [B, S, num_layers * ple_dim]

        // 2. Model projection: project hidden_states → per-layer dims
        let per_layer_projection = self
            .per_layer_model_projection
            .forward(&hidden_states)?;
        let per_layer_projection = (per_layer_projection * self.ple_projection_scale)?;
        // Shape: [B, S, num_layers * ple_dim]

        // Reshape to [B, S, num_layers, ple_dim] for norm
        let shape_prefix = per_layer_projection.dims()[..2].to_vec();
        let per_layer_projection = per_layer_projection
            .reshape((shape_prefix[0], shape_prefix[1], num_layers, ple_dim))?;
        let per_layer_projection = self.per_layer_projection_norm.forward(&per_layer_projection)?;
        // Flatten back to [B, S, num_layers * ple_dim]
        let per_layer_projection = per_layer_projection
            .reshape((shape_prefix[0], shape_prefix[1], num_layers * ple_dim))?;

        // 3. Combine: (projection + token_embeds) * 2^-0.5
        let per_layer_inputs =
            ((per_layer_projection + per_layer_token_embeds)? * self.ple_input_scale)?;

        // RoPE tables
        let total_len = start_pos + seq_len;

        let (full_cos_sliding, full_sin_sliding) = self.rotary_sliding.forward(total_len)?;
        let cos_sliding = full_cos_sliding
            .narrow(0, start_pos, seq_len)?
            .to_dtype(self.dtype)?;
        let sin_sliding = full_sin_sliding
            .narrow(0, start_pos, seq_len)?
            .to_dtype(self.dtype)?;

        let (full_cos_full, full_sin_full) = self.rotary_full.forward(total_len)?;
        let cos_full = full_cos_full
            .narrow(0, start_pos, seq_len)?
            .to_dtype(self.dtype)?;
        let sin_full = full_sin_full
            .narrow(0, start_pos, seq_len)?
            .to_dtype(self.dtype)?;

        // Attention masks
        let sliding_mask = if seq_len > 1 {
            Some(self.build_sliding_mask(seq_len, total_len, start_pos, input_ids.device())?)
        } else {
            None
        };
        let causal_mask = if seq_len > 1 {
            Some(self.build_causal_mask(seq_len, total_len, start_pos, input_ids.device())?)
        } else {
            None
        };

        // Forward pass through layers
        let mut hidden_states = hidden_states;
        let mut shared_kv_states: HashMap<usize, (Tensor, Tensor)> = HashMap::new();

        for i in 0..self.layers.len() {
            let lt = self.layer_types[i];

            let (cos, sin, mask) = match lt {
                LayerType::SlidingAttention => {
                    (&cos_sliding, &sin_sliding, sliding_mask.as_ref())
                }
                LayerType::FullAttention => (&cos_full, &sin_full, causal_mask.as_ref()),
            };

            let rotated_dim = match lt {
                LayerType::FullAttention => Some(self.full_rotated_dim),
                LayerType::SlidingAttention => None,
            };

            let shared_kv = self.kv_sharing_map[i].and_then(|src| shared_kv_states.get(&src));

            // PLE input for this layer: narrow from combined per_layer_inputs
            let ple_input = per_layer_inputs.narrow(D::Minus1, i * ple_dim, ple_dim)?;

            hidden_states = self.layers[i].forward(
                &hidden_states,
                &ple_input,
                cos,
                sin,
                mask,
                rotated_dim,
                shared_kv,
            )?;

            // Store KV state from non-shared layers
            if self.kv_sharing_map[i].is_none() {
                if let Some((ref buf_k, ref buf_v)) = self.layers[i].self_attn.kv_cache {
                    let len = self.layers[i].self_attn.cache_seq_len;
                    let k_view = buf_k.narrow(2, 0, len)?;
                    let v_view = buf_v.narrow(2, 0, len)?;
                    shared_kv_states.insert(i, (k_view, v_view));
                }
            }
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self
            .lm_head
            .forward(&hidden_states.narrow(1, seq_len - 1, 1)?)?;

        // Logit softcapping
        if let Some(cap) = self.config.final_logit_softcapping {
            let cap_t = Tensor::new(cap, logits.device())?.to_dtype(logits.dtype())?;
            let scaled = logits.broadcast_div(&cap_t)?;
            let capped = scaled.tanh()?.broadcast_mul(&cap_t)?;
            Ok(capped)
        } else {
            Ok(logits)
        }
    }

    fn build_sliding_mask(
        &self,
        seq_len: usize,
        total_len: usize,
        start_pos: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let window = self.sliding_window;
        let mut mask_data = vec![0f32; seq_len * total_len];
        for i in 0..seq_len {
            let pos_i = start_pos + i;
            for j in 0..total_len {
                let visible = j <= pos_i && j >= pos_i.saturating_sub(window - 1);
                if !visible {
                    mask_data[i * total_len + j] = -1e9;
                }
            }
        }
        let mask = Tensor::from_vec(mask_data, (seq_len, total_len), device)?
            .to_dtype(self.dtype)?;
        Ok(mask.unsqueeze(0)?.unsqueeze(0)?)
    }

    fn build_causal_mask(
        &self,
        seq_len: usize,
        total_len: usize,
        start_pos: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mut mask_data = vec![0f32; seq_len * total_len];
        for i in 0..seq_len {
            for j in 0..total_len {
                if j > start_pos + i {
                    mask_data[i * total_len + j] = -1e9;
                }
            }
        }
        let mask = Tensor::from_vec(mask_data, (seq_len, total_len), device)?
            .to_dtype(self.dtype)?;
        Ok(mask.unsqueeze(0)?.unsqueeze(0)?)
    }

    // ── KV Cache Management ─────────────────────────────────────────────

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn active_kv_cache_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| {
                l.self_attn
                    .kv_cache
                    .as_ref()
                    .map(|(k, v)| {
                        let k_bytes = k.elem_count() as u64 * k.dtype().size_in_bytes() as u64;
                        let v_bytes = v.elem_count() as u64 * v.dtype().size_in_bytes() as u64;
                        k_bytes + v_bytes
                    })
                    .unwrap_or(0)
            })
            .sum()
    }

    pub fn config(&self) -> &Gemma4TextConfig {
        &self.config
    }

    pub fn model_dtype(&self) -> DType {
        self.dtype
    }
}
