//! MiniCPM5 (OpenBMB) dense decoder.
//!
//! Despite the "MiniCPM" name this checkpoint is architecturally a plain
//! `LlamaForCausalLM` (per `openbmb/MiniCPM5-1B`'s own `config.json` and
//! README: `"architectures": ["LlamaForCausalLM"]`) — GQA, RoPE, SwiGLU,
//! RMSNorm, no attention bias, no QK-norm, no cross-layer attention sharing.
//! Older MiniCPM releases (1–3) used bespoke `scale_emb`/`scale_depth`/
//! `dim_model_base` tricks; none of that applies here. This module is
//! structurally a trimmed copy of `crate::models::hunyuan_dense::modeling`
//! (same GQA/merged-QKV/GGUF-quantized `LinearLayer` pattern) with the
//! QK-norm and CLA (cross-layer attention sharing) branches removed, since
//! MiniCPM5 has neither.
//!
//! Cross-checked against the MIT-licensed `AspadaX/tiny-llm` (a minimal
//! educational Rust MiniCPM5-1B implementation) for the overall
//! embed/RoPE/GQA/SwiGLU/RMSNorm shape; no code was copied from it — it has
//! no KV cache, quantization, or backend plumbing to port, only the
//! architecture description to confirm against.

use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::rotary_emb::rope;
use candle_nn::{linear_no_bias, Linear, RmsNorm, VarBuilder};
use crate::models::modules::rotary::RotaryEmbedding;
use serde::Deserialize;
use std::io::{Read, Seek};
use std::sync::Arc;

// ── GGUF loading helper ──

/// Wraps a parsed GGUF file + reader for convenient tensor loading.
pub struct Gguf<R: Read + Seek> {
    pub ct: gguf_file::Content,
    reader: R,
    device: Device,
    /// Target compute dtype. Dequantized tensors (norms, embeddings) are
    /// cast to this dtype so they match the activations flowing through the
    /// model (e.g. BF16 on CUDA). Quantized linear layers (QMatMul) handle
    /// their own internal dtype and the `LinearLayer` wrapper casts their
    /// output to the input's dtype.
    dtype: DType,
}

impl<R: Read + Seek> Gguf<R> {
    pub fn new(ct: gguf_file::Content, reader: R, device: Device, dtype: DType) -> Self {
        Self {
            ct,
            reader,
            device,
            dtype,
        }
    }

    /// Load a quantized tensor and wrap as a LinearLayer (QMatMul).
    pub fn linear(&mut self, name: &str) -> Result<LinearLayer> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        let qmm = candle_core::quantized::QMatMul::from_arc(Arc::new(ws))?;
        Ok(LinearLayer::Quantized(qmm))
    }

    /// Load a tensor, dequantize, and create an RmsNorm.
    /// The weight is cast to the target `dtype` so it matches activations.
    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        let weight = ws.dequantize(&self.device)?.to_dtype(self.dtype)?;
        Ok(RmsNorm::new(weight, eps))
    }

    /// Load a tensor, dequantize, and create an Embedding.
    /// The weight is cast to the target `dtype` so lookups produce
    /// tensors in the expected compute precision.
    pub fn embedding(&mut self, name: &str, hidden_size: usize) -> Result<candle_nn::Embedding> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        let weight = ws.dequantize(&self.device)?.to_dtype(self.dtype)?;
        Ok(candle_nn::Embedding::new(weight, hidden_size))
    }

    /// Load a raw QTensor by name.
    pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }

    /// Whether the file contains a tensor with this exact name.
    pub fn contains_tensor(&self, name: &str) -> bool {
        self.ct.tensor_infos.contains_key(name)
    }

    /// Access GGUF metadata.
    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }
}

pub use crate::ops::linear::LinearLayer;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: Option<f64>,
    pub attention_bias: Option<bool>,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
}

fn default_true() -> bool {
    true
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn attention_bias(&self) -> bool {
        self.attention_bias.unwrap_or(false)
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_theta.unwrap_or(10000.0)
    }
}

// ── Event-tracking RAII guard ────────────────────────────────────────────
//
// Candle defaults to tracking per-tensor CudaEvents for multi-stream safety.
// Crane uses a single CUDA stream — those events are pure overhead.
// This guard disables event tracking on first use and leaves it disabled.

#[cfg(feature = "cuda")]
struct EventTrackingGuard;

#[cfg(feature = "cuda")]
impl EventTrackingGuard {
    fn disable(device: &candle_core::Device) -> Self {
        if let candle_core::Device::Cuda(dev) = device {
            if dev.is_event_tracking() {
                unsafe { dev.disable_event_tracking() };
            }
        }
        Self
    }
}

// RoPE is applied via candle's fused `rope()` kernel (1 CUDA launch per tensor)
// instead of manual rotate_half + broadcast_mul (~5 launches per tensor).
struct Attention {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    o_proj: LinearLayer,
    /// Merged QKV weight [q_dim + 2*kv_dim, hidden_size] — one gemv instead of 3.
    /// Only set for Standard (non-quantized) weights.
    qkv_proj: Option<Linear>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    /// Pre-allocated KV cache buffer. May be larger than `cache_seq_len` to
    /// allow in-place `slice_set` writes without reallocation.
    kv_cache: Option<(Tensor, Tensor)>,
    /// Number of valid (filled) positions in the KV cache buffer.
    cache_seq_len: usize,
}

impl Attention {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let bias = config.attention_bias();

        let q_proj = if bias {
            LinearLayer::Standard(candle_nn::linear(
                config.hidden_size,
                num_heads * head_dim,
                vb.pp("q_proj"),
            )?)
        } else {
            LinearLayer::Standard(linear_no_bias(
                config.hidden_size,
                num_heads * head_dim,
                vb.pp("q_proj"),
            )?)
        };
        let k_proj = if bias {
            LinearLayer::Standard(candle_nn::linear(
                config.hidden_size,
                num_kv_heads * head_dim,
                vb.pp("k_proj"),
            )?)
        } else {
            LinearLayer::Standard(linear_no_bias(
                config.hidden_size,
                num_kv_heads * head_dim,
                vb.pp("k_proj"),
            )?)
        };
        let v_proj = if bias {
            LinearLayer::Standard(candle_nn::linear(
                config.hidden_size,
                num_kv_heads * head_dim,
                vb.pp("v_proj"),
            )?)
        } else {
            LinearLayer::Standard(linear_no_bias(
                config.hidden_size,
                num_kv_heads * head_dim,
                vb.pp("v_proj"),
            )?)
        };
        let o_proj = if bias {
            LinearLayer::Standard(candle_nn::linear(
                num_heads * head_dim,
                config.hidden_size,
                vb.pp("o_proj"),
            )?)
        } else {
            LinearLayer::Standard(linear_no_bias(
                num_heads * head_dim,
                config.hidden_size,
                vb.pp("o_proj"),
            )?)
        };

        // Create merged QKV projection for Standard weights:
        // Concatenate [q_weight; k_weight; v_weight] along dim 0 so one gemv
        // replaces three.  `narrow` splits are zero-copy views.
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_proj = if let (LinearLayer::Standard(q), LinearLayer::Standard(k), LinearLayer::Standard(v)) =
            (&q_proj, &k_proj, &v_proj)
        {
            let qkv_w = Tensor::cat(&[q.weight(), k.weight(), v.weight()], 0)?;
            let qkv_b = match (q.bias(), k.bias(), v.bias()) {
                (Some(qb), Some(kb), Some(vb)) => Some(Tensor::cat(&[qb, kb, vb], 0)?),
                _ => None,
            };
            Some(Linear::new(qkv_w, qkv_b))
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            qkv_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            q_dim,
            kv_dim,
            kv_cache: None,
            cache_seq_len: 0,
        })
    }

    /// Construct from GGUF quantized weights.
    fn new_from_gguf<R: Read + Seek>(
        config: &Config,
        gg: &mut Gguf<R>,
        layer_idx: usize,
    ) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let prefix = format!("blk.{layer_idx}");

        let q_proj = gg.linear(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.linear(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.linear(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.linear(&format!("{prefix}.attn_output.weight"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            qkv_proj: None, // GGUF quantized — cannot merge
            num_heads,
            num_kv_heads,
            head_dim,
            q_dim: num_heads * head_dim,
            kv_dim: num_kv_heads * head_dim,
            kv_cache: None,
            cache_seq_len: 0,
        })
    }

    /// Update the pre-allocated KV cache with new K,V tensors.
    ///
    /// Uses `slice_set` for O(1) in-place writes when the buffer has room.
    /// Falls back to cat + reallocate when the buffer is full.
    /// Returns (k_full, v_full) views covering all valid cached data.
    fn update_kv_cache(&mut self, k: Tensor, v: Tensor) -> Result<(Tensor, Tensor)> {
        // slice_set requires contiguous tensors; K/V after transpose(1,2) are strided.
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let new_seq_len = k.dim(2)?;
        let cache_seq_len = self.cache_seq_len;

        match self.kv_cache.take() {
            Some((buf_k, buf_v)) => {
                let buf_len = buf_k.dim(2)?;
                let new_total = cache_seq_len + new_seq_len;

                if new_total <= buf_len {
                    // In-place write: O(new_seq_len) instead of O(cache_len)
                    buf_k.slice_set(&k, 2, cache_seq_len)?;
                    buf_v.slice_set(&v, 2, cache_seq_len)?;
                    let k_view = buf_k.narrow(2, 0, new_total)?;
                    let v_view = buf_v.narrow(2, 0, new_total)?;
                    self.kv_cache = Some((buf_k, buf_v));
                    self.cache_seq_len = new_total;
                    Ok((k_view, v_view))
                } else {
                    // Buffer too small: grow with extra room.
                    let cur_k = buf_k.narrow(2, 0, cache_seq_len)?;
                    let cur_v = buf_v.narrow(2, 0, cache_seq_len)?;
                    drop(buf_k);
                    drop(buf_v);
                    let full_k = Tensor::cat(&[&cur_k, &k], 2)?;
                    let full_v = Tensor::cat(&[&cur_v, &v], 2)?;
                    drop(cur_k);
                    drop(cur_v);
                    let total = full_k.dim(2)?;
                    let room = 256; // fixed small room — avoids 2x over-allocation
                    let (b, h, _, d) = full_k.dims4()?;
                    let new_buf_k = Tensor::zeros((b, h, total + room, d), k.dtype(), k.device())?;
                    let new_buf_v = Tensor::zeros((b, h, total + room, d), v.dtype(), v.device())?;
                    new_buf_k.slice_set(&full_k, 2, 0)?;
                    new_buf_v.slice_set(&full_v, 2, 0)?;
                    self.kv_cache = Some((new_buf_k, new_buf_v));
                    self.cache_seq_len = total;
                    Ok((full_k, full_v))
                }
            }
            None => {
                // First use: allocate buffer with extra room.
                let (b, h, s, d) = k.dims4()?;
                let room = 256; // fixed small room — avoids 2x over-allocation
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
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden_states.dims3()?;

        // ── QKV projection: merged (1 gemv) or separate (3 gemv) ──
        let (q, k, v) = if let Some(ref qkv_proj) = self.qkv_proj {
            let qkv = qkv_proj.forward(hidden_states)?; // [B, S, q_dim+2*kv_dim]
            let q = qkv.narrow(D::Minus1, 0, self.q_dim)?;
            let k = qkv.narrow(D::Minus1, self.q_dim, self.kv_dim)?;
            let v = qkv.narrow(D::Minus1, self.q_dim + self.kv_dim, self.kv_dim)?;
            (q, k, v)
        } else {
            let q = self.q_proj.forward(hidden_states)?;
            let k = self.k_proj.forward(hidden_states)?;
            let v = self.v_proj.forward(hidden_states)?;
            (q, k, v)
        };

        // Reshape: [B, S, num_heads * head_dim] -> [B, num_heads, S, head_dim]
        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings via fused kernel.
        let q = rope(&q.contiguous()?, cos, sin)?;
        let k = rope(&k.contiguous()?, cos, sin)?;

        let (k, v) = self.update_kv_cache(k, v)?;

        self.compute_attention(q, k, v, attention_mask, b_sz, seq_len)
    }

    fn compute_attention(
        &self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Option<&Tensor>,
        b_sz: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;

        if n_rep > 1 && seq_len == 1 {
            // ── GQA-grouped SDPA for decode (seq_len=1) ──
            // Keep 4D tensors throughout; candle matmul handles
            // non-contiguous K internally with a single flatten pass.
            let scale = 1.0 / (self.head_dim as f64).sqrt();

            // Q: [B, H, 1, D] → [B, kv_heads, n_rep, D], pre-scaled
            let q_g =
                (q.reshape((b_sz, self.num_kv_heads, n_rep, self.head_dim))? * scale)?;

            // K^T: [B, kv_heads, D, S] — just a view, no copy
            let k_t = k.transpose(2, 3)?;

            // scores: [B, kv_heads, n_rep, S]
            let attn_weights = q_g.matmul(&k_t)?;

            let attn_weights = match attention_mask {
                Some(mask) => {
                    // mask [B, 1, 1, S] broadcasts over kv_heads & n_rep
                    attn_weights.broadcast_add(mask)?
                }
                None => attn_weights,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

            // V: [B, kv_heads, S, D] — matmul handles non-contiguous
            let attn_output = attn_weights.matmul(&v)?; // [B, kv_heads, n_rep, D]

            // Reshape back: → [B, H, D] → [B, 1, H*D]
            let attn_output = attn_output
                .reshape((b_sz, self.num_heads, self.head_dim))?
                .reshape((b_sz, 1, self.num_heads * self.head_dim))?;
            return self.o_proj.forward(&attn_output);
        }

        // ── Standard SDPA for prefill or when n_rep == 1 ──
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

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // [B, num_heads, S, head_dim] -> [B, S, hidden_size]
        let attn_output =
            attn_output
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

/// Gate+up projection: either a merged [2*I, H] weight (Standard) or separate quantized projections.
enum MlpGateUp {
    /// Merged gate+up weight — one gemv instead of two. Standard (BF16/F16/F32) only.
    Merged { gate_up_proj: Linear, intermediate_size: usize },
    /// Separate quantized gate and up projections (GGUF).
    Separate { gate_proj: LinearLayer, up_proj: LinearLayer },
}

struct Mlp {
    gate_up: MlpGateUp,
    down_proj: LinearLayer,
}

impl Mlp {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?;
        let up_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?;
        let down_proj = LinearLayer::Standard(linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?);

        // Merge gate+up into a single weight, then drop the originals to save VRAM.
        let gate_up_w = Tensor::cat(&[gate_proj.weight(), up_proj.weight()], 0)?;
        // gate_proj and up_proj are dropped here — their VRAM is freed.
        let gate_up = MlpGateUp::Merged {
            gate_up_proj: Linear::new(gate_up_w, None),
            intermediate_size: config.intermediate_size,
        };

        Ok(Self { gate_up, down_proj })
    }

    fn new_from_gguf<R: Read + Seek>(gg: &mut Gguf<R>, layer_idx: usize) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let gate_proj = gg.linear(&format!("{prefix}.ffn_gate.weight"))?;
        let up_proj = gg.linear(&format!("{prefix}.ffn_up.weight"))?;
        let down_proj = gg.linear(&format!("{prefix}.ffn_down.weight"))?;
        Ok(Self {
            gate_up: MlpGateUp::Separate {
                gate_proj,
                up_proj,
            },
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.gate_up {
            MlpGateUp::Merged { gate_up_proj, intermediate_size } => {
                let gu = gate_up_proj.forward(x)?; // [B, S, 2*intermediate_size]

                #[cfg(feature = "cuda")]
                {
                    if gu.device().is_cuda() {
                        let activated = crate::ops::fused_silu_mul(
                            &gu.contiguous()?,
                            *intermediate_size,
                        )?;
                        return self.down_proj.forward(&activated);
                    }
                }

                // CPU fallback
                let gate = gu.narrow(D::Minus1, 0, *intermediate_size)?;
                let up = gu.narrow(D::Minus1, *intermediate_size, *intermediate_size)?;
                let gate = candle_nn::Activation::Silu.forward(&gate)?;
                self.down_proj.forward(&(gate * up)?)
            }
            MlpGateUp::Separate { gate_proj, up_proj, .. } => {
                let gate = gate_proj.forward(x)?;
                let gate = candle_nn::Activation::Silu.forward(&gate)?;
                let up = up_proj.forward(x)?;
                self.down_proj.forward(&(gate * up)?)
            }
        }
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(config, vb.pp("self_attn"))?;
        let mlp = Mlp::new(config, vb.pp("mlp"))?;
        let input_layernorm = candle_nn::rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = candle_nn::rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn new_from_gguf<R: Read + Seek>(
        config: &Config,
        gg: &mut Gguf<R>,
        layer_idx: usize,
    ) -> Result<Self> {
        let self_attn = Attention::new_from_gguf(config, gg, layer_idx)?;
        let mlp = Mlp::new_from_gguf(gg, layer_idx)?;
        let prefix = format!("blk.{layer_idx}");
        let input_layernorm =
            gg.rms_norm(&format!("{prefix}.attn_norm.weight"), config.rms_norm_eps)?;
        let post_attention_layernorm =
            gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), config.rms_norm_eps)?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(&hidden_states, cos, sin, attention_mask)?;
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

pub struct MiniCpm5Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: LinearLayer,
    rotary_emb: RotaryEmbedding,
    dtype: DType,
}

impl MiniCpm5Model {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let model_vb = vb.pp("model");
        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            model_vb.pp("embed_tokens"),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let layers_vb = model_vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(config, layers_vb.pp(i))?);
        }

        let norm =
            candle_nn::rms_norm(config.hidden_size, config.rms_norm_eps, model_vb.pp("norm"))?;

        let lm_head = if config.tie_word_embeddings && !vb.contains_tensor("lm_head.weight") {
            LinearLayer::Standard(Linear::new(embed_tokens.embeddings().clone(), None))
        } else {
            LinearLayer::Standard(linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                vb.pp("lm_head"),
            )?)
        };

        let dim = config.head_dim();
        let rotary_emb = RotaryEmbedding::new(
            dim,
            config.max_position_embeddings,
            config.rope_theta(),
            vb.device(),
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            dtype,
        })
    }

    /// Construct from a GGUF file. Reads config from GGUF metadata and loads
    /// all weights as quantized tensors (QMatMul for linear layers, dequantized
    /// for embeddings and norms).
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Determine compute dtype early so Gguf can dequantize to it.
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

        // Detect architecture prefix (e.g. "llama", "minicpm", "minicpm5") —
        // the standardized llama.cpp GGUF metadata/tensor-naming scheme used
        // below works regardless of which of these the converter picked.
        let arch = gg
            .metadata()
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.clone())
            .unwrap_or_else(|| "llama".to_string());

        let num_attention_heads =
            md_get(&format!("{arch}.attention.head_count"))?.to_u32()? as usize;
        let num_kv_heads = md_get(&format!("{arch}.attention.head_count_kv"))?.to_u32()? as usize;
        let head_dim = gg
            .metadata()
            .get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(128) as usize;
        let num_hidden_layers = md_get(&format!("{arch}.block_count"))?.to_u32()? as usize;
        let hidden_size = md_get(&format!("{arch}.embedding_length"))?.to_u32()? as usize;
        let intermediate_size = md_get(&format!("{arch}.feed_forward_length"))?.to_u32()? as usize;
        let max_position_embeddings = gg
            .metadata()
            .get(&format!("{arch}.context_length"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(131_072) as usize;
        let rms_norm_eps = gg
            .metadata()
            .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1e-6) as f64;
        let rope_theta = gg
            .metadata()
            .get(&format!("{arch}.rope.freq_base"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(5_000_000.0) as f64;

        // Check for tied embeddings
        let tie_word_embeddings = !gg.ct.tensor_infos.contains_key("output.weight");

        // Build config
        let config = Config {
            vocab_size: 0, // updated below
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: num_kv_heads,
            head_dim: Some(head_dim),
            hidden_act: "silu".to_string(),
            max_position_embeddings,
            rms_norm_eps,
            rope_theta: Some(rope_theta),
            attention_bias: Some(false),
            tie_word_embeddings,
        };

        // Load embedding
        let embed_tokens = gg.embedding("token_embd.weight", hidden_size)?;
        let actual_vocab_size = embed_tokens.embeddings().dim(0)?;

        // Update config with actual vocab size
        let config = Config {
            vocab_size: actual_vocab_size,
            ..config
        };

        // Build layers
        let mut layers = Vec::with_capacity(num_hidden_layers);
        for i in 0..num_hidden_layers {
            layers.push(DecoderLayer::new_from_gguf(&config, &mut gg, i)?);
        }

        // Final norm
        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;

        // LM head (may be tied to embeddings)
        let lm_head = if tie_word_embeddings {
            LinearLayer::Standard(Linear::new(embed_tokens.embeddings().clone(), None))
        } else {
            gg.linear("output.weight")?
        };

        let rotary_emb = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta(),
            device,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            dtype,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        // Disable per-tensor CUDA event tracking — Crane uses a single stream.
        #[cfg(feature = "cuda")]
        let _event_guard = EventTrackingGuard::disable(input_ids.device());

        let (_b_sz, seq_len) = input_ids.dims2()?;
        // Cast embedding output to the target dtype — the safetensors file may store
        // weights in BF16 which is unsupported for CPU matmul.
        let hidden_states = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;

        let (cos, sin) = self.rotary_emb.forward(start_pos, seq_len)?;
        let cos = cos.to_dtype(self.dtype)?;
        let sin = sin.to_dtype(self.dtype)?;

        // Build causal mask
        let attention_mask = if seq_len > 1 {
            let total_len = start_pos + seq_len;
            // Build [seq_len, total_len] mask: 1.0 where allowed, 0.0 where masked
            let mut mask_data = vec![0f32; seq_len * total_len];
            for i in 0..seq_len {
                // Each query position i can attend to all cached positions + positions 0..=i
                for j in 0..total_len {
                    if j <= start_pos + i {
                        mask_data[i * total_len + j] = 1.0;
                    }
                }
            }
            let mask = Tensor::from_vec(mask_data, (seq_len, total_len), input_ids.device())?;
            // Convert: 0.0 (masked) -> -1e9, 1.0 (attend) -> 0.0
            let mask = mask
                .broadcast_lt(&Tensor::new(0.5f32, input_ids.device())?)?
                .to_dtype(self.dtype)?;
            let mask = (mask * (-1e9f64))?;
            Some(mask.unsqueeze(0)?.unsqueeze(0)?) // [1, 1, seq_len, total_len]
        } else {
            None
        };

        let mut hidden_states = hidden_states;
        for layer in self.layers.iter_mut() {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, attention_mask.as_ref())?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self
            .lm_head
            .forward(&hidden_states.narrow(1, seq_len - 1, 1)?)?;
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
        // rotary_emb tables are static and reusable — do not clear.
    }
}
