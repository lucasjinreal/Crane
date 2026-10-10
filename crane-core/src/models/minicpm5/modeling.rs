//! `MiniCPM5` (`OpenBMB`) dense decoder.
//!
//! Despite the "`MiniCPM`" name this checkpoint is architecturally a plain
//! `LlamaForCausalLM` (per `openbmb/MiniCPM5-1B`'s own `config.json` and
//! README: `"architectures": ["LlamaForCausalLM"]`) — GQA, `RoPE`, `SwiGLU`,
//! `RMSNorm`, no attention bias, no QK-norm, no cross-layer attention sharing.
//! Older `MiniCPM` releases (1–3) used bespoke `scale_emb`/`scale_depth`/
//! `dim_model_base` tricks; none of that applies here. This module is
//! structurally a trimmed copy of `crate::models::hunyuan_dense::modeling`
//! (same GQA/merged-QKV/GGUF-quantized `LinearLayer` pattern) with the
//! QK-norm and CLA (cross-layer attention sharing) branches removed, since
//! `MiniCPM5` has neither.
//!
//! Cross-checked against the MIT-licensed `AspadaX/tiny-llm` (a minimal
//! educational Rust MiniCPM5-1B implementation) for the overall
//! embed/RoPE/GQA/SwiGLU/RMSNorm shape; no code was copied from it — it has
//! no KV cache, quantization, or backend plumbing to port, only the
//! architecture description to confirm against.

use crate::models::modules::kv_cache::{KvCache, KvCacheKind};
use crate::models::modules::rotary::RotaryEmbedding;
use crate::quantized::gguf_file::Gguf;
use crate::quantized::gguf_metadata::GgufMetadata;
use candle_core::quantized::gguf_file;
use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::rotary_emb::rope;
use candle_nn::{Linear, RmsNorm, VarBuilder, linear_no_bias};
use serde::Deserialize;
use std::io::{Read, Seek};

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
    /// Merged QKV weight [`q_dim` + 2*`kv_dim`, `hidden_size`] — one gemv instead of 3.
    /// Only set for Standard (non-quantized) weights.
    qkv_proj: Option<Linear>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    kv_cache: KvCache,
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
        let qkv_proj =
            if let (LinearLayer::Standard(q), LinearLayer::Standard(k), LinearLayer::Standard(v)) =
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
            kv_cache: KvCache::new(KvCacheKind::from_env()),
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
            kv_cache: KvCache::new(KvCacheKind::from_env()),
        })
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

        let (k, v) = self.kv_cache.append(&k, &v)?;

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
            let q_g = (q.reshape((b_sz, self.num_kv_heads, n_rep, self.head_dim))? * scale)?;

            // K^T: [B, kv_heads, D, S] — just a view, no copy
            let k_t = k.transpose(2, 3)?;

            // scores: [B, kv_heads, n_rep, S]
            let attn_weights = q_g.matmul(&k_t)?;

            let attn_weights = match attention_mask {
                Some(mask) => {
                    // mask [B, 1, 1, S] broadcasts over kv_heads & n_rep
                    attn_weights.broadcast_add(mask)?
                },
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
        self.kv_cache.reset();
    }
}

/// Gate+up projection: either a merged [2*I, H] weight (Standard) or separate quantized projections.
enum MlpGateUp {
    /// Merged gate+up weight — one gemv instead of two. Standard (BF16/F16/F32) only.
    Merged {
        gate_up_proj: Linear,
        intermediate_size: usize,
    },
    /// Separate quantized gate and up projections (GGUF).
    Separate {
        gate_proj: LinearLayer,
        up_proj: LinearLayer,
    },
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
            gate_up: MlpGateUp::Separate { gate_proj, up_proj },
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.gate_up {
            MlpGateUp::Merged {
                gate_up_proj,
                intermediate_size,
            } => {
                let gu = gate_up_proj.forward(x)?; // [B, S, 2*intermediate_size]

                #[cfg(feature = "cuda")]
                {
                    if gu.device().is_cuda() {
                        let activated =
                            crate::ops::fused_silu_mul(&gu.contiguous()?, *intermediate_size)?;
                        return self.down_proj.forward(&activated);
                    }
                }

                // CPU fallback
                let gate = gu.narrow(D::Minus1, 0, *intermediate_size)?;
                let up = gu.narrow(D::Minus1, *intermediate_size, *intermediate_size)?;
                let gate = candle_nn::Activation::Silu.forward(&gate)?;
                self.down_proj.forward(&(gate * up)?)
            },
            MlpGateUp::Separate {
                gate_proj, up_proj, ..
            } => {
                let gate = gate_proj.forward(x)?;
                let gate = candle_nn::Activation::Silu.forward(&gate)?;
                let up = up_proj.forward(x)?;
                self.down_proj.forward(&(gate * up)?)
            },
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
        let hidden_states = self
            .self_attn
            .forward(&hidden_states, cos, sin, attention_mask)?;
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
    /// # Errors
    ///
    /// Returns an error if a required tensor is missing or has a mismatched
    /// shape under `vb`, or if building the rotary embeddings fails.
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
    /// all weights as quantized tensors (`QMatMul` for linear layers, dequantized
    /// for embeddings and norms).
    ///
    /// # Errors
    ///
    /// Returns an error if a required metadata key is missing or has an
    /// unexpected type (e.g. `attention.head_count`), or if a required
    /// tensor is missing or cannot be loaded/dequantized from `reader`.
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
        // Detect architecture prefix (e.g. "llama", "minicpm", "minicpm5") —
        // the standardized llama.cpp GGUF metadata/tensor-naming scheme used
        // below works regardless of which of these the converter picked.
        let md = GgufMetadata::new(gg.metadata());
        let arch = md
            .opt_string("general.architecture")
            .unwrap_or_else(|| "llama".to_string());
        let key = |k: &str| format!("{arch}.{k}");

        let num_attention_heads = md.usize(&key("attention.head_count"))?;
        let num_kv_heads = md.usize(&key("attention.head_count_kv"))?;
        let head_dim = md.opt_usize(&key("attention.key_length")).unwrap_or(128);
        let num_hidden_layers = md.usize(&key("block_count"))?;
        let hidden_size = md.usize(&key("embedding_length"))?;
        let intermediate_size = md.usize(&key("feed_forward_length"))?;
        let max_position_embeddings = md.opt_usize(&key("context_length")).unwrap_or(131_072);
        let rms_norm_eps = f64::from(
            md.opt_f32(&key("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-6),
        );
        let rope_theta = f64::from(md.opt_f32(&key("rope.freq_base")).unwrap_or(5_000_000.0));

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

    /// # Errors
    ///
    /// Returns an error if any tensor operation in the forward pass fails,
    /// e.g. due to a shape mismatch in `input_ids`.
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
                .to_dtype(DType::F32)?;
            let mask = (mask * (-1e9f64))?.to_dtype(self.dtype)?;
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

    /// Native maximum context length this checkpoint was trained/configured
    /// for — the precomputed rotary table's row count, since `Config` itself
    /// isn't retained after construction.
    pub fn max_position_embeddings(&self) -> usize {
        self.rotary_emb.max_pos()
    }
}
