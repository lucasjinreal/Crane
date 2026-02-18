use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::rotary_emb::rope;
use candle_nn::{linear_no_bias, Linear, RmsNorm, VarBuilder};
use serde::Deserialize;
use std::io::{Read, Seek};
use std::sync::Arc;

#[cfg(feature = "flash-attn")]
use candle_flash_attn::{flash_attn, flash_attn_varlen};

// ── GGUF loading helper ──

/// Wraps a parsed GGUF file + reader for convenient tensor loading.
pub struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
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

    /// Access GGUF metadata.
    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }
}

// ── Polymorphic linear layer ──

/// A linear layer that can be either a standard (f16/f32) Linear or a
/// quantized QMatMul. Both implement Module::forward identically from the
/// caller's perspective. This allows the same model code to serve both
/// safetensors and GGUF weights with zero duplication.
pub enum LinearLayer {
    Standard(Linear),
    Quantized(candle_core::quantized::QMatMul),
}

impl Module for LinearLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard(l) => l.forward(xs),
            Self::Quantized(q) => {
                // candle's QMatMul internally dequantizes weights to F32
                // and requires F32 input. When the activation pipeline
                // runs in BF16/F16 we must:
                //   1. Cast input to F32 for the quantized matmul
                //   2. Cast output back to the original dtype for
                //      downstream binary ops (residual adds, etc.)
                let input_dtype = xs.dtype();
                let xs_f32 = if input_dtype != DType::F32 {
                    xs.to_dtype(DType::F32)?
                } else {
                    xs.clone()
                };
                let out = q.forward(&xs_f32)?;
                if input_dtype != DType::F32 {
                    out.to_dtype(input_dtype)
                } else {
                    Ok(out)
                }
            }
        }
    }
}

/// Pre-computed metadata for flash_attn_varlen in batched decode.
///
/// Built once per decode round in `step_batch_decode` and shared across
/// all 28 transformer layers, eliminating 84 redundant CPU→GPU tensor
/// transfers per round (3 per layer × 28 layers → just 3 total).
#[allow(dead_code)]
pub(crate) struct VarlenContext {
    pub seqlens_q: Tensor,
    pub seqlens_k: Tensor,
    pub gather_index: Tensor,
    pub max_seqlen_k: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub alpha: Option<f64>,
    pub beta_fast: Option<f64>,
    pub beta_slow: Option<f64>,
    pub factor: Option<f64>,
    pub mscale: Option<f64>,
    pub mscale_all_dim: Option<f64>,
    #[serde(rename = "type")]
    pub rope_type: Option<String>,
}

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
    pub rope_scaling: Option<RopeScaling>,
    pub attention_bias: Option<bool>,
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    pub use_cla: Option<bool>,
    pub cla_share_factor: Option<usize>,
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

// RoPE is applied via candle's fused `rope()` kernel (1 CUDA launch per tensor)
// instead of manual rotate_half + broadcast_mul (~5 launches per tensor).

struct RotaryEmbedding {
    inv_freq: Tensor,
    cos_cache: Option<Tensor>,
    sin_cache: Option<Tensor>,
    cached_len: usize,
}

impl RotaryEmbedding {
    fn new(config: &Config, device: &Device) -> Result<Self> {
        let dim = config.head_dim();
        let rope_theta = config.rope_theta();

        let inv_freq = if let Some(ref scaling) = config.rope_scaling {
            if let Some(alpha) = scaling.alpha {
                if alpha > 0.0 {
                    let base = rope_theta * alpha.powf(dim as f64 / (dim as f64 - 2.0));
                    let inv: Vec<f32> = (0..dim)
                        .step_by(2)
                        .map(|i| 1.0 / base.powf(i as f64 / dim as f64) as f32)
                        .collect();
                    Tensor::new(inv.as_slice(), device)?
                } else {
                    Self::default_inv_freq(dim, rope_theta, device)?
                }
            } else {
                Self::default_inv_freq(dim, rope_theta, device)?
            }
        } else {
            Self::default_inv_freq(dim, rope_theta, device)?
        };

        Ok(Self {
            inv_freq,
            cos_cache: None,
            sin_cache: None,
            cached_len: 0,
        })
    }

    fn default_inv_freq(dim: usize, base: f64, device: &Device) -> Result<Tensor> {
        let inv: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f64 / dim as f64) as f32)
            .collect();
        Tensor::new(inv.as_slice(), device)
    }

    fn forward(&mut self, seq_len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        if seq_len <= self.cached_len {
            if let (Some(ref cos), Some(ref sin)) = (&self.cos_cache, &self.sin_cache) {
                return Ok((cos.narrow(0, 0, seq_len)?, sin.narrow(0, 0, seq_len)?));
            }
        }

        let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;
        // positions: [seq_len], inv_freq: [dim/2]
        let freqs = positions
            .unsqueeze(1)?
            .matmul(&self.inv_freq.unsqueeze(0)?)?; // [seq_len, dim/2]
                                                    // Return [seq_len, dim/2] — candle's fused rope() expects half-dim cos/sin
                                                    // and handles the rotation internally.
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        self.cos_cache = Some(cos.clone());
        self.sin_cache = Some(sin.clone());
        self.cached_len = seq_len;

        Ok((cos, sin))
    }
}

struct Attention {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    o_proj: LinearLayer,
    query_layernorm: Option<RmsNorm>,
    key_layernorm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// Pre-allocated KV cache buffer. May be larger than `cache_seq_len` to
    /// allow in-place `slice_set` writes without reallocation.
    kv_cache: Option<(Tensor, Tensor)>,
    /// Number of valid (filled) positions in the KV cache buffer.
    cache_seq_len: usize,
    /// When true, the KV cache covers the full pre-allocated buffer and
    /// attention operates on the full width (masked). Used for CUDA-graph
    /// compatible decode where tensor shapes must be fixed.
    full_buffer_attn: bool,
    #[cfg(feature = "flash-attn")]
    use_flash_attn: bool,
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

        let (query_layernorm, key_layernorm) = if config.use_qk_norm {
            (
                Some(candle_nn::rms_norm(
                    head_dim,
                    config.rms_norm_eps,
                    vb.pp("query_layernorm"),
                )?),
                Some(candle_nn::rms_norm(
                    head_dim,
                    config.rms_norm_eps,
                    vb.pp("key_layernorm"),
                )?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            query_layernorm,
            key_layernorm,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_cache: None,
            cache_seq_len: 0,
            full_buffer_attn: false,
            #[cfg(feature = "flash-attn")]
            use_flash_attn: true,
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

        let (query_layernorm, key_layernorm) = if config.use_qk_norm {
            (
                Some(gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), config.rms_norm_eps)?),
                Some(gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), config.rms_norm_eps)?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            query_layernorm,
            key_layernorm,
            num_heads,
            num_kv_heads,
            head_dim,
            kv_cache: None,
            cache_seq_len: 0,
            full_buffer_attn: false,
            #[cfg(feature = "flash-attn")]
            use_flash_attn: true,
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
                    let room = total.max(256);
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
                let room = s.max(256);
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

    /// Scatter-based KV cache write using a position tensor.
    ///
    /// Writes new K/V at positions specified by `write_pos` using `scatter_set`.
    /// Returns the **full** pre-allocated buffer (not narrowed), suitable for
    /// attention with a mask. This makes the output shape fixed and compatible
    /// with CUDA graph capture.
    ///
    /// # Arguments
    /// - `k`, `v`: new data `[B, H, 1, D]`
    /// - `write_pos`: `[B]` tensor of u32 write positions
    fn update_kv_cache_scatter(
        &mut self,
        k: Tensor,
        v: Tensor,
        write_pos: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let (b, h, _one, d) = k.dims4()?;

        match self.kv_cache {
            Some((ref buf_k, ref buf_v)) => {
                // Build scatter index: expand [B] → [B, H, 1, D]
                let idx = write_pos
                    .reshape((b, 1, 1, 1))?
                    .expand((b, h, 1, d))?
                    .contiguous()?;
                buf_k.scatter_set(&idx, &k, 2)?;
                buf_v.scatter_set(&idx, &v, 2)?;
                // Return full buffer — caller uses mask for invalid positions.
                Ok((buf_k.clone(), buf_v.clone()))
            }
            None => {
                candle_core::bail!(
                    "update_kv_cache_scatter called without pre-allocated buffer; \
                     call setup_batch_decode first"
                )
            }
        }
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        write_pos: Option<&Tensor>,
        varlen_ctx: Option<&VarlenContext>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden_states.dims3()?;
        let _ = &varlen_ctx; // suppress unused warning when flash-attn is off

        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

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
        // rope() expects xs [B, H, S, D] contiguous, cos/sin [S, D/2] or [B, S, D/2].
        let q = rope(&q.contiguous()?, cos, sin)?;
        let k = rope(&k.contiguous()?, cos, sin)?;

        // Apply QK norm after RoPE
        let q = if let Some(ref norm) = self.query_layernorm {
            norm.forward(&q)?
        } else {
            q
        };
        let k = if let Some(ref norm) = self.key_layernorm {
            norm.forward(&k)?
        } else {
            k
        };

        // KV cache update — scatter-based (graph-compatible) or slice_set.
        let (k, v) = if let Some(wp) = write_pos {
            self.update_kv_cache_scatter(k, v, wp)?
        } else {
            self.update_kv_cache(k, v)?
        };

        // ── Attention computation ──
        // Try flash attention first (eliminates repeat_kv + fuses SDPA into 1 kernel).
        // flash_attn handles GQA natively (num_heads_q != num_heads_kv).
        // Fall back to manual SDPA when a custom attention mask is needed
        // (e.g., batched decode with padding) since flash_attn only supports causal masking.
        // When write_pos is Some (graph-capture mode), K/V are the full pre-allocated
        // buffer — flash_attn cannot be used (it assumes valid-length data), so we
        // always fall through to mask-based SDPA.
        #[cfg(feature = "flash-attn")]
        {
            if self.use_flash_attn && write_pos.is_none() {
                let can_use_flash = attention_mask.is_none() || seq_len > 1;
                if can_use_flash {
                    let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
                    let causal = seq_len > 1;
                    // flash_attn expects layout (B, S, H, D)
                    let q_fa = q.transpose(1, 2)?.contiguous()?;
                    let k_fa = k.transpose(1, 2)?.contiguous()?;
                    let v_fa = v.transpose(1, 2)?.contiguous()?;
                    let attn_output = flash_attn(&q_fa, &k_fa, &v_fa, softmax_scale, causal)?;
                    let attn_output = attn_output.reshape((b_sz, seq_len, ()))?;
                    return self.o_proj.forward(&attn_output);
                }

                // ── flash_attn_varlen for batched decode with heterogeneous KV lengths ──
                // KV cache is RIGHT-ALIGNED: [padding | prefill_data | decode_data].
                // Valid data per sequence is contiguous. We use the pre-computed
                // VarlenContext (built once per round in step_batch_decode) to
                // avoid redundant CPU→GPU transfers across the 28 layers.
                if seq_len == 1 {
                    if let Some(ctx) = varlen_ctx {
                        let total_s = k.dim(2)?;

                        // Q: [B, num_heads, 1, D] → [B, num_heads, D] (total_q = B)
                        let q_packed = q.squeeze(2)?.contiguous()?;

                        // K/V: [B, kv_heads, S, D] → [total_kv, kv_heads, D] via pre-computed index.
                        let k_perm = k.transpose(1, 2)?.contiguous()?;
                        let v_perm = v.transpose(1, 2)?.contiguous()?;
                        let k_flat =
                            k_perm.reshape((b_sz * total_s, self.num_kv_heads, self.head_dim))?;
                        let v_flat =
                            v_perm.reshape((b_sz * total_s, self.num_kv_heads, self.head_dim))?;
                        let k_packed = k_flat.index_select(&ctx.gather_index, 0)?;
                        let v_packed = v_flat.index_select(&ctx.gather_index, 0)?;

                        let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();

                        let attn_output = flash_attn_varlen(
                            &q_packed,
                            &k_packed,
                            &v_packed,
                            &ctx.seqlens_q,
                            &ctx.seqlens_k,
                            1,
                            ctx.max_seqlen_k,
                            softmax_scale,
                            false,
                        )?;
                        // [B, num_heads, D] → [B, 1, hidden_size]
                        let attn_output =
                            attn_output.reshape((b_sz, 1, self.num_heads * self.head_dim))?;
                        return self.o_proj.forward(&attn_output);
                    }
                }
            }
        }

        // ── Manual SDPA fallback (with mask support) ──
        // Use grouped query attention: reshape Q into [B*kv_heads, n_rep, S, D]
        // and use broadcasting with K/V [B*kv_heads, S, D] to avoid the expensive
        // 7x KV head expansion (repeat_kv).
        let n_rep = self.num_heads / self.num_kv_heads;
        let kv_s = k.dim(2)?;

        if n_rep > 1 && seq_len == 1 {
            // ── GQA-grouped SDPA for decode (seq_len=1) ──
            // Q: [B, 28, 1, D] → [B, 4, 7, D] → [B*4, 7, D]
            let q_g = q
                .reshape((b_sz, self.num_kv_heads, n_rep, self.head_dim))?
                .reshape((b_sz * self.num_kv_heads, n_rep, self.head_dim))?
                .contiguous()?;
            // K: [B, 4, S, D] → [B*4, D, S]
            let k_g = k
                .reshape((b_sz * self.num_kv_heads, kv_s, self.head_dim))?
                .transpose(D::Minus2, D::Minus1)?
                .contiguous()?;
            // V: [B, 4, S, D] → [B*4, S, D]
            let v_g = v
                .reshape((b_sz * self.num_kv_heads, kv_s, self.head_dim))?
                .contiguous()?;

            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let attn_weights = (q_g.matmul(&k_g)? * scale)?; // [B*4, 7, S]

            let attn_weights = match attention_mask {
                Some(mask) => {
                    // mask: [B, 1, 1, S] → [B, 4, 1, S] → [B*4, 1, S]
                    let mask_g = mask
                        .expand((b_sz, self.num_kv_heads, 1, kv_s))?
                        .reshape((b_sz * self.num_kv_heads, 1, kv_s))?
                        .contiguous()?;
                    attn_weights.broadcast_add(&mask_g)?
                }
                None => attn_weights,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_output = attn_weights.matmul(&v_g)?; // [B*4, 7, D]

            // Reshape back: [B*4, 7, D] → [B, 28, D] → [B, 1, 28*D]
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

struct Mlp {
    gate_proj: LinearLayer,
    up_proj: LinearLayer,
    down_proj: LinearLayer,
}

impl Mlp {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let gate_proj = LinearLayer::Standard(linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )?);
        let up_proj = LinearLayer::Standard(linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )?);
        let down_proj = LinearLayer::Standard(linear_no_bias(
            config.intermediate_size,
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
        let gate = candle_nn::Activation::Silu.forward(&gate)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
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
        write_pos: Option<&Tensor>,
        varlen_ctx: Option<&VarlenContext>,
    ) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(
            &hidden_states,
            cos,
            sin,
            attention_mask,
            write_pos,
            varlen_ctx,
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

pub struct HunYuanDenseV1 {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: LinearLayer,
    rotary_emb: RotaryEmbedding,
    config: Config,
    dtype: DType,
}

impl HunYuanDenseV1 {
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

        let lm_head = if config.tie_word_embeddings {
            LinearLayer::Standard(Linear::new(embed_tokens.embeddings().clone(), None))
        } else {
            LinearLayer::Standard(linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                vb.pp("lm_head"),
            )?)
        };

        let rotary_emb = RotaryEmbedding::new(config, vb.device())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            config: config.clone(),
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

        // Detect architecture prefix (e.g. "qwen2", "qwen3", "llama")
        let arch = gg
            .metadata()
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.clone())
            .unwrap_or_else(|| "qwen2".to_string());

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
            .unwrap_or(32768) as usize;
        let rms_norm_eps = gg
            .metadata()
            .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1e-6) as f64;
        let rope_theta = gg
            .metadata()
            .get(&format!("{arch}.rope.freq_base"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(10_000.0) as f64;

        // Check for QK norm by probing tensor existence
        let use_qk_norm = gg.ct.tensor_infos.contains_key("blk.0.attn_q_norm.weight");

        // Check for tied embeddings
        let tie_word_embeddings = !gg.ct.tensor_infos.contains_key("output.weight");

        // Build config
        let config = Config {
            vocab_size: 0, // updated below
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads: num_attention_heads,
            num_key_value_heads: num_kv_heads,
            head_dim: Some(head_dim),
            hidden_act: "silu".to_string(),
            max_position_embeddings,
            rms_norm_eps,
            rope_theta: Some(rope_theta),
            rope_scaling: None,
            attention_bias: Some(false),
            use_qk_norm,
            tie_word_embeddings,
            use_cla: None,
            cla_share_factor: None,
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

        let rotary_emb = RotaryEmbedding::new(&config, device)?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            config,
            dtype,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        // Cast embedding output to the target dtype — the safetensors file may store
        // weights in BF16 which is unsupported for CPU matmul.
        let hidden_states = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;

        let total_len = start_pos + seq_len;
        let (full_cos, full_sin) = self.rotary_emb.forward(total_len, input_ids.device())?;
        // Slice for current positions; cast to model dtype (RoPE computes in F32)
        // cos/sin: [seq_len, dim/2] — rope() handles broadcasting.
        let cos = full_cos
            .narrow(0, start_pos, seq_len)?
            .to_dtype(self.dtype)?;
        let sin = full_sin
            .narrow(0, start_pos, seq_len)?
            .to_dtype(self.dtype)?;

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
            hidden_states = layer.forward(
                &hidden_states,
                &cos,
                &sin,
                attention_mask.as_ref(),
                None,
                None,
            )?;
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
        self.rotary_emb.cos_cache = None;
        self.rotary_emb.sin_cache = None;
        self.rotary_emb.cached_len = 0;
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Extract per-layer KV caches. Returns only the valid (filled) portion,
    /// **not** the pre-allocated buffer. Per-sequence storage should keep these
    /// compactly; `setup_batch_decode` will re-create a pre-allocated buffer.
    pub fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.layers
            .iter()
            .map(|l| {
                l.self_attn.kv_cache.as_ref().map(|(k, v)| {
                    let len = l.self_attn.cache_seq_len;
                    if len > 0 && len < k.dim(2).unwrap_or(0) {
                        // Narrow to valid data — still a zero-copy view.
                        (
                            k.narrow(2, 0, len).unwrap_or_else(|_| k.clone()),
                            v.narrow(2, 0, len).unwrap_or_else(|_| v.clone()),
                        )
                    } else {
                        (k.clone(), v.clone())
                    }
                })
            })
            .collect()
    }

    /// Restore per-layer KV caches (e.g. after swapping sequences).
    /// The tensors are stored as-is; `cache_seq_len` is set to their dim(2).
    pub fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        for (layer, cache) in self.layers.iter_mut().zip(caches.into_iter()) {
            let seq_len = cache
                .as_ref()
                .map(|(k, _)| k.dim(2).unwrap_or(0))
                .unwrap_or(0);
            layer.self_attn.kv_cache = cache;
            layer.self_attn.cache_seq_len = seq_len;
        }
    }

    /// Pad per-sequence KV caches to the same length and load into model's
    /// attention layers, preparing for batched decode.
    ///
    /// Returns `(kv_lens, max_kv_len)` where `kv_lens[i]` is the original
    /// number of cached tokens for sequence i, and `max_kv_len` is the
    /// padded length.
    ///
    /// After calling this, use `step_batch_decode` for each decode round,
    /// then `extract_batch_kv` to get clean per-sequence caches back.
    /// `extra_room`: number of decode tokens to pre-allocate in the buffer
    /// (avoids `Tensor::cat` reallocation during multi-round decode).
    pub fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        extra_room: usize,
    ) -> Result<(Vec<usize>, usize)> {
        let kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim();
        let device = self.embed_tokens.embeddings().device();

        // Compute per-sequence KV lengths from the first layer's cache.
        let kv_lens: Vec<usize> = seq_kv_caches
            .iter()
            .map(|caches| {
                caches
                    .first()
                    .and_then(|c| c.as_ref())
                    .map(|(k, _)| k.dim(2).unwrap_or(0))
                    .unwrap_or(0)
            })
            .collect();
        let max_kv_len = kv_lens.iter().copied().max().unwrap_or(0);

        // For each layer, pad and stack KV caches.
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let layer_caches: Vec<&Option<(Tensor, Tensor)>> =
                seq_kv_caches.iter().map(|seq| &seq[layer_idx]).collect();

            let batched_kv = pad_and_stack_kv_caches(
                &layer_caches,
                max_kv_len,
                kv_heads,
                head_dim,
                device,
                self.dtype,
            )?;

            if let Some((k, v)) = batched_kv {
                // Ensure contiguous for slice_set (narrow views from
                // get_kv_caches / extract_batch_kv are strided).
                let k = k.contiguous()?;
                let v = v.contiguous()?;
                if extra_room > 0 {
                    // Pre-allocate buffer with room for decode rounds.
                    let (b, h, s, d) = k.dims4()?;
                    let buf_k = Tensor::zeros((b, h, s + extra_room, d), k.dtype(), k.device())?;
                    let buf_v = Tensor::zeros((b, h, s + extra_room, d), v.dtype(), v.device())?;
                    buf_k.slice_set(&k, 2, 0)?;
                    buf_v.slice_set(&v, 2, 0)?;
                    layer.self_attn.kv_cache = Some((buf_k, buf_v));
                } else {
                    layer.self_attn.kv_cache = Some((k, v));
                }
                layer.self_attn.cache_seq_len = max_kv_len;
            } else {
                layer.self_attn.kv_cache = None;
                layer.self_attn.cache_seq_len = 0;
            }
        }

        Ok((kv_lens, max_kv_len))
    }

    /// Run one batched decode step using the model's existing (batched) KV cache.
    ///
    /// Call `setup_batch_decode` first, then call this repeatedly for each
    /// decode round. The KV cache grows by 1 column each round (via the
    /// attention layer's internal concat).
    ///
    /// # Arguments
    /// - `input_ids` — `[N, 1]` tensor (one token per sequence)
    /// - `positions` — logical position for each sequence's new token
    /// - `attention_mask` — optional `[N, 1, 1, total_kv_after_concat]` mask
    ///
    /// # Returns
    /// Logits tensor `[N, 1, vocab_size]`.
    pub fn step_batch_decode(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> Result<Tensor> {
        let hidden_states = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;

        let max_pos = positions.iter().copied().max().unwrap_or(0) + 1;
        let device = input_ids.device();
        let (full_cos, full_sin) = self.rotary_emb.forward(max_pos, device)?;

        let pos_ids: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        let pos_tensor = Tensor::new(pos_ids.as_slice(), device)?;
        // cos/sin: [N, dim/2] after index_select → [N, 1, dim/2] for rope()
        // rope() accepts 3D cos/sin as [B, S, D/2].
        let cos = full_cos
            .index_select(&pos_tensor, 0)?
            .to_dtype(self.dtype)?
            .unsqueeze(1)?;
        let sin = full_sin
            .index_select(&pos_tensor, 0)?
            .to_dtype(self.dtype)?
            .unsqueeze(1)?;

        // Pre-compute varlen metadata ONCE per round, shared across all layers.
        // This eliminates 84 redundant CPU→GPU transfers per round (3 per layer × 28).
        #[cfg(feature = "flash-attn")]
        let varlen_ctx = if let Some((kv_lens, max_kv)) = batch_kv_info {
            let b_sz = kv_lens.len();
            let cache_s = self.layers[0].self_attn.cache_seq_len;
            let total_s = cache_s + 1; // after update_kv_cache appends 1 token
            let decode_tokens = total_s - max_kv;

            let seqlens_q_data: Vec<u32> = (0..=b_sz).map(|i| i as u32).collect();
            let seqlens_q = Tensor::new(seqlens_q_data.as_slice(), device)?;

            let valid_kv_lens: Vec<usize> = kv_lens.iter().map(|&l| l + decode_tokens).collect();
            let mut seqlens_k_data = vec![0u32; b_sz + 1];
            for i in 0..b_sz {
                seqlens_k_data[i + 1] = seqlens_k_data[i] + valid_kv_lens[i] as u32;
            }
            let seqlens_k = Tensor::new(seqlens_k_data.as_slice(), device)?;

            let total_kv: usize = valid_kv_lens.iter().sum();
            let mut indices: Vec<i64> = Vec::with_capacity(total_kv);
            for i in 0..b_sz {
                let row_base = i * total_s;
                let start = max_kv - kv_lens[i];
                for j in start..total_s {
                    indices.push((row_base + j) as i64);
                }
            }
            let gather_index = Tensor::new(indices.as_slice(), device)?;
            let max_seqlen_k = valid_kv_lens.iter().copied().max().unwrap_or(1);

            Some(VarlenContext {
                seqlens_q,
                seqlens_k,
                gather_index,
                max_seqlen_k,
            })
        } else {
            None
        };

        #[cfg(not(feature = "flash-attn"))]
        let _ = batch_kv_info;

        let mut hidden_states = hidden_states;
        for layer in self.layers.iter_mut() {
            #[cfg(feature = "flash-attn")]
            {
                hidden_states = layer.forward(
                    &hidden_states,
                    &cos,
                    &sin,
                    attention_mask,
                    None,
                    varlen_ctx.as_ref(),
                )?;
            }
            #[cfg(not(feature = "flash-attn"))]
            {
                hidden_states =
                    layer.forward(&hidden_states, &cos, &sin, attention_mask, None, None)?;
            }
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        self.lm_head.forward(&hidden_states) // [N, 1, vocab]
    }

    /// Graph-compatible batched decode step.
    ///
    /// Unlike `step_batch_decode`, all inputs are **pre-allocated tensors** —
    /// no CPU→GPU copies happen inside this function. The caller updates the
    /// tensors' contents (via `slice_set` / `copy_`) between graph replays.
    ///
    /// # Arguments
    /// - `input_ids`: `[N, 1]` token IDs (u32)
    /// - `cos`, `sin`: `[N, 1, dim/2]` rotary-embedding vectors
    /// - `attention_mask`: `[N, 1, 1, max_kv_len]` mask (0.0 = attend, -1e9 = ignore)
    /// - `write_pos`: `[N]` u32 positions to write into the KV buffer
    ///
    /// # Returns
    /// Logits tensor `[N, 1, vocab_size]`.
    pub fn step_batch_decode_graph(
        &mut self,
        input_ids: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: &Tensor,
        write_pos: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let hidden_states = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;

        let mut hidden_states = hidden_states;
        for layer in self.layers.iter_mut() {
            hidden_states = layer.forward(
                &hidden_states,
                cos,
                sin,
                Some(attention_mask),
                Some(write_pos),
                None,
            )?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        self.lm_head.forward(&hidden_states) // [N, 1, vocab]
    }

    /// Extract per-sequence KV caches from the batched state, removing padding.
    ///
    /// Call this after all decode rounds are done to get clean per-sequence
    /// caches for storage. Clears the model's KV cache afterward.
    ///
    /// # Arguments
    /// - `kv_lens` — original per-sequence KV lengths (from `setup_batch_decode`)
    /// - `original_max_kv` — max KV length at setup time (from `setup_batch_decode`)
    /// - `rounds_done` — number of `step_batch_decode` calls completed
    pub fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        let n_seqs = kv_lens.len();
        let num_layers = self.layers.len();
        let mut result: Vec<Vec<Option<(Tensor, Tensor)>>> = (0..n_seqs)
            .map(|_| Vec::with_capacity(num_layers))
            .collect();

        for layer in self.layers.iter_mut() {
            if let Some((ref full_k, ref full_v)) = layer.self_attn.kv_cache {
                for i in 0..n_seqs {
                    let row_k = full_k.narrow(0, i, 1)?;
                    let row_v = full_v.narrow(0, i, 1)?;

                    // Right-aligned layout: valid data is always contiguous.
                    // Prefill at [max_kv - kv_lens[i] .. max_kv],
                    // decode at [max_kv .. max_kv + rounds_done].
                    let total = kv_lens[i] + rounds_done;
                    let offset = original_max_kv - kv_lens[i];
                    let clean = Some((
                        row_k.narrow(2, offset, total)?,
                        row_v.narrow(2, offset, total)?,
                    ));
                    result[i].push(clean);
                }
            } else {
                for i in 0..n_seqs {
                    result[i].push(None);
                }
            }
            layer.self_attn.kv_cache = None;
            layer.self_attn.cache_seq_len = 0;
        }

        Ok(result)
    }

    // ── CUDA Graph mode infrastructure ──

    /// Pre-allocate PERMANENT KV cache buffers for all layers.
    ///
    /// These buffers have fixed shape `[max_batch, kv_heads, max_kv_len, head_dim]`
    /// and their device-memory addresses are baked into CUDA graphs during
    /// capture. Call this once at startup before `capture_decode_graph`.
    pub fn preallocate_graph_kv(
        &mut self,
        max_batch: usize,
        max_kv_len: usize,
    ) -> candle_core::Result<()> {
        let kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim();
        let device = self.embed_tokens.embeddings().device();
        for layer in self.layers.iter_mut() {
            let buf_k = Tensor::zeros(
                (max_batch, kv_heads, max_kv_len, head_dim),
                self.dtype,
                device,
            )?;
            let buf_v = Tensor::zeros(
                (max_batch, kv_heads, max_kv_len, head_dim),
                self.dtype,
                device,
            )?;
            layer.self_attn.kv_cache = Some((buf_k, buf_v));
            layer.self_attn.cache_seq_len = 0;
            layer.self_attn.full_buffer_attn = true;
        }
        Ok(())
    }

    /// Load per-sequence KV caches into the permanent graph buffers.
    ///
    /// Zeroes the entire permanent buffer, then writes each sequence's cached
    /// KV data at batch-row `i` starting at seq-position 0.
    ///
    /// # Returns
    /// `(kv_lens, max_kv_len)` — same semantics as `setup_batch_decode`.
    pub fn load_kv_for_graph(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        let n_seqs = seq_kv_caches.len();

        // Compute per-sequence KV lengths from the first layer's cache.
        let kv_lens: Vec<usize> = seq_kv_caches
            .iter()
            .map(|caches| {
                caches
                    .first()
                    .and_then(|c| c.as_ref())
                    .map(|(k, _)| k.dim(2).unwrap_or(0))
                    .unwrap_or(0)
            })
            .collect();
        let max_kv_len = kv_lens.iter().copied().max().unwrap_or(0);

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let (ref buf_k, ref buf_v) = layer.self_attn.kv_cache.as_ref().ok_or_else(|| {
                candle_core::Error::Msg(
                    "load_kv_for_graph: no pre-allocated KV buffer; \
                     call preallocate_graph_kv first"
                        .into(),
                )
            })?;
            // Zero the entire buffer so dead/padding regions are clean.
            buf_k.zero_set()?;
            buf_v.zero_set()?;

            // Write each sequence's KV at its batch row.
            for (seq_idx, seq_caches) in seq_kv_caches.iter().enumerate() {
                if let Some((ref k, ref v)) = seq_caches[layer_idx] {
                    // k: [1, kv_heads, seq_len, head_dim]
                    // Write into buf[seq_idx, :, 0..seq_len, :]
                    // narrow(0, seq_idx, 1) gives a view of that batch row.
                    let dest_k = buf_k.narrow(0, seq_idx, 1)?;
                    let dest_v = buf_v.narrow(0, seq_idx, 1)?;
                    dest_k.slice_set(&k.contiguous()?, 2, 0)?;
                    dest_v.slice_set(&v.contiguous()?, 2, 0)?;
                }
            }
            layer.self_attn.cache_seq_len = max_kv_len;
        }

        Ok((kv_lens, max_kv_len))
    }

    /// Compute rotary embeddings for given positions.
    ///
    /// Returns `(cos, sin)` with shape `[N, 1, head_dim/2]` in model dtype,
    /// suitable for passing to `step_batch_decode_graph`.
    pub fn compute_rotary(&mut self, positions: &[usize]) -> candle_core::Result<(Tensor, Tensor)> {
        let device = self.embed_tokens.embeddings().device();
        let max_pos = positions.iter().copied().max().unwrap_or(0) + 1;
        let (full_cos, full_sin) = self.rotary_emb.forward(max_pos, device)?;

        let pos_ids: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        let pos_tensor = Tensor::new(pos_ids.as_slice(), device)?;
        let cos = full_cos
            .index_select(&pos_tensor, 0)?
            .to_dtype(self.dtype)?
            .unsqueeze(1)?;
        let sin = full_sin
            .index_select(&pos_tensor, 0)?
            .to_dtype(self.dtype)?
            .unsqueeze(1)?;
        Ok((cos, sin))
    }

    /// Extract per-sequence KV caches from the graph buffers without
    /// deallocating them. The permanent buffers remain intact for the
    /// next graph replay batch.
    ///
    /// Unlike `extract_batch_kv`, this does NOT clear the model's KV cache.
    pub fn extract_graph_kv(
        &self,
        kv_lens: &[usize],
        rounds_done: usize,
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        let n_seqs = kv_lens.len();
        let num_layers = self.layers.len();
        let mut result: Vec<Vec<Option<(Tensor, Tensor)>>> = (0..n_seqs)
            .map(|_| Vec::with_capacity(num_layers))
            .collect();

        for layer in self.layers.iter() {
            if let Some((ref full_k, ref full_v)) = layer.self_attn.kv_cache {
                for i in 0..n_seqs {
                    let total = kv_lens[i] + rounds_done;
                    let row_k = full_k.narrow(0, i, 1)?;
                    let row_v = full_v.narrow(0, i, 1)?;
                    // In graph mode, data is written contiguously at positions
                    // 0..kv_lens[i] (from load_kv_for_graph) then at
                    // kv_lens[i]..kv_lens[i]+rounds_done (from scatter_set during decode).
                    // Wait — scatter_set writes at the POSITION specified, which
                    // is the logical position (varies per sequence). So the data
                    // IS at 0..total contiguously.
                    result[i].push(Some((
                        row_k.narrow(2, 0, total)?.contiguous()?,
                        row_v.narrow(2, 0, total)?.contiguous()?,
                    )));
                }
            } else {
                for i in 0..n_seqs {
                    result[i].push(None);
                }
            }
        }

        Ok(result)
    }

    /// Access the model config.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Access the model dtype.
    pub fn model_dtype(&self) -> DType {
        self.dtype
    }
}

/// Build attention mask for batched decode with padding-aware masking.
///
/// Positions `kv_lens[i]..original_max_kv` are masked as -1e9 for each sequence.
/// Positions before `kv_lens[i]` and after `original_max_kv` are 0.0 (attend).
///
/// Returns `None` if no padding exists (all sequences have the same KV length).
pub fn build_batch_decode_mask(
    kv_lens: &[usize],
    original_max_kv: usize,
    total_width: usize,
    device: &Device,
    dtype: DType,
) -> Result<Option<Tensor>> {
    if kv_lens.iter().all(|&l| l == original_max_kv) {
        return Ok(None);
    }
    let n = kv_lens.len();
    let mut mask_data = vec![0f32; n * total_width];
    for i in 0..n {
        // Right-aligned layout: padding at positions 0..pad_end.
        let pad_end = (original_max_kv - kv_lens[i]).min(total_width);
        for j in 0..pad_end {
            mask_data[i * total_width + j] = -1e9;
        }
    }
    let mask = Tensor::from_vec(mask_data, (n, total_width), device)?.to_dtype(dtype)?;
    Ok(Some(mask.unsqueeze(1)?.unsqueeze(1)?))
}

/// Pad per-sequence KV caches to `max_len` and stack into a batched tensor.
///
/// Returns `Some((K, V))` with shape `[N, kv_heads, max_len, head_dim]`,
/// or `None` if `max_len == 0`.
fn pad_and_stack_kv_caches(
    caches: &[&Option<(Tensor, Tensor)>],
    max_len: usize,
    kv_heads: usize,
    head_dim: usize,
    device: &Device,
    dtype: DType,
) -> Result<Option<(Tensor, Tensor)>> {
    if max_len == 0 {
        return Ok(None);
    }

    let n = caches.len();
    let mut padded_ks = Vec::with_capacity(n);
    let mut padded_vs = Vec::with_capacity(n);

    // Pre-allocate a zero tensor for padding (shared across sequences)
    let max_pad_needed = caches
        .iter()
        .map(|c| match c {
            Some((k, _)) => max_len.saturating_sub(k.dim(2).unwrap_or(0)),
            None => max_len,
        })
        .max()
        .unwrap_or(0);
    let zero_pad = if max_pad_needed > 0 {
        Some(Tensor::zeros(
            (1, kv_heads, max_pad_needed, head_dim),
            dtype,
            device,
        )?)
    } else {
        None
    };

    for cache in caches {
        match cache {
            Some((k, v)) => {
                let cur_len = k.dim(2)?;
                let pad_len = max_len - cur_len;
                if pad_len > 0 {
                    // Right-align: [padding | real_data] so that decode tokens
                    // appended after max_kv form a contiguous valid range with
                    // the pre-existing data, enabling flash_attn_varlen.
                    let pad = zero_pad.as_ref().unwrap().narrow(2, 0, pad_len)?;
                    padded_ks.push(Tensor::cat(&[&pad, k.as_ref()], 2)?);
                    padded_vs.push(Tensor::cat(&[&pad, v.as_ref()], 2)?);
                } else {
                    padded_ks.push(k.clone());
                    padded_vs.push(v.clone());
                }
            }
            None => {
                let zeros = Tensor::zeros((1, kv_heads, max_len, head_dim), dtype, device)?;
                padded_ks.push(zeros.clone());
                padded_vs.push(zeros);
            }
        }
    }

    let stacked_k = Tensor::cat(&padded_ks, 0)?.contiguous()?;
    let stacked_v = Tensor::cat(&padded_vs, 0)?.contiguous()?;
    Ok(Some((stacked_k, stacked_v)))
}
