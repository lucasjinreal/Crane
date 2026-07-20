//! Optimized Qwen3 transformer implementation.
//!
//! Adapted from the HunyuanDense model with full parity on all optimizations:
//!
//! 1. **Pre-allocated KV cache** with in-place `slice_set` writes
//!    — O(new_seq_len) per decode step instead of O(cache_len) `Tensor::cat`.
//! 2. **Fused flash attention for decode and prefill** (CPU, B=1)
//!    — Uses `candle_nn::attention::flash_attn`'s online-softmax kernel
//!      (O(head_dim) working set, native GQA) instead of materializing an
//!      O(context_len) scores tensor. Prefill additionally skips the
//!      GQA K/V expansion (which duplicates K/V n_rep times) and uses
//!      `AttnMask::Causal` so masking is done via loop bounds, not a
//!      materialized mask tensor. Falls back to a GQA-grouped matmul SDPA
//!      on GPU or for batched (B>1) decode, where cuBLAS is already
//!      compute-bound or an explicit per-sequence mask is required, and to
//!      a standard SDPA for GPU or batched (B>1) prefill, or when
//!      num_heads == num_kv_heads (no GQA grouping needed).
//! 3. **Fused `RoPE` kernel** via `candle_nn::rotary_emb::rope_thd()`
//!    — One CUDA launch per Q/K instead of 5 manual tensor ops.
//!    — Applied in BSHD layout (before the transpose to BHSD), so the
//!      reshape output is already contiguous — no `contiguous()` copy
//!      needed before `RoPE`, and QK norm hits the fast fused RmsNorm path.
//!    — Precomputed `[max_pos, head_dim/2]` cos/sin tables (half-width, as
//!      required by the `rope_thd()` API).
//! 4. **GGUF quantization** via the polymorphic `LinearLayer` enum
//!    — Same model code serves both safetensors (f16/f32/bf16) and GGUF weights.
//! 5. **Batched decode infrastructure**
//!    — `setup_batch_decode`, `step_batch_decode`, `extract_batch_kv` enable
//!      GPU-efficient concurrent sequence serving in the engine.
//! 6. **KV cache save/restore**
//!    — `get_kv_caches` / `set_kv_caches` for continuous-batching context swap.
//! 7. **Fused SiLU-mul MLP gate**
//!    — `fused_silu_mul` replaces the `narrow + silu + mul` op chain in each
//!      MLP block, reducing kernel launches and intermediate allocations.
//! 8. **Merged QKV / gate+up projections**
//!    — Q, K, V weights fused into one matmul; gate and up weights fused into
//!      one matmul — halves the number of linear-layer dispatches per layer.

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::attention::AttnMask;
use candle_nn::rotary_emb::rope_thd;
use candle_nn::{linear_no_bias, Linear, RmsNorm, VarBuilder};
use serde::Deserialize;
use std::io::{Read, Seek};

use crate::models::modules::flash_attn::dispatch_flash_attn;
use crate::models::modules::kv_cache;
use crate::models::modules::rotary::RotaryEmbedding;

// Reuse the polymorphic linear layer and GGUF loader from the shared Hunyuan module.
pub use crate::models::hunyuan_dense::modeling::{Gguf, LinearLayer};

// ── Event-tracking RAII guard ────────────────────────────────────────────
//
// Candle defaults to tracking per-tensor CudaEvents for multi-stream safety.
// Crane uses a single CUDA stream — those events are pure overhead.
// This guard disables event tracking on first use and leaves it disabled
// (candle 0.9.x exposes only `disable_event_tracking`, not a re-enable).

#[cfg(feature = "cuda")]
struct EventTrackingGuard;

#[cfg(feature = "cuda")]
impl EventTrackingGuard {
    fn disable(device: &candle_core::Device) -> Self {
        if let candle_core::Device::Cuda(dev) = device {
            if dev.is_event_tracking() {
                // Safety: we ensure sequential use of a single CUDA stream.
                unsafe { dev.disable_event_tracking() };
            }
        }
        Self
    }
}

/// Per-layer, per-sequence KV cache tensors, as returned by
/// [`Qwen3Model::extract_batch_kv`].
pub type BatchKvCache = Vec<Vec<Option<(Tensor, Tensor)>>>;

// ── Config ──────────────────────────────────────────────────────────────

fn default_true() -> bool {
    true
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub max_window_layers: usize,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

// ── Attention ───────────────────────────────────────────────────────────

struct Attention {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    o_proj: LinearLayer,
    /// Merged QKV weight [q_dim + 2*kv_dim, hidden_size] — one gemv instead of 3.
    /// Only set for Standard (non-quantized) weights.
    qkv_proj: Option<Linear>,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    /// Pre-allocated KV cache buffer (may be larger than `cache_seq_len`).
    kv_cache: Option<(Tensor, Tensor)>,
    /// Number of valid (filled) positions in the KV cache buffer.
    cache_seq_len: usize,
}

impl Attention {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let bias = config.attention_bias;

        let make_proj = |in_d: usize, out_d: usize, name: &str| -> Result<LinearLayer> {
            if bias {
                Ok(LinearLayer::Standard(candle_nn::linear(
                    in_d,
                    out_d,
                    vb.pp(name),
                )?))
            } else {
                Ok(LinearLayer::Standard(linear_no_bias(
                    in_d,
                    out_d,
                    vb.pp(name),
                )?))
            }
        };

        let q_proj = make_proj(config.hidden_size, num_heads * head_dim, "q_proj")?;
        let k_proj = make_proj(config.hidden_size, num_kv_heads * head_dim, "k_proj")?;
        let v_proj = make_proj(config.hidden_size, num_kv_heads * head_dim, "v_proj")?;
        let o_proj = make_proj(num_heads * head_dim, config.hidden_size, "o_proj")?;

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

        let (q_norm, k_norm) = if config.use_qk_norm {
            (
                Some(candle_nn::rms_norm(
                    head_dim,
                    config.rms_norm_eps,
                    vb.pp("q_norm"),
                )?),
                Some(candle_nn::rms_norm(
                    head_dim,
                    config.rms_norm_eps,
                    vb.pp("k_norm"),
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
            qkv_proj,
            q_norm,
            k_norm,
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

        let (q_norm, k_norm) = if config.use_qk_norm {
            (
                Some(gg.rms_norm(
                    &format!("{prefix}.attn_q_norm.weight"),
                    config.rms_norm_eps,
                )?),
                Some(gg.rms_norm(
                    &format!("{prefix}.attn_k_norm.weight"),
                    config.rms_norm_eps,
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
            qkv_proj: None, // GGUF quantized — cannot merge
            q_norm,
            k_norm,
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
    fn update_kv_cache(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let cache = self.kv_cache.take();
        let prev_seq_len = std::mem::replace(&mut self.cache_seq_len, 0);
        let update = kv_cache::update_kv_cache(cache, prev_seq_len, k, v)?;
        self.kv_cache = Some(update.buffer);
        self.cache_seq_len = update.seq_len;
        Ok((update.k, update.v))
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden_states.dims3()?;

        // Use merged QKV if available — one gemv instead of three.
        // q/k/v are each [B, S, num_heads * head_dim] here.
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

        // reshape() always returns a contiguous tensor (zero-copy when the
        // input already is, e.g. the separate-projections path; a copy when
        // it isn't, e.g. the merged QKV narrow() path — same as before this
        // change).
        let q = q.reshape((b_sz, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head QK norm (Qwen3 applies before RoPE), applied in BSHD
        // layout while q/k are still contiguous — this hits RmsNorm's fast
        // fused kernel instead of the non-contiguous element-wise fallback.
        let q = if let Some(ref norm) = self.q_norm {
            norm.forward(&q)?
        } else {
            q
        };
        let k = if let Some(ref norm) = self.k_norm {
            norm.forward(&k)?
        } else {
            k
        };

        // Fused RoPE in BSHD layout — q/k are still contiguous here, so no
        // contiguous() copy is needed before rope_thd (unlike rope(), which
        // requires BHSD and would force a copy after the transpose below).
        let q = rope_thd(&q, cos, sin)?;
        let k = rope_thd(&k, cos, sin)?;

        // [B, S, H, D] → [B, H, S, D] for the KV cache and attention below.
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;

        // Update KV cache (pre-allocated with slice_set)
        let (k, v) = self.update_kv_cache(&k, &v)?;

        // ── SDPA ──
        let n_rep = self.num_heads / self.num_kv_heads;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        // 1/sqrt(head_dim) is always small and positive; f64->f32 here
        // only drops precision flash_attn's own f32 accumulator would
        // discard anyway.
        #[allow(clippy::cast_possible_truncation)]
        let scale_f32 = scale as f32;

        if seq_len == 1 && b_sz == 1 && q.device().is_cpu() {
            // ── Fused flash attention for decode (seq_len=1), CPU only ──
            // candle's cpu_flash kernel streams K/V with online softmax
            // (O(head_dim) working set instead of materializing an O(S)
            // scores tensor 3 times), and handles GQA natively via integer
            // division — no Q reshape trick needed. Not available on GPU;
            // cuBLAS matmuls there are compute-bound, so the plain path
            // below is used instead.
            //
            // b_sz == 1 only: candle's flash_attn hard-errors for B>1 with
            // an explicit Mask tensor (only Causal/None are allowed), and
            // crane-serve's continuous-batching decode
            // (`step_batch_decode` / `build_batch_decode_mask`) passes
            // exactly that — an explicit per-sequence padding mask with
            // B>1 — whenever batched sequences have different KV-cache
            // lengths. Single-sequence decode never hits that mask shape,
            // so it's the only case safe to fast-path here.

            // BHSD [B, H, S, D] → BSHD [B, S, H, D], as flash_attn expects.
            // Non-contiguous is fine — the decode kernel indexes by stride.
            let q_bshd = q.transpose(1, 2)?;
            let k_bshd = k.transpose(1, 2)?;
            let v_bshd = v.transpose(1, 2)?;

            let mask = match attention_mask {
                // AttnMask::Mask takes ownership; Tensor is Arc-backed, so
                // this is a refcount bump, not a data copy.
                Some(mask) => AttnMask::Mask(mask.clone()),
                None => AttnMask::None,
            };

            let attn_output = dispatch_flash_attn(&q_bshd, &k_bshd, &v_bshd, scale_f32, mask)?;

            // flash_attn output is BHSD [B, H, 1, D] → [B, 1, H*D]
            let attn_output = attn_output
                .reshape((b_sz, self.num_heads, self.head_dim))?
                .reshape((b_sz, 1, self.num_heads * self.head_dim))?;
            return self.o_proj.forward(&attn_output);
        }

        if b_sz == 1 && q.device().is_cpu() && attention_mask.is_none() {
            // ── Fused flash attention for prefill (seq_len > 1), CPU only ──
            // Same benefits as the decode fast path above, plus it avoids
            // the GQA K/V expansion below (unsqueeze/expand/reshape, which
            // duplicates K and V n_rep times) and never materializes the
            // O(H * S_q * S_kv) score tensor. The causal kernel masks via
            // loop bounds (kv_offset), so no mask tensor is built or read.
            // attention_mask.is_none() guards this: an explicit mask (e.g.
            // a future non-causal caller) can't be expressed via
            // AttnMask::Causal, so such callers fall through to the SDPA
            // path below instead of having their mask silently dropped.

            // BHSD [B, H, S, D] → BSHD [B, S, H, D]
            let q_bshd = q.transpose(1, 2)?;
            let k_bshd = k.transpose(1, 2)?;
            let v_bshd = v.transpose(1, 2)?;

            // The layer's KV cache tracks `cache_seq_len` filled positions;
            // after update_kv_cache, the K/V seq dim equals cache_seq_len,
            // so kv_offset = cache_seq_len - seq_len recovers start_pos.
            let kv_offset = k_bshd.dim(1)? - seq_len;
            let mask = AttnMask::Causal { kv_offset };

            let attn_output = dispatch_flash_attn(&q_bshd, &k_bshd, &v_bshd, scale_f32, mask)?;

            // flash_attn output is BHSD [B, H, S, D] → [B, S, H*D]
            let attn_output = attn_output
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b_sz, seq_len, ()))?;
            return self.o_proj.forward(&attn_output);
        }

        if n_rep > 1 && seq_len == 1 {
            // ── GQA-grouped SDPA for decode (seq_len=1), GPU fallback ──
            // Use 4D tensors throughout so candle's matmul only has to
            // flatten+contiguous the non-contiguous K narrow-view ONCE
            // instead of reshape(contiguous) + transpose + contiguous.

            // Q: [B, H, 1, D] → [B, kv_heads, n_rep, D], pre-scaled
            let q_g =
                (q.reshape((b_sz, self.num_kv_heads, n_rep, self.head_dim))? * scale)?;

            // K^T: [B, kv_heads, D, S] — just a view (0 copies here;
            //       matmul will flatten+contiguous in one pass).
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

        // cuBLAS's strided-batched matmul needs q's (seq, head_dim) slice to
        // be plain row/col-major; after the BSHD->BHSD transpose above it
        // isn't (row stride is num_heads*head_dim, not head_dim) whenever
        // num_heads > 1. The CPU flash-attn fast paths above return before
        // reaching this line and never pay this cost.
        let q = q.contiguous()?;
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // [B, H, S, D] → [B, S, H*D]
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

    fn new_from_gguf<R: Read + Seek>(gg: &mut Gguf<R>, layer_idx: usize, _intermediate_size: usize) -> Result<Self> {
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

                // Use fused CUDA kernel when available: eliminates narrow + silu + mul
                // (3 kernel launches → 1).
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

                // CPU / non-CUDA fallback
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

// ── Decoder Layer ───────────────────────────────────────────────────────

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
        let mlp = Mlp::new_from_gguf(gg, layer_idx, config.intermediate_size)?;
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
        let hidden_states =
            self.self_attn
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

// ── Full Model ──────────────────────────────────────────────────────────

pub struct Qwen3Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: LinearLayer,
    rotary_emb: RotaryEmbedding,
    config: Config,
    dtype: DType,
}

impl Qwen3Model {
    /// Construct from safetensors / HuggingFace checkpoint.
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        Self::new_inner(config, vb.pp("model"), vb)
    }

    /// Construct from a checkpoint where the decoder is nested under a
    /// deeper prefix than the standard `model.*` layout (e.g. Qwen3-ASR's
    /// `model.language_model.*`). `model_vb` must already be scoped to the
    /// decoder's root (what would otherwise be `vb.pp("model")`); `root_vb`
    /// is the checkpoint root, used to resolve an untied `lm_head` sibling.
    pub fn new_from_model_vb(
        config: &Config,
        model_vb: VarBuilder,
        root_vb: VarBuilder,
    ) -> Result<Self> {
        Self::new_inner(config, model_vb, root_vb)
    }

    fn new_inner(config: &Config, model_vb: VarBuilder, root_vb: VarBuilder) -> Result<Self> {
        let dtype = model_vb.dtype();
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
                root_vb.pp("lm_head"),
            )?)
        };

        let rotary_emb = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            model_vb.device(),
        )?;

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
            .unwrap_or_else(|| "qwen3".to_string());

        let num_attention_heads =
            md_get(&format!("{arch}.attention.head_count"))?.to_u32()? as usize;
        let num_kv_heads =
            md_get(&format!("{arch}.attention.head_count_kv"))?.to_u32()? as usize;
        let head_dim = gg
            .metadata()
            .get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(128) as usize;
        let num_hidden_layers = md_get(&format!("{arch}.block_count"))?.to_u32()? as usize;
        let hidden_size = md_get(&format!("{arch}.embedding_length"))?.to_u32()? as usize;
        let intermediate_size =
            md_get(&format!("{arch}.feed_forward_length"))?.to_u32()? as usize;
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
            .unwrap_or(1_000_000.0) as f64;

        let use_qk_norm = gg.ct.tensor_infos.contains_key("blk.0.attn_q_norm.weight");
        let tie_word_embeddings = !gg.ct.tensor_infos.contains_key("output.weight");

        let config = Config {
            vocab_size: 0, // updated below
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads: num_kv_heads,
            head_dim: Some(head_dim),
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            attention_bias: false,
            use_qk_norm,
            tie_word_embeddings,
            sliding_window: None,
            max_window_layers: 0,
            use_sliding_window: false,
            eos_token_id: None,
        };

        let embed_tokens = gg.embedding("token_embd.weight", hidden_size)?;
        let actual_vocab_size = embed_tokens.embeddings().dim(0)?;
        let config = Config {
            vocab_size: actual_vocab_size,
            ..config
        };

        let mut layers = Vec::with_capacity(num_hidden_layers);
        for i in 0..num_hidden_layers {
            layers.push(DecoderLayer::new_from_gguf(&config, &mut gg, i)?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;

        let lm_head = if tie_word_embeddings {
            LinearLayer::Standard(Linear::new(embed_tokens.embeddings().clone(), None))
        } else {
            gg.linear("output.weight")?
        };

        let rotary_emb = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            device,
        )?;

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

    // ── Forward ─────────────────────────────────────────────────────────

    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;

        // Disable event tracking for the duration of the forward pass.
        // Crane uses a single CUDA stream; the per-tensor CudaEvents are
        // unnecessary and cost ~2×cuEventCreate+cuEventRecord per temp tensor.
        #[cfg(feature = "cuda")]
        let _event_guard = EventTrackingGuard::disable(input_ids.device());

        let hidden_states = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;
        self.decode(hidden_states, seq_len, start_pos, input_ids.device())
    }

    /// Same as [`Self::forward`], but starting from a caller-supplied
    /// embedding sequence instead of doing the token embedding lookup
    /// internally. Used by callers (e.g. Qwen3-ASR) that splice in
    /// non-text embeddings (audio, etc.) at specific positions before
    /// running the decoder.
    pub fn forward_embeds(&mut self, inputs_embeds: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len, hidden) = inputs_embeds.dims3()?;
        if hidden != self.config.hidden_size {
            candle_core::bail!(
                "forward_embeds: expected hidden_size {}, got {hidden}",
                self.config.hidden_size,
            );
        }

        #[cfg(feature = "cuda")]
        let _event_guard = EventTrackingGuard::disable(inputs_embeds.device());

        let hidden_states = inputs_embeds.to_dtype(self.dtype)?;
        self.decode(hidden_states, seq_len, start_pos, inputs_embeds.device())
    }

    /// Shared decoder body: rotary embeddings, causal mask, transformer
    /// layers, final norm, and the `lm_head` projection of the last
    /// position. Called by both [`Self::forward`] and
    /// [`Self::forward_embeds`] once they've produced `hidden_states`.
    fn decode(
        &mut self,
        hidden_states: Tensor,
        seq_len: usize,
        start_pos: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let total_len = start_pos + seq_len;
        let (cos, sin) = self.rotary_emb.forward(start_pos, seq_len)?;
        let cos = cos.to_dtype(self.dtype)?;
        let sin = sin.to_dtype(self.dtype)?;

        // Causal mask (only during prefill; skipped for single-token decode,
        // and for CPU/B=1 prefill, where Attention::forward's flash_attn
        // fast path masks via AttnMask::Causal instead of reading this).
        let b_sz = hidden_states.dim(0)?;
        let attention_mask = if seq_len > 1 && !(device.is_cpu() && b_sz == 1) {
            let mut mask_data = vec![0f32; seq_len * total_len];
            for i in 0..seq_len {
                for j in 0..total_len {
                    if j <= start_pos + i {
                        mask_data[i * total_len + j] = 1.0;
                    }
                }
            }
            let mask = Tensor::from_vec(mask_data, (seq_len, total_len), device)?;
            let mask = mask
                .broadcast_lt(&Tensor::new(0.5f32, device)?)?
                .to_dtype(self.dtype)?;
            let mask = (mask * (-1e9f64))?;
            Some(mask.unsqueeze(0)?.unsqueeze(0)?)
        } else {
            None
        };

        let mut hidden_states = hidden_states;
        for layer in self.layers.iter_mut() {
            hidden_states =
                layer.forward(&hidden_states, &cos, &sin, attention_mask.as_ref())?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self
            .lm_head
            .forward(&hidden_states.narrow(1, seq_len - 1, 1)?)?;
        Ok(logits)
    }

    /// The token embedding table, exposed for callers that need to embed
    /// text tokens themselves before splicing in other embeddings (e.g.
    /// Qwen3-ASR's audio/text embedding merge).
    pub fn embed_tokens(&self) -> &candle_nn::Embedding {
        &self.embed_tokens
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

    /// Total bytes held by the model's KV caches (no GPU copies).
    pub fn active_kv_cache_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| {
                l.self_attn
                    .kv_cache
                    .as_ref()
                    .map(|(k, v)| {
                        let k_bytes =
                            k.elem_count() as u64 * k.dtype().size_in_bytes() as u64;
                        let v_bytes =
                            v.elem_count() as u64 * v.dtype().size_in_bytes() as u64;
                        k_bytes + v_bytes
                    })
                    .unwrap_or(0)
            })
            .sum()
    }

    /// Extract per-layer KV caches (valid portion only, zero-copy narrow views).
    ///
    /// The returned views still reference the pre-allocated buffer.  Callers
    /// that need to free the buffer (e.g. batch-decode extract) should use
    /// `Tensor::contiguous()` on their side, or clear `seq.kv_caches` after
    /// consuming the views.
    pub fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.layers
            .iter()
            .map(|l| {
                l.self_attn.kv_cache.as_ref().map(|(k, v)| {
                    let len = l.self_attn.cache_seq_len;
                    if len > 0 && len < k.dim(2).unwrap_or(0) {
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

    /// Restore per-layer KV caches.
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

    // ── Batched Decode ──────────────────────────────────────────────────

    /// Pad per-sequence KV caches to the same length and load into model layers.
    ///
    /// Returns `(kv_lens, max_kv_len)`.
    pub fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        extra_room: usize,
    ) -> Result<(Vec<usize>, usize)> {
        let kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim();
        let device = self.embed_tokens.embeddings().device();

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
                let k = k.contiguous()?;
                let v = v.contiguous()?;
                if extra_room > 0 {
                    let (b, h, s, d) = k.dims4()?;
                    let buf_k =
                        Tensor::zeros((b, h, s + extra_room, d), k.dtype(), k.device())?;
                    let buf_v =
                        Tensor::zeros((b, h, s + extra_room, d), v.dtype(), v.device())?;
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

    /// Run one batched decode step.
    pub fn step_batch_decode(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        _batch_kv_info: Option<(&[usize], usize)>,
    ) -> Result<Tensor> {
        let hidden_states = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;

        let max_pos = positions.iter().copied().max().unwrap_or(0) + 1;
        let device = input_ids.device();
        let (full_cos, full_sin) = self.rotary_emb.forward(0, max_pos)?;
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

        let mut hidden_states = hidden_states;
        for layer in self.layers.iter_mut() {
            hidden_states =
                layer.forward(&hidden_states, &cos, &sin, attention_mask)?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        self.lm_head.forward(&hidden_states) // [N, 1, vocab]
    }

    /// Extract per-sequence KV caches from batched state.
    pub fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> Result<BatchKvCache> {
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
                    let total = kv_lens[i] + rounds_done;
                    let offset = original_max_kv - kv_lens[i];
                    // Contiguous copy — breaks ref to padded batch buffer.
                    let clean = Some((
                        row_k.narrow(2, offset, total)?.contiguous()?,
                        row_v.narrow(2, offset, total)?.contiguous()?,
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

    /// Access the model config.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Access the model dtype.
    pub fn model_dtype(&self) -> DType {
        self.dtype
    }
}

// ── Utilities ───────────────────────────────────────────────────────────

/// Build attention mask for batched decode with padding-aware masking.
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
        let pad_end = (original_max_kv - kv_lens[i]).min(total_width);
        for j in 0..pad_end {
            mask_data[i * total_width + j] = -1e9;
        }
    }
    let mask = Tensor::from_vec(mask_data, (n, total_width), device)?.to_dtype(dtype)?;
    Ok(Some(mask.unsqueeze(1)?.unsqueeze(1)?))
}

/// Pad per-sequence KV caches to `max_len` and stack (right-aligned).
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::attention::flash_attn;
    use candle_nn::VarMap;

    fn tiny_config() -> Config {
        let json = r#"{
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 32,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "attention_bias": false,
            "use_qk_norm": true,
            "tie_word_embeddings": true
        }"#;
        serde_json::from_str(json).expect("tiny_config parse")
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar::<f32>()
            .expect("to_scalar")
    }

    /// `new()` and `new_from_model_vb()` must be equivalent when given
    /// equivalent `VarBuilder` scoping — the split constructor is a pure
    /// refactor, not a behavior change.
    #[test]
    fn test_new_and_new_from_model_vb_equivalence() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let mut model_a = Qwen3Model::new(&cfg, vb.clone()).expect("new");
        let mut model_b = Qwen3Model::new_from_model_vb(&cfg, vb.pp("model"), vb)
            .expect("new_from_model_vb");

        let input_ids = Tensor::new(&[[1u32, 2, 3]], &device).expect("input_ids");
        let out_a = model_a.forward(&input_ids, 0).expect("forward a");
        let out_b = model_b.forward(&input_ids, 0).expect("forward b");

        assert_eq!(out_a.dims(), out_b.dims());
        assert!(max_abs_diff(&out_a, &out_b) < 1e-5);
    }

    /// `forward_embeds` fed the decoder's own token embeddings must match
    /// `forward` given the same token ids.
    #[test]
    fn test_forward_and_forward_embeds_equivalence() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = Qwen3Model::new(&cfg, vb).expect("new");

        let input_ids = Tensor::new(&[[1u32, 2, 3]], &device).expect("input_ids");
        let out_forward = model.forward(&input_ids, 0).expect("forward");
        model.clear_kv_cache();

        let embeds = model
            .embed_tokens()
            .forward(&input_ids)
            .expect("embed lookup")
            .to_dtype(model.model_dtype())
            .expect("dtype cast");
        let out_embeds = model.forward_embeds(&embeds, 0).expect("forward_embeds");

        assert_eq!(out_forward.dims(), out_embeds.dims());
        assert!(max_abs_diff(&out_forward, &out_embeds) < 1e-5);
    }

    /// The CPU `flash_attn` decode path must compute the same attention
    /// output (up to float32 rounding) as the naive 3-pass GQA softmax it
    /// replaced.
    #[test]
    fn test_flash_attn_decode_matches_naive_gqa_softmax() {
        let device = Device::Cpu;
        let (b, kv_heads, n_rep, head_dim, kv_len) = (1usize, 2usize, 2usize, 4usize, 5usize);
        let num_heads = kv_heads * n_rep;

        let q = (Tensor::arange(0f32, (b * num_heads * head_dim) as f32, &device)
            .unwrap()
            .reshape((b, num_heads, 1, head_dim))
            .unwrap()
            * 0.037)
            .unwrap();
        let k = (Tensor::arange(0f32, (b * kv_heads * kv_len * head_dim) as f32, &device)
            .unwrap()
            .reshape((b, kv_heads, kv_len, head_dim))
            .unwrap()
            * 0.021)
            .unwrap();
        let v = (Tensor::arange(0f32, (b * kv_heads * kv_len * head_dim) as f32, &device)
            .unwrap()
            .reshape((b, kv_heads, kv_len, head_dim))
            .unwrap()
            * 0.013)
            .unwrap();
        let mask = Tensor::zeros((b, 1, 1, kv_len), DType::F32, &device).unwrap();

        let scale = 1.0 / (head_dim as f64).sqrt();

        // Old algorithm: reshape Q into groups, 3-pass matmul/softmax/matmul.
        let q_g = (q
            .reshape((b, kv_heads, n_rep, head_dim))
            .unwrap()
            * scale)
            .unwrap();
        let k_t = k.transpose(2, 3).unwrap();
        let attn_weights = q_g.matmul(&k_t).unwrap();
        let attn_weights = attn_weights.broadcast_add(&mask).unwrap();
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights).unwrap();
        let naive_out = attn_weights.matmul(&v).unwrap();
        let naive_out = naive_out.reshape((b, num_heads, head_dim)).unwrap();

        // New algorithm: flash_attn on BSHD-transposed views.
        let q_bshd = q.transpose(1, 2).unwrap();
        let k_bshd = k.transpose(1, 2).unwrap();
        let v_bshd = v.transpose(1, 2).unwrap();
        // 1/sqrt(head_dim) is always small and positive; f64->f32 here
        // only drops precision flash_attn's own f32 accumulator would
        // discard anyway.
        #[allow(clippy::cast_possible_truncation)]
        let scale_f32 = scale as f32;
        let flash_out = flash_attn::<f32>(
            &q_bshd,
            &k_bshd,
            &v_bshd,
            scale_f32,
            AttnMask::None,
            None,
            None,
        )
        .unwrap();
        let flash_out = flash_out.reshape((b, num_heads, head_dim)).unwrap();

        assert!(max_abs_diff(&naive_out, &flash_out) < 1e-4);
    }

    /// The CPU `flash_attn` decode path's `AttnMask::Mask` branch (used
    /// when `attention_mask` is `Some`) must apply the mask the same way
    /// as the naive path's `broadcast_add` before softmax. This branch is
    /// unreachable from any current production caller (single-sequence
    /// decode never builds an explicit mask today), so it needs direct
    /// coverage here.
    #[test]
    fn test_flash_attn_decode_with_explicit_mask() {
        let device = Device::Cpu;
        let (b, kv_heads, n_rep, head_dim, kv_len) = (1usize, 2usize, 2usize, 4usize, 5usize);
        let num_heads = kv_heads * n_rep;

        let q = (Tensor::arange(0f32, (b * num_heads * head_dim) as f32, &device)
            .unwrap()
            .reshape((b, num_heads, 1, head_dim))
            .unwrap()
            * 0.037)
            .unwrap();
        let k = (Tensor::arange(0f32, (b * kv_heads * kv_len * head_dim) as f32, &device)
            .unwrap()
            .reshape((b, kv_heads, kv_len, head_dim))
            .unwrap()
            * 0.021)
            .unwrap();
        let v = (Tensor::arange(0f32, (b * kv_heads * kv_len * head_dim) as f32, &device)
            .unwrap()
            .reshape((b, kv_heads, kv_len, head_dim))
            .unwrap()
            * 0.013)
            .unwrap();

        // Non-trivial additive mask: mask out the first KV position,
        // matching `build_batch_decode_mask`'s -1e9/0.0 convention.
        let mut mask_data = vec![0f32; kv_len];
        mask_data[0] = -1e9;
        let mask = Tensor::from_vec(mask_data, (b, 1, 1, kv_len), &device).unwrap();

        let scale = 1.0 / (head_dim as f64).sqrt();

        // Naive algorithm with the mask applied.
        let q_g = (q
            .reshape((b, kv_heads, n_rep, head_dim))
            .unwrap()
            * scale)
            .unwrap();
        let k_t = k.transpose(2, 3).unwrap();
        let attn_weights = q_g.matmul(&k_t).unwrap();
        let attn_weights = attn_weights.broadcast_add(&mask).unwrap();
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights).unwrap();
        let naive_out = attn_weights.matmul(&v).unwrap();
        let naive_out = naive_out.reshape((b, num_heads, head_dim)).unwrap();

        // flash_attn with the same mask via AttnMask::Mask.
        let q_bshd = q.transpose(1, 2).unwrap();
        let k_bshd = k.transpose(1, 2).unwrap();
        let v_bshd = v.transpose(1, 2).unwrap();
        // 1/sqrt(head_dim) is always small and positive; f64->f32 here
        // only drops precision flash_attn's own f32 accumulator would
        // discard anyway.
        #[allow(clippy::cast_possible_truncation)]
        let scale_f32 = scale as f32;
        let flash_out = flash_attn::<f32>(
            &q_bshd,
            &k_bshd,
            &v_bshd,
            scale_f32,
            AttnMask::Mask(mask),
            None,
            None,
        )
        .unwrap();
        let flash_out = flash_out.reshape((b, num_heads, head_dim)).unwrap();

        assert!(max_abs_diff(&naive_out, &flash_out) < 1e-4);
    }

    /// End-to-end check that the flash_attn decode fast path is wired
    /// correctly inside `Attention::forward` (not just numerically
    /// equivalent in isolation): a prefill followed by a single-token
    /// decode must be deterministic and reproducible on the same model.
    #[test]
    fn test_flash_attn_decode_via_model_forward() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = Qwen3Model::new(&cfg, vb).expect("new");

        let prefill_ids = Tensor::new(&[[1u32, 2, 3]], &device).expect("prefill_ids");
        let decode_ids = Tensor::new(&[[4u32]], &device).expect("decode_ids");

        model.forward(&prefill_ids, 0).expect("prefill");
        // seq_len == 1, b_sz == 1, CPU: hits the flash_attn decode path.
        let out_a = model.forward(&decode_ids, 3).expect("decode a");

        model.clear_kv_cache();
        model.forward(&prefill_ids, 0).expect("prefill again");
        let out_b = model.forward(&decode_ids, 3).expect("decode b");

        assert_eq!(out_a.dims(), out_b.dims());
        assert!(max_abs_diff(&out_a, &out_b) < 1e-5);
    }

    /// The CPU `flash_attn` prefill path (seq_len > 1) must match the naive
    /// GQA-expand/matmul/softmax/matmul path, for both a first prefill
    /// (kv_offset=0) and a continued prefill (kv_offset>0). The naive
    /// reference always runs in F32 (candle's CPU backend doesn't support
    /// `matmul` for BF16); `dtype` only controls what `dispatch_flash_attn`
    /// runs in, so the tolerance reflects that dtype's precision loss.
    fn check_flash_attn_prefill_matches_naive_sdpa(dtype: DType, tol: f32) {
        let device = Device::Cpu;
        let (b, kv_heads, n_rep, head_dim) = (1usize, 2usize, 2usize, 4usize);
        let num_heads = kv_heads * n_rep;
        let seq_len = 3usize;

        for kv_offset in [0usize, 2usize] {
            let kv_len = kv_offset + seq_len;

            let q = (Tensor::arange(0f32, (b * num_heads * seq_len * head_dim) as f32, &device)
                .unwrap()
                .reshape((b, num_heads, seq_len, head_dim))
                .unwrap()
                * 0.037)
                .unwrap();
            let k = (Tensor::arange(0f32, (b * kv_heads * kv_len * head_dim) as f32, &device)
                .unwrap()
                .reshape((b, kv_heads, kv_len, head_dim))
                .unwrap()
                * 0.021)
                .unwrap();
            let v = (Tensor::arange(0f32, (b * kv_heads * kv_len * head_dim) as f32, &device)
                .unwrap()
                .reshape((b, kv_heads, kv_len, head_dim))
                .unwrap()
                * 0.013)
                .unwrap();

            let scale = 1.0 / (head_dim as f64).sqrt();

            // Naive algorithm (always F32): GQA-expand K/V, explicit causal
            // mask, 3-pass SDPA.
            let k_exp = k
                .unsqueeze(2)
                .unwrap()
                .expand((b, kv_heads, n_rep, kv_len, head_dim))
                .unwrap()
                .reshape((b, num_heads, kv_len, head_dim))
                .unwrap();
            let v_exp = v
                .unsqueeze(2)
                .unwrap()
                .expand((b, kv_heads, n_rep, kv_len, head_dim))
                .unwrap()
                .reshape((b, num_heads, kv_len, head_dim))
                .unwrap();

            let mut mask_data = vec![0f32; seq_len * kv_len];
            for i in 0..seq_len {
                for j in 0..kv_len {
                    if j > kv_offset + i {
                        mask_data[i * kv_len + j] = -1e9;
                    }
                }
            }
            let mask = Tensor::from_vec(mask_data, (1, 1, seq_len, kv_len), &device).unwrap();

            let attn_weights = (q.matmul(&k_exp.transpose(2, 3).unwrap()).unwrap() * scale)
                .unwrap();
            let attn_weights = attn_weights.broadcast_add(&mask).unwrap();
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights).unwrap();
            let naive_out = attn_weights.matmul(&v_exp).unwrap();

            // flash_attn algorithm, cast to `dtype`: BSHD + AttnMask::Causal.
            let q_bshd = q.transpose(1, 2).unwrap().to_dtype(dtype).unwrap();
            let k_bshd = k.transpose(1, 2).unwrap().to_dtype(dtype).unwrap();
            let v_bshd = v.transpose(1, 2).unwrap().to_dtype(dtype).unwrap();
            // 1/sqrt(head_dim) is always small and positive; f64->f32 here
            // only drops precision flash_attn's own f32 accumulator would
            // discard anyway.
            #[allow(clippy::cast_possible_truncation)]
            let scale_f32 = scale as f32;
            let flash_out = dispatch_flash_attn(
                &q_bshd,
                &k_bshd,
                &v_bshd,
                scale_f32,
                AttnMask::Causal { kv_offset },
            )
            .unwrap();

            let flash_out_f32 = flash_out.to_dtype(DType::F32).unwrap();
            assert!(
                max_abs_diff(&naive_out, &flash_out_f32) < tol,
                "mismatch at dtype={dtype:?}, kv_offset={kv_offset}",
            );
        }
    }

    #[test]
    fn test_flash_attn_prefill_matches_naive_sdpa() {
        check_flash_attn_prefill_matches_naive_sdpa(DType::F32, 1e-4);
    }

    #[test]
    fn test_flash_attn_prefill_matches_naive_sdpa_bf16() {
        check_flash_attn_prefill_matches_naive_sdpa(DType::BF16, 5e-2);
    }

    #[test]
    fn test_flash_attn_prefill_matches_naive_sdpa_f16() {
        check_flash_attn_prefill_matches_naive_sdpa(DType::F16, 5e-2);
    }

    /// End-to-end check that the flash_attn prefill fast path is wired
    /// correctly inside `Attention::forward`: a multi-token prefill must be
    /// deterministic and reproducible on the same model.
    #[test]
    fn test_flash_attn_prefill_via_model_forward() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = Qwen3Model::new(&cfg, vb).expect("new");

        let prefill_ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device).expect("prefill_ids");

        // seq_len > 1, b_sz == 1, CPU: hits the flash_attn prefill path.
        let out_a = model.forward(&prefill_ids, 0).expect("prefill a");

        model.clear_kv_cache();
        let out_b = model.forward(&prefill_ids, 0).expect("prefill b");

        assert_eq!(out_a.dims(), out_b.dims());
        assert!(max_abs_diff(&out_a, &out_b) < 1e-5);
    }

    /// Chunked prefill (two smaller prefills) must produce the same decode
    /// output as a single large prefill — exercises the flash_attn prefill
    /// path's `kv_offset > 0` case through the full model.
    #[test]
    fn test_flash_attn_chunked_prefill_matches_single() {
        let cfg = tiny_config();
        let device = Device::Cpu;

        // Both models are built from the same VarMap so `VarMap::get`
        // returns the already-initialized tensors on the second build,
        // giving identical weights (including each model's own
        // once-per-construction merged `qkv_proj`, which is a fresh
        // concatenation `Var::set` can't retroactively update).
        let varmap = VarMap::new();
        let mut model_single =
            Qwen3Model::new(&cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device))
                .expect("new single");
        let mut model_chunked =
            Qwen3Model::new(&cfg, VarBuilder::from_varmap(&varmap, DType::F32, &device))
                .expect("new chunked");

        let decode_id = Tensor::new(&[[6u32]], &device).expect("decode_id");

        // Single prefill.
        let single_ids = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device).expect("single_ids");
        model_single.forward(&single_ids, 0).expect("single prefill");
        let out_single = model_single.forward(&decode_id, 5).expect("single decode");

        // Chunked prefill: two prefill calls, then decode.
        let chunk_a = Tensor::new(&[[1u32, 2, 3]], &device).expect("chunk_a");
        let chunk_b = Tensor::new(&[[4u32, 5]], &device).expect("chunk_b");
        model_chunked.forward(&chunk_a, 0).expect("chunk a prefill");
        model_chunked.forward(&chunk_b, 3).expect("chunk b prefill");
        let out_chunked = model_chunked.forward(&decode_id, 5).expect("chunked decode");

        assert_eq!(out_single.dims(), out_chunked.dims());
        assert!(max_abs_diff(&out_single, &out_chunked) < 1e-4);
    }
}
