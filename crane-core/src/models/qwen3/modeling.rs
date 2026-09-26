//! Optimized Qwen3 transformer implementation.
//!
//! Adapted from the `HunyuanDense` model with full parity on all optimizations:
//!
//! 1. **Pre-allocated KV cache** with in-place `slice_set` writes
//!    — `O(new_seq_len)` per decode step instead of `O(cache_len)` `Tensor::cat`.
//! 2. **Fused flash attention for decode and prefill** (CPU, B=1)
//!    — Uses `candle_nn::attention::flash_attn`'s online-softmax kernel
//!    (`O(head_dim)` working set, native GQA) instead of materializing an
//!    `O(context_len)` scores tensor. Prefill additionally skips the
//!    GQA K/V expansion (which duplicates `K/V` `n_rep` times) and uses
//!    `AttnMask::Causal` so masking is done via loop bounds, not a
//!    materialized mask tensor. Falls back to a GQA-grouped matmul SDPA
//!    on GPU or for batched (B>1) decode, where cuBLAS is already
//!    compute-bound or an explicit per-sequence mask is required, and to
//!    a standard SDPA for GPU or batched (B>1) prefill, or when
//!    `num_heads` == `num_kv_heads` (no GQA grouping needed).
//! 3. **Fused `RoPE` kernel** via `candle_nn::rotary_emb::rope_thd()`
//!    — One CUDA launch per Q/K instead of 5 manual tensor ops.
//!    — Applied in BSHD layout (before the transpose to BHSD), so the
//!    reshape output is already contiguous — no `contiguous()` copy
//!    needed before `RoPE`, and QK norm hits the fast fused `RmsNorm` path.
//!    — Precomputed `[max_pos, head_dim/2]` cos/sin tables (half-width, as
//!    required by the `rope_thd()` API).
//! 4. **GGUF quantization** via the polymorphic `LinearLayer` enum
//!    — Same model code serves both safetensors (f16/f32/bf16) and GGUF weights.
//! 5. **Batched decode infrastructure**
//!    — `setup_batch_decode`, `step_batch_decode`, `extract_batch_kv` enable
//!    GPU-efficient concurrent sequence serving in the engine.
//! 6. **KV cache save/restore**
//!    — `get_kv_caches` / `set_kv_caches` for continuous-batching context swap.
//! 7. **Fused SiLU-mul MLP gate**
//!    — `fused_silu_mul` replaces the `narrow + silu + mul` op chain in each
//!    MLP block, reducing kernel launches and intermediate allocations.
//! 8. **Merged QKV / gate+up projections**
//!    — Q, K, V weights fused into one matmul; gate and up weights fused into
//!    one matmul — halves the number of linear-layer dispatches per layer.

use candle_core::quantized::gguf_file;
use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::attention::AttnMask;
use candle_nn::rotary_emb::rope_thd;
use candle_nn::{Linear, RmsNorm, VarBuilder, linear_no_bias};
use serde::Deserialize;
use std::io::{Read, Seek};

use crate::device::{DeviceAssignment, format_budget, greedy_fit_layers, query_gpu_memory};
use crate::models::modules::embedding::EmbeddingLayer;
use crate::models::modules::flash_attn::dispatch_flash_attn;
use crate::models::modules::kv_cache;
use crate::models::modules::moe::{MlpOrMoe, MoeConfig, SparseMoeBlock};
use crate::models::modules::rotary::RotaryEmbedding;
use crate::utils::DeviceExt;
use ribo::utils::log;

// Reuse the polymorphic linear layer and the shared GGUF loader.
pub use crate::ops::linear::LinearLayer;
pub use crate::quantized::gguf_file::Gguf;

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

// Field names and independent bool toggles mirror the HuggingFace
// config.json schema; grouping them would break Deserialize's field mapping.
#[allow(clippy::struct_excessive_bools)]
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
    /// Total number of experts per `MoE` layer. `None` for dense checkpoints.
    #[serde(default)]
    pub num_experts: Option<usize>,
    /// Number of experts activated per token (top-K). `None` for dense checkpoints.
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    /// Hidden dimension of each expert's feed-forward network.
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    /// Whether to renormalize the top-K routing weights to sum to 1.
    #[serde(default)]
    pub norm_topk_prob: Option<bool>,
    /// Every Nth layer is `MoE`; the rest stay dense MLP.
    #[serde(default)]
    pub decoder_sparse_step: Option<usize>,
}

impl Config {
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Bytes of KV cache one sequence consumes per generated token, summed
    /// across every layer (standard full-attention: one K and one V tensor
    /// per layer, no sharing).
    #[must_use]
    pub fn kv_bytes_per_token(&self, dtype_bytes: usize) -> u64 {
        2 * self.num_hidden_layers as u64
            * self.num_key_value_heads as u64
            * self.head_dim() as u64
            * dtype_bytes as u64
    }

    /// Builds the `MoE` configuration for this model, or `None` if this is a
    /// dense (non-`MoE`) checkpoint.
    #[must_use]
    pub fn moe_config(&self) -> Option<MoeConfig> {
        Some(MoeConfig {
            num_experts: self.num_experts?,
            num_experts_per_tok: self.num_experts_per_tok?,
            moe_intermediate_size: self.moe_intermediate_size?,
            norm_topk_prob: self.norm_topk_prob.unwrap_or(true),
            decoder_sparse_step: self.decoder_sparse_step,
        })
    }
}

// ── Attention ───────────────────────────────────────────────────────────

struct Attention {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    o_proj: LinearLayer,
    /// Merged QKV weight [`q_dim` + 2*`kv_dim`, `hidden_size`] — one gemv instead of 3.
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
    // `VarBuilder` is conventionally passed by value throughout this
    // codebase (its `pp`/`device`/`dtype` accessors take `&self` and are
    // cheap to call repeatedly); matching that convention here.
    #[allow(clippy::needless_pass_by_value)]
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

    // q/k/v/b/h/s/d are standard ML tensor-shape notation (query, key,
    // value, batch, heads, seq_len, head_dim), matching the Q/K/V and BHSD
    // terminology already used throughout this function's comments.
    #[allow(clippy::many_single_char_names)]
    // This function's length comes from four densely-commented fast-path
    // branches (CPU flash-attn decode, CPU flash-attn prefill, GQA-grouped
    // SDPA decode, standard SDPA); splitting it up would scatter that
    // rationale across several small functions without simplifying the
    // control flow itself.
    #[allow(clippy::too_many_lines)]
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
        // head_dim is a small model hyperparameter (e.g. <= a few hundred),
        // far below f64's 52-bit mantissa limit.
        #[allow(clippy::cast_precision_loss)]
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
            // Cast back from F32 before `o_proj` — see `dispatch_flash_attn`'s doc.
            let attn_output = attn_output.to_dtype(q.dtype())?;

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
            // Cast back from F32 before `o_proj` — see `dispatch_flash_attn`'s doc.
            let attn_output = attn_output.to_dtype(q.dtype())?;

            // flash_attn output is BHSD [B, H, S, D] → [B, S, H*D]
            let attn_output =
                attn_output
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
            let q_g = (q.reshape((b_sz, self.num_kv_heads, n_rep, self.head_dim))? * scale)?;

            // K^T: [B, kv_heads, D, S] — just a view (0 copies here;
            //       matmul will flatten+contiguous in one pass).
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

// ── MLP ─────────────────────────────────────────────────────────────────

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
    // See `Attention::new`'s comment on `VarBuilder` by-value.
    #[allow(clippy::needless_pass_by_value)]
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

    fn new_from_gguf<R: Read + Seek>(
        gg: &mut Gguf<R>,
        layer_idx: usize,
        _intermediate_size: usize,
    ) -> Result<Self> {
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

                // Use the fused GPU kernel when available: eliminates
                // narrow + silu + mul (3 kernel launches → 1).
                #[cfg(any(feature = "cuda", feature = "rocm", feature = "sycl"))]
                {
                    if gu.device().is_cuda() || gu.device().is_rocm() || gu.device().is_sycl() {
                        let activated =
                            crate::ops::fused_silu_mul(&gu.contiguous()?, *intermediate_size)?;
                        return self.down_proj.forward(&activated);
                    }
                }

                // CPU / non-CUDA fallback
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

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Self::forward(self, xs)
    }
}

// ── Decoder Layer ───────────────────────────────────────────────────────

struct DecoderLayer {
    self_attn: Attention,
    mlp: MlpOrMoe<Mlp>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    // See `Attention::new`'s comment on `VarBuilder` by-value.
    #[allow(clippy::needless_pass_by_value)]
    fn new(
        config: &Config,
        layer_idx: usize,
        vb: VarBuilder,
        expert_device: &Device,
    ) -> Result<Self> {
        let self_attn = Attention::new(config, vb.pp("self_attn"))?;
        let moe_config = config.moe_config();
        // HF's `mlp_only_layers` override (per-layer dense exceptions) isn't
        // modeled here; only the uniform `decoder_sparse_step` stride is.
        let is_moe_layer = moe_config.as_ref().is_some_and(|mc| {
            mc.decoder_sparse_step
                .is_none_or(|step| (layer_idx + 1).is_multiple_of(step))
        });
        let mlp = match moe_config {
            Some(mc) if is_moe_layer => MlpOrMoe::Moe(SparseMoeBlock::new(
                &mc,
                config.hidden_size,
                vb.pp("mlp"),
                expert_device,
            )?),
            _ => MlpOrMoe::Dense(Mlp::new(config, vb.pp("mlp"))?),
        };
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
        expert_device: &Device,
    ) -> Result<Self> {
        let self_attn = Attention::new_from_gguf(config, gg, layer_idx)?;
        let is_moe = gg.contains_tensor(&format!("blk.{layer_idx}.ffn_gate_inp.weight"));
        let mlp = if is_moe {
            let moe_config = config.moe_config().ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "layer {layer_idx} has MoE tensors but Config lacks MoE fields"
                ))
                .bt()
            })?;
            MlpOrMoe::Moe(SparseMoeBlock::new_from_gguf(
                &moe_config,
                gg,
                layer_idx,
                expert_device,
            )?)
        } else {
            MlpOrMoe::Dense(Mlp::new_from_gguf(gg, layer_idx, config.intermediate_size)?)
        };
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
        let hidden_states = self.forward_attn(hidden_states, cos, sin, attention_mask)?;
        self.forward_mlp(&hidden_states)
    }

    /// Attention half: input layernorm, self-attention, residual add.
    ///
    /// Split out from [`Self::forward`] so [`Qwen3Model::decode`] can prune
    /// hidden states to only the output-needing positions between this and
    /// [`Self::forward_mlp`] on the last layer, saving `MoE` expert
    /// dispatches on positions whose hidden states would otherwise be
    /// discarded before `lm_head`. Self-attention must still see every
    /// position (for KV cache correctness), so only the MLP half is
    /// prunable.
    fn forward_attn(
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
        residual + hidden_states
    }

    /// MLP half: post-attention layernorm, dense/`MoE` MLP, residual add.
    /// See [`Self::forward_attn`] for why this is split out.
    fn forward_mlp(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states;
        let hidden_states = self.post_attention_layernorm.forward(hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        residual + hidden_states
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ── Full Model ──────────────────────────────────────────────────────────

pub struct Qwen3Model {
    embed_tokens: EmbeddingLayer,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: LinearLayer,
    rotary_emb: RotaryEmbedding,
    config: Config,
    dtype: DType,
    /// Post-norm hidden states from the most recent forward call.
    ///
    /// Shape is `[B, S, H]` in the common case, but `[B, 1, H]` (last
    /// position only) when the last decoder layer is `MoE` and `seq_len > 1`,
    /// because the `MoE` pruning optimization narrows hidden states before
    /// the final MLP. See [`Self::last_hidden_states`].
    last_hidden_states: Option<Tensor>,
    /// Whether the last decoder layer uses a Mixture-of-Experts MLP,
    /// cached at construction to avoid re-checking on every decode call.
    last_layer_is_moe: bool,
}

/// `MoE` expert metadata read from GGUF, all `None` for dense
/// (non-`MoE`) checkpoints.
struct GgufMoeMetadata {
    num_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    moe_intermediate_size: Option<usize>,
    norm_topk_prob: Option<bool>,
}

/// Reads `MoE` expert metadata from GGUF, if present.
fn read_moe_metadata<R: Read + Seek>(gg: &Gguf<R>, arch: &str) -> GgufMoeMetadata {
    let num_experts = gg
        .metadata()
        .get(&format!("{arch}.expert_count"))
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize)
        // Some dense GGUF exports write an explicit `expert_count = 0`
        // rather than omitting the key; treat that the same as absent.
        .filter(|&n| n > 0);
    let num_experts_per_tok = gg
        .metadata()
        .get(&format!("{arch}.expert_used_count"))
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize);
    let moe_intermediate_size = gg
        .metadata()
        .get(&format!("{arch}.expert_feed_forward_length"))
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize);
    let expert_shared_ffn_length = gg
        .metadata()
        .get(&format!("{arch}.expert_shared_feed_forward_length"))
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize);
    // Qwen3 MoE has no shared experts, so an absent or zero shared-FFN
    // length means the top-K routing weights should be renormalized.
    let norm_topk_prob = num_experts
        .map(|_| expert_shared_ffn_length.is_none() || expert_shared_ffn_length == Some(0));
    GgufMoeMetadata {
        num_experts,
        num_experts_per_tok,
        moe_intermediate_size,
        norm_topk_prob,
    }
}

impl Qwen3Model {
    /// Construct from safetensors / `HuggingFace` checkpoint. Expert weights
    /// (if `MoE`) go on the same device as the rest of the model; use
    /// [`Self::new_with_expert_device`] to place them elsewhere.
    ///
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing or has an
    /// unexpected shape.
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        Self::new_inner(config, vb.pp("model"), vb, &device)
    }

    /// Like [`Self::new`], but places `MoE` expert weights on `expert_device`
    /// rather than the model's main device.
    ///
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing or has an
    /// unexpected shape.
    pub fn new_with_expert_device(
        config: &Config,
        vb: VarBuilder,
        expert_device: &Device,
    ) -> Result<Self> {
        Self::new_inner(config, vb.pp("model"), vb, expert_device)
    }

    /// Construct from a checkpoint where the decoder is nested under a
    /// deeper prefix than the standard `model.*` layout (e.g. Qwen3-ASR's
    /// `model.language_model.*`). `model_vb` must already be scoped to the
    /// decoder's root (what would otherwise be `vb.pp("model")`); `root_vb`
    /// is the checkpoint root, used to resolve an untied `lm_head` sibling.
    /// Expert weights (if `MoE`) go on the same device as `model_vb`; use
    /// [`Self::new_from_model_vb_with_expert_device`] to place them
    /// elsewhere.
    ///
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing or has an
    /// unexpected shape.
    pub fn new_from_model_vb(
        config: &Config,
        model_vb: VarBuilder,
        root_vb: VarBuilder,
    ) -> Result<Self> {
        let device = model_vb.device().clone();
        Self::new_inner(config, model_vb, root_vb, &device)
    }

    /// Like [`Self::new_from_model_vb`], but places `MoE` expert weights on
    /// `expert_device` rather than `model_vb`'s device.
    ///
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing or has an
    /// unexpected shape.
    pub fn new_from_model_vb_with_expert_device(
        config: &Config,
        model_vb: VarBuilder,
        root_vb: VarBuilder,
        expert_device: &Device,
    ) -> Result<Self> {
        Self::new_inner(config, model_vb, root_vb, expert_device)
    }

    // See `Attention::new`'s comment on `VarBuilder` by-value.
    #[allow(clippy::needless_pass_by_value)]
    fn new_inner(
        config: &Config,
        model_vb: VarBuilder,
        root_vb: VarBuilder,
        expert_device: &Device,
    ) -> Result<Self> {
        let dtype = model_vb.dtype();
        let embed_tokens = EmbeddingLayer::Dense(candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            model_vb.pp("embed_tokens"),
        )?);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let layers_vb = model_vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            layers.push(DecoderLayer::new(
                config,
                i,
                layers_vb.pp(i),
                expert_device,
            )?);
        }

        let norm =
            candle_nn::rms_norm(config.hidden_size, config.rms_norm_eps, model_vb.pp("norm"))?;

        // Pre-stored in F32 only when the compute dtype is F16: raw logits
        // over a 100k+ vocab routinely exceed F16's 65504 max, and
        // pre-converting avoids a per-token cast of the weight at the
        // `LinearLayer::forward_logits` call site. BF16/F32 share F32's
        // exponent range and can't overflow, so they stay native -- BF16 in
        // particular must stay BF16 for the CUDA `gpu_argmax` sampling fast
        // path, which only accepts BF16 logits.
        let lm_head_dtype = if dtype == DType::F16 {
            DType::F32
        } else {
            dtype
        };
        let lm_head = if config.tie_word_embeddings {
            embed_tokens.tied_output_upcast_f16(dtype)?
        } else {
            LinearLayer::Standard(Linear::new(
                linear_no_bias(config.hidden_size, config.vocab_size, root_vb.pp("lm_head"))?
                    .weight()
                    .to_dtype(lm_head_dtype)?,
                None,
            ))
        };

        let rotary_emb = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            model_vb.device(),
        )?;

        let last_layer_is_moe = layers
            .last()
            .is_some_and(|l| matches!(l.mlp, MlpOrMoe::Moe(_)));

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            config: config.clone(),
            dtype,
            last_hidden_states: None,
            last_layer_is_moe,
        })
    }

    /// Construct from a GGUF file.
    ///
    /// `devices.main` holds every weight but `MoE` experts; `devices.expert`
    /// holds `MoE` expert weights, if the checkpoint is `MoE`.
    ///
    /// # Errors
    ///
    /// Returns an error if a required tensor or metadata entry is missing
    /// or has an unexpected shape.
    // This function's length comes from reading many independent GGUF
    // metadata keys (attention, RoPE, MoE) into `Config` one field at a
    // time; splitting it up would scatter that flat read-and-assign
    // sequence across several small functions without simplifying it.
    #[allow(clippy::too_many_lines)]
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        devices: impl Into<DeviceAssignment>,
    ) -> Result<Self> {
        let devices = devices.into();
        let device = &devices.main;
        let dtype = if device.is_cuda() {
            DType::BF16
        } else if device.is_metal() || device.is_rocm() || device.is_sycl() {
            DType::F16
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
            .cloned()
            .unwrap_or_else(|| "qwen3".to_string());

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
        let rms_norm_eps = f64::from(
            gg.metadata()
                .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(1e-6),
        );
        let rope_theta = f64::from(
            gg.metadata()
                .get(&format!("{arch}.rope.freq_base"))
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(1_000_000.0),
        );

        let moe_meta = read_moe_metadata(&gg, &arch);

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
            num_experts: moe_meta.num_experts,
            num_experts_per_tok: moe_meta.num_experts_per_tok,
            moe_intermediate_size: moe_meta.moe_intermediate_size,
            norm_topk_prob: moe_meta.norm_topk_prob,
            // Not a standard GGUF metadata key; the GGUF layer-construction
            // path detects MoE-vs-dense per layer by tensor presence instead.
            decoder_sparse_step: None,
        };

        let embed_tokens = gg.quantized_embedding("token_embd.weight", hidden_size)?;
        let actual_vocab_size = embed_tokens.vocab_size();
        let config = Config {
            vocab_size: actual_vocab_size,
            ..config
        };

        let mut layers = Vec::with_capacity(num_hidden_layers);
        for i in 0..num_hidden_layers {
            layers.push(DecoderLayer::new_from_gguf(
                &config,
                &mut gg,
                i,
                &devices.expert,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;

        let lm_head = if tie_word_embeddings {
            embed_tokens.tied_output_upcast_f16(dtype)?
        } else {
            gg.linear("output.weight")?
        };

        let rotary_emb = RotaryEmbedding::new(
            config.head_dim(),
            config.max_position_embeddings,
            config.rope_theta,
            device,
        )?;

        let last_layer_is_moe = layers
            .last()
            .is_some_and(|l| matches!(l.mlp, MlpOrMoe::Moe(_)));

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            config,
            dtype,
            last_hidden_states: None,
            last_layer_is_moe,
        })
    }

    /// Runs a probe forward pass to force GPU backends' lazy first-use
    /// library initialization (rocBLAS/hipRAND/JIT-compiled kernels), then
    /// live-queries free VRAM and greedily promotes CPU-placed `MoE` expert
    /// layers to `main_device` up to `vram_ceiling_bytes` (minus
    /// `runtime_reservation_bytes`). No-op for non-`MoE` checkpoints.
    ///
    /// A static pre-load VRAM estimate consistently undershoots real usage:
    /// rocBLAS/hipRAND/CUDA JIT lazily initialize on first use, so the real
    /// number is only visible once a forward pass actually runs. This is
    /// why placement is decided here, after construction, rather than
    /// while loading.
    ///
    /// # Errors
    ///
    /// Returns an error only if a tensor op unrelated to the promotion
    /// itself fails; an out-of-memory promotion attempt is caught and
    /// logged instead — remaining layers just stay on CPU.
    // One sequential pipeline (probe -> cost estimate -> budget -> promote)
    // sharing local state (`gpu_location`, `layer_costs`) throughout;
    // splitting it up would scatter that shared context across several
    // small functions without simplifying the control flow itself.
    #[allow(clippy::too_many_lines)]
    pub fn promote_experts_to_gpu(
        &mut self,
        main_device: &Device,
        vram_ceiling_bytes: u64,
        runtime_reservation_bytes: u64,
    ) -> Result<()> {
        let Some(moe_config) = self.config.moe_config() else {
            return Ok(());
        };
        if matches!(main_device, Device::Cpu) {
            return Ok(());
        }
        let gpu_location = main_device.location();
        log::info!(
            "Expert placement: running probe forward pass + live VRAM query on {gpu_location:?} \
             before deciding MoE GPU/CPU split (may take a few seconds)"
        );
        let probe_ids = Tensor::new(&[45u32, 546, 456], main_device)?.unsqueeze(0)?;
        if let Err(e) = self.forward(&probe_ids, 0) {
            self.clear_kv_cache();
            log::warn!(
                "expert-placement probe forward failed on {gpu_location:?} (non-fatal, all \
                 experts stay on CPU): {e}"
            );
            return Ok(());
        }
        self.clear_kv_cache();

        let is_moe_layer: Vec<bool> = self
            .layers
            .iter()
            .map(|l| matches!(l.mlp, MlpOrMoe::Moe(_)))
            .collect();
        let total_moe_layers = is_moe_layer.iter().filter(|&&m| m).count();
        // Every MoE layer in a Qwen3 checkpoint has identical expert-tensor
        // shapes (3 projections per expert: gate, up, down), so this cost
        // is uniform across MoE layers; non-MoE layers cost 0 so they never
        // affect the greedy budget below.
        let per_layer_cost = moe_config.num_experts as u64
            * moe_config.moe_intermediate_size as u64
            * self.config.hidden_size as u64
            * 3
            * self.dtype.size_in_bytes() as u64;
        // Layers whose experts already live on `gpu_location` (e.g. a
        // re-promotion pass after `max_seq_len` shrinks and frees headroom)
        // cost 0: they're already promoted, so charging them again would
        // waste budget on a no-op and starve layers still on CPU.
        let layer_costs: Vec<u64> = self
            .layers
            .iter()
            .map(|l| match &l.mlp {
                MlpOrMoe::Moe(block) if block.expert_device().location() != gpu_location => {
                    per_layer_cost
                },
                _ => 0,
            })
            .collect();
        log::debug!(
            "MoE layout: {total_moe_layers} layers, per-layer expert cost estimate={}",
            format_budget(per_layer_cost),
        );

        let Some((free, total)) = query_gpu_memory(main_device) else {
            log::warn!(
                "No live VRAM query available for {gpu_location:?}; skipping expert promotion \
                 (all experts stay on CPU)"
            );
            return Ok(());
        };
        let used = total.saturating_sub(free);
        let ceiling = vram_ceiling_bytes.min(total);
        let remaining = ceiling.saturating_sub(used);
        let available = remaining.saturating_sub(runtime_reservation_bytes);
        log::info!(
            "Live VRAM on {gpu_location:?} after probe: free={}, total={}, used={}, \
             available_for_experts={} (configured limit={})",
            format_budget(free),
            format_budget(total),
            format_budget(used),
            format_budget(available),
            format_budget(vram_ceiling_bytes),
        );

        let promoted: std::collections::HashSet<usize> = greedy_fit_layers(&layer_costs, available)
            .into_iter()
            .filter(|&i| is_moe_layer[i])
            .collect();
        log::debug!(
            "Attempting promotion of {} of {total_moe_layers} MoE layers to {gpu_location:?}: {:?}",
            promoted.len(),
            {
                let mut sorted: Vec<usize> = promoted.iter().copied().collect();
                sorted.sort_unstable();
                sorted
            },
        );

        // The greedy budget above is a heuristic upper bound on what to
        // *attempt* — allocator fragmentation and per-expert allocation
        // overhead mean actual usage can still exceed it even though
        // `promote_experts_to` is itself atomic per layer. A failed
        // promotion here (e.g. real GPU out-of-memory) must not abort
        // model load: stop promoting further layers and leave the rest on
        // CPU — degraded, not fatal.
        let mut gpu_layers = 0usize;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if !promoted.contains(&i) {
                continue;
            }
            let MlpOrMoe::Moe(block) = &mut layer.mlp else {
                continue;
            };
            match block.promote_experts_to(main_device, self.dtype) {
                Ok(()) => {
                    gpu_layers += 1;
                    log::debug!(
                        "layer {i}: promoted to {gpu_location:?} (cost={})",
                        format_budget(layer_costs[i]),
                    );
                },
                Err(e) => {
                    log::warn!(
                        "expert promotion stopped at layer {i} on {gpu_location:?} (device \
                         allocation failed, this and remaining layers stay on CPU): {e}"
                    );
                    break;
                },
            }
        }

        log::info!(
            "Expert placement: {gpu_layers}/{total_moe_layers} MoE layers on {gpu_location:?}, \
             {} on CPU",
            total_moe_layers - gpu_layers,
        );
        Ok(())
    }

    // ── Forward ─────────────────────────────────────────────────────────

    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;

        // Disable event tracking for the duration of the forward pass.
        // Crane uses a single CUDA stream; the per-tensor CudaEvents are
        // unnecessary and cost ~2×cuEventCreate+cuEventRecord per temp tensor.
        #[cfg(feature = "cuda")]
        let _event_guard = EventTrackingGuard::disable(input_ids.device());

        // Outermost pass boundary for `CRANE_PROF=1`: covers the whole
        // forward (embedding lookup through `decode`), mirroring
        // `qwen3_5::prefill::forward`'s use of the same timer.
        let timer = crate::utils::prof::pass(seq_len, input_ids.device());
        let hidden_states = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;
        let out = self.decode(hidden_states, seq_len, start_pos, input_ids.device());
        if let Some(timer) = timer {
            timer.finish(input_ids.device());
        }
        out
    }

    /// Same as [`Self::forward`], but starting from a caller-supplied
    /// embedding sequence instead of doing the token embedding lookup
    /// internally. Used by callers (e.g. Qwen3-ASR) that splice in
    /// non-text embeddings (audio, etc.) at specific positions before
    /// running the decoder.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
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

        let timer = crate::utils::prof::pass(seq_len, inputs_embeds.device());
        let hidden_states = inputs_embeds.to_dtype(self.dtype)?;
        let out = self.decode(hidden_states, seq_len, start_pos, inputs_embeds.device());
        if let Some(timer) = timer {
            timer.finish(inputs_embeds.device());
        }
        out
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
            // Multiply in F32, cast to `self.dtype` last: candle's affine op
            // converts the -1e9 scalar to the tensor's own dtype *before*
            // multiplying, so doing this in F16 turns -1e9 into literal
            // -Inf and then corrupts every *unmasked* (0.0) position to NaN
            // via 0.0 * -Inf. F32 keeps -1e9 finite through the multiply;
            // only the final cast may turn masked positions into -Inf
            // (which softmax handles correctly via max-subtraction).
            let mask = mask
                .broadcast_lt(&Tensor::new(0.5f32, device)?)?
                .to_dtype(DType::F32)?;
            let mask = (mask * (-1e9f64))?.to_dtype(self.dtype)?;
            Some(mask.unsqueeze(0)?.unsqueeze(0)?)
        } else {
            None
        };

        // Only the last position's hidden state feeds `lm_head` below, but
        // every position must still pass through every layer's
        // self-attention (for KV cache correctness). When the last layer is
        // MoE, its expert dispatch is otherwise wasted on the `seq_len - 1`
        // positions this method discards anyway: prune to the last position
        // right after that layer's attention, before its MoE MLP runs.
        // Skipped for single-token decode (`seq_len == 1`, nothing to prune)
        // and for a dense last layer (no MoE dispatch cost to save, and
        // preserves full-sequence `last_hidden_states` for callers like
        // MiniCPM-o's TTS conditioning).
        let prune_last = seq_len > 1 && self.last_layer_is_moe;

        let mut hidden_states = hidden_states;
        if let Some((last, rest)) = self.layers.split_last_mut() {
            for layer in rest {
                hidden_states =
                    layer.forward(&hidden_states, &cos, &sin, attention_mask.as_ref())?;
            }
            if prune_last {
                hidden_states =
                    last.forward_attn(&hidden_states, &cos, &sin, attention_mask.as_ref())?;
                hidden_states = hidden_states.narrow(1, seq_len - 1, 1)?;
                hidden_states = last.forward_mlp(&hidden_states)?;
            } else {
                hidden_states =
                    last.forward(&hidden_states, &cos, &sin, attention_mask.as_ref())?;
            }
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        // Cheap to stash: Tensor is Arc-backed, so this is a refcount bump,
        // not a data copy. Lets callers that need the post-norm hidden
        // states (e.g. MiniCPM-o's TTS conditioning, which needs every
        // generated position's hidden state, not just the last) get them
        // via `last_hidden_states()` without changing this method's return
        // type for every other caller. Already pruned to the last position
        // alone when `prune_last` fired above.
        self.last_hidden_states = Some(hidden_states.clone());
        debug_assert!(
            prune_last || hidden_states.dim(1).is_ok_and(|s| s == seq_len),
            "last_hidden_states seq dim mismatch: expected {seq_len}, got {:?}",
            hidden_states.dim(1),
        );
        let logits = if prune_last {
            self.lm_head.forward_logits(&hidden_states)?
        } else {
            self.lm_head
                .forward_logits(&hidden_states.narrow(1, seq_len - 1, 1)?)?
        };
        Ok(logits)
    }

    /// Post-norm hidden states (pre-`lm_head`) from the most recent
    /// [`Self::forward`]/[`Self::forward_embeds`] call.
    ///
    /// Shape is `[B, S, H]` normally, but `[B, 1, H]` (last position only)
    /// when the last decoder layer is `MoE` and `seq_len > 1`. The `MoE`
    /// pruning optimization narrows hidden states before that layer's MLP,
    /// so only the output-relevant position survives. Callers that need the
    /// full sequence should check `dim(1)`.
    ///
    /// Exists as a side-channel instead of widening every caller's return
    /// type.
    #[must_use]
    pub fn last_hidden_states(&self) -> Option<&Tensor> {
        self.last_hidden_states.as_ref()
    }

    /// The token embedding table, exposed for callers that need to embed
    /// text tokens themselves before splicing in other embeddings (e.g.
    /// Qwen3-ASR's audio/text embedding merge).
    #[must_use]
    pub fn embed_tokens(&self) -> &EmbeddingLayer {
        &self.embed_tokens
    }

    // ── KV Cache Management ─────────────────────────────────────────────

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }

    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Number of tokens currently held in the KV cache (0 before the first
    /// forward pass, or right after `clear_kv_cache`). Needed by callers
    /// that evict from the middle of the cache (e.g. MiniCPM-o's duplex
    /// sliding window) and must keep `start_pos` in later `forward`/
    /// `forward_embeds` calls in sync with the cache's *actual* length
    /// rather than a separately-tracked running position counter.
    #[must_use]
    pub fn kv_cache_len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.self_attn.cache_seq_len)
    }

    /// Total bytes held by the model's KV caches (no GPU copies).
    #[must_use]
    pub fn active_kv_cache_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| {
                l.self_attn.kv_cache.as_ref().map_or(0, |(k, v)| {
                    let k_bytes = k.elem_count() as u64 * k.dtype().size_in_bytes() as u64;
                    let v_bytes = v.elem_count() as u64 * v.dtype().size_in_bytes() as u64;
                    k_bytes + v_bytes
                })
            })
            .sum()
    }

    /// Extract per-layer KV caches (valid portion only, zero-copy narrow views).
    ///
    /// The returned views still reference the pre-allocated buffer.  Callers
    /// that need to free the buffer (e.g. batch-decode extract) should use
    /// `Tensor::contiguous()` on their side, or clear `seq.kv_caches` after
    /// consuming the views.
    #[must_use]
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
        for (layer, cache) in self.layers.iter_mut().zip(caches) {
            let seq_len = cache.as_ref().map_or(0, |(k, _)| k.dim(2).unwrap_or(0));
            layer.self_attn.kv_cache = cache;
            layer.self_attn.cache_seq_len = seq_len;
        }
    }

    // ── Batched Decode ──────────────────────────────────────────────────

    /// Pad per-sequence KV caches to the same length and load into model layers.
    ///
    /// Returns `(kv_lens, max_kv_len)`.
    ///
    /// # Errors
    ///
    /// Returns an error if padding or stacking the KV caches fails.
    // b/h/s/d are standard tensor-shape notation (batch, heads, seq_len,
    // head_dim), matching the BHSD terminology used elsewhere in this file.
    #[allow(clippy::many_single_char_names)]
    pub fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        extra_room: usize,
    ) -> Result<(Vec<usize>, usize)> {
        let kv_heads = self.config.num_key_value_heads;
        let head_dim = self.config.head_dim();
        let device = self.embed_tokens.device();
        let dtype = self.dtype;

        let kv_lens: Vec<usize> = seq_kv_caches
            .iter()
            .map(|caches| {
                caches
                    .first()
                    .and_then(|c| c.as_ref())
                    .map_or(0, |(k, _)| k.dim(2).unwrap_or(0))
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
                &device,
                dtype,
            )?;

            if let Some((k, v)) = batched_kv {
                let k = k.contiguous()?;
                let v = v.contiguous()?;
                if extra_room > 0 {
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

    /// Run one batched decode step.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
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
        // Sequence positions never approach u32::MAX.
        #[allow(clippy::cast_possible_truncation)]
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
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, attention_mask)?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        self.lm_head.forward_logits(&hidden_states) // [N, 1, vocab]
    }

    /// Extract per-sequence KV caches from batched state.
    ///
    /// # Errors
    ///
    /// Returns an error if narrowing or copying the KV cache tensors fails.
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

        for layer in &mut self.layers {
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
                for row in &mut result {
                    row.push(None);
                }
            }
            layer.self_attn.kv_cache = None;
            layer.self_attn.cache_seq_len = 0;
        }

        Ok(result)
    }

    /// Access the model config.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Access the model dtype.
    #[must_use]
    pub fn model_dtype(&self) -> DType {
        self.dtype
    }
}

// ── Utilities ───────────────────────────────────────────────────────────

/// Build attention mask for batched decode with padding-aware masking.
///
/// # Errors
///
/// Returns an error if building the mask tensor fails.
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
    let mut padded_keys = Vec::with_capacity(n);
    let mut padded_values = Vec::with_capacity(n);

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
        if let Some((k, v)) = cache {
            let cur_len = k.dim(2)?;
            let pad_len = max_len - cur_len;
            if pad_len > 0 {
                let pad = zero_pad.as_ref().unwrap().narrow(2, 0, pad_len)?;
                padded_keys.push(Tensor::cat(&[&pad, k], 2)?);
                padded_values.push(Tensor::cat(&[&pad, v], 2)?);
            } else {
                padded_keys.push(k.clone());
                padded_values.push(v.clone());
            }
        } else {
            let zeros = Tensor::zeros((1, kv_heads, max_len, head_dim), dtype, device)?;
            padded_keys.push(zeros.clone());
            padded_values.push(zeros);
        }
    }

    let stacked_k = Tensor::cat(&padded_keys, 0)?.contiguous()?;
    let stacked_v = Tensor::cat(&padded_values, 0)?.contiguous()?;
    Ok(Some((stacked_k, stacked_v)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;
    use candle_nn::attention::flash_attn;

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

    // Formula matches the real Qwen3-Coder-30B-A3B GGUF geometry verified
    // against its own metadata: 48 layers, 4 KV heads, head_dim 128, F16
    // (2 bytes) -> 96 KiB/token.
    #[test]
    fn test_kv_bytes_per_token_matches_qwen3_coder_30b_a3b() {
        let config = Config {
            num_hidden_layers: 48,
            num_key_value_heads: 4,
            head_dim: Some(128),
            ..tiny_config()
        };
        assert_eq!(config.kv_bytes_per_token(2), 96 * 1024);
    }

    // Uses the tiny test config's own geometry: 1 layer, 2 KV heads,
    // head_dim 4, F32 (4 bytes) -> 2*1*2*4*4 = 64 bytes/token.
    #[test]
    fn test_kv_bytes_per_token_tiny_config() {
        assert_eq!(tiny_config().kv_bytes_per_token(4), 64);
    }

    // Dense checkpoints carry no MoE fields, so `moe_config()` must return `None`.
    #[test]
    fn test_moe_config_none_for_dense_checkpoint() {
        assert!(tiny_config().moe_config().is_none());
    }

    // A checkpoint with all five MoE fields set builds a matching `MoeConfig`.
    #[test]
    fn test_moe_config_some_for_full_moe_checkpoint() {
        let config = Config {
            num_experts: Some(8),
            num_experts_per_tok: Some(2),
            moe_intermediate_size: Some(64),
            norm_topk_prob: Some(false),
            decoder_sparse_step: Some(2),
            ..tiny_config()
        };
        let moe_config = config.moe_config().expect("moe_config");
        assert_eq!(moe_config.num_experts, 8);
        assert_eq!(moe_config.num_experts_per_tok, 2);
        assert_eq!(moe_config.moe_intermediate_size, 64);
        assert!(!moe_config.norm_topk_prob);
        assert_eq!(moe_config.decoder_sparse_step, Some(2));
    }

    // A missing required sizing field (here `moe_intermediate_size`) means
    // `moe_config()` must return `None`, even if other MoE fields are set.
    #[test]
    fn test_moe_config_none_when_required_field_missing() {
        let config = Config {
            num_experts: Some(8),
            num_experts_per_tok: Some(2),
            ..tiny_config()
        };
        assert!(config.moe_config().is_none());
    }

    // An absent `norm_topk_prob` defaults to `true`.
    #[test]
    fn test_moe_config_norm_topk_prob_defaults_to_true() {
        let config = Config {
            num_experts: Some(8),
            num_experts_per_tok: Some(2),
            moe_intermediate_size: Some(64),
            ..tiny_config()
        };
        assert!(config.moe_config().expect("moe_config").norm_topk_prob);
    }

    fn moe_layer_config(num_hidden_layers: usize, decoder_sparse_step: Option<usize>) -> Config {
        Config {
            num_hidden_layers,
            num_experts: Some(2),
            num_experts_per_tok: Some(1),
            moe_intermediate_size: Some(32),
            decoder_sparse_step,
            ..tiny_config()
        }
    }

    // `decoder_sparse_step: Some(2)` makes every 2nd layer (1-indexed) MoE;
    // the rest stay dense.
    #[test]
    fn test_moe_layer_selection_respects_decoder_sparse_step() {
        let cfg = moe_layer_config(4, Some(2));
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = Qwen3Model::new(&cfg, vb).expect("new");

        let is_moe: Vec<bool> = model
            .layers
            .iter()
            .map(|l| matches!(l.mlp, MlpOrMoe::Moe(_)))
            .collect();
        assert_eq!(is_moe, vec![false, true, false, true]);
    }

    // An absent `decoder_sparse_step` on a MoE checkpoint means every layer
    // is MoE, matching HF's default of `1`.
    #[test]
    fn test_moe_all_layers_when_decoder_sparse_step_none() {
        let cfg = moe_layer_config(2, None);
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = Qwen3Model::new(&cfg, vb).expect("new");

        assert!(
            model
                .layers
                .iter()
                .all(|l| matches!(l.mlp, MlpOrMoe::Moe(_)))
        );
    }

    // On a CPU-only model, `promote_experts_to_gpu` must return `Ok(())`
    // without panicking (no live VRAM query is possible on CPU, so it
    // takes the early-return path before running the probe forward pass).
    #[test]
    fn test_promote_experts_to_gpu_is_noop_on_cpu() {
        let cfg = moe_layer_config(2, None);
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = Qwen3Model::new(&cfg, vb).expect("new");

        model
            .promote_experts_to_gpu(&device, 1 << 30, 0)
            .expect("promote_experts_to_gpu");
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
        let mut model_b =
            Qwen3Model::new_from_model_vb(&cfg, vb.pp("model"), vb).expect("new_from_model_vb");

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
        let q_g = (q.reshape((b, kv_heads, n_rep, head_dim)).unwrap() * scale).unwrap();
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
        let q_g = (q.reshape((b, kv_heads, n_rep, head_dim)).unwrap() * scale).unwrap();
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

            let attn_weights =
                (q.matmul(&k_exp.transpose(2, 3).unwrap()).unwrap() * scale).unwrap();
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

    /// Regression test for the causal mask's additive penalty overflowing
    /// F16: `b_sz > 1` forces `decode()`'s `broadcast_add` mask path (skips
    /// the CPU/`b_sz==1` flash_attn fast path, which builds its mask via
    /// `AttnMask::Causal` instead and never hits this code). In F16, naively
    /// casting the boolean mask to F16 *before* multiplying by -1e9 makes
    /// candle's affine op convert -1e9 to literal -Inf first, so every
    /// *unmasked* (0.0) position computes `0.0 * -Inf = NaN`.
    #[test]
    fn test_prefill_batch_gt1_f16_mask_stays_finite() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F16, &device);
        let mut model = Qwen3Model::new(&cfg, vb).expect("new");

        // b_sz=2, seq_len=3: not the b_sz==1 CPU fast path, so this hits
        // decode()'s broadcast_add causal mask in F16.
        let prefill_ids = Tensor::new(&[[1u32, 2, 3], [4u32, 5, 6]], &device).expect("ids");
        let logits = model.forward(&prefill_ids, 0).expect("prefill");

        let values = logits
            .to_dtype(DType::F32)
            .expect("to_dtype")
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "logits contain a non-finite value: {values:?}"
        );
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
        model_single
            .forward(&single_ids, 0)
            .expect("single prefill");
        let out_single = model_single.forward(&decode_id, 5).expect("single decode");

        // Chunked prefill: two prefill calls, then decode.
        let chunk_a = Tensor::new(&[[1u32, 2, 3]], &device).expect("chunk_a");
        let chunk_b = Tensor::new(&[[4u32, 5]], &device).expect("chunk_b");
        model_chunked.forward(&chunk_a, 0).expect("chunk a prefill");
        model_chunked.forward(&chunk_b, 3).expect("chunk b prefill");
        let out_chunked = model_chunked
            .forward(&decode_id, 5)
            .expect("chunked decode");

        assert_eq!(out_single.dims(), out_chunked.dims());
        assert!(max_abs_diff(&out_single, &out_chunked) < 1e-4);
    }

    // MoE last-layer pruning (narrowing hidden states to the last position
    // between attention and MLP on the final layer when it is MoE) must not
    // change the model's output. Verify by comparing a multi-token prefill
    // (which triggers the prune path) against single-token-at-a-time
    // decoding through the same weights (which never prunes because
    // `seq_len == 1`).
    #[test]
    fn test_moe_last_layer_prune_matches_incremental_decode() {
        // 2 layers, all MoE (decoder_sparse_step: None), so
        // `prune_last` fires when seq_len > 1.
        let cfg = moe_layer_config(2, None);
        let device = Device::Cpu;
        let varmap = VarMap::new();

        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model_prefill = Qwen3Model::new(&cfg, vb.clone()).expect("model_prefill");
        let mut model_incr = Qwen3Model::new(&cfg, vb).expect("model_incr");

        assert!(
            matches!(
                model_prefill.layers.last().expect("layers").mlp,
                MlpOrMoe::Moe(_)
            ),
            "last layer should be MoE for this test",
        );

        // Multi-token prefill: seq_len=3 > 1, triggers prune path.
        let prefill_ids = Tensor::new(&[[1u32, 2, 3]], &device).expect("prefill_ids");
        let logits_prefill = model_prefill
            .forward(&prefill_ids, 0)
            .expect("prefill forward");

        // Incremental decode: feed tokens one at a time (never prunes).
        let t1 = Tensor::new(&[[1u32]], &device).expect("t1");
        let t2 = Tensor::new(&[[2u32]], &device).expect("t2");
        let t3 = Tensor::new(&[[3u32]], &device).expect("t3");
        model_incr.forward(&t1, 0).expect("incr t1");
        model_incr.forward(&t2, 1).expect("incr t2");
        let logits_incr = model_incr.forward(&t3, 2).expect("incr t3");

        // Both produce logits for the last position given the same
        // context; they must match.
        assert_eq!(logits_prefill.dims(), logits_incr.dims());
        assert!(
            max_abs_diff(&logits_prefill, &logits_incr) < 1e-4,
            "MoE pruned prefill diverged from incremental decode",
        );
    }

    // When the last layer is dense (not MoE), the pruning optimization
    // must not fire and `last_hidden_states` must retain the full sequence
    // dimension.
    #[test]
    fn test_dense_last_layer_preserves_full_hidden_states() {
        // 3 layers, step=2: layers [dense, MoE, dense]. Last is dense,
        // so prune_last stays false.
        let cfg = moe_layer_config(3, Some(2));
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = Qwen3Model::new(&cfg, vb).expect("new");

        assert!(
            matches!(model.layers.last().expect("layers").mlp, MlpOrMoe::Dense(_)),
            "last layer should be dense for this test",
        );

        let ids = Tensor::new(&[[1u32, 2, 3]], &device).expect("ids");
        model.forward(&ids, 0).expect("forward");

        let hidden = model
            .last_hidden_states()
            .expect("last_hidden_states should be Some");
        assert_eq!(
            hidden.dims(),
            &[1, 3, 16],
            "dense last layer should preserve full sequence in hidden states",
        );
    }

    // When pruning fires, `last_hidden_states` should be narrowed to
    // `[B, 1, H]`.
    #[test]
    fn test_moe_last_layer_prune_narrows_hidden_states() {
        let cfg = moe_layer_config(2, None);
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let mut model = Qwen3Model::new(&cfg, vb).expect("new");

        let ids = Tensor::new(&[[1u32, 2, 3]], &device).expect("ids");
        model.forward(&ids, 0).expect("forward");

        let hidden = model
            .last_hidden_states()
            .expect("last_hidden_states should be Some");
        assert_eq!(
            hidden.dims(),
            &[1, 1, 16],
            "MoE pruned path should narrow hidden states to last position",
        );
    }
}
