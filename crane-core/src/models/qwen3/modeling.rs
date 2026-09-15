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
use ribo::utils::log;
use serde::Deserialize;
use std::io::{Read, Seek};

use crate::device::{
    DeviceAssignment, GpuBudget, WeightBudget, format_budget, greedy_fit_layers, query_gpu_memory,
};
use crate::models::modules::embedding::EmbeddingLayer;
use crate::models::modules::flash_attn::dispatch_flash_attn;
use crate::models::modules::moe::{MlpOrMoe, MoeConfig, SparseMoeBlock};
use crate::models::modules::quant_kv_cache;
use crate::models::modules::quant_kv_cache::{FpKvCache, KvCache, KvCacheKind, KvCacheState};
use crate::models::modules::rotary::RotaryEmbedding;
use crate::ops::fused_ops::quant_attn;
use crate::utils::DeviceExt;

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
pub type BatchKvCache = Vec<Vec<Option<KvCacheState>>>;

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

    /// Bytes of KV cache one sequence consumes per generated token when K/V
    /// are stored as `bits`-wide (4 or 8) quantized codes plus a per-token
    /// f32 scale per head, instead of the compute dtype (see
    /// [`crate::models::modules::quant_kv_cache::QuantKvCache`]). Only an
    /// accurate estimate when the fused dequantize-in-attention kernel
    /// (`crate::ops::fused_ops::quant_attn`) covers the sequence's entire
    /// lifetime — see [`Self::kv_bytes_per_token`]'s doc for why plain
    /// compute-dtype pricing is required otherwise.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is neither 4 nor 8.
    #[must_use]
    pub fn quantized_kv_bytes_per_token(&self, bits: u32) -> u64 {
        quant_kv_cache::quantized_kv_bytes_per_token(
            bits,
            self.num_hidden_layers,
            self.num_key_value_heads,
            self.head_dim(),
        )
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
    /// Per-token K/V cache, plain or quantized depending on `CRANE_KV_QUANT`
    /// (see [`KvCacheKind::from_env`]). `KvCache::default()` (the `Fp`
    /// variant) behaves identically to the plain pre-allocated buffer this
    /// replaced.
    kv_cache: KvCache,
}

impl Attention {
    // `VarBuilder` is conventionally passed by value throughout this
    // codebase (its `pp`/`device`/`dtype` accessors take `&self` and are
    // cheap to call repeatedly); matching that convention here.
    #[allow(clippy::needless_pass_by_value)]
    fn new(config: &Config, vb: VarBuilder, kv_kind: KvCacheKind) -> Result<Self> {
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
                Some(crate::models::with_tracing::rms_norm(
                    head_dim,
                    config.rms_norm_eps,
                    vb.pp("q_norm"),
                )?),
                Some(crate::models::with_tracing::rms_norm(
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
            kv_cache: KvCache::new(kv_kind),
        })
    }

    /// Construct from GGUF quantized weights.
    fn new_from_gguf<R: Read + Seek>(
        config: &Config,
        gg: &mut Gguf<R>,
        layer_idx: usize,
        kv_kind: KvCacheKind,
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
            kv_cache: KvCache::new(kv_kind),
        })
    }

    /// Append this step's K/V to the cache and return the full cached K/V in
    /// the compute dtype, ready for attention. See [`KvCache::append`].
    fn update_kv_cache(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        self.kv_cache.append(k, v)
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

        // ── Fused dequantize-in-attention, GPU + quantized cache only
        // (Phase 5) ── Reads int8/int4 codes + scales directly and
        // dequantizes inside the QK/SV matmuls instead of reading a
        // fully-dequantized K/V back from `QuantKvCache`'s scratch buffer,
        // so this step never materializes a full-context-length
        // compute-dtype K/V tensor -- covers both decode (seq_len == 1) and
        // prefill (seq_len > 1). `try_quantized_append` returns `None`
        // (falling through below) for an `Fp` cache, a CPU/Metal device, or
        // `CRANE_QUANT_ATTN_FUSED=0`.
        if let Some(kv_ref) = self.kv_cache.try_quantized_append(&k, &v)? {
            // q is [B, num_heads, seq_len, D]; fold (n_rep, seq_len) into a
            // single R axis per kv head (rep-major, position-minor), the
            // same merge `quant_qk_dot`'s kernel already expects for decode
            // (there seq_len == 1 so R == n_rep).
            let q_g =
                (q.reshape((b_sz, self.num_kv_heads, n_rep * seq_len, self.head_dim))? * scale)?;
            let scores = quant_attn::quant_qk_dot(&q_g, &kv_ref)?;
            let total_kv = scores.dim(D::Minus1)?;
            // The mask varies per query position, not per rep, so unfold R
            // back into (n_rep, seq_len) merged with kv_heads into
            // num_heads before adding it — the exact inverse of the q_g
            // fold above.
            let scores = scores.reshape((b_sz, self.num_heads, seq_len, total_kv))?;
            let scores = match attention_mask {
                Some(mask) => scores.broadcast_add(mask)?,
                None => scores,
            };
            let scores = candle_nn::ops::softmax_last_dim(&scores)?.reshape((
                b_sz,
                self.num_kv_heads,
                n_rep * seq_len,
                total_kv,
            ))?;
            let attn_output = quant_attn::quant_sv_dot(&scores, &kv_ref)?;
            let attn_output = attn_output
                .reshape((b_sz, self.num_heads, seq_len, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b_sz, seq_len, ()))?;
            return self.o_proj.forward(&attn_output);
        }

        // Update KV cache (pre-allocated with slice_set)
        let (k, v) = self.update_kv_cache(&k, &v)?;

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
        self.kv_cache.reset();
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
                #[cfg(any(feature = "cuda", feature = "rocm"))]
                {
                    if gu.device().is_cuda() || gu.device().is_rocm() {
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
        kv_kind: KvCacheKind,
    ) -> Result<Self> {
        let self_attn = Attention::new(config, vb.pp("self_attn"), kv_kind)?;
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
        let input_layernorm = crate::models::with_tracing::rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = crate::models::with_tracing::rms_norm(
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
        kv_kind: KvCacheKind,
    ) -> Result<Self> {
        let self_attn = Attention::new_from_gguf(config, gg, layer_idx, kv_kind)?;
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

// ── Full Model ──────────────────────────────────────────────────────────

pub struct Qwen3Model {
    embed_tokens: EmbeddingLayer,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: LinearLayer,
    rotary_emb: RotaryEmbedding,
    config: Config,
    dtype: DType,
    /// Full-sequence post-norm hidden states from the most recent forward
    /// call — see [`Self::last_hidden_states`].
    last_hidden_states: Option<Tensor>,
    /// KV cache representation in use, selected at construction either
    /// from `CRANE_KV_QUANT` or an explicit `kv_kind` argument (e.g.
    /// `--kv-quant`). Needed by [`Self::extract_batch_kv`] to decide
    /// whether to re-quantize.
    kv_kind: KvCacheKind,
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

/// Quantized on-disk byte size of a GGUF tensor, computed from header
/// metadata alone (no tensor data read).
fn gguf_tensor_bytes(info: &gguf_file::TensorInfo) -> u64 {
    let elem_count = info.shape.elem_count() as u64;
    let block_size = info.ggml_dtype.block_size() as u64;
    let type_size = info.ggml_dtype.type_size() as u64;
    elem_count / block_size * type_size
}

/// Whether a GGUF tensor name is an `MoE` expert weight (packed or
/// per-expert layout) — see [`SparseMoeBlock::new_from_gguf`] for the two
/// layouts. Excludes the router (`ffn_gate_inp`), which always stays on the
/// main device.
fn is_expert_tensor(name: &str) -> bool {
    if name.contains("ffn_gate_exps")
        || name.contains("ffn_up_exps")
        || name.contains("ffn_down_exps")
    {
        return true;
    }
    ["ffn_gate.", "ffn_up.", "ffn_down."].iter().any(|prefix| {
        name.split(prefix)
            .nth(1)
            .and_then(|rest| rest.chars().next())
            .is_some_and(|c| c.is_ascii_digit())
    })
}

/// Extracts the decoder layer index from a `blk.{i}.*` tensor name.
fn expert_tensor_layer(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    rest[..dot].parse().ok()
}

/// Estimated GPU VRAM cost of loading one `MoE` layer's expert weights.
///
/// Packed experts are dequantized to `compute_dtype_bytes` up front (see
/// [`SparseMoeBlock::new_from_gguf`]'s doc comment on `load_packed_experts`),
/// so their cost is the dequantized size, not the on-disk quantized size.
/// Per-expert tensors are loaded quantized but are also dequantized to
/// `compute_dtype_bytes` on promotion (see `LinearLayer::to_device`), so
/// their cost estimate uses the same dequantized size, not the smaller
/// on-disk quantized size.
fn estimate_expert_layer_vram(
    tensor_infos: &std::collections::HashMap<String, gguf_file::TensorInfo>,
    layer_idx: usize,
    is_packed: bool,
    compute_dtype_bytes: usize,
) -> u64 {
    if is_packed {
        ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
            .iter()
            .map(|suffix| {
                let name = format!("blk.{layer_idx}.{suffix}.weight");
                tensor_infos.get(&name).map_or(0, |info| {
                    info.shape.elem_count() as u64 * compute_dtype_bytes as u64
                })
            })
            .sum()
    } else {
        tensor_infos
            .iter()
            .filter(|(name, _)| {
                is_expert_tensor(name) && expert_tensor_layer(name) == Some(layer_idx)
            })
            .map(|(_, info)| info.shape.elem_count() as u64 * compute_dtype_bytes as u64)
            .sum()
    }
}

impl Qwen3Model {
    /// Construct from safetensors / `HuggingFace` checkpoint.
    ///
    /// `gpu_budget` constrains `MoE` expert placement; only consumed once
    /// the checkpoint is `MoE` (see [`crate::device::GpuBudget`]).
    ///
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing or has an
    /// unexpected shape.
    pub fn new(
        config: &Config,
        vb: VarBuilder,
        expert_device: &Device,
        gpu_budget: &GpuBudget,
    ) -> Result<Self> {
        Self::new_inner(config, vb.pp("model"), vb, expert_device, gpu_budget)
    }

    /// Like [`Self::new`], but takes an explicit `kv_kind` (e.g. from a
    /// `--kv-quant` CLI flag) instead of reading `CRANE_KV_QUANT`.
    ///
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing or has an
    /// unexpected shape.
    pub fn new_with_kv_kind(
        config: &Config,
        vb: VarBuilder,
        expert_device: &Device,
        gpu_budget: &GpuBudget,
        kv_kind: KvCacheKind,
    ) -> Result<Self> {
        Self::new_inner_with_kv_kind(
            config,
            vb.pp("model"),
            vb,
            expert_device,
            gpu_budget,
            kv_kind,
        )
    }

    /// Construct from a checkpoint where the decoder is nested under a
    /// deeper prefix than the standard `model.*` layout (e.g. Qwen3-ASR's
    /// `model.language_model.*`). `model_vb` must already be scoped to the
    /// decoder's root (what would otherwise be `vb.pp("model")`); `root_vb`
    /// is the checkpoint root, used to resolve an untied `lm_head` sibling.
    /// `gpu_budget` constrains `MoE` expert placement; only consumed once
    /// the checkpoint is `MoE` (see [`crate::device::GpuBudget`]).
    ///
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing or has an
    /// unexpected shape.
    pub fn new_from_model_vb(
        config: &Config,
        model_vb: VarBuilder,
        root_vb: VarBuilder,
        expert_device: &Device,
        gpu_budget: &GpuBudget,
    ) -> Result<Self> {
        Self::new_inner(config, model_vb, root_vb, expert_device, gpu_budget)
    }

    // See `Attention::new`'s comment on `VarBuilder` by-value.
    #[allow(clippy::needless_pass_by_value)]
    fn new_inner(
        config: &Config,
        model_vb: VarBuilder,
        root_vb: VarBuilder,
        expert_device: &Device,
        gpu_budget: &GpuBudget,
    ) -> Result<Self> {
        Self::new_inner_with_kv_kind(
            config,
            model_vb,
            root_vb,
            expert_device,
            gpu_budget,
            KvCacheKind::from_env(),
        )
    }

    /// Like [`Self::new_inner`], but takes an explicit `kv_kind` instead of
    /// reading `CRANE_KV_QUANT` — used by [`Self::new_with_kv_kind`] and by
    /// tests (avoids mutating process-wide env state from parallel tests).
    // See `Attention::new`'s comment on `VarBuilder` by-value.
    #[allow(clippy::needless_pass_by_value)]
    fn new_inner_with_kv_kind(
        config: &Config,
        model_vb: VarBuilder,
        root_vb: VarBuilder,
        expert_device: &Device,
        _gpu_budget: &GpuBudget,
        kv_kind: KvCacheKind,
    ) -> Result<Self> {
        log::info!("KV cache: {}", kv_kind.describe());
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
                kv_kind,
            )?);
        }

        let norm = crate::models::with_tracing::rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            model_vb.pp("norm"),
        )?;

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

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            config: config.clone(),
            dtype,
            last_hidden_states: None,
            kv_kind,
        })
    }

    /// Construct from a GGUF file.
    ///
    /// `devices.main` holds every weight, including `MoE` experts that fit
    /// in `gpu_budget`; `devices.expert` is not used by this loading path.
    /// `gpu_budget` decides, per `MoE` layer, whether that layer's experts
    /// load to `devices.main` or `Device::Cpu`; only consumed once the
    /// checkpoint is `MoE` (see [`crate::device::GpuBudget`]).
    ///
    /// # Errors
    ///
    /// Returns an error if a required tensor or metadata entry is missing
    /// or has an unexpected shape.
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        devices: &DeviceAssignment,
        gpu_budget: &GpuBudget,
    ) -> Result<Self> {
        Self::from_gguf_with_kv_kind(ct, reader, devices, gpu_budget, KvCacheKind::from_env())
    }

    /// Like [`Self::from_gguf`], but takes an explicit `kv_kind` (e.g. from
    /// a `--kv-quant` CLI flag) instead of reading `CRANE_KV_QUANT`.
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
    pub fn from_gguf_with_kv_kind<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        devices: &DeviceAssignment,
        gpu_budget: &GpuBudget,
        kv_kind: KvCacheKind,
    ) -> Result<Self> {
        let device = &devices.main;
        let dtype = if device.is_cuda() {
            DType::BF16
        } else if device.is_metal() || device.is_rocm() {
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

        // ── Per-layer expert placement ───────────────────────────────────
        //
        // Non-expert weights (attention, norms, embeddings, router) always
        // load to `devices.main`; only MoE expert weights are considered
        // for offloading to `Device::Cpu`, based on `gpu_budget`.
        //
        // `WeightBudget::Limited` loads every MoE layer to CPU here (a safe
        // default) and defers the real GPU/CPU decision to
        // `promote_experts_after_probe`, called once this model exists and
        // can run a probe forward pass — see that method's doc comment for
        // why a static pre-load estimate isn't good enough on its own.
        let is_moe_checkpoint = config.num_experts.is_some_and(|n| n > 0);
        let expert_devices: Vec<Device> = if !is_moe_checkpoint {
            vec![devices.main.clone(); num_hidden_layers]
        } else if gpu_budget.offload_all_experts {
            log::info!("--offload-experts: all MoE expert layers -> CPU");
            vec![Device::Cpu; num_hidden_layers]
        } else {
            match gpu_budget.weight_budget {
                WeightBudget::NoGpu | WeightBudget::Unlimited => {
                    vec![devices.main.clone(); num_hidden_layers]
                },
                WeightBudget::Limited(_) => vec![Device::Cpu; num_hidden_layers],
            }
        };

        log::info!("KV cache: {}", kv_kind.describe());
        log::info!(
            "Loading {num_hidden_layers} layers from GGUF (attention{} weights stay quantized)...",
            if is_moe_checkpoint {
                " + MoE expert"
            } else {
                ""
            },
        );
        let mut layers = Vec::with_capacity(num_hidden_layers);
        for i in 0..num_hidden_layers {
            layers.push(DecoderLayer::new_from_gguf(
                &config,
                &mut gg,
                i,
                &expert_devices[i],
                kv_kind,
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

        let mut model = Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            config,
            dtype,
            last_hidden_states: None,
            kv_kind,
        };

        if is_moe_checkpoint
            && !gpu_budget.offload_all_experts
            && let WeightBudget::Limited(total_vram) = gpu_budget.weight_budget
        {
            // Quantized pricing is only valid when the fused
            // dequantize-in-attention kernel covers the whole sequence
            // lifetime: CUDA/ROCm, `max_concurrent == 1` so batch-decode's
            // to_fp_pair()/from_fp_pair() round-trip (which always peaks at
            // the compute dtype) never applies, and `CRANE_QUANT_ATTN_FUSED`
            // isn't `0` (which forces every append onto the unfused path).
            // Otherwise price at the compute dtype's size — see
            // `Model::kv_bytes_per_token`'s doc comment for the same
            // reasoning.
            let fused_covers_full_lifetime = (device.is_cuda() || device.is_rocm())
                && gpu_budget.max_concurrent == Some(1)
                && !quant_attn::fused_disabled();
            let kv_bytes_per_token = kv_kind.effective_kv_bytes_per_token(
                fused_covers_full_lifetime,
                num_hidden_layers,
                num_kv_heads,
                head_dim,
                dtype.size_in_bytes(),
            );
            let runtime_reservation =
                gpu_budget.runtime_reservation_bytes_for_kv_bytes_per_token(kv_bytes_per_token);
            model.promote_experts_after_probe(
                devices,
                total_vram,
                runtime_reservation,
                &gg.ct.tensor_infos,
                num_hidden_layers,
                dtype.size_in_bytes(),
            )?;
        }

        Ok(model)
    }

    /// Runs a probe forward pass to force GPU backends' lazy first-use
    /// library initialization (rocBLAS/hipRAND/JIT-compiled kernels), then
    /// live-queries actual free VRAM and promotes CPU-placed `MoE` expert
    /// layers to GPU based on real remaining headroom.
    ///
    /// A purely static pre-load estimate (subtracting an estimated
    /// KV-cache reservation from `--gpu-memory-limit`) was found, via
    /// `crane-serve`'s live-queried `record_baseline()` compared across
    /// several real `ROCm` runs, to consistently underestimate actual
    /// post-warmup VRAM usage by several GB — a gap that didn't scale
    /// with the number of GPU-resident expert layers, so it isn't a
    /// weight-sizing bug. It's the cost of compute libraries that don't
    /// initialize until something actually runs on the device, which is
    /// unavoidably *after* model loading decides placement unless loading
    /// itself forces that initialization first.
    ///
    /// Best effort: if the probe forward pass fails, this logs a warning
    /// and leaves every expert on CPU (safe, just unoptimized) rather
    /// than failing model load. If live VRAM querying isn't supported for
    /// this device (e.g. Metal), falls back to the static estimate this
    /// replaces.
    ///
    /// # Errors
    ///
    /// Returns an error only if constructing the probe input tensor fails.
    /// A failed probe forward pass or failed expert promotion is handled
    /// gracefully (logged, affected experts left on CPU) and does not
    /// propagate an error.
    // One sequential pipeline (probe -> cost estimate -> budget -> promote)
    // sharing local state (`gpu_location`, `layer_costs`) throughout;
    // splitting it up would scatter that shared context across several
    // small functions without simplifying the control flow itself — same
    // rationale as `Attention::forward`'s existing `too_many_lines` allow.
    #[allow(clippy::too_many_lines)]
    fn promote_experts_after_probe(
        &mut self,
        devices: &DeviceAssignment,
        total_vram: u64,
        runtime_reservation: u64,
        tensor_infos: &std::collections::HashMap<String, gguf_file::TensorInfo>,
        num_hidden_layers: usize,
        dtype_bytes: usize,
    ) -> Result<()> {
        // `RocmDevice`'s `Debug` output (e.g. `DeviceId(1)`) is a
        // process-wide counter of *how many `RocmDevice`s this process has
        // ever constructed* — it is not the physical GPU ordinal, and is
        // frequently `1` even when the real ordinal is `0`. `location()`
        // is the only way to recover the real ordinal (CUDA/ROCm/Metal
        // `gpu_id`) for logging.
        let gpu_location = devices.main.location();
        log::info!(
            "Expert placement: running probe forward pass + live VRAM query on {gpu_location:?} \
             before deciding MoE GPU/CPU split (may take a few seconds)"
        );
        let probe_ids = Tensor::new(&[45u32, 546, 456], &devices.main)?.unsqueeze(0)?;
        if let Err(e) = self.forward(&probe_ids, 0) {
            log::warn!(
                "expert-placement probe forward failed on {gpu_location:?} (non-fatal, all \
                 experts stay on CPU): {e}"
            );
            return Ok(());
        }
        self.clear_kv_cache();

        // All MoE layers in a checkpoint share the same expert tensor
        // layout; detect it from the first MoE layer found.
        let is_packed = (0..num_hidden_layers)
            .find(|&i| tensor_infos.contains_key(&format!("blk.{i}.ffn_gate_inp.weight")))
            .is_some_and(|i| tensor_infos.contains_key(&format!("blk.{i}.ffn_gate_exps.weight")));

        let is_moe_layer: Vec<bool> = (0..num_hidden_layers)
            .map(|i| tensor_infos.contains_key(&format!("blk.{i}.ffn_gate_inp.weight")))
            .collect();
        // Every MoE layer in a Qwen3 checkpoint has identical expert-tensor
        // shapes, so this cost is uniform across MoE layers; non-MoE
        // layers cost 0 so they never affect the greedy budget below.
        let layer_costs: Vec<u64> = (0..num_hidden_layers)
            .map(|i| {
                if is_moe_layer[i] {
                    estimate_expert_layer_vram(tensor_infos, i, is_packed, dtype_bytes)
                } else {
                    0
                }
            })
            .collect();
        let total_moe_layers = is_moe_layer.iter().filter(|&&m| m).count();
        log::debug!(
            "MoE layout: {total_moe_layers} layers, packed={is_packed}, \
             per-layer expert cost estimate={}",
            format_budget(layer_costs.iter().copied().find(|&c| c > 0).unwrap_or(0)),
        );

        let available = if let Some((free, total)) = query_gpu_memory(&devices.main) {
            let used = total.saturating_sub(free);
            let ceiling = total_vram.min(total);
            let remaining = ceiling.saturating_sub(used);
            let available = remaining.saturating_sub(runtime_reservation);
            log::info!(
                "Live VRAM on {gpu_location:?} after probe: free={}, total={}, used={}, \
                 available_for_experts={} (configured limit={})",
                format_budget(free),
                format_budget(total),
                format_budget(used),
                format_budget(available),
                format_budget(total_vram),
            );
            available
        } else {
            // `token_embd.weight` stays quantized (`gg.quantized_embedding()`),
            // so it costs its on-disk size like every other non-expert
            // tensor and needs no special-casing here.
            let non_expert_bytes: u64 = tensor_infos
                .iter()
                .filter(|(name, _)| !is_expert_tensor(name))
                .map(|(_, info)| gguf_tensor_bytes(info))
                .sum::<u64>();
            let weight_budget = total_vram.saturating_sub(runtime_reservation);
            let available = weight_budget.saturating_sub(non_expert_bytes);
            log::warn!(
                "No live VRAM query available for {gpu_location:?}; falling back to static \
                 estimate: total={}, runtime_reserved={}, non_expert_weights={}, \
                 expert_budget={}",
                format_budget(total_vram),
                format_budget(runtime_reservation),
                format_budget(non_expert_bytes),
                format_budget(available),
            );
            available
        };

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
        // overhead (128 experts, each its own device transfer) mean actual
        // usage can still exceed it even though `promote_experts_to` is
        // itself atomic per layer. A failed promotion here (e.g. real GPU
        // out-of-memory) must not abort model load: stop promoting further
        // layers and leave the rest on CPU — degraded, not fatal.
        let mut gpu_layers = 0usize;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if !promoted.contains(&i) {
                continue;
            }
            let MlpOrMoe::Moe(block) = &mut layer.mlp else {
                continue;
            };
            // Re-query immediately before attempting, not the stale
            // snapshot from before this loop started: this tells us
            // whether a failure below reflects state that already
            // changed by the time we act, or whether it was already
            // consistent with the earlier estimate right up to the
            // moment of the actual allocation call.
            if let Some((free_before, total_before)) = query_gpu_memory(&devices.main) {
                log::debug!(
                    "layer {i}: live VRAM immediately before attempt on {gpu_location:?}: \
                     free={}, total={}, layer_cost={}",
                    format_budget(free_before),
                    format_budget(total_before),
                    format_budget(layer_costs[i]),
                );
            }
            match block.promote_experts_to(&devices.main, self.dtype) {
                Ok(()) => {
                    gpu_layers += 1;
                    log::debug!(
                        "layer {i}: promoted to {gpu_location:?} (cost={})",
                        format_budget(layer_costs[i]),
                    );
                },
                Err(e) => {
                    // Re-query right after the failure: if free VRAM
                    // dropped far more than `layer_costs[i]` despite the
                    // allocation itself failing, that points at the
                    // underlying GPU allocator reserving/growing a much
                    // larger pool on a failed attempt, rather than at
                    // Crane's own cost estimate being wrong.
                    if let Some((free_after, total_after)) = query_gpu_memory(&devices.main) {
                        log::warn!(
                            "layer {i}: live VRAM immediately after the failed attempt on \
                             {gpu_location:?}: free={}, total={} (compare to the \"before\" line \
                             above to see if the failed allocation itself consumed VRAM)",
                            format_budget(free_after),
                            format_budget(total_after),
                        );
                    }
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
        // Final checkpoint before returning to the caller (crane-serve's
        // own separate `Model::warmup()` + `record_baseline()` run next):
        // if VRAM usage jumps between this line and that later baseline,
        // the growth happened *after* model loading, not during it.
        if let Some((free, total)) = query_gpu_memory(&devices.main) {
            log::info!(
                "Live VRAM on {gpu_location:?} at end of from_gguf(): free={}, total={}, used={}",
                format_budget(free),
                format_budget(total),
                format_budget(total.saturating_sub(free)),
            );
        }
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

        let hidden_states = self.embed_tokens.forward(input_ids)?.to_dtype(self.dtype)?;
        self.decode(hidden_states, seq_len, start_pos, input_ids.device())
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

        let mut hidden_states = hidden_states;
        for layer in &mut self.layers {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, attention_mask.as_ref())?;
        }

        let hidden_states = self.norm.forward(&hidden_states)?;
        // Cheap to stash: Tensor is Arc-backed, so this is a refcount bump,
        // not a data copy. Lets callers that need the full-sequence
        // post-norm hidden states (e.g. MiniCPM-o's TTS conditioning, which
        // needs every generated position's hidden state, not just the
        // last) get them via `last_hidden_states()` without changing this
        // method's return type for every other caller.
        self.last_hidden_states = Some(hidden_states.clone());
        let logits = self
            .lm_head
            .forward_logits(&hidden_states.narrow(1, seq_len - 1, 1)?)?;
        Ok(logits)
    }

    /// Full-sequence post-norm hidden states (`[B, S, H]`, pre-`lm_head`)
    /// from the most recent [`Self::forward`]/[`Self::forward_embeds`] call.
    /// See the field doc on why this exists instead of widening every
    /// caller's return type.
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
        self.layers
            .first()
            .map_or(0, |l| l.self_attn.kv_cache.len())
    }

    /// Total bytes held by the model's KV caches (no GPU copies). Reflects
    /// the real, smaller footprint when `CRANE_KV_QUANT` is active, since
    /// [`KvCacheBackend::byte_size`] accounts for whatever representation
    /// each layer's cache actually stores.
    #[must_use]
    pub fn active_kv_cache_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| l.self_attn.kv_cache.byte_size() as u64)
            .sum()
    }

    /// Extract per-layer KV cache state (valid portion only). Plain or
    /// quantized depending on the active `CRANE_KV_QUANT` setting — see
    /// [`KvCacheState`].
    #[must_use]
    pub fn get_kv_caches(&self) -> Vec<Option<KvCacheState>> {
        self.layers
            .iter()
            .map(|l| l.self_attn.kv_cache.extract().ok().flatten())
            .collect()
    }

    /// Restore per-layer KV cache state extracted by [`Self::get_kv_caches`].
    pub fn set_kv_caches(&mut self, caches: Vec<Option<KvCacheState>>) {
        for (layer, cache) in self.layers.iter_mut().zip(caches) {
            match cache {
                Some(state) => {
                    if let Err(e) = layer.self_attn.kv_cache.install(state) {
                        log::warn!("set_kv_caches: failed to install layer state, resetting: {e}");
                        layer.self_attn.kv_cache.reset();
                    }
                },
                None => layer.self_attn.kv_cache.reset(),
            }
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
        seq_kv_caches: &[Vec<Option<KvCacheState>>],
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
                    .and_then(|s| s.seq_len().ok())
                    .unwrap_or(0)
            })
            .collect();
        let max_kv_len = kv_lens.iter().copied().max().unwrap_or(0);

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // The batch itself always runs in plain compute dtype regardless
            // of each sequence's stored representation — dequantize here,
            // re-quantize on extract (see `extract_batch_kv`).
            let layer_caches: Vec<Option<(Tensor, Tensor)>> = seq_kv_caches
                .iter()
                .map(|seq| {
                    seq[layer_idx]
                        .as_ref()
                        .map(|s| s.to_fp_pair(dtype))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            let layer_caches_refs: Vec<&Option<(Tensor, Tensor)>> = layer_caches.iter().collect();

            let batched_kv = pad_and_stack_kv_caches(
                &layer_caches_refs,
                max_kv_len,
                kv_heads,
                head_dim,
                &device,
                dtype,
            )?;

            let mut fp_cache = FpKvCache::new();
            if let Some((k, v)) = batched_kv {
                let k = k.contiguous()?;
                let v = v.contiguous()?;
                if extra_room > 0 {
                    let (b, h, s, d) = k.dims4()?;
                    let buf_k = Tensor::zeros((b, h, s + extra_room, d), k.dtype(), k.device())?;
                    let buf_v = Tensor::zeros((b, h, s + extra_room, d), v.dtype(), v.device())?;
                    buf_k.slice_set(&k, 2, 0)?;
                    buf_v.slice_set(&v, 2, 0)?;
                    fp_cache.install_with_headroom(buf_k, buf_v, max_kv_len);
                } else {
                    fp_cache.install_with_headroom(k, v, max_kv_len);
                }
            }
            layer.self_attn.kv_cache = KvCache::Fp(fp_cache);
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
        let mut result: Vec<Vec<Option<KvCacheState>>> = (0..n_seqs)
            .map(|_| Vec::with_capacity(num_layers))
            .collect();
        let dtype = self.dtype;
        let kv_kind = self.kv_kind;

        // Extract every layer's batched state and reset it to a fresh cache
        // of the correct kind up front, before any fallible per-sequence
        // slicing below. Otherwise a failure partway through a combined
        // loop would leave already-processed layers reset while later
        // layers keep stale batched (Fp) state — a mixed-variant model.
        let mut layer_states = Vec::with_capacity(num_layers);
        for layer in &mut self.layers {
            layer_states.push(layer.self_attn.kv_cache.extract()?);
            layer.self_attn.kv_cache = KvCache::new(kv_kind);
        }

        for state in layer_states {
            if let Some(state) = state {
                // The batch always runs in plain compute dtype (see
                // `setup_batch_decode`); `to_fp_pair` is a no-op cast here.
                let (full_k, full_v) = state.to_fp_pair(dtype)?;
                for i in 0..n_seqs {
                    let row_k = full_k.narrow(0, i, 1)?;
                    let row_v = full_v.narrow(0, i, 1)?;
                    let total = kv_lens[i] + rounds_done;
                    let offset = original_max_kv - kv_lens[i];
                    // Contiguous copy — breaks ref to padded batch buffer.
                    let k = row_k.narrow(2, offset, total)?.contiguous()?;
                    let v = row_v.narrow(2, offset, total)?.contiguous()?;
                    // Re-quantize into this sequence's stored representation.
                    result[i].push(Some(KvCacheState::from_fp_pair(&k, &v, kv_kind)?));
                }
            } else {
                for row in &mut result {
                    row.push(None);
                }
            }
        }

        Ok(result)
    }

    /// Access the model config.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The KV cache representation this model was constructed with, either
    /// from `CRANE_KV_QUANT` (see [`KvCacheKind::from_env`]) or an explicit
    /// `kv_kind` argument (see [`KvCacheKind::parse`]).
    #[must_use]
    pub fn kv_kind(&self) -> KvCacheKind {
        self.kv_kind
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
                let pad = zero_pad
                    .as_ref()
                    .expect("zero_pad is Some when pad_len > 0 because max_pad_needed > 0")
                    .narrow(2, 0, pad_len)?;
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

    // int8: 1 byte/code + 4 byte f32 scale per head -> 2*48*4*(128+4) =
    // 50,688 bytes/token for the Qwen3-Coder-30B-A3B geometry, about half
    // the 96 KiB (98,304 bytes) f16 pricing above.
    #[test]
    fn test_quantized_kv_bytes_per_token_int8_matches_qwen3_coder_30b_a3b() {
        let config = Config {
            num_hidden_layers: 48,
            num_key_value_heads: 4,
            head_dim: Some(128),
            ..tiny_config()
        };
        assert_eq!(
            config.quantized_kv_bytes_per_token(8),
            2 * 48 * 4 * (128 + 4)
        );
    }

    // int4: nibble-packed codes (head_dim/2 bytes) + 4 byte f32 scale per
    // head -> 2*48*4*(64+4) bytes/token, about half int8's cost.
    #[test]
    fn test_quantized_kv_bytes_per_token_int4_matches_qwen3_coder_30b_a3b() {
        let config = Config {
            num_hidden_layers: 48,
            num_key_value_heads: 4,
            head_dim: Some(128),
            ..tiny_config()
        };
        assert_eq!(
            config.quantized_kv_bytes_per_token(4),
            2 * 48 * 4 * (64 + 4)
        );
    }

    // Both quantized bit widths must cost strictly less than compute-dtype
    // (f16) pricing for the same geometry -- otherwise there's no point
    // ever using quantized pricing.
    #[test]
    fn test_quantized_kv_bytes_per_token_smaller_than_fp16() {
        let config = Config {
            num_hidden_layers: 48,
            num_key_value_heads: 4,
            head_dim: Some(128),
            ..tiny_config()
        };
        let fp16_bytes = config.kv_bytes_per_token(2);
        assert!(config.quantized_kv_bytes_per_token(8) < fp16_bytes);
        assert!(config.quantized_kv_bytes_per_token(4) < fp16_bytes);
    }

    #[test]
    #[should_panic(expected = "bits must be 4 or 8")]
    fn test_quantized_kv_bytes_per_token_rejects_invalid_bits() {
        let _ = tiny_config().quantized_kv_bytes_per_token(16);
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

    // Packed and per-expert tensor names are recognized; router and
    // attention tensor names are not mistaken for expert weights.
    #[test]
    fn test_is_expert_tensor() {
        assert!(is_expert_tensor("blk.0.ffn_gate_exps.weight"));
        assert!(is_expert_tensor("blk.0.ffn_up_exps.weight"));
        assert!(is_expert_tensor("blk.0.ffn_down_exps.weight"));
        assert!(is_expert_tensor("blk.5.ffn_gate.3.weight"));
        assert!(is_expert_tensor("blk.5.ffn_up.3.weight"));
        assert!(is_expert_tensor("blk.5.ffn_down.3.weight"));
        assert!(!is_expert_tensor("blk.0.ffn_gate_inp.weight"));
        assert!(!is_expert_tensor("blk.0.attn_q.weight"));
        assert!(!is_expert_tensor("token_embd.weight"));
        // Dense MLP tensors (no trailing expert index) from a mixed
        // dense+MoE checkpoint must not be mistaken for expert weights.
        assert!(!is_expert_tensor("blk.5.ffn_gate.weight"));
        assert!(!is_expert_tensor("blk.5.ffn_up.weight"));
        assert!(!is_expert_tensor("blk.5.ffn_down.weight"));
    }

    // Layer index is parsed out of the `blk.{i}.` prefix for both expert
    // tensor layouts; non-`blk`-prefixed tensors have no layer.
    #[test]
    fn test_expert_tensor_layer() {
        assert_eq!(expert_tensor_layer("blk.5.ffn_gate_exps.weight"), Some(5));
        assert_eq!(expert_tensor_layer("blk.12.ffn_gate.3.weight"), Some(12));
        assert_eq!(expert_tensor_layer("token_embd.weight"), None);
    }

    // Byte size matches candle's own quantized allocation formula:
    // elem_count / block_size * type_size.
    #[test]
    fn test_gguf_tensor_bytes() {
        use candle_core::quantized::GgmlDType;
        let info = gguf_file::TensorInfo {
            ggml_dtype: GgmlDType::Q4K,
            shape: candle_core::Shape::from(256usize),
            offset: 0,
        };
        let expected = GgmlDType::Q4K.type_size() as u64;
        assert_eq!(gguf_tensor_bytes(&info), expected);
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
        let model = Qwen3Model::new(&cfg, vb, &device, &GpuBudget::default()).expect("new");

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
        let model = Qwen3Model::new(&cfg, vb, &device, &GpuBudget::default()).expect("new");

        assert!(
            model
                .layers
                .iter()
                .all(|l| matches!(l.mlp, MlpOrMoe::Moe(_)))
        );
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

        let mut model_a =
            Qwen3Model::new(&cfg, vb.clone(), &device, &GpuBudget::default()).expect("new");
        let mut model_b =
            Qwen3Model::new_from_model_vb(&cfg, vb.pp("model"), vb, &device, &GpuBudget::default())
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
        let mut model = Qwen3Model::new(&cfg, vb, &device, &GpuBudget::default()).expect("new");

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

    /// With identical weights (one `VarMap` reused across three models),
    /// quantized KV cache's forward output must stay close to the lossless
    /// `Fp` baseline through both prefill and a decode step — not just
    /// structurally valid, but numerically sane. Precise error bounds on
    /// the quantization math itself are covered in
    /// `crate::models::modules::quant_kv_cache`'s own tests; this is an
    /// integration check that the model wiring doesn't introduce its own
    /// corruption (e.g. wrong dtype, wrong axis) on top of that.
    #[test]
    fn test_quantized_kv_forward_close_to_fp() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let mut model_fp = Qwen3Model::new_inner_with_kv_kind(
            &cfg,
            vb.clone(),
            vb.clone(),
            &device,
            &GpuBudget::default(),
            KvCacheKind::Fp,
        )
        .expect("fp model");
        let mut model_int8 = Qwen3Model::new_inner_with_kv_kind(
            &cfg,
            vb.clone(),
            vb.clone(),
            &device,
            &GpuBudget::default(),
            KvCacheKind::Int8,
        )
        .expect("int8 model");
        let mut model_int4 = Qwen3Model::new_inner_with_kv_kind(
            &cfg,
            vb.clone(),
            vb.clone(),
            &device,
            &GpuBudget::default(),
            KvCacheKind::Int4,
        )
        .expect("int4 model");

        let prompt = Tensor::new(&[[1u32, 2, 3, 4, 5]], &device).expect("prompt");
        model_fp.forward(&prompt, 0).expect("fp prefill");
        model_int8.forward(&prompt, 0).expect("int8 prefill");
        model_int4.forward(&prompt, 0).expect("int4 prefill");

        let next = Tensor::new(&[[6u32]], &device).expect("next token");
        let out_fp = model_fp.forward(&next, 5).expect("fp decode");
        let out_int8 = model_int8.forward(&next, 5).expect("int8 decode");
        let out_int4 = model_int4.forward(&next, 5).expect("int4 decode");

        assert_eq!(out_fp.dims(), out_int8.dims());
        assert_eq!(out_fp.dims(), out_int4.dims());

        let scale = max_abs_diff(&out_fp, &Tensor::zeros_like(&out_fp).expect("zeros")).max(1e-3);
        let diff8 = max_abs_diff(&out_fp, &out_int8);
        let diff4 = max_abs_diff(&out_fp, &out_int4);
        assert!(
            diff8 < scale,
            "int8 diverged too far from fp: diff={diff8} scale={scale}"
        );
        assert!(
            diff4 < scale,
            "int4 diverged too far from fp: diff={diff4} scale={scale}"
        );
        assert!(!diff8.is_nan() && !diff4.is_nan());
    }

    /// The capability the KV-swap interface redesign exists for: a
    /// quantized sequence's cache must survive being extracted (simulating
    /// eviction), another sequence running in between, and reinstalling —
    /// producing an *identical* continuation to an uninterrupted run, since
    /// `get_kv_caches`/`set_kv_caches` only narrow/reinstall the existing
    /// quantized tensors (no re-quantization happens on this path, unlike
    /// the batch-decode path — see `setup_batch_decode`'s doc comment).
    #[test]
    fn test_kv_swap_round_trip_preserves_quantized_continuation() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let mut reference = Qwen3Model::new_inner_with_kv_kind(
            &cfg,
            vb.clone(),
            vb.clone(),
            &device,
            &GpuBudget::default(),
            KvCacheKind::Int8,
        )
        .expect("reference model");
        let prompt = Tensor::new(&[[1u32, 2, 3]], &device).expect("prompt");
        reference.forward(&prompt, 0).expect("reference prefill");
        let next = Tensor::new(&[[4u32]], &device).expect("next");
        let reference_out = reference.forward(&next, 3).expect("reference decode");

        let mut model = Qwen3Model::new_inner_with_kv_kind(
            &cfg,
            vb.clone(),
            vb.clone(),
            &device,
            &GpuBudget::default(),
            KvCacheKind::Int8,
        )
        .expect("model");
        model.forward(&prompt, 0).expect("prefill");
        let saved = model.get_kv_caches();
        assert!(saved.iter().all(Option::is_some));
        assert_eq!(saved[0].as_ref().expect("state").kind(), KvCacheKind::Int8);

        // Another sequence runs on the same model in between.
        model.clear_kv_cache();
        let other_prompt = Tensor::new(&[[9u32, 8]], &device).expect("other prompt");
        model
            .forward(&other_prompt, 0)
            .expect("other sequence prefill");

        model.clear_kv_cache();
        model.set_kv_caches(saved);
        let resumed_out = model.forward(&next, 3).expect("resumed decode");

        assert_eq!(reference_out.dims(), resumed_out.dims());
        let diff = max_abs_diff(&reference_out, &resumed_out);
        assert!(
            diff < 1e-4,
            "KV swap round trip diverged from uninterrupted continuation: {diff}"
        );
    }

    /// Quantized KV must also survive the batch-decode path (dequantize at
    /// `setup_batch_decode`, re-quantize at `extract_batch_kv` — see that
    /// pair's doc comments), used when multiple sequences decode together.
    /// Also checks the batched logits stay numerically close to an
    /// identical-weights `Fp` model run through the same batch sequence,
    /// not just structurally valid.
    #[test]
    fn test_batch_decode_round_trips_quantized_kv() {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let mut model = Qwen3Model::new_inner_with_kv_kind(
            &cfg,
            vb.clone(),
            vb.clone(),
            &device,
            &GpuBudget::default(),
            KvCacheKind::Int8,
        )
        .expect("model");
        // Same varmap (identical weights) at `Fp` — the numerical baseline
        // the batch-decode logits must stay close to.
        let mut model_fp = Qwen3Model::new_inner_with_kv_kind(
            &cfg,
            vb.clone(),
            vb.clone(),
            &device,
            &GpuBudget::default(),
            KvCacheKind::Fp,
        )
        .expect("fp model");

        let prompt_a = Tensor::new(&[[1u32, 2, 3]], &device).expect("prompt a");
        let prompt_b = Tensor::new(&[[4u32, 5]], &device).expect("prompt b");

        model.forward(&prompt_a, 0).expect("prefill a");
        let cache_a = model.get_kv_caches();
        model.clear_kv_cache();

        model.forward(&prompt_b, 0).expect("prefill b");
        let cache_b = model.get_kv_caches();
        model.clear_kv_cache();

        model_fp.forward(&prompt_a, 0).expect("fp prefill a");
        let cache_a_fp = model_fp.get_kv_caches();
        model_fp.clear_kv_cache();

        model_fp.forward(&prompt_b, 0).expect("fp prefill b");
        let cache_b_fp = model_fp.get_kv_caches();
        model_fp.clear_kv_cache();

        assert_eq!(cache_a[0].as_ref().expect("a").kind(), KvCacheKind::Int8);
        assert_eq!(cache_b[0].as_ref().expect("b").kind(), KvCacheKind::Int8);

        let (kv_lens, max_kv_len) = model
            .setup_batch_decode(&[vec![cache_a[0].clone()], vec![cache_b[0].clone()]], 4)
            .expect("setup_batch_decode");
        assert_eq!(kv_lens, vec![3, 2]);
        assert_eq!(max_kv_len, 3);

        let (kv_lens_fp, max_kv_len_fp) = model_fp
            .setup_batch_decode(
                &[vec![cache_a_fp[0].clone()], vec![cache_b_fp[0].clone()]],
                4,
            )
            .expect("fp setup_batch_decode");
        assert_eq!(kv_lens_fp, kv_lens);
        assert_eq!(max_kv_len_fp, max_kv_len);

        // Width covers the K length *after* this round's append (kv_lens'
        // max, plus the one new token each sequence appends this round) —
        // matching `crane-serve/src/engine/mod.rs`'s `mask_width =
        // original_max_kv + round + 1` for round 0.
        let mask =
            build_batch_decode_mask(&kv_lens, max_kv_len, max_kv_len + 1, &device, DType::F32)
                .expect("mask");
        let tokens = Tensor::new(&[6u32, 7], &device)
            .expect("tokens")
            .reshape((2, 1))
            .expect("reshape");
        let positions = [3usize, 2usize];
        let logits = model
            .step_batch_decode(&tokens, &positions, mask.as_ref(), None)
            .expect("step_batch_decode");
        assert_eq!(logits.dims()[0], 2);
        let flat = logits
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1");
        assert!(!flat.iter().any(|v| v.is_nan() || v.is_infinite()));

        let logits_fp = model_fp
            .step_batch_decode(&tokens, &positions, mask.as_ref(), None)
            .expect("fp step_batch_decode");
        assert_eq!(logits.dims(), logits_fp.dims());
        let scale =
            max_abs_diff(&logits_fp, &Tensor::zeros_like(&logits_fp).expect("zeros")).max(1e-3);
        let diff = max_abs_diff(&logits, &logits_fp);
        assert!(
            diff < scale,
            "batch-decode int8 diverged too far from fp: diff={diff} scale={scale}"
        );

        let extracted = model
            .extract_batch_kv(&kv_lens, max_kv_len, 1)
            .expect("extract_batch_kv");
        assert_eq!(extracted.len(), 2);
        let state_a = extracted[0][0].as_ref().expect("extracted a");
        let state_b = extracted[1][0].as_ref().expect("extracted b");
        assert_eq!(state_a.kind(), KvCacheKind::Int8);
        assert_eq!(state_b.kind(), KvCacheKind::Int8);
        assert_eq!(state_a.seq_len().expect("seq_len"), 4); // 3 + 1 round
        assert_eq!(state_b.seq_len().expect("seq_len"), 3); // 2 + 1 round
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
        let mut model = Qwen3Model::new(&cfg, vb, &device, &GpuBudget::default()).expect("new");

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
        let mut model = Qwen3Model::new(&cfg, vb, &device, &GpuBudget::default()).expect("new");

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
        let mut model = Qwen3Model::new(&cfg, vb, &device, &GpuBudget::default()).expect("new");

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
        let mut model_single = Qwen3Model::new(
            &cfg,
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
            &device,
            &GpuBudget::default(),
        )
        .expect("new single");
        let mut model_chunked = Qwen3Model::new(
            &cfg,
            VarBuilder::from_varmap(&varmap, DType::F32, &device),
            &device,
            &GpuBudget::default(),
        )
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
}
