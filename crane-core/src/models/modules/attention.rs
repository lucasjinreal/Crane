//! Shared grouped-query attention (`GqaAttention`) for standard causal transformer decoders.
//!
//! Implements the core pre-norm GQA pattern used across Qwen3-TTS, Qwen2.5, and related models.
//!
//! # Weight names
//!
//! The [`VarBuilder`] passed to [`GqaAttention::new`] is expected to hold:
//! - `q_proj.weight` (and `q_proj.bias` when `cfg.qkv_bias = true`)
//! - `k_proj.weight` (and `k_proj.bias` when `cfg.qkv_bias = true`)
//! - `v_proj.weight` (and `v_proj.bias` when `cfg.qkv_bias = true`)
//! - `o_proj.weight` (and `o_proj.bias` when `cfg.o_bias = true`)
//! - `q_norm.weight` and `k_norm.weight` (only when `cfg.use_qk_norm = true`)
//!
//! These names match the safetensors checkpoint layout for Qwen3-TTS and Qwen2.5.

use candle_core::{DType, Module, Result, Tensor, D};
use candle_nn::attention::AttnMask;
use candle_nn::VarBuilder;

use super::flash_attn::dispatch_flash_attn;
use super::kv_cache;
use crate::models::utils::repeat_kv;
use crate::models::with_tracing::{linear_b, Linear, RmsNorm};

/// Which rotary position embedding convention to apply in [`GqaAttention::forward`].
#[derive(Debug, Clone, Copy)]
pub enum RopeMode {
    /// No rotary embeddings — `cos_sin` is ignored.
    None,
    /// Half-split rotation (`d` paired with `d + half_dim`). Used by
    /// HuggingFace-converted checkpoints (Qwen2.5, Qwen3-TTS).
    HalfSplit,
    /// Interleaved rotation (`2d` paired with `2d + 1`). Required by
    /// Mistral-native checkpoints such as Voxtral TTS.
    Interleaved,
}

/// Configuration for [`GqaAttention`].
///
/// All dimension and feature-flag information needed to construct the layer.
/// Causality is enforced by the caller via `attention_mask`, not by the
/// attention layer itself.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Copy)]
pub struct AttentionConfig {
    /// Hidden dimension — input and output size of the attention layer.
    pub dim: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads. Must evenly divide `n_heads`.
    pub n_kv_heads: usize,
    /// Dimension of each attention head (`dim / n_heads` for most models).
    pub head_dim: usize,
    /// Whether Q, K, V projections include a bias term.
    pub qkv_bias: bool,
    /// Whether the output projection includes a bias term.
    pub o_bias: bool,
    /// Which rotary position embedding convention to apply. When set to any
    /// variant other than [`RopeMode::None`], `cos_sin` must be `Some` in
    /// [`GqaAttention::forward`].
    pub rope_mode: RopeMode,
    /// Whether to apply per-head RMS normalization to Q and K after reshaping.
    pub use_qk_norm: bool,
    /// Epsilon for RMS normalizations. Used by QK-norm (when `use_qk_norm =
    /// true`) and by [`TransformerBlock`] for the block's input and
    /// post-attention layer norms.
    pub norm_eps: f64,
}

/// Grouped-query attention with an accumulated KV cache.
///
/// See the [module documentation](self) for the full forward-pass description
/// and the expected weight-name layout.
///
/// Call [`GqaAttention::clear_kv_cache`] between independent sequences.
#[derive(Debug)]
pub struct GqaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    cfg: AttentionConfig,
    /// Precomputed `cfg.n_heads / cfg.n_kv_heads`.
    n_kv_groups: usize,
    kv_cache: Option<(Tensor, Tensor)>,
    /// Number of valid (filled) positions in `kv_cache`'s pre-allocated buffer.
    cache_seq_len: usize,
}

impl Clone for GqaAttention {
    /// Clone the layer with the KV cache reset to empty.
    ///
    /// Projection weights and norms are cloned as-is, but the accumulated KV
    /// cache is **not** carried over — the clone starts with a fresh cache.
    /// This prevents an accidental deep-copy of potentially large cached tensors.
    fn clone(&self) -> Self {
        Self {
            q_proj: self.q_proj.clone(),
            k_proj: self.k_proj.clone(),
            v_proj: self.v_proj.clone(),
            o_proj: self.o_proj.clone(),
            q_norm: self.q_norm.clone(),
            k_norm: self.k_norm.clone(),
            cfg: self.cfg,
            n_kv_groups: self.n_kv_groups,
            kv_cache: None,
            cache_seq_len: 0,
        }
    }
}

impl GqaAttention {
    /// Create a new `GqaAttention` layer.
    ///
    /// # Arguments
    ///
    /// * `cfg` — Configuration controlling dimensions, bias, `RoPE`, and QK-norm.
    /// * `vb` — [`VarBuilder`] scoped to this layer. Loads the weights listed in
    ///   the [module documentation](self).
    ///
    /// # Errors
    ///
    /// Returns a candle error if `cfg.n_heads` is not divisible by `cfg.n_kv_heads`,
    /// or if any weight-loading operation fails.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(cfg: AttentionConfig, vb: VarBuilder) -> Result<Self> {
        if !cfg.n_heads.is_multiple_of(cfg.n_kv_heads) {
            candle_core::bail!(
                "n_heads ({}) must be divisible by n_kv_heads ({})",
                cfg.n_heads,
                cfg.n_kv_heads
            );
        }
        let q_proj = linear_b(
            cfg.dim,
            cfg.n_heads * cfg.head_dim,
            cfg.qkv_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.dim,
            cfg.n_kv_heads * cfg.head_dim,
            cfg.qkv_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.dim,
            cfg.n_kv_heads * cfg.head_dim,
            cfg.qkv_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            cfg.n_heads * cfg.head_dim,
            cfg.dim,
            cfg.o_bias,
            vb.pp("o_proj"),
        )?;
        let (q_norm, k_norm) = if cfg.use_qk_norm {
            (
                Some(RmsNorm::new(cfg.head_dim, cfg.norm_eps, vb.pp("q_norm"))?),
                Some(RmsNorm::new(cfg.head_dim, cfg.norm_eps, vb.pp("k_norm"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            n_kv_groups: cfg.n_heads / cfg.n_kv_heads,
            cfg,
            kv_cache: None,
            cache_seq_len: 0,
        })
    }

    /// Run grouped-query attention on `hidden_states`.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` — Input tensor of shape `[batch, seq_len, dim]`.
    /// * `cos_sin` — Pre-computed `RoPE` tables `(cos, sin)`, each of shape
    ///   `[seq_len, head_dim / 2]`. May be in any floating-point dtype; they are
    ///   cast to the query dtype internally so BF16/F16 models work without an
    ///   explicit conversion by the caller. **Must be `Some` when `rope_mode !=
    ///   RopeMode::None`.**
    /// * `attention_mask` — Optional additive mask broadcastable to
    ///   `[batch, n_heads, q_seq_len, kv_seq_len]`. Set blocked positions to
    ///   `f32::NEG_INFINITY` (or a sufficiently large negative value) to prevent
    ///   attention flow. During single-token decode (`q_seq_len == 1`), the
    ///   mask's head dimension must be `1` (i.e. must not vary per head) —
    ///   the decode fast paths broadcast one mask row across all heads.
    ///
    /// # Returns
    ///
    /// A tensor of shape `[batch, seq_len, dim]` — the attended and projected output.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `cos_sin` is `None` but `rope_mode` is not
    /// `RopeMode::None`, or if any tensor operation fails.
    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        cos_sin: Option<(&Tensor, &Tensor)>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, q_seq_len, _) = hidden_states.dims3()?;

        // 1. Project Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // 2. Reshape to [batch, heads, q_seq_len, head_dim]
        let q = q
            .reshape((b_sz, q_seq_len, self.cfg.n_heads, self.cfg.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_seq_len, self.cfg.n_kv_heads, self.cfg.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_seq_len, self.cfg.n_kv_heads, self.cfg.head_dim))?
            .transpose(1, 2)?;

        // 3. Optional per-head QK-norm (RMS norm applied to the head_dim dimension)
        let q = match &self.q_norm {
            Some(norm) => norm.forward(&q)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(&k)?,
            None => k,
        };

        // 4. Optional RoPE — cast cos/sin to Q dtype for BF16/F16 models
        let (q, k) = match self.cfg.rope_mode {
            RopeMode::None => (q, k),
            mode => {
                let Some((cos, sin)) = cos_sin else {
                    candle_core::bail!(
                        "`cos_sin` must be `Some` when `rope_mode` is not `RopeMode::None`"
                    )
                };
                let cos = cos.to_dtype(q.dtype())?;
                let sin = sin.to_dtype(q.dtype())?;
                let rope_fn: fn(
                    &Tensor,
                    &Tensor,
                    &Tensor,
                ) -> candle_core::Result<Tensor> = match mode {
                    RopeMode::HalfSplit => candle_nn::rotary_emb::rope,
                    RopeMode::Interleaved => candle_nn::rotary_emb::rope_i,
                    RopeMode::None => unreachable!(),
                };
                (
                    rope_fn(&q.contiguous()?, &cos, &sin)?,
                    rope_fn(&k.contiguous()?, &cos, &sin)?,
                )
            }
        };

        // 5. KV cache: pre-allocated buffer with in-place `slice_set` writes
        let (k, v) = self.update_kv_cache(&k, &v)?;

        // 6. Scaled dot-product attention
        let n_rep = self.n_kv_groups;
        // head_dim is bounded by model config (typically 2-256), well within
        // f64's exact integer range.
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (self.cfg.head_dim as f64).sqrt();
        // 1/sqrt(head_dim) is always small and positive; f64->f32 here only
        // drops precision flash_attn's own f32 accumulator would discard anyway.
        #[allow(clippy::cast_possible_truncation)]
        let scale_f32 = scale as f32;

        if q_seq_len == 1 && b_sz == 1 && q.device().is_cpu() {
            // ── Fused flash attention for decode (seq_len=1), CPU only ──
            // candle's cpu_flash kernel streams K/V with online softmax
            // (O(head_dim) working set instead of materializing an O(S)
            // scores tensor 3 times), and handles GQA natively via integer
            // division — no K/V repeat needed. b_sz == 1 only: candle's
            // flash_attn hard-errors for B>1 with an explicit Mask tensor
            // (only Causal/None are allowed).
            return self.flash_attn_decode(&q, &k, &v, attention_mask, scale_f32, b_sz);
        }

        // No unconditional flash-attn prefill fast path here: unlike the
        // qwen3-specific model (always a causal decoder), `GqaAttention` is
        // documented to leave causality entirely up to the caller's
        // `attention_mask` — `None` means full (non-causal) attention, as
        // used by Voxtral's bidirectional `AcousticTransformer`. Guessing
        // `AttnMask::Causal` whenever the mask is absent would silently
        // break that contract for any single-sequence CPU caller.

        if n_rep > 1 && q_seq_len == 1 {
            // ── GQA-grouped SDPA for decode (seq_len=1), GPU fallback ──
            // Reshape Q to group queries with their KV head instead of
            // repeating K/V n_rep times.
            let q_g = (q.reshape((b_sz, self.cfg.n_kv_heads, n_rep, self.cfg.head_dim))?
                * scale)?;
            let k_t = k.transpose(2, 3)?;
            let attn_weights = q_g.matmul(&k_t)?;
            let attn_weights = match attention_mask {
                Some(mask) => attn_weights.broadcast_add(mask)?,
                None => attn_weights,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_output = attn_weights.matmul(&v)?;

            // [B, n_kv_heads, n_rep, D] → [B, 1, H*D]; flattening (n_kv_heads,
            // n_rep, D) matches (H, D) since H = n_kv_heads * n_rep.
            let attn_output =
                attn_output.reshape((b_sz, 1, self.cfg.n_heads * self.cfg.head_dim))?;
            return self.o_proj.forward(&attn_output);
        }

        // ── Standard SDPA for prefill or when n_rep == 1 ──
        // Softmax in F32 for numerical stability.
        let k = repeat_kv(k, n_rep)?.contiguous()?;
        let v = repeat_kv(v, n_rep)?.contiguous()?;
        // Q may be non-contiguous after transpose(1,2) when RoPE was not applied.
        let q = q.contiguous()?;
        let attn_weights = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? * scale)?;
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };
        let input_dtype = attn_weights.dtype();
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32)?)?
            .to_dtype(input_dtype)?;
        let attn_output = attn_weights.matmul(&v)?;

        // 7. Reassemble heads and project back to the hidden dimension
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b_sz, q_seq_len, ()))?;
        self.o_proj.forward(&attn_output)
    }

    /// Clear the accumulated KV cache.
    ///
    /// Must be called between independent sequences to avoid stale context
    /// contaminating the next sequence.
    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
        self.cache_seq_len = 0;
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

    /// Fused flash-attention decode path: `q_seq_len == 1`, `b_sz == 1`, CPU only.
    ///
    /// `q`, `k`, `v` are in `[batch, heads, seq, head_dim]` layout. See the
    /// call site in [`Self::forward`] for why this path is gated the way it is.
    fn flash_attn_decode(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        scale_f32: f32,
        b_sz: usize,
    ) -> Result<Tensor> {
        // Non-contiguous is fine — the decode kernel indexes by stride.
        let q_bshd = q.transpose(1, 2)?;
        let k_bshd = k.transpose(1, 2)?;
        let v_bshd = v.transpose(1, 2)?;

        let mask = match attention_mask {
            Some(mask) => {
                debug_assert!(
                    mask.dim(1).is_ok_and(|d| d == 1),
                    "CPU flash-attn decode broadcasts one mask row across all heads; \
                     a mask with head dim != 1 would be silently misapplied"
                );
                // AttnMask::Mask takes ownership; Tensor is Arc-backed, so
                // this is a refcount bump, not a data copy.
                AttnMask::Mask(mask.clone())
            }
            None => AttnMask::None,
        };

        let attn_output = dispatch_flash_attn(&q_bshd, &k_bshd, &v_bshd, scale_f32, mask)?;

        // flash_attn output is BHSD [B, H, 1, D] → [B, 1, H*D].
        let attn_output = attn_output
            .reshape((b_sz, self.cfg.n_heads, self.cfg.head_dim))?
            .reshape((b_sz, 1, self.cfg.n_heads * self.cfg.head_dim))?;
        self.o_proj.forward(&attn_output)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    // Base config used by most tests: dim=16, 4 heads, head_dim=4, no bias/rope/norm.
    fn base_cfg() -> AttentionConfig {
        AttentionConfig {
            dim: 16,
            n_heads: 4,
            n_kv_heads: 4,
            head_dim: 4,
            qkv_bias: false,
            o_bias: false,
            rope_mode: RopeMode::None,
            use_qk_norm: false,
            norm_eps: 1e-6,
        }
    }

    /// Build a `VarBuilder` with zero-valued projection weights (and ones for
    /// QK-norm weights) that matches the given `cfg`.
    ///
    /// Includes QKV bias tensors when `cfg.qkv_bias = true`, output bias when
    /// `cfg.o_bias = true`, and norm weights only when `cfg.use_qk_norm = true`.
    fn make_vb(cfg: &AttentionConfig) -> VarBuilder<'static> {
        let device = &Device::Cpu;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let q_out = cfg.n_heads * cfg.head_dim;
        let kv_out = cfg.n_kv_heads * cfg.head_dim;

        let zeros =
            |shape: (usize, usize)| Tensor::zeros(shape, DType::F32, device).expect("zeros weight");
        let zeros1 = |n: usize| Tensor::zeros(n, DType::F32, device).expect("zeros bias");
        let ones1 = |n: usize| Tensor::ones(n, DType::F32, device).expect("ones norm");

        t.insert("q_proj.weight".into(), zeros((q_out, cfg.dim)));
        t.insert("k_proj.weight".into(), zeros((kv_out, cfg.dim)));
        t.insert("v_proj.weight".into(), zeros((kv_out, cfg.dim)));
        t.insert("o_proj.weight".into(), zeros((cfg.dim, q_out)));

        if cfg.qkv_bias {
            t.insert("q_proj.bias".into(), zeros1(q_out));
            t.insert("k_proj.bias".into(), zeros1(kv_out));
            t.insert("v_proj.bias".into(), zeros1(kv_out));
        }
        if cfg.o_bias {
            t.insert("o_proj.bias".into(), zeros1(cfg.dim));
        }

        if cfg.use_qk_norm {
            t.insert("q_norm.weight".into(), ones1(cfg.head_dim));
            t.insert("k_norm.weight".into(), ones1(cfg.head_dim));
        }

        VarBuilder::from_tensors(t, DType::F32, device)
    }

    /// Create a `GqaAttention` with zero projection weights from `cfg`.
    fn zeros_attn(cfg: AttentionConfig) -> GqaAttention {
        GqaAttention::new(cfg, make_vb(&cfg)).expect("GqaAttention::new")
    }

    /// Build a `VarBuilder` with small non-zero weights for tests that need
    /// non-trivial outputs. All weights follow `(i + 1) * scale`.
    ///
    /// Includes QKV bias tensors when `cfg.qkv_bias = true`, output bias when
    /// `cfg.o_bias = true`, and norm weights only when `cfg.use_qk_norm = true`
    /// (mirrors `make_vb`'s conditional logic).
    fn nonzero_vb(cfg: &AttentionConfig, scale: f32) -> VarBuilder<'static> {
        let device = &Device::Cpu;
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let q_out = cfg.n_heads * cfg.head_dim;
        let kv_out = cfg.n_kv_heads * cfg.head_dim;

        let fill = |rows: usize, cols: usize| {
            let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 + 1.0) * scale).collect();
            Tensor::from_vec(data, (rows, cols), device).expect("weight")
        };
        let fill1 = |n: usize| {
            let data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * scale).collect();
            Tensor::from_vec(data, n, device).expect("bias")
        };

        t.insert("q_proj.weight".into(), fill(q_out, cfg.dim));
        t.insert("k_proj.weight".into(), fill(kv_out, cfg.dim));
        t.insert("v_proj.weight".into(), fill(kv_out, cfg.dim));
        t.insert("o_proj.weight".into(), fill(cfg.dim, q_out));

        if cfg.qkv_bias {
            t.insert("q_proj.bias".into(), fill1(q_out));
            t.insert("k_proj.bias".into(), fill1(kv_out));
            t.insert("v_proj.bias".into(), fill1(kv_out));
        }
        if cfg.o_bias {
            t.insert("o_proj.bias".into(), fill1(cfg.dim));
        }

        if cfg.use_qk_norm {
            let ones1 = |n: usize| Tensor::ones(n, DType::F32, device).expect("ones norm");
            t.insert("q_norm.weight".into(), ones1(cfg.head_dim));
            t.insert("k_norm.weight".into(), ones1(cfg.head_dim));
        }

        VarBuilder::from_tensors(t, DType::F32, device)
    }

    /// Build an input tensor where every element is distinct: `x[b][t][d] = (index + 1) * 0.01`.
    ///
    /// Using position-varying values ensures that different token positions produce
    /// different Q, K, V projections, which is required for tests that verify the
    /// effect of masking, KV cache, QK-norm, or RoPE.
    fn varying_input(batch: usize, seq: usize, dim: usize) -> Tensor {
        let data: Vec<f32> = (0..batch * seq * dim)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        Tensor::from_vec(data, (batch, seq, dim), &Device::Cpu).expect("varying_input")
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

    // -------------------------------------------------------------------------
    // Shape tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_output_shape() {
        let mut attn = zeros_attn(base_cfg());
        let x = Tensor::zeros((2, 5, 16), DType::F32, &Device::Cpu).expect("zeros");
        let y = attn.forward(&x, None, None).expect("forward");
        assert_eq!(y.dims(), &[2, 5, 16]);
    }

    #[test]
    fn test_output_shape_gqa() {
        // 8 query heads, 2 KV heads → n_kv_groups = 4
        let cfg = AttentionConfig {
            dim: 16,
            n_heads: 8,
            n_kv_heads: 2,
            head_dim: 2,
            ..base_cfg()
        };
        let mut attn = zeros_attn(cfg);
        let x = Tensor::zeros((1, 3, 16), DType::F32, &Device::Cpu).expect("zeros");
        let y = attn.forward(&x, None, None).expect("forward");
        assert_eq!(y.dims(), &[1, 3, 16]);
    }

    #[test]
    fn test_single_token_decode_shape() {
        let mut attn = zeros_attn(base_cfg());
        let x = Tensor::zeros((1, 1, 16), DType::F32, &Device::Cpu).expect("zeros");
        let y = attn.forward(&x, None, None).expect("forward");
        assert_eq!(y.dims(), &[1, 1, 16]);
    }

    // -------------------------------------------------------------------------
    // Zero-weight / zero-input tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_zero_weights_give_zero_output() {
        let mut attn = zeros_attn(base_cfg());
        let x = Tensor::ones((1, 4, 16), DType::F32, &Device::Cpu).expect("ones");
        let y = attn.forward(&x, None, None).expect("forward");
        let max_val: f32 = y
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            max_val < 1e-8,
            "expected zero output for zero weights, got max={max_val}"
        );
    }

    #[test]
    fn test_zero_input_gives_zero_output() {
        let cfg = base_cfg();
        let mut attn = GqaAttention::new(cfg, nonzero_vb(&cfg, 0.01)).expect("new");
        let x = Tensor::zeros((2, 3, 16), DType::F32, &Device::Cpu).expect("zeros");
        let y = attn.forward(&x, None, None).expect("forward");
        let max_val: f32 = y
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            max_val < 1e-8,
            "expected zero output for zero input, got max={max_val}"
        );
    }

    #[test]
    fn test_nonzero_input_gives_nonzero_output() {
        // Verify the forward pass produces non-zero output for non-zero inputs and weights.
        // This complements the zero-input test by checking the path actually computes
        // something meaningful when both inputs and weights are non-trivial.
        let cfg = base_cfg();
        let mut attn = GqaAttention::new(cfg, nonzero_vb(&cfg, 0.01)).expect("new");
        let x = varying_input(1, 4, cfg.dim);
        let y = attn.forward(&x, None, None).expect("forward");
        let max_val: f32 = y
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar()
            .expect("scalar");
        assert!(
            max_val > 1e-6,
            "non-zero input with non-zero weights must produce non-zero output, got max={max_val}"
        );
    }

    // -------------------------------------------------------------------------
    // QK-norm tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_qk_norm_enabled_output_shape() {
        let cfg = AttentionConfig {
            use_qk_norm: true,
            ..base_cfg()
        };
        let mut attn = zeros_attn(cfg);
        let x = Tensor::zeros((1, 5, 16), DType::F32, &Device::Cpu).expect("zeros");
        let y = attn.forward(&x, None, None).expect("forward with qk_norm");
        assert_eq!(y.dims(), &[1, 5, 16]);
    }

    #[test]
    fn test_qk_norm_disabled_no_norm_weights_needed() {
        // VarBuilder holds no norm weights; construction must succeed when use_qk_norm=false.
        let cfg = AttentionConfig {
            use_qk_norm: false,
            ..base_cfg()
        };
        let vb = make_vb(&cfg); // norm keys absent
        let mut attn = GqaAttention::new(cfg, vb).expect("new without norm weights");
        let x = Tensor::zeros((1, 2, 16), DType::F32, &Device::Cpu).expect("zeros");
        let y = attn.forward(&x, None, None).expect("forward");
        assert_eq!(y.dims(), &[1, 2, 16]);
    }

    #[test]
    fn test_qk_norm_changes_output() {
        // Same projection weights; QK-norm normalizes Q and K to unit RMS, which
        // changes the attention weight distribution for non-trivial (position-varying)
        // inputs. Use distinct embeddings per token so Q/K projections differ.
        let device = &Device::Cpu;
        let cfg_no_norm = base_cfg();
        let cfg_norm = AttentionConfig {
            use_qk_norm: true,
            ..base_cfg()
        };
        let mut attn_no_norm =
            GqaAttention::new(cfg_no_norm, nonzero_vb(&cfg_no_norm, 0.1)).expect("no norm");

        let q_out = cfg_norm.n_heads * cfg_norm.head_dim;
        let kv_out = cfg_norm.n_kv_heads * cfg_norm.head_dim;
        let vb_norm = {
            let mut map: HashMap<String, Tensor> = HashMap::new();
            let fill = |rows: usize, cols: usize| {
                let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 + 1.0) * 0.1).collect();
                Tensor::from_vec(data, (rows, cols), device).expect("weight")
            };
            map.insert("q_proj.weight".into(), fill(q_out, cfg_norm.dim));
            map.insert("k_proj.weight".into(), fill(kv_out, cfg_norm.dim));
            map.insert("v_proj.weight".into(), fill(kv_out, cfg_norm.dim));
            map.insert("o_proj.weight".into(), fill(cfg_norm.dim, q_out));
            map.insert(
                "q_norm.weight".into(),
                Tensor::ones(cfg_norm.head_dim, DType::F32, device).expect("ones"),
            );
            map.insert(
                "k_norm.weight".into(),
                Tensor::ones(cfg_norm.head_dim, DType::F32, device).expect("ones"),
            );
            VarBuilder::from_tensors(map, DType::F32, device)
        };
        let mut attn_norm = GqaAttention::new(cfg_norm, vb_norm).expect("with norm");

        // Position-varying input: each token has a different embedding so Q/K differ per position.
        let x = varying_input(1, 3, cfg_norm.dim);
        let y_no_norm = attn_no_norm
            .forward(&x, None, None)
            .expect("no norm forward");
        let y_norm = attn_norm.forward(&x, None, None).expect("norm forward");
        assert!(
            max_abs_diff(&y_no_norm, &y_norm) > 1e-6,
            "qk_norm must change the output for non-trivial inputs"
        );
    }

    // -------------------------------------------------------------------------
    // RoPE tests
    // -------------------------------------------------------------------------

    /// Build trivial cos=1, sin=0 tables (position-0 RoPE is the identity).
    fn identity_cos_sin(seq_len: usize, half_dim: usize) -> (Tensor, Tensor) {
        let cos = Tensor::ones((seq_len, half_dim), DType::F32, &Device::Cpu).expect("cos ones");
        let sin = Tensor::zeros((seq_len, half_dim), DType::F32, &Device::Cpu).expect("sin zeros");
        (cos, sin)
    }

    #[test]
    fn test_rope_enabled_output_shape() {
        let cfg = AttentionConfig {
            rope_mode: RopeMode::HalfSplit,
            ..base_cfg()
        };
        let mut attn = zeros_attn(cfg);
        let x = Tensor::zeros((1, 3, 16), DType::F32, &Device::Cpu).expect("zeros");
        let (cos, sin) = identity_cos_sin(3, cfg.head_dim / 2);
        let y = attn
            .forward(&x, Some((&cos, &sin)), None)
            .expect("forward with rope");
        assert_eq!(y.dims(), &[1, 3, 16]);
    }

    #[test]
    fn test_interleaved_rope_output_shape() {
        let cfg = AttentionConfig {
            rope_mode: RopeMode::Interleaved,
            ..base_cfg()
        };
        let mut attn = zeros_attn(cfg);
        let x = Tensor::zeros((1, 3, 16), DType::F32, &Device::Cpu).expect("zeros");
        let (cos, sin) = identity_cos_sin(3, cfg.head_dim / 2);
        let y = attn
            .forward(&x, Some((&cos, &sin)), None)
            .expect("forward with interleaved rope");
        assert_eq!(y.dims(), &[1, 3, 16]);
    }

    #[test]
    fn test_rope_disabled_accepts_none_cos_sin() {
        let cfg = AttentionConfig {
            rope_mode: RopeMode::None,
            ..base_cfg()
        };
        let mut attn = zeros_attn(cfg);
        let x = Tensor::zeros((1, 3, 16), DType::F32, &Device::Cpu).expect("zeros");
        let y = attn.forward(&x, None, None).expect("forward without rope");
        assert_eq!(y.dims(), &[1, 3, 16]);
    }

    #[test]
    fn test_rope_disabled_ignores_cos_sin() {
        // When rope_mode=None, cos_sin is silently ignored regardless of whether
        // it is Some or None. Callers may always pass cos_sin without branching.
        let cfg = base_cfg(); // rope_mode: RopeMode::None
        let vb1 = nonzero_vb(&cfg, 0.02);
        let vb2 = nonzero_vb(&cfg, 0.02);

        let mut attn_none = GqaAttention::new(cfg, vb1).expect("attn none");
        let mut attn_some = GqaAttention::new(cfg, vb2).expect("attn some");

        let x = varying_input(1, 3, cfg.dim);
        let (cos, sin) = identity_cos_sin(3, cfg.head_dim / 2);

        let y_none = attn_none.forward(&x, None, None).expect("no cos_sin");
        let y_some = attn_some
            .forward(&x, Some((&cos, &sin)), None)
            .expect("with cos_sin ignored");

        assert!(
            max_abs_diff(&y_none, &y_some) < 1e-6,
            "cos_sin must be silently ignored when rope_mode=None"
        );
    }

    #[test]
    fn test_rope_enabled_none_cos_sin_errors() {
        let cfg = AttentionConfig {
            rope_mode: RopeMode::HalfSplit,
            ..base_cfg()
        };
        let mut attn = zeros_attn(cfg);
        let x = Tensor::zeros((1, 3, 16), DType::F32, &Device::Cpu).expect("zeros");
        assert!(
            attn.forward(&x, None, None).is_err(),
            "rope_mode != None with cos_sin=None must return an error"
        );
    }

    #[test]
    fn test_rope_identity_at_position_zero_preserves_output() {
        // cos=1, sin=0 is a no-op rotation, so RoPE at position 0 must not
        // change the output compared to the same model without RoPE.
        let cfg_rope = AttentionConfig {
            rope_mode: RopeMode::HalfSplit,
            ..base_cfg()
        };
        let cfg_no_rope = base_cfg();

        // Identical non-zero weights for both variants
        let cfg = cfg_rope;
        let vb_rope = nonzero_vb(&cfg, 0.05);
        let vb_no_rope = nonzero_vb(&cfg_no_rope, 0.05);

        let mut attn_rope = GqaAttention::new(cfg_rope, vb_rope).expect("rope attn");
        let mut attn_no_rope = GqaAttention::new(cfg_no_rope, vb_no_rope).expect("no rope attn");

        let x = Tensor::ones((1, 4, 16), DType::F32, &Device::Cpu).expect("ones");
        let (cos, sin) = identity_cos_sin(4, cfg.head_dim / 2);
        let y_rope = attn_rope
            .forward(&x, Some((&cos, &sin)), None)
            .expect("rope forward");
        let y_no_rope = attn_no_rope
            .forward(&x, None, None)
            .expect("no rope forward");
        assert!(
            max_abs_diff(&y_rope, &y_no_rope) < 1e-5,
            "position-0 RoPE (identity rotation) must not change the output"
        );
    }

    #[test]
    fn test_rope_non_identity_changes_output() {
        // Position-varying RoPE rotates Q and K at each position by a different angle,
        // changing the attention pattern. Verify this produces a different output from
        // the identical model without RoPE.
        //
        // For the change to be observable, two conditions are needed:
        // 1. cos/sin must vary per position (otherwise all positions get the same
        //    rotation and Q @ K^T is unchanged by the orthogonal transform).
        // 2. Q/K must be non-zero (position-varying input ensures different per-token Q/K).
        let device = &Device::Cpu;
        let cfg_rope = AttentionConfig {
            rope_mode: RopeMode::HalfSplit,
            ..base_cfg()
        };
        let cfg_no_rope = base_cfg();
        let vb_rope = nonzero_vb(&cfg_rope, 0.05);
        let vb_no_rope = nonzero_vb(&cfg_no_rope, 0.05);

        let mut attn_rope = GqaAttention::new(cfg_rope, vb_rope).expect("rope attn");
        let mut attn_no_rope = GqaAttention::new(cfg_no_rope, vb_no_rope).expect("no rope attn");

        // Position-varying input so Q and K differ per position.
        let x = varying_input(1, 4, cfg_rope.dim);

        // Build position-varying cos/sin: for head_dim=4, half_dim=2.
        // inv_freq = [1.0, 0.1] (theta=10, dim=4).
        // At position t: freq[i] = t * inv_freq[i].
        let inv_freq = [1.0_f32, 0.1_f32];
        let seq_len = 4usize;
        let half_dim = cfg_rope.head_dim / 2;
        let mut cos_data = Vec::with_capacity(seq_len * half_dim);
        let mut sin_data = Vec::with_capacity(seq_len * half_dim);
        for t in 0..seq_len {
            for &freq in &inv_freq {
                let angle = (t as f32) * freq;
                cos_data.push(angle.cos());
                sin_data.push(angle.sin());
            }
        }
        let cos = Tensor::from_vec(cos_data, (seq_len, half_dim), device).expect("cos");
        let sin = Tensor::from_vec(sin_data, (seq_len, half_dim), device).expect("sin");

        let y_rope = attn_rope
            .forward(&x, Some((&cos, &sin)), None)
            .expect("rope forward");
        let y_no_rope = attn_no_rope
            .forward(&x, None, None)
            .expect("no rope forward");
        assert!(
            max_abs_diff(&y_rope, &y_no_rope) > 1e-6,
            "position-varying RoPE must change the output"
        );
    }

    // -------------------------------------------------------------------------
    // KV cache tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_kv_cache_decode_step_shape() {
        // Prefill 4 tokens, then decode 1 — output must be [1, 1, dim].
        let mut attn = zeros_attn(base_cfg());
        let device = &Device::Cpu;
        let prefill = Tensor::zeros((1, 4, 16), DType::F32, device).expect("prefill");
        attn.forward(&prefill, None, None).expect("prefill forward");
        let decode = Tensor::zeros((1, 1, 16), DType::F32, device).expect("decode token");
        let y = attn.forward(&decode, None, None).expect("decode forward");
        assert_eq!(
            y.dims(),
            &[1, 1, 16],
            "decode output must have seq_len=1, not the accumulated KV length"
        );
    }

    #[test]
    fn test_kv_cache_affects_output() {
        // A prefill pass changes the KV cache; the subsequent decode step must
        // produce a different result than a fresh single-token forward.
        let cfg = base_cfg();
        let vb1 = nonzero_vb(&cfg, 0.02);
        let vb2 = nonzero_vb(&cfg, 0.02);

        let mut attn_fresh = GqaAttention::new(cfg, vb1).expect("fresh");
        let mut attn_cached = GqaAttention::new(cfg, vb2).expect("cached");

        // Position-varying decode token and prefill so each cached KV entry is distinct.
        // With uniform inputs the cached K/V entries would all equal the decode K/V, and
        // softmax over identical logits gives the same output regardless of KV length.
        let token = varying_input(1, 1, cfg.dim);
        let prefill = varying_input(1, 4, cfg.dim);
        attn_cached
            .forward(&prefill, None, None)
            .expect("prefill forward");

        let y_fresh = attn_fresh
            .forward(&token, None, None)
            .expect("fresh forward");
        let y_cached = attn_cached
            .forward(&token, None, None)
            .expect("cached forward");

        // Both outputs must have the same shape
        assert_eq!(y_fresh.dims(), y_cached.dims());
        // The cached state adds 4 more KV entries from a different context, which
        // changes the attention distribution and hence the output.
        assert!(
            max_abs_diff(&y_fresh, &y_cached) > 1e-6,
            "cached context must change the attention output"
        );
    }

    #[test]
    fn test_kv_cache_buffer_overflow() {
        // `update_kv_cache` pre-allocates a buffer with 256 slots of room and
        // reallocates (cat + zeros) once that room is exhausted. Decode past
        // that boundary and verify the output still matches a single-shot
        // masked prefill over the same tokens — i.e. the reallocation
        // preserves all previously cached K/V content.
        let cfg = base_cfg();
        let vb1 = nonzero_vb(&cfg, 0.02);
        let vb2 = nonzero_vb(&cfg, 0.02);
        let device = &Device::Cpu;

        let mut attn_prefill = GqaAttention::new(cfg, vb1).expect("prefill attn");
        let mut attn_incr = GqaAttention::new(cfg, vb2).expect("incremental attn");

        // First prefill call allocates a buffer sized seq_len + 256; decoding
        // 256 more one-token steps exactly exhausts that room, so the next
        // step (the 257th) must hit the overflow/reallocation branch.
        let prefill_len = 4usize;
        let decode_steps = 257usize;
        let seq_len = prefill_len + decode_steps;
        let x = varying_input(1, seq_len, cfg.dim);

        let n = seq_len;
        let mask_data: Vec<f32> = (0..n * n)
            .map(|i| {
                if (i % n) > (i / n) {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let mask = Tensor::from_vec(mask_data, (1, 1, n, n), device).expect("causal mask");
        let out_prefill = attn_prefill
            .forward(&x, None, Some(&mask))
            .expect("prefill forward");
        let last_prefill = out_prefill.narrow(1, seq_len - 1, 1).expect("narrow prefill");

        let prefill_tokens = x.narrow(1, 0, prefill_len).expect("prefill tokens");
        let mut out_last = attn_incr
            .forward(&prefill_tokens, None, None)
            .expect("incremental prefill");
        for i in 0..decode_steps {
            let token = x.narrow(1, prefill_len + i, 1).expect("decode token");
            out_last = attn_incr.forward(&token, None, None).expect("decode step");
        }
        // Unnormalized attention over 261 accumulated positions makes the
        // output magnitude large (no LayerNorm at this layer in isolation),
        // so compare relative rather than absolute error.
        let mag = last_prefill
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar::<f32>()
            .expect("to_scalar");
        let rel_diff = max_abs_diff(&out_last, &last_prefill) / mag;
        assert!(
            rel_diff < 1e-5,
            "decode past the KV buffer's 256-slot room must match full prefill (rel_diff={rel_diff})"
        );
    }

    #[test]
    fn test_clear_kv_cache_resets_to_fresh_state() {
        let cfg = base_cfg();
        let vb = nonzero_vb(&cfg, 0.02);
        let device = &Device::Cpu;
        let mut attn = GqaAttention::new(cfg, vb).expect("new");

        let x = Tensor::ones((1, 3, 16), DType::F32, device).expect("ones");

        // First forward (no cache)
        let y_fresh = attn.forward(&x, None, None).expect("first forward");

        // clear_kv_cache resets state — next forward is equivalent to the first
        attn.clear_kv_cache();
        let y_reset = attn.forward(&x, None, None).expect("forward after clear");

        assert!(
            max_abs_diff(&y_fresh, &y_reset) < 1e-6,
            "output after clear_kv_cache must match fresh forward"
        );
    }

    #[test]
    fn test_incremental_decode_matches_full_prefill() {
        // Verify the core KV-cache correctness property: processing tokens one at a
        // time (incremental decode) must produce the same last-token output as
        // processing all tokens in one shot with a causal attention mask (full prefill).
        let cfg = base_cfg();
        let vb1 = nonzero_vb(&cfg, 0.02);
        let vb2 = nonzero_vb(&cfg, 0.02);
        let device = &Device::Cpu;

        let mut attn_prefill = GqaAttention::new(cfg, vb1).expect("prefill attn");
        let mut attn_incr = GqaAttention::new(cfg, vb2).expect("incremental attn");

        // Position-varying input so each token has a distinct K/V.
        let seq_len = 3usize;
        let x = varying_input(1, seq_len, cfg.dim);

        // Full prefill with a strict causal mask: position i attends to positions ≤ i.
        let n = seq_len;
        let mask_data: Vec<f32> = (0..n * n)
            .map(|i| {
                if (i % n) > (i / n) {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let mask = Tensor::from_vec(mask_data, (1, 1, n, n), device).expect("causal mask");
        let out_prefill = attn_prefill
            .forward(&x, None, Some(&mask))
            .expect("prefill forward");
        // Extract the last token's output.
        let last_prefill = out_prefill
            .narrow(1, seq_len - 1, 1)
            .expect("narrow prefill");

        // Incremental decode: feed one token at a time; no explicit mask required
        // because each step's query attends only to the accumulated KV cache.
        let t0 = x.narrow(1, 0, 1).expect("t0");
        let t1 = x.narrow(1, 1, 1).expect("t1");
        let t2 = x.narrow(1, 2, 1).expect("t2");
        attn_incr.forward(&t0, None, None).expect("step 0");
        attn_incr.forward(&t1, None, None).expect("step 1");
        let out_last = attn_incr.forward(&t2, None, None).expect("step 2");

        assert!(
            max_abs_diff(&out_last, &last_prefill) < 1e-5,
            "incremental decode must match full prefill at the last token position"
        );
    }

    #[test]
    fn test_incremental_decode_matches_full_prefill_gqa() {
        // Same property as `test_incremental_decode_matches_full_prefill`,
        // but with n_kv_groups > 1 so decode steps exercise the
        // GQA-grouped SDPA path instead of the n_rep==1 flash-attn path.
        // b_sz=2 is required for this: the CPU flash-attn decode fast path
        // (forward's `q_seq_len == 1 && b_sz == 1 && q.device().is_cpu()`
        // branch) is unconditional on `n_rep`, so a b_sz=1 CPU test would
        // take that branch instead of the GQA-grouped one being tested here.
        let cfg = AttentionConfig {
            dim: 16,
            n_heads: 8,
            n_kv_heads: 2,
            head_dim: 2,
            ..base_cfg()
        };
        let vb1 = nonzero_vb(&cfg, 0.02);
        let vb2 = nonzero_vb(&cfg, 0.02);
        let device = &Device::Cpu;

        let mut attn_prefill = GqaAttention::new(cfg, vb1).expect("prefill attn");
        let mut attn_incr = GqaAttention::new(cfg, vb2).expect("incremental attn");

        let seq_len = 3usize;
        let x = varying_input(2, seq_len, cfg.dim);

        let n = seq_len;
        let mask_data: Vec<f32> = (0..n * n)
            .map(|i| {
                if (i % n) > (i / n) {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let mask = Tensor::from_vec(mask_data, (1, 1, n, n), device).expect("causal mask");
        let out_prefill = attn_prefill
            .forward(&x, None, Some(&mask))
            .expect("prefill forward");
        let last_prefill = out_prefill
            .narrow(1, seq_len - 1, 1)
            .expect("narrow prefill");

        let t0 = x.narrow(1, 0, 1).expect("t0");
        let t1 = x.narrow(1, 1, 1).expect("t1");
        let t2 = x.narrow(1, 2, 1).expect("t2");
        attn_incr.forward(&t0, None, None).expect("step 0");
        attn_incr.forward(&t1, None, None).expect("step 1");
        let out_last = attn_incr.forward(&t2, None, None).expect("step 2");

        assert!(
            max_abs_diff(&out_last, &last_prefill) < 1e-5,
            "GQA-grouped incremental decode must match full prefill at the last token position"
        );
    }

    #[test]
    fn test_flash_attn_decode_with_explicit_mask() {
        // Verify `GqaAttention::forward`'s CPU flash-attn decode branch
        // (`q_seq_len == 1 && b_sz == 1 && q.device().is_cpu()`) actually
        // wires an explicit `Some(mask)` through as `AttnMask::Mask` rather
        // than silently dropping it (e.g. by always passing `AttnMask::None`).
        let cfg = base_cfg();
        let vb1 = nonzero_vb(&cfg, 0.02);
        let vb2 = nonzero_vb(&cfg, 0.02);
        let device = &Device::Cpu;

        let mut attn_no_mask = GqaAttention::new(cfg, vb1).expect("no mask attn");
        let mut attn_masked = GqaAttention::new(cfg, vb2).expect("masked attn");

        let prefill_len = 3usize;
        let x = varying_input(1, prefill_len + 1, cfg.dim);
        let prefill = x.narrow(1, 0, prefill_len).expect("prefill tokens");
        let decode_tok = x.narrow(1, prefill_len, 1).expect("decode token");

        attn_no_mask
            .forward(&prefill, None, None)
            .expect("prefill no mask");
        attn_masked
            .forward(&prefill, None, None)
            .expect("prefill masked");

        let y_no_mask = attn_no_mask
            .forward(&decode_tok, None, None)
            .expect("decode no mask");

        // Non-trivial additive mask: force attention onto only the oldest
        // KV position (mask every other position to -1e9), matching
        // crane-serve's -1e9/0.0 padding-mask convention. `nonzero_vb`'s
        // smoothly-increasing weights make dot products grow with position,
        // so unmasked softmax already saturates almost entirely on the
        // newest position — masking down to *that same* position wouldn't
        // move the output. Forcing attention onto the oldest position
        // instead guarantees a detectable difference.
        let kv_len = prefill_len + 1;
        let mut mask_data = vec![-1e9f32; kv_len];
        mask_data[0] = 0.0;
        let mask = Tensor::from_vec(mask_data, (1, 1, 1, kv_len), device).expect("mask");
        let y_masked = attn_masked
            .forward(&decode_tok, None, Some(&mask))
            .expect("decode masked");

        assert_eq!(y_no_mask.dims(), y_masked.dims());
        assert!(
            max_abs_diff(&y_no_mask, &y_masked) > 1e-4,
            "explicit mask must change the CPU flash-attn decode output; diff was too small"
        );
    }

    #[test]
    fn test_clear_kv_cache_then_reuse() {
        // Verify that clear + multiple forward passes work correctly.
        let mut attn = zeros_attn(base_cfg());
        let device = &Device::Cpu;

        let x = Tensor::zeros((1, 2, 16), DType::F32, device).expect("zeros");
        attn.forward(&x, None, None).expect("first");
        attn.clear_kv_cache();
        attn.forward(&x, None, None).expect("second after clear");
        attn.clear_kv_cache();
        let y = attn.forward(&x, None, None).expect("third after clear");
        assert_eq!(y.dims(), &[1, 2, 16]);
    }

    // -------------------------------------------------------------------------
    // Attention mask test
    // -------------------------------------------------------------------------

    #[test]
    fn test_attention_mask_changes_output() {
        let cfg = base_cfg();
        let vb1 = nonzero_vb(&cfg, 0.01);
        let vb2 = nonzero_vb(&cfg, 0.01);
        let device = &Device::Cpu;

        let mut attn_no_mask = GqaAttention::new(cfg, vb1).expect("no mask");
        let mut attn_masked = GqaAttention::new(cfg, vb2).expect("masked");

        // Position-varying input: each token has a distinct embedding so Q and K
        // differ per position. With uniform inputs, all Q/K are identical and the
        // causal mask cannot change the (already uniform) attention distribution.
        let x = varying_input(1, 4, cfg.dim);
        // Strict causal mask: position i may only attend to positions ≤ i.
        let mask_data: Vec<f32> = (0..4 * 4)
            .map(|i| {
                if (i % 4) > (i / 4) {
                    f32::NEG_INFINITY
                } else {
                    0.0
                }
            })
            .collect();
        let mask = Tensor::from_vec(mask_data, (1, 1, 4, 4), device).expect("mask");

        let y_no_mask = attn_no_mask
            .forward(&x, None, None)
            .expect("no mask forward");
        let y_masked = attn_masked
            .forward(&x, None, Some(&mask))
            .expect("masked forward");

        assert_eq!(y_no_mask.dims(), y_masked.dims());
        assert!(
            max_abs_diff(&y_no_mask, &y_masked) > 1e-4,
            "causal mask must change the output; diff was too small"
        );
    }

    // -------------------------------------------------------------------------
    // Bias tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bias_true_output_shape() {
        let cfg = AttentionConfig {
            qkv_bias: true,
            o_bias: true,
            ..base_cfg()
        };
        let vb = make_vb(&cfg); // includes zero bias tensors
        let mut attn = GqaAttention::new(cfg, vb).expect("new with bias");
        let x = Tensor::zeros((1, 3, 16), DType::F32, &Device::Cpu).expect("zeros");
        let y = attn.forward(&x, None, None).expect("forward");
        assert_eq!(y.dims(), &[1, 3, 16]);
    }

    #[test]
    fn test_bias_false_output_shape() {
        let cfg = AttentionConfig {
            qkv_bias: false,
            o_bias: false,
            ..base_cfg()
        };
        let vb = make_vb(&cfg); // no bias tensors present
        let mut attn = GqaAttention::new(cfg, vb).expect("new without bias");
        let x = Tensor::zeros((1, 3, 16), DType::F32, &Device::Cpu).expect("zeros");
        let y = attn.forward(&x, None, None).expect("forward");
        assert_eq!(y.dims(), &[1, 3, 16]);
    }

    #[test]
    fn test_nonzero_bias_changes_output() {
        // A non-zero bias shifts Q/K/V projections and must change the output.
        let cfg = AttentionConfig {
            qkv_bias: true,
            o_bias: true,
            ..base_cfg()
        };
        let device = &Device::Cpu;
        let q_out = cfg.n_heads * cfg.head_dim;
        let kv_out = cfg.n_kv_heads * cfg.head_dim;

        let build_vb = |bias_val: f32| {
            let mut t: HashMap<String, Tensor> = HashMap::new();
            let fill = |rows: usize, cols: usize| {
                let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32 + 1.0) * 0.01).collect();
                Tensor::from_vec(data, (rows, cols), device).expect("weight")
            };
            let scalar = |n: usize| Tensor::full(bias_val, n, device).expect("bias");
            t.insert("q_proj.weight".into(), fill(q_out, cfg.dim));
            t.insert("k_proj.weight".into(), fill(kv_out, cfg.dim));
            t.insert("v_proj.weight".into(), fill(kv_out, cfg.dim));
            t.insert("o_proj.weight".into(), fill(cfg.dim, q_out));
            t.insert("q_proj.bias".into(), scalar(q_out));
            t.insert("k_proj.bias".into(), scalar(kv_out));
            t.insert("v_proj.bias".into(), scalar(kv_out));
            t.insert("o_proj.bias".into(), scalar(cfg.dim));
            VarBuilder::from_tensors(t, DType::F32, device)
        };

        let mut attn_no_bias_val = GqaAttention::new(cfg, build_vb(0.0)).expect("zero bias");
        let mut attn_nonzero_bias = GqaAttention::new(cfg, build_vb(1.0)).expect("nonzero bias");

        let x = Tensor::ones((1, 2, 16), DType::F32, device).expect("ones");
        let y0 = attn_no_bias_val
            .forward(&x, None, None)
            .expect("zero bias forward");
        let y1 = attn_nonzero_bias
            .forward(&x, None, None)
            .expect("nonzero bias forward");
        assert!(
            max_abs_diff(&y0, &y1) > 1e-4,
            "non-zero bias must change the output"
        );
    }

    #[test]
    fn test_asymmetric_bias_qkv_only() {
        // qkv_bias=true, o_bias=false is the Qwen2.5 layout: Q/K/V projections have
        // bias tensors in the checkpoint but the output projection does not.
        let cfg = AttentionConfig {
            qkv_bias: true,
            o_bias: false,
            ..base_cfg()
        };
        let vb = make_vb(&cfg); // inserts QKV bias tensors, omits o_proj.bias
        let mut attn = GqaAttention::new(cfg, vb).expect("asymmetric bias construction");
        let x = Tensor::zeros((1, 3, 16), DType::F32, &Device::Cpu).expect("zeros");
        let y = attn.forward(&x, None, None).expect("forward");
        assert_eq!(y.dims(), &[1, 3, 16]);
    }

    // -------------------------------------------------------------------------
    // Constructor validation
    // -------------------------------------------------------------------------

    #[test]
    fn test_n_kv_heads_not_dividing_n_heads_errors() {
        let cfg = AttentionConfig {
            n_heads: 7,
            n_kv_heads: 3, // 7 % 3 != 0
            ..base_cfg()
        };
        // Make a VarBuilder that at least has the projection weights (dims may be wrong
        // but the divisibility check fires first)
        let device = &Device::Cpu;
        let dummy = Tensor::zeros((1, 1), DType::F32, device).expect("dummy");
        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert("q_proj.weight".into(), dummy.clone());
        t.insert("k_proj.weight".into(), dummy.clone());
        t.insert("v_proj.weight".into(), dummy.clone());
        t.insert("o_proj.weight".into(), dummy);
        let vb = VarBuilder::from_tensors(t, DType::F32, device);
        assert!(
            GqaAttention::new(cfg, vb).is_err(),
            "n_kv_heads not dividing n_heads must produce an error"
        );
    }

    // -------------------------------------------------------------------------
    // Batch consistency
    // -------------------------------------------------------------------------

    #[test]
    fn test_batch_consistency() {
        // Identical rows in a batch must produce identical output rows.
        let cfg = base_cfg();
        // Two VarBuilders with identical weights are needed because VarBuilder does
        // not implement Clone; they are built from the same shape and scale.
        let vb1 = nonzero_vb(&cfg, 0.01);
        let vb2 = nonzero_vb(&cfg, 0.01);
        let device = &Device::Cpu;

        let row: Vec<f32> = (0..cfg.dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let single = Tensor::from_vec(row, (1, 1, cfg.dim), device).expect("single row");
        let batch = Tensor::cat(&[&single, &single, &single], 0).expect("cat batch");

        let mut attn_single = GqaAttention::new(cfg, vb1).expect("single");
        let mut attn_batch = GqaAttention::new(cfg, vb2).expect("batch");

        let out_single = attn_single
            .forward(&single, None, None)
            .expect("single forward");
        let out_batch = attn_batch
            .forward(&batch, None, None)
            .expect("batch forward");

        for b in 0..3_usize {
            let row_b = out_batch.narrow(0, b, 1).expect("narrow");
            let diff = max_abs_diff(&row_b, &out_single);
            assert!(
                diff < 1e-6,
                "batch row {b} differs from single-row result: max_diff={diff}"
            );
        }
    }
}
