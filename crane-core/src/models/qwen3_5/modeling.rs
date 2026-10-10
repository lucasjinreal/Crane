//! Qwen 3.5 transformer layer: hybrid full-attention + linear-attention stack.
//!
//! Layout (per layer):
//!   residual = `input_layernorm(x)`
//!   `attn_or_gdn_out` = `full_or_linear`(residual, …)   // dispatched by `layer_types`
//!   x = x + `attn_or_gdn_out`
//!   residual2 = `post_attention_layernorm(x)`
//!   x = x + mlp(residual2)
//!
//! The full-attention path uses MRoPE-interleaved rotary embeddings and gated
//! output (`attn_output_gate: true` in the config). The linear-attention path
//! uses [`crate::ops::gdn::GatedDeltaNet`].
//!
//! Per-layer GDN state is held by [`super::Qwen3_5TextModel`] (not by the
//! layer), so that continuous-batching can save/restore state per request.

use candle_core::quantized::GgmlDType;
use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use std::io::{Read, Seek};

use crate::models::modules::moe::{MlpOrMoe, SparseMoeBlock};
use crate::ops::linear::{LinearLayer, linear_layer};
use crate::quantized::gguf_file::Gguf;

// ── Qwen 3.5 RMSNorm (unit-offset) ───────────────────────────────────────

/// `RMSNorm` as used by Qwen 3.5: `x / rms(x) * (1 + weight)`.
///
/// Unlike the standard (Llama/Qwen3) `RMSNorm` — which scales by `weight` — Qwen
/// 3.5 adds a unit offset (`1 + weight`, Gemma-style). HF source:
/// `output = self._norm(x.float()) * (1.0 + self.weight.float())`. The stored
/// weights have mean ~0.24, so omitting the `+1` shrinks every normalized
/// activation ~5x and compounds across layers.
///
/// This is NOT used by the GDN gated norm ([`crate::ops::gdn::RmsNormGated`]), which
/// scales by plain `weight` in HF.
///
/// The unit offset is folded into the stored scale at load rather than applied
/// per call, which leaves the forward pass as a single fused `rms_norm`. That
/// matters far more than it looks: this norm runs 48 times per decoded token
/// (two per block, plus per-head Q/K norms), and as an op chain it was ~10
/// launches each — 6.2 ms of the 21.8 ms the CPU spent submitting a token's
/// work, measured with `CRANE_PROF=1`.
#[derive(Clone)]
pub struct Qwen35RmsNorm {
    /// `1 + weight` for HF layouts; plain `weight` for GGUF, where llama.cpp's
    /// converter already folded the `+1` in (mean ~1.24 vs ~0.24 raw).
    alpha: Tensor,
    eps: f32,
}

impl Qwen35RmsNorm {
    /// # Errors
    ///
    /// Returns an error if the `weight` tensor is missing or has an unexpected shape.
    pub fn load(size: usize, eps: f64, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self::from_folded(weight.affine(1.0, 1.0)?, eps))
    }

    /// Construct from a scale that already includes the `+1` unit offset
    /// (GGUF layout).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_folded(alpha: Tensor, eps: f64) -> Self {
        Self {
            alpha,
            eps: eps as f32,
        }
    }
}

impl Module for Qwen35RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // candle's fused `rms_norm` normalizes in f32 internally regardless of
        // the tensor dtype, so this keeps the previous f32 accumulation. It
        // does require `alpha` to share `x`'s dtype and `x` to be contiguous;
        // both hold already on every call site here, so both conversions are
        // no-op clones rather than kernels.
        let alpha = self.alpha.to_dtype(x.dtype())?;
        candle_nn::ops::rms_norm(&x.contiguous()?, &alpha, self.eps)
    }
}

use super::config::{LayerType, TextConfig};
use crate::models::modules::kv_cache::KvCache;
use crate::ops::gdn::{GatedDeltaNet, GdnDims, GdnInputProjectionKind, GdnLayerCache};

// ── MRoPE rotary embedding ─────────────────────────────────────────────

/// Multimodal Rotary Position Embedding (interleaved variant, used by Qwen 3.5).
///
/// For text-only inference the `position_ids` are identical across all three
/// sections, so this reduces to standard `RoPE` applied to the first
/// `rot_dim = head_dim * partial_rotary_factor` components of each head.
///
/// Precomputes `cos` and `sin` tables of shape `[max_pos, rot_dim/2]`. Use
/// [`apply_mrope`] to rotate a query/key tensor.
pub struct MRotaryEmbedding {
    cos_table: Tensor,
    sin_table: Tensor,
    rot_dim: usize,
    /// Doubled `mrope_section` (`[22, 22, 20]` for Qwen 3.5). Cached for the
    /// vision-path gather.
    mrope_section_doubled: Vec<usize>,
}

impl MRotaryEmbedding {
    /// # Errors
    ///
    /// Returns an error if building the cos/sin tables fails.
    pub fn new(cfg: &TextConfig, device: &Device) -> Result<Self> {
        Self::from_params(
            cfg.rot_dim(),
            cfg.rope_theta(),
            cfg.max_position_embeddings,
            cfg.mrope_section(),
            device,
        )
    }

    /// Build the tables from raw rotary parameters, for configs other than
    /// Qwen 3.5's that share its interleaved `MRoPE` (e.g. Qwen4-Exp).
    ///
    /// # Errors
    ///
    /// Returns an error if building the cos/sin tables fails.
    pub fn from_params(
        rot_dim: usize,
        rope_theta: f64,
        max_pos: usize,
        mrope_section: &[usize],
        device: &Device,
    ) -> Result<Self> {
        // rope_theta is a positive frequency base (e.g. 10_000 or 10_000_000);
        // computing the RoPE table in f32 matches HF's own float32 rotary math.
        #[allow(clippy::cast_possible_truncation)]
        let base = rope_theta as f32;

        // cos/sin tables have shape `[S, rot_dim/2]` — exactly the slice of the
        // head that receives rotary embeddings. `apply_mrope` rotates only the
        // first `rot_dim` components of each head (HF's partial-rotary scheme:
        // `q_rot = q[..., :rot_dim]`), so the tables must NOT be padded to
        // `head_dim/2`. Padding them to the full head and rotating the whole
        // head (the previous approach) pairs dim `i` with dim `i+head_dim/2`,
        // whereas HF pairs `i` with `i+rot_dim/2` inside the rotary slice — a
        // different rotation entirely.
        let half_rot = rot_dim / 2;
        // half_rot and rot_dim are at most head_dim (a small model dimension,
        // e.g. <=512), well within f32's 24-bit exact-integer range.
        #[allow(clippy::cast_precision_loss)]
        let inv: Vec<f32> = (0..half_rot)
            .map(|i| 1.0 / base.powf(i as f32 * 2.0 / rot_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv.as_slice(), device)?;

        // max_pos (max_position_embeddings) stays well below 2^24 (16_777_216),
        // the largest integer f32 can represent exactly, even for long-context
        // configs (e.g. 262_144 or 1_000_000).
        #[allow(clippy::cast_precision_loss)]
        let positions: Vec<f32> = (0..max_pos).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?; // [max_pos, half_rot]

        let cos_table = freqs.cos()?.contiguous()?;
        let sin_table = freqs.sin()?.contiguous()?;

        let mrope_section_doubled: Vec<usize> = mrope_section.iter().map(|s| s * 2).collect();

        Ok(Self {
            cos_table,
            sin_table,
            rot_dim,
            mrope_section_doubled,
        })
    }

    /// Slice cos/sin for positions `[start, start+seq_len)`. Used by the
    /// text-only path where all three axes share the same position.
    ///
    /// # Errors
    ///
    /// Returns an error if `start + seq_len` exceeds the precomputed table.
    pub fn cos_sin(&self, start: usize, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos_table.narrow(0, start, seq_len)?;
        let sin = self.sin_table.narrow(0, start, seq_len)?;
        Ok((cos, sin))
    }

    /// Build cos/sin for a sequence with **per-token 3D position ids** (vision).
    ///
    /// `position_ids` has shape `[3, S]` on the same device as the tables:
    /// row 0 = temporal position (T), row 1 = height position (H),
    /// row 2 = width position (W). Each row is gathered against
    /// `cos_table` / `sin_table` to produce a `[S, rot_dim/2]` per-axis tensor;
    /// the three per-axis tensors are then combined under the interleaved
    /// `MRoPE` scheme using `mrope_section` (see HF's
    /// `apply_multimodal_rotary_pos_emb` for the reference).
    ///
    /// Returns `(cos, sin)` of shape `[S, rot_dim/2]`, ready to feed into
    /// [`apply_mrope`] (candle's `rotary_emb::rope` pairs each entry with
    /// its `i + rot_dim/2` counterpart inside the rotary slice, matching the
    /// HF behavior of the `q * cos + rotate_half(q) * sin` formulation when
    /// the cos/sin tables are not pair-duplicated).
    ///
    /// # Errors
    ///
    /// Returns an error if `position_ids` has an unexpected shape or contains
    /// out-of-range indices.
    pub fn cos_sin_with_position_ids(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_three, seq_len) = position_ids.dims2()?;
        let half_rot = self.rot_dim / 2;

        // Gather one cos/sin slice per axis. position_ids must be U32 (candle's
        // index_select requirement); the caller (vlm.rs::build_position_ids)
        // builds the tensor from a `Vec<u32>`.
        let mut per_axis_cos: Vec<Tensor> = Vec::with_capacity(3);
        let mut per_axis_sin: Vec<Tensor> = Vec::with_capacity(3);
        for axis in 0..3 {
            let ids = position_ids.narrow(0, axis, 1)?.squeeze(0)?;
            let cos = self.cos_table.index_select(&ids, 0)?; // [S, half_rot]
            let sin = self.sin_table.index_select(&ids, 0)?;
            per_axis_cos.push(cos);
            per_axis_sin.push(sin);
        }

        // Interleaved MRoPE, matching HF `Qwen3_5TextRotaryEmbedding
        // .apply_interleaved_mrope`: this is an INDEX-interleave, not a
        // contiguous chunking. Start from the T axis everywhere, then let H
        // claim columns `slice(1, 3*section[1], 3)` and W claim columns
        // `slice(2, 3*section[2], 3)` — each axis keeps the frequency index of
        // the column it lands in.
        //
        // For `mrope_section = [11, 11, 10]` (half_rot = 32) the ownership is
        //   H -> 1, 4, 7, ..., 31   (11 columns)
        //   W -> 2, 5, 8, ..., 29   (10 columns)
        //   T -> 0, 3, 6, ..., 30   (the remaining 11)
        // i.e. column i is served by axis (i % 3) until each section runs out.
        //
        // Note this is a no-op when T == H == W (the text-only case): all three
        // gathers are identical, so the result reduces to the plain rope table.
        // `mrope_section_doubled` is empty for configs with no `mrope_section`
        // (e.g. MiniCPM-V-4.6's `qwen3_5_text` backbone, which drives this
        // path with T==H==W and no interleaving at all) — `.get(dim)` instead
        // of `[dim]` avoids an out-of-bounds panic; `unwrap_or(0)` makes
        // `limit` collapse to 0 below, so the H/W reassignment loop never
        // runs and every column stays on the T axis, which is exactly
        // correct when T==H==W.
        let mut axis_of = vec![0usize; half_rot];
        for (dim, offset) in [(1usize, 1usize), (2usize, 2usize)] {
            let section = self.mrope_section_doubled.get(dim).copied().unwrap_or(0) / 2;
            let limit = (section * 3).min(half_rot);
            let mut i = offset;
            while i < limit {
                axis_of[i] = dim;
                i += 3;
            }
        }

        // Combine with per-axis 0/1 masks so the whole thing stays a couple of
        // fused elementwise ops instead of `half_rot` narrow/cat calls.
        let dtype = self.cos_table.dtype();
        let device = self.cos_table.device();
        let mut cos = Tensor::zeros((seq_len, half_rot), dtype, device)?;
        let mut sin = Tensor::zeros((seq_len, half_rot), dtype, device)?;
        for axis in 0..3 {
            let mask: Vec<f32> = axis_of
                .iter()
                .map(|a| if *a == axis { 1.0 } else { 0.0 })
                .collect();
            if mask.iter().all(|v| *v == 0.0) {
                continue;
            }
            let mask = Tensor::from_vec(mask, (1, half_rot), device)?.to_dtype(dtype)?;
            cos = (cos + per_axis_cos[axis].broadcast_mul(&mask)?)?;
            sin = (sin + per_axis_sin[axis].broadcast_mul(&mask)?)?;
        }

        let cos = cos.contiguous()?;
        let sin = sin.contiguous()?;
        debug_assert_eq!(cos.dims(), &[seq_len, half_rot]);
        Ok((cos, sin))
    }

    #[must_use]
    pub fn rot_dim(&self) -> usize {
        self.rot_dim
    }
}

/// Apply rotary embeddings to a query/key tensor `[B, H, S, D]`.
///
/// Only the first `rot_dim` components of each head are rotated; the remaining
/// `D - rot_dim` components are passed through unchanged. This mirrors HF's
/// partial-rotary scheme (`q_rot = q[..., :rot_dim]`, `q_pass = q[..., rot_dim:]`).
/// `cos`/`sin` tables have shape `[S, rot_dim/2]`.
///
/// `candle_nn::rotary_emb::rope` is the rotate-half (non-interleaved / GPT-NeoX)
/// variant — it pairs component `i` with `i + rot_dim/2` *within the slice we
/// hand it*, which matches HF's `rotate_half` over the rotary slice. We must
/// slice first; rotating the full head would pair `i` with `i + head_dim/2`.
///
/// # Errors
///
/// Returns an error if `x`'s shape is incompatible with `rot_dim`.
pub fn apply_mrope(x: &Tensor, cos: &Tensor, sin: &Tensor, rot_dim: usize) -> Result<Tensor> {
    let (_b, _h, _seq_len, head_dim) = x.dims4()?;
    let dtype = x.dtype();
    // `cos`/`sin` are kept in f32; the rope op requires its inputs to share a
    // dtype, so for half-precision activations (F16/BF16 on GPU) we rotate in
    // f32 and cast back. This also matches HF, which applies RoPE in float.
    let rope_f32 = |t: &Tensor| -> Result<Tensor> {
        let r = candle_nn::rotary_emb::rope(&t.to_dtype(DType::F32)?.contiguous()?, cos, sin)?;
        r.to_dtype(dtype)
    };
    if rot_dim == head_dim {
        return rope_f32(x);
    }
    let x_rot = rope_f32(&x.narrow(D::Minus1, 0, rot_dim)?)?;
    let x_pass = x.narrow(D::Minus1, rot_dim, head_dim - rot_dim)?;
    Tensor::cat(&[&x_rot, &x_pass], D::Minus1)?.contiguous()
}

/// Rotary tables already sliced to the positions of the current forward call,
/// together with the partial-rotary width they cover.
///
/// Under chunked prefill the slice is taken at `start_pos + chunk_offset`, so
/// carrying it as one value keeps the absolute-position contract in a single
/// place instead of three parallel parameters threaded through every layer.
#[derive(Clone, Copy)]
pub struct RopeSlice<'a> {
    pub cos: &'a Tensor,
    pub sin: &'a Tensor,
    pub rot_dim: usize,
}

// ── Full-attention layer ────────────────────────────────────────────────

/// Standard softmax attention layer for Qwen 3.5's `full_attention` blocks.
///
/// Differences from the regular Qwen 3 attention:
/// - `q_proj` outputs `num_heads * head_dim * 2`; the second half is a sigmoid
///   gate applied to the attention output.
/// - Per-head QK-norm is always present (`q_norm`, `k_norm` of size `head_dim`).
/// - `RoPE` is MRoPE-interleaved applied only to the first `rot_dim` components.
///   `CRANE_ATTN_EXPAND=1` forces the legacy GQA-expansion path (decode and
///   prefill) instead of the grouped matmul. The two are mathematically
///   identical, so this exists to A/B them: same binary, same weights, one
///   variable.
fn legacy_attn_expand() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("CRANE_ATTN_EXPAND").is_ok_and(|v| !matches!(v.trim(), "" | "0"))
    })
}

/// Shape of a [`FullAttention`] layer, independent of which model config it
/// comes from.
#[derive(Debug, Clone, Copy)]
pub struct AttentionDims {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    /// `q_proj` also emits a per-head sigmoid gate on the attention output.
    pub output_gate: bool,
}

impl From<&TextConfig> for AttentionDims {
    fn from(cfg: &TextConfig) -> Self {
        Self {
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            rms_norm_eps: cfg.rms_norm_eps,
            output_gate: cfg.attn_output_gate,
        }
    }
}

pub struct FullAttention {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    o_proj: LinearLayer,
    q_norm: Qwen35RmsNorm,
    k_norm: Qwen35RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    has_output_gate: bool,
    /// Dtype the K/V cache and attention run in when it differs from the
    /// activations' (see [`Self::set_attention_dtype`]); `None` follows them.
    attn_dtype: Option<DType>,
}

impl FullAttention {
    /// Run the K/V cache and attention in `dtype` instead of the activations'
    /// dtype: Q/K/V are cast after `RoPE` and the output back before the gate.
    /// Lets F32 activations (no casts around every quantized linear) keep a
    /// half-precision cache, which is what long contexts are sized by.
    pub fn set_attention_dtype(&mut self, dtype: Option<DType>) {
        self.attn_dtype = dtype;
    }

    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing or has an unexpected shape.
    pub fn load(cfg: &TextConfig, vb: &VarBuilder, quant: Option<GgmlDType>) -> Result<Self> {
        Self::load_dims(AttentionDims::from(cfg), cfg.hidden_size, vb, quant)
    }

    /// [`Self::load`] for any config that shares this attention (HF layout:
    /// `q_proj`, `k_proj`, `v_proj`, `o_proj`, `q_norm`, `k_norm`).
    ///
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing or has an unexpected shape.
    pub fn load_dims(
        dims: AttentionDims,
        hidden_size: usize,
        vb: &VarBuilder,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let AttentionDims {
            num_heads,
            num_kv_heads,
            head_dim,
            rms_norm_eps,
            output_gate,
        } = dims;
        let q_out = if output_gate {
            num_heads * head_dim * 2
        } else {
            num_heads * head_dim
        };
        let q_proj = linear_layer(hidden_size, q_out, vb.pp("q_proj"), quant)?;
        let k_proj = linear_layer(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"), quant)?;
        let v_proj = linear_layer(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"), quant)?;
        let o_proj = linear_layer(num_heads * head_dim, hidden_size, vb.pp("o_proj"), quant)?;
        let q_norm = Qwen35RmsNorm::load(head_dim, rms_norm_eps, &vb.pp("q_norm"))?;
        let k_norm = Qwen35RmsNorm::load(head_dim, rms_norm_eps, &vb.pp("k_norm"))?;

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
            has_output_gate: output_gate,
            attn_dtype: None,
        })
    }

    /// Query heads.
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Construct from GGUF quantized weights (llama.cpp `qwen35` layout).
    ///
    /// `attn_q` keeps HF's fused `[query | gate]` per-head layout (2× rows
    /// when `attn_output_gate`); per-head q/k norms are stored with the `+1`
    /// unit offset already folded in.
    ///
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing.
    pub fn from_gguf<R: Read + Seek>(
        cfg: &TextConfig,
        gg: &mut Gguf<R>,
        layer_idx: usize,
    ) -> Result<Self> {
        Self::from_gguf_dims(AttentionDims::from(cfg), gg, layer_idx)
    }

    /// [`Self::from_gguf`] for any config sharing the llama.cpp layout
    /// (`attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_{q,k}_norm`).
    ///
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing.
    pub fn from_gguf_dims<R: Read + Seek>(
        dims: AttentionDims,
        gg: &mut Gguf<R>,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let q_proj = gg.linear(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.linear(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.linear(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.linear(&format!("{prefix}.attn_output.weight"))?;

        let q_norm = Qwen35RmsNorm::from_folded(
            gg.dequant_tensor(&format!("{prefix}.attn_q_norm.weight"))?,
            dims.rms_norm_eps,
        );
        let k_norm = Qwen35RmsNorm::from_folded(
            gg.dequant_tensor(&format!("{prefix}.attn_k_norm.weight"))?,
            dims.rms_norm_eps,
        );

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: dims.num_heads,
            num_kv_heads: dims.num_kv_heads,
            head_dim: dims.head_dim,
            has_output_gate: dims.output_gate,
            attn_dtype: None,
        })
    }

    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    // q/k/v/b/h/s/d are standard ML tensor-shape notation (query, key, value,
    // batch, heads, seq_len, head_dim), matching the terminology used
    // throughout this function's comments.
    #[allow(clippy::many_single_char_names)]
    pub fn forward(
        &self,
        x: &Tensor,
        rope: RopeSlice<'_>,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q_out = self.q_proj.forward(x)?;
        let k_proj_out = self.k_proj.forward(x)?;
        let v_proj_out = self.v_proj.forward(x)?;
        let k = k_proj_out;
        let v = v_proj_out;

        let (q, gate) = if self.has_output_gate {
            // HF splits `[query | gate]` PER HEAD, not on the flat axis:
            //   q_proj(x).view(B, S, num_heads, head_dim*2).chunk(2, dim=-1)
            // so for each head the first `head_dim` is the query and the next
            // `head_dim` is the gate. Splitting the flat 4096 in half instead
            // interleaves heads' query/gate and scrambles q_norm. Re-flatten
            // both back to `[B, S, num_heads*head_dim]` in head order.
            let flat = self.num_heads * self.head_dim;
            let qh = q_out.reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?;
            // Straight to attention layout `[B, H, S, D]`. Materializing the
            // flat `[B, S, H*D]` form first and transposing afterwards cost two
            // full copies of Q where narrowing then transposing costs one.
            let q = qh
                .narrow(D::Minus1, 0, self.head_dim)?
                .transpose(1, 2)?
                .contiguous()?;
            let gate = qh
                .narrow(D::Minus1, self.head_dim, self.head_dim)?
                .contiguous()?
                .reshape((b_sz, seq_len, flat))?;
            (q, Some(gate))
        } else {
            let q = q_out
                .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            (q, None)
        };
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q = apply_mrope(&q, rope.cos, rope.sin, rope.rot_dim)?;
        let k = apply_mrope(&k, rope.cos, rope.sin, rope.rot_dim)?;
        let out_dtype = q.dtype();
        let (q, k, v) = match self.attn_dtype {
            Some(dt) if dt != out_dtype => (q.to_dtype(dt)?, k.to_dtype(dt)?, v.to_dtype(dt)?),
            _ => (q, k, v),
        };
        // The model builds its mask in the attention dtype; a caller-supplied
        // one may not match.
        let attention_mask = match attention_mask {
            Some(m) if m.dtype() != q.dtype() => Some(m.to_dtype(q.dtype())?),
            Some(m) => Some(m.clone()),
            None => None,
        };
        let attention_mask = attention_mask.as_ref();

        // Append this step's K/V to the cache (post-RoPE, pre-GQA-expand) and
        // continue with the full cached K/V. During incremental decode this is
        // what lets the attention see the whole context from a single token.
        let (k, v) = match kv_cache {
            Some(cache) => cache.append(&k, &v)?,
            None => (k, v),
        };

        let n_rep = self.num_heads / self.num_kv_heads;
        #[allow(clippy::cast_precision_loss)] // head_dim is small (<=512 in practice)
        let scale = 1.0 / (self.head_dim as f64).sqrt();

        let y = if legacy_attn_expand() {
            expanded_sdpa(&q, &k, &v, attention_mask, scale, n_rep)?
        } else {
            grouped_sdpa(&q, &k, &v, attention_mask, scale, n_rep)?
        }
        .to_dtype(out_dtype)?;

        // Qwen 3.5 gates the attention output before `o_proj`.
        let y = match gate {
            Some(g) => {
                let gate = candle_nn::ops::sigmoid(&g.to_dtype(y.dtype())?)?;
                y.broadcast_mul(&gate)?
            },
            None => y,
        };
        self.o_proj.forward(&y)
    }
}

/// Most queries per attention slice. Prefill chunks can be large (a packed
/// `MoE` is cheapest per token in big chunks), but attention scores grow as
/// `heads x queries x context`, so [`grouped_sdpa`] walks a chunk's queries in
/// slices sized by [`attn_query_slice`].
const ATTN_QUERY_CHUNK: usize = 512;

/// Bytes of attention scores one slice may produce (f16 or f32 alike, as a
/// bound): keeps a 512-query slice of 24 heads up to ~2.7k context and
/// shrinks it beyond.
const ATTN_SCORE_BUDGET: usize = 128 << 20;

/// Queries per attention slice for `cells` of context: at most
/// [`ATTN_QUERY_CHUNK`], fewer once `heads x queries x cells` f32 scores would
/// pass [`ATTN_SCORE_BUDGET`], and at least 16.
pub(crate) fn attn_query_slice(heads: usize, cells: usize) -> usize {
    (ATTN_SCORE_BUDGET / (heads * cells * 4).max(1)).clamp(16, ATTN_QUERY_CHUNK)
}

/// GQA attention without expanding K/V: the `n_rep` query heads that share a
/// KV head are folded into the matmul's row dimension, so K and V are read
/// once at their stored `kv_heads` width. Queries go in slices of
/// [`attn_query_slice`], so the score matrix stays bounded however long the
/// prefill chunk and the context are.
///
/// Expanding them instead (see [`expanded_sdpa`]) materializes `k_rep`,
/// `v_rep` and `k_t`, each `[B, num_heads, cells, D]`: on Qwen3.8-27B (24 q /
/// 4 KV heads) that was 1.7 GB of traffic per decoded token at 2912 tokens of
/// context, and on Qwen3.8-Flash-Next (24 / 2) ~1.2 GB of prefill memory at
/// 32k. Ported from `models/qwen3/modeling.rs`'s decode path.
///
/// `q` is `[B, num_heads, S, D]` with heads ordered `kv_head * n_rep + r`;
/// `k`/`v` are `[B, kv_heads, cells, D]`; `mask` is additive and broadcasts to
/// `[B, 1, S, cells]`. Returns `[B, S, num_heads * D]`.
fn grouped_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
    n_rep: usize,
) -> Result<Tensor> {
    let (b_sz, num_heads, seq_len, _) = q.dims4()?;
    let cells = k.dim(2)?;
    let slice = attn_query_slice(b_sz * num_heads, cells);
    if seq_len <= slice {
        return grouped_sdpa_slice(q, k, v, mask, scale, n_rep);
    }
    let v = v.contiguous()?;
    let mut outs = Vec::with_capacity(seq_len.div_ceil(slice));
    let mut offset = 0;
    while offset < seq_len {
        let len = slice.min(seq_len - offset);
        let q = q.narrow(2, offset, len)?.contiguous()?;
        let mask = mask.map(|m| m.narrow(D::Minus2, offset, len)).transpose()?;
        outs.push(grouped_sdpa_slice(&q, k, &v, mask.as_ref(), scale, n_rep)?);
        offset += len;
    }
    Tensor::cat(&outs, 1)
}

/// [`grouped_sdpa`] over one slice of queries.
fn grouped_sdpa_slice(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
    n_rep: usize,
) -> Result<Tensor> {
    let (b_sz, num_heads, seq_len, head_dim) = q.dims4()?;
    let (_, kv_heads, cells, _) = k.dims4()?;
    // Heads sharing a KV head are adjacent, so this is a free view.
    let q_g = (q.reshape((b_sz, kv_heads, n_rep * seq_len, head_dim))? * scale)?;
    let k_t = k.transpose(2, 3)?; // [B, kv_heads, D, cells] — view only
    let scores = q_g.matmul(&k_t)?; // [B, kv_heads, n_rep * S, cells]
    let scores = match mask {
        // Split the rows back into (n_rep, S) so a [.., S, cells] mask
        // broadcasts over kv_heads and n_rep.
        Some(mask) => scores
            .reshape((b_sz, kv_heads, n_rep, seq_len, cells))?
            .broadcast_add(&mask.unsqueeze(1)?)?
            .reshape((b_sz, kv_heads, n_rep * seq_len, cells))?,
        None => scores,
    };
    let weights = candle_nn::ops::softmax_last_dim(&scores)?;
    weights
        .matmul(&v.contiguous()?)? // [B, kv_heads, n_rep * S, D]
        .reshape((b_sz, num_heads, seq_len, head_dim))?
        .transpose(1, 2)?
        .reshape((b_sz, seq_len, num_heads * head_dim))
}

/// The same attention with K/V expanded to every query head, kept for A/B
/// comparisons (`CRANE_ATTN_EXPAND=1`); see [`grouped_sdpa`].
// q/k/v/b/s/d are the standard attention-shape names.
#[allow(clippy::many_single_char_names)]
fn expanded_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
    n_rep: usize,
) -> Result<Tensor> {
    let (b_sz, num_heads, seq_len, _) = q.dims4()?;
    let expand = |t: &Tensor| -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(t.clone());
        }
        let (b, kv_heads, s, d) = t.dims4()?;
        t.unsqueeze(2)?
            .expand((b, kv_heads, n_rep, s, d))?
            .contiguous()?
            .reshape((b, num_heads, s, d))
    };
    let k_t = expand(k)?.transpose(D::Minus2, D::Minus1)?.contiguous()?;
    let attn_logits = (q.matmul(&k_t)? * scale)?;
    let attn_weights = match mask {
        Some(mask) => attn_weights_with_mask(&attn_logits, mask)?,
        None => attn_logits,
    };
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
    attn_weights
        .matmul(&expand(v)?)?
        .transpose(1, 2)?
        .reshape((b_sz, seq_len, ()))
}

fn attn_weights_with_mask(attn_logits: &Tensor, mask: &Tensor) -> Result<Tensor> {
    // HF applies the mask (shape `[B, 1, S_q, S_k]` for additive causal mask)
    // via `attn_weights + mask` and softmax. We do the same.
    attn_logits.broadcast_add(mask)
}

// ── MLP ────────────────────────────────────────────────────────────────

/// Standard `SwiGLU` MLP: `down(silu(gate(x)) * up(x))`.
pub struct Mlp {
    gate: LinearLayer,
    up: LinearLayer,
    down: LinearLayer,
}

impl Mlp {
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing or has an unexpected shape.
    pub fn load(cfg: &TextConfig, vb: &VarBuilder, quant: Option<GgmlDType>) -> Result<Self> {
        let gate = linear_layer(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("gate_proj"),
            quant,
        )?;
        let up = linear_layer(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("up_proj"),
            quant,
        )?;
        let down = linear_layer(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("down_proj"),
            quant,
        )?;
        Ok(Self { gate, up, down })
    }

    /// Construct from GGUF quantized weights.
    ///
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing.
    pub fn from_gguf<R: Read + Seek>(gg: &mut Gguf<R>, layer_idx: usize) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let gate = gg.linear(&format!("{prefix}.ffn_gate.weight"))?;
        let up = gg.linear(&format!("{prefix}.ffn_up.weight"))?;
        let down = gg.linear(&format!("{prefix}.ffn_down.weight"))?;
        Ok(Self { gate, up, down })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(x)?;
        let up = self.up.forward(x)?;
        let h = crate::ops::fused_ops::swiglu::swiglu(&gate, &up)?;
        self.down.forward(&h)
    }
}

// ── DecoderLayer ────────────────────────────────────────────────────────

/// One transformer block. `LayerImpl` selects which attention path runs.
///
/// Per-layer GDN state is passed in via the [`DecoderLayer::forward`] signature
/// (a `&mut Option<GdnLayerCache>`); the model holds the canonical cache array.
pub struct DecoderLayer {
    layer_impl: LayerImpl,
    input_layernorm: Qwen35RmsNorm,
    post_attention_layernorm: Qwen35RmsNorm,
    mlp: MlpOrMoe<Mlp>,
    /// Pre-computed dims for the GDN path. `None` for full-attention blocks.
    gdn_dims: Option<GdnDims>,
}

enum LayerImpl {
    FullAttention(FullAttention),
    LinearAttention(GatedDeltaNet),
}

impl DecoderLayer {
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing or has an unexpected shape.
    pub fn load(
        cfg: &TextConfig,
        layer_idx: usize,
        layer_type: LayerType,
        vb: VarBuilder,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let input_layernorm =
            Qwen35RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, &vb.pp("input_layernorm"))?;
        let post_attention_layernorm = Qwen35RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            &vb.pp("post_attention_layernorm"),
        )?;
        // `MoE` experts load dense: `quant` (in-situ quantization) applies to
        // the attention/GDN projections and dense MLPs only.
        let mlp = match cfg.moe_config() {
            Some(moe) => MlpOrMoe::Moe(SparseMoeBlock::new(
                &moe,
                layer_idx,
                cfg.hidden_size,
                vb.pp("mlp"),
                vb.device(),
            )?),
            None => MlpOrMoe::Dense(Mlp::load(cfg, &vb.pp("mlp"), quant)?),
        };

        let (layer_impl, gdn_dims) = match layer_type {
            LayerType::FullAttention => (
                LayerImpl::FullAttention(FullAttention::load(cfg, &vb.pp("self_attn"), quant)?),
                None,
            ),
            LayerType::LinearAttention => {
                let dims = GdnDims::new(cfg);
                let gdn = GatedDeltaNet::load(vb, cfg, GdnInputProjectionKind::Split, quant)?;
                (LayerImpl::LinearAttention(gdn), Some(dims))
            },
        };

        Ok(Self {
            layer_impl,
            input_layernorm,
            post_attention_layernorm,
            mlp,
            gdn_dims,
        })
    }

    /// Construct from GGUF quantized weights (llama.cpp `qwen35` layout).
    ///
    /// Block norms (`attn_norm`, `post_attention_norm`) carry the folded `+1`
    /// offset; linear-attention blocks load through
    /// [`GatedDeltaNet::from_gguf`], which documents their tensor naming and
    /// value-head order.
    ///
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing.
    pub fn from_gguf<R: Read + Seek>(
        cfg: &TextConfig,
        layer_type: LayerType,
        gg: &mut Gguf<R>,
        layer_idx: usize,
        expert_device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let input_layernorm = Qwen35RmsNorm::from_folded(
            gg.dequant_tensor(&format!("{prefix}.attn_norm.weight"))?,
            cfg.rms_norm_eps,
        );
        let post_attention_layernorm = Qwen35RmsNorm::from_folded(
            gg.dequant_tensor(&format!("{prefix}.post_attention_norm.weight"))?,
            cfg.rms_norm_eps,
        );
        // Layer kind follows tensor presence, as in the Qwen3 loader.
        let mlp = match cfg.moe_config() {
            Some(moe) if gg.contains_tensor(&format!("{prefix}.ffn_gate_inp.weight")) => {
                MlpOrMoe::Moe(SparseMoeBlock::new_from_gguf(
                    &moe,
                    gg,
                    layer_idx,
                    expert_device,
                )?)
            },
            _ => MlpOrMoe::Dense(Mlp::from_gguf(gg, layer_idx)?),
        };

        let (layer_impl, gdn_dims) = match layer_type {
            LayerType::FullAttention => (
                LayerImpl::FullAttention(FullAttention::from_gguf(cfg, gg, layer_idx)?),
                None,
            ),
            LayerType::LinearAttention => {
                let (gdn, dims) = GatedDeltaNet::from_gguf(gg, layer_idx, cfg)?;
                (LayerImpl::LinearAttention(gdn), Some(dims))
            },
        };

        Ok(Self {
            layer_impl,
            input_layernorm,
            post_attention_layernorm,
            mlp,
            gdn_dims,
        })
    }

    /// See [`FullAttention::set_attention_dtype`]; no-op for GDN layers.
    pub fn set_attention_dtype(&mut self, dtype: Option<DType>) {
        if let LayerImpl::FullAttention(attn) = &mut self.layer_impl {
            attn.set_attention_dtype(dtype);
        }
    }

    #[must_use]
    pub fn is_linear(&self) -> bool {
        matches!(self.layer_impl, LayerImpl::LinearAttention(_))
    }

    /// Forward pass. For `LinearAttention` blocks pass `Some(gdn_cache)` and
    /// `None` for `attn_cache`; for `FullAttention` blocks pass `Some(attn_cache)`
    /// and `None` for `gdn_cache`.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward(
        &self,
        x: &Tensor,
        rope: RopeSlice<'_>,
        attention_mask: Option<&Tensor>,
        gdn_cache: Option<&mut GdnLayerCache>,
        attn_cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        use crate::utils::prof::{Span, timed};

        let residual = x;
        let normed = timed(Span::BlockNorm, || self.input_layernorm.forward(x))?;

        let attn_out = match &self.layer_impl {
            LayerImpl::FullAttention(attn) => {
                debug_assert!(
                    gdn_cache.is_none(),
                    "full-attention layer should not receive a GDN cache"
                );
                timed(Span::Attn, || {
                    attn.forward(&normed, rope, attention_mask, attn_cache)
                })?
            },
            LayerImpl::LinearAttention(gdn) => {
                debug_assert!(
                    attn_cache.is_none(),
                    "linear-attention layer should not receive a KV cache"
                );
                let cache = gdn_cache.ok_or_else(|| {
                    candle_core::Error::Msg("GDN cache missing for linear-attention layer".into())
                })?;
                let dims = self.gdn_dims.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("GDN dims missing for linear-attention layer".into())
                })?;
                timed(Span::Gdn, || gdn.forward(&normed, dims, cache))?
            },
        };

        let x = timed(Span::Resid, || residual + attn_out)?;

        let residual2 = &x;
        let normed2 = timed(Span::BlockNorm, || {
            self.post_attention_layernorm.forward(&x)
        })?;

        let mlp_out = timed(Span::Mlp, || self.mlp.forward(&normed2))?;

        timed(Span::Resid, || residual2 + mlp_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen3.8-Flash-Next has 24 query heads: full 512-query slices up to
    /// ~2.7k cells, then shrinking so scores stay within the budget.
    #[test]
    fn attention_slices_shrink_with_context() {
        assert_eq!(attn_query_slice(24, 1), ATTN_QUERY_CHUNK);
        assert_eq!(attn_query_slice(24, 2048), ATTN_QUERY_CHUNK);
        assert_eq!(attn_query_slice(24, 32_768), 42);
        assert!(24 * attn_query_slice(24, 32_768) * 32_768 * 4 <= ATTN_SCORE_BUDGET);
        assert_eq!(attn_query_slice(24, 1 << 20), 16);
    }

    /// The grouped GQA attention equals the expanded one, for decode and
    /// prefill shapes, with and without a causal mask, on every device this
    /// build has.
    #[test]
    fn grouped_sdpa_matches_expanded() -> anyhow::Result<()> {
        #[allow(unused_mut)]
        let mut devices = vec![Device::Cpu];
        #[cfg(feature = "sycl")]
        if candle_core::utils::sycl_is_available() {
            devices.push(Device::new_sycl(0)?);
        }
        // Qwen3.8-27B (24 q / 4 KV) and Qwen3.8-Flash-Next (24 / 2) layouts,
        // then Ornith-1.5-35B (16 / 2) with queries in more than one slice:
        // 512-query slices, and budget-shrunk ones (419 at 5000 cells).
        for (heads, kv_heads, seq, cells) in [
            (24, 4, 1, 300),
            (24, 2, 1, 77),
            (24, 4, 37, 300),
            (24, 2, 64, 64),
            (16, 2, 700, 900),
            (16, 2, 600, 5000),
        ] {
            let n_rep = heads / kv_heads;
            let d = 32;
            let cpu = Device::Cpu;
            let q = Tensor::randn(0f32, 1.0, (1, heads, seq, d), &cpu)?;
            let k = Tensor::randn(0f32, 1.0, (1, kv_heads, cells, d), &cpu)?;
            let v = Tensor::randn(0f32, 1.0, (1, kv_heads, cells, d), &cpu)?;
            let mask = super::super::prefill::causal_mask(seq, cells - seq, &cpu, DType::F32)?;
            for dev in &devices {
                let on = |t: &Tensor| t.to_device(dev);
                for m in [None, Some(&mask)] {
                    let m = m.map(on).transpose()?;
                    let want =
                        expanded_sdpa(&on(&q)?, &on(&k)?, &on(&v)?, m.as_ref(), 0.17, n_rep)?;
                    let got = grouped_sdpa(&on(&q)?, &on(&k)?, &on(&v)?, m.as_ref(), 0.17, n_rep)?;
                    assert_eq!(got.dims(), &[1, seq, heads * d]);
                    let diff = (got - want)?.abs()?.max_all()?.to_scalar::<f32>()?;
                    assert!(
                        diff < 1e-4,
                        "{dev:?} heads {heads}/{kv_heads} seq {seq} masked {}: diff {diff}",
                        m.is_some()
                    );
                }
            }
        }
        Ok(())
    }
}
