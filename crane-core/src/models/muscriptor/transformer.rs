//! Streaming causal transformer used by `MuScriptor`.
//!
//! Faithful port of the upstream `muscriptor/modules/transformer.py`:
//!
//! * Sinusoidal positions are added **once at the stack level**, in fp32,
//!   using a per-batch running offset counter so a decode step only adds
//!   position `offset + [0]` before projection.
//! * The KV cache is a [`LayerState`] holding one full K and one full V
//!   tensor, pre-allocated to `max_seq_len` and written in place via
//!   `slice_set` (the same pattern as `models::modules::kv_cache`) — an
//!   earlier `Tensor::cat`-per-step version grew the cache without bound
//!   and both corrupted decoding and OOM'd at large `max_gen_len`.
//! * Attention is bottom-right aligned. Two fast paths hit the simple
//!   matmul/softmax kernel: `T_q == 1` (decode, no mask needed) and
//!   `T_q == T_k` (prefill, full causal).
//! * No `RoPE`, no QK-norm, no GQA — plain multi-head attention.
//! * Linear projections carry no bias.
//! * Feed-forward uses GELU.
//! * Pre-norm `LayerNorm`, eps = `1e-5`.

use candle_core::quantized::GgmlDType;
use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::layer_norm::LayerNorm;
use candle_nn::{Linear, VarBuilder};

use crate::models::muscriptor::config::VariantConfig;
use crate::ops::linear::{LinearLayer, linear_layer, quantize_linear};

// ── Sinusoidal positions ────────────────────────────────────────────────

/// `[T, dim]` sinusoidal embedding matching the upstream's
/// `create_sin_embedding`: `positions / max_period ** (i / (half_dim -
/// 1))`, half-cos half-sin. Always computed in fp32. `positions` may be
/// any shape; the result is `[..., dim]` with the same leading dims.
///
/// # Errors
///
/// Returns an error if any tensor op (dtype cast, reshape, broadcast, or
/// concatenation) fails, e.g. because `dim` is odd or `positions` has an
/// unsupported dtype.
pub fn create_sin_embedding(
    positions: &Tensor,
    dim: usize,
    max_period: f64,
    dtype: DType,
) -> Result<Tensor> {
    let half_dim = dim / 2;
    let positions = positions.to_dtype(DType::F32)?;
    let lead = positions.dims();
    // Append a trailing 1 so positions broadcasts cleanly against
    // `[1, 1, half_dim]` to `[..., half_dim]`.
    let mut exp_shape: Vec<usize> = lead.to_vec();
    exp_shape.push(1);
    let positions = positions.reshape(exp_shape.as_slice())?.contiguous()?;

    // Broadcastable `[1, 1, half_dim]` divisor: `max_period ** (i / (half_dim - 1))`.
    // `affine` only builds the linear exponent `i * (-ln(max_period)/(half_dim-1))`;
    // without `.exp()` this is `positions * exponent` (linear ramp) instead of
    // `positions / max_period ** exponent` (geometric frequency falloff) — every
    // frequency band past the first collapses into high-frequency noise, which
    // corrupts positional information for any sequence longer than a couple of
    // tokens (see the doc comment above for the intended formula).
    let adim = Tensor::arange(0, half_dim as u32, positions.device())?
        .to_dtype(DType::F32)?
        .reshape((1, 1, ()))?;
    let scale = adim
        .affine(-(max_period.ln()) / ((half_dim - 1) as f64), 0.0)?
        .exp()?;
    let phase = positions.broadcast_mul(&scale)?;
    let cos = phase.cos()?;
    let sin = phase.sin()?;
    let emb = Tensor::cat(&[cos, sin], D::Minus1)?.to_dtype(dtype)?;
    Ok(emb)
}

// ── State ────────────────────────────────────────────────────────────────

/// One attention layer's running KV cache (full K, full V) plus the
/// number of valid positions.
pub struct LayerState {
    /// K prefix, BHSD: `[B, H, max_seq_len, D]`, filled `[..., :seq_len, :]`.
    pub k: Tensor,
    /// V prefix, same layout as `k`.
    pub v: Tensor,
    /// Number of valid (filled) positions.
    pub seq_len: usize,
    /// Total buffer capacity. Stays constant for the lifetime of the
    /// state; protects `slice_set` writes from out-of-range access.
    pub max_seq_len: usize,
}

impl LayerState {
    /// Advance `seq_len` by `by`. Host-side; no `.item()` call.
    #[inline]
    pub fn advance(&mut self, by: usize) {
        self.seq_len = self.seq_len.saturating_add(by);
    }
}

/// All per-layer states plus a per-batch on-device offset counter.
pub struct TransformerState {
    pub layers: Vec<LayerState>,
    /// Per-batch starting position for the next step. Single-batch
    /// callers shadow this with `layers[0].seq_len`; the field exists
    /// for future batched callers (the upstream's `state["offsets"]`).
    pub offsets: Tensor,
}

// ── Attention ────────────────────────────────────────────────────────────

struct StreamingMultiheadAttention {
    /// `[3 * E, E]` projection — stored under the flattened key
    /// `in_proj_weight` in the upstream checkpoints (matches the
    /// upstream's `nn.Linear(..., bias=False).weight` exposed via
    /// `weight` then aliased in `state_dict`). Fetched as a raw tensor and
    /// wrapped in a bias-free `Linear` (not via `linear_layer`/
    /// `linear_no_bias`'s `vb.pp("weight")` convention), because the key
    /// isn't nested under `in_proj/weight`.
    in_proj: LinearLayer,
    out_proj: LinearLayer,
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
}

impl StreamingMultiheadAttention {
    fn new(vb: VarBuilder, config: &VariantConfig, quant: Option<GgmlDType>) -> Result<Self> {
        let embed_dim = config.dim;
        let head_dim = config.head_dim();
        let num_heads = config.num_heads;
        let in_proj_weight = vb
            .get((3 * embed_dim, embed_dim), "in_proj_weight")?
            .contiguous()?;
        let in_proj = Linear::new(in_proj_weight, None);
        let in_proj = match quant {
            Some(dt) => quantize_linear(in_proj, dt)?,
            None => LinearLayer::Standard(in_proj),
        };
        let out_proj = linear_layer(embed_dim, embed_dim, vb.pp("out_proj"), quant)?;
        Ok(Self {
            in_proj,
            out_proj,
            num_heads,
            head_dim,
            embed_dim,
        })
    }

    /// One attention call. `query` is already position-encoded by the
    /// stack-level caller; the MHA itself does not add positional
    /// embedding.
    fn forward(&self, query: &Tensor, state: &mut LayerState) -> Result<Tensor> {
        let (b, q_len, _) = query.dims3()?;
        let h = self.num_heads;
        let d = self.head_dim;

        // Flatten the (B, T) batch dims for the matmul (candle's
        // Tensor::matmul requires matching rank or rank-2 inputs;
        // rank-3 × rank-2 is rejected).
        let query_flat = query.reshape((b * q_len, self.embed_dim))?;
        let projected = self.in_proj.forward(&query_flat)?;
        let packed = projected.reshape((b, q_len, 3, h, d))?;
        let q = packed.narrow(2, 0, 1)?.squeeze(2)?;
        let k = packed.narrow(2, 1, 1)?.squeeze(2)?;
        let v = packed.narrow(2, 2, 1)?.squeeze(2)?;

        // K, V from projection come out BSHD. Rearrange to BHSD for the
        // cache + matmul attention.
        let k = k.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v.permute((0, 2, 1, 3))?.contiguous()?;

        // Append to the cache in place. The buffer is pre-sized to
        // `state.max_seq_len`, so a `slice_set` write at `state.seq_len`
        // is O(q_len) and doesn't allocate. (The earlier
        // `Tensor::cat`-per-step version grew the cache without bound
        // and OOM'd at large `max_gen_len`; switching to `slice_set`
        // mirrors the `models::modules::kv_cache::FpKvCache`
        // pre-allocated buffer pattern.)
        let end = state.seq_len;
        let new_end = end + q_len;
        if new_end > state.max_seq_len {
            candle_core::bail!(
                "KV cache overflow: tried to write at offset {new_end}, max is {}",
                state.max_seq_len
            );
        }
        state.k.slice_set(&k, 2, end)?;
        state.v.slice_set(&v, 2, end)?;
        // Views over the populated prefix in BHSD layout.
        let k_view = state.k.narrow(2, 0, new_end)?;
        let v_view = state.v.narrow(2, 0, new_end)?;

        // Q for the attention matmul — flatten (B, T) the same way.
        let q_bhsd = q.reshape((b * q_len, h, d))?; // [B*T, H, D]
        let q_bhsd = q_bhsd
            .reshape((b, q_len, h, d))?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let k_bhsd = k_view;
        let v_bhsd = v_view;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        // Must match `scores`' dtype (== q/k's, i.e. the model's compute
        // dtype) for `broadcast_mul` — under f16/bf16 an F32 constant here
        // is a dtype-mismatch error, not a silent upcast.
        let scale_t = Tensor::new(&[scale as f32], q_bhsd.device())?.to_dtype(q_bhsd.dtype())?;

        let attn_bhsd = if q_len == 1 {
            // Decode — bottom-right aligned, no masking.
            let scores = q_bhsd.matmul(&k_bhsd.transpose(D::Minus2, D::Minus1)?)?;
            let scores = scores.broadcast_mul(&scale_t)?;
            let weights = candle_nn::ops::softmax_last_dim(&scores)?;
            weights.matmul(&v_bhsd)?
        } else if q_len == k_bhsd.dim(2)? {
            // Prefill — square, standard causal. `where_cond`'s CPU/CUDA
            // backends only accept U8/U32/I* conditions; the
            // `on_true` (kept scores) and `on_false` (-inf) must share
            // the same dtype (F32 here).
            let scores = q_bhsd.matmul(&k_bhsd.transpose(D::Minus2, D::Minus1)?)?;
            let scores = scores.broadcast_mul(&scale_t)?;
            let mask_u8 = causal_mask(q_len, q_bhsd.device())?
                .unsqueeze(0)?
                .unsqueeze(0)?
                .broadcast_as(scores.shape())?
                .to_dtype(DType::U8)?;
            let neg_inf = neg_inf_like(&scores)?;
            let scores = mask_u8.where_cond(&scores, &neg_inf)?;
            let weights = candle_nn::ops::softmax_last_dim(&scores)?;
            weights.matmul(&v_bhsd)?
        } else {
            // Rectangular case — bottom-right aligned causal.
            let k_len = k_bhsd.dim(2)?;
            let scores = q_bhsd.matmul(&k_bhsd.transpose(D::Minus2, D::Minus1)?)?;
            let scores = scores.broadcast_mul(&scale_t)?;
            let mask_u8 = bottom_right_causal_mask(q_len, k_len, q_bhsd.device())?
                .unsqueeze(0)?
                .unsqueeze(0)?
                .broadcast_as(scores.shape())?
                .to_dtype(DType::U8)?;
            let neg_inf = neg_inf_like(&scores)?;
            let scores = mask_u8.where_cond(&scores, &neg_inf)?;
            let weights = candle_nn::ops::softmax_last_dim(&scores)?;
            weights.matmul(&v_bhsd)?
        };

        // BHSD → BSHD → flatten H*D → out_proj (out_proj expects 2D).
        let attn = attn_bhsd
            .permute((0, 2, 1, 3))?
            .reshape((b * q_len, h * d))?;
        let out_flat = self.out_proj.forward(&attn)?;
        let out = out_flat.reshape((b, q_len, self.embed_dim))?;
        Ok(out)
    }
}

/// `[T, T]` causal mask: lower-triangular (`1` in the lower triangle,
/// `0` above) — matches `is_causal=True` in SDPA, top-left aligned,
/// fine for prefill where query and key spans are the same.
fn causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0u8; t * t];
    for i in 0..t {
        for j in 0..=i {
            data[i * t + j] = 1;
        }
    }
    Tensor::from_vec(data, (t, t), device)?.to_dtype(DType::F32)
}

/// `[T_q, T_k]` bottom-right-aligned causal mask. Row i can attend to
/// columns `j <= (T_k - T_q) + i`. Used when query and key spans are
/// rectangular (the streaming decode case where keys are the cache and
/// query is the new row).
fn bottom_right_causal_mask(t_q: usize, t_k: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0u8; t_q * t_k];
    let offset = t_k as i64 - t_q as i64;
    for i in 0..t_q {
        let bound = (offset + i as i64).max(0);
        for j in 0..=bound {
            if (j as usize) < t_k {
                data[i * t_k + j as usize] = 1;
            }
        }
    }
    Tensor::from_vec(data, (t_q, t_k), device)?.to_dtype(DType::F32)
}

fn neg_inf_like(t: &Tensor) -> Result<Tensor> {
    // `where_cond`'s on_true/on_false must share a dtype (`scores`' dtype,
    // i.e. the model's compute dtype) — an F32 literal built via
    // `Tensor::full` is a dtype-mismatch error under f16/bf16, not a
    // silent upcast.
    Tensor::full(f32::NEG_INFINITY, t.dims(), t.device())?
        .to_dtype(t.dtype())?
        .broadcast_as(t.shape())
}

// ── Layer + Stack ───────────────────────────────────────────────────────

struct StreamingTransformerLayer {
    self_attn: StreamingMultiheadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    linear1: LinearLayer,
    linear2: LinearLayer,
}

impl StreamingTransformerLayer {
    fn new(vb: VarBuilder, config: &VariantConfig, quant: Option<GgmlDType>) -> Result<Self> {
        let self_attn = StreamingMultiheadAttention::new(vb.pp("self_attn"), config, quant)?;
        let norm1 = candle_nn::layer_norm(config.dim, 1e-5, vb.pp("norm1"))?;
        let norm2 = candle_nn::layer_norm(config.dim, 1e-5, vb.pp("norm2"))?;
        let linear1 = linear_layer(
            config.dim,
            config.dim_feedforward(),
            vb.pp("linear1"),
            quant,
        )?;
        let linear2 = linear_layer(
            config.dim_feedforward(),
            config.dim,
            vb.pp("linear2"),
            quant,
        )?;
        Ok(Self {
            self_attn,
            norm1,
            norm2,
            linear1,
            linear2,
        })
    }

    fn forward(&self, x: &Tensor, state: &mut LayerState) -> Result<Tensor> {
        let normed = self.norm1.forward(x)?;
        let attn = self.self_attn.forward(&normed, state)?;
        let x = (x + attn)?;
        let normed = self.norm2.forward(&x)?;
        let h = self.linear1.forward(&normed)?;
        let h = h.gelu()?;
        let h = self.linear2.forward(&h)?;
        let out = (&x + &h)?;
        Ok(out)
    }
}

/// Whole-stack streaming transformer. Wrapped (not exposed) — public
/// callers go through [`super::LMModel`].
pub(crate) struct StreamingTransformer {
    layers: Vec<StreamingTransformerLayer>,
    max_period: f64,
    embed_dim: usize,
}

impl StreamingTransformer {
    pub(crate) fn new(
        vb: VarBuilder,
        config: &VariantConfig,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_layers);
        let layers_vb = vb.pp("layers");
        for i in 0..config.num_layers {
            layers.push(StreamingTransformerLayer::new(
                layers_vb.pp(i),
                config,
                quant,
            )?);
        }
        Ok(Self {
            layers,
            max_period: config.max_period,
            embed_dim: config.dim,
        })
    }

    pub(crate) fn init_state(
        &self,
        batch_size: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<TransformerState> {
        // Pre-allocate the KV buffers at the full expected size so
        // `slice_set` writes are in-place (no growth, no allocation
        // per decode step). Without this the cache grew unboundedly
        // and the 1.3B model OOM'd after a few hundred decode steps.
        let layers = self
            .layers
            .iter()
            .map(|_| {
                let k = Tensor::zeros(
                    (
                        batch_size,
                        self.layers[0].self_attn.num_heads,
                        max_seq_len,
                        self.layers[0].self_attn.head_dim,
                    ),
                    dtype,
                    device,
                )?;
                let v = Tensor::zeros(
                    (
                        batch_size,
                        self.layers[0].self_attn.num_heads,
                        max_seq_len,
                        self.layers[0].self_attn.head_dim,
                    ),
                    dtype,
                    device,
                )?;
                Ok(LayerState {
                    k,
                    v,
                    seq_len: 0,
                    max_seq_len,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = Tensor::zeros((batch_size,), DType::I64, device)?;
        Ok(TransformerState { layers, offsets })
    }

    /// Forward through the stack. The caller has already token-embedded
    /// (and mel-conditioned) the input; this method adds sinusoidal
    /// positions once at the stack level and then runs every layer.
    pub(crate) fn forward(&self, x: &Tensor, states: &mut TransformerState) -> Result<Tensor> {
        let offset = states.layers[0].seq_len as i64;
        let t = x.dim(1)?;

        let positions = Tensor::arange(offset as u32, (offset + t as i64) as u32, x.device())?
            .to_dtype(DType::F32)?
            // [1, t]
            .reshape((1, t))?;
        // `create_sin_embedding` reshapes positions internally to add a
        // trailing 1, so the result shares `x`'s leading dims; just
        // broadcast to match exactly.
        let pos_emb = create_sin_embedding(&positions, self.embed_dim, self.max_period, x.dtype())?
            .broadcast_as(x.shape())?;
        let h = (x + pos_emb)?;

        let mut h = h;
        for (layer, st) in self.layers.iter().zip(states.layers.iter_mut()) {
            h = layer.forward(&h, st)?;
        }
        Ok(h)
    }

    /// Advance every layer's `seq_len` cursor by `by` positions.
    pub(crate) fn advance(&self, states: &mut TransformerState, by: usize) {
        for st in &mut states.layers {
            st.advance(by);
        }
    }
}
