//! The Qwen2 decoder backbone (`model.language_model.*` in the checkpoint).
//!
//! Thin wrapper around the shared [`TransformerBlock`]/[`RotaryEmbedding`]
//! stack — see those modules' doc comments for the weight-name contract.
//! [`KugelAudioDecoder::forward_embeds`] takes `inputs_embeds` and returns
//! hidden states at every position (the diffusion head uses them as condition
//! at each speech position). No sliding-window attention (reference
//! checkpoint has `use_sliding_window: false`). `lm_head` lives in
//! `model.rs` — `lm_head.weight` is a sibling of `model.*`, not nested under
//! `language_model.*`.
//!
//! # In-situ quantization (ISQ)
//!
//! [`KugelAudioDecoder::new_with_quant`] quantizes every layer's Q/K/V/O
//! and MLP projections to a `GgmlDType` as they're loaded. The ~7B-parameter
//! backbone is the bulk of the ~18.7GB checkpoint, so this is the lever for
//! low-VRAM/low-RAM machines. Needs its own [`QuantizedLayer`] rather than
//! the shared [`TransformerBlock`]: Qwen2 uses biased Q/K/V projections but
//! `QMatMul` has no bias — [`crate::ops::linear::quantize_linear`] carries
//! the bias alongside the quantized weight, but `TransformerBlock`'s
//! `with_tracing::Linear` fields can't hold a `LinearLayer`. `QuantizedLayer`
//! otherwise mirrors `TransformerBlock`/`GqaAttention` exactly (pre-norm +
//! standard GQA math, no flash). `QMatMul` runs identically on CUDA and Metal.

#![allow(clippy::needless_pass_by_value)] // VarBuilder by-value is the candle idiom
#![allow(clippy::cast_precision_loss)] // head_dim->f64 sqrt is intentional (bounded)
#![allow(clippy::too_many_arguments)] // ISQ loader has many params
#![allow(clippy::missing_errors_doc)] // Result-returning helpers: errors are candle tensor errors
#![allow(clippy::missing_panics_doc)] // expect() in tests / debug_asserts only
#![allow(clippy::must_use_candidate)] // getters are conventionally used at call sites
#![allow(clippy::doc_markdown)] // KugelAudio is the model name, not generic Markdown text

use candle_core::quantized::GgmlDType;
use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, VarBuilder};

use crate::models::modules::attention::{AttentionConfig, RopeMode};
use crate::models::modules::embedding::EmbeddingLayer;
use crate::models::modules::kv_cache;
use crate::models::modules::rotary::RotaryEmbedding;
use crate::models::modules::transformer::TransformerBlock;
use crate::models::utils::repeat_kv;
use crate::models::with_tracing::RmsNorm;
use crate::ops::linear::{LinearLayer, quantize_linear_onto};

use super::config::DecoderConfig;

/// GQA attention built from [`LinearLayer`] projections — can't reuse
/// [`GqaAttention`] since that holds `with_tracing::Linear`.
///
/// [`GqaAttention`]: crate::models::modules::attention::GqaAttention
#[derive(Clone)]
struct QuantizedAttention {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    o_proj: LinearLayer,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_kv_groups: usize,
    kv_cache: Option<(Tensor, Tensor)>,
    cache_seq_len: usize,
}

impl QuantizedAttention {
    /// `vb_cpu` must be scoped to [`Device::Cpu`] — see
    /// [`quantize_linear_onto`] and the module doc comment.
    fn new(
        cfg: &DecoderConfig,
        quant: GgmlDType,
        vb_cpu: VarBuilder,
        target_device: &Device,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        Ok(Self {
            // Qwen2 keeps biased Q/K/V projections, no output bias.
            q_proj: quantize_linear_onto(
                cfg.hidden_size,
                q_dim,
                true,
                vb_cpu.pp("q_proj"),
                quant,
                target_device,
            )?,
            k_proj: quantize_linear_onto(
                cfg.hidden_size,
                kv_dim,
                true,
                vb_cpu.pp("k_proj"),
                quant,
                target_device,
            )?,
            v_proj: quantize_linear_onto(
                cfg.hidden_size,
                kv_dim,
                true,
                vb_cpu.pp("v_proj"),
                quant,
                target_device,
            )?,
            o_proj: quantize_linear_onto(
                q_dim,
                cfg.hidden_size,
                false,
                vb_cpu.pp("o_proj"),
                quant,
                target_device,
            )?,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim,
            n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            kv_cache: None,
            cache_seq_len: 0,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        cos_sin: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b_sz, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (cos, sin) = cos_sin;
        let cos = cos.to_dtype(q.dtype())?;
        let sin = sin.to_dtype(q.dtype())?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;

        let cache = self.kv_cache.take();
        let prev_seq_len = std::mem::replace(&mut self.cache_seq_len, 0);
        let update = kv_cache::update_kv_cache(cache, prev_seq_len, &k, &v)?;
        self.kv_cache = Some(update.buffer);
        self.cache_seq_len = update.seq_len;

        let k = repeat_kv(update.k, self.n_kv_groups)?.contiguous()?;
        let v = repeat_kv(update.v, self.n_kv_groups)?.contiguous()?;
        let q = q.contiguous()?;

        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? * scale)?;
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };
        let input_dtype = attn_weights.dtype();
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32)?)?
            .to_dtype(input_dtype)?;
        let attn_output = attn_weights.matmul(&v)?;
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

/// SwiGLU FFN built from [`LinearLayer`] projections — mirrors
/// [`crate::models::modules::ffn::SwiGluFfn`], minus the merged-gate and
/// CUDA `fused_silu_mul` optimizations (don't apply to quantized weights).
#[derive(Clone)]
struct QuantizedFfn {
    gate_proj: LinearLayer,
    up_proj: LinearLayer,
    down_proj: LinearLayer,
    activation: Activation,
}

impl QuantizedFfn {
    /// `vb_cpu` must be scoped to [`Device::Cpu`] — see [`QuantizedAttention::new`].
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        activation: Activation,
        quant: GgmlDType,
        vb_cpu: VarBuilder,
        target_device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: quantize_linear_onto(
                hidden_size,
                intermediate_size,
                false,
                vb_cpu.pp("gate_proj"),
                quant,
                target_device,
            )?,
            up_proj: quantize_linear_onto(
                hidden_size,
                intermediate_size,
                false,
                vb_cpu.pp("up_proj"),
                quant,
                target_device,
            )?,
            down_proj: quantize_linear_onto(
                intermediate_size,
                hidden_size,
                false,
                vb_cpu.pp("down_proj"),
                quant,
                target_device,
            )?,
            activation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.apply(&self.activation)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Pre-norm decoder layer over [`QuantizedAttention`]/[`QuantizedFfn`] —
/// quantized counterpart of [`TransformerBlock`].
#[derive(Clone)]
struct QuantizedLayer {
    self_attn: QuantizedAttention,
    ffn: QuantizedFfn,
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
}

impl QuantizedLayer {
    /// `vb` stays scoped to the target device (only used here for the two
    /// small `RmsNorm` weights); `vb_cpu` must be scoped to [`Device::Cpu`].
    fn new(
        cfg: &DecoderConfig,
        quant: GgmlDType,
        vb: VarBuilder,
        vb_cpu: VarBuilder,
    ) -> Result<Self> {
        let target_device = vb.device();
        Ok(Self {
            self_attn: QuantizedAttention::new(cfg, quant, vb_cpu.pp("self_attn"), target_device)?,
            ffn: QuantizedFfn::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.hidden_act,
                quant,
                vb_cpu.pp("mlp"),
                target_device,
            )?,
            input_norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attn_norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &mut self,
        x: &Tensor,
        cos_sin: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = x;
        let h = self.input_norm.forward(x)?;
        let h = self.self_attn.forward(&h, cos_sin, attention_mask)?;
        let h = (residual + h)?;
        let residual = &h;
        let out = self.post_attn_norm.forward(&h)?;
        let out = self.ffn.forward(&out)?;
        residual + out
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// The decoder's per-layer stack: `Standard` is the unquantized
/// [`TransformerBlock`]-based path; `Quantized` is used when ISQ is requested.
#[derive(Clone)]
enum DecoderLayers {
    Standard(Vec<TransformerBlock>),
    Quantized(Vec<QuantizedLayer>),
}

/// `Clone` resets every layer's KV cache (see [`TransformerBlock`]'s `Clone`
/// impl) while sharing the underlying weight tensors (`Tensor` is `Arc`-backed).
/// [`super::model::KugelAudioModel::generate`] uses this to run a second,
/// independently-cached CFG-negative decoder stream without doubling memory.
#[derive(Clone)]
pub struct KugelAudioDecoder {
    embed_tokens: EmbeddingLayer,
    layers: DecoderLayers,
    norm: RmsNorm,
    rotary_emb: RotaryEmbedding,
    device: Device,
    dtype: DType,
}

impl KugelAudioDecoder {
    pub fn new(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_quant(cfg, vb, None)
    }

    /// Like [`Self::new`], but with `quant: Some((dtype, vb_cpu))` quantizes
    /// every layer's projections in-situ to `dtype` as they're loaded — see
    /// the module doc comment. `None` is exactly [`Self::new`]'s behavior.
    ///
    /// `vb_cpu` must be a *second* `VarBuilder` over the same checkpoint,
    /// scoped identically to `vb` but with device [`Device::Cpu`] (cheap —
    /// same mmap, a second lightweight handle) — see
    /// [`quantize_linear_onto`]'s doc comment.
    ///
    /// `embed_tokens` stays dense even under ISQ: `QTensor::quantize` on its
    /// full `vocab_size (152064) * hidden_size` table spikes wired memory
    /// by 7-11GB+ on a real checkpoint — wildly disproportionate to the ~1GB
    /// the table takes in F16, and to every per-layer projection's
    /// well-behaved quantization. High row-count tensors apparently hit some
    /// non-linear cost in candle's CPU quantizer; rather than chase that
    /// upstream, this just doesn't quantize the one tensor big enough to
    /// trigger it. `lm_head` (same shape) is left dense for the same reason
    /// — see `model.rs`'s `from_pretrained_with_quant`.
    pub fn new_with_quant(
        cfg: &DecoderConfig,
        vb: VarBuilder,
        quant: Option<(GgmlDType, VarBuilder)>,
    ) -> Result<Self> {
        let embed_tokens = EmbeddingLayer::Dense(candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("embed_tokens"),
        )?);
        let head_dim = cfg.head_dim();
        let rotary_emb = RotaryEmbedding::new(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.device(),
        )?;
        let vb_l = vb.pp("layers");
        let layers = match quant {
            None => {
                let attn_cfg = AttentionConfig {
                    dim: cfg.hidden_size,
                    n_heads: cfg.num_attention_heads,
                    n_kv_heads: cfg.num_key_value_heads,
                    head_dim,
                    qkv_bias: true,
                    o_bias: false,
                    rope_mode: RopeMode::HalfSplit,
                    use_qk_norm: false,
                    norm_eps: cfg.rms_norm_eps,
                    causal: true,
                };
                let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
                for i in 0..cfg.num_hidden_layers {
                    layers.push(TransformerBlock::new(
                        &attn_cfg,
                        cfg.intermediate_size,
                        cfg.hidden_act,
                        vb_l.pp(i),
                    )?);
                    // Drain Metal's staging-buffer pool per layer instead of
                    // letting it grow for the whole ~7B-parameter stack.
                    crate::models::utils::release_load_staging(vb.device());
                }
                DecoderLayers::Standard(layers)
            },
            Some((dt, vb_cpu)) => {
                debug_assert!(
                    vb_cpu.device().is_cpu(),
                    "KugelAudioDecoder::new_with_quant: vb_cpu must be scoped to Device::Cpu"
                );
                let vb_cpu_l = vb_cpu.pp("layers");
                let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
                for i in 0..cfg.num_hidden_layers {
                    layers.push(QuantizedLayer::new(cfg, dt, vb_l.pp(i), vb_cpu_l.pp(i))?);
                    crate::models::utils::release_load_staging(vb.device());
                }
                DecoderLayers::Quantized(layers)
            },
        };
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Text-token embedding lookup — the half of KugelAudio's input
    /// embedding that isn't overwritten by spliced-in speech features.
    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    fn causal_mask(&self, b_size: usize, tgt_len: usize, seqlen_offset: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    /// Run the decoder on pre-built input embeddings, returning hidden
    /// states at every position (post final norm, pre `lm_head`).
    /// `seqlen_offset` is the number of already-cached positions (0 for a
    /// fresh prefill).
    /// # Errors
    ///
    /// Returns a candle error if any layer's forward pass fails.
    pub fn forward_embeds(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let (cos, sin) = self.rotary_emb.forward(seqlen_offset, seq_len)?;
        let cos = cos.to_dtype(self.dtype)?;
        let sin = sin.to_dtype(self.dtype)?;
        // `QuantizedAttention` is a separate, hand-rolled matmul SDPA (not
        // `GqaAttention`) where `None` means no masking at all, not "causal
        // by default" — it still needs this explicit mask. `Standard`
        // layers pass `None` instead: exactly what `GqaAttention`'s
        // `causal: true` dispatch already builds internally, letting it
        // reach the fused kernel rather than hand-building the same mask
        // here.
        let quantized_mask = match (&self.layers, seq_len <= 1) {
            (DecoderLayers::Quantized(_), false) => {
                Some(self.causal_mask(b_size, seq_len, seqlen_offset)?)
            },
            _ => None,
        };
        let mut xs = inputs_embeds.clone();
        match &mut self.layers {
            DecoderLayers::Standard(layers) => {
                for layer in layers.iter_mut() {
                    xs = layer.forward(&xs, Some((&cos, &sin)), None)?;
                }
            },
            DecoderLayers::Quantized(layers) => {
                for layer in layers.iter_mut() {
                    xs = layer.forward(&xs, (&cos, &sin), quantized_mask.as_ref())?;
                }
            },
        }
        xs.apply(&self.norm)
    }

    pub fn clear_kv_cache(&mut self) {
        match &mut self.layers {
            DecoderLayers::Standard(layers) => {
                for layer in layers.iter_mut() {
                    layer.clear_kv_cache();
                }
            },
            DecoderLayers::Quantized(layers) => {
                for layer in layers.iter_mut() {
                    layer.clear_kv_cache();
                }
            },
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::Activation;
    use std::collections::HashMap;

    /// `hidden_size`/`q_dim`/`intermediate_size` are all multiples of 32
    /// (Q4_0/Q8_0's block size) so the quantized tests below actually
    /// exercise `LinearLayer::Quantized` instead of silently falling back
    /// to `Standard`.
    fn small_cfg() -> DecoderConfig {
        DecoderConfig {
            vocab_size: 50,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            hidden_act: Activation::Silu,
            tie_word_embeddings: false,
        }
    }

    /// Populate every tensor [`KugelAudioDecoder::new`] needs, keyed by
    /// exact weight name (see [`super::super::modules::transformer`]'s doc
    /// comment for the shared Qwen2 layout).
    fn make_vb(cfg: &DecoderConfig, dtype: DType, device: &Device) -> VarBuilder<'static> {
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let fill = |shape: &[usize]| -> Tensor {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| 0.02 * ((i % 13) as f32 - 6.0)).collect();
            Tensor::from_vec(data, shape, device).unwrap()
        };
        let ones = |n: usize| Tensor::ones(n, DType::F32, device).unwrap();
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;

        t.insert("embed_tokens.weight".into(), fill(&[cfg.vocab_size, h]));
        for i in 0..cfg.num_hidden_layers {
            let p = format!("layers.{i}");
            t.insert(format!("{p}.self_attn.q_proj.weight"), fill(&[q_dim, h]));
            t.insert(format!("{p}.self_attn.q_proj.bias"), fill(&[q_dim]));
            t.insert(format!("{p}.self_attn.k_proj.weight"), fill(&[kv_dim, h]));
            t.insert(format!("{p}.self_attn.k_proj.bias"), fill(&[kv_dim]));
            t.insert(format!("{p}.self_attn.v_proj.weight"), fill(&[kv_dim, h]));
            t.insert(format!("{p}.self_attn.v_proj.bias"), fill(&[kv_dim]));
            t.insert(format!("{p}.self_attn.o_proj.weight"), fill(&[h, q_dim]));
            t.insert(
                format!("{p}.mlp.gate_proj.weight"),
                fill(&[cfg.intermediate_size, h]),
            );
            t.insert(
                format!("{p}.mlp.up_proj.weight"),
                fill(&[cfg.intermediate_size, h]),
            );
            t.insert(
                format!("{p}.mlp.down_proj.weight"),
                fill(&[h, cfg.intermediate_size]),
            );
            t.insert(format!("{p}.input_layernorm.weight"), ones(h));
            t.insert(format!("{p}.post_attention_layernorm.weight"), ones(h));
        }
        t.insert("norm.weight".into(), ones(h));

        VarBuilder::from_tensors(t, dtype, device)
    }

    /// Prefill (4 tokens) then a decode step (1 token, `seqlen_offset = 4`)
    /// through a small synthetic decoder — checks shapes and finiteness.
    fn prefill_then_decode(device: &Device, dtype: DType, quant: Option<GgmlDType>) {
        let cfg = small_cfg();
        let vb = make_vb(&cfg, dtype, device);
        // A separate CPU-scoped VarBuilder, as `quantize_linear_onto`
        // requires — values don't need to match `vb`'s for this check.
        let quant = quant.map(|dt| (dt, make_vb(&cfg, DType::F32, &Device::Cpu)));
        let mut decoder =
            KugelAudioDecoder::new_with_quant(&cfg, vb, quant).expect("build decoder");

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), device).unwrap();
        let embeds = decoder
            .embed_tokens(&input_ids)
            .expect("embed_tokens")
            .to_dtype(dtype)
            .unwrap();
        assert_eq!(embeds.dims(), &[1, 4, cfg.hidden_size]);

        let hidden = decoder.forward_embeds(&embeds, 0).expect("prefill");
        assert_eq!(hidden.dims(), &[1, 4, cfg.hidden_size]);
        let max_abs: f32 = hidden
            .to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(max_abs.is_finite());

        let next_id = Tensor::from_vec(vec![5u32], (1, 1), device).unwrap();
        let next_embed = decoder
            .embed_tokens(&next_id)
            .expect("embed next token")
            .to_dtype(dtype)
            .unwrap();
        let decode_hidden = decoder.forward_embeds(&next_embed, 4).expect("decode step");
        assert_eq!(decode_hidden.dims(), &[1, 1, cfg.hidden_size]);
        let decode_max_abs: f32 = decode_hidden
            .to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(decode_max_abs.is_finite());
    }

    #[test]
    fn prefill_then_decode_shapes_and_finite_on_cpu() {
        prefill_then_decode(&Device::Cpu, DType::F32, None);
    }

    /// Same prefill/decode check on Metal with F16 (the macOS dtype/device).
    /// Skipped where Metal isn't available.
    #[test]
    fn prefill_then_decode_shapes_and_finite_on_metal() {
        if !candle_core::utils::metal_is_available() {
            return;
        }
        let device = Device::new_metal(0).expect("metal device");
        prefill_then_decode(&device, DType::F16, None);
    }

    /// Same check with 4-bit (`Q4_0`) ISQ on CPU. `QMatMul` is
    /// backend-agnostic, so this and the Metal/CUDA variants exercise the
    /// same code.
    #[test]
    fn prefill_then_decode_shapes_and_finite_quantized_on_cpu() {
        prefill_then_decode(&Device::Cpu, DType::F32, Some(GgmlDType::Q4_0));
    }

    /// Same 4-bit quantized check on Metal. Skipped where Metal isn't available.
    #[test]
    fn prefill_then_decode_shapes_and_finite_quantized_on_metal() {
        if !candle_core::utils::metal_is_available() {
            return;
        }
        let device = Device::new_metal(0).expect("metal device");
        prefill_then_decode(&device, DType::F16, Some(GgmlDType::Q4_0));
    }

    /// Same 4-bit quantized check on CUDA. Skipped without the `cuda`
    /// feature or when no CUDA device is available.
    #[test]
    #[cfg(feature = "cuda")]
    fn prefill_then_decode_shapes_and_finite_quantized_on_cuda() {
        if !candle_core::utils::cuda_is_available() {
            return;
        }
        let device = Device::new_cuda(0).expect("cuda device");
        prefill_then_decode(&device, DType::BF16, Some(GgmlDType::Q4_0));
    }

    /// The quantized path must agree (loosely — Q4_0 at this 32-wide scale
    /// is not a tight numerical match) with the unquantized path on
    /// identical weights, so ISQ isn't silently falling back to `Standard`
    /// nor producing garbage.
    #[test]
    fn quantized_output_is_finite_and_same_order_of_magnitude_as_unquantized() {
        let device = Device::Cpu;
        let cfg = small_cfg();
        let vb = make_vb(&cfg, DType::F32, &device);
        let mut plain = KugelAudioDecoder::new_with_quant(&cfg, vb.clone(), None).unwrap();
        // Same underlying weights as `vb` (already Cpu-scoped) for a
        // meaningful comparison, not just "both produce finite output."
        let mut quantized =
            KugelAudioDecoder::new_with_quant(&cfg, vb.clone(), Some((GgmlDType::Q4_0, vb)))
                .unwrap();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let embeds = plain.embed_tokens(&input_ids).unwrap();
        let plain_out = plain.forward_embeds(&embeds, 0).unwrap();
        let quantized_out = quantized.forward_embeds(&embeds, 0).unwrap();

        assert_eq!(plain_out.dims(), quantized_out.dims());
        let quantized_max: f32 = quantized_out
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(quantized_max.is_finite());
        let plain_max: f32 = plain_out
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        // Loose sanity bound, not a precision claim.
        assert!(
            quantized_max < plain_max.max(1.0) * 10.0,
            "quantized output magnitude ({quantized_max}) diverged from unquantized ({plain_max})"
        );
    }
}
