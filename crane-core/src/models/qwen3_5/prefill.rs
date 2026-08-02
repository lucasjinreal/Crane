//! Chunked prefill for Qwen 3.5.
//!
//! A single forward pass over the whole prompt makes the full-attention layers
//! materialise a `[B, heads, S, S]` score matrix, so peak VRAM grows with S².
//! On a 16 GB card that is what kills long prompts — not the K/V cache, which
//! is linear and small (only 1 layer in 4 is full attention).
//!
//! Feeding the prompt through the existing caches in fixed-size chunks makes
//! the peak linear in context length: attention for one chunk costs
//! `chunk × total`, and the GDN layers carry their constant-size recurrent and
//! conv state across the boundary. Chunking engages only above the chunk size,
//! so short prompts and decode steps take exactly the path they took before.
//!
//! The chunk size is [`DEFAULT_CHUNK`] tokens, overridable with
//! `CRANE_PREFILL_CHUNK`; `0` disables chunking.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::model::Qwen3_5TextModel;

/// Prefill chunk size in tokens when `CRANE_PREFILL_CHUNK` is unset.
pub const DEFAULT_CHUNK: usize = 512;

/// Chunk size from `CRANE_PREFILL_CHUNK`, or [`DEFAULT_CHUNK`]. An explicit
/// `0` (or an unparseable value) disables chunking, restoring the single-pass
/// behaviour for A/B comparisons.
pub fn chunk_size() -> usize {
    match std::env::var("CRANE_PREFILL_CHUNK") {
        Ok(v) => v.trim().parse().unwrap_or(0),
        Err(_) => DEFAULT_CHUNK,
    }
}

/// Forward pass, chunking the prefill when the prompt is long enough to be
/// worth it. Returns next-token logits `[B, vocab_size]`.
pub(super) fn forward(
    model: &mut Qwen3_5TextModel,
    input_ids: &Tensor,
    start_pos: usize,
    attention_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (_b, seq_len) = input_ids.dims2()?;
    let chunk = chunk_size();

    // A caller-supplied mask is shaped for the whole sequence at once; slicing
    // it per chunk is the caller's business, so honour it verbatim.
    if attention_mask.is_some() || chunk == 0 || seq_len <= chunk {
        let mask = match attention_mask {
            Some(m) => Some(m.clone()),
            // Decode (`seq_len == 1`) attends to everything cached, so the
            // mask is a no-op there and is skipped.
            None if seq_len > 1 => Some(causal_mask(
                seq_len,
                start_pos,
                model.device(),
                model.dtype(),
            )?),
            None => None,
        };
        let hidden = model.forward_layers(input_ids, start_pos, mask.as_ref())?;
        return model.head(&last_position(&hidden)?);
    }

    let mut last = None;
    let mut offset = 0;
    while offset < seq_len {
        let len = chunk.min(seq_len - offset);
        let ids = input_ids.narrow(1, offset, len)?.contiguous()?;
        let mask = causal_mask(len, start_pos + offset, model.device(), model.dtype())?;
        let hidden = model.forward_layers(&ids, start_pos + offset, Some(&mask))?;
        // Only the final position is ever projected, so each chunk's hidden
        // states are dropped here rather than accumulated.
        last = Some(last_position(&hidden)?);
        offset += len;
    }

    let hidden = last.ok_or_else(|| anyhow::anyhow!("empty prefill: zero-length input"))?;
    model.head(&hidden)
}

/// Last position of `[B, S, H]` as an owned `[B, 1, H]`.
///
/// Owned, not a view: a view would keep the whole chunk's hidden states alive
/// across the next chunk's forward pass.
fn last_position(hidden: &Tensor) -> Result<Tensor> {
    let s = hidden.dim(1)?;
    Ok(hidden.narrow(1, s - 1, 1)?.contiguous()?)
}

/// Additive causal mask `[1, 1, seq_q, start_pos + seq_q]`: row `i` is the
/// query at absolute position `start_pos + i` and holds `-inf` for every key
/// column beyond it.
///
/// Built at exactly `seq_q × total_k`. The previous version built the whole
/// `total_k × total_k` square and narrowed it, which both reintroduces the
/// quadratic allocation chunking exists to avoid and silently drops
/// `start_pos` — a chunk starting mid-prompt would get row 0's mask for its
/// first token instead of row `start_pos`'s.
pub(super) fn causal_mask(
    seq_q: usize,
    start_pos: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let total_k = start_pos + seq_q;
    let mut data: Vec<f32> = Vec::with_capacity(seq_q * total_k);
    for q in start_pos..total_k {
        for k in 0..total_k {
            data.push(if k > q { f32::NEG_INFINITY } else { 0.0 });
        }
    }
    Ok(Tensor::from_vec(data, (seq_q, total_k), device)?
        .to_dtype(dtype)?
        .unsqueeze(0)?
        .unsqueeze(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen3_5::config::{Config, HiddenAct, RopeParameters, TextConfig};
    use candle_core::Shape;
    use candle_nn::var_builder::SimpleBackend;
    use candle_nn::{Init, VarBuilder};

    /// Deterministic-per-model random weights for any requested tensor name.
    ///
    /// Sidesteps having to spell out every checkpoint tensor name; the model is
    /// built once and both paths run against the same weights, so the values
    /// only have to be non-degenerate, not reproducible across processes.
    struct RandWeights;

    impl SimpleBackend for RandWeights {
        fn get(
            &self,
            s: Shape,
            name: &str,
            _h: Init,
            dtype: DType,
            dev: &Device,
        ) -> candle_core::Result<Tensor> {
            let fan_in = s.dims().last().copied().unwrap_or(1) as f64;
            let t = Tensor::randn(0f32, (1.0 / fan_in.sqrt()) as f32, s, dev)?;
            // `A_log` drives the GDN decay `exp(-exp(A_log) * softplus(..))`.
            // Keep it well negative so decay stays near 1 and the recurrent
            // state actually survives a chunk boundary — otherwise the state
            // hand-off this test exists to check decays into the noise floor.
            let t = if name.ends_with("A_log") {
                t.affine(0.1, -2.0)?
            } else {
                t
            };
            t.to_dtype(dtype)
        }

        fn get_unchecked(&self, _n: &str, _d: DType, _v: &Device) -> candle_core::Result<Tensor> {
            candle_core::bail!("RandWeights requires a shape")
        }

        fn contains_tensor(&self, _name: &str) -> bool {
            false
        }
    }

    /// Small hybrid config: 8 layers, so 2 full-attention blocks among 6 GDN
    /// blocks (`full_attention_interval = 4`).
    fn tiny_config() -> Config {
        Config {
            text_config: TextConfig {
                head_dim: 16,
                vocab_size: 64,
                hidden_size: 32,
                intermediate_size: 48,
                num_hidden_layers: 8,
                num_attention_heads: 4,
                num_key_value_heads: 2,
                hidden_act: HiddenAct::Silu,
                max_position_embeddings: 128,
                rms_norm_eps: 1e-6,
                rope_parameters: RopeParameters {
                    rope_theta: 10_000.0,
                    mrope_section: Vec::new(),
                    partial_rotary_factor: 0.5,
                    mrope_interleaved: true,
                },
                full_attention_interval: 4,
                linear_conv_kernel_dim: 4,
                linear_key_head_dim: 8,
                linear_value_head_dim: 8,
                linear_num_key_heads: 2,
                linear_num_value_heads: 4,
                tie_word_embeddings: true,
                attn_output_gate: true,
            },
            vision_config: None,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
            tie_word_embeddings: true,
        }
    }

    fn logits(model: &mut Qwen3_5TextModel, ids: &Tensor) -> Vec<f32> {
        model.reset_gdn_caches().expect("reset caches");
        model
            .forward(ids, 0, None)
            .expect("forward")
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1")
    }

    /// Chunked and unchunked prefill must agree to near bit-exactness: the
    /// arithmetic is identical, only the batching of it differs. This covers
    /// every hand-off across a chunk boundary at once — the GDN conv state, the
    /// GDN recurrent state, the K/V cache, RoPE offsets and the causal mask.
    #[test]
    fn chunked_prefill_matches_single_pass() {
        let dev = Device::Cpu;
        let cfg = tiny_config();
        let vb = VarBuilder::from_backend(Box::new(RandWeights), DType::F32, dev.clone());
        let mut model = Qwen3_5TextModel::new(&cfg, vb, &dev, DType::F32, None).expect("build");

        // 23 tokens with chunk 5 gives 4 full chunks and a length-3 remainder;
        // 7 gives a length-2 one. Both are shorter than the conv kernel (4).
        let ids: Vec<u32> = (0..23u32).map(|i| (i * 7 + 3) % 64).collect();
        let ids = Tensor::new(ids.as_slice(), &dev)
            .expect("ids")
            .unsqueeze(0)
            .expect("batch dim");

        // SAFETY: `set_var`/`remove_var` are process-global; this test owns the
        // variable and Rust's test harness runs each test binary's threads in
        // the same process, so keep the whole chunked/unchunked comparison here
        // rather than splitting it across tests that could interleave.
        unsafe { std::env::set_var("CRANE_PREFILL_CHUNK", "0") };
        let single = logits(&mut model, &ids);

        for chunk in ["5", "7", "1"] {
            unsafe { std::env::set_var("CRANE_PREFILL_CHUNK", chunk) };
            let chunked = logits(&mut model, &ids);
            assert_eq!(single.len(), chunked.len());
            for (i, (a, b)) in single.iter().zip(&chunked).enumerate() {
                assert!(
                    (a - b).abs() < 1e-4,
                    "chunk={chunk} logit[{i}]: {a} != {b}"
                );
            }
        }
        unsafe { std::env::remove_var("CRANE_PREFILL_CHUNK") };

        // A degenerate model would pass the comparison trivially; make sure the
        // logits actually carry signal.
        let max = single.iter().cloned().fold(f32::MIN, f32::max);
        let min = single.iter().cloned().fold(f32::MAX, f32::min);
        assert!(max - min > 1e-3, "logits are flat ({min}..{max})");
        assert!(single.iter().all(|v| v.is_finite()), "non-finite logits");
    }

    /// Row `i` of the mask must be the causal row for absolute position
    /// `start_pos + i`, not for `i`.
    #[test]
    fn causal_mask_uses_absolute_positions() {
        let dev = Device::Cpu;
        let mask = causal_mask(2, 3, &dev, DType::F32).expect("mask");
        assert_eq!(mask.dims(), &[1, 1, 2, 5]);
        let v = mask
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1");
        // Row 0 is absolute position 3: keys 0..=3 visible, key 4 masked.
        assert_eq!(&v[0..5], &[0.0, 0.0, 0.0, 0.0, f32::NEG_INFINITY]);
        // Row 1 is absolute position 4: everything visible.
        assert_eq!(&v[5..10], &[0.0; 5]);
    }
}
