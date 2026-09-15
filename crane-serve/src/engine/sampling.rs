//! Token sampling utilities.
//!
//! Includes:
//! - Repetition penalty (in-place, GPU-friendly)
//! - Gumbel-max sampling (GPU-native, no CPU round-trip)
//! - Top-k / top-p filtering
//!
//! All routines are designed for zero-copy GPU operation where possible: on a
//! device with the fused ops (CUDA or ROCm) the only host transfer per token is
//! the sampled index itself. Set `CRANE_SAMPLE_TRACE=1` to log the path taken
//! and its latency at `debug` level.

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use tracing::debug;

use super::grammar::{TokenMask, apply_grammar_mask, suppress_eos_inplace};
use super::sequence::Sequence;

/// Whether `device` has `crane_core::ops::topk_indices` / `gpu_argmax` backed
/// by real kernels, i.e. whether the device-side sampling path applies.
///
/// Off it, sampling copies the whole logits vector to the host and sorts there
/// — ~1 MB and ~24 ms per decoded token at a 250 K vocabulary. ROCm used to be
/// excluded here because the fused ops were CUDA-only; they are not any more.
#[must_use]
fn has_gpu_sampling(device: &Device) -> bool {
    crate::is_gpu_device(device)
}

/// Emit one `CRANE_SAMPLE_TRACE=1` line naming the path [`sample`] took.
///
/// Every exit of [`sample`] traces, including the device-side top-k ones. They
/// used to be silent, which left the trace facility dead on exactly the
/// configuration people benchmark (`top_k` + `top_p` together).
#[allow(clippy::cast_possible_truncation)]
fn trace_sample(seq_id: &str, path: &'static str, seq: &Sequence, top_k: usize, t0: Instant) {
    debug!(
        id = %seq_id,
        path,
        top_k,
        top_p = ?seq.top_p,
        temp = ?seq.temperature,
        total_us = t0.elapsed().as_micros() as u64,
        "sample"
    );
}

/// Persistent buffers for GPU-side top-k/top-p sampling.
///
/// Reuses GPU allocations across steps to avoid repeated mallocs.
pub struct SamplingBuffers {
    pub topk_cumsum_mats: HashMap<usize, Tensor>,
    pub topk_shift_bufs: HashMap<usize, Tensor>,
    pub topk_shift_idxs: HashMap<usize, Tensor>,
    pub topk_neg_vecs: HashMap<usize, Tensor>,
}

impl Default for SamplingBuffers {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplingBuffers {
    #[must_use]
    pub fn new() -> Self {
        Self {
            topk_cumsum_mats: HashMap::new(),
            topk_shift_bufs: HashMap::new(),
            topk_shift_idxs: HashMap::new(),
            topk_neg_vecs: HashMap::new(),
        }
    }

    /// # Errors
    ///
    /// Returns an error if tensor allocation on `device` fails.
    pub fn get_topk_neg_vec(&mut self, k: usize, device: &Device) -> candle_core::Result<Tensor> {
        if let Some(t) = self.topk_neg_vecs.get(&k)
            && t.device().same_device(device)
        {
            return Ok(t.clone());
        }
        let t = Tensor::full(-1e9f32, k, device)?;
        self.topk_neg_vecs.insert(k, t.clone());
        Ok(t)
    }

    /// # Errors
    ///
    /// Returns an error if `k <= 1` or tensor allocation on `device` fails.
    pub fn get_topk_shift_idx(&mut self, k: usize, device: &Device) -> candle_core::Result<Tensor> {
        if let Some(t) = self.topk_shift_idxs.get(&k)
            && t.device().same_device(device)
        {
            return Ok(t.clone());
        }
        if k <= 1 {
            candle_core::bail!("get_topk_shift_idx expects k > 1")
        }
        #[allow(clippy::cast_possible_truncation)]
        let t = Tensor::arange(1u32, k as u32, device)?;
        self.topk_shift_idxs.insert(k, t.clone());
        Ok(t)
    }

    /// # Errors
    ///
    /// Returns an error if tensor allocation on `device` fails.
    pub fn get_topk_shift_buf(
        &mut self,
        k: usize,
        device: &Device,
        dtype: DType,
    ) -> candle_core::Result<Tensor> {
        if let Some(t) = self.topk_shift_bufs.get(&k)
            && t.device().same_device(device)
            && t.dtype() == dtype
        {
            return Ok(t.clone());
        }
        let t = Tensor::zeros(k, dtype, device)?;
        self.topk_shift_bufs.insert(k, t.clone());
        Ok(t)
    }

    /// # Errors
    ///
    /// Returns an error if tensor allocation on `device` fails.
    pub fn get_topk_cumsum_mat(
        &mut self,
        k: usize,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        if let Some(t) = self.topk_cumsum_mats.get(&k)
            && t.device().same_device(device)
        {
            return Ok(t.clone());
        }
        let mut data = Vec::with_capacity(k * k);
        for row in 0..k {
            for col in 0..k {
                data.push(if row <= col { 1f32 } else { 0f32 });
            }
        }
        let t = Tensor::from_vec(data, (k, k), device)?;
        self.topk_cumsum_mats.insert(k, t.clone());
        Ok(t)
    }
}

/// Sample a token from logits for a specific sequence.
///
/// Supports:
/// - Greedy decoding (temperature ≤ 0)
/// - Top-k filtering with GPU-native Gumbel-max sampling
/// - Top-p (nucleus) filtering with cumulative softmax masking
/// - CPU fallback via `LogitsProcessor` when needed
///
/// # Errors
///
/// Returns an error if a tensor operation fails.
// The branching by device/top-k/top-p is one cohesive decode path; splitting
// it up would scatter state across smaller functions rather than clarify it.
#[allow(clippy::too_many_lines)]
pub fn sample(
    seq_id: &str,
    seq: &mut Sequence,
    logits: &Tensor,
    buffers: &mut SamplingBuffers,
) -> Result<u32> {
    let trace = std::env::var("CRANE_SAMPLE_TRACE").ok().as_deref() == Some("1");
    let t0 = Instant::now();

    // ── Fast path: greedy + no repetition penalty ──────────────────────
    // Skip the bf16→f32 conversion and use GPU argmax directly on bf16
    // logits.  Saves one dtype-conversion kernel + less DtoH.
    let greedy = match seq.temperature {
        Some(t) => t <= 0.0,
        None => false,
    };
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        // `repetition_penalty`/`frequency_penalty`/`presence_penalty` are
        // each compared against their exact "disabled" sentinel, not a
        // computed float, so strict equality is correct.
        #[allow(clippy::float_cmp)]
        if greedy
            && seq.repetition_penalty == 1.0
            && seq.frequency_penalty == 0.0
            && seq.presence_penalty == 0.0
            && seq.grammar.as_ref().is_none_or(|g| {
                matches!(g.token_mask(), TokenMask::Unconstrained) && g.allows_eos()
            })
            && has_gpu_sampling(logits.device())
        {
            let flat = logits.squeeze(0)?.squeeze(0)?;
            let token = crane_core::ops::gpu_argmax(&flat)?;
            if trace {
                let t_done = Instant::now();
                #[allow(clippy::cast_possible_truncation)]
                let total_us = t_done.duration_since(t0).as_micros() as u64;
                tracing::debug!(
                    id = %seq_id,
                    total_us,
                    "sample(gpu_argmax_fast)"
                );
            }
            return Ok(token);
        }
    }

    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
    let t_preprocessed = Instant::now();

    // Shared trailing-context window for both penalty types below.
    let start_at = seq.tokens.len().saturating_sub(seq.repeat_last_n);

    apply_penalties_inplace(
        &logits,
        seq.repetition_penalty,
        seq.frequency_penalty,
        seq.presence_penalty,
        &seq.tokens[start_at..],
    )
    .map_err(anyhow::Error::from)?;
    let t_penalty_applied = Instant::now();

    // ── Grammar constraint (e.g. the tool-call XML skeleton) ───────────
    // `token_mask()` never includes EOS ids on its own (a grammar holds no
    // request-specific EOS state): for `AllowOnly`, it's added to the
    // allow-list when permitted and left out (so `apply_grammar_mask`
    // blanks it) when not; for `Unconstrained`, there's no allow-list to
    // exclude it from, so it's suppressed separately when not permitted.
    let logits = if let Some(g) = seq.grammar.as_ref() {
        match g.token_mask() {
            TokenMask::Unconstrained => {
                if !g.allows_eos() {
                    suppress_eos_inplace(&logits, &seq.eos_token_id)
                        .map_err(anyhow::Error::from)?;
                }
                logits
            },
            TokenMask::AllowOnly(mut ids) => {
                if g.allows_eos() {
                    ids.extend_from_slice(&seq.eos_token_id);
                }
                if super::grammar_trace_enabled() {
                    let idx = Tensor::new(ids.as_slice(), logits.device())?;
                    let vals = logits
                        .gather(&idx, candle_core::D::Minus1)?
                        .to_vec1::<f32>()?;
                    let pairs: Vec<String> = ids
                        .iter()
                        .zip(vals.iter())
                        .map(|(id, v)| format!("{id}={v:.3}"))
                        .collect();
                    debug!(id = %seq_id, logits = %pairs.join(","), "CRANE_GRAMMAR_TRACE pre-mask logits");
                }
                apply_grammar_mask(&logits, &TokenMask::AllowOnly(ids))
                    .map_err(anyhow::Error::from)?
            },
        }
    } else {
        logits
    };

    if greedy {
        return Ok(logits.argmax(0)?.to_scalar::<u32>()?);
    }

    if has_gpu_sampling(logits.device()) {
        let top_p = seq.top_p.unwrap_or(1.0);
        let top_p_active = top_p > 0.0 && top_p < 1.0;
        let vocab = logits.dim(0)?;
        let temperature = seq.temperature.unwrap_or(1.0);

        let mut top_k = seq.top_k.unwrap_or(0);
        if top_k == 0 && top_p_active {
            // For large vocabularies (>64 K tokens) where top_k was NOT
            // explicitly requested, avoid the expensive GPU topk kernel.
            // Fall back to CPU LogitsProcessor which handles temperature +
            // top-p natively and only needs a ~600 KB DtoH copy.
            // Set CRANE_FORCE_GPU_TOPK=1 to override this heuristic.
            if vocab > 65536 && std::env::var("CRANE_FORCE_GPU_TOPK").ok().as_deref() != Some("1") {
                let next_token = seq.logits_processor.sample(&logits)?;
                if trace {
                    let t_done = Instant::now();
                    #[allow(clippy::cast_possible_truncation)]
                    let total_us = t_done.duration_since(t0).as_micros() as u64;
                    debug!(
                        id = %seq_id,
                        vocab,
                        top_p = ?seq.top_p,
                        temp = ?seq.temperature,
                        total_us,
                        "sample(cpu_logits_processor)"
                    );
                }
                return Ok(next_token);
            }
            top_k = std::env::var("CRANE_TOPP_FALLBACK_TOPK")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(64);
        }
        top_k = top_k.min(64).min(vocab);

        if top_k > 0 && top_k < vocab {
            let topk_idx =
                crane_core::ops::topk_indices(&logits, top_k).map_err(anyhow::Error::from)?;
            let topk_logits = logits.gather(&topk_idx, candle_core::D::Minus1)?;
            let t_after_topk = Instant::now();

            if std::env::var("CRANE_TOPK_SAMPLE_ON_CPU").ok().as_deref() == Some("1") {
                let idx_cpu = topk_idx.to_vec1::<u32>()?;
                let logits_cpu = topk_logits.to_vec1::<f32>()?;
                let cpu_logits = Tensor::from_vec(logits_cpu, top_k, &Device::Cpu)?;

                let pos = seq.logits_processor.sample(&cpu_logits)?;
                let token = idx_cpu
                    .get(pos as usize)
                    .copied()
                    .unwrap_or_else(|| idx_cpu[0]);

                if trace {
                    let t_done = Instant::now();
                    #[allow(clippy::cast_possible_truncation)]
                    let (prep_us, rep_us, topk_us, total_us) = (
                        t_preprocessed.duration_since(t0).as_micros() as u64,
                        t_penalty_applied.duration_since(t_preprocessed).as_micros() as u64,
                        t_after_topk.duration_since(t_penalty_applied).as_micros() as u64,
                        t_done.duration_since(t0).as_micros() as u64,
                    );
                    debug!(
                        id = %seq_id,
                        top_k,
                        top_p = ?seq.top_p,
                        temp = ?seq.temperature,
                        prep_us,
                        rep_us,
                        topk_us,
                        total_us,
                        "sample(topk->cpu)"
                    );
                }
                return Ok(token);
            }

            if top_p_active {
                let scaled = (&topk_logits / temperature)?;
                let probs = candle_nn::ops::softmax_last_dim(&scaled)?;
                let cumsum_mat = buffers.get_topk_cumsum_mat(top_k, logits.device())?;
                let cumsum = probs
                    .reshape((1, top_k))?
                    .matmul(&cumsum_mat)?
                    .reshape(top_k)?;
                let mask_le = cumsum.le(top_p)?;

                let shift = buffers.get_topk_shift_buf(top_k, logits.device(), mask_le.dtype())?;
                shift.zero_set()?;
                if top_k > 1 {
                    let idx = buffers.get_topk_shift_idx(top_k, logits.device())?;
                    let src = mask_le.narrow(candle_core::D::Minus1, 0, top_k - 1)?;
                    shift.scatter_set(&idx, &src, candle_core::D::Minus1)?;
                }
                let mask = (&mask_le + &shift)?.gt(0f64)?;

                let neg = buffers.get_topk_neg_vec(top_k, logits.device())?;
                let masked = mask.where_cond(&topk_logits, &neg)?;
                let mut pos = sample_gumbel_max_idx(&masked, temperature)?;
                if pos.rank() == 0 {
                    pos = pos.unsqueeze(0)?;
                }
                let token = topk_idx.gather(&pos, candle_core::D::Minus1)?;
                let token = token.squeeze(0)?.to_scalar::<u32>()?;
                if trace {
                    trace_sample(seq_id, "gpu_topk_topp", seq, top_k, t0);
                }
                return Ok(token);
            }

            let mut pos = sample_gumbel_max_idx(&topk_logits, temperature)?;
            if pos.rank() == 0 {
                pos = pos.unsqueeze(0)?;
            }
            let token = topk_idx.gather(&pos, candle_core::D::Minus1)?;
            let token = token.squeeze(0)?.to_scalar::<u32>()?;
            if trace {
                trace_sample(seq_id, "gpu_topk", seq, top_k, t0);
            }
            return Ok(token);
        }
    }

    let top_p = seq.top_p.unwrap_or(1.0);
    if top_p <= 0.0 || top_p >= 1.0 {
        let temperature = seq.temperature.unwrap_or(1.0);
        let idx = sample_gumbel_max_idx(&logits, temperature).map_err(anyhow::Error::from)?;
        let token = idx.to_scalar::<u32>()?;
        if trace {
            trace_sample(seq_id, "gumbel_full_vocab", seq, 0, t0);
        }
        return Ok(token);
    }

    let next_token = seq.logits_processor.sample(&logits)?;
    if trace {
        trace_sample(seq_id, "cpu_logits_processor_fallback", seq, 0, t0);
    }
    Ok(next_token)
}

/// Gumbel-max trick for GPU-native categorical sampling.
///
/// # Errors
///
/// Returns an error if a tensor operation fails.
// `temperature == 1.0` is the exact "no scaling" sentinel, not a computed value.
#[allow(clippy::float_cmp)]
pub fn sample_gumbel_max_idx(logits: &Tensor, temperature: f64) -> candle_core::Result<Tensor> {
    if temperature <= 0.0 {
        return logits.argmax(candle_core::D::Minus1);
    }
    let minus_g = logits.rand_like(1e-7, 0.999)?.log()?.neg()?.log()?;
    if temperature == 1.0 {
        (logits - minus_g)?.argmax(candle_core::D::Minus1)
    } else {
        ((logits / temperature)? - minus_g)?.argmax(candle_core::D::Minus1)
    }
}

/// Apply repetition, frequency, and presence penalties to `logits` in-place
/// (GPU-friendly scatter/gather).
///
/// `repetition_penalty` (1.0 = disabled) is a flat multiplicative penalty:
/// the same factor is applied whether a token appeared once or a hundred
/// times. `frequency_penalty` (0.0 = disabled) is subtracted once per
/// occurrence of a token in `context`, so it grows with repeat count and can
/// break short repetition loops (period a few tokens) that a flat penalty
/// cannot — such loops keep every token in the loop at the same "seen"
/// penalty forever, never changing their relative ranking.
/// `presence_penalty` (0.0 = disabled) is subtracted once for any token
/// that appears at all, regardless of count. Negative `frequency_penalty`/
/// `presence_penalty` values are permitted (per the `OpenAI` API) and boost
/// rather than penalize a token's logit.
///
/// `context` is normally the caller's trailing `repeat_last_n`-token window,
/// not the full generated output, so occurrences older than that window are
/// not counted for any of the three penalties.
///
/// All three penalties are computed from a single token-occurrence count
/// and applied via one gather + scatter round-trip, rather than one
/// round-trip per penalty type, since this runs once per decode step.
/// Repetition penalty (multiplicative) is applied first, then
/// frequency/presence penalty (additive) is subtracted from the result.
///
/// # Errors
///
/// Returns an error if a tensor operation fails.
pub fn apply_penalties_inplace(
    logits: &Tensor,
    repetition_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    context: &[u32],
) -> candle_core::Result<()> {
    // `repetition_penalty`/`frequency_penalty`/`presence_penalty` are each
    // compared against their exact "disabled" sentinel, not a computed
    // float, so strict equality is correct.
    #[allow(clippy::float_cmp)]
    let repetition_active = repetition_penalty != 1.0;
    #[allow(clippy::float_cmp)]
    let freq_presence_active = frequency_penalty != 0.0 || presence_penalty != 0.0;
    if context.is_empty() || (!repetition_active && !freq_presence_active) {
        return Ok(());
    }

    let mut counts: HashMap<u32, u32> = HashMap::with_capacity(context.len());
    for &t in context {
        *counts.entry(t).or_insert(0) += 1;
    }

    let mut token_ids: Vec<u32> = counts.keys().copied().collect();
    token_ids.sort_unstable();

    let idx = Tensor::new(token_ids.as_slice(), logits.device())?;
    let selected = logits.gather(&idx, candle_core::D::Minus1)?;

    let selected = if repetition_active {
        let mask = selected.ge(0f64)?;
        let on_true = (&selected / f64::from(repetition_penalty))?;
        let on_false = (&selected * f64::from(repetition_penalty))?;
        mask.where_cond(&on_true, &on_false)?
    } else {
        selected
    };

    let updated = if freq_presence_active {
        let deltas: Vec<f32> = token_ids
            .iter()
            .map(|id| {
                // Counts are bounded by `context.len()`, which is always far
                // smaller than f32's 2^24 exact-integer limit.
                #[allow(clippy::cast_precision_loss)]
                let count = counts[id] as f32;
                count * frequency_penalty + presence_penalty
            })
            .collect();
        let penalty_tensor = Tensor::new(deltas.as_slice(), logits.device())?;
        (&selected - &penalty_tensor)?
    } else {
        selected
    };

    logits.scatter_set(&idx, &updated, candle_core::D::Minus1)
}

/// Generate a random seed from system time.
#[must_use]
pub fn rand_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    #[allow(clippy::cast_possible_truncation)]
    let seed = nanos as u64;
    seed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(values: &[f32]) -> Tensor {
        Tensor::new(values, &Device::Cpu).unwrap()
    }

    #[test]
    // All three penalties disabled (sentinel values) must leave logits untouched.
    fn all_penalties_noop_when_disabled() {
        let t = logits(&[1.0, 2.0, 3.0]);
        apply_penalties_inplace(&t, 1.0, 0.0, 0.0, &[0, 1, 2]).unwrap();
        assert_eq!(t.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    // Empty context must leave logits untouched even with penalties set.
    fn all_penalties_noop_when_context_empty() {
        let t = logits(&[1.0, 2.0, 3.0]);
        apply_penalties_inplace(&t, 1.1, 0.5, 0.5, &[]).unwrap();
        assert_eq!(t.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    // repetition_penalty alone: positive logits are divided by the penalty,
    // negative logits are multiplied by it, mirroring the pre-merge
    // `apply_repeat_penalty_inplace` behavior for a single-token context.
    fn repetition_penalty_alone_scales_by_sign() {
        let t = logits(&[10.0, -10.0, 3.0]);
        apply_penalties_inplace(&t, 2.0, 0.0, 0.0, &[0, 1]).unwrap();
        let out = t.to_vec1::<f32>().unwrap();
        assert!(
            (out[0] - 5.0).abs() < 1e-6,
            "positive logit should be divided"
        );
        assert!(
            (out[1] - -20.0).abs() < 1e-6,
            "negative logit should be multiplied"
        );
        assert!(
            (out[2] - 3.0).abs() < 1e-6,
            "unseen token must be untouched"
        );
    }

    #[test]
    // Combining repetition_penalty with frequency_penalty in one call must
    // apply repetition's multiplicative scaling first, then subtract the
    // frequency penalty from the *scaled* result, not compute them
    // independently from the original logit.
    fn repetition_and_frequency_penalty_combine_in_order() {
        let t = logits(&[10.0, 10.0]);
        apply_penalties_inplace(&t, 2.0, 1.0, 0.0, &[0, 0, 1]).unwrap();
        let out = t.to_vec1::<f32>().unwrap();
        // token 0: (10.0 / 2.0) - (2 occurrences * 1.0) = 3.0
        assert!((out[0] - 3.0).abs() < 1e-6);
        // token 1: (10.0 / 2.0) - (1 occurrence * 1.0) = 4.0
        assert!((out[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    // frequency_penalty must scale with occurrence count: token 0 appears
    // three times and should be penalized 3x more than token 1, which
    // appears once. Token 2 never appears and is untouched.
    fn frequency_penalty_scales_with_occurrence_count() {
        let t = logits(&[10.0, 10.0, 10.0]);
        apply_penalties_inplace(&t, 1.0, 0.5, 0.0, &[0, 0, 0, 1]).unwrap();
        assert_eq!(
            t.to_vec1::<f32>().unwrap(),
            vec![10.0 - 1.5, 10.0 - 0.5, 10.0]
        );
    }

    #[test]
    // presence_penalty is flat: token 0 (3 occurrences) and token 1 (1
    // occurrence) must be penalized by the exact same amount.
    fn presence_penalty_is_flat_regardless_of_count() {
        let t = logits(&[10.0, 10.0, 10.0]);
        apply_penalties_inplace(&t, 1.0, 0.0, 0.5, &[0, 0, 0, 1]).unwrap();
        assert_eq!(
            t.to_vec1::<f32>().unwrap(),
            vec![10.0 - 0.5, 10.0 - 0.5, 10.0]
        );
    }

    #[test]
    // Simulates a stuck decode loop: token 0 has a slightly higher logit
    // than token 1 and has therefore been picked repeatedly (5 times) while
    // token 1 was picked once. A flat penalty (like repetition_penalty)
    // treats both as equally "seen" and can never flip the argmax, so the
    // loop never breaks. frequency_penalty grows with occurrence count and
    // must flip the ranking here, breaking the loop.
    fn frequency_penalty_breaks_short_repeat_cycle() {
        let t = logits(&[5.0, 4.9]);
        let context = [0, 1, 0, 0, 0, 0];
        apply_penalties_inplace(&t, 1.0, 0.1, 0.0, &context).unwrap();
        let out = t.to_vec1::<f32>().unwrap();
        assert!((out[0] - 4.5).abs() < 1e-6);
        assert!((out[1] - 4.8).abs() < 1e-6);
        assert!(out[1] > out[0], "frequency penalty should flip the ranking");
    }

    #[test]
    // Both penalties active at once must combine additively: token 0's
    // total penalty is `count * frequency_penalty + presence_penalty`, not
    // just one or the other.
    fn frequency_and_presence_penalty_combine_additively() {
        let t = logits(&[10.0, 10.0, 10.0]);
        apply_penalties_inplace(&t, 1.0, 0.5, 0.2, &[0, 0, 0, 1]).unwrap();
        assert_eq!(
            t.to_vec1::<f32>().unwrap(),
            vec![10.0 - (3.0 * 0.5 + 0.2), 10.0 - (1.0 * 0.5 + 0.2), 10.0]
        );
    }

    #[test]
    // OpenAI's API permits negative frequency_penalty values, which must
    // boost (not penalize) a repeated token's logit.
    fn negative_frequency_penalty_boosts_logit() {
        let t = logits(&[10.0, 10.0]);
        apply_penalties_inplace(&t, 1.0, -0.5, 0.0, &[0, 0, 1]).unwrap();
        let out = t.to_vec1::<f32>().unwrap();
        assert!(
            out[0] > 10.0,
            "negative frequency_penalty should boost the logit"
        );
        assert!((out[0] - 11.0).abs() < 1e-6);
        assert!((out[1] - 10.5).abs() < 1e-6);
    }

    #[test]
    // OpenAI's API permits negative presence_penalty values, which must
    // boost (not penalize) the logit of any token that appeared at all.
    fn negative_presence_penalty_boosts_logit() {
        let t = logits(&[10.0, 10.0, 10.0]);
        apply_penalties_inplace(&t, 1.0, 0.0, -0.5, &[0, 1]).unwrap();
        let out = t.to_vec1::<f32>().unwrap();
        assert!(
            out[0] > 10.0,
            "negative presence_penalty should boost the logit"
        );
        assert!((out[0] - 10.5).abs() < 1e-6);
        assert!((out[1] - 10.5).abs() < 1e-6);
        assert!(
            (out[2] - 10.0).abs() < 1e-6,
            "unseen token must be untouched"
        );
    }
}
