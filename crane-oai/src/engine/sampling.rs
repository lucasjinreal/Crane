//! Token sampling utilities.
//!
//! Includes:
//! - Repetition penalty (in-place, GPU-friendly)
//! - Gumbel-max sampling (GPU-native, no CPU round-trip)
//! - Top-k / top-p filtering
//!
//! All routines are designed for zero-copy GPU operation where possible.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use tracing::debug;

use super::sequence::Sequence;

/// Persistent buffers for GPU-side top-k/top-p sampling.
///
/// Reuses GPU allocations across steps to avoid repeated mallocs.
pub struct SamplingBuffers {
    pub topk_cumsum_mats: HashMap<usize, Tensor>,
    pub topk_shift_bufs: HashMap<usize, Tensor>,
    pub topk_shift_idxs: HashMap<usize, Tensor>,
    pub topk_neg_vecs: HashMap<usize, Tensor>,
}

impl SamplingBuffers {
    pub fn new() -> Self {
        Self {
            topk_cumsum_mats: HashMap::new(),
            topk_shift_bufs: HashMap::new(),
            topk_shift_idxs: HashMap::new(),
            topk_neg_vecs: HashMap::new(),
        }
    }

    pub fn get_topk_neg_vec(
        &mut self,
        k: usize,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        if let Some(t) = self.topk_neg_vecs.get(&k) {
            if t.device().same_device(device) {
                return Ok(t.clone());
            }
        }
        let t = Tensor::full(-1e9f32, k, device)?;
        self.topk_neg_vecs.insert(k, t.clone());
        Ok(t)
    }

    pub fn get_topk_shift_idx(
        &mut self,
        k: usize,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        if let Some(t) = self.topk_shift_idxs.get(&k) {
            if t.device().same_device(device) {
                return Ok(t.clone());
            }
        }
        if k <= 1 {
            candle_core::bail!("get_topk_shift_idx expects k > 1")
        }
        let t = Tensor::arange(1u32, k as u32, device)?;
        self.topk_shift_idxs.insert(k, t.clone());
        Ok(t)
    }

    pub fn get_topk_shift_buf(
        &mut self,
        k: usize,
        device: &Device,
        dtype: DType,
    ) -> candle_core::Result<Tensor> {
        if let Some(t) = self.topk_shift_bufs.get(&k) {
            if t.device().same_device(device) && t.dtype() == dtype {
                return Ok(t.clone());
            }
        }
        let t = Tensor::zeros(k, dtype, device)?;
        self.topk_shift_bufs.insert(k, t.clone());
        Ok(t)
    }

    pub fn get_topk_cumsum_mat(
        &mut self,
        k: usize,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        if let Some(t) = self.topk_cumsum_mats.get(&k) {
            if t.device().same_device(device) {
                return Ok(t.clone());
            }
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
pub fn sample(
    seq_id: &str,
    seq: &mut Sequence,
    logits: &Tensor,
    buffers: &mut SamplingBuffers,
) -> Result<u32> {
    let trace = std::env::var("CRANE_SAMPLE_TRACE").ok().as_deref() == Some("1");
    let t0 = Instant::now();

    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
    let t_after_prep = Instant::now();

    if seq.repetition_penalty != 1.0 {
        let start_at = seq.tokens.len().saturating_sub(seq.repeat_last_n);
        apply_repeat_penalty_inplace(&logits, seq.repetition_penalty, &seq.tokens[start_at..])
            .map_err(anyhow::Error::from)?;
    }
    let t_after_rep = Instant::now();

    let greedy = match seq.temperature {
        Some(t) => t <= 0.0,
        None => false,
    };
    if greedy {
        return Ok(logits.argmax(0)?.to_scalar::<u32>()?);
    }

    if logits.device().is_cuda() {
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
            if vocab > 65536
                && std::env::var("CRANE_FORCE_GPU_TOPK")
                    .ok()
                    .as_deref()
                    != Some("1")
            {
                let next_token = seq.logits_processor.sample(&logits)?;
                if trace {
                    let t_done = Instant::now();
                    debug!(
                        id = %seq_id,
                        vocab,
                        top_p = ?seq.top_p,
                        temp = ?seq.temperature,
                        total_us = t_done.duration_since(t0).as_micros() as u64,
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
            let topk_idx = logits.topk_indices(top_k).map_err(anyhow::Error::from)?;
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
                    debug!(
                        id = %seq_id,
                        top_k,
                        top_p = ?seq.top_p,
                        temp = ?seq.temperature,
                        prep_us = t_after_prep.duration_since(t0).as_micros() as u64,
                        rep_us = t_after_rep.duration_since(t_after_prep).as_micros() as u64,
                        topk_us = t_after_topk.duration_since(t_after_rep).as_micros() as u64,
                        total_us = t_done.duration_since(t0).as_micros() as u64,
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

                let shift =
                    buffers.get_topk_shift_buf(top_k, logits.device(), mask_le.dtype())?;
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
                return Ok(token.squeeze(0)?.to_scalar::<u32>()?);
            }

            let mut pos = sample_gumbel_max_idx(&topk_logits, temperature)?;
            if pos.rank() == 0 {
                pos = pos.unsqueeze(0)?;
            }
            let token = topk_idx.gather(&pos, candle_core::D::Minus1)?;
            return Ok(token.squeeze(0)?.to_scalar::<u32>()?);
        }
    }

    let top_p = seq.top_p.unwrap_or(1.0);
    if top_p <= 0.0 || top_p >= 1.0 {
        let temperature = seq.temperature.unwrap_or(1.0);
        let idx = sample_gumbel_max_idx(&logits, temperature).map_err(anyhow::Error::from)?;
        return Ok(idx.to_scalar::<u32>()?);
    }

    let next_token = seq.logits_processor.sample(&logits)?;
    Ok(next_token)
}

/// Gumbel-max trick for GPU-native categorical sampling.
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

/// Apply repetition penalty in-place (GPU-friendly scatter/gather).
pub fn apply_repeat_penalty_inplace(
    logits: &Tensor,
    penalty: f32,
    context: &[u32],
) -> candle_core::Result<()> {
    if context.is_empty() {
        return Ok(());
    }

    let mut unique: HashSet<u32> = HashSet::with_capacity(context.len());
    for &t in context {
        unique.insert(t);
    }
    if unique.is_empty() {
        return Ok(());
    }
    let mut token_ids: Vec<u32> = unique.into_iter().collect();
    token_ids.sort_unstable();

    let idx = Tensor::new(token_ids.as_slice(), logits.device())?;
    let selected = logits.gather(&idx, candle_core::D::Minus1)?;
    let mask = selected.ge(0f64)?;
    let on_true = (&selected / penalty as f64)?;
    let on_false = (&selected * penalty as f64)?;
    let updated = mask.where_cond(&on_true, &on_false)?;
    logits.scatter_set(&idx, &updated, candle_core::D::Minus1)
}

/// Generate a random seed from system time.
pub fn rand_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rand_seed_is_nonzero() {
        let seed = rand_seed();
        assert_ne!(seed, 0);
    }

    #[test]
    fn rand_seed_varies_across_calls() {
        let s1 = rand_seed();
        // Spin a bit to ensure time advances.
        std::thread::sleep(std::time::Duration::from_millis(1));
        let s2 = rand_seed();
        assert_ne!(s1, s2);
    }

    // ── SamplingBuffers tests ──

    #[test]
    fn sampling_buffers_new_is_empty() {
        let b = SamplingBuffers::new();
        assert!(b.topk_cumsum_mats.is_empty());
        assert!(b.topk_shift_bufs.is_empty());
        assert!(b.topk_shift_idxs.is_empty());
        assert!(b.topk_neg_vecs.is_empty());
    }

    #[test]
    fn get_topk_neg_vec_creates_and_caches() {
        let mut b = SamplingBuffers::new();
        let dev = Device::Cpu;

        let v1 = b.get_topk_neg_vec(5, &dev).unwrap();
        assert_eq!(v1.dims(), &[5]);
        // All values should be -1e9.
        let vals: Vec<f32> = v1.to_vec1().unwrap();
        for v in &vals {
            assert!((*v - (-1e9f32)).abs() < 1.0);
        }

        // Second call should return cached version.
        assert!(b.topk_neg_vecs.contains_key(&5));
        let v2 = b.get_topk_neg_vec(5, &dev).unwrap();
        assert_eq!(v2.dims(), &[5]);
    }

    #[test]
    fn get_topk_shift_idx_creates_range() {
        let mut b = SamplingBuffers::new();
        let dev = Device::Cpu;

        let idx = b.get_topk_shift_idx(5, &dev).unwrap();
        let vals: Vec<u32> = idx.to_vec1().unwrap();
        assert_eq!(vals, vec![1, 2, 3, 4]);
    }

    #[test]
    fn get_topk_shift_idx_k1_fails() {
        let mut b = SamplingBuffers::new();
        let dev = Device::Cpu;
        assert!(b.get_topk_shift_idx(1, &dev).is_err());
    }

    #[test]
    fn get_topk_shift_buf_zeros() {
        let mut b = SamplingBuffers::new();
        let dev = Device::Cpu;

        let buf = b.get_topk_shift_buf(4, &dev, DType::F32).unwrap();
        let vals: Vec<f32> = buf.to_vec1().unwrap();
        assert_eq!(vals, vec![0.0; 4]);
    }

    #[test]
    fn get_topk_cumsum_mat_upper_triangular() {
        let mut b = SamplingBuffers::new();
        let dev = Device::Cpu;

        let mat = b.get_topk_cumsum_mat(3, &dev).unwrap();
        assert_eq!(mat.dims(), &[3, 3]);
        let vals: Vec<Vec<f32>> = mat.to_vec2().unwrap();
        // Upper triangular with 1s and 0s below diagonal.
        // Row 0 (row <= col for all): [1, 1, 1]
        // Row 1 (row=1 <= col for col>=1): [0, 1, 1]
        // Row 2 (row=2 <= col for col>=2): [0, 0, 1]
        assert_eq!(vals[0], vec![1.0, 1.0, 1.0]);
        assert_eq!(vals[1], vec![0.0, 1.0, 1.0]);
        assert_eq!(vals[2], vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn cumsum_mat_cached_on_second_call() {
        let mut b = SamplingBuffers::new();
        let dev = Device::Cpu;

        let _ = b.get_topk_cumsum_mat(4, &dev).unwrap();
        assert!(b.topk_cumsum_mats.contains_key(&4));

        let mat2 = b.get_topk_cumsum_mat(4, &dev).unwrap();
        assert_eq!(mat2.dims(), &[4, 4]);
    }

    // ── apply_repeat_penalty_inplace tests ──

    #[test]
    fn repeat_penalty_empty_context_is_noop() {
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu).unwrap();
        apply_repeat_penalty_inplace(&logits, 2.0, &[]).unwrap();
        let vals: Vec<f32> = logits.to_vec1().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn repeat_penalty_reduces_positive_logits() {
        let logits = Tensor::new(&[4.0f32, 2.0, 6.0, 1.0], &Device::Cpu).unwrap();
        // Penalize tokens 0 and 2 with penalty=2.0.
        apply_repeat_penalty_inplace(&logits, 2.0, &[0, 2]).unwrap();
        let vals: Vec<f32> = logits.to_vec1().unwrap();
        // Positive logits are divided by penalty.
        assert!((vals[0] - 2.0).abs() < 0.01); // 4.0 / 2.0
        assert!((vals[1] - 2.0).abs() < 0.01); // untouched
        assert!((vals[2] - 3.0).abs() < 0.01); // 6.0 / 2.0
        assert!((vals[3] - 1.0).abs() < 0.01); // untouched
    }

    #[test]
    fn repeat_penalty_amplifies_negative_logits() {
        let logits = Tensor::new(&[-4.0f32, 2.0, -6.0], &Device::Cpu).unwrap();
        // Penalize tokens 0 and 2 with penalty=2.0.
        apply_repeat_penalty_inplace(&logits, 2.0, &[0, 2]).unwrap();
        let vals: Vec<f32> = logits.to_vec1().unwrap();
        // Negative logits are multiplied by penalty (making them more negative).
        assert!((vals[0] - (-8.0)).abs() < 0.01); // -4.0 * 2.0
        assert!((vals[1] - 2.0).abs() < 0.01); // untouched
        assert!((vals[2] - (-12.0)).abs() < 0.01); // -6.0 * 2.0
    }

    #[test]
    fn repeat_penalty_deduplicates_context() {
        let logits = Tensor::new(&[4.0f32, 2.0, 6.0], &Device::Cpu).unwrap();
        // Duplicate tokens in context should not double-penalize.
        apply_repeat_penalty_inplace(&logits, 2.0, &[0, 0, 0]).unwrap();
        let vals: Vec<f32> = logits.to_vec1().unwrap();
        assert!((vals[0] - 2.0).abs() < 0.01); // 4.0 / 2.0
    }

    #[test]
    fn repeat_penalty_no_effect_with_1() {
        let logits = Tensor::new(&[4.0f32, -2.0, 6.0], &Device::Cpu).unwrap();
        apply_repeat_penalty_inplace(&logits, 1.0, &[0, 1, 2]).unwrap();
        let vals: Vec<f32> = logits.to_vec1().unwrap();
        assert!((vals[0] - 4.0).abs() < 0.01);
        assert!((vals[1] - (-2.0)).abs() < 0.01);
        assert!((vals[2] - 6.0).abs() < 0.01);
    }
}
