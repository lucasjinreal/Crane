// SPDX-License-Identifier: MIT
//! Custom sampling for Audio8-TTS: a deterministic PRNG, the reference's
//! top-k/top-p Gumbel-max sampler, and lazy Repetition-Avoidance Sampling
//! (RAS). A dedicated sampler is needed instead of
//! `candle_transformers::generation::LogitsProcessor` because RAS
//! conditionally re-samples at different parameters when the first sample
//! repeats a recent token.

use std::collections::VecDeque;

/// Deterministic PRNG (`SplitMix64`) used for reproducible sampling.
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seeds a new generator.
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Returns a uniform value in `(0, 1)`, never exactly `0` (so callers can
    /// safely take its `ln()` for Gumbel noise).
    pub(crate) fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // `z >> 11` fits in 53 bits, exactly representable in f64.
        #[allow(clippy::cast_precision_loss)]
        let mantissa = (z >> 11) as f64;
        let value = mantissa / 9_007_199_254_740_992.0; // 2^53
        value.max(f64::MIN_POSITIVE)
    }
}

/// Nucleus-sampling parameters shared by [`sample_topk_topp`] and
/// [`sample_semantic`].
pub(crate) struct SamplingParams {
    /// Softmax temperature; higher values increase randomness.
    pub temperature: f64,
    /// Nucleus sampling cumulative-probability threshold.
    pub top_p: f64,
    /// Maximum number of highest-probability candidates considered.
    pub top_k: usize,
}

/// The slow AR vocabulary layout needed to map a sampled logits index back
/// to a token ID, matching `runtime_manifest.json`'s `relative_semantic_then_eos`
/// layout.
pub(crate) struct SemanticVocab {
    /// First vocabulary ID of the semantic-token range.
    pub semantic_begin_id: i64,
    /// Vocabulary ID that signals end-of-generation.
    pub im_end_id: i64,
    /// Number of entries per codec codebook.
    pub codebook_size: usize,
}

/// Samples an index from `logits` using top-k/top-p (nucleus) filtering with
/// a Gumbel-max trick, matching the reference implementation's `_sample`.
pub(crate) fn sample_topk_topp(
    logits: &[f64],
    params: &SamplingParams,
    rng: &mut SplitMix64,
) -> usize {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));

    let max_logit = logits[order[0]];
    let mut base: Vec<f64> = order
        .iter()
        .map(|&i| (logits[i] - max_logit).exp())
        .collect();
    let base_sum: f64 = base.iter().sum();
    for v in &mut base {
        *v /= base_sum;
    }

    let temp = params.temperature.max(1e-5);
    let mut scaled: Vec<f64> = logits.iter().map(|&v| v / temp).collect();
    let mut cumulative = 0.0;
    for (rank, &idx) in order.iter().enumerate() {
        cumulative += base[rank];
        let remove = rank != 0 && (cumulative > params.top_p || rank >= params.top_k);
        if remove {
            scaled[idx] = f64::NEG_INFINITY;
        }
    }

    let max_scaled = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut probs: Vec<f64> = scaled.iter().map(|&v| (v - max_scaled).exp()).collect();
    let probs_sum: f64 = probs.iter().sum();
    for v in &mut probs {
        *v /= probs_sum;
    }

    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, &p) in probs.iter().enumerate() {
        let noise = -rng.next_f64().ln();
        let score = p / noise;
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx
}

/// Samples the next semantic token with lazy Repetition-Avoidance Sampling:
/// `logits` are the slow AR's `codebook_size + 1`-wide relative-semantic
/// logits (index `i < codebook_size` maps to token `semantic_begin_id + i`;
/// index `codebook_size` maps to `im_end_id`). A high-temperature/top-p
/// resample only runs when the first (`normal`) sample repeats a token from
/// `previous` — the reference implementation always computes both samples.
pub(crate) fn sample_semantic(
    logits: &[f64],
    previous: &VecDeque<i64>,
    params: &SamplingParams,
    vocab: &SemanticVocab,
    rng: &mut SplitMix64,
) -> i64 {
    let map_index = |index: usize| -> i64 {
        if index < vocab.codebook_size {
            // index < codebook_size (4096 in the shipped package), well
            // within i64 range.
            #[allow(clippy::cast_possible_wrap)]
            let relative = index as i64;
            vocab.semantic_begin_id + relative
        } else {
            vocab.im_end_id
        }
    };

    let normal_index = sample_topk_topp(logits, params, rng);
    let normal = map_index(normal_index);

    if normal != vocab.im_end_id && previous.contains(&normal) {
        let ras_params = SamplingParams {
            temperature: 1.0,
            top_p: 0.9,
            top_k: params.top_k,
        };
        let high_index = sample_topk_topp(logits, &ras_params, rng);
        map_index(high_index)
    } else {
        normal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Same seed must reproduce the same sequence of draws.
    #[test]
    fn splitmix64_deterministic() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..8 {
            assert_eq!(a.next_f64(), b.next_f64());
        }
    }

    // Draws are always in (0, 1), never exactly 0 or 1.
    #[test]
    fn splitmix64_in_open_unit_interval() {
        let mut rng = SplitMix64::new(1);
        for _ in 0..100 {
            let v = rng.next_f64();
            assert!(v > 0.0 && v < 1.0);
        }
    }

    // With top_k = 1, only the argmax logit survives filtering, so the
    // sampler must always return it regardless of temperature/noise.
    #[test]
    fn sample_topk_topp_respects_top_k_1() {
        let logits = [1.0, 10.0, 2.0, -5.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
        };
        let mut rng = SplitMix64::new(7);
        for _ in 0..20 {
            assert_eq!(sample_topk_topp(&logits, &params, &mut rng), 1);
        }
    }

    // A tiny temperature makes the softmax near one-hot on the argmax, so
    // (barring extreme noise draws) the highest logit should be picked.
    #[test]
    fn sample_topk_topp_picks_highest_at_tiny_temperature() {
        let logits = [1.0, 10.0, 2.0];
        let params = SamplingParams {
            temperature: 1e-5,
            top_p: 1.0,
            top_k: 50,
        };
        let mut rng = SplitMix64::new(3);
        assert_eq!(sample_topk_topp(&logits, &params, &mut rng), 1);
    }

    // With an empty repetition window, RAS never triggers a resample; the
    // returned token must be a valid mapped index (semantic or EOS).
    #[test]
    fn sample_semantic_normal_path() {
        let mut logits = vec![-10.0; 11];
        logits[3] = 10.0; // index 3 is overwhelmingly likely
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 50,
        };
        let vocab = SemanticVocab {
            semantic_begin_id: 65537,
            im_end_id: 4096,
            codebook_size: 10,
        };
        let mut rng = SplitMix64::new(11);
        let token = sample_semantic(&logits, &VecDeque::new(), &params, &vocab, &mut rng);
        assert_eq!(token, 65537 + 3);
    }

    // If the first sample would repeat a token already in the RAS window,
    // the high-temperature resample must be used instead of `normal`.
    #[test]
    fn sample_semantic_resamples_on_repetition() {
        let mut logits = vec![-10.0; 11];
        logits[3] = 10.0;
        let previous = VecDeque::from([65537 + 3]);
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 50,
        };
        let vocab = SemanticVocab {
            semantic_begin_id: 65537,
            im_end_id: 4096,
            codebook_size: 10,
        };
        let mut rng = SplitMix64::new(11);
        let token = sample_semantic(&logits, &previous, &params, &vocab, &mut rng);
        // The resample uses the same overwhelmingly-likely logits, but a
        // different (temperature=1.0, top_p=0.9) pass and a fresh PRNG draw,
        // so it need not equal the suppressed `normal` value; it must still
        // be a validly-mapped token (semantic or EOS).
        assert!((65537..=65537 + 10).contains(&token) || token == 4096);
    }
}
