//! Engine statistics — lock-free counters shared with API handlers.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Lock-free engine statistics counters.
pub struct EngineStats {
    pub total_requests: AtomicU64,
    pub completed_requests: AtomicU64,
    pub cancelled_requests: AtomicU64,
    pub failed_requests: AtomicU64,
    pub total_prompt_tokens: AtomicU64,
    pub total_completion_tokens: AtomicU64,
    pub total_prefill_time_us: AtomicU64,
    pub total_decode_steps: AtomicU64,
    pub total_decode_time_us: AtomicU64,
    pub total_kv_swap_count: AtomicU64,
    pub active_sequences: AtomicU64,
    pub waiting_sequences: AtomicU64,
    /// Cumulative time-to-first-token across all completed requests, in
    /// microseconds. Paired with `ttft_count` to compute `avg_ttft_ms`.
    pub total_ttft_us: AtomicU64,
    /// Number of requests that have contributed a TTFT sample.
    pub ttft_count: AtomicU64,
    /// Set once a fatal, device-context-poisoning error occurs (e.g. an
    /// illegal GPU memory access). Unset means the engine is healthy.
    /// First occurrence wins; later errors don't overwrite it.
    pub fatal_error: OnceLock<String>,
}

impl Default for EngineStats {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineStats {
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            completed_requests: AtomicU64::new(0),
            cancelled_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            total_completion_tokens: AtomicU64::new(0),
            total_prefill_time_us: AtomicU64::new(0),
            total_decode_steps: AtomicU64::new(0),
            total_decode_time_us: AtomicU64::new(0),
            total_kv_swap_count: AtomicU64::new(0),
            active_sequences: AtomicU64::new(0),
            waiting_sequences: AtomicU64::new(0),
            total_ttft_us: AtomicU64::new(0),
            ttft_count: AtomicU64::new(0),
            fatal_error: OnceLock::new(),
        }
    }

    /// Record a fatal, device-context-poisoning error. Only the first
    /// occurrence is kept; subsequent calls are no-ops.
    pub fn set_fatal_error(&self, msg: &str) {
        if self.fatal_error.get().is_none() {
            let _ = self.fatal_error.set(msg.to_string());
        }
    }

    /// Returns the recorded fatal error message, if any.
    #[must_use]
    pub fn get_fatal_error(&self) -> Option<String> {
        self.fatal_error.get().cloned()
    }

    /// Snapshot for JSON serialization.
    pub fn snapshot(&self) -> StatsSnapshot {
        let total_decode = self.total_decode_steps.load(Ordering::Relaxed);
        let total_decode_us = self.total_decode_time_us.load(Ordering::Relaxed);
        #[allow(clippy::cast_precision_loss)]
        let avg_decode_tok_s = if total_decode_us > 0 {
            (total_decode as f64) / (total_decode_us as f64 / 1_000_000.0)
        } else {
            0.0
        };
        let total_prefill_us = self.total_prefill_time_us.load(Ordering::Relaxed);
        let total_prompt = self.total_prompt_tokens.load(Ordering::Relaxed);
        #[allow(clippy::cast_precision_loss)]
        let avg_prefill_tok_s = if total_prefill_us > 0 {
            (total_prompt as f64) / (total_prefill_us as f64 / 1_000_000.0)
        } else {
            0.0
        };
        let ttft_count = self.ttft_count.load(Ordering::Relaxed);
        let total_ttft_us = self.total_ttft_us.load(Ordering::Relaxed);
        #[allow(clippy::cast_precision_loss)]
        let avg_ttft_ms = if ttft_count > 0 {
            (total_ttft_us as f64 / 1000.0) / ttft_count as f64
        } else {
            0.0
        };
        StatsSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),
            completed_requests: self.completed_requests.load(Ordering::Relaxed),
            cancelled_requests: self.cancelled_requests.load(Ordering::Relaxed),
            failed_requests: self.failed_requests.load(Ordering::Relaxed),
            total_prompt_tokens: total_prompt,
            total_completion_tokens: self.total_completion_tokens.load(Ordering::Relaxed),
            active_sequences: self.active_sequences.load(Ordering::Relaxed),
            waiting_sequences: self.waiting_sequences.load(Ordering::Relaxed),
            total_kv_swaps: self.total_kv_swap_count.load(Ordering::Relaxed),
            avg_decode_tokens_per_sec: avg_decode_tok_s,
            avg_prefill_tokens_per_sec: avg_prefill_tok_s,
            avg_ttft_ms,
            fatal_error: self.get_fatal_error(),
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct StatsSnapshot {
    pub total_requests: u64,
    pub completed_requests: u64,
    pub cancelled_requests: u64,
    pub failed_requests: u64,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub active_sequences: u64,
    pub waiting_sequences: u64,
    pub total_kv_swaps: u64,
    pub avg_decode_tokens_per_sec: f64,
    pub avg_prefill_tokens_per_sec: f64,
    /// Average time-to-first-token across completed requests, in
    /// milliseconds. `0.0` until at least one request has completed.
    pub avg_ttft_ms: f64,
    /// Recorded fatal GPU error message, if the engine has hit one. Omitted
    /// from the JSON response entirely when the engine is healthy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fatal_error: Option<String>,
}

/// Narrow allowlist of error-message signatures that indicate the GPU
/// device context itself is poisoned (not just a single request failing).
/// Deliberately conservative: a per-request OOM or bad-input error must
/// not trip this, since that would incorrectly fail the whole engine.
#[must_use]
pub(crate) fn is_fatal_gpu_error(msg: &str) -> bool {
    const FATAL_SIGNATURES: &[&str] = &[
        "rocrand",
        "illegal memory access",
        "launch failed",
        "launch failure",
        "an illegal instruction",
        "device-side assert",
        "context is destroyed",
    ];
    let lower = msg.to_lowercase();
    FATAL_SIGNATURES.iter().any(|sig| lower.contains(sig))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn new_stats_are_zero() {
        let s = EngineStats::new();
        assert_eq!(s.total_requests.load(Ordering::Relaxed), 0);
        assert_eq!(s.completed_requests.load(Ordering::Relaxed), 0);
        assert_eq!(s.cancelled_requests.load(Ordering::Relaxed), 0);
        assert_eq!(s.failed_requests.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_prompt_tokens.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_completion_tokens.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_prefill_time_us.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_decode_steps.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_decode_time_us.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_kv_swap_count.load(Ordering::Relaxed), 0);
        assert_eq!(s.active_sequences.load(Ordering::Relaxed), 0);
        assert_eq!(s.waiting_sequences.load(Ordering::Relaxed), 0);
        assert_eq!(s.total_ttft_us.load(Ordering::Relaxed), 0);
        assert_eq!(s.ttft_count.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn snapshot_copies_all_counters() {
        let s = EngineStats::new();
        s.total_requests.store(10, Ordering::Relaxed);
        s.completed_requests.store(7, Ordering::Relaxed);
        s.cancelled_requests.store(2, Ordering::Relaxed);
        s.failed_requests.store(1, Ordering::Relaxed);
        s.total_prompt_tokens.store(500, Ordering::Relaxed);
        s.total_completion_tokens.store(1000, Ordering::Relaxed);
        s.total_kv_swap_count.store(3, Ordering::Relaxed);
        s.active_sequences.store(4, Ordering::Relaxed);
        s.waiting_sequences.store(2, Ordering::Relaxed);

        let snap = s.snapshot();
        assert_eq!(snap.total_requests, 10);
        assert_eq!(snap.completed_requests, 7);
        assert_eq!(snap.cancelled_requests, 2);
        assert_eq!(snap.failed_requests, 1);
        assert_eq!(snap.total_prompt_tokens, 500);
        assert_eq!(snap.total_completion_tokens, 1000);
        assert_eq!(snap.total_kv_swaps, 3);
        assert_eq!(snap.active_sequences, 4);
        assert_eq!(snap.waiting_sequences, 2);
    }

    #[test]
    fn snapshot_decode_rate_calculation() {
        let s = EngineStats::new();
        // 100 decode steps in 1 second (1_000_000 μs)
        s.total_decode_steps.store(100, Ordering::Relaxed);
        s.total_decode_time_us.store(1_000_000, Ordering::Relaxed);

        let snap = s.snapshot();
        assert!((snap.avg_decode_tokens_per_sec - 100.0).abs() < 0.01);
    }

    #[test]
    fn snapshot_prefill_rate_calculation() {
        let s = EngineStats::new();
        // 500 prompt tokens prefilled in 0.5 seconds (500_000 μs)
        s.total_prompt_tokens.store(500, Ordering::Relaxed);
        s.total_prefill_time_us.store(500_000, Ordering::Relaxed);

        let snap = s.snapshot();
        assert!((snap.avg_prefill_tokens_per_sec - 1000.0).abs() < 0.01);
    }

    #[test]
    fn snapshot_zero_time_gives_zero_rate() {
        let s = EngineStats::new();
        s.total_decode_steps.store(50, Ordering::Relaxed);
        // time stays 0
        let snap = s.snapshot();
        // The zero-time branch returns the literal 0.0, so exact comparison is correct here.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(snap.avg_decode_tokens_per_sec, 0.0);
            assert_eq!(snap.avg_prefill_tokens_per_sec, 0.0);
        }
    }

    #[test]
    fn snapshot_avg_ttft_ms_calculation() {
        let s = EngineStats::new();
        // 3 requests totaling 900_000 microseconds => avg 300ms.
        s.total_ttft_us.store(900_000, Ordering::Relaxed);
        s.ttft_count.store(3, Ordering::Relaxed);

        let snap = s.snapshot();
        assert!((snap.avg_ttft_ms - 300.0).abs() < 0.01);
    }

    #[test]
    fn snapshot_avg_ttft_ms_zero_when_no_samples() {
        let s = EngineStats::new();
        let snap = s.snapshot();
        // The zero-count branch returns the literal 0.0, so exact comparison is correct here.
        #[allow(clippy::float_cmp)]
        {
            assert_eq!(snap.avg_ttft_ms, 0.0);
        }
    }

    #[test]
    fn snapshot_serializes_to_json() {
        let s = EngineStats::new();
        s.total_requests.store(5, Ordering::Relaxed);
        let snap = s.snapshot();
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("\"total_requests\":5"));
        assert!(json.contains("avg_decode_tokens_per_sec"));
        assert!(json.contains("avg_prefill_tokens_per_sec"));
    }

    #[test]
    fn atomic_fetch_add_works() {
        let s = EngineStats::new();
        s.total_requests.fetch_add(1, Ordering::Relaxed);
        s.total_requests.fetch_add(1, Ordering::Relaxed);
        s.total_requests.fetch_add(1, Ordering::Relaxed);
        assert_eq!(s.snapshot().total_requests, 3);
    }

    #[test]
    fn fatal_error_is_none_when_healthy() {
        let s = EngineStats::new();
        assert_eq!(s.get_fatal_error(), None);
    }

    #[test]
    fn set_fatal_error_records_message() {
        let s = EngineStats::new();
        s.set_fatal_error(
            "Batched decode failed: DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, \
             \"an illegal memory access was encountered\")",
        );
        assert_eq!(
            s.get_fatal_error(),
            Some(
                "Batched decode failed: DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, \
                 \"an illegal memory access was encountered\")"
                    .to_string()
            )
        );
    }

    #[test]
    fn set_fatal_error_first_occurrence_wins() {
        let s = EngineStats::new();
        s.set_fatal_error("first fatal error");
        s.set_fatal_error("second fatal error");
        assert_eq!(s.get_fatal_error(), Some("first fatal error".to_string()));
    }

    #[test]
    fn snapshot_includes_fatal_error() {
        let s = EngineStats::new();
        s.set_fatal_error("rocrand generate_uniform failed with status 107");
        let snap = s.snapshot();
        assert_eq!(
            snap.fatal_error,
            Some("rocrand generate_uniform failed with status 107".to_string())
        );
    }

    #[test]
    fn snapshot_json_omits_fatal_error_when_healthy() {
        let s = EngineStats::new();
        let snap = s.snapshot();
        let json = serde_json::to_string(&snap).unwrap();
        assert!(!json.contains("fatal_error"));
    }

    #[test]
    fn snapshot_json_includes_fatal_error_when_set() {
        let s = EngineStats::new();
        s.set_fatal_error("CUDA error: launch failed");
        let snap = s.snapshot();
        let json = serde_json::to_string(&snap).unwrap();
        assert!(json.contains("\"fatal_error\":\"CUDA error: launch failed\""));
    }

    #[test]
    fn is_fatal_gpu_error_matches_known_signatures() {
        assert!(is_fatal_gpu_error(
            "rocrand generate_uniform failed with status 107"
        ));
        assert!(is_fatal_gpu_error(
            "Batched decode failed: DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, \
             \"an illegal memory access was encountered\")"
        ));
        assert!(is_fatal_gpu_error("an illegal instruction was encountered"));
        assert!(is_fatal_gpu_error(
            "fused_rms_norm launch failed: some hip error"
        ));
        assert!(is_fatal_gpu_error(
            "Batched decode failed: DriverError(CUDA_ERROR_LAUNCH_FAILED, \
             \"unspecified launch failure\")"
        ));
        assert!(is_fatal_gpu_error("device-side assert triggered"));
        assert!(is_fatal_gpu_error(
            "Batched decode failed: DriverError(CUDA_ERROR_CONTEXT_IS_DESTROYED, \
             \"context is destroyed\")"
        ));
    }

    #[test]
    fn is_fatal_gpu_error_is_case_insensitive() {
        assert!(is_fatal_gpu_error("ROCRAND status failure"));
        assert!(is_fatal_gpu_error("Illegal Memory Access detected"));
    }

    #[test]
    fn is_fatal_gpu_error_rejects_ordinary_errors() {
        assert!(!is_fatal_gpu_error("Sampling failed: invalid probability"));
        assert!(!is_fatal_gpu_error("out of memory"));
        assert!(!is_fatal_gpu_error("Empty prefill"));
        assert!(!is_fatal_gpu_error("invalid argument"));
        assert!(!is_fatal_gpu_error(
            "KV cache budget exceeded during decode"
        ));
    }
}
