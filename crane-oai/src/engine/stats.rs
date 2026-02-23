//! Engine statistics — lock-free counters shared with API handlers.

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
}

impl EngineStats {
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
        }
    }

    /// Snapshot for JSON serialization.
    pub fn snapshot(&self) -> StatsSnapshot {
        let total_decode = self.total_decode_steps.load(Ordering::Relaxed);
        let total_decode_us = self.total_decode_time_us.load(Ordering::Relaxed);
        let avg_decode_tok_s = if total_decode_us > 0 {
            (total_decode as f64) / (total_decode_us as f64 / 1_000_000.0)
        } else {
            0.0
        };
        let total_prefill_us = self.total_prefill_time_us.load(Ordering::Relaxed);
        let total_prompt = self.total_prompt_tokens.load(Ordering::Relaxed);
        let avg_prefill_tok_s = if total_prefill_us > 0 {
            (total_prompt as f64) / (total_prefill_us as f64 / 1_000_000.0)
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
        assert_eq!(snap.avg_decode_tokens_per_sec, 0.0);
        assert_eq!(snap.avg_prefill_tokens_per_sec, 0.0);
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
}
