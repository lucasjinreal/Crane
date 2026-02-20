use std::collections::VecDeque;

/// Scheduling decision for a single engine step.
pub struct SchedulerOutput {
    /// Sequence IDs to process this step, in order.
    pub batch: Vec<String>,
    /// Whether this step is a prefill step (the first sequence is being prefilled).
    pub is_prefill: bool,
}

/// Simple FIFO scheduler with prefill-decode interleaving.
///
/// Invariants:
///   - After a prefill step, if there are running sequences, the next
///     step is always a decode (prevents prefill-starvation under load).
///   - A prefill step processes exactly ONE new sequence (its full prompt).
///   - A decode step processes ALL running sequences (one token each).
///   - `max_running` limits how many sequences can be in decode phase
///     simultaneously (controls peak KV-cache memory).
pub struct Scheduler {
    /// Sequences waiting for prefill, FIFO.
    pub waiting: VecDeque<String>,
    /// Sequences in decode phase, FIFO.
    pub running: VecDeque<String>,
    /// Maximum concurrent decode sequences.
    pub max_running: usize,
    /// Whether the last scheduled step was a prefill.
    /// Used to interleave decode between consecutive prefills
    /// so running sequences are not starved.
    last_was_prefill: bool,
}

impl Scheduler {
    pub fn new(max_running: usize) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            max_running,
            last_was_prefill: false,
        }
    }

    /// Add a new sequence to the waiting queue.
    pub fn add(&mut self, seq_id: String) {
        self.waiting.push_back(seq_id);
    }

    /// Remove a sequence from all queues (on completion or error).
    pub fn remove(&mut self, seq_id: &str) {
        self.waiting.retain(|id| id != seq_id);
        self.running.retain(|id| id != seq_id);
    }

    /// Decide what to do next.
    ///
    /// Returns `None` if there is no work (engine should wait for new requests).
    ///
    /// Uses prefill-decode interleaving: after a prefill, if there are running
    /// sequences, always do a decode step before the next prefill. This prevents
    /// running sequences from being starved under high concurrency.
    pub fn schedule(&mut self) -> Option<SchedulerOutput> {
        // If the last step was a prefill and we have running sequences,
        // force a decode step to avoid starving running sequences.
        let force_decode = self.last_was_prefill && !self.running.is_empty();

        // Priority 1: Prefill a waiting sequence if there's capacity
        // and we're not forcing decode.
        if !force_decode && !self.waiting.is_empty() && self.running.len() < self.max_running {
            let seq_id = self.waiting.pop_front().unwrap();
            self.last_was_prefill = true;
            return Some(SchedulerOutput {
                batch: vec![seq_id],
                is_prefill: true,
            });
        }

        // Priority 2: Decode all running sequences.
        if !self.running.is_empty() {
            self.last_was_prefill = false;
            let batch: Vec<String> = self.running.iter().cloned().collect();
            return Some(SchedulerOutput {
                batch,
                is_prefill: false,
            });
        }

        // Priority 3: Nothing running but waiting has items — prefill.
        if !self.waiting.is_empty() {
            let seq_id = self.waiting.pop_front().unwrap();
            self.last_was_prefill = true;
            return Some(SchedulerOutput {
                batch: vec![seq_id],
                is_prefill: true,
            });
        }

        None // No work.
    }

    /// Reset the interleave flag (e.g. when the engine defers a prefill
    /// and manually forces a decode step).
    pub fn reset_prefill_flag(&mut self) {
        self.last_was_prefill = false;
    }

    /// Move a sequence from waiting state to running state (called after prefill).
    pub fn promote_to_running(&mut self, seq_id: String) {
        self.running.push_back(seq_id);
    }

    /// Total active sequences (waiting + running).
    #[allow(dead_code)]
    pub fn active_count(&self) -> usize {
        self.waiting.len() + self.running.len()
    }

    /// Whether there is any work pending.
    #[allow(dead_code)]
    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_scheduler_is_empty() {
        let mut s = Scheduler::new(4);
        assert_eq!(s.active_count(), 0);
        assert!(!s.has_work());
        assert!(s.schedule().is_none());
    }

    #[test]
    fn add_puts_into_waiting() {
        let mut s = Scheduler::new(4);
        s.add("req-1".into());
        assert_eq!(s.waiting.len(), 1);
        assert_eq!(s.running.len(), 0);
        assert_eq!(s.active_count(), 1);
        assert!(s.has_work());
    }

    #[test]
    fn schedule_prefers_prefill_over_decode() {
        let mut s = Scheduler::new(4);
        // req-1 is already running (via promote).
        s.promote_to_running("req-1".into());
        // req-2 is waiting.
        s.add("req-2".into());

        let out = s.schedule().unwrap();
        // Should prefill req-2 first (priority 1: waiting has items and capacity available).
        assert!(out.is_prefill);
        assert_eq!(out.batch, vec!["req-2".to_string()]);
    }

    #[test]
    fn schedule_prefill_returns_single_sequence() {
        let mut s = Scheduler::new(4);
        s.add("req-1".into());
        s.add("req-2".into());

        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        // Only one sequence per prefill step.
        assert_eq!(out.batch.len(), 1);
        assert_eq!(out.batch[0], "req-1");
    }

    #[test]
    fn schedule_decode_returns_all_running() {
        let mut s = Scheduler::new(4);
        s.promote_to_running("req-1".into());
        s.promote_to_running("req-2".into());
        s.promote_to_running("req-3".into());

        let out = s.schedule().unwrap();
        assert!(!out.is_prefill);
        assert_eq!(out.batch.len(), 3);
    }

    #[test]
    fn schedule_respects_max_running() {
        let mut s = Scheduler::new(2);
        s.promote_to_running("req-1".into());
        s.promote_to_running("req-2".into());
        s.add("req-3".into()); // waiting, but at capacity

        let out = s.schedule().unwrap();
        // Can't prefill because running is at max_running.
        assert!(!out.is_prefill);
        assert_eq!(out.batch.len(), 2);
    }

    #[test]
    fn remove_from_waiting() {
        let mut s = Scheduler::new(4);
        s.add("req-1".into());
        s.add("req-2".into());
        s.remove("req-1");
        assert_eq!(s.waiting.len(), 1);
        assert_eq!(s.waiting[0], "req-2");
    }

    #[test]
    fn remove_from_running() {
        let mut s = Scheduler::new(4);
        s.promote_to_running("req-1".into());
        s.promote_to_running("req-2".into());
        s.remove("req-1");
        assert_eq!(s.running.len(), 1);
        assert_eq!(s.running[0], "req-2");
    }

    #[test]
    fn remove_nonexistent_is_no_op() {
        let mut s = Scheduler::new(4);
        s.add("req-1".into());
        s.remove("req-999"); // doesn't exist
        assert_eq!(s.active_count(), 1);
    }

    #[test]
    fn promote_to_running_adds_to_running_queue() {
        let mut s = Scheduler::new(4);
        s.promote_to_running("req-1".into());
        assert_eq!(s.running.len(), 1);
        assert_eq!(s.running[0], "req-1");
    }

    #[test]
    fn fifo_order_maintained() {
        let mut s = Scheduler::new(4);
        s.add("a".into());
        s.add("b".into());
        s.add("c".into());

        // First schedule should pick "a" (FIFO).
        let out = s.schedule().unwrap();
        assert_eq!(out.batch[0], "a");

        // Next should pick "b".
        let out = s.schedule().unwrap();
        assert_eq!(out.batch[0], "b");
    }

    #[test]
    fn schedule_none_when_empty() {
        let mut s = Scheduler::new(4);
        assert!(s.schedule().is_none());
    }

    #[test]
    fn full_lifecycle() {
        let mut s = Scheduler::new(2);

        // Add 3 requests.
        s.add("r1".into());
        s.add("r2".into());
        s.add("r3".into());
        assert_eq!(s.active_count(), 3);

        // Step 1: Prefill r1 (nothing running, so prefill is fine).
        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        assert_eq!(out.batch[0], "r1");
        s.promote_to_running("r1".into());

        // Step 2: Interleave — last was prefill, running=[r1] → force decode.
        let out = s.schedule().unwrap();
        assert!(!out.is_prefill);
        assert_eq!(out.batch, vec!["r1"]);

        // Step 3: Now prefill r2 (interleave flag reset).
        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        assert_eq!(out.batch[0], "r2");
        s.promote_to_running("r2".into());

        // Step 4: Interleave — force decode [r1, r2].
        let out = s.schedule().unwrap();
        assert!(!out.is_prefill);
        assert_eq!(out.batch.len(), 2);

        // Step 5: At max_running=2 with r3 waiting → can't prefill, decode.
        let out = s.schedule().unwrap();
        assert!(!out.is_prefill);
        assert_eq!(out.batch.len(), 2);

        // Finish r1, now room for r3.
        s.remove("r1");
        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        assert_eq!(out.batch[0], "r3");
    }

    #[test]
    fn interleave_prefill_decode() {
        let mut s = Scheduler::new(8);
        // Simulate 4 requests arriving at once.
        s.add("a".into());
        s.add("b".into());
        s.add("c".into());
        s.add("d".into());

        // Step 1: prefill "a" (nothing running).
        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        assert_eq!(out.batch[0], "a");
        s.promote_to_running("a".into());

        // Step 2: forced decode [a] (last was prefill).
        let out = s.schedule().unwrap();
        assert!(!out.is_prefill);
        assert_eq!(out.batch, vec!["a"]);

        // Step 3: prefill "b".
        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        assert_eq!(out.batch[0], "b");
        s.promote_to_running("b".into());

        // Step 4: forced decode [a, b].
        let out = s.schedule().unwrap();
        assert!(!out.is_prefill);
        assert_eq!(out.batch.len(), 2);

        // Step 5: prefill "c".
        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        assert_eq!(out.batch[0], "c");
        s.promote_to_running("c".into());

        // Step 6: forced decode [a, b, c].
        let out = s.schedule().unwrap();
        assert!(!out.is_prefill);
        assert_eq!(out.batch.len(), 3);

        // Step 7: prefill "d".
        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        assert_eq!(out.batch[0], "d");
        s.promote_to_running("d".into());

        // Step 8: forced decode [a, b, c, d].
        let out = s.schedule().unwrap();
        assert!(!out.is_prefill);
        assert_eq!(out.batch.len(), 4);

        // Step 9: nothing waiting, running has items → decode.
        let out = s.schedule().unwrap();
        assert!(!out.is_prefill);
        assert_eq!(out.batch.len(), 4);
    }

    #[test]
    fn reset_prefill_flag_allows_immediate_retry() {
        let mut s = Scheduler::new(4);
        s.add("r1".into());
        s.add("r2".into());

        // Prefill r1.
        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        s.promote_to_running("r1".into());

        // Normally would force decode, but reset flag simulates
        // budget-gate deferring + manual decode.
        s.reset_prefill_flag();

        // Now prefill r2 immediately (no forced decode).
        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        assert_eq!(out.batch[0], "r2");
    }
}
