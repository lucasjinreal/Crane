use std::collections::VecDeque;

/// Scheduling decision for a single engine step.
pub struct SchedulerOutput {
    /// Sequence IDs to process this step, in order.
    pub batch: Vec<String>,
    /// Whether this step is a prefill step (the first sequence is being prefilled).
    pub is_prefill: bool,
}

/// Simple FIFO scheduler with prefill-priority batching.
///
/// Invariants:
///   - Prefills are prioritized: waiting sequences are prefilled as fast as
///     possible (one at a time) to build up the decode batch quickly. This
///     maximizes GPU utilization since larger batch decodes are more efficient.
///   - Once `running` reaches `max_running` (or `effective_max_running`),
///     no more prefills are admitted — the scheduler only decodes.
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
    /// Dynamic cap on running sequences, set after eviction.
    ///
    /// When eviction occurs, the current running count (post-eviction) is
    /// stored here to prevent immediately re-admitting sequences that would
    /// exceed the KV budget again. Reset to `None` when a sequence finishes
    /// naturally, allowing the system to try admitting more.
    pub effective_max_running: Option<usize>,
}

impl Scheduler {
    pub fn new(max_running: usize) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            max_running,
            effective_max_running: None,
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
    /// Prioritizes prefilling waiting sequences up to `max_running` (or `effective_max_running`)
    /// to build up the batch size for efficient decoding.
    pub fn schedule(&mut self) -> Option<SchedulerOutput> {
        // Priority 1: Prefill a waiting sequence if there's capacity.
        let max = self.effective_max_running.unwrap_or(self.max_running);
        if !self.waiting.is_empty() && self.running.len() < max {
            let seq_id = self.waiting.pop_front().unwrap();
            return Some(SchedulerOutput {
                batch: vec![seq_id],
                is_prefill: true,
            });
        }

        // Priority 2: Decode all running sequences.
        if !self.running.is_empty() {
            let batch: Vec<String> = self.running.iter().cloned().collect();
            return Some(SchedulerOutput {
                batch,
                is_prefill: false,
            });
        }

        // Priority 3: Nothing running but waiting has items — prefill.
        // (This happens if max is 0, which shouldn't normally happen, but just in case).
        if !self.waiting.is_empty() {
            let seq_id = self.waiting.pop_front().unwrap();
            return Some(SchedulerOutput {
                batch: vec![seq_id],
                is_prefill: true,
            });
        }

        None // No work.
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

        // Step 1: Prefill r1 (nothing running, capacity available).
        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        assert_eq!(out.batch[0], "r1");
        s.promote_to_running("r1".into());

        // Step 2: Prefill r2 (running=1 < max=2, capacity available).
        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        assert_eq!(out.batch[0], "r2");
        s.promote_to_running("r2".into());

        // Step 3: At max_running=2 with r3 waiting → can't prefill, decode.
        let out = s.schedule().unwrap();
        assert!(!out.is_prefill);
        assert_eq!(out.batch.len(), 2);

        // Step 4: Still at capacity → decode.
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
    fn consecutive_prefills_then_decode() {
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

        // Step 2: prefill "b" (capacity available, no interleaving).
        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        assert_eq!(out.batch[0], "b");
        s.promote_to_running("b".into());

        // Step 3: prefill "c".
        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        assert_eq!(out.batch[0], "c");
        s.promote_to_running("c".into());

        // Step 4: prefill "d".
        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        assert_eq!(out.batch[0], "d");
        s.promote_to_running("d".into());

        // Step 5: nothing waiting, running has items → decode.
        let out = s.schedule().unwrap();
        assert!(!out.is_prefill);
        assert_eq!(out.batch.len(), 4);
    }

    #[test]
    fn effective_max_running_prevents_prefill() {
        let mut s = Scheduler::new(8);
        s.promote_to_running("r1".into());
        s.promote_to_running("r2".into());
        s.promote_to_running("r3".into());
        s.add("r4".into()); // waiting

        // Without cap: would prefill r4 (3 < 8).
        // With cap set to 3: should decode instead.
        s.effective_max_running = Some(3);

        let out = s.schedule().unwrap();
        assert!(!out.is_prefill, "Should decode, not prefill (capped at 3)");
        assert_eq!(out.batch.len(), 3);

        // Remove one running → running=2 < cap=3 → now can prefill.
        s.remove("r1");
        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        assert_eq!(out.batch[0], "r4");
    }

    #[test]
    fn effective_max_running_reset_allows_admission() {
        let mut s = Scheduler::new(8);
        s.promote_to_running("r1".into());
        s.promote_to_running("r2".into());
        s.add("r3".into());

        // Cap at 2 → can't admit r3.
        s.effective_max_running = Some(2);
        let out = s.schedule().unwrap();
        assert!(!out.is_prefill);

        // Lift cap → can now admit r3.
        s.effective_max_running = None;
        let out = s.schedule().unwrap();
        assert!(out.is_prefill);
        assert_eq!(out.batch[0], "r3");
    }
}
