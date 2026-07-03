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
    #[must_use]
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
        if self.running.len() < max
            && let Some(seq_id) = self.waiting.pop_front()
        {
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
        if let Some(seq_id) = self.waiting.pop_front() {
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
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.waiting.len() + self.running.len()
    }

    /// Whether there is any work pending.
    #[allow(dead_code)]
    #[must_use]
    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }
}
