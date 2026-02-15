use std::collections::VecDeque;

/// Scheduling decision for a single engine step.
pub struct SchedulerOutput {
    /// Sequence IDs to process this step, in order.
    pub batch: Vec<String>,
    /// Whether this step is a prefill step (the first sequence is being prefilled).
    pub is_prefill: bool,
}

/// Simple FIFO scheduler with prefill priority.
///
/// Invariants:
///   - Prefill has priority over decode (new requests get served first
///     so they can start streaming sooner).
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
}

impl Scheduler {
    pub fn new(max_running: usize) -> Self {
        Self {
            waiting: VecDeque::new(),
            running: VecDeque::new(),
            max_running,
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
    pub fn schedule(&mut self) -> Option<SchedulerOutput> {
        // Priority 1: Prefill a waiting sequence if there's capacity.
        if !self.waiting.is_empty() && self.running.len() < self.max_running {
            let seq_id = self.waiting.pop_front().unwrap();
            // It will be moved to `running` after prefill completes.
            return Some(SchedulerOutput {
                batch: vec![seq_id],
                is_prefill: true,
            });
        }

        // Priority 2: Decode all running sequences (round-robin, one token each).
        if !self.running.is_empty() {
            let batch: Vec<String> = self.running.iter().cloned().collect();
            return Some(SchedulerOutput {
                batch,
                is_prefill: false,
            });
        }

        // Priority 3: If we're at max capacity but have waiting sequences,
        // still do a decode step to free up a slot eventually.
        // (Already covered by the running check above.)

        None // No work.
    }

    /// Move a sequence from waiting state to running state (called after prefill).
    pub fn promote_to_running(&mut self, seq_id: String) {
        self.running.push_back(seq_id);
    }

    /// Total active sequences (waiting + running).
    pub fn active_count(&self) -> usize {
        self.waiting.len() + self.running.len()
    }

    /// Whether there is any work pending.
    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }
}
