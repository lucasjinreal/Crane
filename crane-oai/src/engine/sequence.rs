use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use tokio::sync::mpsc;

/// Per-request lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SequenceStatus {
    /// Queued, waiting for prefill.
    Waiting,
    /// Actively decoding (KV cache allocated).
    Running,
    /// Generation complete.
    Finished,
}

/// A single in-flight generation request managed by the engine.
#[allow(dead_code)]
pub struct Sequence {
    // ── identity ──
    pub id: String,
    pub status: SequenceStatus,

    // ── token state ──
    /// Full token list: prompt ++ generated.
    pub tokens: Vec<u32>,
    /// Length of the original prompt (tokens before generation started).
    pub prompt_len: usize,

    // ── KV cache (one entry per transformer layer) ──
    /// Saved KV caches when this sequence is not the one loaded in the model.
    /// Each element is `(K, V)` for a layer, or `None` for fresh layers.
    pub kv_caches: Vec<Option<(Tensor, Tensor)>>,

    // ── sampling ──
    pub logits_processor: LogitsProcessor,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub max_tokens: usize,
    pub eos_token_id: Vec<u32>,
    pub repetition_penalty: f32,
    pub repeat_last_n: usize,

    // ── response channel ──
    /// Sends `EngineResponse` chunks back to the API handler.
    pub response_tx: mpsc::UnboundedSender<super::EngineResponse>,
}

impl Sequence {
    /// Number of tokens generated so far (excluding prompt).
    pub fn num_generated(&self) -> usize {
        self.tokens.len().saturating_sub(self.prompt_len)
    }

    /// Whether generation should stop.
    pub fn should_stop(&self) -> bool {
        if self.num_generated() >= self.max_tokens {
            return true;
        }
        if let Some(&last) = self.tokens.last() {
            if self.eos_token_id.contains(&last) {
                return true;
            }
        }
        false
    }

    /// The KV cache covers tokens `0..start_pos` when we do the next forward.
    /// For a fresh sequence, `start_pos = 0`.
    /// After prefill of N prompt tokens, `start_pos = N`.
    /// During decode, `start_pos = tokens.len() - 1` (everything except the latest token).
    pub fn start_pos(&self) -> usize {
        // After prefill the kv_caches cover prompt_len tokens.
        // During decode each step adds one token, so the cache covers
        // tokens.len() - 1 positions (the new token hasn't been cached yet).
        if self.status == SequenceStatus::Waiting {
            0
        } else {
            self.tokens.len().saturating_sub(1)
        }
    }

    /// Tokens to feed into the next forward step.
    pub fn next_input_ids(&self) -> &[u32] {
        if self.status == SequenceStatus::Waiting {
            // Prefill: feed all prompt tokens.
            &self.tokens[..self.prompt_len]
        } else {
            // Decode: feed only the last generated token.
            &self.tokens[self.tokens.len() - 1..]
        }
    }

    /// Finish reason string for the OpenAI response.
    pub fn finish_reason(&self) -> &'static str {
        if let Some(&last) = self.tokens.last() {
            if self.eos_token_id.contains(&last) {
                return "stop";
            }
        }
        "length"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_transformers::generation::LogitsProcessor;

    /// Helper: build a minimal Sequence for testing.
    fn make_seq(
        prompt: &[u32],
        generated: &[u32],
        max_tokens: usize,
        eos_token_id: u32,
        status: SequenceStatus,
    ) -> Sequence {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut tokens = prompt.to_vec();
        tokens.extend_from_slice(generated);
        Sequence {
            id: "test-seq".into(),
            status,
            tokens,
            prompt_len: prompt.len(),
            kv_caches: vec![],
            logits_processor: LogitsProcessor::new(42, Some(0.8), Some(0.95)),
            temperature: Some(0.8),
            top_p: Some(0.95),
            top_k: Some(40),
            max_tokens,
            eos_token_id: vec![eos_token_id],
            repetition_penalty: 1.0,
            repeat_last_n: 64,
            response_tx: tx,
        }
    }

    #[test]
    fn num_generated_no_generation() {
        let seq = make_seq(&[1, 2, 3], &[], 10, 0, SequenceStatus::Waiting);
        assert_eq!(seq.num_generated(), 0);
    }

    #[test]
    fn num_generated_with_tokens() {
        let seq = make_seq(&[1, 2, 3], &[10, 11, 12], 10, 0, SequenceStatus::Running);
        assert_eq!(seq.num_generated(), 3);
    }

    #[test]
    fn should_stop_at_max_tokens() {
        let seq = make_seq(&[1, 2], &[10, 11], 2, 999, SequenceStatus::Running);
        assert!(seq.should_stop());
    }

    #[test]
    fn should_stop_on_eos() {
        let seq = make_seq(&[1, 2], &[10, 2], 100, 2, SequenceStatus::Running);
        assert!(seq.should_stop());
    }

    #[test]
    fn should_not_stop_mid_generation() {
        let seq = make_seq(&[1, 2], &[10], 100, 999, SequenceStatus::Running);
        assert!(!seq.should_stop());
    }

    #[test]
    fn should_stop_empty_with_max_zero() {
        let seq = make_seq(&[1], &[], 0, 999, SequenceStatus::Running);
        assert!(seq.should_stop());
    }

    #[test]
    fn start_pos_waiting_is_zero() {
        let seq = make_seq(&[1, 2, 3], &[], 10, 0, SequenceStatus::Waiting);
        assert_eq!(seq.start_pos(), 0);
    }

    #[test]
    fn start_pos_running_after_prefill() {
        // Prompt of 3 tokens, 1 generated => total 4 tokens, start_pos = 3
        let seq = make_seq(&[1, 2, 3], &[10], 10, 0, SequenceStatus::Running);
        assert_eq!(seq.start_pos(), 3); // tokens.len() - 1 = 4 - 1
    }

    #[test]
    fn start_pos_running_no_generated() {
        // Just moved to Running but no token generated yet
        let seq = make_seq(&[1, 2, 3], &[], 10, 0, SequenceStatus::Running);
        assert_eq!(seq.start_pos(), 2); // tokens.len() - 1 = 3 - 1
    }

    #[test]
    fn next_input_ids_waiting_returns_prompt() {
        let seq = make_seq(&[1, 2, 3], &[], 10, 0, SequenceStatus::Waiting);
        assert_eq!(seq.next_input_ids(), &[1, 2, 3]);
    }

    #[test]
    fn next_input_ids_running_returns_last_token() {
        let seq = make_seq(&[1, 2, 3], &[10, 11], 10, 0, SequenceStatus::Running);
        assert_eq!(seq.next_input_ids(), &[11]);
    }

    #[test]
    fn finish_reason_eos() {
        let seq = make_seq(&[1, 2], &[10, 42], 100, 42, SequenceStatus::Running);
        assert_eq!(seq.finish_reason(), "stop");
    }

    #[test]
    fn finish_reason_length() {
        let seq = make_seq(&[1, 2], &[10, 11], 100, 999, SequenceStatus::Running);
        assert_eq!(seq.finish_reason(), "length");
    }

    #[test]
    fn finish_reason_prompt_eos() {
        // Prompt ends with EOS but no generation — still "stop"
        let seq = make_seq(&[1, 2, 42], &[], 100, 42, SequenceStatus::Waiting);
        assert_eq!(seq.finish_reason(), "stop");
    }

    #[test]
    fn sequence_status_enum_eq() {
        assert_eq!(SequenceStatus::Waiting, SequenceStatus::Waiting);
        assert_ne!(SequenceStatus::Waiting, SequenceStatus::Running);
        assert_ne!(SequenceStatus::Running, SequenceStatus::Finished);
    }
}
