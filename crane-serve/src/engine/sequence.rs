use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use tokio::sync::mpsc;

/// Compute the total GPU memory (in bytes) held by a set of KV caches.
#[must_use]
pub fn kv_cache_bytes(caches: &[Option<(Tensor, Tensor)>]) -> u64 {
    caches
        .iter()
        .filter_map(|c| c.as_ref())
        .map(|(k, v)| {
            let k_bytes = k.elem_count() as u64 * k.dtype().size_in_bytes() as u64;
            let v_bytes = v.elem_count() as u64 * v.dtype().size_in_bytes() as u64;
            k_bytes + v_bytes
        })
        .sum()
}

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
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub repeat_last_n: usize,

    // ── stop sequences ──
    /// String sequences that terminate generation when produced.
    pub stop_sequences: Vec<String>,
    /// Decoded text accumulated since the last successful `take_safe_text`
    /// call. Used both to detect stop-sequence matches and, via
    /// `take_safe_text`, to withhold text that could still extend into a
    /// stop sequence from being streamed to the client.
    pub unsent_text: String,

    // ── response channel ──
    /// Sends `EngineResponse` chunks back to the API handler.
    pub response_tx: mpsc::UnboundedSender<super::EngineResponse>,
}

impl Sequence {
    /// Number of tokens generated so far (excluding prompt).
    #[must_use]
    pub fn num_generated(&self) -> usize {
        self.tokens.len().saturating_sub(self.prompt_len)
    }

    /// Whether generation should stop.
    #[must_use]
    pub fn should_stop(&self) -> bool {
        if self.num_generated() >= self.max_tokens {
            return true;
        }
        if let Some(&last) = self.tokens.last()
            && self.eos_token_id.contains(&last)
        {
            return true;
        }
        self.stop_sequence_match().is_some()
    }

    /// The first stop sequence that `unsent_text` currently ends with, if any.
    #[must_use]
    pub fn stop_sequence_match(&self) -> Option<&str> {
        self.stop_sequences
            .iter()
            .find(|s| !s.is_empty() && self.unsent_text.ends_with(s.as_str()))
            .map(String::as_str)
    }

    /// Longest suffix of `unsent_text` that is also a prefix of some stop
    /// sequence. This much text must be withheld from streaming, since a
    /// later token could still extend it into a full stop-sequence match.
    fn stop_prefix_overlap(&self) -> usize {
        let text = self.unsent_text.as_str();
        let mut max_overlap = 0usize;
        for stop in &self.stop_sequences {
            if stop.is_empty() {
                continue;
            }
            let mut boundaries: Vec<usize> = stop.char_indices().map(|(i, _)| i).collect();
            boundaries.push(stop.len());
            for k in boundaries {
                if k == 0 || k > text.len() {
                    continue;
                }
                if text.ends_with(&stop[..k]) {
                    max_overlap = max_overlap.max(k);
                }
            }
        }
        max_overlap
    }

    /// Drains and returns the prefix of `unsent_text` that is safe to stream
    /// now, withholding any tail that could still extend into a stop
    /// sequence. Returns `None` if nothing new is safe to send.
    #[must_use]
    pub fn take_safe_text(&mut self) -> Option<String> {
        if self.stop_sequences.is_empty() {
            return (!self.unsent_text.is_empty()).then(|| std::mem::take(&mut self.unsent_text));
        }
        let safe_len = self.unsent_text.len() - self.stop_prefix_overlap();
        if safe_len == 0 {
            return None;
        }
        Some(self.unsent_text.drain(..safe_len).collect())
    }

    /// Drains and returns any text preceding a fully-matched stop sequence
    /// that hasn't been streamed yet. Only meaningful once `should_stop` has
    /// returned `true` because of a stop-sequence match.
    #[must_use]
    pub fn take_pre_stop_text(&mut self) -> Option<String> {
        let stop_len = self.stop_sequence_match()?.len();
        let pre_len = self.unsent_text.len() - stop_len;
        if pre_len == 0 {
            return None;
        }
        Some(self.unsent_text.drain(..pre_len).collect())
    }

    /// The KV cache covers tokens `0..start_pos` when we do the next forward.
    /// For a fresh sequence, `start_pos = 0`.
    /// After prefill of N prompt tokens, `start_pos = N`.
    /// During decode, `start_pos = tokens.len() - 1` (everything except the latest token).
    #[must_use]
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
    #[must_use]
    pub fn next_input_ids(&self) -> &[u32] {
        if self.status == SequenceStatus::Waiting {
            // Prefill: feed all prompt tokens.
            &self.tokens[..self.prompt_len]
        } else {
            // Decode: feed only the last generated token.
            &self.tokens[self.tokens.len() - 1..]
        }
    }

    /// Finish reason string for the `OpenAI` response.
    #[must_use]
    pub fn finish_reason(&self) -> &'static str {
        if let Some(&last) = self.tokens.last()
            && self.eos_token_id.contains(&last)
        {
            return "stop";
        }
        if self.stop_sequence_match().is_some() {
            return "stop";
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
            stop_sequences: vec![],
            unsent_text: String::new(),
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
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

    /// Helper: build a Sequence with stop sequences and accumulated unsent text.
    fn make_seq_with_stop(stop_sequences: &[&str], unsent_text: &str) -> Sequence {
        let mut seq = make_seq(&[1, 2], &[10, 11], 100, 999, SequenceStatus::Running);
        seq.stop_sequences = stop_sequences.iter().map(|s| (*s).to_string()).collect();
        seq.unsent_text = unsent_text.to_string();
        seq
    }

    #[test]
    fn should_stop_on_stop_sequence() {
        let seq = make_seq_with_stop(&["```"], "print('hi')\n```");
        assert!(seq.should_stop());
    }

    #[test]
    fn should_not_stop_partial_stop_sequence() {
        let seq = make_seq_with_stop(&["```"], "print('hi')\n``");
        assert!(!seq.should_stop());
    }

    #[test]
    fn stop_sequence_match_returns_match() {
        let seq = make_seq_with_stop(&["END", "```"], "some text```");
        assert_eq!(seq.stop_sequence_match(), Some("```"));
    }

    #[test]
    fn stop_sequence_match_none_when_no_match() {
        let seq = make_seq_with_stop(&["END", "```"], "some text");
        assert_eq!(seq.stop_sequence_match(), None);
    }

    #[test]
    fn finish_reason_stop_sequence() {
        let seq = make_seq_with_stop(&["```"], "hello```");
        assert_eq!(seq.finish_reason(), "stop");
    }

    #[test]
    fn multiple_stop_sequences_first_match_wins() {
        let seq = make_seq_with_stop(&["lo", "hello"], "hello");
        assert_eq!(seq.stop_sequence_match(), Some("lo"));
    }

    #[test]
    fn take_safe_text_returns_everything_when_no_stop_sequences() {
        let mut seq = make_seq_with_stop(&[], "hello world");
        assert_eq!(seq.take_safe_text(), Some("hello world".to_string()));
        assert_eq!(seq.unsent_text, "");
    }

    #[test]
    fn take_safe_text_withholds_stop_prefix_tail() {
        // "``" is a proper prefix of "```", so it must be withheld: a later
        // token could turn it into a full match.
        let mut seq = make_seq_with_stop(&["```"], "print('hi')\n``");
        assert_eq!(seq.take_safe_text(), Some("print('hi')\n".to_string()));
        assert_eq!(seq.unsent_text, "``");
    }

    #[test]
    fn take_safe_text_releases_withheld_tail_once_it_cannot_match() {
        let mut seq = make_seq_with_stop(&["```"], "print('hi')\n``");
        seq.take_safe_text();
        // The next token proves the withheld "``" is not becoming a stop
        // sequence — appending non-matching text should release it all.
        seq.unsent_text.push_str("x");
        assert_eq!(seq.take_safe_text(), Some("``x".to_string()));
    }

    #[test]
    fn take_safe_text_none_when_everything_withheld() {
        let mut seq = make_seq_with_stop(&["```"], "``");
        assert_eq!(seq.take_safe_text(), None);
        assert_eq!(seq.unsent_text, "``");
    }

    #[test]
    fn take_safe_text_ignores_empty_stop_sequence() {
        let mut seq = make_seq_with_stop(&[""], "hello");
        assert_eq!(seq.take_safe_text(), Some("hello".to_string()));
    }

    #[test]
    fn take_safe_text_handles_utf8_char_boundary() {
        // Stop sequence contains a multi-byte character; the withheld tail
        // must land on a valid UTF-8 boundary.
        let mut seq = make_seq_with_stop(&["→END"], "hello →");
        assert_eq!(seq.take_safe_text(), Some("hello ".to_string()));
        assert_eq!(seq.unsent_text, "→");
    }

    #[test]
    fn take_pre_stop_text_returns_text_before_match() {
        let mut seq = make_seq_with_stop(&["```"], "print('hi')\n```");
        assert_eq!(seq.take_pre_stop_text(), Some("print('hi')\n".to_string()));
        assert_eq!(seq.unsent_text, "```");
    }

    #[test]
    fn take_pre_stop_text_none_when_match_is_whole_text() {
        let mut seq = make_seq_with_stop(&["```"], "```");
        assert_eq!(seq.take_pre_stop_text(), None);
    }

    #[test]
    fn multi_token_stop_sequence_accumulation() {
        // Simulate tokens arriving one at a time until the stop sequence
        // completes across several `take_safe_text` calls.
        let mut seq = make_seq_with_stop(&["```"], "");
        let mut streamed = String::new();

        for chunk in ["hi", "\n", "`", "`", "`"] {
            seq.unsent_text.push_str(chunk);
            if seq.stop_sequence_match().is_some() {
                break;
            }
            if let Some(text) = seq.take_safe_text() {
                streamed.push_str(&text);
            }
        }

        assert_eq!(streamed, "hi\n");
        assert_eq!(seq.take_pre_stop_text(), None);
    }
}
