use std::time::Instant;

use crane_core::{LogitsProcessor, Tensor};
use tokio::sync::mpsc;
use tracing::{debug, trace};

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
    /// Leading prompt tokens whose model state is already loaded, so prefill
    /// can start partway in. Set by the engine's prefix cache when this
    /// request's prompt extends what the model already holds; `0` otherwise.
    ///
    /// Only ever an *exact* prefix length. The hybrid layers carry a recurrent
    /// state that summarises everything fed so far, so it can be resumed only
    /// at the position it actually reached — unlike an attention KV cache,
    /// there is no truncating it to an arbitrary shorter prefix.
    pub cached_prefix_len: usize,

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

    // ── grammar ──
    /// Grammar constraint on what the sampler may produce next (e.g. the
    /// tool-call XML skeleton). `None` when the request offers no tools,
    /// or once the grammar reports [`super::grammar::GrammarConstraint::is_finished`].
    pub grammar: Option<Box<dyn super::grammar::GrammarConstraint>>,

    // ── stop sequences ──
    /// String sequences that terminate generation when produced.
    pub stop_sequences: Vec<String>,
    /// Decoded text accumulated since the last successful `take_safe_text`
    /// call. Used both to detect stop-sequence matches and, via
    /// `take_safe_text`, to withhold text that could still extend into a
    /// stop sequence from being streamed to the client.
    pub unsent_text: String,

    // ── timing ──
    /// Set to `Instant::now()` the moment the first generated token exists
    /// (end of prefill). Used to compute [`Self::decode_tokens_per_sec`];
    /// `None` before prefill has produced that first token. Reset on every
    /// prefill completion, including a re-prefill after KV-cache eviction,
    /// so `decode_tokens_per_sec` reflects only the most recent decode
    /// stint rather than cumulative throughput across preemptions.
    pub decode_start: Option<Instant>,
    /// Set at request arrival (sequence construction). Used with
    /// `first_token_at` to compute [`Self::ttft_ms`].
    pub created_at: Instant,
    /// Set once, the first time a generated-text chunk is sent to the
    /// client via `EngineResponse::Token`. `None` until then.
    pub first_token_at: Option<Instant>,

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
        let generated = self.num_generated();
        if generated >= self.max_tokens {
            debug!(
                id = %self.id,
                generated,
                max_tokens = self.max_tokens,
                "Stop: max tokens reached",
            );
            return true;
        }
        if let Some(&last) = self.tokens.last()
            && self.eos_token_id.contains(&last)
        {
            debug!(id = %self.id, token_id = last, "Stop: EOS token");
            return true;
        }
        if let Some(matched) = self.stop_sequence_match() {
            debug!(id = %self.id, matched, "Stop: stop sequence matched");
            return true;
        }
        false
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
        let overlap = self.stop_prefix_overlap();
        if overlap > 0 {
            trace!(
                id = %self.id,
                overlap,
                unsent = %self.unsent_text,
                "Withholding possible stop-sequence prefix",
            );
        }
        let safe_len = self.unsent_text.len() - overlap;
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
            self.cached_prefix_len
        } else {
            self.tokens.len().saturating_sub(1)
        }
    }

    /// Tokens to feed into the next forward step.
    #[must_use]
    pub fn next_input_ids(&self) -> &[u32] {
        if self.status == SequenceStatus::Waiting {
            // Prefill: feed the prompt tokens the model does not already hold.
            &self.tokens[self.cached_prefix_len..self.prompt_len]
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

    /// Decode-phase token rate since [`Self::decode_start`]. Excludes the
    /// first generated token, since it was produced by prefill rather than
    /// decode and attributing it here would inflate short generations.
    /// Returns `0.0` before decode has started or fewer than two tokens
    /// have been generated.
    #[must_use]
    pub fn decode_tokens_per_sec(&self) -> f64 {
        let Some(start) = self.decode_start else {
            return 0.0;
        };
        let decoded = self.num_generated().saturating_sub(1);
        if decoded == 0 {
            return 0.0;
        }
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed <= 0.0 {
            return 0.0;
        }
        // decoded is bounded by max_tokens, far below 2^53, so f64 precision loss is not a concern.
        #[allow(clippy::cast_precision_loss)]
        let tok_s = decoded as f64 / elapsed;
        tok_s
    }

    /// Time-to-first-token in milliseconds: the delay between request
    /// arrival ([`Self::created_at`]) and the first generated-text chunk
    /// being sent to the client. Returns `None` until a token has been sent.
    #[must_use]
    pub fn ttft_ms(&self) -> Option<u64> {
        let first = self.first_token_at?;
        let duration = first.duration_since(self.created_at);
        // TTFT is bounded by request timeouts (seconds), far below u64::MAX ms.
        #[allow(clippy::cast_possible_truncation)]
        Some(duration.as_millis() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            cached_prefix_len: 0,
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
            grammar: None,
            stop_sequences: vec![],
            unsent_text: String::new(),
            decode_start: None,
            created_at: Instant::now(),
            first_token_at: None,
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
    fn cached_prefix_shifts_prefill_start_and_input() {
        // The engine found the model already holds tokens [1, 2]; prefill must
        // feed only the remainder and start at the absolute position 2 so RoPE
        // and the KV cache line up with what is already there.
        let mut seq = make_seq(&[1, 2, 3, 4], &[], 10, 0, SequenceStatus::Waiting);
        seq.cached_prefix_len = 2;
        assert_eq!(seq.start_pos(), 2);
        assert_eq!(seq.next_input_ids(), &[3, 4]);
    }

    #[test]
    fn cached_prefix_does_not_affect_decode() {
        let mut seq = make_seq(&[1, 2, 3], &[10, 11], 10, 0, SequenceStatus::Running);
        seq.cached_prefix_len = 2;
        assert_eq!(seq.next_input_ids(), &[11]);
        assert_eq!(seq.start_pos(), 4);
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

    // ── decode_tokens_per_sec ────────────────────────────────────────────

    #[test]
    fn decode_tokens_per_sec_zero_before_decode_start() {
        let seq = make_seq(&[1, 2, 3], &[10, 11], 10, 0, SequenceStatus::Running);
        assert_eq!(seq.decode_start, None);
        assert_eq!(seq.decode_tokens_per_sec(), 0.0);
    }

    #[test]
    fn decode_tokens_per_sec_zero_with_fewer_than_two_generated() {
        let mut seq = make_seq(&[1, 2, 3], &[10], 10, 0, SequenceStatus::Running);
        seq.decode_start = Some(Instant::now() - std::time::Duration::from_secs(1));
        assert_eq!(seq.decode_tokens_per_sec(), 0.0);
    }

    #[test]
    fn decode_tokens_per_sec_computes_rate_since_decode_start() {
        // 5 tokens generated, first one attributed to prefill, so 4 tokens
        // over a backdated 2-second window should read as ~2.0 tok/s.
        let mut seq = make_seq(
            &[1, 2, 3],
            &[10, 11, 12, 13, 14],
            10,
            0,
            SequenceStatus::Running,
        );
        seq.decode_start = Some(Instant::now() - std::time::Duration::from_secs(2));
        let tok_s = seq.decode_tokens_per_sec();
        assert!((tok_s - 2.0).abs() < 0.1, "expected ~2.0, got {tok_s}");
    }

    #[test]
    fn decode_tokens_per_sec_zero_when_decode_start_in_future() {
        // Instant::elapsed() saturates to zero when the reference instant is
        // in the future, which must not divide-by-zero.
        let mut seq = make_seq(&[1, 2, 3], &[10, 11], 10, 0, SequenceStatus::Running);
        seq.decode_start = Some(Instant::now() + std::time::Duration::from_secs(100));
        assert_eq!(seq.decode_tokens_per_sec(), 0.0);
    }

    // ── ttft_ms ──────────────────────────────────────────────────────────

    #[test]
    fn ttft_ms_none_before_first_token() {
        let seq = make_seq(&[1, 2, 3], &[], 10, 0, SequenceStatus::Waiting);
        assert_eq!(seq.first_token_at, None);
        assert_eq!(seq.ttft_ms(), None);
    }

    #[test]
    fn ttft_ms_computes_duration_since_created_at() {
        let mut seq = make_seq(&[1, 2, 3], &[10], 10, 0, SequenceStatus::Running);
        seq.created_at = Instant::now() - std::time::Duration::from_millis(250);
        seq.first_token_at = Some(Instant::now());
        let ttft = seq.ttft_ms().expect("ttft should be set");
        assert!((200..=400).contains(&ttft), "expected ~250ms, got {ttft}");
    }
}
