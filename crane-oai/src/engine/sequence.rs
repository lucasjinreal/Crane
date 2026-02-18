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
    pub eos_token_id: u32,
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
            if last == self.eos_token_id {
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
            if last == self.eos_token_id {
                return "stop";
            }
        }
        "length"
    }
}
