pub mod scheduler;
pub mod sequence;

use std::collections::HashMap;

use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crane_core::models::hunyuan_dense::Model;
use crane_core::utils::token_output_stream::TokenOutputStream;

use scheduler::{Scheduler, SchedulerOutput};
use sequence::{Sequence, SequenceStatus};

// ── Public types shared between engine thread and API handlers ──

/// A request from an API handler to the engine.
pub struct EngineRequest {
    pub id: String,
    pub tokens: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub repetition_penalty: f32,
    pub eos_token_id: u32,
    pub response_tx: mpsc::UnboundedSender<EngineResponse>,
}

/// A response chunk from the engine to an API handler.
#[derive(Debug, Clone)]
pub enum EngineResponse {
    /// A newly generated text delta (for streaming).
    Token {
        text: String,
        token_id: u32,
    },
    /// Generation finished.
    Finished {
        full_text: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: String,
    },
    /// An error occurred.
    Error(String),
}

/// Handle returned to API handlers for sending requests.
#[derive(Clone)]
pub struct EngineHandle {
    request_tx: mpsc::UnboundedSender<EngineRequest>,
}

impl EngineHandle {
    /// Submit a new generation request. Returns a receiver for response chunks.
    pub fn submit(
        &self,
        id: String,
        tokens: Vec<u32>,
        max_tokens: usize,
        temperature: Option<f64>,
        top_p: Option<f64>,
        repetition_penalty: f32,
        eos_token_id: u32,
    ) -> Result<mpsc::UnboundedReceiver<EngineResponse>> {
        let (response_tx, response_rx) = mpsc::unbounded_channel();
        self.request_tx
            .send(EngineRequest {
                id,
                tokens,
                max_tokens,
                temperature,
                top_p,
                repetition_penalty,
                eos_token_id,
                response_tx,
            })
            .map_err(|_| anyhow::anyhow!("Engine thread has shut down"))?;
        Ok(response_rx)
    }
}

// ── Inference engine ──

/// Continuous-batching inference engine.
///
/// Runs on a dedicated OS thread (model forward passes are synchronous).
/// Communicates with async API handlers via channels.
///
/// # Architecture (inspired by nano-vllm / vLLM)
///
/// ```text
/// API handlers ──(request channel)──► Engine thread
///       ◄──(per-request response channel)──┘
///
/// Engine loop:
///   1. Drain new requests from channel
///   2. Scheduler picks next batch (prefill priority)
///   3. For each sequence in batch:
///      a. Swap KV caches (load sequence's cache into model)
///      b. Forward step (model.forward_step)
///      c. Sample next token
///      d. Send token delta to response channel
///      e. Swap KV caches back (save model's cache to sequence)
///      f. If done → send Finished, clean up
///   4. If idle → blocking wait for new request
/// ```
///
/// KV cache swapping is O(1) because candle Tensors are Arc-based.
pub struct InferenceEngine {
    model: Model,
    sequences: HashMap<String, Sequence>,
    /// Per-sequence incremental text decoder (handles BPE boundaries).
    token_streams: HashMap<String, TokenOutputStream>,
    scheduler: Scheduler,
    request_rx: mpsc::UnboundedReceiver<EngineRequest>,
    /// ID of the sequence whose KV cache is currently loaded in the model.
    active_seq_id: Option<String>,
    num_layers: usize,
}

impl InferenceEngine {
    /// Create the engine and return a handle for submitting requests.
    pub fn new(model: Model, max_concurrent: usize) -> (Self, EngineHandle) {
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        let num_layers = model.num_layers();
        let engine = Self {
            model,
            sequences: HashMap::new(),
            token_streams: HashMap::new(),
            scheduler: Scheduler::new(max_concurrent),
            request_rx,
            active_seq_id: None,
            num_layers,
        };
        let handle = EngineHandle { request_tx };
        (engine, handle)
    }

    /// Run the engine loop (blocking — call from a dedicated thread).
    pub fn run(mut self) {
        info!(
            "Engine started (max_concurrent={})",
            self.scheduler.max_running
        );

        loop {
            // 1. Drain all pending requests.
            self.drain_requests();

            // 2. Get scheduling decision.
            let output = self.scheduler.schedule();

            match output {
                Some(output) => self.execute_step(output),
                None => {
                    // No work — block until a request arrives.
                    match self.request_rx.blocking_recv() {
                        Some(req) => self.accept_request(req),
                        None => {
                            info!("Engine channel closed, shutting down");
                            return;
                        }
                    }
                }
            }
        }
    }

    // ── Request handling ──

    fn drain_requests(&mut self) {
        while let Ok(req) = self.request_rx.try_recv() {
            self.accept_request(req);
        }
    }

    fn accept_request(&mut self, req: EngineRequest) {
        let prompt_len = req.tokens.len();
        let tokenizer = self.model.tokenizer.tokenizer.clone();

        debug!(
            id = %req.id,
            prompt_len,
            max_tokens = req.max_tokens,
            "Accepted request"
        );

        let seq = Sequence {
            id: req.id.clone(),
            status: SequenceStatus::Waiting,
            tokens: req.tokens,
            prompt_len,
            kv_caches: vec![None; self.num_layers],
            logits_processor: LogitsProcessor::new(
                rand_seed(),
                req.temperature,
                req.top_p,
            ),
            max_tokens: req.max_tokens,
            eos_token_id: req.eos_token_id,
            repetition_penalty: req.repetition_penalty,
            repeat_last_n: 64,
            response_tx: req.response_tx,
        };

        let stream = TokenOutputStream::new(tokenizer);

        self.sequences.insert(req.id.clone(), seq);
        self.token_streams.insert(req.id.clone(), stream);
        self.scheduler.add(req.id);
    }

    // ── Step execution ──

    fn execute_step(&mut self, output: SchedulerOutput) {
        if output.is_prefill {
            // Prefill: exactly one sequence.
            debug_assert_eq!(output.batch.len(), 1);
            let seq_id = &output.batch[0];
            self.step_prefill(seq_id.clone());
        } else {
            // Decode: all running sequences, one token each.
            for seq_id in output.batch {
                self.step_decode(seq_id);
            }
            // Check for new requests between decode rounds.
            self.drain_requests();
        }
    }

    /// Prefill a new sequence: run the full prompt through the model.
    fn step_prefill(&mut self, seq_id: String) {
        // Swap in empty KV cache.
        self.swap_in(&seq_id);

        let (input_ids, start_pos) = {
            let seq = self.sequences.get(&seq_id).unwrap();
            (seq.next_input_ids().to_vec(), seq.start_pos())
        };

        // Forward pass (full prompt).
        let logits = match self.model.forward_step(&input_ids, start_pos) {
            Ok(l) => l,
            Err(e) => {
                self.send_error(&seq_id, &format!("Prefill forward failed: {e}"));
                return;
            }
        };

        // Sample first token.
        let next_token = match self.sample(&seq_id, &logits) {
            Ok(t) => t,
            Err(e) => {
                self.send_error(&seq_id, &format!("Sampling failed: {e}"));
                return;
            }
        };

        // Swap out KV cache.
        self.swap_out(&seq_id);

        // Update sequence state.
        {
            let seq = self.sequences.get_mut(&seq_id).unwrap();
            seq.tokens.push(next_token);
            seq.status = SequenceStatus::Running;
        }

        // Send token.
        self.send_token(&seq_id, next_token);

        // Check if done.
        if self.sequences.get(&seq_id).unwrap().should_stop() {
            self.finish_sequence(&seq_id);
        } else {
            self.scheduler.promote_to_running(seq_id);
        }
    }

    /// Decode one token for a running sequence.
    fn step_decode(&mut self, seq_id: String) {
        self.swap_in(&seq_id);

        let (input_ids, start_pos) = {
            let seq = self.sequences.get(&seq_id).unwrap();
            (seq.next_input_ids().to_vec(), seq.start_pos())
        };

        // Forward pass (single token).
        let logits = match self.model.forward_step(&input_ids, start_pos) {
            Ok(l) => l,
            Err(e) => {
                self.send_error(&seq_id, &format!("Decode forward failed: {e}"));
                self.swap_out(&seq_id);
                return;
            }
        };

        // Sample.
        let next_token = match self.sample(&seq_id, &logits) {
            Ok(t) => t,
            Err(e) => {
                self.send_error(&seq_id, &format!("Sampling failed: {e}"));
                self.swap_out(&seq_id);
                return;
            }
        };

        self.swap_out(&seq_id);

        // Update.
        {
            let seq = self.sequences.get_mut(&seq_id).unwrap();
            seq.tokens.push(next_token);
        }

        self.send_token(&seq_id, next_token);

        if self.sequences.get(&seq_id).unwrap().should_stop() {
            self.finish_sequence(&seq_id);
        }
    }

    // ── KV cache management ──

    /// Load a sequence's KV cache into the model.
    /// Saves the currently loaded sequence's cache first if different.
    fn swap_in(&mut self, seq_id: &str) {
        if self.active_seq_id.as_deref() == Some(seq_id) {
            return; // Already loaded.
        }

        // Save current if any.
        if let Some(ref prev_id) = self.active_seq_id.clone() {
            let caches = self.model.get_kv_caches();
            if let Some(prev_seq) = self.sequences.get_mut(prev_id) {
                prev_seq.kv_caches = caches;
            }
        }

        // Load target.
        let caches = self
            .sequences
            .get(seq_id)
            .map(|s| s.kv_caches.clone())
            .unwrap_or_else(|| vec![None; self.num_layers]);
        self.model.set_kv_caches(caches);
        self.active_seq_id = Some(seq_id.to_string());
    }

    /// Save the model's current KV cache back to the sequence.
    fn swap_out(&mut self, seq_id: &str) {
        if self.active_seq_id.as_deref() == Some(seq_id) {
            let caches = self.model.get_kv_caches();
            if let Some(seq) = self.sequences.get_mut(seq_id) {
                seq.kv_caches = caches;
            }
            // Don't clear active_seq_id — saves a swap if we come back.
        }
    }

    // ── Sampling ──

    fn sample(&mut self, seq_id: &str, logits: &Tensor) -> Result<u32> {
        let seq = self.sequences.get_mut(seq_id).unwrap();

        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let logits = if seq.repetition_penalty != 1.0 {
            let start_at = seq.tokens.len().saturating_sub(seq.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                seq.repetition_penalty,
                &seq.tokens[start_at..],
            )?
        } else {
            logits
        };

        let next_token = seq.logits_processor.sample(&logits)?;
        Ok(next_token)
    }

    // ── Response sending ──

    fn send_token(&mut self, seq_id: &str, token_id: u32) {
        // Incremental text decode via TokenOutputStream.
        let text = if let Some(stream) = self.token_streams.get_mut(seq_id) {
            match stream.next_token(token_id) {
                Ok(Some(t)) => t,
                Ok(None) => return, // BPE boundary, not yet decodable.
                Err(e) => {
                    warn!(seq_id, "Token decode error: {e}");
                    return;
                }
            }
        } else {
            return;
        };

        if let Some(seq) = self.sequences.get(seq_id) {
            let _ = seq.response_tx.send(EngineResponse::Token { text, token_id });
        }
    }

    fn send_error(&mut self, seq_id: &str, msg: &str) {
        error!(seq_id, "Engine error: {msg}");
        if let Some(seq) = self.sequences.get(seq_id) {
            let _ = seq.response_tx.send(EngineResponse::Error(msg.to_string()));
        }
        self.cleanup_sequence(seq_id);
    }

    fn finish_sequence(&mut self, seq_id: &str) {
        // Flush remaining text from the token stream.
        let remaining = self
            .token_streams
            .get(seq_id)
            .and_then(|s| s.decode_rest().ok().flatten())
            .unwrap_or_default();

        if !remaining.is_empty() {
            if let Some(seq) = self.sequences.get(seq_id) {
                let _ = seq.response_tx.send(EngineResponse::Token {
                    text: remaining,
                    token_id: 0,
                });
            }
        }

        // Send finish message.
        if let Some(seq) = self.sequences.get(seq_id) {
            let generated_ids = &seq.tokens[seq.prompt_len..];
            let full_text = self
                .model
                .tokenizer
                .tokenizer
                .decode(generated_ids, true)
                .unwrap_or_default();

            let _ = seq.response_tx.send(EngineResponse::Finished {
                full_text,
                prompt_tokens: seq.prompt_len,
                completion_tokens: seq.num_generated(),
                finish_reason: seq.finish_reason().to_string(),
            });
        }

        self.cleanup_sequence(seq_id);
    }

    fn cleanup_sequence(&mut self, seq_id: &str) {
        self.sequences.remove(seq_id);
        self.token_streams.remove(seq_id);
        self.scheduler.remove(seq_id);

        // If this was the active sequence, mark as no longer loaded.
        if self.active_seq_id.as_deref() == Some(seq_id) {
            self.model.clear_kv_cache();
            self.active_seq_id = None;
        }

        debug!(seq_id, "Sequence cleaned up");
    }
}

fn rand_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
