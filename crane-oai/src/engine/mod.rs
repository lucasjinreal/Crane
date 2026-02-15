pub mod scheduler;
pub mod sequence;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

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
    Token { text: String, token_id: u32 },
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
    pub stats: Arc<EngineStats>,
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

// ── Engine statistics (lock-free, shared with API handlers) ──

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
    fn new() -> Self {
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

#[derive(Debug, Clone, serde::Serialize)]
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

// ── Inference engine ──

/// Continuous-batching inference engine.
///
/// Runs on a dedicated OS thread (model forward passes are synchronous).
/// Communicates with async API handlers via channels.
///
/// # Architecture
///
/// ```text
/// API handlers ──(request channel)──► Engine thread
///       ◄──(per-request response channel)──┘
///
/// Engine loop (each iteration = one "step"):
///   1. Drain new requests from channel
///   2. Detect & cancel disconnected clients
///   3. Scheduler picks next batch (prefill > decode)
///   4. Prefill step: run full prompt for ONE new sequence
///      - Single sequence, no KV swap needed
///   5. Decode step: iterate running sequences
///      - Prioritize the currently-loaded sequence to avoid KV swaps
///      - Swap KV caches only when switching sequences
///      - Produce one token per sequence per round
///      - Check for cancellations between sequences
///   6. If idle → blocking wait for new request
/// ```
///
/// # GPU utilization strategy
///
/// Without paged attention, true batched decode (concatenating multiple
/// sequences into one forward pass) is not feasible with candle's
/// concat-based KV cache. Instead, we maximize GPU utilization by:
///
/// - **Minimizing KV cache swaps**: Decode multiple tokens for the
///   currently-loaded sequence before switching to the next one
///   (controlled by `decode_tokens_per_seq`).
/// - **Prefill priority**: New sequences get prefilled immediately,
///   which is a large compute-bound operation that utilizes the GPU well.
/// - **Cancellation detection**: Freed GPU time from cancelled requests
///   is immediately available for other sequences.
pub struct InferenceEngine {
    model: Model,
    sequences: HashMap<String, Sequence>,
    token_streams: HashMap<String, TokenOutputStream>,
    scheduler: Scheduler,
    request_rx: mpsc::UnboundedReceiver<EngineRequest>,
    active_seq_id: Option<String>,
    num_layers: usize,
    stats: Arc<EngineStats>,
    /// How many tokens to decode for one sequence before switching.
    /// Higher = fewer KV swaps but worse latency fairness.
    decode_tokens_per_seq: usize,
    /// Engine start time for uptime calculation.
    start_time: Instant,
    /// Step counter for periodic stats logging.
    step_counter: u64,
}

impl InferenceEngine {
    /// Create the engine and return a handle for submitting requests.
    pub fn new(
        model: Model,
        max_concurrent: usize,
        decode_tokens_per_seq: usize,
    ) -> (Self, EngineHandle) {
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        let num_layers = model.num_layers();
        let stats = Arc::new(EngineStats::new());
        let engine = Self {
            model,
            sequences: HashMap::new(),
            token_streams: HashMap::new(),
            scheduler: Scheduler::new(max_concurrent),
            request_rx,
            active_seq_id: None,
            num_layers,
            stats: stats.clone(),
            decode_tokens_per_seq: decode_tokens_per_seq.max(1),
            start_time: Instant::now(),
            step_counter: 0,
        };
        let handle = EngineHandle {
            request_tx,
            stats,
        };
        (engine, handle)
    }

    /// Run the engine loop (blocking — call from a dedicated thread).
    pub fn run(mut self) {
        info!(
            "Engine started (max_concurrent={}, decode_tokens_per_seq={})",
            self.scheduler.max_running, self.decode_tokens_per_seq,
        );

        loop {
            // 1. Drain all pending requests.
            self.drain_requests();

            // 2. Detect and cancel disconnected clients.
            self.check_cancelled();

            // 3. Update queue stats.
            self.stats
                .active_sequences
                .store(self.scheduler.running.len() as u64, Ordering::Relaxed);
            self.stats
                .waiting_sequences
                .store(self.scheduler.waiting.len() as u64, Ordering::Relaxed);

            // 4. Get scheduling decision.
            let output = self.scheduler.schedule();

            match output {
                Some(output) => {
                    self.execute_step(output);
                    self.step_counter += 1;

                    // Periodic stats logging (every 50 steps).
                    if self.step_counter % 50 == 0 {
                        self.log_stats();
                    }
                }
                None => {
                    // No work — block until a request arrives.
                    match self.request_rx.blocking_recv() {
                        Some(req) => self.accept_request(req),
                        None => {
                            info!("Engine channel closed, shutting down");
                            self.log_stats();
                            return;
                        }
                    }
                }
            }
        }
    }

    fn log_stats(&self) {
        let snap = self.stats.snapshot();
        let uptime = self.start_time.elapsed().as_secs();
        info!(
            "Engine stats | uptime={}s | requests: total={} completed={} cancelled={} failed={} | \
             sequences: active={} waiting={} | \
             tokens: prompt={} completion={} | \
             kv_swaps={} | \
             speed: prefill={:.1} tok/s decode={:.1} tok/s",
            uptime,
            snap.total_requests,
            snap.completed_requests,
            snap.cancelled_requests,
            snap.failed_requests,
            snap.active_sequences,
            snap.waiting_sequences,
            snap.total_prompt_tokens,
            snap.total_completion_tokens,
            snap.total_kv_swaps,
            snap.avg_prefill_tokens_per_sec,
            snap.avg_decode_tokens_per_sec,
        );
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

        info!(
            id = %req.id,
            prompt_len,
            max_tokens = req.max_tokens,
            temp = ?req.temperature,
            top_p = ?req.top_p,
            rep_penalty = req.repetition_penalty,
            "New request accepted (queue: waiting={} running={})",
            self.scheduler.waiting.len() + 1,
            self.scheduler.running.len(),
        );

        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_prompt_tokens
            .fetch_add(prompt_len as u64, Ordering::Relaxed);

        let seq = Sequence {
            id: req.id.clone(),
            status: SequenceStatus::Waiting,
            tokens: req.tokens,
            prompt_len,
            kv_caches: vec![None; self.num_layers],
            logits_processor: LogitsProcessor::new(rand_seed(), req.temperature, req.top_p),
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

    // ── Cancellation detection ──

    /// Check if any client has disconnected (response channel closed).
    /// If so, cancel the sequence immediately to free resources.
    fn check_cancelled(&mut self) {
        let cancelled: Vec<String> = self
            .sequences
            .iter()
            .filter(|(_, seq)| seq.response_tx.is_closed())
            .map(|(id, _)| id.clone())
            .collect();

        for id in cancelled {
            warn!(id = %id, "Client disconnected, cancelling sequence");
            self.stats.cancelled_requests.fetch_add(1, Ordering::Relaxed);
            self.cleanup_sequence(&id);
        }
    }

    // ── Step execution ──

    fn execute_step(&mut self, output: SchedulerOutput) {
        if output.is_prefill {
            debug_assert_eq!(output.batch.len(), 1);
            let seq_id = &output.batch[0];
            self.step_prefill(seq_id.clone());
        } else {
            // Decode step: iterate sequences with multi-token decode.
            self.step_decode_batch(output.batch);
        }
    }

    /// Prefill a new sequence: run the full prompt through the model.
    fn step_prefill(&mut self, seq_id: String) {
        let t0 = Instant::now();

        // Swap in empty KV cache.
        self.swap_in(&seq_id);

        let (input_ids, start_pos) = {
            let seq = self.sequences.get(&seq_id).unwrap();
            (seq.next_input_ids().to_vec(), seq.start_pos())
        };

        let prompt_len = input_ids.len();

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

        let prefill_us = t0.elapsed().as_micros() as u64;
        self.stats
            .total_prefill_time_us
            .fetch_add(prefill_us, Ordering::Relaxed);

        let prefill_tok_s = if prefill_us > 0 {
            (prompt_len as f64) / (prefill_us as f64 / 1_000_000.0)
        } else {
            0.0
        };

        // Update sequence state.
        {
            let seq = self.sequences.get_mut(&seq_id).unwrap();
            seq.tokens.push(next_token);
            seq.status = SequenceStatus::Running;
        }

        info!(
            id = %seq_id,
            prompt_len,
            prefill_ms = prefill_us / 1000,
            prefill_tok_s = format!("{:.1}", prefill_tok_s),
            "Prefill complete, first token generated",
        );

        // Send token.
        self.send_token(&seq_id, next_token);

        // Check if done after first token.
        if self.sequences.get(&seq_id).unwrap().should_stop() {
            self.finish_sequence(&seq_id);
        } else {
            self.scheduler.promote_to_running(seq_id);
        }
    }

    /// Decode step for all running sequences.
    ///
    /// Key optimization: for each sequence, decode `decode_tokens_per_seq`
    /// tokens before moving to the next, reducing KV cache swap overhead.
    /// Between sequences, check for new requests and cancellations.
    fn step_decode_batch(&mut self, batch: Vec<String>) {
        let batch_size = batch.len();
        let t0 = Instant::now();
        let mut total_tokens_this_step = 0u64;
        let mut finished_ids = Vec::new();

        for seq_id in &batch {
            // Check for cancellation before processing.
            if !self.sequences.contains_key(seq_id) {
                continue; // Already cleaned up.
            }
            if self.sequences[seq_id].response_tx.is_closed() {
                warn!(id = %seq_id, "Client disconnected during decode batch");
                self.stats.cancelled_requests.fetch_add(1, Ordering::Relaxed);
                self.cleanup_sequence(seq_id);
                continue;
            }

            // Swap in this sequence's KV cache.
            self.swap_in(seq_id);

            // Decode multiple tokens for this sequence before switching.
            for _tok_idx in 0..self.decode_tokens_per_seq {
                let (input_ids, start_pos) = {
                    let seq = self.sequences.get(seq_id).unwrap();
                    (seq.next_input_ids().to_vec(), seq.start_pos())
                };

                let logits = match self.model.forward_step(&input_ids, start_pos) {
                    Ok(l) => l,
                    Err(e) => {
                        self.send_error(seq_id, &format!("Decode forward failed: {e}"));
                        break;
                    }
                };

                let next_token = match self.sample(seq_id, &logits) {
                    Ok(t) => t,
                    Err(e) => {
                        self.send_error(seq_id, &format!("Sampling failed: {e}"));
                        break;
                    }
                };

                // Update sequence.
                {
                    let seq = self.sequences.get_mut(seq_id).unwrap();
                    seq.tokens.push(next_token);
                }

                total_tokens_this_step += 1;
                self.stats.total_decode_steps.fetch_add(1, Ordering::Relaxed);

                self.send_token(seq_id, next_token);

                // Check termination.
                if self.sequences.get(seq_id).map_or(true, |s| s.should_stop()) {
                    finished_ids.push(seq_id.clone());
                    break;
                }

                // Check for client disconnect mid-generation.
                if self.sequences.get(seq_id).map_or(true, |s| s.response_tx.is_closed()) {
                    warn!(id = %seq_id, "Client disconnected mid-decode");
                    self.stats.cancelled_requests.fetch_add(1, Ordering::Relaxed);
                    self.cleanup_sequence(seq_id);
                    break;
                }
            }

            // Swap out before moving to next sequence.
            self.swap_out(seq_id);
        }

        // Finish completed sequences.
        for id in &finished_ids {
            self.finish_sequence(id);
        }

        let decode_us = t0.elapsed().as_micros() as u64;
        self.stats
            .total_decode_time_us
            .fetch_add(decode_us, Ordering::Relaxed);

        if total_tokens_this_step > 0 {
            let tok_s = if decode_us > 0 {
                (total_tokens_this_step as f64) / (decode_us as f64 / 1_000_000.0)
            } else {
                0.0
            };
            debug!(
                batch_size,
                tokens = total_tokens_this_step,
                decode_ms = decode_us / 1000,
                tok_s = format!("{:.1}", tok_s),
                "Decode step complete",
            );
        }

        // Drain new requests that arrived during decode.
        self.drain_requests();
        self.check_cancelled();
    }

    // ── KV cache management ──

    /// Load a sequence's KV cache into the model.
    fn swap_in(&mut self, seq_id: &str) {
        if self.active_seq_id.as_deref() == Some(seq_id) {
            return;
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

        self.stats.total_kv_swap_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Save the model's current KV cache back to the sequence.
    fn swap_out(&mut self, seq_id: &str) {
        if self.active_seq_id.as_deref() == Some(seq_id) {
            let caches = self.model.get_kv_caches();
            if let Some(seq) = self.sequences.get_mut(seq_id) {
                seq.kv_caches = caches;
            }
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
        let text = if let Some(stream) = self.token_streams.get_mut(seq_id) {
            match stream.next_token(token_id) {
                Ok(Some(t)) => t,
                Ok(None) => return,
                Err(e) => {
                    warn!(id = %seq_id, "Token decode error: {e}");
                    return;
                }
            }
        } else {
            return;
        };

        if let Some(seq) = self.sequences.get(seq_id) {
            // If send fails, client is disconnected.
            if seq.response_tx.send(EngineResponse::Token { text, token_id }).is_err() {
                debug!(id = %seq_id, "Response channel closed (client disconnected)");
            }
        }
    }

    fn send_error(&mut self, seq_id: &str, msg: &str) {
        error!(id = %seq_id, "Engine error: {msg}");
        if let Some(seq) = self.sequences.get(seq_id) {
            let _ = seq.response_tx.send(EngineResponse::Error(msg.to_string()));
        }
        self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
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
            let completion_tokens = seq.num_generated();
            let full_text = self
                .model
                .tokenizer
                .tokenizer
                .decode(generated_ids, true)
                .unwrap_or_default();

            let finish_reason = seq.finish_reason().to_string();

            info!(
                id = %seq_id,
                prompt_tokens = seq.prompt_len,
                completion_tokens,
                finish_reason = %finish_reason,
                "Sequence finished",
            );

            let _ = seq.response_tx.send(EngineResponse::Finished {
                full_text,
                prompt_tokens: seq.prompt_len,
                completion_tokens,
                finish_reason,
            });

            self.stats
                .total_completion_tokens
                .fetch_add(completion_tokens as u64, Ordering::Relaxed);
            self.stats
                .completed_requests
                .fetch_add(1, Ordering::Relaxed);
        }

        self.cleanup_sequence(seq_id);
    }

    fn cleanup_sequence(&mut self, seq_id: &str) {
        self.sequences.remove(seq_id);
        self.token_streams.remove(seq_id);
        self.scheduler.remove(seq_id);

        if self.active_seq_id.as_deref() == Some(seq_id) {
            self.model.clear_kv_cache();
            self.active_seq_id = None;
        }

        debug!(id = %seq_id, "Sequence cleaned up");
    }
}

fn rand_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
