//! Continuous-batching inference engine.
//!
//! # Architecture
//!
//! ```text
//! API handlers ──(request channel)──► Engine thread
//!       ◄──(per-request response channel)──┘
//!
//! Engine loop (each iteration = one "step"):
//!   1. Drain new requests from channel
//!   2. Detect & cancel disconnected clients
//!   3. Scheduler picks next batch (prefill > decode)
//!   4. Prefill step: run full prompt for ONE new sequence
//!   5. Decode step: batched or sequential forward for running sequences
//!   6. If idle → blocking wait for new request
//! ```
//!
//! # Module layout
//!
//! | Module          | Responsibility                                   |
//! |-----------------|--------------------------------------------------|
//! | `types`         | Public request/response types + `EngineHandle`   |
//! | `stats`         | Lock-free counters shared with API layer          |
//! | `sampling`      | Token sampling (top-k, top-p, Gumbel-max, etc.) |
//! | `scheduler`     | FIFO scheduler with prefill priority              |
//! | `sequence`      | Per-request lifecycle state                       |
//! | `backend`       | `ModelBackend` trait + concrete implementations   |
//! | `model_factory` | Auto-detection and factory creation               |

pub mod backend;
pub mod model_factory;
pub mod sampling;
pub mod scheduler;
pub mod sequence;
pub mod stats;
pub mod types;

// Re-export commonly used items for convenience.
pub use stats::{EngineStats, StatsSnapshot};
pub use types::{EngineHandle, EngineRequest, EngineResponse};

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use candle_core::Tensor;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use backend::ModelBackend;
use crane_core::utils::token_output_stream::TokenOutputStream;
use sampling::SamplingBuffers;
use scheduler::{Scheduler, SchedulerOutput};
use sequence::{Sequence, SequenceStatus};

// ─────────────────────────────────────────────────────────────
//  InferenceEngine
// ─────────────────────────────────────────────────────────────

/// Continuous-batching inference engine.
///
/// Runs on a dedicated OS thread (model forward passes are synchronous).
/// Communicates with async API handlers via channels.
pub struct InferenceEngine {
    model: Box<dyn ModelBackend>,
    sequences: HashMap<String, Sequence>,
    token_streams: HashMap<String, TokenOutputStream>,
    scheduler: Scheduler,
    request_rx: mpsc::UnboundedReceiver<EngineRequest>,
    active_seq_id: Option<String>,
    num_layers: usize,
    stats: Arc<EngineStats>,
    /// How many tokens to decode for one sequence before switching.
    decode_tokens_per_seq: usize,
    /// Engine start time for uptime calculation.
    start_time: Instant,
    /// Step counter for periodic stats logging.
    step_counter: u64,
    sampling_buffers: SamplingBuffers,
}

impl InferenceEngine {
    /// Create the engine and return a handle for submitting requests.
    pub fn new(
        model: Box<dyn ModelBackend>,
        max_concurrent: usize,
        decode_tokens_per_seq: usize,
    ) -> (Self, EngineHandle) {
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        let num_layers = model.num_layers();

        // Cap max_concurrent to 1 for models without KV cache swapping.
        let effective_max = if model.supports_kv_swap() {
            max_concurrent
        } else {
            1.min(max_concurrent)
        };
        if effective_max != max_concurrent {
            info!(
                "Model does not support KV swap — limiting max_concurrent to {}",
                effective_max
            );
        }

        let stats = Arc::new(EngineStats::new());
        let engine = Self {
            model,
            sequences: HashMap::new(),
            token_streams: HashMap::new(),
            scheduler: Scheduler::new(effective_max),
            request_rx,
            active_seq_id: None,
            num_layers,
            stats: stats.clone(),
            decode_tokens_per_seq: decode_tokens_per_seq.max(1),
            start_time: Instant::now(),
            step_counter: 0,
            sampling_buffers: SamplingBuffers::new(),
        };
        let handle = EngineHandle {
            request_tx,
            stats,
        };
        (engine, handle)
    }

    // ─────────────────────────────────────────────────────────
    //  Main loop
    // ─────────────────────────────────────────────────────────

    /// Run the engine loop (blocking — call from a dedicated thread).
    pub fn run(mut self) {
        info!(
            "Engine started (max_concurrent={}, decode_tokens_per_seq={})",
            self.scheduler.max_running, self.decode_tokens_per_seq,
        );

        loop {
            self.drain_requests();
            self.check_cancelled();

            self.stats
                .active_sequences
                .store(self.scheduler.running.len() as u64, Ordering::Relaxed);
            self.stats
                .waiting_sequences
                .store(self.scheduler.waiting.len() as u64, Ordering::Relaxed);

            let output = self.scheduler.schedule();

            match output {
                Some(output) => {
                    self.execute_step(output);
                    self.step_counter += 1;

                    if self.step_counter % 50 == 0 {
                        self.log_stats();
                    }
                }
                None => {
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

    // ─────────────────────────────────────────────────────────
    //  Request handling
    // ─────────────────────────────────────────────────────────

    fn drain_requests(&mut self) {
        while let Ok(req) = self.request_rx.try_recv() {
            self.accept_request(req);
        }
    }

    fn accept_request(&mut self, req: EngineRequest) {
        let prompt_len = req.tokens.len();
        let tokenizer = self.model.tokenizer().clone();

        info!(
            id = %req.id,
            prompt_len,
            max_tokens = req.max_tokens,
            temp = ?req.temperature,
            top_p = ?req.top_p,
            top_k = ?req.top_k,
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
            logits_processor: candle_transformers::generation::LogitsProcessor::new(
                sampling::rand_seed(),
                req.temperature,
                req.top_p,
            ),
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
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

    // ─────────────────────────────────────────────────────────
    //  Cancellation detection
    // ─────────────────────────────────────────────────────────

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

    // ─────────────────────────────────────────────────────────
    //  Step execution dispatch
    // ─────────────────────────────────────────────────────────

    fn execute_step(&mut self, output: SchedulerOutput) {
        if output.is_prefill {
            debug_assert_eq!(output.batch.len(), 1);
            let seq_id = &output.batch[0];
            self.step_prefill(seq_id.clone());
        } else if self.model.supports_batch_decode() {
            self.step_decode_batch(output.batch);
        } else {
            self.step_decode_sequential(output.batch);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Prefill
    // ─────────────────────────────────────────────────────────

    fn step_prefill(&mut self, seq_id: String) {
        let t0 = Instant::now();

        self.swap_in(&seq_id);

        let (input_ids, start_pos) = {
            let seq = self.sequences.get(&seq_id).unwrap();
            (seq.next_input_ids().to_vec(), seq.start_pos())
        };

        let prompt_len = input_ids.len();

        let logits = match self.model.forward_step(&input_ids, start_pos) {
            Ok(l) => l,
            Err(e) => {
                self.send_error(&seq_id, &format!("Prefill forward failed: {e}"));
                return;
            }
        };

        let next_token = {
            let seq = self.sequences.get_mut(&seq_id).unwrap();
            match sampling::sample(&seq_id, seq, &logits, &mut self.sampling_buffers) {
                Ok(t) => t,
                Err(e) => {
                    self.send_error(&seq_id, &format!("Sampling failed: {e}"));
                    return;
                }
            }
        };

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

        self.send_token(&seq_id, next_token);

        if self.sequences.get(&seq_id).unwrap().should_stop() {
            self.finish_sequence(&seq_id);
        } else {
            self.scheduler.promote_to_running(seq_id);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Batched decode
    // ─────────────────────────────────────────────────────────

    /// Decode step for all running sequences — TRUE BATCHED forward.
    ///
    /// Uses **lazy eviction**: when a sequence completes or is cancelled
    /// mid-loop, it stays in the batch tensor (wasting trivial compute)
    /// rather than triggering an expensive extract→re-setup cycle.
    fn step_decode_batch(&mut self, batch: Vec<String>) {
        let t0 = Instant::now();

        // Filter cancelled sequences.
        let cancelled: Vec<String> = batch
            .iter()
            .filter(|id| {
                self.sequences
                    .get(id.as_str())
                    .map_or(true, |s| s.response_tx.is_closed())
            })
            .cloned()
            .collect();
        for id in &cancelled {
            warn!(id = %id, "Client disconnected before decode batch");
            self.stats.cancelled_requests.fetch_add(1, Ordering::Relaxed);
            self.cleanup_sequence(id);
        }
        let batch: Vec<String> = batch
            .into_iter()
            .filter(|id| !cancelled.contains(id))
            .collect();
        if batch.is_empty() {
            return;
        }

        let batch_size = batch.len();

        // Flush model's internal KV cache state.
        if let Some(ref prev_id) = self.active_seq_id.take() {
            if self.sequences.contains_key(prev_id) {
                let caches = self.model.get_kv_caches();
                if let Some(seq) = self.sequences.get_mut(prev_id) {
                    seq.kv_caches = caches;
                }
            }
            self.model.clear_kv_cache();
        }

        // Collect KV caches and setup batched decode.
        let kv_caches: Vec<Vec<Option<(Tensor, Tensor)>>> = batch
            .iter()
            .map(|id| self.sequences.get(id).unwrap().kv_caches.clone())
            .collect();

        let (kv_lens, original_max_kv) =
            match self
                .model
                .setup_batch_decode(&kv_caches, self.decode_tokens_per_seq)
            {
                Ok(r) => r,
                Err(e) => {
                    error!("Batch decode setup failed: {e}");
                    for seq_id in &batch {
                        self.send_error(seq_id, &format!("Batch decode setup failed: {e}"));
                    }
                    return;
                }
            };
        drop(kv_caches);

        let t_setup = t0.elapsed();

        // Pre-build attention mask.
        let max_total_width = original_max_kv + self.decode_tokens_per_seq;
        let full_mask = match self.model.build_batch_decode_mask(
            &kv_lens,
            original_max_kv,
            max_total_width,
        ) {
            Ok(m) => m,
            Err(e) => {
                error!("Mask build failed: {e}");
                self.model.clear_kv_cache();
                return;
            }
        };

        // Multi-round decode loop with lazy eviction.
        let mut total_tokens_this_step = 0u64;
        let mut rounds_done = 0usize;
        let mut alive = vec![true; batch.len()];
        let mut pending_finish: Vec<String> = Vec::new();
        let mut pending_cancel: Vec<String> = Vec::new();

        let mut positions: Vec<usize> = batch
            .iter()
            .map(|id| self.sequences.get(id).unwrap().start_pos())
            .collect();

        let mut last_tokens: Vec<u32> = batch
            .iter()
            .map(|id| *self.sequences.get(id).unwrap().tokens.last().unwrap())
            .collect();

        for round in 0..self.decode_tokens_per_seq {
            if alive.iter().all(|a| !a) {
                break;
            }

            let tokens: Vec<u32> = (0..batch.len())
                .map(|i| {
                    if alive[i] {
                        *self.sequences.get(&batch[i]).unwrap().tokens.last().unwrap()
                    } else {
                        last_tokens[i]
                    }
                })
                .collect();

            let input_ids = match crane_core::fused_ops::copy_from_slice_u32(
                &tokens,
                self.model.device(),
            )
            .and_then(|t| t.reshape((batch_size, 1)))
            {
                Ok(t) => t,
                Err(e) => {
                    error!("Decode input_ids upload failed: {e}");
                    self.model.clear_kv_cache();
                    return;
                }
            };

            let mask_width = original_max_kv + round + 1;
            let mask_for_round = match &full_mask {
                Some(full) => full.narrow(3, 0, mask_width).ok(),
                None => None,
            };

            let logits = match self.model.step_batch_decode(
                &input_ids,
                &positions,
                mask_for_round.as_ref(),
                Some((&kv_lens, original_max_kv)),
            ) {
                Ok(l) => l,
                Err(e) => {
                    error!("Batched decode forward failed (round {round}): {e}");
                    for (i, seq_id) in batch.iter().enumerate() {
                        if alive[i] {
                            self.send_error(seq_id, &format!("Batched decode failed: {e}"));
                        }
                    }
                    self.model.clear_kv_cache();
                    return;
                }
            };

            rounds_done += 1;

            for (i, seq_id) in batch.iter().enumerate() {
                if !alive[i] {
                    continue;
                }

                let seq_logits = match logits.narrow(0, i, 1) {
                    Ok(l) => l,
                    Err(e) => {
                        self.send_error(seq_id, &format!("Logits extraction failed: {e}"));
                        alive[i] = false;
                        continue;
                    }
                };

                let next_token = {
                    let seq = self.sequences.get_mut(seq_id).unwrap();
                    match sampling::sample(seq_id, seq, &seq_logits, &mut self.sampling_buffers) {
                        Ok(t) => t,
                        Err(e) => {
                            self.send_error(seq_id, &format!("Sampling failed: {e}"));
                            alive[i] = false;
                            continue;
                        }
                    }
                };

                if let Some(seq) = self.sequences.get_mut(seq_id) {
                    seq.tokens.push(next_token);
                }
                last_tokens[i] = next_token;

                total_tokens_this_step += 1;
                self.stats.total_decode_steps.fetch_add(1, Ordering::Relaxed);

                self.send_token(seq_id, next_token);

                if self.sequences.get(seq_id).map_or(true, |s| s.should_stop()) {
                    alive[i] = false;
                    pending_finish.push(seq_id.clone());
                } else if self
                    .sequences
                    .get(seq_id)
                    .map_or(true, |s| s.response_tx.is_closed())
                {
                    warn!(id = %seq_id, "Client disconnected mid-batch-decode");
                    alive[i] = false;
                    pending_cancel.push(seq_id.clone());
                }
            }

            for p in positions.iter_mut() {
                *p += 1;
            }
        }

        // Extract per-sequence KV caches.
        if rounds_done > 0 {
            match self
                .model
                .extract_batch_kv(&kv_lens, original_max_kv, rounds_done)
            {
                Ok(extracted) => {
                    for (i, seq_id) in batch.iter().enumerate() {
                        if alive[i] {
                            if let Some(seq) = self.sequences.get_mut(seq_id) {
                                if i < extracted.len() {
                                    seq.kv_caches = extracted[i].clone();
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Final KV extraction failed: {e}");
                    self.model.clear_kv_cache();
                }
            }
        }

        for id in &pending_finish {
            self.finish_sequence(id);
        }
        for id in &pending_cancel {
            self.stats.cancelled_requests.fetch_add(1, Ordering::Relaxed);
            self.cleanup_sequence(id);
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
                rounds = rounds_done,
                finished = pending_finish.len(),
                setup_ms = t_setup.as_millis() as u64,
                decode_ms = decode_us / 1000,
                tok_s = format!("{:.1}", tok_s),
                "Batched decode step complete",
            );
        }

        self.drain_requests();
        self.check_cancelled();
    }

    // ─────────────────────────────────────────────────────────
    //  Sequential decode
    // ─────────────────────────────────────────────────────────

    /// Sequential decode for backends without batch decode support.
    fn step_decode_sequential(&mut self, batch: Vec<String>) {
        let t0 = Instant::now();
        let mut total_tokens: u64 = 0;

        for seq_id in &batch {
            if self
                .sequences
                .get(seq_id)
                .map_or(true, |s| s.response_tx.is_closed())
            {
                self.stats
                    .cancelled_requests
                    .fetch_add(1, Ordering::Relaxed);
                self.cleanup_sequence(seq_id);
                continue;
            }

            self.swap_in(seq_id);

            for _round in 0..self.decode_tokens_per_seq {
                let (input_ids, start_pos) = {
                    let seq = match self.sequences.get(seq_id) {
                        Some(s) => s,
                        None => break,
                    };
                    (seq.next_input_ids().to_vec(), seq.start_pos())
                };

                let logits = match self.model.forward_step(&input_ids, start_pos) {
                    Ok(l) => l,
                    Err(e) => {
                        self.send_error(seq_id, &format!("Decode forward failed: {e}"));
                        break;
                    }
                };

                let next_token = {
                    let seq = self.sequences.get_mut(seq_id).unwrap();
                    match sampling::sample(seq_id, seq, &logits, &mut self.sampling_buffers) {
                        Ok(t) => t,
                        Err(e) => {
                            self.send_error(seq_id, &format!("Sampling failed: {e}"));
                            break;
                        }
                    }
                };

                if let Some(seq) = self.sequences.get_mut(seq_id) {
                    seq.tokens.push(next_token);
                }

                total_tokens += 1;
                self.stats
                    .total_decode_steps
                    .fetch_add(1, Ordering::Relaxed);

                self.send_token(seq_id, next_token);

                if self
                    .sequences
                    .get(seq_id)
                    .map_or(true, |s| s.should_stop())
                {
                    self.finish_sequence(seq_id);
                    break;
                }

                if self
                    .sequences
                    .get(seq_id)
                    .map_or(true, |s| s.response_tx.is_closed())
                {
                    warn!(id = %seq_id, "Client disconnected mid-decode");
                    self.stats
                        .cancelled_requests
                        .fetch_add(1, Ordering::Relaxed);
                    self.cleanup_sequence(seq_id);
                    break;
                }
            }

            self.swap_out(seq_id);
        }

        let decode_us = t0.elapsed().as_micros() as u64;
        self.stats
            .total_decode_time_us
            .fetch_add(decode_us, Ordering::Relaxed);

        if total_tokens > 0 {
            let tok_s = if decode_us > 0 {
                (total_tokens as f64) / (decode_us as f64 / 1_000_000.0)
            } else {
                0.0
            };
            debug!(
                tokens = total_tokens,
                decode_ms = decode_us / 1000,
                tok_s = format!("{:.1}", tok_s),
                "Sequential decode step complete",
            );
        }

        self.drain_requests();
        self.check_cancelled();
    }

    // ─────────────────────────────────────────────────────────
    //  KV cache management
    // ─────────────────────────────────────────────────────────

    fn swap_in(&mut self, seq_id: &str) {
        if self.active_seq_id.as_deref() == Some(seq_id) {
            return;
        }

        if !self.model.supports_kv_swap() {
            if self.active_seq_id.as_deref() != Some(seq_id) {
                self.model.clear_kv_cache();
                self.active_seq_id = Some(seq_id.to_string());
            }
            return;
        }

        if let Some(ref prev_id) = self.active_seq_id.clone() {
            let caches = self.model.get_kv_caches();
            if let Some(prev_seq) = self.sequences.get_mut(prev_id) {
                prev_seq.kv_caches = caches;
            }
        }

        let caches = self
            .sequences
            .get(seq_id)
            .map(|s| s.kv_caches.clone())
            .unwrap_or_else(|| vec![None; self.num_layers]);
        self.model.set_kv_caches(caches);
        self.active_seq_id = Some(seq_id.to_string());

        self.stats
            .total_kv_swap_count
            .fetch_add(1, Ordering::Relaxed);
    }

    fn swap_out(&mut self, seq_id: &str) {
        if !self.model.supports_kv_swap() {
            return;
        }
        if self.active_seq_id.as_deref() == Some(seq_id) {
            let caches = self.model.get_kv_caches();
            if let Some(seq) = self.sequences.get_mut(seq_id) {
                seq.kv_caches = caches;
            }
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Response sending
    // ─────────────────────────────────────────────────────────

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
            if seq
                .response_tx
                .send(EngineResponse::Token { text, token_id })
                .is_err()
            {
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

        if let Some(seq) = self.sequences.get(seq_id) {
            let generated_ids = &seq.tokens[seq.prompt_len..];
            let completion_tokens = seq.num_generated();
            let full_text = self
                .model
                .tokenizer()
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
