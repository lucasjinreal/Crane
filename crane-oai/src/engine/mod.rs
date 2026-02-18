pub mod scheduler;
pub mod sequence;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crane_core::models::hunyuan_dense::Model;
use crane_core::models::hunyuan_dense::modeling::build_batch_decode_mask;
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
    pub top_k: Option<usize>,
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
        top_k: Option<usize>,
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
                top_k,
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
///      - Single sequence, uses model's internal KV cache
///   5. Decode step: TRUE BATCHED forward for ALL running sequences
///      - Collect all sequences' KV caches
///      - Pad + stack into batched tensors
///      - ONE forward pass with batch dim N
///      - Sample per-sequence, distribute updated KV caches
///   6. If idle → blocking wait for new request
/// ```
///
/// # GPU utilization strategy
///
/// Decode uses `Model::forward_batch_decode` which pads all sequences'
/// KV caches to the same length, stacks them into `[N, kv_heads, max_len,
/// head_dim]`, and runs ONE forward pass through the entire model. This
/// gives the GPU a much larger workload per kernel launch compared to N
/// sequential single-token forward passes.
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
    decode_input_ids_buf: Option<Tensor>,
    topk_cumsum_mats: HashMap<usize, Tensor>,
    topk_shift_bufs: HashMap<usize, Tensor>,
    topk_shift_idxs: HashMap<usize, Tensor>,
    topk_neg_vecs: HashMap<usize, Tensor>,
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
            decode_input_ids_buf: None,
            topk_cumsum_mats: HashMap::new(),
            topk_shift_bufs: HashMap::new(),
            topk_shift_idxs: HashMap::new(),
            topk_neg_vecs: HashMap::new(),
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
            logits_processor: LogitsProcessor::new(rand_seed(), req.temperature, req.top_p),
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

    /// Decode step for all running sequences — TRUE BATCHED forward.
    ///
    /// Uses **lazy eviction**: when a sequence completes or is cancelled
    /// mid-loop, it stays in the batch tensor (wasting trivial compute)
    /// rather than triggering an expensive extract→re-setup cycle.
    /// KV caches are extracted exactly ONCE at the end.
    fn step_decode_batch(&mut self, batch: Vec<String>) {
        let t0 = Instant::now();

        // 1. Filter cancelled sequences before the expensive forward pass.
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

        if self.model.device.is_cuda() && self.decode_input_ids_buf.is_none() {
            let max_batch = self.scheduler.max_running;
            self.decode_input_ids_buf = match Tensor::zeros((max_batch, 1), DType::U32, &self.model.device) {
                Ok(t) => Some(t),
                Err(e) => {
                    error!("Decode input_ids buffer alloc failed: {e}");
                    None
                }
            };
        }

        // 2. Flush model's internal KV cache state (stale from prior prefill).
        if let Some(ref prev_id) = self.active_seq_id.take() {
            if self.sequences.contains_key(prev_id) {
                let caches = self.model.get_kv_caches();
                if let Some(seq) = self.sequences.get_mut(prev_id) {
                    seq.kv_caches = caches;
                }
            }
            self.model.clear_kv_cache();
        }

        // 3. Collect KV caches and SETUP batched decode (pad+stack — done ONCE).
        let kv_caches: Vec<Vec<Option<(candle_core::Tensor, candle_core::Tensor)>>> = batch
            .iter()
            .map(|id| self.sequences.get(id).unwrap().kv_caches.clone())
            .collect();

        let (kv_lens, original_max_kv) = match self.model.setup_batch_decode(&kv_caches, self.decode_tokens_per_seq) {
            Ok(r) => r,
            Err(e) => {
                error!("Batch decode setup failed: {e}");
                for seq_id in &batch {
                    self.send_error(seq_id, &format!("Batch decode setup failed: {e}"));
                }
                return;
            }
        };
        drop(kv_caches); // Free per-sequence cache references.

        let t_setup = t0.elapsed();

        // 4. Pre-build attention mask at maximum width.
        let max_total_width = original_max_kv + self.decode_tokens_per_seq;
        let full_mask = match build_batch_decode_mask(
            &kv_lens,
            original_max_kv,
            max_total_width,
            &self.model.device,
            self.model.dtype,
        ) {
            Ok(m) => m,
            Err(e) => {
                error!("Mask build failed: {e}");
                self.model.clear_kv_cache();
                return;
            }
        };

        // 5. Multi-round decode loop with LAZY EVICTION.
        //    When a sequence finishes or is cancelled, we mark it as dead but
        //    keep it in the batch tensor. We feed dummy tokens for dead slots
        //    and skip sampling. This avoids the costly extract→re-setup cycle.
        let mut total_tokens_this_step = 0u64;
        let mut rounds_done = 0usize;
        let mut alive = vec![true; batch.len()]; // track which slots are still active
        let mut pending_finish: Vec<String> = Vec::new();
        let mut pending_cancel: Vec<String> = Vec::new();

        // Track per-sequence positions (grows by 1 each round for ALL slots).
        let mut positions: Vec<usize> = batch
            .iter()
            .map(|id| self.sequences.get(id).unwrap().start_pos())
            .collect();

        // Remember last token for each slot (used as dummy input for dead slots).
        let mut last_tokens: Vec<u32> = batch
            .iter()
            .map(|id| *self.sequences.get(id).unwrap().tokens.last().unwrap())
            .collect();

        for round in 0..self.decode_tokens_per_seq {
            // If all sequences are dead, no point continuing.
            if alive.iter().all(|a| !a) {
                break;
            }

            // Collect input tokens — alive slots use their last generated token,
            // dead slots reuse their last known token (output will be discarded).
            let tokens: Vec<u32> = (0..batch.len())
                .map(|i| {
                    if alive[i] {
                        *self.sequences.get(&batch[i]).unwrap().tokens.last().unwrap()
                    } else {
                        last_tokens[i]
                    }
                })
                .collect();

            let input_ids = if let Some(buf) = self.decode_input_ids_buf.as_ref() {
                if let Err(e) = buf.copy_from_slice_u32(&tokens) {
                    error!("Decode input_ids upload failed: {e}");
                    self.model.clear_kv_cache();
                    return;
                }
                match buf.narrow(0, 0, batch_size) {
                    Ok(t) => t,
                    Err(e) => {
                        error!("Decode input_ids narrow failed: {e}");
                        self.model.clear_kv_cache();
                        return;
                    }
                }
            } else {
                match Tensor::new(tokens.as_slice(), &self.model.device) {
                    Ok(t) => match t.reshape((batch_size, 1)) {
                        Ok(t) => t,
                        Err(e) => {
                            error!("Decode input_ids reshape failed: {e}");
                            self.model.clear_kv_cache();
                            return;
                        }
                    },
                    Err(e) => {
                        error!("Decode input_ids creation failed: {e}");
                        self.model.clear_kv_cache();
                        return;
                    }
                }
            };

            // Narrow the pre-built mask to correct width for this round.
            let mask_width = original_max_kv + round + 1;
            let mask_for_round = match &full_mask {
                Some(full) => full.narrow(3, 0, mask_width).ok(),
                None => None,
            };

            // ONE batched forward pass (includes dead slots — trivial waste).
            let logits = match self.model.step_batch_decode_with_input_ids(
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

            // Sample and update only ALIVE sequences.
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

                let next_token = match self.sample(seq_id, &seq_logits) {
                    Ok(t) => t,
                    Err(e) => {
                        self.send_error(seq_id, &format!("Sampling failed: {e}"));
                        alive[i] = false;
                        continue;
                    }
                };

                if let Some(seq) = self.sequences.get_mut(seq_id) {
                    seq.tokens.push(next_token);
                }
                last_tokens[i] = next_token;

                total_tokens_this_step += 1;
                self.stats.total_decode_steps.fetch_add(1, Ordering::Relaxed);

                self.send_token(seq_id, next_token);

                // Check completion.
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

            // Advance positions for ALL slots (KV cache grows uniformly).
            for p in positions.iter_mut() {
                *p += 1;
            }
        }

        // 6. Extract per-sequence KV caches (done ONCE, regardless of how many
        //    sequences finished mid-loop).
        if rounds_done > 0 {
            match self.model.extract_batch_kv(&kv_lens, original_max_kv, rounds_done) {
                Ok(extracted) => {
                    // Save KV only for sequences that are still alive (not finished/cancelled).
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

        // 7. Now finish/cancel sequences AFTER KV is safely extracted.
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
            let alive_count = alive.iter().filter(|a| **a).count();
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

        let trace = std::env::var("CRANE_SAMPLE_TRACE").ok().as_deref() == Some("1");
        let t0 = Instant::now();

        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let t_after_prep = Instant::now();
        if seq.repetition_penalty != 1.0 {
            let start_at = seq.tokens.len().saturating_sub(seq.repeat_last_n);
            apply_repeat_penalty_inplace(&logits, seq.repetition_penalty, &seq.tokens[start_at..])
                .map_err(anyhow::Error::from)?;
        }
        let t_after_rep = Instant::now();

        let greedy = match seq.temperature {
            Some(t) => t <= 0.0,
            None => false,
        };
        if greedy {
            return Ok(logits.argmax(0)?.to_scalar::<u32>()?);
        }

        if logits.device().is_cuda() {
            let top_p = seq.top_p.unwrap_or(1.0);
            let top_p_active = top_p > 0.0 && top_p < 1.0;
            let vocab = logits.dim(0)?;
            let temperature = seq.temperature.unwrap_or(1.0);

            let mut top_k = seq.top_k.unwrap_or(0);
            if top_k == 0 && top_p_active {
                top_k = std::env::var("CRANE_TOPP_FALLBACK_TOPK")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(64);
            }
            top_k = top_k.min(64).min(vocab);

            if top_k > 0 && top_k < vocab {
                let topk_idx = logits.topk_indices(top_k).map_err(anyhow::Error::from)?;
                let topk_logits = logits.gather(&topk_idx, candle_core::D::Minus1)?;
                let t_after_topk = Instant::now();

                if std::env::var("CRANE_TOPK_SAMPLE_ON_CPU").ok().as_deref() == Some("1") {
                    let idx_cpu = topk_idx.to_vec1::<u32>()?;
                    let logits_cpu = topk_logits.to_vec1::<f32>()?;
                    let cpu_logits = Tensor::from_vec(logits_cpu, top_k, &Device::Cpu)?;

                    let pos = seq.logits_processor.sample(&cpu_logits)?;
                    let token = idx_cpu
                        .get(pos as usize)
                        .copied()
                        .unwrap_or_else(|| idx_cpu[0]);

                    if trace {
                        let t_done = Instant::now();
                        debug!(
                            id = %seq_id,
                            top_k,
                            top_p = ?seq.top_p,
                            temp = ?seq.temperature,
                            prep_us = t_after_prep.duration_since(t0).as_micros() as u64,
                            rep_us = t_after_rep.duration_since(t_after_prep).as_micros() as u64,
                            topk_us = t_after_topk.duration_since(t_after_rep).as_micros() as u64,
                            total_us = t_done.duration_since(t0).as_micros() as u64,
                            "sample(topk->cpu)"
                        );
                    }
                    return Ok(token);
                }

                if top_p_active {
                    let scaled = (&topk_logits / temperature)?;
                    let probs = candle_nn::ops::softmax_last_dim(&scaled)?;
                    let cumsum_mat = self.get_topk_cumsum_mat(top_k, logits.device())?;
                    let cumsum = probs.reshape((1, top_k))?.matmul(&cumsum_mat)?.reshape(top_k)?;
                    let mask_le = cumsum.le(top_p)?;

                    let shift = self.get_topk_shift_buf(top_k, logits.device(), mask_le.dtype())?;
                    shift.zero_set()?;
                    if top_k > 1 {
                        let idx = self.get_topk_shift_idx(top_k, logits.device())?;
                        let src = mask_le.narrow(candle_core::D::Minus1, 0, top_k - 1)?;
                        shift.scatter_set(&idx, &src, candle_core::D::Minus1)?;
                    }
                    let mask = (&mask_le + &shift)?.gt(0f64)?;

                    let neg = self.get_topk_neg_vec(top_k, logits.device())?;
                    let masked = mask.where_cond(&topk_logits, &neg)?;
                    let mut pos = sample_gumbel_max_idx(&masked, temperature)?;
                    if pos.rank() == 0 {
                        pos = pos.unsqueeze(0)?;
                    }
                    let token = topk_idx.gather(&pos, candle_core::D::Minus1)?;
                    return Ok(token.squeeze(0)?.to_scalar::<u32>()?);
                }

                let mut pos = sample_gumbel_max_idx(&topk_logits, temperature)?;
                if pos.rank() == 0 {
                    pos = pos.unsqueeze(0)?;
                }
                let token = topk_idx.gather(&pos, candle_core::D::Minus1)?;
                return Ok(token.squeeze(0)?.to_scalar::<u32>()?);
            }
        }

        let top_p = seq.top_p.unwrap_or(1.0);
        if top_p <= 0.0 || top_p >= 1.0 {
            let temperature = seq.temperature.unwrap_or(1.0);
            let idx = sample_gumbel_max_idx(&logits, temperature).map_err(anyhow::Error::from)?;
            return Ok(idx.to_scalar::<u32>()?);
        }

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

fn sample_gumbel_max_idx(logits: &Tensor, temperature: f64) -> candle_core::Result<Tensor> {
    if temperature <= 0.0 {
        return logits.argmax(candle_core::D::Minus1);
    }
    let minus_g = logits.rand_like(1e-7, 0.999)?.log()?.neg()?.log()?;
    if temperature == 1.0 {
        (logits - minus_g)?.argmax(candle_core::D::Minus1)
    } else {
        ((logits / temperature)? - minus_g)?.argmax(candle_core::D::Minus1)
    }
}

fn apply_repeat_penalty_inplace(logits: &Tensor, penalty: f32, context: &[u32]) -> candle_core::Result<()> {
    if context.is_empty() {
        return Ok(());
    }

    let mut unique: HashSet<u32> = HashSet::with_capacity(context.len());
    for &t in context {
        unique.insert(t);
    }
    if unique.is_empty() {
        return Ok(());
    }
    let mut token_ids: Vec<u32> = unique.into_iter().collect();
    token_ids.sort_unstable();

    let idx = Tensor::new(token_ids.as_slice(), logits.device())?;
    let selected = logits.gather(&idx, candle_core::D::Minus1)?;
    let mask = selected.ge(0f64)?;
    let on_true = (&selected / penalty as f64)?;
    let on_false = (&selected * penalty as f64)?;
    let updated = mask.where_cond(&on_true, &on_false)?;
    logits.scatter_set(&idx, &updated, candle_core::D::Minus1)
}

impl InferenceEngine {
    fn get_topk_neg_vec(&mut self, k: usize, device: &candle_core::Device) -> candle_core::Result<Tensor> {
        if let Some(t) = self.topk_neg_vecs.get(&k) {
            if t.device().same_device(device) {
                return Ok(t.clone());
            }
        }
        let t = Tensor::full(-1e9f32, k, device)?;
        self.topk_neg_vecs.insert(k, t.clone());
        Ok(t)
    }

    fn get_topk_shift_idx(&mut self, k: usize, device: &candle_core::Device) -> candle_core::Result<Tensor> {
        if let Some(t) = self.topk_shift_idxs.get(&k) {
            if t.device().same_device(device) {
                return Ok(t.clone());
            }
        }
        if k <= 1 {
            candle_core::bail!("get_topk_shift_idx expects k > 1")
        }
        let t = Tensor::arange(1u32, k as u32, device)?;
        self.topk_shift_idxs.insert(k, t.clone());
        Ok(t)
    }

    fn get_topk_shift_buf(
        &mut self,
        k: usize,
        device: &candle_core::Device,
        dtype: candle_core::DType,
    ) -> candle_core::Result<Tensor> {
        if let Some(t) = self.topk_shift_bufs.get(&k) {
            if t.device().same_device(device) && t.dtype() == dtype {
                return Ok(t.clone());
            }
        }
        let t = Tensor::zeros(k, dtype, device)?;
        self.topk_shift_bufs.insert(k, t.clone());
        Ok(t)
    }

    fn get_topk_cumsum_mat(&mut self, k: usize, device: &candle_core::Device) -> candle_core::Result<Tensor> {
        if let Some(t) = self.topk_cumsum_mats.get(&k) {
            if t.device().same_device(device) {
                return Ok(t.clone());
            }
        }
        let mut data = Vec::with_capacity(k * k);
        for row in 0..k {
            for col in 0..k {
                data.push(if row <= col { 1f32 } else { 0f32 });
            }
        }
        let t = Tensor::from_vec(data, (k, k), device)?;
        self.topk_cumsum_mats.insert(k, t.clone());
        Ok(t)
    }
}

fn rand_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}
