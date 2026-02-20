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

use candle_core::{Device, Tensor};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use backend::ModelBackend;
use crane_core::utils::token_output_stream::TokenOutputStream;
use sampling::SamplingBuffers;
use scheduler::{Scheduler, SchedulerOutput};
use sequence::{Sequence, SequenceStatus};

// ─────────────────────────────────────────────────────────────
//  Memory configuration
// ─────────────────────────────────────────────────────────────

/// Configuration for GPU memory limits.
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    /// Maximum tokens per sequence (prompt + completion). 0 = unlimited.
    pub max_seq_len: usize,
    /// GPU memory limit in bytes. 0 = unlimited.
    /// This is an **absolute** limit on total GPU memory usage.
    pub gpu_memory_limit_bytes: u64,
    /// Baseline GPU memory recorded after model load + warmup.
    /// The memory gate compares `(current_used - baseline)` against
    /// `(gpu_memory_limit_bytes - baseline)` so that the limit represents
    /// the *total* allowed usage, not just KV-cache growth.
    pub baseline_gpu_bytes: u64,
}

impl MemoryConfig {
    /// Parse memory configuration from CLI arguments.
    ///
    /// `gpu_memory_limit` accepts:
    ///   - Absolute sizes: "5G", "8G", "5120M", "5368709120" (bytes)
    ///   - Utilization fraction: "0.7" (70% of total GPU memory)
    pub fn parse(max_seq_len: usize, gpu_memory_limit: Option<&str>, device: &Device) -> Self {
        let gpu_memory_limit_bytes = match gpu_memory_limit {
            Some(s) => Self::parse_memory_limit(s, device),
            None => 0,
        };
        Self {
            max_seq_len,
            gpu_memory_limit_bytes,
            baseline_gpu_bytes: 0,
        }
    }

    fn parse_memory_limit(s: &str, device: &Device) -> u64 {
        let s = s.trim();
        if s.is_empty() || s == "0" {
            return 0;
        }

        // Try absolute sizes: "5G", "8G", "5120M", "1024K"
        let upper = s.to_uppercase();
        if upper.ends_with('G') {
            if let Ok(n) = upper[..upper.len() - 1].trim().parse::<f64>() {
                return (n * (1u64 << 30) as f64) as u64;
            }
        }
        if upper.ends_with('M') {
            if let Ok(n) = upper[..upper.len() - 1].trim().parse::<f64>() {
                return (n * (1u64 << 20) as f64) as u64;
            }
        }

        // Try as a fraction (0.0 - 1.0)
        if let Ok(frac) = s.parse::<f64>() {
            if (0.0..=1.0).contains(&frac) {
                let total = Self::query_total_gpu_memory(device);
                if total > 0 {
                    return (frac * total as f64) as u64;
                }
            }
            // If > 1.0, treat as bytes
            if frac > 1.0 {
                return frac as u64;
            }
        }

        tracing::warn!("Could not parse gpu_memory_limit '{}', ignoring", s);
        0
    }

    /// Record baseline GPU memory (call after model load + warmup).
    pub fn record_baseline(&mut self, device: &Device) {
        let (used, _total) = query_gpu_memory_usage(device);
        self.baseline_gpu_bytes = used;
    }

    /// Query total GPU memory (bytes). Returns 0 if unavailable.
    fn query_total_gpu_memory(_device: &Device) -> u64 {
        #[cfg(feature = "cuda")]
        {
            if let Device::Cuda(_) = _device {
                if let Ok((_free, total)) =
                    candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
                {
                    return total as u64;
                }
            }
        }
        0
    }
}

/// Query current GPU memory usage. Returns (used_bytes, total_bytes).
/// Returns (0, 0) if not on CUDA.
fn query_gpu_memory_usage(_device: &Device) -> (u64, u64) {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = _device {
            if let Ok((free, total)) =
                candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
            {
                return ((total - free) as u64, total as u64);
            }
        }
    }
    (0, 0)
}

/// Format a byte count as a human-readable string (used in engine log messages).
fn format_bytes_engine(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.1}G", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.0}M", bytes as f64 / (1u64 << 20) as f64)
    } else {
        format!("{}B", bytes)
    }
}

// ─────────────────────────────────────────────────────────────
//  InferenceEngine
// ─────────────────────────────────────────────────────────────

/// KV-to-GPU overhead factor.
///
/// `tracked_kv_bytes` only captures live per-sequence KV cache tensors, which
/// is roughly 15-20% of the *real* GPU memory consumed.  Batch-decode setup
/// creates padded copies, the CUDA caching allocator retains freed blocks,
/// and forward-pass intermediates add extra pressure.  Empirically the ratio
/// between actual GPU growth over baseline and tracked KV bytes is 5-8×.
///
/// We use 6× so that `kv_budget = (limit - baseline) / 6`.  This gives the
/// engine a realistic estimate of how much KV it can afford before the GPU
/// runs out of memory.
const KV_GPU_OVERHEAD_FACTOR: u64 = 6;

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
    /// Memory configuration for VRAM limits.
    memory_config: MemoryConfig,
    /// Timestamp of last memory-limit warning (to throttle log spam).
    last_mem_warn: Instant,
    /// Tracked total KV cache bytes across all sequences (not relying on
    /// `cuMemGetInfo` which includes CUDA allocator pool bloat).
    tracked_kv_bytes: u64,
    /// Steps remaining before cuMemGetInfo checks are re-enabled after eviction.
    /// The CUDA caching allocator doesn't instantly reflect freed memory, so we
    /// grant a short cooldown after preemption to avoid a deadlock where
    /// cuMemGetInfo always reports over-limit.
    eviction_cooldown: u32,
}

impl InferenceEngine {
    /// Create the engine and return a handle for submitting requests.
    pub fn new(
        model: Box<dyn ModelBackend>,
        max_concurrent: usize,
        decode_tokens_per_seq: usize,
        memory_config: MemoryConfig,
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
            memory_config,
            last_mem_warn: Instant::now() - std::time::Duration::from_secs(60),
            tracked_kv_bytes: 0,
            eviction_cooldown: 0,
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
        // Log effective memory budget.
        let baseline = self.memory_config.baseline_gpu_bytes;
        let limit = self.memory_config.gpu_memory_limit_bytes;
        if limit > 0 {
            let kv_budget = self.kv_budget_bytes();
            if kv_budget == 0 || limit <= baseline {
                warn!(
                    "gpu_memory_limit ({}) <= model baseline ({}). \
                     KV-cache budget is 0 — all sequences will be immediately preempted.",
                    format_bytes_engine(limit),
                    format_bytes_engine(baseline),
                );
            } else {
                info!(
                    "Memory budget: total_limit={}, model_baseline={}, kv_budget={} (overhead={}x, also checked by cuMemGetInfo)",
                    format_bytes_engine(limit),
                    format_bytes_engine(baseline),
                    format_bytes_engine(kv_budget),
                    KV_GPU_OVERHEAD_FACTOR,
                );
            }
        }
        info!(
            "Engine started (max_concurrent={}, decode_tokens_per_seq={}, max_seq_len={})",
            self.scheduler.max_running,
            self.decode_tokens_per_seq,
            if self.memory_config.max_seq_len == 0 { "unlimited".to_string() } else { self.memory_config.max_seq_len.to_string() },
        );

        loop {
            self.drain_requests();
            self.check_cancelled();

            // Decrement eviction cooldown (cuMemGetInfo grace period).
            self.eviction_cooldown = self.eviction_cooldown.saturating_sub(1);

            self.stats
                .active_sequences
                .store(self.scheduler.running.len() as u64, Ordering::Relaxed);
            self.stats
                .waiting_sequences
                .store(self.scheduler.waiting.len() as u64, Ordering::Relaxed);

            let output = self.scheduler.schedule();

            match output {
                Some(output) => {
                    // KV cache budget gate: if a prefill is scheduled but we're
                    // over the KV budget, first try to evict (preempt) the
                    // largest running sequence to make room. If still over,
                    // defer the prefill and drain existing sequences.
                    if output.is_prefill && self.is_over_kv_budget() {
                        // Attempt eviction before deferring.
                        self.evict_if_needed();

                        if self.is_over_kv_budget() && !self.scheduler.running.is_empty() {
                            // Still over budget and have running sequences to drain.
                            for seq_id in &output.batch {
                                self.scheduler.waiting.push_front(seq_id.clone());
                            }
                            let decode_batch: Vec<String> =
                                self.scheduler.running.iter().cloned().collect();
                            let decode_output = SchedulerOutput {
                                batch: decode_batch,
                                is_prefill: false,
                            };
                            self.execute_step(decode_output);
                        } else {
                            // Budget OK after eviction (or nothing running) — proceed.
                            self.execute_step(output);
                        }
                    } else {
                        self.execute_step(output);
                    }
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
        let (gpu_used, gpu_total) = query_gpu_memory_usage(self.model.device());
        let budget = self.kv_budget_bytes();
        let budget_info = if budget < u64::MAX {
            format!(" kv_budget: {}", format_bytes_engine(budget))
        } else {
            String::new()
        };
        let gpu_info = if gpu_total > 0 {
            format!(
                " | gpu_mem: {:.1}G/{:.1}G ({:.0}%) | kv_cache: {}{}",
                gpu_used as f64 / (1u64 << 30) as f64,
                gpu_total as f64 / (1u64 << 30) as f64,
                gpu_used as f64 / gpu_total as f64 * 100.0,
                format_bytes_engine(self.tracked_kv_bytes),
                budget_info,
            )
        } else {
            format!(" | kv_cache: {}{}", format_bytes_engine(self.tracked_kv_bytes), budget_info)
        };
        info!(
            "Engine stats | uptime={}s | requests: total={} completed={} cancelled={} failed={} | \
             sequences: active={} waiting={} | \
             tokens: prompt={} completion={} | \
             kv_swaps={} | \
             speed: prefill={:.1} tok/s decode={:.1} tok/s{}",
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
            gpu_info,
        );
    }

    // ─────────────────────────────────────────────────────────
    //  Memory management
    // ─────────────────────────────────────────────────────────

    /// Recount `tracked_kv_bytes` from all sequences.
    /// For the active sequence, bytes are in the model (uses `active_kv_cache_bytes`).
    /// For other sequences, bytes are stored in `seq.kv_caches`.
    fn recount_kv_bytes(&mut self) {
        let mut total: u64 = 0;
        for (id, seq) in &self.sequences {
            if self.active_seq_id.as_deref() == Some(id.as_str()) {
                total += self.model.active_kv_cache_bytes();
            } else {
                total += sequence::kv_cache_bytes(&seq.kv_caches);
            }
        }
        self.tracked_kv_bytes = total;
    }

    /// KV cache budget **in KV-cache bytes** (not raw GPU bytes).
    ///
    /// Each byte of live KV cache costs roughly `KV_GPU_OVERHEAD_FACTOR` bytes
    /// of real GPU memory (due to padded batch copies, CUDA pool bloat, and
    /// forward-pass intermediates).  The budget is therefore:
    ///
    /// ```text
    /// kv_budget = (gpu_limit - baseline) / KV_GPU_OVERHEAD_FACTOR
    /// ```
    ///
    /// Returns `u64::MAX` when no limit is configured.
    fn kv_budget_bytes(&self) -> u64 {
        let limit = self.memory_config.gpu_memory_limit_bytes;
        if limit == 0 {
            return u64::MAX;
        }
        let raw = limit.saturating_sub(self.memory_config.baseline_gpu_bytes);
        raw / KV_GPU_OVERHEAD_FACTOR
    }

    /// Check whether the engine should block new prefills due to memory
    /// pressure.  Two complementary checks:
    ///
    /// 1. **KV budget** — `tracked_kv_bytes > kv_budget_bytes()`.  This is the
    ///    primary admission control, using an overhead factor to estimate real
    ///    GPU cost from the tracked KV cache bytes.
    ///
    /// 2. **cuMemGetInfo hard safety** — if actual GPU memory (as reported by
    ///    the driver) exceeds the configured limit, block prefills.  This
    ///    catches cases where the overhead factor underestimates.  The check
    ///    is skipped during `eviction_cooldown` to avoid a deadlock (the CUDA
    ///    caching allocator doesn't instantly reflect freed memory).
    fn is_over_kv_budget(&mut self) -> bool {
        let limit = self.memory_config.gpu_memory_limit_bytes;
        if limit == 0 {
            return false;
        }

        let budget = self.kv_budget_bytes();
        if budget == 0 {
            return true; // limit <= baseline
        }

        // Check 1: tracked KV bytes vs overhead-adjusted budget.
        if self.tracked_kv_bytes > budget {
            let now = Instant::now();
            if now.duration_since(self.last_mem_warn).as_secs() >= 5 {
                self.last_mem_warn = now;
                warn!(
                    "KV budget exceeded: kv_used={} > kv_budget={} (limit={} baseline={} overhead={}x)",
                    format_bytes_engine(self.tracked_kv_bytes),
                    format_bytes_engine(budget),
                    format_bytes_engine(limit),
                    format_bytes_engine(self.memory_config.baseline_gpu_bytes),
                    KV_GPU_OVERHEAD_FACTOR,
                );
            }
            return true;
        }

        // Check 2: cuMemGetInfo hard safety (skip during cooldown).
        if self.eviction_cooldown == 0 {
            let (gpu_used, _) = query_gpu_memory_usage(self.model.device());
            if gpu_used > 0 && gpu_used > limit {
                let now = Instant::now();
                if now.duration_since(self.last_mem_warn).as_secs() >= 5 {
                    self.last_mem_warn = now;
                    warn!(
                        "GPU memory hard limit exceeded: gpu_used={} > limit={} (kv_tracked={})",
                        format_bytes_engine(gpu_used),
                        format_bytes_engine(limit),
                        format_bytes_engine(self.tracked_kv_bytes),
                    );
                }
                return true;
            }
        }

        false
    }

    /// Preempt (evict) running sequences until KV usage is within budget.
    ///
    /// Eviction policy: **longest-output-first** — the sequence that has
    /// generated the most tokens (and therefore holds the largest KV cache)
    /// is evicted first. Its KV cache is dropped and it is moved back to
    /// the waiting queue for later re-prefill.
    ///
    /// This mirrors sglang's retraction strategy.
    fn evict_if_needed(&mut self) {
        let budget = self.kv_budget_bytes();
        if budget == u64::MAX {
            return;
        }

        while self.tracked_kv_bytes > budget && !self.scheduler.running.is_empty() {
            // Find the running sequence with the most generated tokens (largest KV).
            let victim_id = self
                .scheduler
                .running
                .iter()
                .filter_map(|id| {
                    self.sequences.get(id).map(|seq| (id.clone(), seq.tokens.len()))
                })
                .max_by_key(|(_, len)| *len)
                .map(|(id, _)| id);

            let victim_id = match victim_id {
                Some(id) => id,
                None => break,
            };

            // Compute bytes being freed.
            let freed = self
                .sequences
                .get(&victim_id)
                .map(|seq| sequence::kv_cache_bytes(&seq.kv_caches))
                .unwrap_or(0);

            info!(
                id = %victim_id,
                freed_bytes = %format_bytes_engine(freed),
                kv_used = %format_bytes_engine(self.tracked_kv_bytes),
                kv_budget = %format_bytes_engine(budget),
                "Preempting sequence (KV cache eviction) — will re-prefill later",
            );

            // If this sequence's KV is currently loaded in the model, clear it.
            if self.active_seq_id.as_deref() == Some(&victim_id) {
                self.model.clear_kv_cache();
                self.active_seq_id = None;
            }

            // Drop KV caches and reset sequence state to Waiting.
            if let Some(seq) = self.sequences.get_mut(&victim_id) {
                seq.kv_caches = vec![None; self.num_layers];
                seq.status = SequenceStatus::Waiting;
                // Reset tokens to just the prompt to allow re-prefill.
                seq.tokens.truncate(seq.prompt_len);
            }

            self.tracked_kv_bytes = self.tracked_kv_bytes.saturating_sub(freed);

            // Move from running back to waiting (back, not front — avoid
            // immediate re-prefill which would cause thrashing).
            self.scheduler.running.retain(|id| id != &victim_id);
            self.scheduler.waiting.push_back(victim_id);
        }

        // Cap effective max_running to the post-eviction running count.
        // This prevents the scheduler from admitting new sequences that
        // would immediately exceed the budget again (eviction thrashing).
        // The cap is lifted when a sequence finishes naturally.
        let post_eviction_running = self.scheduler.running.len();
        self.scheduler.effective_max_running = Some(post_eviction_running);
        info!(
            "Eviction complete: capping concurrent sequences at {} (was {})",
            post_eviction_running, self.scheduler.max_running,
        );

        // Grant a cooldown period so the cuMemGetInfo hard-safety check
        // doesn't immediately re-trigger (CUDA pool retains freed blocks).
        self.eviction_cooldown = 5;
    }

    /// Effective max_tokens for a request, taking server-level max_seq_len into account.
    fn effective_max_tokens(&self, prompt_len: usize, requested_max_tokens: usize) -> usize {
        if self.memory_config.max_seq_len == 0 {
            return requested_max_tokens;
        }
        let remaining = self.memory_config.max_seq_len.saturating_sub(prompt_len);
        requested_max_tokens.min(remaining)
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

        // Reject prompts that already exceed max_seq_len.
        if self.memory_config.max_seq_len > 0 && prompt_len > self.memory_config.max_seq_len {
            warn!(
                id = %req.id,
                prompt_len,
                max_seq_len = self.memory_config.max_seq_len,
                "Prompt exceeds max_seq_len, rejecting request",
            );
            let _ = req.response_tx.send(EngineResponse::Error(
                format!(
                    "Prompt length ({}) exceeds server max_seq_len ({})",
                    prompt_len, self.memory_config.max_seq_len,
                ),
            ));
            self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // Cap max_tokens to respect max_seq_len.
        let effective_max_tokens = self.effective_max_tokens(prompt_len, req.max_tokens);

        info!(
            id = %req.id,
            prompt_len,
            max_tokens = effective_max_tokens,
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
            max_tokens: effective_max_tokens,
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
        } else if self.model.supports_batch_decode() && output.batch.len() > 1 {
            // True batched decode only when there are multiple sequences.
            // For a single sequence the sequential path is far cheaper: it
            // keeps the KV cache resident in the model and avoids the
            // extract→pad→stack→extract GPU-copy cycle that batch decode
            // performs every scheduling round.
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
        self.recount_kv_bytes();

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

        // Now that setup_batch_decode has consumed the KV views (building its
        // own padded buffer), drop the per-sequence cache references.  With
        // zero-copy narrow views from get_kv_caches(), these still pin the
        // old pre-allocated buffers — clearing them here lets CUDA free that
        // VRAM before the decode loop allocates intermediates.
        for seq_id in &batch {
            if let Some(seq) = self.sequences.get_mut(seq_id) {
                seq.kv_caches = vec![None; self.num_layers];
            }
        }

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
                    // KV caches changed for multiple sequences — recount.
                    self.recount_kv_bytes();
                }
                Err(e) => {
                    error!("Final KV extraction failed: {e}");
                    self.model.clear_kv_cache();
                    self.recount_kv_bytes();
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

        // Save previous active sequence's KV cache from the model.
        if let Some(ref prev_id) = self.active_seq_id.clone() {
            let caches = self.model.get_kv_caches();
            if let Some(prev_seq) = self.sequences.get_mut(prev_id) {
                prev_seq.kv_caches = caches;
            }
        }

        // Load new sequence's KV cache into the model.
        let caches = self
            .sequences
            .get(seq_id)
            .map(|s| s.kv_caches.clone())
            .unwrap_or_else(|| vec![None; self.num_layers]);
        self.model.set_kv_caches(caches);
        self.active_seq_id = Some(seq_id.to_string());

        self.recount_kv_bytes();
        self.stats
            .total_kv_swap_count
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Mark that the model finished processing `seq_id` for this scheduling
    /// round.  Instead of extracting full KV caches (expensive GPU copies),
    /// we only update byte tracking from the model's internal state.
    /// The actual KV tensors remain in the model and are saved lazily by
    /// `swap_in` when switching to a different sequence.
    fn swap_out(&mut self, seq_id: &str) {
        if !self.model.supports_kv_swap() {
            return;
        }
        if self.active_seq_id.as_deref() != Some(seq_id) {
            return;
        }
        // Drop stale seq cache references (from the last swap_in) to free
        // GPU memory.  swap_in will extract fresh caches from the model
        // when switching to a different sequence.
        if let Some(seq) = self.sequences.get_mut(seq_id) {
            if seq.kv_caches.iter().any(|c| c.is_some()) {
                seq.kv_caches = vec![None; seq.kv_caches.len()];
            }
        }
        self.recount_kv_bytes();
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
        // Subtract this sequence's KV bytes from the tracked total.
        // If active, bytes are in the model (not in seq.kv_caches).
        let freed = if self.active_seq_id.as_deref() == Some(seq_id) {
            self.model.active_kv_cache_bytes()
        } else if let Some(seq) = self.sequences.get(seq_id) {
            sequence::kv_cache_bytes(&seq.kv_caches)
        } else {
            0
        };
        self.tracked_kv_bytes = self.tracked_kv_bytes.saturating_sub(freed);

        self.sequences.remove(seq_id);
        self.token_streams.remove(seq_id);
        self.scheduler.remove(seq_id);

        if self.active_seq_id.as_deref() == Some(seq_id) {
            self.active_seq_id = None;
        }
        self.model.clear_kv_cache();

        // Only lift the eviction cap when the system has drained all
        // waiting sequences. Under sustained load, keeping the cap prevents
        // repeated eviction-readmit cycles (e.g., cap=6 → finish → admit 7th →
        // evict → cap=6 → repeat). Once the load subsides and all waiting
        // sequences are served, we reset so the next burst can try full
        // concurrency again.
        if self.scheduler.effective_max_running.is_some()
            && self.scheduler.waiting.is_empty()
        {
            debug!(
                "Eviction cap lifted (no waiting sequences, load subsided)"
            );
            self.scheduler.effective_max_running = None;
        }

        debug!(id = %seq_id, "Sequence cleaned up");
    }
}
