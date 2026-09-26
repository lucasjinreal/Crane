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
//! | `memory`        | GPU memory-limit parsing + usage queries          |

pub mod backend;
pub mod grammar;
pub(crate) mod memory;
pub mod model_factory;
pub mod sampling;
pub mod scheduler;
pub mod sequence;
pub mod stats;
pub mod types;

// Re-export commonly used items for convenience.
pub use memory::MemoryConfig;
pub use stats::{EngineStats, StatsSnapshot};
pub use types::{EngineHandle, EngineRequest, EngineResponse, GenerationParams};

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::Ordering;
use std::time::Instant;

use crane_core::{LogitsProcessor, Tensor};
use tokio::sync::mpsc;
use tracing::{debug, error, info, trace, warn};

use backend::ModelBackend;
use crane_core::utils::token_output_stream::TokenOutputStream;
use memory::{floor_kv_budget, format_bytes_engine, query_gpu_memory_usage};
use sampling::SamplingBuffers;
use scheduler::{Scheduler, SchedulerOutput};
use sequence::{Sequence, SequenceStatus};

/// Maximum number of prompt tokens fed to a single `forward_step` during
/// prefill. Larger prompts are split into chunks of this size so each forward
/// step's intermediate state (causal-mask allocation, attention logits, GDN
/// recurrent scratch) stays bounded. See [`InferenceEngine::step_prefill`].
///
/// The engine also hands this size to the model so it does not re-chunk each
/// pass underneath; every pass dequantizes the whole weight set, so a smaller
/// inner chunk multiplies that work for no benefit.
const PREFILL_CHUNK_SIZE: usize = 2048;

/// Whether to log every tool-call grammar state transition and the
/// resulting next-token mask (`CRANE_GRAMMAR_TRACE=1`).
///
/// Diagnostic for confirming whether `grammar::tool_call_skeleton`'s
/// `<tool_call>` wrapper enforcement is actually engaging for a request,
/// as opposed to silently degrading to unconstrained (e.g. because
/// `VocabByteTable::tokens_matching_prefix` found no token able to start
/// a forced literal — see [`InferenceEngine::new`]'s startup sanity log).
#[must_use]
fn grammar_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("CRANE_GRAMMAR_TRACE").as_deref() == Ok("1"))
}

/// Whether to log real/tracked VRAM usage on every engine step
/// (`CRANE_VRAM_TRACE=1`).
///
/// Diagnostic for confirming or ruling out a real-VRAM leak over a long
/// decode run (e.g. a caching allocator never reusing buffers for a
/// strictly growing KV length) — unlike [`InferenceEngine::log_stats`],
/// which only logs every 50 steps, this fires every step so a leak can be
/// seen trending well before a crash a few hundred tokens in.
#[must_use]
fn vram_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("CRANE_VRAM_TRACE").as_deref() == Ok("1"))
}

/// `CRANE_PREFIX_CACHE=0` disables reusing the model state across requests,
/// for A/B against a full prefill.
fn prefix_cache_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("CRANE_PREFIX_CACHE").as_deref(), Ok("0")))
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
///
/// Also reused by `crate::derive_safe_max_seq_len` for the same reason: a
/// naive raw-KV-bytes budget (no overhead factor) undercounts real usage by
/// the same 5-8× and lets a single long-running session's own prefill blow
/// past physical VRAM (verified in production — see git history).
pub(crate) const KV_GPU_OVERHEAD_FACTOR: u64 = 6;

/// Continuous-batching inference engine.
///
/// What the model's in-place caches currently correspond to.
///
/// Only used for backends that cannot swap per-sequence caches
/// (`!ModelBackend::supports_kv_swap()`), where exactly one sequence's state
/// lives in the model at a time. When the next request's prompt extends these
/// tokens, that state is already what prefill would recompute, so prefill can
/// start at `tokens.len()`.
///
/// Reuse requires an **exact** prefix: the hybrid GDN layers carry a recurrent
/// state summarising everything fed so far, so it resumes only at the position
/// it actually reached — unlike an attention KV cache there is no truncating
/// it back to an arbitrary shorter prefix.
#[derive(Default)]
struct PrefixCache {
    /// Tokens the snapshot covers: the previous request's prompt, not its
    /// prompt plus completion. A chat client re-renders the assistant turn
    /// through the chat template, so the next prompt rarely contains the
    /// generated token ids verbatim.
    tokens: Vec<u32>,
    /// Layer state as of the end of that prompt's prefill.
    snapshot: Option<crane_core::models::qwen3_5::StateSnapshot>,
    /// KV bytes this cache keeps resident in the model after its sequence is
    /// gone, so the engine's memory budget still accounts for them.
    bytes: u64,
}

impl PrefixCache {
    /// Length of the longest common prefix of `a` and `b`.
    fn lcp(a: &[u32], b: &[u32]) -> usize {
        a.iter().zip(b).take_while(|(x, y)| x == y).count()
    }

    /// How much of `prompt` the model already holds, or 0 if nothing usable.
    ///
    /// Requires a *strict* prefix: were the cached tokens the whole prompt,
    /// prefill would have nothing to run and no logits to sample, so the final
    /// token is always recomputed.
    fn reusable_len(&self, prompt: &[u32]) -> usize {
        let n = self.tokens.len();
        if n == 0 || n >= prompt.len() || prompt[..n] != self.tokens[..] {
            return 0;
        }
        n
    }
}

/// Runs on a dedicated OS thread (model forward passes are synchronous).
/// Communicates with async API handlers via channels.
pub struct InferenceEngine {
    model: Box<dyn ModelBackend>,
    sequences: HashMap<String, Sequence>,
    token_streams: HashMap<String, TokenOutputStream>,
    /// Tokens whose state the model currently holds; see [`PrefixCache`].
    prefix_cache: PrefixCache,
    /// Prefill requests that reused cached state, and tokens so skipped.
    prefix_hits: u64,
    prefix_misses: u64,
    prefix_tokens_saved: u64,
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
    /// Token-id → decoded-text lookup for the whole vocabulary, built once
    /// from the loaded model's tokenizer. Shared (via `Arc`) into every
    /// sequence's grammar constraint, since it depends only on the model,
    /// not the request. `None` for models that don't use the tool-call XML
    /// skeleton grammar (see [`InferenceEngine::new`]'s `uses_xml_tool_format`
    /// parameter).
    vocab_byte_table: Option<Arc<grammar::VocabByteTable>>,
}

impl InferenceEngine {
    /// Create the engine and return a handle for submitting requests.
    ///
    /// `uses_xml_tool_format` gates the tool-call XML skeleton grammar
    /// (see `grammar::tool_call_skeleton`): only Qwen3-Coder checkpoints
    /// emit that format, so the grammar must stay off for every other
    /// model or it would corrupt their (already well-formed) tool calls.
    pub fn new(
        model: Box<dyn ModelBackend>,
        max_concurrent: usize,
        decode_tokens_per_seq: usize,
        memory_config: MemoryConfig,
        uses_xml_tool_format: bool,
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
            info!("Model does not support KV swap — limiting max_concurrent to {effective_max}");
        }

        let vocab_byte_table = uses_xml_tool_format.then(|| {
            let table = Arc::new(grammar::VocabByteTable::build(model.tokenizer()));
            // One-time sanity check: does the vocab contain any token able to
            // *start* forcing each literal the tool-call grammar relies on? A
            // zero here means `apply_grammar_mask` will always see an empty
            // allow-list for that literal and silently degrade to
            // unconstrained (see `VocabByteTable::tokens_matching_prefix`'s
            // doc) — i.e. the grammar would never actually constrain anything
            // at that point, without erroring.
            debug!(
                wrapper_open_from_idle = table.tokens_matching_prefix("\n<function=", 0).len(),
                call_close = table.tokens_matching_prefix("\n</tool_call>", 0).len(),
                "Grammar vocab sanity check (0 = that literal can never be forced)",
            );
            table
        });
        let stats = Arc::new(EngineStats::new());
        let engine = Self {
            model,
            sequences: HashMap::new(),
            token_streams: HashMap::new(),
            prefix_cache: PrefixCache::default(),
            prefix_hits: 0,
            prefix_misses: 0,
            prefix_tokens_saved: 0,
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
            last_mem_warn: Instant::now()
                .checked_sub(std::time::Duration::from_mins(1))
                .unwrap_or_else(Instant::now),
            tracked_kv_bytes: 0,
            eviction_cooldown: 0,
            vocab_byte_table,
        };
        let handle = EngineHandle { request_tx, stats };
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
        let max_seq_len_str = if self.memory_config.max_seq_len == 0 {
            "unlimited".to_string()
        } else {
            self.memory_config.max_seq_len.to_string()
        };
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

            // These two warnings are mutually exclusive: floor_kv_budget only
            // raises the budget when kv_bytes_per_token() is Some, so a raise
            // is only observable when the backend does report a rate.
            let raw_budget = limit.saturating_sub(baseline) / KV_GPU_OVERHEAD_FACTOR;
            if kv_budget > raw_budget {
                warn!(
                    "KV budget {} raised to {} to fit one full sequence (max_seq_len={}) \
                     (gpu_memory_limit may be too low for this context length)",
                    format_bytes_engine(raw_budget),
                    format_bytes_engine(kv_budget),
                    max_seq_len_str,
                );
            } else if self.model.kv_bytes_per_token().is_none() {
                warn!(
                    "Model backend does not report kv_bytes_per_token; cannot verify \
                     the KV budget can fit one full max_seq_len={} sequence \
                     (gpu_memory_limit may be too low for this context length)",
                    max_seq_len_str,
                );
            }
        }
        // Match the model's inner chunking to ours so it does not split each
        // pass again; `CRANE_PREFILL_CHUNK` still overrides both.
        crane_core::models::qwen3_5::set_default_prefill_chunk(PREFILL_CHUNK_SIZE);

        info!(
            "Engine started (max_concurrent={}, decode_tokens_per_seq={}, max_seq_len={})",
            self.scheduler.max_running, self.decode_tokens_per_seq, max_seq_len_str,
        );

        // Install candle's private, affinity-pinned rayon pool for the
        // engine's lifetime so every forward pass's matmuls run on warm
        // worker threads instead of rayon's ambient global pool.
        let device = self.model.device().clone();
        device.with_context(|| {
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
                        } else if output.is_prefill {
                            self.execute_step(output);
                        } else if self.is_over_kv_budget() {
                            // Decode step over budget — re-check, since KV
                            // usage grows every decode step and a lone
                            // session's own growth is never checked at
                            // prefill time (there is no new prefill here).
                            let all_evicted = self.evict_if_needed();
                            if all_evicted {
                                // Eviction only parked the lone oversized
                                // sequence(s) back in `waiting`, which the
                                // scheduler would immediately re-prefill
                                // into the same budget violation — abort
                                // them instead of looping forever.
                                for seq_id in &output.batch {
                                    if self.sequences.contains_key(seq_id) {
                                        self.send_error(
                                            seq_id,
                                            "KV cache budget exceeded during decode",
                                        );
                                    }
                                }
                            }
                            // Eviction/abort may have removed sequences from
                            // `running` since `output` was scheduled — only
                            // decode survivors.
                            let surviving: Vec<String> = output
                                .batch
                                .into_iter()
                                .filter(|id| self.scheduler.running.contains(id))
                                .collect();
                            if !surviving.is_empty() {
                                self.execute_step(SchedulerOutput {
                                    batch: surviving,
                                    is_prefill: false,
                                });
                            }
                        } else {
                            self.execute_step(output);
                        }
                        self.step_counter += 1;

                        if self.step_counter.is_multiple_of(50) {
                            self.log_stats();
                        }

                        if vram_trace_enabled() {
                            let (gpu_used, gpu_total) = query_gpu_memory_usage(self.model.device());
                            debug!(
                                step = self.step_counter,
                                tracked_kv = %format_bytes_engine(self.tracked_kv_bytes),
                                gpu_used = %format_bytes_engine(gpu_used),
                                gpu_total = %format_bytes_engine(gpu_total),
                                "CRANE_VRAM_TRACE",
                            );
                        }
                    },
                    None => {
                        if let Some(req) = self.request_rx.blocking_recv() {
                            self.accept_request(req);
                        } else {
                            info!("Engine channel closed, shutting down");
                            self.log_stats();
                            return;
                        }
                    },
                }
            }
        });
    }

    fn log_stats(&self) {
        let snap = self.stats.snapshot();
        let uptime = self.start_time.elapsed().as_secs();
        let (gpu_used, gpu_total) = query_gpu_memory_usage(self.model.device());
        let budget = self.kv_budget_bytes();
        let prefix_info = if self.prefix_hits + self.prefix_misses > 0 {
            format!(
                " | prefix_cache: {}/{} hit, {} tokens skipped",
                self.prefix_hits,
                self.prefix_hits + self.prefix_misses,
                self.prefix_tokens_saved,
            )
        } else {
            String::new()
        };
        let budget_info = if budget < u64::MAX {
            format!(" kv_budget: {}", format_bytes_engine(budget))
        } else {
            String::new()
        };
        #[allow(clippy::cast_precision_loss)]
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
            format!(
                " | kv_cache: {}{}",
                format_bytes_engine(self.tracked_kv_bytes),
                budget_info
            )
        };
        info!(
            "Engine stats | uptime={}s | requests: total={} completed={} cancelled={} failed={} | \
             sequences: active={} waiting={} | \
             tokens: prompt={} completion={} | \
             kv_swaps={} | \
             speed: prefill={:.1} tok/s decode={:.1} tok/s{}{}",
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
            prefix_info,
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
    /// Floored at what one full `max_seq_len` sequence needs — a configured
    /// `max_seq_len` must be satisfiable by at least one sequence, or
    /// eviction has nothing else to blame and loops forever evicting the
    /// only sequence, re-prefilling it, and evicting it again (observed in
    /// the field: a single request whose own KV footprint alone exceeded
    /// this budget never completed). The floor only applies when the backend
    /// reports `kv_bytes_per_token`; backends that return `None`
    /// (hybrid/shared KV architectures) get the un-floored budget, and
    /// `run()`'s startup log warns that satisfiability couldn't be verified.
    ///
    /// Returns `u64::MAX` when no limit is configured.
    fn kv_budget_bytes(&self) -> u64 {
        let limit = self.memory_config.gpu_memory_limit_bytes;
        if limit == 0 {
            return u64::MAX;
        }
        let raw = limit.saturating_sub(self.memory_config.baseline_gpu_bytes);
        let budget = raw / KV_GPU_OVERHEAD_FACTOR;
        floor_kv_budget(
            budget,
            self.model.kv_bytes_per_token(),
            self.memory_config.max_seq_len,
        )
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

    /// Selects the running sequence with the longest token list (and
    /// therefore the largest KV cache) as the eviction or abort victim.
    ///
    /// Returns `None` when `running` is empty or no running sequence is
    /// found in `self.sequences`.
    fn largest_running_victim(&self) -> Option<String> {
        self.scheduler
            .running
            .iter()
            .filter_map(|id| {
                self.sequences
                    .get(id)
                    .map(|seq| (id.clone(), seq.tokens.len()))
            })
            .max_by_key(|(_, len)| *len)
            .map(|(id, _)| id)
    }

    /// Preempt (evict) running sequences until KV usage is within budget.
    ///
    /// Eviction policy: **longest-output-first** — the sequence that has
    /// generated the most tokens (and therefore holds the largest KV cache)
    /// is evicted first. Its KV cache is dropped and it is moved back to
    /// the waiting queue for later re-prefill.
    ///
    /// Returns `true` when eviction had to empty `running` entirely (all
    /// sequences evicted). This signals "futile eviction" — typically a
    /// lone oversized sequence whose own KV cache alone exceeds the
    /// budget. Evicting it only parks it back in `waiting`, from which the
    /// scheduler would immediately re-prefill it into the same budget
    /// violation; the caller must abort such sequences instead.
    ///
    /// This mirrors sglang's retraction strategy.
    fn evict_if_needed(&mut self) -> bool {
        let budget = self.kv_budget_bytes();
        if budget == u64::MAX {
            return false;
        }

        let had_running = !self.scheduler.running.is_empty();

        while self.tracked_kv_bytes > budget && !self.scheduler.running.is_empty() {
            let Some(victim_id) = self.largest_running_victim() else {
                break;
            };

            // Compute bytes being freed. If active, bytes are in the model
            // (not in seq.kv_caches) — mirrors cleanup_sequence's branch.
            // Must run before clear_kv_cache() below while the cache is
            // still live.
            let freed = if self.active_seq_id.as_deref() == Some(&victim_id) {
                self.model.active_kv_cache_bytes()
            } else {
                self.sequences
                    .get(&victim_id)
                    .map_or(0, |seq| sequence::kv_cache_bytes(&seq.kv_caches))
            };

            info!(
                id = %victim_id,
                freed_bytes = %format_bytes_engine(freed),
                kv_used = %format_bytes_engine(self.tracked_kv_bytes),
                kv_budget = %format_bytes_engine(budget),
                "Preempting sequence (KV cache eviction) — will re-prefill later",
            );

            // If this sequence's KV is currently loaded in the model, clear it.
            if self.active_seq_id.as_deref() == Some(&victim_id) {
                self.clear_kv_cache_best_effort("eviction");
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

        had_running && self.scheduler.running.is_empty()
    }

    /// Effective `max_tokens` for a request, taking server-level `max_seq_len` into account.
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
        // Reject requests that were already queued when the engine hit a
        // fatal GPU error. `EngineHandle::submit` rejects new requests once
        // this flag is set, but a request enqueued just before the flag was
        // set can still reach here.
        if let Some(err) = self.stats.get_fatal_error() {
            let _ = req.response_tx.send(EngineResponse::Error(format!(
                "Engine is unavailable due to a fatal GPU error: {err}"
            )));
            self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
            return;
        }

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
            let _ = req.response_tx.send(EngineResponse::Error(format!(
                "Prompt length ({}) exceeds server max_seq_len ({})",
                prompt_len, self.memory_config.max_seq_len,
            )));
            self.stats.failed_requests.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // Cap max_tokens to respect max_seq_len.
        let effective_max_tokens = self.effective_max_tokens(prompt_len, req.max_tokens);

        info!(
            id = %req.id,
            prompt_len,
            max_tokens = effective_max_tokens,
            "New request accepted (queue: waiting={} running={})",
            self.scheduler.waiting.len() + 1,
            self.scheduler.running.len(),
        );
        debug!(
            id = %req.id,
            temp = ?req.temperature,
            top_p = ?req.top_p,
            top_k = ?req.top_k,
            rep_penalty = format!("{:.2}", req.repetition_penalty),
            freq_penalty = req.frequency_penalty,
            pres_penalty = req.presence_penalty,
            stop_sequences = ?req.stop,
            eos_token_ids = ?req.eos_token_id,
            "Sampling parameters",
        );

        self.stats.total_requests.fetch_add(1, Ordering::Relaxed);
        self.stats
            .total_prompt_tokens
            .fetch_add(prompt_len as u64, Ordering::Relaxed);

        // Constrains the tool-call XML skeleton so Qwen3-Coder's documented
        // quirk of occasionally omitting the `<tool_call>` opener can't
        // reach the client. `None` when the request offers no tools, or when
        // the loaded model isn't Qwen3-Coder (`vocab_byte_table` is only
        // built for that format — see `InferenceEngine::new`).
        let grammar_constructed = !req.tool_names.is_empty() && self.vocab_byte_table.is_some();
        let grammar_status = if req.tool_names.is_empty() {
            "no tools offered, none constructed"
        } else if self.vocab_byte_table.is_none() {
            "tools offered, but model doesn't use the XML tool-call format — none constructed"
        } else {
            "constructed"
        };
        if grammar_constructed {
            info!(
                id = %req.id,
                tool_count = req.tool_names.len(),
                tool_names = ?req.tool_names,
                "Grammar constraint: {}",
                grammar_status,
            );
        } else {
            debug!(
                id = %req.id,
                tool_count = req.tool_names.len(),
                tool_names = ?req.tool_names,
                "Grammar constraint: {}",
                grammar_status,
            );
        }
        let grammar: Option<Box<dyn grammar::GrammarConstraint>> = if req.tool_names.is_empty() {
            None
        } else {
            self.vocab_byte_table.as_ref().map(|vocab_byte_table| {
                Box::new(grammar::tool_call_skeleton::ToolCallSkeleton::new(
                    req.tool_names,
                    vocab_byte_table.clone(),
                )) as Box<dyn grammar::GrammarConstraint>
            })
        };

        let seq = Sequence {
            cached_prefix_len: 0,
            id: req.id.clone(),
            status: SequenceStatus::Waiting,
            tokens: req.tokens,
            prompt_len,
            kv_caches: vec![None; self.num_layers],
            logits_processor: LogitsProcessor::new(
                sampling::rand_seed(),
                req.temperature,
                req.top_p,
            ),
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            max_tokens: effective_max_tokens,
            eos_token_id: req.eos_token_id,
            grammar,
            repetition_penalty: req.repetition_penalty,
            frequency_penalty: req.frequency_penalty,
            presence_penalty: req.presence_penalty,
            repeat_last_n: 64,
            stop_sequences: req.stop,
            unsent_text: String::new(),
            decode_start: None,
            created_at: Instant::now(),
            first_token_at: None,
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
            .inspect(|(id, seq)| {
                trace!(
                    id = %id,
                    is_closed = seq.response_tx.is_closed(),
                    "check_cancelled: sequence tx state",
                );
            })
            .filter(|(_, seq)| seq.response_tx.is_closed())
            .map(|(id, _)| id.clone())
            .collect();

        if !cancelled.is_empty() {
            debug!(count = cancelled.len(), "Detected cancelled sequences");
        }

        for id in cancelled {
            warn!(id = %id, "Client disconnected, cancelling sequence");
            self.stats
                .cancelled_requests
                .fetch_add(1, Ordering::Relaxed);
            self.cleanup_sequence(&id);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Step execution dispatch
    // ─────────────────────────────────────────────────────────

    /// Run one scheduler step, containing any panic to the sequences involved.
    ///
    /// The inference thread owns the only handle to the model, so an escaping
    /// panic takes the whole engine with it and every later request fails with
    /// "Engine thread has shut down". Instead, fail the step's sequences and
    /// reset the model's caches — a panic mid-forward says nothing about what
    /// state they are in.
    fn execute_step(&mut self, output: SchedulerOutput) {
        let batch = output.batch.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.execute_step_inner(output);
        }));
        let Err(panic) = result else {
            return;
        };
        let msg = panic_message(&panic);
        error!(
            "Inference step panicked, failing {} sequence(s): {msg}",
            batch.len()
        );
        for seq_id in &batch {
            self.send_error(seq_id, &format!("Inference step panicked: {msg}"));
        }
        self.clear_kv_cache_best_effort("panic recovery");
    }

    fn execute_step_inner(&mut self, output: SchedulerOutput) {
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
            self.step_decode_sequential(&output.batch);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Prefill
    // ─────────────────────────────────────────────────────────

    /// Record the model state, which currently covers `covered` tokens of this
    /// sequence's prompt.
    fn capture_prefix_cache(&mut self, seq_id: &str, covered: usize) {
        if self.model.supports_kv_swap() || !prefix_cache_enabled() || covered == 0 {
            return;
        }
        let Some(prompt) = self
            .sequences
            .get(seq_id)
            .filter(|s| covered <= s.prompt_len)
            .map(|s| s.tokens[..covered].to_vec())
        else {
            return;
        };
        let Some(snapshot) = self.model.snapshot_state() else {
            return;
        };
        self.prefix_cache.tokens = prompt;
        self.prefix_cache.snapshot = Some(snapshot);
    }

    /// Forget the retained state, releasing the bytes it was accounted for.
    /// Call wherever the model's caches are about to be reset.
    fn drop_prefix_cache(&mut self) {
        self.tracked_kv_bytes = self
            .tracked_kv_bytes
            .saturating_sub(self.prefix_cache.bytes);
        self.prefix_cache.bytes = 0;
        self.prefix_cache.tokens.clear();
        self.prefix_cache.snapshot = None;
    }

    /// If the model already holds a prefix of this sequence's prompt, mark how
    /// much and keep that state; returns the number of tokens reused.
    ///
    /// Restricted to backends without per-sequence cache swapping: those are
    /// the ones where the model holds exactly one sequence's state, so "what
    /// the model holds" is a single well-defined thing.
    fn claim_prefix_cache(&mut self, seq_id: &str) -> usize {
        if self.model.supports_kv_swap() || !prefix_cache_enabled() {
            return 0;
        }
        let Some(seq) = self.sequences.get(seq_id) else {
            return 0;
        };
        let prompt = &seq.tokens[..seq.prompt_len];
        let reused = self.prefix_cache.reusable_len(prompt);
        if reused == 0 {
            self.prefix_misses += 1;
            if !self.prefix_cache.tokens.is_empty() {
                debug!(
                    id = %seq_id,
                    prompt_len = prompt.len(),
                    cached_len = self.prefix_cache.tokens.len(),
                    common_prefix = PrefixCache::lcp(prompt, &self.prefix_cache.tokens),
                    "Prefix cache miss",
                );
            }
            return 0;
        }
        let Some(snapshot) = self.prefix_cache.snapshot.clone() else {
            return 0;
        };
        if let Err(e) = self.model.restore_state(&snapshot) {
            warn!(id = %seq_id, "Prefix cache restore failed, prefilling in full: {e}");
            self.drop_prefix_cache();
            return 0;
        }
        if let Some(seq) = self.sequences.get_mut(seq_id) {
            seq.cached_prefix_len = reused;
        }
        // The state stays in the model; claim it for this sequence without the
        // reset `swap_in` would do. The sequence accounts for those bytes from
        // here on, so release the cache's claim on them.
        self.tracked_kv_bytes = self
            .tracked_kv_bytes
            .saturating_sub(self.prefix_cache.bytes);
        self.prefix_cache.bytes = 0;
        self.active_seq_id = Some(seq_id.to_string());
        self.prefix_hits += 1;
        self.prefix_tokens_saved += reused as u64;
        info!(
            id = %seq_id,
            reused_tokens = reused,
            "Prefix cache hit: skipping prefill of the shared prefix",
        );
        reused
    }

    fn step_prefill(&mut self, seq_id: String) {
        let t0 = Instant::now();

        // Reuse the state already in the model when this prompt extends it.
        // Must be decided before `swap_in`, which is what would otherwise
        // throw that state away.
        let reused = self.claim_prefix_cache(&seq_id);
        if reused == 0 {
            // A failed cache reset would leave the previous sequence's state in
            // place, so this request cannot run — fail it rather than answer
            // from corrupt state.
            if let Err(e) = self.swap_in(&seq_id) {
                self.send_error(&seq_id, &format!("Cache reset before prefill failed: {e}"));
                return;
            }
        }

        let (input_ids, start_pos) = {
            let seq = self.sequences.get(&seq_id).unwrap();
            (seq.next_input_ids().to_vec(), seq.start_pos())
        };

        let prompt_len = input_ids.len();

        // Checkpoint one token short of the prompt's end. The prompt ends with
        // the generation prompt (`<|im_start|>assistant\n`); next turn the
        // reply text follows that position and BPE re-tokenizes across the
        // join, so the last token of this prompt is not the token at that
        // index next time even though everything before it is identical.
        // Holding one token back costs one extra single-token forward and
        // makes the checkpoint land where the next prompt still agrees.
        let hold_back =
            usize::from(prefix_cache_enabled() && !self.model.supports_kv_swap() && prompt_len > 0);
        let checkpoint_at = prompt_len - hold_back;

        let mut logits = None;
        let mut processed = 0usize;
        while processed < checkpoint_at {
            let chunk_end = (processed + PREFILL_CHUNK_SIZE).min(checkpoint_at);
            let chunk = &input_ids[processed..chunk_end];
            let chunk_start_pos = start_pos + processed;
            logits = match self.model.forward_step(chunk, chunk_start_pos) {
                Ok(l) => Some(l),
                Err(e) => {
                    self.send_error(
                        &seq_id,
                        &format!("Prefill forward failed at chunk {processed}/{prompt_len}: {e}"),
                    );
                    return;
                },
            };
            processed = chunk_end;

            // A large prompt's prefill is chunked across multiple forward
            // passes with no other yield point back to the engine loop, so
            // a disconnect mid-prefill would otherwise run undetected until
            // every remaining chunk finishes (see check_cancelled(), which
            // only runs between steps).
            if self
                .sequences
                .get(&seq_id)
                .is_none_or(|s| s.response_tx.is_closed())
            {
                warn!(
                    id = %seq_id,
                    tokens_processed = processed,
                    prompt_len,
                    "Client disconnected mid-prefill",
                );
                self.stats
                    .cancelled_requests
                    .fetch_add(1, Ordering::Relaxed);
                self.cleanup_sequence(&seq_id);
                return;
            }
        }

        // State now covers everything up to `checkpoint_at`; record it before
        // the held-back token moves it past where the next prompt agrees.
        if hold_back > 0 {
            // `prompt_len` counts only the tokens this pass feeds, so offset
            // by whatever a prefix-cache hit already covered.
            self.capture_prefix_cache(&seq_id, start_pos + checkpoint_at);
        }

        if processed < prompt_len {
            let chunk = &input_ids[processed..prompt_len];
            let chunk_start_pos = start_pos + processed;
            logits = match self.model.forward_step(chunk, chunk_start_pos) {
                Ok(l) => Some(l),
                Err(e) => {
                    self.send_error(
                        &seq_id,
                        &format!("Prefill forward failed at final token: {e}"),
                    );
                    return;
                },
            };
        }

        let logits = match logits {
            Some(l) => l,
            None => {
                self.send_error(&seq_id, "Empty prefill");
                return;
            },
        };

        let next_token = {
            let seq = self.sequences.get_mut(&seq_id).unwrap();
            match sampling::sample(&seq_id, seq, &logits, &mut self.sampling_buffers) {
                Ok(t) => t,
                Err(e) => {
                    self.send_error(&seq_id, &format!("Sampling failed: {e}"));
                    return;
                },
            }
        };

        self.swap_out(&seq_id);

        #[allow(clippy::cast_possible_truncation)]
        let prefill_us = t0.elapsed().as_micros() as u64;
        self.stats
            .total_prefill_time_us
            .fetch_add(prefill_us, Ordering::Relaxed);

        #[allow(clippy::cast_precision_loss)]
        let prefill_tok_s = if prefill_us > 0 {
            (prompt_len as f64) / (prefill_us as f64 / 1_000_000.0)
        } else {
            0.0
        };

        {
            let seq = self.sequences.get_mut(&seq_id).unwrap();
            seq.tokens.push(next_token);
            seq.status = SequenceStatus::Running;
            seq.decode_start = Some(Instant::now());
        }

        info!(
            id = %seq_id,
            prompt_len,
            prefill_chunks = prompt_len.div_ceil(PREFILL_CHUNK_SIZE),
            prefill_ms = prefill_us / 1000,
            prefill_tok_s = format!("{:.1}", prefill_tok_s),
            "Prefill complete, first token generated",
        );

        self.accumulate_token(&seq_id, next_token);

        if self.sequences.get(&seq_id).unwrap().should_stop() {
            self.finish_sequence(&seq_id);
        } else {
            self.flush_pending_text(&seq_id);
            self.scheduler.promote_to_running(seq_id);
        }
    }

    // ─────────────────────────────────────────────────────────
    //  Batched decode
    // ─────────────────────────────────────────────────────────

    /// Remove cancelled/disconnected sequences from a decode batch, cleaning
    /// each one up. Returns the filtered batch.
    fn filter_cancelled_batch(&mut self, batch: Vec<String>) -> Vec<String> {
        let cancelled: Vec<String> = batch
            .iter()
            .filter(|id| {
                self.sequences
                    .get(id.as_str())
                    .is_none_or(|s| s.response_tx.is_closed())
            })
            .cloned()
            .collect();
        for id in &cancelled {
            warn!(id = %id, "Client disconnected before decode batch");
            self.stats
                .cancelled_requests
                .fetch_add(1, Ordering::Relaxed);
            self.cleanup_sequence(id);
        }
        batch
            .into_iter()
            .filter(|id| !cancelled.contains(id))
            .collect()
    }

    /// Flush the model's active single-sequence KV cache before batch decode
    /// takes over, saving it back to the owning sequence.
    fn flush_active_kv_for_batch(&mut self) {
        if let Some(ref prev_id) = self.active_seq_id.take() {
            if self.sequences.contains_key(prev_id) {
                let caches = self.model.get_kv_caches();
                if let Some(seq) = self.sequences.get_mut(prev_id) {
                    seq.kv_caches = caches;
                }
            }
            self.clear_kv_cache_best_effort("swap out");
        }
        self.recount_kv_bytes();
    }

    /// Extract per-sequence KV caches from the model's batched buffer back
    /// into `self.sequences` after a batch decode step.
    fn save_extracted_batch_kv(
        &mut self,
        batch: &[String],
        alive: &[bool],
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) {
        if rounds_done == 0 {
            return;
        }
        match self
            .model
            .extract_batch_kv(kv_lens, original_max_kv, rounds_done)
        {
            Ok(extracted) => {
                for (i, seq_id) in batch.iter().enumerate() {
                    if alive[i]
                        && let Some(seq) = self.sequences.get_mut(seq_id)
                        && i < extracted.len()
                    {
                        seq.kv_caches.clone_from(&extracted[i]);
                    }
                }
                // KV caches changed for multiple sequences — recount.
                self.recount_kv_bytes();
            },
            Err(e) => {
                error!("Final KV extraction failed: {e}");
                self.clear_kv_cache_best_effort("final KV extraction");
                self.recount_kv_bytes();
            },
        }
    }

    /// Finalize sequences that completed or were cancelled during a decode
    /// round before the round loop itself exited (normally or via an error
    /// return). Must be called on every exit path out of
    /// `step_decode_batch`'s round loop, not just the success path, or these
    /// sequences leak: their client never gets a terminal response and their
    /// `seq_id` is never freed from the scheduler.
    fn drain_pending_completions(&mut self, pending_finish: &[String], pending_cancel: &[String]) {
        for id in pending_finish {
            self.finish_sequence(id);
        }
        for id in pending_cancel {
            self.stats
                .cancelled_requests
                .fetch_add(1, Ordering::Relaxed);
            self.cleanup_sequence(id);
        }
    }

    /// Decode step for all running sequences — TRUE BATCHED forward.
    ///
    /// Uses **lazy eviction**: when a sequence completes or is cancelled
    /// mid-loop, it stays in the batch tensor (wasting trivial compute)
    /// rather than triggering an expensive extract→re-setup cycle.
    // The setup, multi-round decode loop, and finalization below are one
    // cohesive decode step; splitting them further would scatter shared
    // local state (batch, kv_lens, alive, positions) across functions.
    #[allow(clippy::too_many_lines)]
    fn step_decode_batch(&mut self, batch: Vec<String>) {
        let t0 = Instant::now();

        let batch = self.filter_cancelled_batch(batch);
        if batch.is_empty() {
            return;
        }

        let batch_size = batch.len();

        self.flush_active_kv_for_batch();

        // Collect KV caches and setup batched decode.
        let kv_caches: Vec<Vec<Option<(Tensor, Tensor)>>> = batch
            .iter()
            .map(|id| self.sequences.get(id).unwrap().kv_caches.clone())
            .collect();

        let (kv_lens, original_max_kv) = match self
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
            },
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
        let full_mask =
            match self
                .model
                .build_batch_decode_mask(&kv_lens, original_max_kv, max_total_width)
            {
                Ok(m) => m,
                Err(e) => {
                    for seq_id in &batch {
                        self.send_error(seq_id, &format!("Mask build failed: {e}"));
                    }
                    self.clear_kv_cache_best_effort("mask build");
                    return;
                },
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
                        *self
                            .sequences
                            .get(&batch[i])
                            .unwrap()
                            .tokens
                            .last()
                            .unwrap()
                    } else {
                        last_tokens[i]
                    }
                })
                .collect();

            let input_ids = match crane_core::ops::copy_from_slice_u32(&tokens, self.model.device())
                .and_then(|t| t.reshape((batch_size, 1)))
            {
                Ok(t) => t,
                Err(e) => {
                    for (i, seq_id) in batch.iter().enumerate() {
                        if alive[i] {
                            self.send_error(
                                seq_id,
                                &format!("Decode input_ids upload failed: {e}"),
                            );
                        }
                    }
                    self.clear_kv_cache_best_effort("decode input upload");
                    self.drain_pending_completions(&pending_finish, &pending_cancel);
                    return;
                },
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
                    self.clear_kv_cache_best_effort("batched decode");
                    self.drain_pending_completions(&pending_finish, &pending_cancel);
                    return;
                },
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
                    },
                };

                let next_token = {
                    let seq = self.sequences.get_mut(seq_id).unwrap();
                    match sampling::sample(seq_id, seq, &seq_logits, &mut self.sampling_buffers) {
                        Ok(t) => t,
                        Err(e) => {
                            self.send_error(seq_id, &format!("Sampling failed: {e}"));
                            alive[i] = false;
                            continue;
                        },
                    }
                };

                if let Some(seq) = self.sequences.get_mut(seq_id) {
                    seq.tokens.push(next_token);
                }
                last_tokens[i] = next_token;

                total_tokens_this_step += 1;
                self.stats
                    .total_decode_steps
                    .fetch_add(1, Ordering::Relaxed);

                self.accumulate_token(seq_id, next_token);

                if self.sequences.get(seq_id).is_none_or(Sequence::should_stop) {
                    alive[i] = false;
                    pending_finish.push(seq_id.clone());
                } else if self
                    .sequences
                    .get(seq_id)
                    .is_none_or(|s| s.response_tx.is_closed())
                {
                    warn!(id = %seq_id, "Client disconnected mid-batch-decode");
                    alive[i] = false;
                    pending_cancel.push(seq_id.clone());
                } else {
                    self.flush_pending_text(seq_id);
                }
            }

            for p in &mut positions {
                *p += 1;
            }
        }

        self.save_extracted_batch_kv(&batch, &alive, &kv_lens, original_max_kv, rounds_done);

        self.drain_pending_completions(&pending_finish, &pending_cancel);

        #[allow(clippy::cast_possible_truncation)]
        let decode_us = t0.elapsed().as_micros() as u64;
        self.stats
            .total_decode_time_us
            .fetch_add(decode_us, Ordering::Relaxed);

        if total_tokens_this_step > 0 {
            #[allow(clippy::cast_precision_loss)]
            let tok_s = if decode_us > 0 {
                (total_tokens_this_step as f64) / (decode_us as f64 / 1_000_000.0)
            } else {
                0.0
            };
            #[allow(clippy::cast_possible_truncation)]
            let setup_ms = t_setup.as_millis() as u64;
            debug!(
                batch_size,
                tokens = total_tokens_this_step,
                rounds = rounds_done,
                finished = pending_finish.len(),
                setup_ms,
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
    fn step_decode_sequential(&mut self, batch: &[String]) {
        let t0 = Instant::now();
        let mut total_tokens: u64 = 0;

        for seq_id in batch {
            if self
                .sequences
                .get(seq_id)
                .is_none_or(|s| s.response_tx.is_closed())
            {
                self.stats
                    .cancelled_requests
                    .fetch_add(1, Ordering::Relaxed);
                self.cleanup_sequence(seq_id);
                continue;
            }

            if let Err(e) = self.swap_in(seq_id) {
                self.send_error(seq_id, &format!("Cache reset before decode failed: {e}"));
                continue;
            }

            for _round in 0..self.decode_tokens_per_seq {
                let (input_ids, start_pos) = {
                    let Some(seq) = self.sequences.get(seq_id) else {
                        break;
                    };
                    (seq.next_input_ids().to_vec(), seq.start_pos())
                };

                let logits = match self.model.forward_step(&input_ids, start_pos) {
                    Ok(l) => l,
                    Err(e) => {
                        self.send_error(seq_id, &format!("Decode forward failed: {e}"));
                        break;
                    },
                };

                let next_token = {
                    let seq = self.sequences.get_mut(seq_id).unwrap();
                    match sampling::sample(seq_id, seq, &logits, &mut self.sampling_buffers) {
                        Ok(t) => t,
                        Err(e) => {
                            self.send_error(seq_id, &format!("Sampling failed: {e}"));
                            break;
                        },
                    }
                };

                if let Some(seq) = self.sequences.get_mut(seq_id) {
                    seq.tokens.push(next_token);
                }

                total_tokens += 1;
                self.stats
                    .total_decode_steps
                    .fetch_add(1, Ordering::Relaxed);

                self.accumulate_token(seq_id, next_token);

                if self.sequences.get(seq_id).is_none_or(Sequence::should_stop) {
                    self.finish_sequence(seq_id);
                    break;
                }

                if self
                    .sequences
                    .get(seq_id)
                    .is_none_or(|s| s.response_tx.is_closed())
                {
                    warn!(id = %seq_id, "Client disconnected mid-decode");
                    self.stats
                        .cancelled_requests
                        .fetch_add(1, Ordering::Relaxed);
                    self.cleanup_sequence(seq_id);
                    break;
                }

                self.flush_pending_text(seq_id);
            }

            self.swap_out(seq_id);
        }

        #[allow(clippy::cast_possible_truncation)]
        let decode_us = t0.elapsed().as_micros() as u64;
        self.stats
            .total_decode_time_us
            .fetch_add(decode_us, Ordering::Relaxed);

        if total_tokens > 0 {
            #[allow(clippy::cast_precision_loss)]
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

    /// Clear the model's caches where there is no longer a sequence to fail —
    /// eviction, teardown, or an error path that has already reported itself.
    /// A failure here is worth knowing about but must not stop the engine.
    fn clear_kv_cache_best_effort(&mut self, context: &str) {
        self.drop_prefix_cache();
        if let Err(e) = self.model.clear_kv_cache() {
            error!("{context}: KV cache reset failed: {e}");
        }
    }

    /// Make `seq_id` the sequence whose state the model holds.
    ///
    /// # Errors
    ///
    /// Returns an error if the model's caches could not be reset, which would
    /// leave the previous sequence's state in place and silently corrupt this
    /// one's output. The caller fails the sequence instead.
    fn swap_in(&mut self, seq_id: &str) -> anyhow::Result<()> {
        if self.active_seq_id.as_deref() == Some(seq_id) {
            return Ok(());
        }

        if !self.model.supports_kv_swap() {
            self.drop_prefix_cache();
            self.model.clear_kv_cache()?;
            self.active_seq_id = Some(seq_id.to_string());
            return Ok(());
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
            .get_mut(seq_id)
            .map_or_else(|| vec![None; self.num_layers], |s| s.kv_caches.clone());
        self.model.set_kv_caches(caches);
        self.active_seq_id = Some(seq_id.to_string());

        self.recount_kv_bytes();
        self.stats
            .total_kv_swap_count
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
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
        if let Some(seq) = self.sequences.get_mut(seq_id)
            && seq.kv_caches.iter().any(Option::is_some)
        {
            seq.kv_caches = vec![None; seq.kv_caches.len()];
        }
        self.recount_kv_bytes();
    }

    // ─────────────────────────────────────────────────────────
    //  Response sending
    // ─────────────────────────────────────────────────────────

    /// Decodes a newly sampled token and buffers its text on the sequence's
    /// `unsent_text`, without streaming it yet. Streaming happens separately
    /// via `flush_pending_text`, once it's confirmed the buffered text isn't
    /// part of a stop-sequence match still in progress.
    fn accumulate_token(&mut self, seq_id: &str, token_id: u32) {
        let text = if let Some(stream) = self.token_streams.get_mut(seq_id) {
            match stream.next_token(token_id) {
                Ok(Some(t)) => t,
                Ok(None) => return,
                Err(e) => {
                    warn!(id = %seq_id, "Token decode error: {e}");
                    return;
                },
            }
        } else {
            return;
        };

        if let Some(seq) = self.sequences.get_mut(seq_id) {
            seq.unsent_text.push_str(&text);
            if let Some(g) = seq.grammar.as_mut() {
                g.advance(token_id, &text);
                if grammar_trace_enabled() {
                    let mask_desc = match g.token_mask() {
                        grammar::TokenMask::Unconstrained => "unconstrained".to_string(),
                        grammar::TokenMask::AllowOnly(ids) => format!("allow_only({})", ids.len()),
                    };
                    debug!(
                        id = %seq_id,
                        token_id,
                        text = %text,
                        allows_eos = g.allows_eos(),
                        next_mask = %mask_desc,
                        "CRANE_GRAMMAR_TRACE",
                    );
                }
                if g.is_finished() {
                    seq.grammar = None;
                }
            }
        }
    }

    /// Streams whatever text is currently safe to send for a sequence that
    /// did not just stop — i.e. any buffered text not withheld as a possible
    /// stop-sequence prefix. Must only be called after confirming
    /// `should_stop()` is false, otherwise a completed stop sequence could
    /// be streamed before `finish_sequence` truncates it.
    fn flush_pending_text(&mut self, seq_id: &str) {
        let Some(text) = self
            .sequences
            .get_mut(seq_id)
            .and_then(Sequence::take_safe_text)
        else {
            return;
        };

        if let Some(seq) = self.sequences.get_mut(seq_id) {
            if seq.first_token_at.is_none() {
                seq.first_token_at = Some(Instant::now());
            }
            if seq
                .response_tx
                .send(EngineResponse::Token { text, token_id: 0 })
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
        if stats::is_fatal_gpu_error(msg) {
            self.stats.set_fatal_error(msg);
        }
        self.cleanup_sequence(seq_id);
    }

    fn finish_sequence(&mut self, seq_id: &str) {
        // A stop sequence has already been produced, so flushing the
        // tokenizer's remaining buffered text would stream part of the stop
        // sequence itself to the client. Only the text preceding the match
        // (already withheld from streaming) is still owed to the client.
        let stopped_by_stop_sequence = self
            .sequences
            .get(seq_id)
            .is_some_and(|s| s.stop_sequence_match().is_some());

        if stopped_by_stop_sequence
            && let Some(seq) = self.sequences.get(seq_id)
            && let Some(matched) = seq.stop_sequence_match()
        {
            debug!(id = %seq_id, matched, "Finishing sequence on stop-sequence match");
        }

        let remaining = if stopped_by_stop_sequence {
            self.sequences
                .get_mut(seq_id)
                .and_then(Sequence::take_pre_stop_text)
                .unwrap_or_default()
        } else {
            // Flush the text buffered by the token that just triggered
            // EOS/max-tokens, followed by the tokenizer's withheld UTF-8 tail.
            let mut remaining = self
                .sequences
                .get_mut(seq_id)
                .map(|s| std::mem::take(&mut s.unsent_text))
                .unwrap_or_default();
            if let Some(rest) = self
                .token_streams
                .get_mut(seq_id)
                .and_then(|s| s.decode_rest().ok().flatten())
            {
                remaining.push_str(&rest);
            }
            remaining
        };

        if !remaining.is_empty()
            && let Some(seq) = self.sequences.get_mut(seq_id)
        {
            if seq.first_token_at.is_none() {
                seq.first_token_at = Some(Instant::now());
            }
            let _ = seq.response_tx.send(EngineResponse::Token {
                text: remaining,
                token_id: 0,
            });
        }

        if let Some(seq) = self.sequences.get(seq_id) {
            let generated_ids = &seq.tokens[seq.prompt_len..];
            let completion_tokens = seq.num_generated();
            let mut full_text = self
                .model
                .tokenizer()
                .decode(generated_ids, true)
                .unwrap_or_default();

            if stopped_by_stop_sequence {
                if let Some(pos) = full_text.rfind(seq.unsent_text.as_str()) {
                    full_text.truncate(pos);
                } else {
                    warn!(
                        id = %seq_id,
                        "Stop sequence matched but not found in final decoded text; \
                         returning untruncated text",
                    );
                }
            }

            let finish_reason = if stopped_by_stop_sequence {
                "stop".to_string()
            } else {
                seq.finish_reason().to_string()
            };
            let decode_tok_s = seq.decode_tokens_per_sec();
            let ttft_ms = seq.ttft_ms();
            let total_secs = seq.created_at.elapsed().as_secs_f64();
            // total_tokens is bounded by prompt_len + max_tokens, far below 2^53.
            #[allow(clippy::cast_precision_loss)]
            let total_tok_s = if total_secs > 0.0 {
                (seq.prompt_len + completion_tokens) as f64 / total_secs
            } else {
                0.0
            };

            info!(
                id = %seq_id,
                prompt_tokens = seq.prompt_len,
                completion_tokens,
                finish_reason = %finish_reason,
                decode_tok_s = format!("{decode_tok_s:.1} tok/s"),
                ttft_ms = ttft_ms.map(|v| format!("{v} ms")),
                total_time = format!("{:.0} ms", total_secs * 1000.0),
                total_tok_s = format!("{total_tok_s:.1} tok/s"),
                "Sequence finished",
            );

            let _ = seq.response_tx.send(EngineResponse::Finished {
                full_text,
                prompt_tokens: seq.prompt_len,
                completion_tokens,
                finish_reason,
                ttft_ms,
                decode_tokens_per_sec: decode_tok_s,
            });

            self.stats
                .total_completion_tokens
                .fetch_add(completion_tokens as u64, Ordering::Relaxed);
            self.stats
                .completed_requests
                .fetch_add(1, Ordering::Relaxed);
            if let Some(ttft) = ttft_ms {
                self.stats
                    .total_ttft_us
                    .fetch_add(ttft * 1000, Ordering::Relaxed);
                self.stats.ttft_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Keep the finished sequence's state in the model so the prefix cache
        // captured during its prefill stays restorable for the next request.
        let keep = self.prefix_cache.snapshot.is_some();
        self.cleanup_sequence_inner(seq_id, keep);
    }

    fn cleanup_sequence(&mut self, seq_id: &str) {
        self.cleanup_sequence_inner(seq_id, false);
    }

    /// `keep_model_state` leaves the model's caches loaded so the prefix cache
    /// can hand them to the next request. Everything else is unchanged.
    fn cleanup_sequence_inner(&mut self, seq_id: &str, keep_model_state: bool) {
        trace!(id = %seq_id, "cleanup_sequence: removing sequence from engine state");

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
        if keep_model_state {
            // The bytes stay resident under the prefix cache's name, so move
            // them across rather than letting the budget forget about them.
            self.prefix_cache.bytes = freed;
            self.tracked_kv_bytes = self.tracked_kv_bytes.saturating_add(freed);
        } else {
            self.clear_kv_cache_best_effort("sequence cleanup");
        }

        // Only lift the eviction cap when the system has drained all
        // waiting sequences. Under sustained load, keeping the cap prevents
        // repeated eviction-readmit cycles (e.g., cap=6 → finish → admit 7th →
        // evict → cap=6 → repeat). Once the load subsides and all waiting
        // sequences are served, we reset so the next burst can try full
        // concurrency again.
        if self.scheduler.effective_max_running.is_some() && self.scheduler.waiting.is_empty() {
            debug!("Eviction cap lifted (no waiting sequences, load subsided)");
            self.scheduler.effective_max_running = None;
        }

        debug!(id = %seq_id, "Sequence cleaned up");
    }
}

/// Best-effort text of a caught panic.
///
/// `panic!("literal")` carries a `&'static str`, `panic!("{x}")` and
/// `.expect(..)` carry a `String`, and anything else is opaque.
fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| panic.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic payload".to_string())
}

#[cfg(test)]
mod panic_message_tests {
    use super::panic_message;

    fn caught(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let err = std::panic::catch_unwind(f).unwrap_err();
        std::panic::set_hook(prev);
        panic_message(&err)
    }

    #[test]
    #[allow(clippy::unnecessary_literal_unwrap)] // the panic is the point
    fn reads_str_and_string_payloads() {
        assert_eq!(caught(|| panic!("a literal")), "a literal");
        let n = 7;
        assert_eq!(caught(move || panic!("formatted {n}")), "formatted 7");
        assert_eq!(
            caught(|| Err::<(), _>("boom").expect("with context")),
            "with context: \"boom\""
        );
    }

    #[test]
    fn falls_back_for_an_opaque_payload() {
        assert_eq!(
            caught(|| std::panic::panic_any(42u32)),
            "unknown panic payload"
        );
    }
}

#[cfg(test)]
mod prefix_cache_tests {
    use super::PrefixCache;

    fn cache(tokens: &[u32]) -> PrefixCache {
        PrefixCache {
            tokens: tokens.to_vec(),
            snapshot: None,
            bytes: 0,
        }
    }

    #[test]
    fn reuses_a_strict_prefix() {
        assert_eq!(cache(&[1, 2, 3]).reusable_len(&[1, 2, 3, 4, 5]), 3);
    }

    #[test]
    fn rejects_divergence_inside_the_cached_span() {
        // The recurrent state summarises all three tokens, so it cannot be
        // rolled back to the common prefix of 2.
        assert_eq!(cache(&[1, 2, 3]).reusable_len(&[1, 2, 9, 4]), 0);
    }

    #[test]
    fn rejects_an_exact_match_so_prefill_still_has_a_token() {
        assert_eq!(cache(&[1, 2, 3]).reusable_len(&[1, 2, 3]), 0);
    }

    #[test]
    fn rejects_a_cache_longer_than_the_prompt() {
        assert_eq!(cache(&[1, 2, 3, 4]).reusable_len(&[1, 2]), 0);
    }

    #[test]
    fn empty_cache_reuses_nothing() {
        assert_eq!(cache(&[]).reusable_len(&[1, 2, 3]), 0);
    }
}
