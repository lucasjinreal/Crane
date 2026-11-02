// SPDX-License-Identifier: MIT

//! Per-expert GPU residency cache for Mixture-of-Experts layers.
//!
//! Replaces whole-layer expert promotion (all-or-nothing per layer) with
//! per-expert granularity: a fixed-size pool of GPU slots holds whichever
//! experts are currently judged "hot" from live routing traffic, corrected
//! periodically by [`ExpertCache::decay_and_swap`]. The initial placement
//! is seeded cheaply (first experts by index, no calibration pass) and
//! corrected over time by swapping a hot CPU-resident expert into a cold
//! GPU-resident expert's slot once live hit counts show a clear, sustained
//! mismatch.

use candle_core::quantized::{GgmlDType, QTensor, ggml_file::qtensor_from_ggml};
use candle_core::{Device, Result};
use ribo::utils::log;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// Token count the initial seed placement serves, uncorrected, before the
/// first decay/swap window closes. Short enough that traffic patterns
/// haven't been judged yet, but long enough to populate non-zero hit
/// counts for the experts current traffic actually routes to.
const WARMUP_TOKENS: u64 = 128;

/// Default [`SwapTuning::aggressive_windows`]: number of short-interval
/// windows after warmup (or after a steady-state re-entry) that
/// aggressively correct the biggest mismatches before relaxing to the
/// steady cadence. Overridden by `CRANE_MOE_AGGRESSIVE_WINDOWS`.
const AGGRESSIVE_WINDOWS_DEFAULT: u32 = 4;

/// Decay/swap-check interval, in tokens, during the aggressive phase.
const AGGRESSIVE_WINDOW_TOKENS: u64 = 256;

/// Decay/swap-check interval, in tokens, during the steady phase.
const STEADY_WINDOW_TOKENS: u64 = 2048;

/// Default [`SwapTuning::swap_margin`]: hysteresis margin a CPU-resident
/// expert must out-hit the coldest GPU-resident expert by before a swap
/// triggers. Keeps two similarly-hot experts from flip-flopping every
/// window. Unvalidated against real traffic — overridden by
/// `CRANE_MOE_SWAP_MARGIN`.
const SWAP_MARGIN_DEFAULT: f64 = 2.0;

/// Default [`SwapTuning::min_hot_count`]: a CPU-resident expert's hit count
/// must reach at least this many hits before it's eligible to trigger a
/// swap, even if it clears [`SWAP_MARGIN_DEFAULT`]. Without this floor, a
/// GPU-resident expert whose count has decayed to 0 makes *any* nonzero
/// CPU hit count "exceed the margin" (anything is greater than `0 *
/// margin`), so swaps fire on single-digit noise rather than a real
/// traffic shift — observed on real-traffic logs as 25%+ of layers
/// swapping every steady-state window, perpetually re-triggering the
/// aggressive phase. Unvalidated — overridden by `CRANE_MOE_SWAP_MIN_HOT`.
const MIN_HOT_COUNT_DEFAULT: u64 = 4;

/// Default [`SwapTuning::reentry_fraction`]: fraction of layers that must
/// swap in one steady-state window to signal a traffic shift and re-enter
/// the aggressive phase. Overridden by `CRANE_MOE_REENTRY_FRACTION`.
const REENTRY_FRACTION_DEFAULT: f64 = 0.25;

/// Reads `key` from the environment and parses it as `T`, falling back to
/// `default` if the variable is unset or fails to parse. Used by
/// [`SwapTuning::from_env`] so each decay/swap knob can be tuned from real
/// traffic without a rebuild.
fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Decay/swap decision thresholds, read once from the environment at
/// [`ExpertCache::new`] time rather than on the per-token hot path, so
/// real-traffic tuning doesn't need a rebuild. Defaults match this
/// module's original hardcoded values, so behavior is unchanged unless a
/// variable is set.
#[derive(Clone, Copy, Debug)]
struct SwapTuning {
    swap_margin: f64,
    min_hot_count: u64,
    reentry_fraction: f64,
    aggressive_windows: u32,
}

impl SwapTuning {
    /// Reads `CRANE_MOE_SWAP_MARGIN`, `CRANE_MOE_SWAP_MIN_HOT`,
    /// `CRANE_MOE_REENTRY_FRACTION`, and `CRANE_MOE_AGGRESSIVE_WINDOWS`,
    /// falling back to this module's defaults for any unset or unparsable
    /// variable.
    fn from_env() -> Self {
        Self {
            swap_margin: env_or("CRANE_MOE_SWAP_MARGIN", SWAP_MARGIN_DEFAULT),
            min_hot_count: env_or("CRANE_MOE_SWAP_MIN_HOT", MIN_HOT_COUNT_DEFAULT),
            reentry_fraction: env_or("CRANE_MOE_REENTRY_FRACTION", REENTRY_FRACTION_DEFAULT),
            aggressive_windows: env_or("CRANE_MOE_AGGRESSIVE_WINDOWS", AGGRESSIVE_WINDOWS_DEFAULT),
        }
    }
}

/// Residency sentinel meaning "this expert is CPU-resident" rather than a
/// valid [`LayerPool`] index.
const CPU_SENTINEL: u32 = u32::MAX;

/// Index into a layer's slot within a [`LayerPool`]'s per-layer packed
/// `QTensor` (shape `[slots_per_layer, rows, cols]`) — dimension 0 of that
/// tensor, not a global offset across layers. The same numeric value in
/// `gate_up_pool` and `down_pool` refers to the two weight halves of the
/// same cached expert, scoped to one layer. This is also the index the
/// mixed GPU/CPU dispatch path built on top of this cache passes as an
/// expert id to `QTensor::indexed_moe_forward` on that layer's pool tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotIdx(u32);

impl SlotIdx {
    /// Returns the raw slot index as a `u32`.
    ///
    /// Used by the mixed GPU/CPU dispatch path to build the `[n_gpu, 1]`
    /// expert-id tensor passed to `QTensor::indexed_moe_forward` on a pool
    /// tensor.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

/// Where an expert's weights live for a given `(layer, expert)` pair.
/// Mutable at runtime: [`ExpertCache::decay_and_swap`] can flip an entry
/// between `Gpu`/`Cpu` between scheduler steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    /// Expert's weights are resident in this GPU slot.
    Gpu(SlotIdx),
    /// Expert stays CPU-resident, dispatched via the CPU batched path.
    Cpu,
}

/// Adaptive decay/swap schedule phase: short intervals right after seeding
/// to correct the arbitrary initial placement quickly, then long intervals
/// once placement has settled, with automatic re-entry if traffic shifts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// First [`WARMUP_TOKENS`] tokens after seeding: no decay, no swaps.
    Warmup,
    /// First [`SwapTuning::aggressive_windows`] windows after warmup: short
    /// decay interval, correcting the arbitrary seed's biggest mismatches.
    Aggressive {
        /// Windows completed so far in this aggressive run.
        windows_elapsed: u32,
    },
    /// Remainder of server lifetime: long decay interval. Re-enters
    /// `Aggressive { windows_elapsed: 0 }` if a window swaps more than
    /// [`SwapTuning::reentry_fraction`] of layers (a traffic shift).
    Steady,
}

impl Phase {
    /// Token count a window must accumulate before this phase's
    /// decay/swap check runs.
    const fn window_tokens(self) -> u64 {
        match self {
            Phase::Warmup => WARMUP_TOKENS,
            Phase::Aggressive { .. } => AGGRESSIVE_WINDOW_TOKENS,
            Phase::Steady => STEADY_WINDOW_TOKENS,
        }
    }
}

/// [`ExpertCache`]'s mutable phase/window bookkeeping. `Mutex`-guarded
/// rather than atomics: transitions read-then-write multiple fields
/// together, and `on_step` is called at most once per scheduler step —
/// contention is a non-issue.
struct PhaseState {
    phase: Phase,
    tokens_in_window: u64,
}

/// VRAM budget resolved into expert-cache slot counts. Produced by
/// [`ExpertCacheBudget::compute`] from measured free VRAM; consumed by
/// [`ExpertCache::new`] to size both slot pools.
#[derive(Clone, Copy, Debug)]
pub struct ExpertCacheBudget {
    /// Raw VRAM-budget-derived slot count before per-layer clamping to
    /// `n_experts`. Not the actual number of slots allocated — that's
    /// `n_layers * slots_per_layer`, which can be far smaller once
    /// `slots_per_layer` is clamped to `n_experts`.
    pub raw_budget_slots: usize,
    /// Slots allocated uniformly to every layer (floor division of
    /// `raw_budget_slots` by `n_layers`, clamped to `n_experts`).
    /// Remainder slots from the division are unused.
    pub slots_per_layer: usize,
    /// Largest per-expert byte cost of the `gate_up` projection blob across
    /// all `MoE` layers (layers can use different GGML quantization types,
    /// e.g. a dynamic-quant GGUF mixing `Q4_K`/`Q5_K` per layer, as long as they
    /// share the same tensor shape). Used only to conservatively size
    /// `slots_per_layer` so every layer's quota fits regardless of which
    /// layer it lands on — not necessarily any individual layer's real cost.
    pub gate_up_blob_bytes: usize,
    /// Largest per-expert byte cost of the down projection blob across all
    /// `MoE` layers. See [`Self::gate_up_blob_bytes`].
    pub down_blob_bytes: usize,
    /// Number of `MoE` layers the cache spans.
    pub n_layers: usize,
    /// Number of experts per `MoE` layer.
    pub n_experts: usize,
}

impl ExpertCacheBudget {
    /// Computes slot counts from measured free VRAM. Returns `None` if the
    /// budget can't fit at least one expert-pair slot per layer — too
    /// little VRAM for per-expert caching to be useful.
    #[must_use]
    pub fn compute(
        free_bytes: u64,
        safety_margin: u64,
        gate_up_blob_bytes: usize,
        down_blob_bytes: usize,
        n_layers: usize,
        n_experts: usize,
    ) -> Option<Self> {
        let budget = free_bytes.checked_sub(safety_margin)?;
        let combined_blob = (gate_up_blob_bytes + down_blob_bytes) as u64;
        if combined_blob == 0 || n_layers == 0 {
            return None;
        }
        #[allow(clippy::cast_possible_truncation)]
        // VRAM byte counts fit comfortably in usize on every platform this
        // runs on (64-bit CUDA/ROCm hosts); `usize` here is just a natural
        // slot-count type, not a sized buffer index.
        let raw_budget_slots = (budget / combined_blob) as usize;
        if raw_budget_slots < n_layers {
            return None;
        }
        let slots_per_layer = (raw_budget_slots / n_layers).min(n_experts);
        Some(Self {
            raw_budget_slots,
            slots_per_layer,
            gate_up_blob_bytes,
            down_blob_bytes,
            n_layers,
            n_experts,
        })
    }
}

/// One expert projection's GPU-resident slots: one packed `QTensor` per
/// layer (shape `[slots_per_layer, rows, cols]`), not one `QTensor` per
/// slot. This lets the mixed GPU/CPU dispatch path pass a layer's pool
/// tensor straight to `QTensor::indexed_moe_forward` with local slot
/// indices as expert ids — one kernel launch per layer, not one per routed
/// expert.
///
/// Candle's `QTensor` has no partial-write or sub-region view, so a layer's
/// pool tensor can't be updated in place: [`Self::rebuild_layer`] instead
/// recomposes the whole layer's bytes in a reusable host `staging` buffer
/// (copying every still-resident expert's bytes from its CPU source, plus
/// the newly admitted one) and uploads it as a brand-new `QTensor`, which
/// [`Self::install_layer`] then swaps in. This is only ever called from the
/// seed (once per layer) or a rate-limited live swap (at most one per layer
/// per decay window), so rebuilding the whole layer rather than one slot is
/// cheap relative to the H2D upload it already requires.
///
/// Each layer has its own [`RwLock`] rather than one lock guarding the whole
/// pool: [`Self::pool_for_layer`] is the forward-pass hot path, called once
/// per layer per step, and unrelated layers are read concurrently. A
/// per-layer lock means those reads never contend with each other — only a
/// read and a concurrent write to the *same* layer would block, and
/// [`ExpertCache::decay_and_swap`]'s documented quiescence invariant (never
/// called concurrently with a forward pass) means that never happens in
/// practice either.
struct LayerPool {
    per_layer: Box<[RwLock<Option<Arc<QTensor>>>]>,
    /// Host staging buffer, sized to `slots_per_layer * max(blob_bytes)`
    /// (the largest per-layer cost, since layers can differ in GGML
    /// quantization type while sharing the same tensor shape) and reused
    /// across every [`Self::rebuild_layer`] call (seed and swap alike)
    /// rather than reallocated per call. A layer with a smaller per-expert
    /// cost than the max only uses a leading portion of the buffer;
    /// `qtensor_from_ggml` reads exactly as many bytes as that layer's own
    /// `blob_bytes`/`ggml_dtype` call for, ignoring the rest.
    staging: Mutex<AlignedBuf>,
    /// Per-expert byte cost, one entry per layer (layers can use different
    /// GGML quantization types, e.g. dynamic-quant GGUFs mixing `Q4_K`/`Q5_K`
    /// per layer, as long as every layer shares the same `expert_rows`/
    /// `expert_cols`).
    blob_bytes: Box<[usize]>,
    /// GGML quantization type, one entry per layer. See [`Self::blob_bytes`].
    ggml_dtype: Box<[GgmlDType]>,
    expert_rows: usize,
    expert_cols: usize,
    slots_per_layer: usize,
}

/// Byte buffer backed by `u64`-aligned storage. `qtensor_from_ggml` (via
/// `from_raw_data` in candle's `ggml_file.rs`) reinterprets a raw byte
/// buffer as `&[T]` for `GgmlDType`'s block type `T` (`f32`, `f16`/`bf16`,
/// or a quantized block struct), which is sound only if the buffer is
/// aligned to `align_of::<T>()`; a plain `Vec<u8>` carries no such
/// guarantee. 8 bytes covers every `T` `qtensor_from_ggml` supports.
struct AlignedBuf {
    words: Vec<u64>,
    byte_len: usize,
}

impl AlignedBuf {
    fn zeroed(byte_len: usize) -> Self {
        Self {
            words: vec![0u64; byte_len.div_ceil(8)],
            byte_len,
        }
    }
}

impl std::ops::Deref for AlignedBuf {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: `words` has no padding bytes, and `byte_len <=
        // words.len() * 8` by construction in `zeroed`.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr().cast::<u8>(), self.byte_len) }
    }
}

impl std::ops::DerefMut for AlignedBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: same invariant as `deref`.
        unsafe {
            std::slice::from_raw_parts_mut(self.words.as_mut_ptr().cast::<u8>(), self.byte_len)
        }
    }
}

/// Per-expert byte cost of a packed `[n_experts, rows, cols]` quantized
/// expert tensor, at `t`'s own GGML block encoding. Integer division is
/// exact: GGUF packing requires element counts to be whole multiples of the
/// quantization block size. Shared by [`ExpertCache::new`] (computing each
/// layer's own cost) and its callers (computing the max across layers to
/// size the VRAM budget) — layers can use different GGML quantization types
/// (e.g. dynamic-quant GGUFs mixing `Q4_K`/`Q5_K` per layer) as long as they
/// share the same `rows`/`cols`.
pub(crate) fn blob_bytes_per_expert(t: &QTensor) -> usize {
    let dims = t.shape().dims();
    let elems_per_expert = dims[1] * dims[2];
    elems_per_expert / t.dtype().block_size() * t.dtype().type_size()
}

impl LayerPool {
    /// `blob_bytes`/`ggml_dtype` must each have exactly `n_layers` entries,
    /// one per layer. Takes them by value: the caller (`ExpertCache::new`)
    /// builds fresh `Vec`s solely for this call and `LayerPool` owns them
    /// for its lifetime, so this is a move, not a clone.
    fn new(
        n_layers: usize,
        slots_per_layer: usize,
        blob_bytes: Vec<usize>,
        ggml_dtype: Vec<GgmlDType>,
        rows: usize,
        cols: usize,
    ) -> Self {
        debug_assert_eq!(blob_bytes.len(), n_layers);
        debug_assert_eq!(ggml_dtype.len(), n_layers);
        let max_blob_bytes = blob_bytes.iter().copied().max().unwrap_or(0);
        Self {
            per_layer: (0..n_layers).map(|_| RwLock::new(None)).collect(),
            staging: Mutex::new(AlignedBuf::zeroed(slots_per_layer * max_blob_bytes)),
            blob_bytes: blob_bytes.into_boxed_slice(),
            ggml_dtype: ggml_dtype.into_boxed_slice(),
            expert_rows: rows,
            expert_cols: cols,
            slots_per_layer,
        }
    }

    /// Recomposes layer `layer`'s packed pool tensor from `source`'s raw
    /// bytes, placing each `Gpu`-resident expert (per `residency`) at its
    /// assigned slot in the staging buffer, then uploads the whole buffer as
    /// a new `QTensor`. Does not install the result — see
    /// [`Self::install_layer`]. The staging buffer is reused across calls and
    /// never re-zeroed, so full slot coverage is load-bearing, not
    /// defensive: every call must write exactly `slots_per_layer` slots, or
    /// an uncovered slot would silently keep stale bytes left over from a
    /// previous, unrelated `rebuild_layer` call (possibly for a different
    /// layer, since the buffer and its mutex are shared across layers). In
    /// practice this holds because [`ExpertCache::new`]'s seeding loop always
    /// fills every slot before the first call, and every live swap keeps the
    /// slot count constant.
    ///
    /// Locks `staging` for the duration of the host copy plus the GPU
    /// upload. Safe because this only ever runs on the single-threaded
    /// scheduler loop between steps — see [`ExpertCache::decay_and_swap`]'s
    /// quiescence invariant, this method's only caller alongside
    /// [`ExpertCache::new`]'s one-time seed. If a previous call panicked
    /// while holding the lock, the lock is recovered rather than left
    /// poisoned: the mutex only guards buffer access, not any invariant that
    /// survives a panic, since every live slot is unconditionally
    /// overwritten below.
    ///
    /// # Errors
    ///
    /// Returns an error if `source`'s raw bytes can't be read, if an
    /// expert's byte range is out of bounds in `source`, if a residency slot
    /// index is out of bounds for the staging buffer, or if the staging
    /// buffer can't be reinterpreted as a `[slots_per_layer, expert_rows,
    /// expert_cols]` tensor of `ggml_dtype` on `device`.
    fn rebuild_layer(
        &self,
        source: &QTensor,
        residency: &[AtomicU32],
        layer: usize,
        n_experts: usize,
        device: &Device,
    ) -> Result<Arc<QTensor>> {
        let raw = source.data()?;
        let mut staging = self
            .staging
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let blob_bytes = self.blob_bytes[layer];
        let base = layer * n_experts;
        let mut slots_written = 0usize;
        for expert in 0..n_experts {
            let slot_val = residency[base + expert].load(Ordering::Relaxed);
            if slot_val == CPU_SENTINEL {
                continue;
            }
            if slot_val as usize >= self.slots_per_layer {
                candle_core::bail!(
                    "residency slot {slot_val} for expert {expert} out of range: \
                     {} slots per layer",
                    self.slots_per_layer,
                );
            }
            let src = self.expert_slice(&raw, layer, expert)?;
            let dst_start = slot_val as usize * blob_bytes;
            let dst_end = dst_start + blob_bytes;
            if dst_end > staging.len() {
                candle_core::bail!(
                    "residency slot {slot_val} for expert {expert} out of range: \
                     byte range {dst_start}..{dst_end} exceeds staging buffer \
                     ({} bytes, {} slots)",
                    staging.len(),
                    self.slots_per_layer,
                );
            }
            staging[dst_start..dst_end].copy_from_slice(src);
            slots_written += 1;
        }
        debug_assert_eq!(
            slots_written, self.slots_per_layer,
            "rebuild_layer must fill every slot: the staging buffer is reused \
             across calls and never re-zeroed, so a skipped slot would keep \
             stale bytes from an earlier call",
        );

        let tensor = qtensor_from_ggml(
            self.ggml_dtype[layer],
            &staging,
            vec![self.slots_per_layer, self.expert_rows, self.expert_cols],
            device,
        )?;
        Ok(Arc::new(tensor))
    }

    /// Installs `qt` as layer `layer`'s current pool tensor, replacing
    /// whatever was there before.
    fn install_layer(&self, layer: usize, qt: Arc<QTensor>) {
        *self.per_layer[layer]
            .write()
            .expect("LayerPool per_layer lock not poisoned") = Some(qt);
    }

    /// Current packed pool tensor for layer `layer`, shape
    /// `[slots_per_layer, expert_rows, expert_cols]`.
    ///
    /// # Panics
    ///
    /// Panics if `layer` hasn't been seeded yet. [`ExpertCache::new`] seeds
    /// every layer before any residency entry can report `Gpu`, so this
    /// never fires for a `layer` reached through [`ExpertCache::residency`].
    fn pool_for_layer(&self, layer: usize) -> Arc<QTensor> {
        Arc::clone(
            self.per_layer[layer]
                .read()
                .expect("LayerPool per_layer lock not poisoned")
                .as_ref()
                .expect("layer must be seeded before its pool is read"),
        )
    }

    /// Byte-slices `expert_idx` out of layer `layer`'s packed
    /// `[num_experts, rows, cols]` tensor's raw data, using that layer's own
    /// `blob_bytes`. Same formula as `moe.rs`'s `slice_packed_qtensor`, with
    /// the same out-of-bounds guard.
    ///
    /// # Errors
    ///
    /// Returns an error if the computed byte range exceeds `raw`'s length
    /// (a mismatched `blob_bytes`/`expert_idx`/source-tensor combination).
    fn expert_slice<'a>(&self, raw: &'a [u8], layer: usize, expert_idx: usize) -> Result<&'a [u8]> {
        let blob_bytes = self.blob_bytes[layer];
        let start = expert_idx * blob_bytes;
        let end = start + blob_bytes;
        if end > raw.len() {
            candle_core::bail!(
                "expert {expert_idx} byte range {start}..{end} exceeds raw data ({} bytes)",
                raw.len(),
            );
        }
        Ok(&raw[start..end])
    }
}

/// Per-expert GPU residency cache for one model's `MoE` layers.
///
/// Holds two [`LayerPool`]s (`gate_up` and down projections), a flat
/// `[n_layers * n_experts]` residency table, and per-expert hit counters
/// fed by [`Self::record_hit`]. [`Self::on_step`] drives the
/// warmup/aggressive/steady schedule that periodically decays counts and
/// swaps a cold GPU-resident expert for a hot CPU-resident one, using
/// thresholds from [`Self::tuning`] instead of hardcoded constants.
/// [`Self::record_fast_path`]/[`Self::record_mixed_path`] track how often
/// a dispatch actually avoided CPU work — the metric that decides whether
/// per-expert residency can speed up decode at all, as opposed to the
/// swap/decay mechanics alone.
///
/// Lives behind `Arc` once constructed: the forward path reads
/// [`Self::residency`]/calls [`Self::record_hit`] from inside model
/// layers, while the scheduler calls [`Self::on_step`] between steps. All
/// fields are `Send + Sync` accordingly — lock-free atomics for the
/// per-token hot path, a per-layer `RwLock` (see [`LayerPool`]) for the
/// rarely-touched slot storage, and a `Mutex` for phase bookkeeping.
pub struct ExpertCache {
    gate_up_pool: LayerPool,
    down_pool: LayerPool,
    /// Flat `[n_layers * n_experts]`; [`CPU_SENTINEL`] means `Cpu`,
    /// otherwise the value is a [`SlotIdx`] valid in both pools, scoped to
    /// the expert's layer.
    residency: Box<[AtomicU32]>,
    /// Flat `[n_layers * n_experts]` decaying routing-hit counters.
    hit_counts: Box<[AtomicU64]>,
    /// Calls to the dispatch classifier that found every routed expert
    /// GPU-resident (no CPU dispatch needed). Reset each window by
    /// [`Self::take_fast_path_stats`].
    fast_path_calls: AtomicU64,
    /// Calls to the dispatch classifier that found at least one routed
    /// expert CPU-resident. Reset each window by
    /// [`Self::take_fast_path_stats`].
    mixed_path_calls: AtomicU64,
    phase_state: Mutex<PhaseState>,
    /// Decay/swap decision thresholds; see [`SwapTuning`].
    tuning: SwapTuning,
    n_layers: usize,
    n_experts: usize,
    slots_per_layer: usize,
    /// CPU-resident packed per-layer source tensors, read from on a swap
    /// to fetch a newly hot expert's raw bytes.
    gate_up_sources: Vec<Arc<QTensor>>,
    down_sources: Vec<Arc<QTensor>>,
    device: Device,
}

/// Restores two `residency` entries to given pre-swap values on `Drop`
/// unless [`Self::disarm`] was called first. Guards [`ExpertCache::try_swap`]
/// against both an `Err` return and a panic unwind from either pool's
/// `rebuild_layer` call: in either case, Rust runs `Drop` impls of
/// in-scope locals while unwinding, so the rollback fires exactly when the
/// swap didn't complete, with no duplicated rollback logic in a `match`
/// arm.
struct ResidencyRollback<'a> {
    residency: &'a [AtomicU32],
    hot_idx: usize,
    evicted_idx: usize,
    pre_hot: u32,
    pre_evicted: u32,
    armed: bool,
}

impl ResidencyRollback<'_> {
    /// Marks the swap as having completed successfully, so `Drop` leaves
    /// the (now-correct) post-swap residency values in place.
    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for ResidencyRollback<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.residency[self.hot_idx].store(self.pre_hot, Ordering::Relaxed);
            self.residency[self.evicted_idx].store(self.pre_evicted, Ordering::Relaxed);
        }
    }
}

impl ExpertCache {
    /// Allocates both slot pools and seeds an arbitrary initial placement:
    /// each layer's first `slots_per_layer` experts by index, not ranked
    /// by any calibration pass — placement correctness comes from
    /// [`Self::decay_and_swap`] over live traffic, not from getting the
    /// seed right.
    ///
    /// `gate_up_exps`/`down_exps` must be CPU-resident packed `[n_experts,
    /// rows, cols]` tensors, one pair per `MoE` layer, in layer order.
    ///
    /// # Errors
    ///
    /// Returns an error if `gate_up_exps`/`down_exps` don't have exactly
    /// `budget.n_layers` entries, if either source tensor's shape isn't
    /// 3-dimensional, if any layer's shape diverges from layer 0's (`MoE`
    /// GGML quantization type may still vary per layer, e.g. a dynamic-quant
    /// GGUF mixing `Q4_K`/`Q5_K` across layers — only shape must be uniform), if
    /// reading a source tensor's raw bytes fails, or if a byte range can't
    /// be reinterpreted as a valid per-expert `QTensor` on `device`.
    #[allow(clippy::too_many_lines)] // validate, build pools, seed, construct
    pub fn new(
        budget: ExpertCacheBudget,
        gate_up_exps: Vec<Arc<QTensor>>,
        down_exps: Vec<Arc<QTensor>>,
        device: &Device,
    ) -> Result<Self> {
        let ExpertCacheBudget {
            slots_per_layer,
            n_layers,
            n_experts,
            ..
        } = budget;

        if gate_up_exps.len() != n_layers || down_exps.len() != n_layers {
            candle_core::bail!(
                "ExpertCache::new: gate_up_exps.len()={}, down_exps.len()={} must both equal budget.n_layers={n_layers}",
                gate_up_exps.len(),
                down_exps.len(),
            );
        }

        let gate_up_dims = gate_up_exps[0].shape().dims().to_vec();
        let down_dims = down_exps[0].shape().dims().to_vec();
        if gate_up_dims.len() != 3 || down_dims.len() != 3 {
            candle_core::bail!(
                "ExpertCache::new: expected 3D [n_experts, rows, cols] source tensors, got gate_up shape {gate_up_dims:?}, down shape {down_dims:?}",
            );
        }
        if let Some(i) = gate_up_exps
            .iter()
            .position(|t| t.shape().dims() != gate_up_dims)
        {
            candle_core::bail!(
                "ExpertCache::new: layer {i}'s gate_up shape {:?} diverges from layer 0's {gate_up_dims:?}",
                gate_up_exps[i].shape().dims(),
            );
        }
        if let Some(i) = down_exps.iter().position(|t| t.shape().dims() != down_dims) {
            candle_core::bail!(
                "ExpertCache::new: layer {i}'s down shape {:?} diverges from layer 0's {down_dims:?}",
                down_exps[i].shape().dims(),
            );
        }

        let gate_up_dtypes: Vec<GgmlDType> = gate_up_exps.iter().map(|t| t.dtype()).collect();
        let gate_up_blob_bytes: Vec<usize> = gate_up_exps
            .iter()
            .map(|t| blob_bytes_per_expert(t))
            .collect();
        let down_dtypes: Vec<GgmlDType> = down_exps.iter().map(|t| t.dtype()).collect();
        let down_blob_bytes: Vec<usize> =
            down_exps.iter().map(|t| blob_bytes_per_expert(t)).collect();

        let gate_up_pool = LayerPool::new(
            n_layers,
            slots_per_layer,
            gate_up_blob_bytes,
            gate_up_dtypes,
            gate_up_dims[1],
            gate_up_dims[2],
        );
        let down_pool = LayerPool::new(
            n_layers,
            slots_per_layer,
            down_blob_bytes,
            down_dtypes,
            down_dims[1],
            down_dims[2],
        );

        let residency: Box<[AtomicU32]> = (0..n_layers * n_experts)
            .map(|_| AtomicU32::new(CPU_SENTINEL))
            .collect();
        let hit_counts: Box<[AtomicU64]> = (0..n_layers * n_experts)
            .map(|_| AtomicU64::new(0))
            .collect();

        for layer in 0..n_layers {
            let base = layer * n_experts;
            for local_idx in 0..slots_per_layer {
                #[allow(clippy::cast_possible_truncation)]
                // Slot counts are bounded by VRAM capacity divided by a
                // ~1 MB-scale blob size — far below u32::MAX.
                let slot = SlotIdx(local_idx as u32);
                residency[base + local_idx].store(slot.0, Ordering::Relaxed);
            }

            let gu_tensor = gate_up_pool.rebuild_layer(
                &gate_up_exps[layer],
                &residency,
                layer,
                n_experts,
                device,
            )?;
            gate_up_pool.install_layer(layer, gu_tensor);

            let dn_tensor =
                down_pool.rebuild_layer(&down_exps[layer], &residency, layer, n_experts, device)?;
            down_pool.install_layer(layer, dn_tensor);
        }

        Ok(Self {
            gate_up_pool,
            down_pool,
            residency,
            hit_counts,
            fast_path_calls: AtomicU64::new(0),
            mixed_path_calls: AtomicU64::new(0),
            phase_state: Mutex::new(PhaseState {
                phase: Phase::Warmup,
                tokens_in_window: 0,
            }),
            tuning: SwapTuning::from_env(),
            n_layers,
            n_experts,
            slots_per_layer,
            gate_up_sources: gate_up_exps,
            down_sources: down_exps,
            device: device.clone(),
        })
    }

    /// Current residency of `(layer, expert)`. Lock-free; callers must not
    /// cache the result across a scheduler step boundary, since
    /// [`Self::decay_and_swap`] can change it between steps.
    #[must_use]
    pub fn residency(&self, layer: usize, expert: usize) -> Residency {
        let val = self.residency[layer * self.n_experts + expert].load(Ordering::Relaxed);
        if val == CPU_SENTINEL {
            Residency::Cpu
        } else {
            Residency::Gpu(SlotIdx(val))
        }
    }

    /// Records one routing hit for `(layer, expert)`. Lock-free; called on
    /// every real dispatch, not just during a calibration pass — see this
    /// module's doc comment for why counting is always on.
    pub fn record_hit(&self, layer: usize, expert: usize) {
        self.hit_counts[layer * self.n_experts + expert].fetch_add(1, Ordering::Relaxed);
    }

    /// Records one dispatch that found every routed expert GPU-resident, so
    /// no CPU fallback was needed. Lock-free. Called by the dispatch
    /// classifier (`moe.rs`'s `partition_by_residency`) once per `MoE`
    /// layer per forward step, alongside [`Self::record_mixed_path`].
    pub fn record_fast_path(&self) {
        self.fast_path_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Records one dispatch that found at least one routed expert
    /// CPU-resident, falling onto the mixed GPU/CPU path. Lock-free. See
    /// [`Self::record_fast_path`].
    pub fn record_mixed_path(&self) {
        self.mixed_path_calls.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns `(fast_path_calls, mixed_path_calls)` accumulated since the
    /// last call (or since construction), then resets both counters to 0.
    /// Used once per decay/swap window by [`Self::on_step`] so the logged
    /// rate reflects recent traffic, not a lifetime-diluted average.
    /// `pub(crate)` rather than private so `moe.rs`'s tests can verify
    /// [`Self::record_fast_path`]/[`Self::record_mixed_path`] are actually
    /// invoked by the dispatch classifier, not just unit-testable in
    /// isolation.
    pub(crate) fn take_fast_path_stats(&self) -> (u64, u64) {
        (
            self.fast_path_calls.swap(0, Ordering::Relaxed),
            self.mixed_path_calls.swap(0, Ordering::Relaxed),
        )
    }

    /// GPU-resident `gate_up` packed pool tensor for `layer`, shape
    /// `[slots_per_layer, rows, cols]`. Callers (the mixed GPU/CPU dispatch
    /// path) pass `Residency::Gpu(SlotIdx)` values from this layer as expert ids
    /// to `QTensor::indexed_moe_forward` on the returned tensor — never an
    /// original expert index, and never a `SlotIdx` from a different layer.
    #[must_use]
    pub fn gate_up_pool(&self, layer: usize) -> Arc<QTensor> {
        self.gate_up_pool.pool_for_layer(layer)
    }

    /// GPU-resident down packed pool tensor for `layer`. Same shape and
    /// `SlotIdx` scoping rules as [`Self::gate_up_pool`].
    #[must_use]
    pub fn down_pool(&self, layer: usize) -> Arc<QTensor> {
        self.down_pool.pool_for_layer(layer)
    }

    /// Number of GPU-resident slots per layer.
    #[must_use]
    pub const fn slots_per_layer(&self) -> usize {
        self.slots_per_layer
    }

    /// Number of experts per `MoE` layer.
    #[must_use]
    pub const fn n_experts(&self) -> usize {
        self.n_experts
    }

    /// Number of `MoE` layers this cache spans.
    #[must_use]
    pub const fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// Advances the token clock by `tokens_this_step` and runs
    /// [`Self::decay_and_swap`] if the current phase's window has closed.
    /// Owns the warmup/aggressive/steady phase transitions; callers (the
    /// scheduler's per-step loop) just report how many tokens this step
    /// produced. Must be called between scheduler steps, never from
    /// inside a forward pass.
    ///
    /// # Correctness
    ///
    /// Must be called by a single thread (the scheduler's step loop), not
    /// concurrently from multiple threads: it reads `phase_state`, releases
    /// the lock, calls [`Self::decay_and_swap`], then reacquires `phase_state`
    /// to record the transition. Concurrent callers could interleave between
    /// those two acquisitions and double-fire a decay/swap window. This
    /// invariant is doc-only, not type-enforced, since `ExpertCache` lives
    /// behind `Arc` for concurrent forward-pass reads; when wiring this
    /// into the scheduler, consider wrapping that `Arc<ExpertCache>` in a
    /// scheduler-owned newtype with `&mut self` on `on_step` so the
    /// single-caller invariant is enforced by the borrow checker instead.
    ///
    /// # Panics
    ///
    /// Panics if the internal phase-state mutex was poisoned by a prior
    /// panic while holding the lock.
    ///
    /// # Errors
    ///
    /// Returns an error if a triggered swap's weight upload fails (see
    /// [`Self::try_swap`]).
    pub fn on_step(&self, tokens_this_step: u64) -> Result<usize> {
        let phase = {
            let mut state = self
                .phase_state
                .lock()
                .expect("phase_state mutex not poisoned");
            state.tokens_in_window += tokens_this_step;
            if state.tokens_in_window < state.phase.window_tokens() {
                return Ok(0);
            }
            state.tokens_in_window = 0;

            if state.phase == Phase::Warmup {
                // Warmup's window closing just ends the no-swap grace
                // period; the first real decay/swap check happens on the
                // next (aggressive) window.
                state.phase = Phase::Aggressive { windows_elapsed: 0 };
                return Ok(0);
            }
            state.phase
        };

        let swaps = self.decay_and_swap()?;

        // Logs the metric that actually decides whether per-expert
        // residency can speed up decode: how often a dispatch avoided CPU
        // work entirely, not just whether swaps are converging. See this
        // module's doc comment and `moe.rs`'s `partition_by_residency`.
        let (fast, mixed) = self.take_fast_path_stats();
        let total = fast + mixed;
        if total > 0 {
            #[allow(clippy::cast_precision_loss)]
            // Call counts within one decay/swap window, far below f64's
            // exact-integer range in practice.
            let pct = fast as f64 / total as f64 * 100.0;
            log::info!("expert cache: fast-path {fast}/{total} calls ({pct:.1}%) this window");
        }

        let mut state = self
            .phase_state
            .lock()
            .expect("phase_state mutex not poisoned");
        match phase {
            Phase::Warmup => unreachable!("warmup handled above"),
            Phase::Aggressive { windows_elapsed } => {
                let next = windows_elapsed + 1;
                state.phase = if next >= self.tuning.aggressive_windows {
                    Phase::Steady
                } else {
                    Phase::Aggressive {
                        windows_elapsed: next,
                    }
                };
            },
            Phase::Steady => {
                #[allow(
                    clippy::cast_precision_loss,
                    clippy::cast_sign_loss,
                    clippy::cast_possible_truncation
                )]
                // n_layers is bounded by realistic model sizes (tens to
                // low hundreds), far below f64's exact-integer range.
                let threshold =
                    (self.n_layers as f64 * self.tuning.reentry_fraction).ceil() as usize;
                if swaps >= threshold {
                    log::info!(
                        "expert cache: {swaps}/{} layers swapped in one window, re-entering aggressive phase",
                        self.n_layers
                    );
                    state.phase = Phase::Aggressive { windows_elapsed: 0 };
                }
            },
        }
        Ok(swaps)
    }

    /// Halves all hit counts, then for each layer swaps its coldest
    /// GPU-resident expert for its hottest CPU-resident expert if the hot
    /// expert's count reaches [`SwapTuning::min_hot_count`] *and* the
    /// hot/cold ratio exceeds [`SwapTuning::swap_margin`]. The hit-count
    /// floor matters independently of the ratio: once a GPU-resident
    /// expert's count decays to 0, any nonzero CPU count would otherwise
    /// "exceed the margin" and swap on noise alone. At most one swap per
    /// layer per call. Returns the number of swaps performed.
    ///
    /// # Correctness
    ///
    /// Must be called between scheduler steps, never concurrent with a
    /// forward pass reading [`Self::residency`] or calling
    /// [`Self::gate_up_pool`]/[`Self::down_pool`]: [`Self::try_swap`]
    /// rebuilds both layer pool tensors and updates the residency table as
    /// separate steps, so a concurrent reader could observe a layer
    /// mid-swap (e.g. a rebuilt `gate_up` pool paired with a stale `down`
    /// pool). The scheduler's step boundary provides this quiescence
    /// guarantee — see [`Self::on_step`]'s doc comment.
    ///
    /// # Errors
    ///
    /// Returns an error if a swap's weight upload fails.
    pub fn decay_and_swap(&self) -> Result<usize> {
        for count in &self.hit_counts {
            let old = count.load(Ordering::Relaxed);
            count.store(old / 2, Ordering::Relaxed);
        }

        let mut total_swaps = 0;
        for layer in 0..self.n_layers {
            let base = layer * self.n_experts;
            let mut coldest_gpu: Option<(usize, u64, u32)> = None;
            let mut hottest_cpu: Option<(usize, u64)> = None;

            for expert in 0..self.n_experts {
                let count = self.hit_counts[base + expert].load(Ordering::Relaxed);
                let slot_val = self.residency[base + expert].load(Ordering::Relaxed);
                if slot_val == CPU_SENTINEL {
                    if hottest_cpu.is_none_or(|(_, hot)| count > hot) {
                        hottest_cpu = Some((expert, count));
                    }
                } else if coldest_gpu.is_none_or(|(_, cold, _)| count < cold) {
                    coldest_gpu = Some((expert, count, slot_val));
                }
            }

            if let (Some((cold_expert, cold_count, cold_slot)), Some((hot_expert, hot_count))) =
                (coldest_gpu, hottest_cpu)
            {
                #[allow(clippy::cast_precision_loss)]
                // Hit counts are decayed every window (halved), so they
                // stay far below f64's exact-integer range in practice.
                let exceeds_margin = hot_count >= self.tuning.min_hot_count
                    && (hot_count as f64) > (cold_count as f64) * self.tuning.swap_margin;
                if exceeds_margin {
                    self.try_swap(layer, SlotIdx(cold_slot), hot_expert, cold_expert)?;
                    total_swaps += 1;
                }
            }
        }
        Ok(total_swaps)
    }

    /// Rebuilds layer `layer`'s two pool tensors with `hot_expert` occupying
    /// `cold_slot` in place of `evicted_expert`, then installs both and
    /// marks `hot_expert` `Gpu(cold_slot)`, `evicted_expert` `Cpu`.
    ///
    /// # Correctness
    ///
    /// Same quiescence requirement as [`Self::decay_and_swap`], its only
    /// caller: `try_swap` rebuilds both layer pool tensors and the
    /// residency table as separate steps, so this must never run
    /// concurrently with a forward pass.
    ///
    /// # Errors
    ///
    /// Returns an error if either pool's rebuild fails (a bad byte range or
    /// a failed GPU upload). Residency is tentatively flipped before the
    /// rebuilds run (so [`LayerPool::rebuild_layer`] picks up the new
    /// mapping) and rolled back to its pre-swap state before either pool
    /// tensor is installed, whether a rebuild returns `Err` or panics — a
    /// failed or aborted swap leaves both pools and the residency table
    /// exactly as they were, no dead or mixed slot possible. See
    /// [`ResidencyRollback`].
    fn try_swap(
        &self,
        layer: usize,
        cold_slot: SlotIdx,
        hot_expert: usize,
        evicted_expert: usize,
    ) -> Result<()> {
        let base = layer * self.n_experts;
        let hot_idx = base + hot_expert;
        let evicted_idx = base + evicted_expert;

        let pre_hot = self.residency[hot_idx].load(Ordering::Relaxed);
        let pre_evicted = self.residency[evicted_idx].load(Ordering::Relaxed);

        self.residency[hot_idx].store(cold_slot.0, Ordering::Relaxed);
        self.residency[evicted_idx].store(CPU_SENTINEL, Ordering::Relaxed);
        let rollback = ResidencyRollback {
            residency: &self.residency,
            hot_idx,
            evicted_idx,
            pre_hot,
            pre_evicted,
            armed: true,
        };

        let new_gu = self.gate_up_pool.rebuild_layer(
            &self.gate_up_sources[layer],
            &self.residency,
            layer,
            self.n_experts,
            &self.device,
        )?;
        let new_dn = self.down_pool.rebuild_layer(
            &self.down_sources[layer],
            &self.residency,
            layer,
            self.n_experts,
            &self.device,
        )?;
        rollback.disarm();

        self.gate_up_pool.install_layer(layer, new_gu);
        self.down_pool.install_layer(layer, new_dn);
        log::debug!(
            "expert cache: layer {layer}: slot {} now expert {hot_expert} (evicted expert {evicted_expert})",
            cold_slot.0
        );
        Ok(())
    }

    /// Current phase, for test observability only.
    #[cfg(test)]
    fn current_phase(&self) -> Phase {
        self.phase_state
            .lock()
            .expect("phase_state mutex not poisoned")
            .phase
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    /// Builds a packed `[n_experts, rows, cols]` `QTensor` where every
    /// element of expert `i` is `(i + 1) as f32`, quantized losslessly
    /// (`GgmlDType::F32`, block_size=1) so values round-trip exactly.
    fn make_packed_experts(n_experts: usize, rows: usize, cols: usize) -> Arc<QTensor> {
        make_packed_experts_as(n_experts, rows, cols, GgmlDType::F32)
    }

    /// Like [`make_packed_experts`], but quantized to an explicit `dtype`
    /// instead of always `F32`. Used to build test fixtures with
    /// per-layer-varying `GgmlDType` (e.g. mimicking a dynamic-quant GGUF).
    fn make_packed_experts_as(
        n_experts: usize,
        rows: usize,
        cols: usize,
        dtype: GgmlDType,
    ) -> Arc<QTensor> {
        let data: Vec<f32> = (0..n_experts * rows * cols)
            .map(|idx| ((idx / (rows * cols)) + 1) as f32)
            .collect();
        let tensor = Tensor::from_vec(data, (n_experts, rows, cols), &Device::Cpu).expect("tensor");
        Arc::new(QTensor::quantize(&tensor, dtype).expect("quantize"))
    }

    /// Per-expert byte cost of a `[rows, cols]` F32 blob (block_size=1, so
    /// this is just `rows * cols * type_size()`). `n_experts` is unused by
    /// the formula but kept as a parameter so call sites read the same as
    /// `make_packed_experts`'s shape.
    fn blob_bytes(_n_experts: usize, rows: usize, cols: usize) -> usize {
        rows * cols * GgmlDType::F32.type_size() / GgmlDType::F32.block_size()
    }

    fn make_cache(
        n_layers: usize,
        n_experts: usize,
        rows: usize,
        cols: usize,
        slots_per_layer: usize,
    ) -> ExpertCache {
        let gate_up_exps: Vec<_> = (0..n_layers)
            .map(|_| make_packed_experts(n_experts, rows, 2 * cols))
            .collect();
        let down_exps: Vec<_> = (0..n_layers)
            .map(|_| make_packed_experts(n_experts, cols, rows))
            .collect();

        let gu_blob = blob_bytes(n_experts, rows, 2 * cols);
        let dn_blob = blob_bytes(n_experts, cols, rows);

        let budget = ExpertCacheBudget {
            raw_budget_slots: n_layers * slots_per_layer,
            slots_per_layer,
            gate_up_blob_bytes: gu_blob,
            down_blob_bytes: dn_blob,
            n_layers,
            n_experts,
        };
        ExpertCache::new(budget, gate_up_exps, down_exps, &Device::Cpu).expect("ExpertCache::new")
    }

    // Verifies budget math for a simple exact-division case.
    #[test]
    fn test_budget_basic() {
        let budget = ExpertCacheBudget::compute(1_000_000, 0, 100, 100, 4, 8).expect("budget");
        assert_eq!(budget.raw_budget_slots, 5_000);
        assert_eq!(budget.slots_per_layer, 8); // 5_000 / 4 = 1_250, clamped to n_experts=8
    }

    // Verifies compute() refuses a budget too small for 1 slot per layer.
    #[test]
    fn test_budget_returns_none_insufficient() {
        let budget = ExpertCacheBudget::compute(100, 0, 1000, 1000, 10, 8);
        assert!(budget.is_none());
    }

    // Verifies slots_per_layer never exceeds n_experts even when the raw
    // VRAM budget would otherwise allow more slots than experts exist.
    #[test]
    fn test_budget_clamps_to_n_experts() {
        let budget = ExpertCacheBudget::compute(1_000_000, 0, 10, 10, 2, 4).expect("budget");
        assert!(budget.slots_per_layer <= 4);
    }

    // Verifies ExpertCache::new returns an error (not a panic) when
    // gate_up_exps/down_exps don't have exactly budget.n_layers entries.
    #[test]
    fn test_new_rejects_layer_count_mismatch() {
        let n_layers = 2;
        let n_experts = 4;
        let rows = 4;
        let cols = 8;
        let slots_per_layer = 2;
        // Only n_layers tensors, but the budget below claims n_layers + 1.
        let gate_up_exps: Vec<_> = (0..n_layers)
            .map(|_| make_packed_experts(n_experts, rows, 2 * cols))
            .collect();
        let down_exps: Vec<_> = (0..n_layers)
            .map(|_| make_packed_experts(n_experts, cols, rows))
            .collect();
        let budget = ExpertCacheBudget {
            raw_budget_slots: (n_layers + 1) * slots_per_layer,
            slots_per_layer,
            gate_up_blob_bytes: blob_bytes(n_experts, rows, 2 * cols),
            down_blob_bytes: blob_bytes(n_experts, cols, rows),
            n_layers: n_layers + 1,
            n_experts,
        };
        let result = ExpertCache::new(budget, gate_up_exps, down_exps, &Device::Cpu);
        assert!(result.is_err());
    }

    // Verifies ExpertCache::new returns an error when a source tensor isn't
    // 3-dimensional, instead of panicking on an out-of-bounds dims index.
    #[test]
    fn test_new_rejects_non_3d_shape() {
        let bad_shape = Tensor::from_vec(vec![1.0f32; 16], (4, 4), &Device::Cpu).expect("tensor");
        let bad_gate_up =
            Arc::new(QTensor::quantize(&bad_shape, GgmlDType::F32).expect("quantize"));
        let down_exps = vec![make_packed_experts(4, 4, 8)];
        let budget = ExpertCacheBudget {
            raw_budget_slots: 2,
            slots_per_layer: 2,
            gate_up_blob_bytes: 16 * GgmlDType::F32.type_size(),
            down_blob_bytes: blob_bytes(4, 8, 4),
            n_layers: 1,
            n_experts: 4,
        };
        let result = ExpertCache::new(budget, vec![bad_gate_up], down_exps, &Device::Cpu);
        assert!(result.is_err());
    }

    // Verifies ExpertCache::new rejects a layer whose tensor shape diverges
    // from layer 0's, even though per-layer GGML dtype is allowed to vary
    // (see test_seed_with_per_layer_varying_dtype) -- shape must still be
    // uniform across layers.
    #[test]
    fn test_new_rejects_shape_mismatch_across_layers() {
        let n_experts = 4;
        let rows = 4;
        let cols = 8;
        let slots_per_layer = 2;
        let gate_up_exps = vec![
            make_packed_experts(n_experts, rows, 2 * cols),
            make_packed_experts(n_experts, rows + 1, 2 * cols), // shape mismatch
        ];
        let down_exps = vec![
            make_packed_experts(n_experts, cols, rows),
            make_packed_experts(n_experts, cols, rows),
        ];
        let budget = ExpertCacheBudget {
            raw_budget_slots: 2 * slots_per_layer,
            slots_per_layer,
            gate_up_blob_bytes: blob_bytes(n_experts, rows, 2 * cols),
            down_blob_bytes: blob_bytes(n_experts, cols, rows),
            n_layers: 2,
            n_experts,
        };
        let result = ExpertCache::new(budget, gate_up_exps, down_exps, &Device::Cpu);
        assert!(result.is_err());
    }

    // Verifies LayerPool::expert_slice returns an error rather than panicking
    // when the computed byte range exceeds the raw data's length.
    #[test]
    fn test_expert_slice_rejects_out_of_bounds() {
        let pool = LayerPool::new(1, 1, vec![100], vec![GgmlDType::F32], 4, 8);
        let raw = vec![0u8; 50];
        assert!(pool.expert_slice(&raw, 0, 0).is_err());
    }

    // Verifies the initial seed marks the first `slots_per_layer` experts
    // per layer Gpu and the rest Cpu.
    #[test]
    fn test_seed_residency() {
        let cache = make_cache(2, 6, 4, 8, 3);
        for layer in 0..2 {
            for expert in 0..3 {
                assert!(matches!(cache.residency(layer, expert), Residency::Gpu(_)));
            }
            for expert in 3..6 {
                assert_eq!(cache.residency(layer, expert), Residency::Cpu);
            }
        }
    }

    // Verifies seeded slot indices are layer-local (0..slots_per_layer
    // within each layer's own pool tensor), not a global offset across
    // layers.
    #[test]
    fn test_seed_slot_assignment() {
        let cache = make_cache(2, 4, 4, 8, 2);
        assert_eq!(cache.residency(0, 0), Residency::Gpu(SlotIdx(0)));
        assert_eq!(cache.residency(0, 1), Residency::Gpu(SlotIdx(1)));
        assert_eq!(cache.residency(1, 0), Residency::Gpu(SlotIdx(0)));
        assert_eq!(cache.residency(1, 1), Residency::Gpu(SlotIdx(1)));
    }

    // Verifies record_hit increments the correct counter.
    #[test]
    fn test_record_hit() {
        let cache = make_cache(1, 4, 4, 8, 2);
        for _ in 0..5 {
            cache.record_hit(0, 2);
        }
        assert_eq!(cache.hit_counts[2].load(Ordering::Relaxed), 5);
    }

    // Verifies decay_and_swap halves every counter (floor division).
    #[test]
    fn test_decay_halves() {
        let cache = make_cache(1, 4, 4, 8, 2);
        for _ in 0..7 {
            cache.record_hit(0, 0);
        }
        cache.decay_and_swap().expect("decay_and_swap");
        assert_eq!(cache.hit_counts[0].load(Ordering::Relaxed), 3);
    }

    // Verifies record_fast_path/record_mixed_path increment independently.
    #[test]
    fn test_fast_and_mixed_path_counters() {
        let cache = make_cache(1, 4, 4, 8, 2);
        cache.record_fast_path();
        cache.record_fast_path();
        cache.record_mixed_path();
        assert_eq!(cache.fast_path_calls.load(Ordering::Relaxed), 2);
        assert_eq!(cache.mixed_path_calls.load(Ordering::Relaxed), 1);
    }

    // Verifies take_fast_path_stats returns the accumulated counts and
    // resets both counters to 0, so the next window's log line reflects
    // only traffic since the last read.
    #[test]
    fn test_take_fast_path_stats_resets() {
        let cache = make_cache(1, 4, 4, 8, 2);
        cache.record_fast_path();
        cache.record_mixed_path();
        cache.record_mixed_path();
        assert_eq!(cache.take_fast_path_stats(), (1, 2));
        assert_eq!(cache.take_fast_path_stats(), (0, 0));
    }

    // Verifies env_or falls back to its default when the key is unset.
    // Uses a key guaranteed absent from the test process's environment
    // rather than std::env::set_var, so this can't race cargo test's
    // parallel test threads.
    #[test]
    fn test_env_or_default_when_unset() {
        assert!((env_or::<f64>("CRANE_MOE_TEST_DEFINITELY_UNSET_KEY", 2.0) - 2.0).abs() < 1e-9);
    }

    // Verifies a CPU expert well above SwapTuning::swap_margin over the coldest GPU
    // expert gets promoted, evicting the cold one. slots_per_layer=1 so
    // there is exactly one (deterministic) coldest GPU expert to compare
    // against — with more GPU slots, an untouched one would sit at count
    // 0 regardless of what this test hits, masking the margin check.
    #[test]
    fn test_swap_triggers_above_margin() {
        let cache = make_cache(1, 4, 4, 8, 1);
        // Expert 0 is Gpu-seeded; 1,2,3 are Cpu.
        for _ in 0..100 {
            cache.record_hit(0, 3); // hottest cpu
        }
        cache.decay_and_swap().expect("decay_and_swap");
        assert!(matches!(cache.residency(0, 3), Residency::Gpu(_)));
        assert_eq!(cache.residency(0, 0), Residency::Cpu);
    }

    // Verifies no swap happens when the hot/cold ratio is below
    // SwapTuning::swap_margin. slots_per_layer=1 for the same reason as
    // test_swap_triggers_above_margin.
    #[test]
    fn test_swap_skipped_below_margin() {
        let cache = make_cache(1, 4, 4, 8, 1);
        for _ in 0..10 {
            cache.record_hit(0, 0); // the lone gpu expert
        }
        for _ in 0..10 {
            cache.record_hit(0, 2); // hottest cpu, but not >2x the gpu count
        }
        cache.decay_and_swap().expect("decay_and_swap");
        assert!(matches!(cache.residency(0, 0), Residency::Gpu(_)));
        assert_eq!(cache.residency(0, 2), Residency::Cpu);
    }

    // Verifies a swap is skipped when the hottest CPU expert's count is
    // nonzero but below SwapTuning::min_hot_count, even though a 0-count
    // coldest GPU expert would otherwise make any nonzero count "exceed
    // the margin" (hot > 0 * margin holds for any hot > 0). Reproduces the
    // real-traffic thrashing this floor exists to stop: 25%+ of layers
    // swapping every steady-state window on single-digit noise.
    #[test]
    fn test_swap_skipped_below_min_hot_count() {
        let cache = make_cache(1, 4, 4, 8, 1);
        // Expert 0 is Gpu-seeded (0 hits, untouched); 1,2,3 are Cpu.
        for _ in 0..6 {
            cache.record_hit(0, 3); // decays to 3, below MIN_HOT_COUNT_DEFAULT=4
        }
        cache.decay_and_swap().expect("decay_and_swap");
        assert!(matches!(cache.residency(0, 0), Residency::Gpu(_)));
        assert_eq!(cache.residency(0, 3), Residency::Cpu);
    }

    // Verifies at most one swap happens per layer per decay_and_swap call,
    // even when multiple CPU experts are hot enough to qualify.
    #[test]
    fn test_swap_rate_limit() {
        let cache = make_cache(1, 4, 4, 8, 1);
        // Expert 0 is Gpu-seeded; 1,2,3 are Cpu.
        for _ in 0..100 {
            cache.record_hit(0, 2);
            cache.record_hit(0, 3);
        }
        let swaps = cache.decay_and_swap().expect("decay_and_swap");
        assert_eq!(swaps, 1);
    }

    // Verifies the phase transitions Warmup -> Aggressive once WARMUP_TOKENS
    // have accumulated.
    #[test]
    fn test_warmup_to_aggressive() {
        let cache = make_cache(1, 4, 4, 8, 2);
        assert_eq!(cache.current_phase(), Phase::Warmup);
        let swaps = cache.on_step(WARMUP_TOKENS).expect("on_step");
        assert_eq!(swaps, 0);
        assert_eq!(
            cache.current_phase(),
            Phase::Aggressive { windows_elapsed: 0 }
        );
    }

    // Verifies the phase transitions Aggressive -> Steady after
    // SwapTuning::aggressive_windows windows have closed.
    #[test]
    fn test_aggressive_to_steady() {
        let cache = make_cache(1, 4, 4, 8, 2);
        cache.on_step(WARMUP_TOKENS).expect("warmup");
        for _ in 0..AGGRESSIVE_WINDOWS_DEFAULT {
            cache
                .on_step(AGGRESSIVE_WINDOW_TOKENS)
                .expect("aggressive window");
        }
        assert_eq!(cache.current_phase(), Phase::Steady);
    }

    // Verifies a steady-state window with swaps on a large fraction of
    // layers re-enters the aggressive phase.
    #[test]
    fn test_steady_reentry() {
        let cache = make_cache(4, 4, 4, 8, 1);
        cache.on_step(WARMUP_TOKENS).expect("warmup");
        for _ in 0..AGGRESSIVE_WINDOWS_DEFAULT {
            cache
                .on_step(AGGRESSIVE_WINDOW_TOKENS)
                .expect("aggressive window");
        }
        assert_eq!(cache.current_phase(), Phase::Steady);

        // Force every layer's Cpu expert far hotter than its Gpu expert.
        for layer in 0..4 {
            for _ in 0..1000 {
                cache.record_hit(layer, 2);
            }
        }
        cache.on_step(STEADY_WINDOW_TOKENS).expect("steady window");
        assert_eq!(
            cache.current_phase(),
            Phase::Aggressive { windows_elapsed: 0 }
        );
    }

    // Verifies a seeded slot's dequantized data matches its source expert,
    // sliced out of the per-layer pool tensor at the slot's row.
    #[test]
    fn test_slot_data_correctness() {
        let cache = make_cache(1, 4, 4, 8, 2);
        let Residency::Gpu(slot) = cache.residency(0, 1) else {
            panic!("expected expert 1 to be Gpu-resident");
        };
        let vals = cache
            .gate_up_pool(0)
            .dequantize(&Device::Cpu)
            .expect("dequantize")
            .narrow(0, slot.0 as usize, 1)
            .expect("narrow")
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1");
        assert!(vals.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    // Verifies a seeded slot's raw bytes exactly match its source expert's
    // bytes (byte-for-byte, not just dequantized values) -- the Step 4
    // "Verify" bar: upload a known expert, read back, byte-compare.
    #[test]
    fn test_seed_byte_exact() {
        let n_experts = 4;
        let rows = 4;
        let cols = 8;
        let cache = make_cache(1, n_experts, rows, cols, 2);
        let Residency::Gpu(slot) = cache.residency(0, 1) else {
            panic!("expected expert 1 to be Gpu-resident");
        };
        let gu_blob = blob_bytes(n_experts, rows, 2 * cols);

        let pool = cache.gate_up_pool(0);
        let pool_raw = pool.data().expect("pool data");
        let slot_start = slot.0 as usize * gu_blob;
        let actual = &pool_raw[slot_start..slot_start + gu_blob];

        let source_raw = cache.gate_up_sources[0].data().expect("source data");
        let expected = &source_raw[gu_blob..2 * gu_blob]; // expert 1

        assert_eq!(actual, expected);
    }

    // Verifies a live swap's rebuilt slot byte-matches the newly swapped-in
    // expert's source bytes, not the evicted expert's stale data -- the
    // Step 4 "Verify" bar's second half: the same check, for one live swap.
    #[test]
    fn test_swap_byte_exact() {
        let n_experts = 4;
        let rows = 4;
        let cols = 8;
        let cache = make_cache(1, n_experts, rows, cols, 1);
        // Expert 0 is Gpu-seeded; 1,2,3 are Cpu.
        for _ in 0..100 {
            cache.record_hit(0, 3); // hottest cpu
        }
        cache.decay_and_swap().expect("decay_and_swap");
        let Residency::Gpu(slot) = cache.residency(0, 3) else {
            panic!("expected expert 3 to be Gpu-resident after swap");
        };
        let gu_blob = blob_bytes(n_experts, rows, 2 * cols);

        let pool = cache.gate_up_pool(0);
        let pool_raw = pool.data().expect("pool data");
        let slot_start = slot.0 as usize * gu_blob;
        let actual = &pool_raw[slot_start..slot_start + gu_blob];

        let source_raw = cache.gate_up_sources[0].data().expect("source data");
        let expected = &source_raw[3 * gu_blob..4 * gu_blob]; // expert 3

        assert_eq!(actual, expected);
    }

    // Verifies ExpertCache::new accepts layers with different GGML
    // quantization types (e.g. a dynamic-quant GGUF mixing Q4_K/Q5_K per
    // layer) as long as shape is uniform, and seeds each layer's pool bytes
    // from that layer's own dtype -- not layer 0's.
    #[test]
    fn test_seed_with_per_layer_varying_dtype() {
        let n_experts = 4;
        let rows = 4;
        let cols = 8;
        let slots_per_layer = 2;

        let gate_up_exps = vec![
            make_packed_experts_as(n_experts, rows, 2 * cols, GgmlDType::F32),
            make_packed_experts_as(n_experts, rows, 2 * cols, GgmlDType::F16),
        ];
        let down_exps = vec![
            make_packed_experts_as(n_experts, cols, rows, GgmlDType::F32),
            make_packed_experts_as(n_experts, cols, rows, GgmlDType::F16),
        ];
        let gu_blob_max =
            blob_bytes_per_expert(&gate_up_exps[0]).max(blob_bytes_per_expert(&gate_up_exps[1]));
        let dn_blob_max =
            blob_bytes_per_expert(&down_exps[0]).max(blob_bytes_per_expert(&down_exps[1]));
        let budget = ExpertCacheBudget {
            raw_budget_slots: 2 * slots_per_layer,
            slots_per_layer,
            gate_up_blob_bytes: gu_blob_max,
            down_blob_bytes: dn_blob_max,
            n_layers: 2,
            n_experts,
        };
        let cache = ExpertCache::new(budget, gate_up_exps, down_exps, &Device::Cpu)
            .expect("ExpertCache::new must accept per-layer-varying dtype");

        // Seeding fills slot `local_idx` with expert `local_idx` for
        // `local_idx < slots_per_layer` (see ExpertCache::new's seed loop).
        for layer in 0..2 {
            let blob = blob_bytes_per_expert(&cache.gate_up_sources[layer]);
            let pool = cache.gate_up_pool(layer);
            let pool_raw = pool.data().expect("pool data");
            let source_raw = cache.gate_up_sources[layer].data().expect("source data");
            for local_idx in 0..slots_per_layer {
                let actual = &pool_raw[local_idx * blob..(local_idx + 1) * blob];
                let expected = &source_raw[local_idx * blob..(local_idx + 1) * blob];
                assert_eq!(
                    actual, expected,
                    "layer {layer} slot {local_idx} bytes mismatch"
                );
            }
        }
    }

    // Live-swap counterpart to test_seed_with_per_layer_varying_dtype:
    // verifies a swap on a non-layer-0 dtype (F16) correctly rebuilds using
    // that layer's own dtype/blob_bytes, not layer 0's (F32).
    #[test]
    fn test_swap_preserves_per_layer_dtype() {
        let n_experts = 4;
        let rows = 4;
        let cols = 8;
        let slots_per_layer = 1;

        let gate_up_exps = vec![
            make_packed_experts_as(n_experts, rows, 2 * cols, GgmlDType::F32),
            make_packed_experts_as(n_experts, rows, 2 * cols, GgmlDType::F16),
        ];
        let down_exps = vec![
            make_packed_experts_as(n_experts, cols, rows, GgmlDType::F32),
            make_packed_experts_as(n_experts, cols, rows, GgmlDType::F16),
        ];
        let gu_blob_max =
            blob_bytes_per_expert(&gate_up_exps[0]).max(blob_bytes_per_expert(&gate_up_exps[1]));
        let dn_blob_max =
            blob_bytes_per_expert(&down_exps[0]).max(blob_bytes_per_expert(&down_exps[1]));
        let budget = ExpertCacheBudget {
            raw_budget_slots: 2 * slots_per_layer,
            slots_per_layer,
            gate_up_blob_bytes: gu_blob_max,
            down_blob_bytes: dn_blob_max,
            n_layers: 2,
            n_experts,
        };
        let cache = ExpertCache::new(budget, gate_up_exps, down_exps, &Device::Cpu)
            .expect("ExpertCache::new");

        // Layer 1 (F16): expert 0 is Gpu-seeded; 1,2,3 are Cpu. Force expert 3 hot.
        for _ in 0..100 {
            cache.record_hit(1, 3);
        }
        cache.decay_and_swap().expect("decay_and_swap");
        let Residency::Gpu(slot) = cache.residency(1, 3) else {
            panic!("expected expert 3 to be Gpu-resident on layer 1 after swap");
        };

        let gu_blob = blob_bytes_per_expert(&cache.gate_up_sources[1]);
        let pool = cache.gate_up_pool(1);
        let pool_raw = pool.data().expect("pool data");
        let slot_start = slot.0 as usize * gu_blob;
        let actual = &pool_raw[slot_start..slot_start + gu_blob];

        let source_raw = cache.gate_up_sources[1].data().expect("source data");
        let expected = &source_raw[3 * gu_blob..4 * gu_blob]; // expert 3

        assert_eq!(actual, expected);
    }

    // Verifies try_swap rolls residency back to its pre-swap state if the
    // down pool's rebuild fails after the gate_up pool's rebuild already
    // succeeded -- no dead slot, no silently-mismatched pools.
    #[test]
    fn test_swap_rollback_on_partial_failure() {
        let n_experts = 4;
        let rows = 4;
        let cols = 8;
        let mut cache = make_cache(1, n_experts, rows, cols, 1);
        // Corrupt the down source to one expert short, so expert 3's byte
        // range is out of bounds and rebuild_layer fails for the down pool
        // only (gate_up's source is untouched and rebuilds fine).
        cache.down_sources[0] = make_packed_experts(n_experts - 1, cols, rows);

        // Expert 0 is Gpu-seeded; 1,2,3 are Cpu.
        for _ in 0..100 {
            cache.record_hit(0, 3); // would otherwise evict expert 0
        }
        assert!(cache.decay_and_swap().is_err());
        assert!(matches!(cache.residency(0, 0), Residency::Gpu(_)));
        assert_eq!(cache.residency(0, 3), Residency::Cpu);
    }

    // Verifies the staging buffer is reused (same allocation) across the
    // seed and a subsequent live swap, not reallocated per rebuild.
    #[test]
    fn test_staging_buffer_reused() {
        let cache = make_cache(1, 4, 4, 8, 2);
        let ptr_before = cache
            .gate_up_pool
            .staging
            .lock()
            .expect("staging lock")
            .as_ptr();

        for _ in 0..100 {
            cache.record_hit(0, 3);
        }
        cache.decay_and_swap().expect("decay_and_swap");

        let ptr_after = cache
            .gate_up_pool
            .staging
            .lock()
            .expect("staging lock")
            .as_ptr();
        assert_eq!(ptr_before, ptr_after);
    }
}
