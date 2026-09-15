// SPDX-License-Identifier: MIT

//! Device assignment and VRAM budgeting for models with offloadable
//! sub-components (e.g. MoE experts).

use candle_core::Device;

/// Fixed margin for allocator fragmentation and small runtime allocations,
/// on top of KV-tensor storage. Does not scale with batch size or hidden
/// dim, so it does not cover prefill/decode activation memory.
///
/// Defined here (rather than in `crane-serve`, which owns the runtime
/// KV-eviction budget that also uses this margin) because `crane-core`
/// cannot depend on `crane-serve` but `crane-serve` can and does depend on
/// `crane-core` — this is the only direction that lets both sides share a
/// single constant instead of drifting copies.
pub const KV_SAFETY_MARGIN_BYTES: u64 = 256 * (1 << 20);

/// Multiplier applied to raw per-sequence KV storage to account for the
/// transient overlap during batched-decode setup, where old per-sequence
/// KV caches and the newly built padded batch buffer coexist in VRAM
/// simultaneously before the old caches are dropped (see
/// `InferenceEngine::step_decode_batch`'s call into
/// `ModelBackend::setup_batch_decode`). With a single concurrent sequence
/// there is nothing to pad against, so the real cost is close to 1x raw
/// storage; with more than one, doubling covers the overlap when
/// concurrent sequences have roughly similar lengths.
///
/// This is a heuristic for the common case, not a worst-case bound: with
/// heavily skewed sequence lengths (one near `max_seq_len`, several much
/// shorter ones sharing the same decode step), the padded buffer can
/// approach `max_concurrent` times the longest sequence, exceeding this
/// factor. Scaling the factor with `max_concurrent` to cover that case
/// would reintroduce the over-reservation this function's callers exist
/// to avoid (see git history — the flat `6x` factor this replaced).
/// `InferenceEngine::is_over_kv_budget`'s `cuMemGetInfo` hard-safety
/// check is the backstop for skewed-length cases where this heuristic
/// underestimates.
#[must_use]
pub const fn kv_batch_factor(max_concurrent: usize) -> u64 {
    if max_concurrent > 1 { 2 } else { 1 }
}

/// Estimate total VRAM consumed by KV caches from raw KV-tensor bytes:
/// `raw_kv_bytes` scaled by [`kv_batch_factor`] for batched-decode
/// overlap, plus [`KV_SAFETY_MARGIN_BYTES`] for allocator fragmentation.
///
/// The single source of truth for this formula — [`GpuBudget::runtime_reservation_bytes`]
/// and `crane-serve`'s KV-eviction budget both call this instead of each
/// keeping their own copy. Inverse: [`kv_budget_from_headroom`].
#[must_use]
pub const fn kv_vram_overhead(raw_kv_bytes: u64, max_concurrent: usize) -> u64 {
    raw_kv_bytes * kv_batch_factor(max_concurrent) + KV_SAFETY_MARGIN_BYTES
}

/// Inverse of [`kv_vram_overhead`]: given available VRAM headroom, compute
/// how many raw KV-cache bytes can be stored within it. Subtracts
/// [`KV_SAFETY_MARGIN_BYTES`] and divides by [`kv_batch_factor`].
#[must_use]
pub const fn kv_budget_from_headroom(headroom_bytes: u64, max_concurrent: usize) -> u64 {
    headroom_bytes.saturating_sub(KV_SAFETY_MARGIN_BYTES) / kv_batch_factor(max_concurrent)
}

/// Bundles the primary inference device with the device MoE expert weights
/// load onto.
///
/// Two adjacent same-typed `&Device` parameters invite an unchecked swap at
/// call sites; bundling them into named fields makes that swap a compile-time
/// impossibility instead of a silent bug.
#[derive(Debug, Clone)]
pub struct DeviceAssignment {
    /// Device for model weights and inference (everything but MoE experts).
    pub main: Device,
    /// Device for MoE expert weights. Same as `main` when expert offloading
    /// is not needed; ignored by models and formats without MoE experts.
    pub expert: Device,
}

impl DeviceAssignment {
    /// All weights, including MoE experts, on the same device.
    pub fn uniform(device: &Device) -> Self {
        Self {
            main: device.clone(),
            expert: device.clone(),
        }
    }
}

/// VRAM budget available for model weights, used to decide per-layer
/// expert placement during model loading.
///
/// Constructed from `--gpu-memory-limit` and `--offload-experts` in
/// `crane-serve` and threaded through to `Qwen3Backend`. Consumed by
/// `Qwen3Model::from_gguf()`'s loading path to decide which `MoE` layers
/// load expert weights to GPU vs CPU.
#[derive(Debug, Clone, Default)]
pub struct GpuBudget {
    /// VRAM ceiling for model weights.
    pub weight_budget: WeightBudget,
    /// When `true`, force all `MoE` expert weights to CPU regardless of
    /// `weight_budget` (`--offload-experts`).
    pub offload_all_experts: bool,
    /// Maximum concurrent sequences the engine will serve, used to estimate
    /// KV cache VRAM at model-load time. `None` when the caller has no
    /// CLI-configured value; [`Self::runtime_reservation_bytes`] then falls
    /// back to a conservative default.
    pub max_concurrent: Option<usize>,
    /// Maximum tokens (prompt + completion) per sequence, used to estimate
    /// KV cache VRAM at model-load time. `None` or `Some(0)` means
    /// unlimited; [`Self::runtime_reservation_bytes`] then falls back to a
    /// conservative default.
    pub max_seq_len: Option<usize>,
}

impl GpuBudget {
    /// No GPU available (`--cpu` mode): all weights, including experts,
    /// must load to CPU.
    #[must_use]
    pub fn cpu() -> Self {
        Self {
            weight_budget: WeightBudget::NoGpu,
            offload_all_experts: false,
            ..Self::default()
        }
    }

    /// Derives a budget from a device for callers with no CLI-configured
    /// VRAM ceiling: [`Self::cpu()`] when `device` is a CPU device,
    /// otherwise an unlimited GPU budget.
    #[must_use]
    pub fn for_device(device: &Device) -> Self {
        if device.is_cpu() {
            Self::cpu()
        } else {
            Self::default()
        }
    }

    /// Estimated VRAM bytes consumed at runtime by `MoE` model KV caches and
    /// a safety margin, given model geometry. Used to subtract runtime needs
    /// from [`WeightBudget::Limited`] before deciding per-layer expert
    /// placement. Returns `0` for [`WeightBudget::NoGpu`] or
    /// [`WeightBudget::Unlimited`], since there is no weight budget to
    /// subtract from.
    ///
    /// Computes raw per-sequence KV storage, then calls [`kv_vram_overhead`]
    /// to account for batched-decode overlap and the safety margin; it is
    /// not a plain tensor-size calculation.
    ///
    /// `num_layers`, `num_kv_heads`, and `head_dim` come from the model's
    /// own config; `dtype_bytes` should always be the compute dtype's
    /// `size_in_bytes()`, even for a quantized KV cache whose fused
    /// dequantize-in-attention kernel (`crate::ops::fused_ops::quant_attn`)
    /// doesn't yet cover the sequence's entire lifetime on the active
    /// backend — batch decode, KV-swap/preemption, and unfused prefill/decode
    /// all fully dequantize to the compute dtype in that case, so reserving
    /// at the smaller quantized storage size would under-claim VRAM against
    /// that worst case. Callers that *can* rely on the fused kernel for the
    /// full lifetime (CUDA/ROCm, `max_concurrent == 1`) should call
    /// [`Self::runtime_reservation_bytes_for_kv_bytes_per_token`] instead,
    /// passing `Config::quantized_kv_bytes_per_token`'s result.
    /// [`Self::max_concurrent`] defaults to `1` and [`Self::max_seq_len`]
    /// defaults to `4096` when unset, since the real values may not be
    /// known yet at the point this is called.
    #[must_use]
    pub fn runtime_reservation_bytes(
        &self,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype_bytes: usize,
    ) -> u64 {
        let kv_bytes_per_token =
            2 * num_layers as u64 * num_kv_heads as u64 * head_dim as u64 * dtype_bytes as u64;
        self.runtime_reservation_bytes_for_kv_bytes_per_token(kv_bytes_per_token)
    }

    /// Same estimate as [`Self::runtime_reservation_bytes`], but takes an
    /// already-computed per-sequence, per-token KV byte cost directly —
    /// for callers pricing a quantized KV cache
    /// (`Config::quantized_kv_bytes_per_token`), whose cost doesn't
    /// decompose into a uniform per-element `dtype_bytes` factor (the
    /// per-token scale overhead doesn't scale with `head_dim`).
    #[must_use]
    pub fn runtime_reservation_bytes_for_kv_bytes_per_token(&self, kv_bytes_per_token: u64) -> u64 {
        const DEFAULT_SEQ_LEN: usize = 4096;

        if matches!(
            self.weight_budget,
            WeightBudget::NoGpu | WeightBudget::Unlimited
        ) {
            return 0;
        }

        let max_concurrent = self.max_concurrent.unwrap_or(1);
        let effective_seq_len = self
            .max_seq_len
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_SEQ_LEN) as u64;

        let kv_storage = max_concurrent as u64 * effective_seq_len * kv_bytes_per_token;

        kv_vram_overhead(kv_storage, max_concurrent)
    }
}

/// Live-queries `(free_bytes, total_bytes)` for `device`.
///
/// `None` for CPU, or for a GPU backend with no query support here (e.g.
/// Metal) — callers should fall back to a static estimate in that case.
/// Mirrors `crane-serve/src/engine/memory.rs`'s `query_gpu_memory_usage`,
/// duplicated rather than reused: `crane-core` cannot depend on
/// `crane-serve` (`crane-serve` → `crane` → `crane-core` is the only
/// allowed direction).
#[must_use]
pub fn query_gpu_memory(_device: &Device) -> Option<(u64, u64)> {
    #[cfg(feature = "cuda")]
    {
        if let Device::Cuda(_) = _device
            && let Ok((free, total)) =
                candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
        {
            return Some((free as u64, total as u64));
        }
    }
    #[cfg(feature = "rocm")]
    {
        if let Device::Rocm(_) = _device
            && let Ok(info) = candle_core::rocm_backend::rocm_rs::hip::memory_info()
        {
            return Some((info.free as u64, info.total as u64));
        }
    }
    None
}

/// Greedily selects layer indices (in order) whose cumulative
/// `layer_costs` fit within `budget`.
///
/// First-fit in index order: earlier layers are preferred when not
/// everything fits, matching the existing placement heuristic (early
/// layers on GPU, later layers fall back to CPU).
#[must_use]
pub fn greedy_fit_layers(layer_costs: &[u64], budget: u64) -> Vec<usize> {
    let mut remaining = budget;
    let mut selected = Vec::new();
    for (idx, &cost) in layer_costs.iter().enumerate() {
        if cost <= remaining {
            remaining -= cost;
            selected.push(idx);
        }
    }
    selected
}

/// Formats a byte count for log messages (e.g. `"8.5G"`, `"512M"`, `"1024B"`).
pub(crate) fn format_budget(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        // Byte counts are far below f64's 52-bit mantissa limit.
        #[allow(clippy::cast_precision_loss)]
        let gb = bytes as f64 / (1u64 << 30) as f64;
        format!("{gb:.1}G")
    } else if bytes >= 1 << 20 {
        #[allow(clippy::cast_precision_loss)]
        let mb = bytes as f64 / (1u64 << 20) as f64;
        format!("{mb:.0}M")
    } else {
        format!("{bytes}B")
    }
}

/// VRAM ceiling for model weights.
///
/// Distinguishes "no GPU device" from "GPU present with no configured
/// limit" so callers can't conflate the two by reading a bare `None`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum WeightBudget {
    /// No GPU device available (`--cpu` mode): all weights, including
    /// experts, must load to CPU.
    NoGpu,
    /// GPU present, no configured VRAM ceiling: prefer GPU for everything.
    #[default]
    Unlimited,
    /// GPU present with a VRAM budget, in bytes, for model weights (total
    /// VRAM minus estimated runtime needs like KV cache and activations).
    Limited(u64),
}

#[cfg(test)]
mod tests {
    use super::*;

    // Verifies `uniform` assigns the same device to both fields.
    #[test]
    fn uniform_assigns_same_device_to_both_fields() {
        let device = Device::Cpu;
        let assignment = DeviceAssignment::uniform(&device);
        assert!(assignment.main.same_device(&assignment.expert));
    }

    // Verifies `cpu()` reports no GPU device and does not force offload.
    #[test]
    fn gpu_budget_cpu_mode() {
        let budget = GpuBudget::cpu();
        assert_eq!(budget.weight_budget, WeightBudget::NoGpu);
        assert!(!budget.offload_all_experts);
    }

    // Verifies the default budget is unlimited and does not force offload.
    #[test]
    fn gpu_budget_default_no_limit() {
        let budget = GpuBudget::default();
        assert_eq!(budget.weight_budget, WeightBudget::Unlimited);
        assert!(!budget.offload_all_experts);
    }

    // Verifies `cpu()` and `default()` are distinct, unlike the old
    // `Option<usize>`-based encoding where both collapsed to `None`.
    #[test]
    fn gpu_budget_cpu_and_default_are_distinct() {
        assert_ne!(
            GpuBudget::cpu().weight_budget,
            GpuBudget::default().weight_budget
        );
    }

    // Verifies `for_device` picks `NoGpu` for a CPU device and `Unlimited`
    // otherwise, so a caller with no CLI budget doesn't silently claim GPU
    // headroom while actually running on CPU.
    #[test]
    fn gpu_budget_for_device_matches_device_kind() {
        assert_eq!(
            GpuBudget::for_device(&Device::Cpu).weight_budget,
            WeightBudget::NoGpu
        );
    }

    // Verifies the KV-cache + safety-margin formula with explicit
    // max_concurrent/max_seq_len values. max_concurrent=8 > 1, so the
    // batch-padding factor of 2 applies.
    #[test]
    fn runtime_reservation_bytes_with_explicit_values() {
        let budget = GpuBudget {
            weight_budget: WeightBudget::Limited(16 << 30),
            offload_all_experts: false,
            max_concurrent: Some(8),
            max_seq_len: Some(2048),
        };
        let kv_storage: u64 = 8 * 2048 * 2 * 48 * 4 * 128 * 2;
        let expected = kv_storage * 2 + KV_SAFETY_MARGIN_BYTES;
        assert_eq!(budget.runtime_reservation_bytes(48, 4, 128, 2), expected);
    }

    // Verifies unset max_concurrent/max_seq_len fall back to conservative
    // defaults (1 concurrent sequence, 4096-token horizon) rather than
    // underestimating the reservation as zero. max_concurrent defaults to
    // 1, so no batch-padding factor applies.
    #[test]
    fn runtime_reservation_bytes_defaults_when_unset() {
        let budget = GpuBudget {
            weight_budget: WeightBudget::Limited(16 << 30),
            offload_all_experts: false,
            max_concurrent: None,
            max_seq_len: None,
        };
        let kv_storage: u64 = 1 * 4096 * 2 * 48 * 4 * 128 * 2;
        let expected = kv_storage + KV_SAFETY_MARGIN_BYTES;
        assert_eq!(budget.runtime_reservation_bytes(48, 4, 128, 2), expected);
    }

    // Verifies no runtime reservation is computed when there is no weight
    // budget to subtract it from.
    #[test]
    fn runtime_reservation_bytes_zero_for_no_gpu_and_unlimited() {
        assert_eq!(GpuBudget::cpu().runtime_reservation_bytes(48, 4, 128, 2), 0);
        assert_eq!(
            GpuBudget::default().runtime_reservation_bytes(48, 4, 128, 2),
            0
        );
    }

    // Verifies a large explicit max_seq_len (e.g. a long-context deployment)
    // is not silently capped at the 4096 unset-fallback default, which
    // would under-reserve KV-cache VRAM and risk a runtime GPU OOM.
    // max_concurrent=1, so no batch-padding factor applies.
    #[test]
    fn runtime_reservation_bytes_respects_large_seq_len() {
        let budget = GpuBudget {
            weight_budget: WeightBudget::Limited(16 << 30),
            offload_all_experts: false,
            max_concurrent: Some(1),
            max_seq_len: Some(32768),
        };
        let expected_kv_bytes: u64 = 1 * 32768 * 2 * 48 * 4 * 128 * 2;
        let expected = expected_kv_bytes + KV_SAFETY_MARGIN_BYTES;
        assert_eq!(budget.runtime_reservation_bytes(48, 4, 128, 2), expected);
    }

    // Verifies the batch-padding factor: 1 for max_concurrent in {0, 1}
    // (nothing to pad against), 2 for anything greater (worst case: every
    // slot padded to the longest).
    #[test]
    fn kv_batch_factor_thresholds_at_one() {
        assert_eq!(kv_batch_factor(0), 1);
        assert_eq!(kv_batch_factor(1), 1);
        assert_eq!(kv_batch_factor(2), 2);
        assert_eq!(kv_batch_factor(16), 2);
    }

    // Verifies kv_budget_from_headroom is the exact inverse of
    // kv_vram_overhead across the batch-factor threshold, so the two
    // formulas used by crane-core (load-time reservation) and
    // crane-serve (runtime KV budget) cannot silently drift apart.
    #[test]
    fn kv_overhead_round_trip() {
        for &n in &[0usize, 1, 2, 8, 16] {
            let raw: u64 = 1_000_000;
            let overhead = kv_vram_overhead(raw, n);
            assert_eq!(kv_budget_from_headroom(overhead, n), raw);
        }
    }

    // Verifies the safety margin is fully present even when raw KV
    // storage is negligible (max_concurrent=1, max_seq_len=1) rather than
    // being silently absorbed or truncated.
    #[test]
    fn runtime_reservation_bytes_includes_full_margin_with_minimal_kv() {
        let budget = GpuBudget {
            weight_budget: WeightBudget::Limited(1 << 30),
            offload_all_experts: false,
            max_concurrent: Some(1),
            max_seq_len: Some(1),
        };
        let kv_storage: u64 = 1 * 1 * 2 * 48 * 4 * 128 * 2;
        let expected = kv_storage + KV_SAFETY_MARGIN_BYTES;
        assert_eq!(budget.runtime_reservation_bytes(48, 4, 128, 2), expected);
    }

    // Verifies runtime_reservation_bytes_for_kv_bytes_per_token computes the
    // same result as runtime_reservation_bytes when fed the equivalent
    // per-token byte cost (2 * layers * kv_heads * head_dim * dtype_bytes) —
    // the two must agree since the latter is defined in terms of the former.
    #[test]
    fn runtime_reservation_bytes_for_kv_bytes_per_token_matches_decomposed_form() {
        let budget = GpuBudget {
            weight_budget: WeightBudget::Limited(16 << 30),
            offload_all_experts: false,
            max_concurrent: Some(8),
            max_seq_len: Some(2048),
        };
        let kv_bytes_per_token: u64 = 2 * 48 * 4 * 128 * 2;
        assert_eq!(
            budget.runtime_reservation_bytes_for_kv_bytes_per_token(kv_bytes_per_token),
            budget.runtime_reservation_bytes(48, 4, 128, 2)
        );
    }

    // A quantized per-token byte cost (int8: ~head_dim + 4 bytes/head,
    // instead of head_dim * 2 for f16) must yield a strictly smaller
    // reservation than compute-dtype pricing — this is the whole point of
    // pricing at the quantized size once the fused kernel covers the full
    // sequence lifetime.
    #[test]
    fn runtime_reservation_bytes_for_kv_bytes_per_token_quantized_is_smaller() {
        let budget = GpuBudget {
            weight_budget: WeightBudget::Limited(16 << 30),
            offload_all_experts: false,
            max_concurrent: Some(1),
            max_seq_len: Some(131_072),
        };
        let fp_bytes_per_token: u64 = 2 * 48 * 4 * 128 * 2; // f16
        let int8_bytes_per_token: u64 = 2 * 48 * 4 * (128 + 4); // codes + scale
        let fp_reservation =
            budget.runtime_reservation_bytes_for_kv_bytes_per_token(fp_bytes_per_token);
        let int8_reservation =
            budget.runtime_reservation_bytes_for_kv_bytes_per_token(int8_bytes_per_token);
        assert!(
            int8_reservation < fp_reservation,
            "int8 reservation ({int8_reservation}) should be smaller than fp ({fp_reservation})"
        );
    }

    // Verifies the human-readable size formatting used in placement logs.
    #[test]
    fn test_format_budget() {
        assert_eq!(format_budget(1 << 30), "1.0G");
        assert_eq!(format_budget(3 * (1 << 30) / 2), "1.5G");
        assert_eq!(format_budget(1 << 20), "1M");
        assert_eq!(format_budget(512), "512B");
    }

    // Verifies every layer is selected when the budget comfortably covers
    // the total cost.
    #[test]
    fn greedy_fit_layers_all_fit() {
        assert_eq!(greedy_fit_layers(&[10, 20, 30], 100), vec![0, 1, 2]);
    }

    // Verifies first-fit-in-order: earlier layers are preferred, but a
    // layer that doesn't fit is skipped (not a hard stop) — a smaller
    // layer further along can still fit in the leftover space. Matches
    // the original inline loop this was extracted from
    // (`Qwen3Model::from_gguf`), which has no `break` on a miss.
    #[test]
    fn greedy_fit_layers_skips_layers_that_do_not_fit() {
        assert_eq!(greedy_fit_layers(&[10, 10, 10, 1], 25), vec![0, 1, 3]);
    }

    // Verifies a zero budget selects nothing.
    #[test]
    fn greedy_fit_layers_zero_budget_selects_none() {
        assert_eq!(greedy_fit_layers(&[1, 2, 3], 0), Vec::<usize>::new());
    }

    // Verifies a single layer costing exactly the budget is still selected
    // (boundary: `cost <= remaining`, not `cost < remaining`).
    #[test]
    fn greedy_fit_layers_exact_fit_boundary() {
        assert_eq!(greedy_fit_layers(&[50], 50), vec![0]);
    }

    // Verifies an empty cost list selects nothing regardless of budget.
    #[test]
    fn greedy_fit_layers_empty_costs() {
        assert_eq!(greedy_fit_layers(&[], 1000), Vec::<usize>::new());
    }
}
