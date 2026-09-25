// SPDX-License-Identifier: MIT

//! Device assignment and reusable VRAM-accounting primitives for models
//! with offloadable sub-components (e.g. MoE experts).

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

/// Fallback sequence length used for KV budgeting when `max_seq_len` is
/// unset or `0` (unlimited).
pub const DEFAULT_KV_SEQ_LEN: usize = 4096;

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
/// Used at model-load time to reserve VRAM headroom for the KV cache
/// before deciding how much is left for other uses (e.g. MoE expert
/// placement). Inverse: [`kv_budget_from_headroom`].
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

impl From<&Device> for DeviceAssignment {
    fn from(device: &Device) -> Self {
        Self::uniform(device)
    }
}

impl From<&DeviceAssignment> for DeviceAssignment {
    fn from(devices: &DeviceAssignment) -> Self {
        devices.clone()
    }
}

/// Live-queries `(free_bytes, total_bytes)` for `device`.
///
/// `None` for CPU, or for a GPU backend with no query support here (e.g.
/// Metal) — callers should fall back to a static estimate in that case.
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
pub fn format_budget(bytes: u64) -> String {
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

    // Verifies `From<&Device>` matches `uniform`, so callers that only have
    // a plain `&Device` can rely on `Into<DeviceAssignment>` conversion.
    #[test]
    fn from_device_matches_uniform() {
        let device = Device::Cpu;
        let assignment: DeviceAssignment = (&device).into();
        assert!(assignment.main.same_device(&assignment.expert));
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
    // layer further along can still fit in the leftover space.
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
