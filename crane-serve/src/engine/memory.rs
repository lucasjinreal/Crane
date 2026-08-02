//! GPU memory accounting: limit parsing and live usage queries.
//!
//! Split out of `engine/mod.rs` (which was well past the workspace's 400-line
//! file cap) — this is the one section of that file with no dependency on
//! `InferenceEngine` itself, so it lifts out cleanly.

use candle_core::Device;

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
    ///   - Absolute sizes: "5G", "8GB", "12GiB", "5120M", "5120MiB", "5368709120" (bytes)
    ///   - Utilization fraction: "0.7" (70% of total GPU memory)
    #[must_use]
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

    /// Suffixes accepted by [`Self::parse_memory_limit`], longest first so a
    /// suffix like "GIB" isn't shadowed by a same-letter prefix like "G".
    /// `format_bytes_engine` below renders sizes with binary-prefix math under
    /// decimal-looking labels ("G" = 2^30), so every suffix here — decimal
    /// spelling ("GB") included — is treated as power-of-two too, to match
    /// what operators see reflected back in our own logs.
    const SIZE_SUFFIXES: [(&'static str, u32); 6] =
        [("GIB", 30), ("GB", 30), ("G", 30), ("MIB", 20), ("MB", 20), ("M", 20)];

    fn parse_memory_limit(s: &str, device: &Device) -> u64 {
        let s = s.trim();
        if s.is_empty() || s == "0" {
            return 0;
        }

        // Try absolute sizes: "5G", "8GB", "12GiB", "5120M", "5120MiB"
        let upper = s.to_uppercase();
        for (suffix, shift) in Self::SIZE_SUFFIXES {
            if let Some(digits) = upper.strip_suffix(suffix)
                && let Ok(n) = digits.trim().parse::<f64>()
            {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    clippy::cast_precision_loss
                )]
                return (n * (1u64 << shift) as f64) as u64;
            }
        }

        // Try as a fraction (0.0 - 1.0)
        if let Ok(frac) = s.parse::<f64>() {
            if (0.0..=1.0).contains(&frac) {
                let total = Self::query_total_gpu_memory(device);
                if total > 0 {
                    #[allow(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        clippy::cast_precision_loss
                    )]
                    return (frac * total as f64) as u64;
                }
            }
            // If > 1.0, treat as bytes
            if frac > 1.0 {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                return frac as u64;
            }
        }

        tracing::warn!("Could not parse gpu_memory_limit '{s}', ignoring");
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
        #[cfg(feature = "rocm")]
        {
            if let Device::Rocm(_) = _device
                && let Ok(info) = candle_core::rocm_backend::rocm_rs::hip::memory_info()
            {
                return info.total as u64;
            }
        }
        0
    }
}

/// Query current GPU memory usage. Returns (`used_bytes`, `total_bytes`).
/// Returns (0, 0) if the device is neither CUDA nor ROCm (or the query fails).
pub(super) fn query_gpu_memory_usage(_device: &Device) -> (u64, u64) {
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
    #[cfg(feature = "rocm")]
    {
        if let Device::Rocm(_) = _device
            && let Ok(info) = candle_core::rocm_backend::rocm_rs::hip::memory_info()
        {
            return ((info.total - info.free) as u64, info.total as u64);
        }
    }
    (0, 0)
}

/// Format a byte count as a human-readable string (used in engine log messages).
pub(super) fn format_bytes_engine(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        #[allow(clippy::cast_precision_loss)]
        return format!("{:.1}G", bytes as f64 / (1u64 << 30) as f64);
    }
    if bytes >= 1 << 20 {
        #[allow(clippy::cast_precision_loss)]
        return format!("{:.0}M", bytes as f64 / (1u64 << 20) as f64);
    }
    format!("{bytes}B")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn parses_gib_suffix() {
        assert_eq!(MemoryConfig::parse_memory_limit("12GiB", &cpu()), 12 * (1u64 << 30));
    }

    #[test]
    fn parses_gb_suffix() {
        assert_eq!(MemoryConfig::parse_memory_limit("8GB", &cpu()), 8 * (1u64 << 30));
    }

    #[test]
    fn parses_bare_g_suffix() {
        assert_eq!(MemoryConfig::parse_memory_limit("5G", &cpu()), 5 * (1u64 << 30));
    }

    #[test]
    fn parses_mib_suffix() {
        assert_eq!(MemoryConfig::parse_memory_limit("512MiB", &cpu()), 512 * (1u64 << 20));
    }

    #[test]
    fn parses_mb_suffix() {
        assert_eq!(MemoryConfig::parse_memory_limit("5120MB", &cpu()), 5120 * (1u64 << 20));
    }

    #[test]
    fn parses_bare_m_suffix() {
        assert_eq!(MemoryConfig::parse_memory_limit("5120M", &cpu()), 5120 * (1u64 << 20));
    }

    #[test]
    fn parses_suffix_case_insensitively() {
        assert_eq!(MemoryConfig::parse_memory_limit("12gib", &cpu()), 12 * (1u64 << 30));
    }

    #[test]
    fn parses_fraction_as_zero_on_cpu() {
        // No CUDA/ROCm device to query a total from, so a fraction resolves
        // to 0 (unlimited) rather than a bogus non-zero value.
        assert_eq!(MemoryConfig::parse_memory_limit("0.9", &cpu()), 0);
    }

    #[test]
    fn parses_bare_bytes() {
        assert_eq!(MemoryConfig::parse_memory_limit("5368709120", &cpu()), 5_368_709_120);
    }

    #[test]
    fn empty_and_zero_are_unlimited() {
        assert_eq!(MemoryConfig::parse_memory_limit("", &cpu()), 0);
        assert_eq!(MemoryConfig::parse_memory_limit("0", &cpu()), 0);
    }

    #[test]
    fn rejects_garbage_and_warns() {
        assert_eq!(MemoryConfig::parse_memory_limit("not-a-size", &cpu()), 0);
    }

    #[test]
    fn format_bytes_engine_rounds_to_binary_units() {
        assert_eq!(format_bytes_engine(1u64 << 30), "1.0G");
        assert_eq!(format_bytes_engine(1u64 << 20), "1M");
        assert_eq!(format_bytes_engine(512), "512B");
    }
}
