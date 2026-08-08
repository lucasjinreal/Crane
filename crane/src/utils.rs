use crate::common::config::DeviceConfig;

#[cfg(feature = "cuda")]
use std::process::Command;

#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub used_bytes: u64,
    pub total_bytes: u64,
}

pub fn gpu_memory_info(device: &DeviceConfig) -> Option<GpuMemoryInfo> {
    match device {
        DeviceConfig::Cuda(index) => cuda_gpu_memory_info(*index as usize),
        DeviceConfig::Rocm(index) => rocm_gpu_memory_info(*index as usize),
        _ => None,
    }
}

#[cfg(feature = "cuda")]
fn cuda_gpu_memory_info(index: usize) -> Option<GpuMemoryInfo> {
    nvml_gpu_memory_info(index).or_else(|| nvidia_smi_gpu_memory_info(index))
}

#[cfg(not(feature = "cuda"))]
fn cuda_gpu_memory_info(_index: usize) -> Option<GpuMemoryInfo> {
    None
}

#[cfg(feature = "cuda")]
fn nvml_gpu_memory_info(index: usize) -> Option<GpuMemoryInfo> {
    let nvml = nvml_wrapper::Nvml::init().ok()?;
    let device = nvml.device_by_index(index as u32).ok()?;
    let mem = device.memory_info().ok()?;
    Some(GpuMemoryInfo {
        used_bytes: mem.used,
        total_bytes: mem.total,
    })
}

#[cfg(feature = "cuda")]
fn nvidia_smi_gpu_memory_info(index: usize) -> Option<GpuMemoryInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
            "-i",
            &index.to_string(),
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8(output.stdout)
        .ok()?
        .lines()
        .next()?
        .trim()
        .to_string();
    let mut parts = line.split(',').map(|v| v.trim().parse::<u64>().ok());
    let used_mb = parts.next()??;
    let total_mb = parts.next()??;
    Some(GpuMemoryInfo {
        used_bytes: used_mb * 1024 * 1024,
        total_bytes: total_mb * 1024 * 1024,
    })
}

// Go through candle's `rocm_rs` re-export rather than depending on `rocm-rs`
// directly, so there is only ever one copy of the HIP bindings linked in.
//
// `hipMemGetInfo` (and thus `memory_info()`) reports on whichever device is
// current for this thread, not an arbitrary index — same limitation the
// crane-serve engine lives with, so `index` is accepted for interface parity
// with the CUDA path but unused here.
#[cfg(feature = "rocm")]
fn rocm_gpu_memory_info(_index: usize) -> Option<GpuMemoryInfo> {
    let info = candle_core::rocm_backend::rocm_rs::hip::memory_info().ok()?;
    Some(GpuMemoryInfo {
        used_bytes: (info.total - info.free) as u64,
        total_bytes: info.total as u64,
    })
}

#[cfg(not(feature = "rocm"))]
fn rocm_gpu_memory_info(_index: usize) -> Option<GpuMemoryInfo> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_memory_info_returns_none_for_cpu() {
        assert!(gpu_memory_info(&DeviceConfig::Cpu).is_none());
    }

    #[test]
    fn gpu_memory_info_returns_none_for_metal() {
        assert!(gpu_memory_info(&DeviceConfig::Metal).is_none());
    }

    // Hardware-independent: without the `rocm` feature the ROCm query path is
    // compiled out entirely, so this is deterministic in any build environment.
    #[cfg(not(feature = "rocm"))]
    #[test]
    fn gpu_memory_info_rocm_is_none_without_rocm_feature() {
        assert!(gpu_memory_info(&DeviceConfig::Rocm(0)).is_none());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn gpu_memory_info_cuda_is_none_without_cuda_feature() {
        assert!(gpu_memory_info(&DeviceConfig::Cuda(0)).is_none());
    }
}
