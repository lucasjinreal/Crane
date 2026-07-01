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

#[cfg(not(feature = "cuda"))]
fn nvml_gpu_memory_info(_index: usize) -> Option<GpuMemoryInfo> {
    None
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

#[cfg(not(feature = "cuda"))]
fn nvidia_smi_gpu_memory_info(_index: usize) -> Option<GpuMemoryInfo> {
    None
}
