pub mod cast_var_builder;
pub mod image_utils;
pub mod prof;
pub mod quantized_var_builder;
pub mod sycl_env;
pub mod token_output_stream;
pub mod tokenizer_utils;
pub mod utils;

use anyhow::Result;
use candle_core::{
    Device,
    utils::{cuda_is_available as candle_cuda_is_available, metal_is_available},
};

pub fn select_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        // Intel oneAPI / SYCL (proof-of-concept). Only reachable when built
        // `--features sycl` against the candle fork that provides the backend.
        #[cfg(feature = "sycl")]
        if candle_core::utils::sycl_is_available() {
            return Ok(Device::new_sycl(0)?);
        }
        Ok(Device::Cpu)
    }
}

pub fn cuda_is_available() -> bool {
    candle_cuda_is_available()
}

/// Adds ROCm/HIP and SYCL device detection to `Device`.
pub trait DeviceExt {
    /// Returns `true` if this device is a ROCm/HIP device.
    fn is_rocm(&self) -> bool;
    /// Returns `true` if this device is an Intel oneAPI/SYCL device.
    fn is_sycl(&self) -> bool;
}

impl DeviceExt for Device {
    fn is_rocm(&self) -> bool {
        false
    }

    fn is_sycl(&self) -> bool {
        false
    }
}
