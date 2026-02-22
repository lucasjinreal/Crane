pub mod image_utils;
pub mod tokenizer_utils;
pub mod token_output_stream;
pub mod utils;

use anyhow::Result;
use candle_core::{
    utils::{cuda_is_available as candle_cuda_is_available, metal_is_available},
    Device,
};

pub fn select_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

pub fn cuda_is_available() -> bool {
    candle_cuda_is_available()
}
