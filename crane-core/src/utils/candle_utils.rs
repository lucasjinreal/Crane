use anyhow::Result;
use candle_core::{
    utils::{cuda_is_available, metal_is_available},
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
