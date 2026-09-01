//! A [`VarBuilder`] backend that converts checkpoint weights to the compute
//! dtype on the CPU, avoiding a redundant on-device staging buffer per weight.

use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::VarBuilder;
use candle_nn::var_builder::SimpleBackend;

/// Wraps a safetensors collection so [`SimpleBackend::get`] casts on the CPU
/// and uploads once. See the module docs for why this matters.
pub struct CpuCastSafetensors {
    inner: candle_core::safetensors::MmapedSafetensors,
}

impl CpuCastSafetensors {
    /// Loads `name` to the CPU, converts it there, then moves it to `dev`.
    fn load_cast(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let cpu = self.inner.load(name, &Device::Cpu)?;
        let cast = cpu.to_dtype(dtype)?;
        // Drop the stored-dtype copy before uploading rather than chaining.
        drop(cpu);
        cast.to_device(dev)
    }
}

impl SimpleBackend for CpuCastSafetensors {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.load_cast(name, dtype, dev)?;
        if tensor.shape() != &s {
            candle_core::bail!(
                "shape mismatch for {name}, expected {s:?}, got {:?}",
                tensor.shape()
            )
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.load_cast(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.get(name).is_ok()
    }
}

/// Drop-in replacement for [`VarBuilder::from_mmaped_safetensors`] that
/// converts to `dtype` on the CPU rather than on `device`. See module docs.
///
/// # Safety
///
/// Same contract as [`VarBuilder::from_mmaped_safetensors`]: files are
/// memory-mapped and must not change while the result is alive.
pub unsafe fn from_mmaped_safetensors_cpu_cast<'a, P: AsRef<std::path::Path>>(
    paths: &[P],
    dtype: DType,
    device: &Device,
) -> candle_core::Result<VarBuilder<'a>> {
    let inner = unsafe { candle_core::safetensors::MmapedSafetensors::multi(paths)? };
    Ok(VarBuilder::from_backend(
        Box::new(CpuCastSafetensors { inner }),
        dtype,
        device.clone(),
    ))
}
