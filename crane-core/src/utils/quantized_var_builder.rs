//! Varbuilder for Loading gguf files
//!
//! `VarBuilder` is a utility to store quantized tensors from a [GGUF model file](https://huggingface.co/docs/hub/gguf).
//! These tensors can be loaded from disk using `from_gguf` or from an in-memory
//! buffer using `from_gguf_buffer`.

use candle_core::quantized::QTensor;
use candle_core::{Device, Result, Shape};
use std::sync::Arc;

// VarBuilder specialized for QTensors
#[derive(Clone)]
pub struct VarBuilder {
    data: Arc<std::collections::HashMap<String, Arc<QTensor>>>,
    path: Vec<String>,
    device: Device,
}

impl VarBuilder {
    /// # Errors
    ///
    /// Returns an error if `p` cannot be memory-mapped or does not contain a
    /// valid GGUF file.
    pub fn from_gguf<P: AsRef<std::path::Path>>(p: P, device: &Device) -> Result<Self> {
        let mmap =
            crate::quantized::gguf_file::mmap_gguf_file(p).map_err(candle_core::Error::wrap)?;
        Self::from_gguf_buffer(mmap.as_ref(), device)
    }

    /// # Errors
    ///
    /// Returns an error if `buffer` is not a valid GGUF file or if any of
    /// its tensors fails to decode (including i-quant tensors).
    pub fn from_gguf_buffer(buffer: &[u8], device: &Device) -> Result<Self> {
        // Go through `Gguf` so i-quant tensors are decoded (see
        // `crate::quantized::iquant`) instead of misread.
        let content = crate::quantized::extended_gguf::read_content_lenient(buffer)?;
        let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        let mut gg = crate::quantized::gguf_file::Gguf::new(
            content,
            std::io::Cursor::new(buffer),
            device.clone(),
            candle_core::DType::F32,
        );
        let mut data = std::collections::HashMap::new();
        for tensor_name in names {
            let tensor = gg.tensor(&tensor_name)?;
            data.insert(tensor_name, Arc::new(tensor));
        }
        Ok(Self {
            data: Arc::new(data),
            path: Vec::new(),
            device: device.clone(),
        })
    }

    pub fn pp<S: ToString + ?Sized>(&self, s: &S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            data: self.data.clone(),
            path,
            device: self.device.clone(),
        }
    }

    fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    /// # Errors
    ///
    /// Returns an error if no tensor named `name` (under the current path)
    /// exists, or if its shape does not match `s`.
    pub fn get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);
        match self.data.get(&path) {
            None => {
                candle_core::bail!("cannot find tensor {path}")
            },
            Some(qtensor) => {
                let shape = s.into();
                if qtensor.shape() != &shape {
                    candle_core::bail!(
                        "shape mismatch for {name}, got {:?}, expected {shape:?}",
                        qtensor.shape()
                    )
                }
                Ok(qtensor.clone())
            },
        }
    }

    /// # Errors
    ///
    /// Returns an error if no tensor named `name` (under the current path)
    /// exists.
    pub fn get_no_shape(&self, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);
        match self.data.get(&path) {
            None => {
                candle_core::bail!("cannot find tensor {name}")
            },
            Some(qtensor) => Ok(qtensor.clone()),
        }
    }

    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    #[must_use]
    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
}
