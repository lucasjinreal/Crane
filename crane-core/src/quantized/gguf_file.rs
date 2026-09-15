// SPDX-License-Identifier: MIT

//! Shared GGUF file loading: memory-mapping and typed tensor access.
//!
//! Used by every model that supports GGUF checkpoints (`hunyuan_dense`,
//! `gemma4`, `qwen3`, `qwen3_5`, `minicpm5`, `minicpmo`).

use candle_core::quantized::{QTensor, gguf_file};
use candle_core::{DType, Device, Result};
use candle_nn::RmsNorm;
use std::io::{Read, Seek};
use std::sync::Arc;

/// Opens and memory-maps a GGUF file for zero-syscall tensor reads.
///
/// The returned `Mmap` can be wrapped in a `std::io::Cursor` and passed
/// anywhere a `Read + Seek` reader is expected (e.g. [`Gguf::new`]), letting
/// tensor loads page data in from disk on demand instead of going through
/// per-tensor `seek`/`read_exact` syscalls.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or memory-mapped.
pub fn mmap_gguf_file(path: impl AsRef<std::path::Path>) -> std::io::Result<memmap2::Mmap> {
    let file = std::fs::File::open(path)?;
    // SAFETY: the caller must not truncate or replace this file on disk while
    // the returned mapping is alive. Doing so raises SIGBUS on a later page-in
    // (e.g. during tensor loading), which is unrecoverable and not something a
    // `Result` can catch. Crane itself never writes to model files it has
    // loaded; this only holds if external tooling (re-downloads, redeploys)
    // avoids replacing a model file path while a server process has it mapped.
    unsafe { memmap2::Mmap::map(&file) }
}

/// Wraps a parsed GGUF file + reader for convenient tensor loading.
pub struct Gguf<R: Read + Seek> {
    pub ct: gguf_file::Content,
    reader: R,
    device: Device,
    /// Target compute dtype. Dequantized tensors (norms, embeddings) are
    /// cast to this dtype so they match the activations flowing through the
    /// model (e.g. BF16 on CUDA). Quantized linear layers (`QMatMul`) handle
    /// their own internal dtype and the `LinearLayer` wrapper casts their
    /// output to the input's dtype.
    dtype: DType,
}

impl<R: Read + Seek> Gguf<R> {
    pub fn new(ct: gguf_file::Content, reader: R, device: Device, dtype: DType) -> Self {
        Self {
            ct,
            reader,
            device,
            dtype,
        }
    }

    /// Load a quantized tensor and wrap as a `LinearLayer` (`QMatMul`).
    ///
    /// # Errors
    ///
    /// Returns an error if the named tensor is missing or malformed.
    pub fn linear(&mut self, name: &str) -> Result<crate::ops::linear::LinearLayer> {
        let device = self.device.clone();
        self.linear_on(name, &device)
    }

    /// Load a quantized tensor onto `device` and wrap as a `LinearLayer` (`QMatMul`).
    ///
    /// Identical to [`Self::linear`] but places the weight on a caller-chosen
    /// device instead of `self.device` — used for `MoE` expert offloading where
    /// experts may live on a different device (e.g. CPU) than the rest of the
    /// model.
    ///
    /// # Errors
    /// Returns an error if the tensor is missing from the GGUF file, the
    /// quantization type is unsupported, or the `QMatMul` construction fails.
    pub fn linear_on(
        &mut self,
        name: &str,
        device: &Device,
    ) -> Result<crate::ops::linear::LinearLayer> {
        let ws = self.ct.tensor(&mut self.reader, name, device)?;
        let qmm = candle_core::quantized::QMatMul::from_arc(Arc::new(ws))?;
        Ok(crate::ops::linear::LinearLayer::Quantized(qmm))
    }

    /// Load a tensor, dequantize, and create an `RmsNorm`.
    /// The weight is cast to the target `dtype` so it matches activations.
    ///
    /// # Errors
    ///
    /// Returns an error if the named tensor is missing or malformed.
    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        let weight = ws.dequantize(&self.device)?.to_dtype(self.dtype)?;
        Ok(RmsNorm::new(weight, eps))
    }

    /// Load an embedding table that stays quantized when it can, dequantizing
    /// only the rows a forward pass gathers.
    ///
    /// Prefer this over [`Self::embedding`] for large vocabularies: a 248k-row
    /// table costs ~2.4 GiB dense in BF16 versus ~0.7 GiB as `Q4_K`.
    ///
    /// # Errors
    ///
    /// Returns an error if the named tensor is missing or malformed.
    pub fn quantized_embedding(
        &mut self,
        name: &str,
        hidden_size: usize,
    ) -> Result<crate::models::modules::embedding::EmbeddingLayer> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        crate::models::modules::embedding::EmbeddingLayer::from_qtensor(ws, hidden_size, self.dtype)
    }

    /// Load a tensor, dequantize, and create an Embedding.
    /// The weight is cast to the target `dtype` so lookups produce
    /// tensors in the expected compute precision.
    ///
    /// # Errors
    ///
    /// Returns an error if the named tensor is missing or malformed.
    pub fn embedding(&mut self, name: &str, hidden_size: usize) -> Result<candle_nn::Embedding> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        let weight = ws.dequantize(&self.device)?.to_dtype(self.dtype)?;
        Ok(candle_nn::Embedding::new(weight, hidden_size))
    }

    /// Load a raw `QTensor` by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the named tensor is missing.
    pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }

    /// Load a raw `QTensor` by name onto `device`.
    ///
    /// Identical to [`Self::tensor`] but places the weight on a
    /// caller-chosen device instead of `self.device` — used for `MoE`
    /// packed expert tensors, which must load directly onto
    /// `expert_device` rather than the main model device.
    ///
    /// # Errors
    ///
    /// Returns an error if the named tensor is missing.
    pub fn tensor_on(&mut self, name: &str, device: &Device) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, device)
    }

    /// Load a tensor, dequantize, and cast to the target compute dtype.
    /// For small full-precision tensors (norm weights, biases, conv kernels).
    ///
    /// # Errors
    ///
    /// Returns an error if the named tensor is missing or malformed.
    pub fn dequant_tensor(&mut self, name: &str) -> Result<candle_core::Tensor> {
        let device = self.device.clone();
        self.dequant_tensor_on(name, &device)
    }

    /// Load a tensor onto `device`, dequantize, and cast to the target compute
    /// dtype.
    ///
    /// Identical to [`Self::dequant_tensor`] but places the result on a
    /// caller-chosen device instead of `self.device` — used for `MoE` packed
    /// expert tensors, which must be dequantized directly onto
    /// `expert_device` rather than the main model device.
    ///
    /// # Errors
    /// Returns an error if the tensor is missing from the GGUF file, the
    /// quantization type is unsupported, or dequantization fails.
    pub fn dequant_tensor_on(
        &mut self,
        name: &str,
        device: &Device,
    ) -> Result<candle_core::Tensor> {
        let ws = self.ct.tensor(&mut self.reader, name, device)?;
        ws.dequantize(device)?.to_dtype(self.dtype)
    }

    /// Whether the file contains a tensor with this exact name.
    pub fn contains_tensor(&self, name: &str) -> bool {
        self.ct.tensor_infos.contains_key(name)
    }

    /// Access GGUF metadata.
    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }
}

#[cfg(test)]
mod mmap_gguf_tests {
    use super::mmap_gguf_file;
    use candle_core::Tensor;
    use candle_core::quantized::{GgmlDType, QTensor, gguf_file};

    /// Writes a single quantized tensor to a real GGUF file on disk, for
    /// exercising the mmap read path against a real file.
    fn write_test_gguf(path: &std::path::Path) {
        let src = Tensor::arange(0f32, 4. * 32., &candle_core::Device::Cpu)
            .unwrap()
            .reshape((4, 32))
            .unwrap();
        let qtensor = QTensor::quantize(&src, GgmlDType::Q4_0).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        gguf_file::write(&mut file, &[], &[("weight", &qtensor)]).unwrap();
    }

    // Reading a GGUF tensor through the mmap'd path must return the exact
    // same bytes as reading it through a plain `File` reader, since the mmap
    // change (`mmap_gguf_file`) is only meant to swap the I/O mechanism, not
    // the data it produces.
    #[test]
    fn mmap_read_matches_direct_file_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.gguf");
        write_test_gguf(&path);

        let mmap = mmap_gguf_file(&path).unwrap();
        let mut mmap_cursor = std::io::Cursor::new(mmap.as_ref());
        let mmap_content = gguf_file::Content::read(&mut mmap_cursor).unwrap();
        let mmap_tensor = mmap_content
            .tensor(&mut mmap_cursor, "weight", &candle_core::Device::Cpu)
            .unwrap();

        let mut file = std::fs::File::open(&path).unwrap();
        let file_content = gguf_file::Content::read(&mut file).unwrap();
        let file_tensor = file_content
            .tensor(&mut file, "weight", &candle_core::Device::Cpu)
            .unwrap();

        let mmap_data = mmap_tensor.data().unwrap();
        let file_data = file_tensor.data().unwrap();
        assert_eq!(mmap_data, file_data);
    }

    // A missing path must surface as an `Err`, not panic (e.g. on the
    // `unsafe` `Mmap::map` call).
    #[test]
    fn mmap_missing_file_returns_err() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.gguf");
        assert!(mmap_gguf_file(&path).is_err());
    }
}
