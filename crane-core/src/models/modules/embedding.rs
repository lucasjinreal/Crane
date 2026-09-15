//! Embedding table that can stay quantized in device memory.
//!
//! The default GGUF path dequantizes `token_embd.weight` to the compute dtype
//! at load. For small vocabularies that is free, but modern Qwen checkpoints
//! carry a 248320-row table: on Qwen 3.8-27B it is 0.67 GiB of Q4_K on disk
//! and **2.37 GiB** once expanded to BF16 — more than the KV cache, spent
//! entirely on rows that are never read. A forward pass touches at most
//! `seq_len` of them.
//!
//! [`EmbeddingLayer::Quantized`] keeps the table in its on-disk format and
//! dequantizes only the gathered rows, via candle's `get_rows_*` kernels
//! (CUDA/Metal) or its CPU block decoder. The saving is the whole difference
//! between the two figures above.

use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};

use crate::ops::linear::LinearLayer;
use candle_core::{DType, Device, Module, Result, Tensor};

/// `CRANE_EMBED_DENSE=1` restores the old behaviour — dequantize the whole
/// table at load. An escape hatch if a backend's row gather ever misbehaves,
/// and the A/B switch the equivalence test uses to compare both paths on one
/// checkpoint.
fn dense_forced() -> bool {
    std::env::var("CRANE_EMBED_DENSE").is_ok_and(|v| v != "0" && !v.is_empty())
}

/// An embedding table, dense or quantized.
pub enum EmbeddingLayer {
    Dense(candle_nn::Embedding),
    Quantized {
        weight: Arc<QTensor>,
        /// Compute dtype; the gather kernels always produce F32.
        dtype: DType,
    },
}

impl EmbeddingLayer {
    /// Wrap a GGUF tensor, keeping it quantized when that is both possible and
    /// worthwhile.
    ///
    /// Falls back to a dense table when the weight is already unquantized
    /// (nothing to save) or when the row gather is unsupported here.
    ///
    /// That second condition is settled by *attempting* a one-row lookup
    /// rather than by enumerating dtypes and backends: candle's own
    /// preconditions (row length a whole number of blocks, a `get_rows_*`
    /// kernel for this dtype) then apply themselves, and any combination it
    /// cannot serve degrades at load instead of failing on the first token.
    ///
    /// `hidden_size` is used only to build the dense fallback; the quantized
    /// gather reads the row length from the tensor's own shape.
    pub fn from_qtensor(weight: QTensor, hidden_size: usize, dtype: DType) -> Result<Self> {
        let device = weight.device();
        let is_quantized = !matches!(
            weight.dtype(),
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        );

        if is_quantized && !dense_forced() {
            let weight = Arc::new(weight);
            let probe =
                Tensor::zeros((1,), DType::U32, &device).and_then(|ids| weight.embedding(&ids));
            match probe {
                Ok(_) => return Ok(Self::Quantized { weight, dtype }),
                Err(e) => {
                    eprintln!(
                        "[embedding] quantized row gather unavailable ({e}); \
                         falling back to a dense table"
                    );
                    // Reclaim the QTensor so the fallback can dequantize it.
                    let weight = Arc::try_unwrap(weight)
                        .map_err(|_| candle_core::Error::Msg("embedding probe leaked".into()))?;
                    return Self::dense(&weight, hidden_size, dtype, &device);
                },
            }
        }
        Self::dense(&weight, hidden_size, dtype, &device)
    }

    fn dense(weight: &QTensor, hidden_size: usize, dtype: DType, device: &Device) -> Result<Self> {
        let w = weight.dequantize(device)?.to_dtype(dtype)?;
        Ok(Self::Dense(candle_nn::Embedding::new(w, hidden_size)))
    }

    pub fn dense_from_tensor(weight: Tensor, hidden_size: usize) -> Self {
        Self::Dense(candle_nn::Embedding::new(weight, hidden_size))
    }

    /// Gather rows for `ids`, in the compute dtype.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(e) => e.forward(ids),
            Self::Quantized { weight, dtype } => weight.embedding(ids)?.to_dtype(*dtype),
        }
    }

    /// The output projection for a checkpoint with tied weights.
    ///
    /// Tying means the lm_head *is* this table, so a quantized table yields a
    /// quantized `QMatMul` over the very same buffer — no second copy, which
    /// is the other half of the memory saving on tied models.
    ///
    /// Returns a [`LinearLayer`] rather than a bare `QMatMul` on purpose: a
    /// dense table must become `LinearLayer::Standard`, because
    /// `LinearLayer::Quantized` up-casts its input to F32 for candle's
    /// dequantizing matmul, and `QMatMul::Tensor` is a plain matmul that does
    /// no such casting — pairing them fails with "dtype mismatch in matmul,
    /// lhs: F32, rhs: BF16" on the first forward.
    pub fn tied_output(&self) -> Result<LinearLayer> {
        match self {
            Self::Dense(e) => Ok(LinearLayer::Standard(candle_nn::Linear::new(
                e.embeddings().clone(),
                None,
            ))),
            Self::Quantized { weight, .. } => {
                Ok(LinearLayer::Quantized(QMatMul::from_arc(weight.clone())?))
            },
        }
    }

    /// [`Self::tied_output`], pre-converted to F32 when `dtype` is F16.
    ///
    /// Raw logits over a 100k+ vocab routinely exceed F16's 65504 max, so a
    /// tied F16 checkpoint pre-stores the projection in F32 to avoid
    /// overflow, matching `LinearLayer::forward_logits`'s untied `Quantized`
    /// path, which already computes in F32 internally. BF16/F32 share F32's
    /// exponent range and can't overflow, so they stay native — BF16 in
    /// particular must stay BF16 for the CUDA `gpu_argmax` sampling fast
    /// path, which only accepts BF16 logits.
    pub fn tied_output_upcast_f16(&self, dtype: DType) -> Result<LinearLayer> {
        Ok(match self.tied_output()? {
            LinearLayer::Standard(l) if dtype == DType::F16 => LinearLayer::Standard(
                candle_nn::Linear::new(l.weight().to_dtype(DType::F32)?, None),
            ),
            other => other,
        })
    }

    /// The dense weight, when this table is dense.
    pub fn dense_weight(&self) -> Option<&Tensor> {
        match self {
            Self::Dense(e) => Some(e.embeddings()),
            Self::Quantized { .. } => None,
        }
    }

    /// The device this table's weight is resident on.
    #[must_use]
    pub fn device(&self) -> Device {
        match self {
            Self::Dense(e) => e.embeddings().device().clone(),
            Self::Quantized { weight, .. } => weight.device(),
        }
    }

    /// Vocabulary size, i.e. the table's row count.
    ///
    /// # Panics
    ///
    /// Panics if the underlying weight tensor is 0-dimensional. All
    /// construction paths ([`Self::from_qtensor`], [`Self::dense`],
    /// [`Self::dense_from_tensor`]) produce a 2-D `(vocab, hidden)` table, so
    /// this cannot happen in practice.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Dense(e) => e.embeddings().dims()[0],
            Self::Quantized { weight, .. } => weight.shape().dims()[0],
        }
    }

    /// Resident size of the table in bytes.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::Dense(e) => e.embeddings().elem_count() * e.embeddings().dtype().size_in_bytes(),
            Self::Quantized { weight, .. } => weight.storage_size_in_bytes(),
        }
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Quantized { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// A quantized table must gather the same rows a dense one would. Q8_0 is
    /// accurate enough to compare against the original values directly.
    #[test]
    fn quantized_lookup_matches_dense_lookup() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (64usize, 256usize);
        let data: Vec<f32> = (0..vocab * hidden)
            .map(|i| (i % 97) as f32 * 0.01)
            .collect();
        let dense_w = Tensor::from_vec(data, (vocab, hidden), &dev)?;

        let q = QTensor::quantize(&dense_w, GgmlDType::Q8_0)?;
        let quantized = EmbeddingLayer::from_qtensor(q, hidden, DType::F32)?;
        assert!(quantized.is_quantized(), "Q8_0 table should stay quantized");

        let dense = EmbeddingLayer::dense_from_tensor(dense_w, hidden);

        let ids = Tensor::new(&[7u32, 0, 63, 7], &dev)?;
        let a = quantized.forward(&ids)?;
        let b = dense.forward(&ids)?;
        assert_eq!(a.dims(), b.dims());

        let diff = (a - b)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 0.02, "quantized lookup drifted by {diff}");
        Ok(())
    }

    /// A *dense* tied table must still work with non-F32 activations.
    ///
    /// Wrapping it as `LinearLayer::Quantized` type-checks but fails on the
    /// first forward with "dtype mismatch in matmul, lhs: F32, rhs: BF16",
    /// because that variant up-casts its input for candle's dequantizing
    /// matmul while `QMatMul::Tensor` does no casting of its own. That is how
    /// `CRANE_EMBED_DENSE=1` broke on tied checkpoints.
    #[test]
    fn dense_tied_output_accepts_non_f32_activations() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (16usize, 32usize);
        // Non-F32 activations are the trigger; the loader uses BF16 on CUDA
        // and F16 on Metal/ROCm. F16 here because candle's CPU matmul has no
        // BF16 kernel, and the distinction that matters is only "not F32".
        let w = Tensor::zeros((vocab, hidden), DType::F16, &dev)?;
        let layer = EmbeddingLayer::dense_from_tensor(w, hidden);

        assert!(matches!(layer.tied_output()?, LinearLayer::Standard(_)));

        let x = Tensor::zeros((1, 1, hidden), DType::F16, &dev)?;
        let y = layer.tied_output()?.forward(&x)?;
        assert_eq!(y.dims(), &[1, 1, vocab]);
        Ok(())
    }

    /// The quantized table must be markedly smaller — that is the whole point.
    #[test]
    fn quantized_table_is_smaller_than_dense() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (64usize, 256usize);
        let w = Tensor::zeros((vocab, hidden), DType::F32, &dev)?;
        let q = QTensor::quantize(&w, GgmlDType::Q4K)?;

        let quantized = EmbeddingLayer::from_qtensor(q, hidden, DType::F32)?;
        let dense = EmbeddingLayer::dense_from_tensor(w, hidden);
        assert!(
            quantized.size_in_bytes() * 4 < dense.size_in_bytes(),
            "Q4_K table ({}) should be far under a third of dense F32 ({})",
            quantized.size_in_bytes(),
            dense.size_in_bytes()
        );
        Ok(())
    }

    /// Tied models must reuse the one buffer for both lookup and output
    /// projection — a dense copy of the table would give back the saving.
    #[test]
    fn tied_output_reuses_the_quantized_table() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (64usize, 256usize);
        let w = Tensor::zeros((vocab, hidden), DType::F32, &dev)?;
        let layer = EmbeddingLayer::from_qtensor(
            QTensor::quantize(&w, GgmlDType::Q4K)?,
            hidden,
            DType::F32,
        )?;

        assert!(matches!(
            layer.tied_output()?,
            LinearLayer::Quantized(QMatMul::QTensor(_))
        ));

        // …and it projects to vocab logits, i.e. it is the transposed matmul
        // a tied lm_head needs.
        let x = Tensor::zeros((1, hidden), DType::F32, &dev)?;
        assert_eq!(layer.tied_output()?.forward(&x)?.dims(), &[1, vocab]);
        Ok(())
    }

    /// A tied F16 checkpoint must upcast the projection to F32 to avoid
    /// overflowing F16's 65504 max on vocab-sized logits.
    #[test]
    fn tied_output_upcast_f16_promotes_dense_f16_to_f32() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (16usize, 32usize);
        let w = Tensor::zeros((vocab, hidden), DType::F16, &dev)?;
        let layer = EmbeddingLayer::dense_from_tensor(w, hidden);

        match layer.tied_output_upcast_f16(DType::F16)? {
            LinearLayer::Standard(l) => assert_eq!(l.weight().dtype(), DType::F32),
            LinearLayer::Quantized(_) => panic!("dense table must not become quantized"),
        }
        Ok(())
    }

    /// A tied BF16 checkpoint must stay BF16 -- CUDA's `gpu_argmax` sampling
    /// fast path only accepts BF16 logits.
    #[test]
    fn tied_output_upcast_f16_leaves_bf16_untouched() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (16usize, 32usize);
        let w = Tensor::zeros((vocab, hidden), DType::BF16, &dev)?;
        let layer = EmbeddingLayer::dense_from_tensor(w, hidden);

        match layer.tied_output_upcast_f16(DType::BF16)? {
            LinearLayer::Standard(l) => assert_eq!(l.weight().dtype(), DType::BF16),
            LinearLayer::Quantized(_) => panic!("dense table must not become quantized"),
        }
        Ok(())
    }

    /// A quantized tied table stays quantized even under F16 compute -- the
    /// `Quantized` `LinearLayer` variant already computes in F32 internally.
    #[test]
    fn tied_output_upcast_f16_leaves_quantized_table_quantized() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (64usize, 256usize);
        let w = Tensor::zeros((vocab, hidden), DType::F32, &dev)?;
        let layer = EmbeddingLayer::from_qtensor(
            QTensor::quantize(&w, GgmlDType::Q4K)?,
            hidden,
            DType::F16,
        )?;

        assert!(matches!(
            layer.tied_output_upcast_f16(DType::F16)?,
            LinearLayer::Quantized(QMatMul::QTensor(_))
        ));
        Ok(())
    }

    /// An unquantized GGUF tensor has nothing to gain from the gather path.
    #[test]
    fn unquantized_weights_stay_dense() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::zeros((8, 256), DType::F32, &dev)?;
        let q = QTensor::quantize(&w, GgmlDType::F32)?;
        let layer = EmbeddingLayer::from_qtensor(q, 256, DType::F32)?;
        assert!(!layer.is_quantized());
        Ok(())
    }
}
