//! Polymorphic linear layer shared by the safetensors, in-situ-quantized (ISQ)
//! and GGUF loading paths.
//!
//! [`LinearLayer`] lives here (rather than in a model module) so that shared
//! ops like [`crate::ops::gdn`] can use it without depending on a specific
//! model; `models::hunyuan_dense::modeling` re-exports it for its existing
//! users (hunyuan, qwen3).

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, VarBuilder};
use std::sync::Arc;

/// A linear layer that can be either a standard (f16/f32) Linear or a
/// quantized QMatMul. Both implement Module::forward identically from the
/// caller's perspective. This allows the same model code to serve both
/// safetensors and GGUF weights with zero duplication.
pub enum LinearLayer {
    Standard(Linear),
    Quantized(QMatMul),
}

impl Module for LinearLayer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard(l) => l.forward(xs),
            Self::Quantized(q) => {
                // candle's QMatMul internally dequantizes weights to F32
                // and requires F32 input. When the activation pipeline
                // runs in BF16/F16 we must:
                //   1. Cast input to F32 for the quantized matmul
                //   2. Cast output back to the original dtype for
                //      downstream binary ops (residual adds, etc.)
                let input_dtype = xs.dtype();
                let xs_f32 = if input_dtype != DType::F32 {
                    xs.to_dtype(DType::F32)?
                } else {
                    xs.clone()
                };
                let out = q.forward(&xs_f32)?;
                if input_dtype != DType::F32 {
                    out.to_dtype(input_dtype)
                } else {
                    Ok(out)
                }
            }
        }
    }
}

/// Parse a quantization level name as accepted by `--quant` / `CRANE_ISQ`
/// (e.g. `q4_0`, `q8_0`, `q4k` / `q4_k`, case-insensitive).
pub fn parse_ggml_dtype(name: &str) -> Result<GgmlDType> {
    let normalized = name.trim().to_lowercase().replace("_k", "k");
    let dt = match normalized.as_str() {
        "q4_0" => GgmlDType::Q4_0,
        "q4_1" => GgmlDType::Q4_1,
        "q5_0" => GgmlDType::Q5_0,
        "q5_1" => GgmlDType::Q5_1,
        "q8_0" => GgmlDType::Q8_0,
        "q2k" => GgmlDType::Q2K,
        "q3k" => GgmlDType::Q3K,
        "q4k" => GgmlDType::Q4K,
        "q5k" => GgmlDType::Q5K,
        "q6k" => GgmlDType::Q6K,
        _ => candle_core::bail!(
            "unknown quantization level '{name}' (expected one of q4_0, q4_1, q5_0, q5_1, q8_0, q2k, q3k, q4k, q5k, q6k)"
        ),
    };
    Ok(dt)
}

/// Quantize a loaded linear's weight in place (ISQ), returning a
/// [`LinearLayer::Quantized`].
///
/// K-quants need the input dim to be a multiple of 256; when it isn't, fall
/// back to `Q8_0` (block size 32) so oddly-shaped projections still shrink
/// instead of erroring out. Bias-carrying linears are not supported (QMatMul
/// has no bias) — all Qwen 3.5 linears are bias-free.
pub fn quantize_linear(linear: Linear, dtype: GgmlDType) -> Result<LinearLayer> {
    if linear.bias().is_some() {
        candle_core::bail!("ISQ does not support linears with a bias");
    }
    let weight = linear.weight();
    let in_dim = weight.dim(candle_core::D::Minus1)?;
    let dtype = if in_dim % dtype.block_size() == 0 {
        dtype
    } else {
        GgmlDType::Q8_0
    };
    if in_dim % dtype.block_size() != 0 {
        // Even Q8_0 can't represent this shape; keep it in full precision.
        return Ok(LinearLayer::Standard(linear));
    }
    let qt = QTensor::quantize(weight, dtype)?;
    Ok(LinearLayer::Quantized(QMatMul::from_arc(Arc::new(qt))?))
}

/// Load a bias-free linear from `vb`, optionally quantizing it at load time.
///
/// With `quant: None` this is exactly `linear_no_bias` wrapped in
/// [`LinearLayer::Standard`]; with `Some(dtype)` the bf16/f16 weight is
/// quantized immediately and dropped, keeping peak memory near the quantized
/// size when loading from mmaped safetensors.
pub fn linear_layer(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
    quant: Option<GgmlDType>,
) -> Result<LinearLayer> {
    let linear = linear_no_bias(in_dim, out_dim, vb)?;
    match quant {
        None => Ok(LinearLayer::Standard(linear)),
        Some(dt) => quantize_linear(linear, dt),
    }
}
