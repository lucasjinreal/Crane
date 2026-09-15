//! Polymorphic linear layer shared by the safetensors, in-situ-quantized (ISQ)
//! and GGUF loading paths.
//!
//! [`LinearLayer`] lives here (rather than in a model module) so that shared
//! ops like [`crate::ops::gdn`] can use it without depending on a specific
//! model; `models::hunyuan_dense::modeling` re-exports it for its existing
//! users (hunyuan, qwen3).

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder, linear_no_bias};
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
            },
        }
    }
}

impl LinearLayer {
    /// Projects in F32, returning F32 output regardless of the stored weight
    /// dtype.
    ///
    /// [`Module::forward`] casts `Quantized` output back to the input dtype,
    /// which overflows to Inf for wide projections (e.g. a vocab-sized
    /// `lm_head`) in F16 — F16's max representable value is 65504, and raw
    /// logits routinely exceed it. Use this instead of `forward` at the
    /// final logit projection so downstream sampling always sees finite
    /// values.
    ///
    /// # Errors
    ///
    /// Returns an error if the dtype cast or underlying matmul fails.
    pub fn forward_f32(&self, xs: &Tensor) -> Result<Tensor> {
        let xs_f32 = if xs.dtype() == DType::F32 {
            xs.clone()
        } else {
            xs.to_dtype(DType::F32)?
        };
        match self {
            Self::Standard(l) => {
                let weight = l.weight();
                let weight_f32 = if weight.dtype() == DType::F32 {
                    weight.clone()
                } else {
                    weight.to_dtype(DType::F32)?
                };
                let bias_f32 = l.bias().map(|b| b.to_dtype(DType::F32)).transpose()?;
                Linear::new(weight_f32, bias_f32).forward(&xs_f32)
            },
            // QMatMul already dequantizes and computes in F32 internally;
            // with F32 input the matmul runs natively, no cast needed.
            Self::Quantized(q) => q.forward(&xs_f32),
        }
    }

    /// Vocab projection that avoids F16 overflow on wide `lm_head` matmuls
    /// while leaving other dtypes on their normal, cheaper path.
    ///
    /// F16 inputs are upcast via [`Self::forward_f32`] since F16's max
    /// representable value (~65504) is easily exceeded by raw logits over a
    /// 100k+ vocab. BF16 and F32 inputs pass through [`Module::forward`]
    /// unchanged: BF16 shares F32's 8-bit exponent (max ~3.4e38) and cannot
    /// overflow, so upcasting it would only waste memory and, for backends
    /// whose sampling fast path requires BF16 logits, break sampling.
    ///
    /// # Errors
    ///
    /// Returns an error if the dtype cast or underlying matmul fails.
    pub fn forward_logits(&self, xs: &Tensor) -> Result<Tensor> {
        if xs.dtype() == DType::F16 {
            self.forward_f32(xs)
        } else {
            self.forward(xs)
        }
    }

    /// Moves this layer's weights to `device`, in `dtype`.
    ///
    /// `Standard` moves its tensors directly, casting to `dtype`. `QTensor`
    /// has no device-transfer primitive, so `Quantized` dequantizes (via
    /// `QMatMul::dequantize_f16`, then casts to `dtype`) and returns a
    /// `Standard` layer on `device` — this loses the quantized memory
    /// footprint for the moved weight, a deliberate tradeoff for promoting
    /// an expert from CPU to GPU once real headroom is known (see
    /// `Qwen3Model::from_gguf`'s post-warmup promotion pass). `dtype` must
    /// match the model's compute dtype (`Qwen3Model::dtype`): candle's
    /// matmul requires both operands to share a dtype, so a promoted
    /// weight left in the wrong dtype fails on its very next forward pass.
    ///
    /// # Errors
    ///
    /// Returns an error if the dequantization, dtype cast, or device
    /// transfer fails.
    pub fn to_device(&self, device: &Device, dtype: DType) -> Result<LinearLayer> {
        match self {
            Self::Standard(l) => {
                let weight = l.weight().to_device(device)?.to_dtype(dtype)?;
                let bias = l
                    .bias()
                    .map(|b| b.to_device(device)?.to_dtype(dtype))
                    .transpose()?;
                Ok(Self::Standard(Linear::new(weight, bias)))
            },
            Self::Quantized(q) => {
                let weight = q.dequantize_f16()?.to_device(device)?.to_dtype(dtype)?;
                Ok(Self::Standard(Linear::new(weight, None)))
            },
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // A 4x2 weight and matching input whose dot products (4 * 300 * 300 =
    // 360,000) exceed F16's ~65504 max but fit comfortably in F32 — the
    // exact overflow shape of a wide `lm_head` projection at model scale.
    fn overflow_weight_and_input() -> (Tensor, Tensor) {
        let device = Device::Cpu;
        let weight = Tensor::from_vec(vec![300.0_f32; 4 * 4], (4, 4), &device).unwrap();
        let input = Tensor::from_vec(vec![300.0_f32; 4], (1, 4), &device).unwrap();
        (weight, input)
    }

    // Verifies a `Standard` layer with an F32-stored weight produces a
    // finite F32 result for inputs that would overflow if computed in F16.
    #[test]
    fn forward_f32_standard_avoids_f16_overflow() {
        let (weight, input) = overflow_weight_and_input();
        let layer = LinearLayer::Standard(Linear::new(weight, None));
        let input_f16 = input.to_dtype(DType::F16).unwrap();

        let out = layer.forward_f32(&input_f16).unwrap();
        assert_eq!(out.dtype(), DType::F32);
        let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
        assert!((values[0] - 360_000.0).abs() < 1.0);
    }

    // Demonstrates the bug `forward_f32` fixes: the same weight/input pair,
    // computed via the regular `Module::forward` path in F16, overflows to
    // Inf instead of the finite value above.
    #[test]
    fn forward_standard_f16_overflows_to_inf() {
        let (weight, input) = overflow_weight_and_input();
        let layer = LinearLayer::Standard(Linear::new(weight.to_dtype(DType::F16).unwrap(), None));
        let input_f16 = input.to_dtype(DType::F16).unwrap();

        let out = layer.forward(&input_f16).unwrap();
        let values = out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(values[0].is_infinite());
    }

    // Verifies a `Quantized` layer's `forward_f32` returns finite F32
    // output even when the caller passes an F16 input.
    #[test]
    fn forward_f32_quantized_returns_f32() {
        let (weight, input) = overflow_weight_and_input();
        let qt = QTensor::quantize(&weight, GgmlDType::F32).unwrap();
        let layer = LinearLayer::Quantized(QMatMul::from_arc(Arc::new(qt)).unwrap());
        let input_f16 = input.to_dtype(DType::F16).unwrap();

        let out = layer.forward_f32(&input_f16).unwrap();
        assert_eq!(out.dtype(), DType::F32);
        let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
        assert!((values[0] - 360_000.0).abs() < 1.0);
    }

    // `forward_f32` on already-F32 input/weight should be a pure passthrough
    // with no unnecessary casting, and still produce the correct value.
    #[test]
    fn forward_f32_noop_when_already_f32() {
        let (weight, input) = overflow_weight_and_input();
        let layer = LinearLayer::Standard(Linear::new(weight, None));

        let out = layer.forward_f32(&input).unwrap();
        assert_eq!(out.dtype(), DType::F32);
        let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((values[0] - 360_000.0).abs() < 1.0);
    }

    // `forward_logits` must upcast F16 to F32, the only dtype narrow enough
    // to overflow at this magnitude.
    #[test]
    fn forward_logits_f16_upcasts_to_f32() {
        let (weight, input) = overflow_weight_and_input();
        let layer = LinearLayer::Standard(Linear::new(weight.to_dtype(DType::F16).unwrap(), None));
        let input_f16 = input.to_dtype(DType::F16).unwrap();

        let out = layer.forward_logits(&input_f16).unwrap();
        assert_eq!(out.dtype(), DType::F32);
        let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
        assert!((values[0] - 360_000.0).abs() < 1.0);
    }

    // BF16 shares F32's exponent range, so `forward_logits` must leave it on
    // the regular `forward` path instead of upcasting it. Uses `Quantized`
    // rather than `Standard` because candle's CPU backend has no BF16 matmul
    // kernel; `QMatMul` sidesteps that by always matmul-ing in F32 and only
    // `to_dtype`-casting the result back to BF16, which CPU does support.
    #[test]
    fn forward_logits_bf16_stays_bf16() {
        let (weight, input) = overflow_weight_and_input();
        let qt = QTensor::quantize(&weight, GgmlDType::F32).unwrap();
        let layer = LinearLayer::Quantized(QMatMul::from_arc(Arc::new(qt)).unwrap());
        let input_bf16 = input.to_dtype(DType::BF16).unwrap();

        let out = layer.forward_logits(&input_bf16).unwrap();
        assert_eq!(out.dtype(), DType::BF16);
        let values = out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(values.iter().all(|v| v.is_finite()));
    }

    // F32 input should pass straight through `forward_logits` unchanged.
    #[test]
    fn forward_logits_f32_stays_f32() {
        let (weight, input) = overflow_weight_and_input();
        let layer = LinearLayer::Standard(Linear::new(weight, None));

        let out = layer.forward_logits(&input).unwrap();
        assert_eq!(out.dtype(), DType::F32);
        let values = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((values[0] - 360_000.0).abs() < 1.0);
    }

    // Verifies `Standard::to_device` preserves the forward-pass output
    // (a CPU->CPU move is a data copy, not a value change).
    #[test]
    fn to_device_standard_preserves_output() {
        let (weight, input) = overflow_weight_and_input();
        let layer = LinearLayer::Standard(Linear::new(weight, None));
        let before = layer.forward(&input).unwrap();

        let moved = layer.to_device(&Device::Cpu, DType::F32).unwrap();
        assert!(matches!(moved, LinearLayer::Standard(_)));
        let after = moved.forward(&input).unwrap();

        assert_eq!(
            before.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            after.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
    }

    // Verifies `Quantized::to_device` dequantizes to a `Standard` layer on
    // the target device and produces (near-)identical output — a promoted
    // expert must still forward correctly, just no longer quantized.
    #[test]
    fn to_device_quantized_dequantizes_and_preserves_output() {
        let (weight, input) = overflow_weight_and_input();
        let qt = QTensor::quantize(&weight, GgmlDType::F32).unwrap();
        let layer = LinearLayer::Quantized(QMatMul::from_arc(Arc::new(qt)).unwrap());
        let before = layer.forward_f32(&input).unwrap();

        let moved = layer.to_device(&Device::Cpu, DType::F32).unwrap();
        assert!(matches!(moved, LinearLayer::Standard(_)));
        let after = moved.forward_f32(&input).unwrap();

        let before_vals = before.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let after_vals = after.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (b, a) in before_vals.iter().zip(after_vals.iter()) {
            assert!((b - a).abs() < 1.0, "before={b} after={a}");
        }
    }

    // Verifies `to_device` casts a promoted quantized expert's weight to
    // the requested compute dtype rather than hard-coding F16, and that
    // plain `forward` (not `forward_f32`) succeeds afterward with a
    // matching-dtype input — this is the path that broke when `to_device`
    // ignored its caller's compute dtype and always dequantized to F16.
    #[test]
    fn to_device_quantized_casts_to_requested_dtype() {
        let (weight, input) = overflow_weight_and_input();
        let qt = QTensor::quantize(&weight, GgmlDType::F32).unwrap();
        let layer = LinearLayer::Quantized(QMatMul::from_arc(Arc::new(qt)).unwrap());

        let moved = layer.to_device(&Device::Cpu, DType::F32).unwrap();
        let LinearLayer::Standard(l) = &moved else {
            panic!("expected Standard after to_device");
        };
        assert_eq!(l.weight().dtype(), DType::F32);

        let out = moved.forward(&input).unwrap();
        assert_eq!(out.dtype(), DType::F32);
    }
}
