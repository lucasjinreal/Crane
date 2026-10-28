// SPDX-License-Identifier: MIT
//! Test helpers shared by the per-model `enable_flash_causal()` tests:
//! deterministic weights and a logits comparison. GPU builds only, since
//! the vendored kernels exist nowhere else.

use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::{Init, VarBuilder};

use crate::ops::fused_ops::fattn::test_support::test_gpu_device;

/// Builds a `VarBuilder` whose tensors derive from a hash of their name, so
/// two models built from separate calls get identical weights.
pub(crate) fn hash_weights(dev: &Device, dtype: DType) -> VarBuilder<'static> {
    VarBuilder::from_backend(Box::new(HashWeights), dtype, dev.clone())
}

/// Runs a model built twice from the same weights over a prefill, a
/// continuation prefill and a decode step, with `enable_flash` applied to
/// one copy. The opted-in logits must stay within the tolerance of the
/// default ones. They must also differ bitwise on the prefills, since equal
/// logits would mean the vendored kernel never ran.
///
/// `tolerances` is the max absolute logits difference allowed per dtype.
pub(crate) fn assert_flash_tracks_default<M>(
    name: &str,
    vocab: usize,
    tolerances: [(DType, f32); 2],
    build: impl Fn(&Device, DType) -> M,
    forward: impl Fn(&mut M, &Tensor, usize) -> Tensor,
    enable_flash: impl Fn(&mut M),
) {
    let dev = test_gpu_device();
    for (dtype, tol) in tolerances {
        let mut default = build(&dev, dtype);
        let mut flash = build(&dev, dtype);
        enable_flash(&mut flash);

        let mut start = 0;
        for (step, len) in [40usize, 17, 1].into_iter().enumerate() {
            let ids: Vec<u32> = (start..start + len)
                .map(|i| ((i * 7 + 3) % vocab) as u32)
                .collect();
            let ids = Tensor::new(ids.as_slice(), &dev)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let want = forward(&mut default, &ids, start);
            let got = forward(&mut flash, &ids, start);
            let (max, equal) = compare_logits(&got, &want);
            println!("{name} {dtype:?} step {step} len {len}: max_abs={max:.5} equal={equal}");
            assert!(max <= tol, "{name} {dtype:?} step {step}: {max} > {tol}");
            if len > 1 {
                assert!(!equal, "{name} {dtype:?} step {step}: kernel did not run");
            }
            start += len;
        }
    }
}

/// Max absolute difference of two logits tensors, and whether they are
/// bitwise equal. Panics on non-finite values.
pub(crate) fn compare_logits(a: &Tensor, b: &Tensor) -> (f32, bool) {
    let flat = |t: &Tensor| {
        t.to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let (a, b) = (flat(a), flat(b));
    assert!(
        a.iter().chain(&b).all(|x| x.is_finite()),
        "non-finite logits"
    );
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    let equal = a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
    (max, equal)
}

/// Deterministic weights for any requested tensor name.
struct HashWeights;

impl SimpleBackend for HashWeights {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _h: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let mut state = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        });
        let fan_in = s.dims().last().copied().unwrap_or(1) as f32;
        let norm = name.contains("norm");
        let vals: Vec<f32> = (0..s.elem_count())
            .map(|_| {
                // xorshift64* mapped to [-1, 1).
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let r = state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40;
                let u = (r as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
                if name.ends_with("layer_scalar") {
                    1.0
                } else if name.ends_with("A_log") {
                    // Drives the GDN decay, so it must stay well negative.
                    0.1 * u - 2.0
                } else if norm {
                    1.0 + 0.1 * u
                } else {
                    u / fan_in.sqrt()
                }
            })
            .collect();
        // Cast on the CPU first so every backend sees identical bits.
        Tensor::from_vec(vals, s, &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(dev)
    }

    fn get_unchecked(&self, _n: &str, _d: DType, _v: &Device) -> candle_core::Result<Tensor> {
        candle_core::bail!("HashWeights requires a shape")
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        false
    }
}
