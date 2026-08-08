//! Numeric cross-checks for Crane's own ROCm/HIP kernels.
//!
//! Each kernel is compared against the device-portable path it replaces, on the
//! same device, so a divergence is the kernel's and not the model's. Run with:
//!
//! ```bash
//! ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
//!   cargo test -p crane-core --release --features rocm --test rocm_kernels
//! ```
//!
//! Every test returns early when the machine has no ROCm device, matching how
//! candle's own backend tests behave.

#![cfg(all(feature = "rocm", not(feature = "cuda")))]

use candle_core::{D, DType, Device, Result, Tensor};
use crane_core::ops::gdn::{gated_delta_rule_recurrence, gdn_recurrence_rocm};

fn device() -> Option<Device> {
    Device::new_rocm(0).ok()
}

/// Cosine similarity of two tensors flattened to vectors.
fn cos_sim(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len(), "shape mismatch");
    let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    Ok(dot / (na * nb))
}

/// The fused recurrence against the op-by-op reference it replaces.
///
/// `K = 128` picks the register-resident specialization, which is the path
/// every supported model actually takes.
#[test]
fn gdn_recurrence_matches_the_portable_reference() -> Result<()> {
    for kdim in [128, 64] {
        let Some(dev) = device() else {
            return Ok(());
        };
        let (b, s, h, vdim) = (1usize, 24usize, 4usize, 128usize);

        // Portable layout: [B, S, H, *]. `g` is the log-decay, so keep it small
        // and negative; `beta` is a write strength in (0, 1).
        let q = Tensor::randn(0f32, 1.0, (b, s, h, kdim), &dev)?;
        let k = Tensor::randn(0f32, 1.0, (b, s, h, kdim), &dev)?;
        let v = Tensor::randn(0f32, 1.0, (b, s, h, vdim), &dev)?;
        let g = Tensor::randn(0f32, 1.0, (b, s, h), &dev)?.affine(0.01, -0.05)?;
        let beta = candle_nn::ops::sigmoid(&Tensor::randn(0f32, 1.0, (b, s, h), &dev)?)?;
        let state = Tensor::randn(0f32, 0.1, (b, h, kdim, vdim), &dev)?;

        let mut state_ref = state.clone();
        let y_ref = gated_delta_rule_recurrence(&q, &k, &v, &g, &beta, &mut state_ref)?;

        // Kernel layout: [BH, S, *], and it takes `q` already scaled by
        // 1/sqrt(K) — the reference applies that scale itself.
        let bh = b * h;
        let flat3 = |t: &Tensor| -> Result<Tensor> {
            t.transpose(1, 2)?.contiguous()?.reshape((bh, s, ()))
        };
        let flat2 =
            |t: &Tensor| -> Result<Tensor> { t.transpose(1, 2)?.contiguous()?.reshape((bh, s)) };
        let scale = 1.0 / (kdim as f64).sqrt();
        let (y, state_out) = gdn_recurrence_rocm(
            &flat3(&q)?.affine(scale, 0.0)?,
            &flat3(&k)?,
            &flat3(&v)?,
            &flat2(&g)?,
            &flat2(&beta)?,
            &state.reshape((bh, kdim, vdim))?.contiguous()?,
        )?;
        // Back to the reference's [B, S, H, V].
        let y = y.reshape((b, h, s, vdim))?.transpose(1, 2)?.contiguous()?;

        let sim_y = cos_sim(&y, &y_ref)?;
        let sim_state = cos_sim(&state_out.reshape((b, h, kdim, vdim))?, &state_ref)?;
        assert!(sim_y >= 0.9999, "K={kdim}: y cos={sim_y}");
        assert!(sim_state >= 0.9999, "K={kdim}: state cos={sim_state}");
    }
    Ok(())
}

/// The device top-k against the host sort it replaces, at a realistic
/// vocabulary and over the `k` values the sampler actually asks for.
///
/// `k` matters to the kernel in three ways: it splits the LDS buffer between
/// the running best-k and the staging area, it sets the size of the bitonic
/// prefix a flush sorts, and it is the stride of the stage-1 output. So the
/// list covers the sampler's default (40), values on and either side of a wave
/// boundary, a value that is not a multiple of anything (37), and the largest
/// `k` the kernel accepts.
#[test]
fn topk_indices_matches_the_host_sort() -> Result<()> {
    let Some(dev) = device() else {
        return Ok(());
    };
    let vocab = 248_320usize;
    let logits_cpu = Tensor::randn(0f32, 4.0, vocab, &Device::Cpu)?;
    let logits = logits_cpu.to_device(&dev)?;

    for k in [1usize, 20, 31, 32, 33, 37, 40, 63, 64, 128, 512] {
        let got = crane_core::ops::topk_indices(&logits, k)?.to_vec1::<u32>()?;
        let want = crane_core::ops::portable::topk_indices(&logits_cpu, k)?.to_vec1::<u32>()?;
        assert_eq!(got, want, "k={k}");
    }
    Ok(())
}

/// Above the kernel's maximum `k` the host sort takes over, rather than the
/// kernel silently truncating to what its buffer holds.
#[test]
fn topk_indices_falls_back_above_the_kernel_maximum() -> Result<()> {
    let Some(dev) = device() else {
        return Ok(());
    };
    let vocab = 4096usize;
    let logits_cpu = Tensor::randn(0f32, 4.0, vocab, &Device::Cpu)?;
    let logits = logits_cpu.to_device(&dev)?;

    // 512 is the maximum; above it the answer must still be full and correct.
    for k in [513usize, 4096] {
        let got = crane_core::ops::topk_indices(&logits, k)?.to_vec1::<u32>()?;
        let want = crane_core::ops::portable::topk_indices(&logits_cpu, k)?.to_vec1::<u32>()?;
        assert_eq!(got.len(), k, "k={k}");
        assert_eq!(got, want, "k={k}");
    }
    Ok(())
}

/// Ties must not duplicate or drop an index.
///
/// A vocabulary of 4 distinct values repeated 60 K times each is the worst case
/// for a selection that breaks ties by value alone: every candidate in the
/// top-k is equal to every other, so nothing but the index distinguishes them.
/// Both paths order equal values by ascending index, so the answer is exactly
/// the first `k` indices of the largest value.
#[test]
fn topk_indices_breaks_ties_by_index() -> Result<()> {
    let Some(dev) = device() else {
        return Ok(());
    };
    let vocab = 240_000usize;
    let values: Vec<f32> = (0..vocab).map(|i| ((i % 4) as f32) * 0.5).collect();
    let logits_cpu = Tensor::new(values.as_slice(), &Device::Cpu)?;
    let logits = logits_cpu.to_device(&dev)?;

    for k in [1usize, 40, 64] {
        let got = crane_core::ops::topk_indices(&logits, k)?.to_vec1::<u32>()?;
        // Value 1.5 sits at every index ≡ 3 (mod 4).
        let want: Vec<u32> = (0..k).map(|j| (j * 4 + 3) as u32).collect();
        assert_eq!(got, want, "k={k}");

        let unique: std::collections::HashSet<u32> = got.iter().copied().collect();
        assert_eq!(unique.len(), k, "k={k}: duplicate index");
    }
    Ok(())
}

/// A short vector still has to work: the grid is derived from the length, and
/// `k == n` is the degenerate case.
#[test]
fn topk_indices_handles_a_short_vector() -> Result<()> {
    let Some(dev) = device() else {
        return Ok(());
    };
    let values: Vec<f32> = vec![0.5, -3.0, 7.25, 1.0, 7.5];
    let logits = Tensor::new(values.as_slice(), &dev)?;
    let got = crane_core::ops::topk_indices(&logits, 5)?.to_vec1::<u32>()?;
    assert_eq!(got, vec![4, 2, 3, 0, 1]);
    Ok(())
}

/// The launch geometry is derived from the row length, so lengths that do not
/// divide the block span, and lengths that straddle the one-block/many-block
/// boundary, both have to land on the same answer as the host.
#[test]
fn topk_indices_matches_the_host_sort_at_awkward_lengths() -> Result<()> {
    let Some(dev) = device() else {
        return Ok(());
    };
    // 1024 elements/block/iteration, 4096 per block at the default grid.
    for n in [1usize, 2, 40, 1023, 1024, 1025, 4095, 4097, 12_289, 65_537] {
        let logits_cpu = Tensor::randn(0f32, 4.0, n, &Device::Cpu)?;
        let logits = logits_cpu.to_device(&dev)?;
        for k in [1usize, 7, 40] {
            if k > n {
                continue;
            }
            let got = crane_core::ops::topk_indices(&logits, k)?.to_vec1::<u32>()?;
            let want = crane_core::ops::portable::topk_indices(&logits_cpu, k)?.to_vec1::<u32>()?;
            assert_eq!(got, want, "n={n} k={k}");
        }
    }
    Ok(())
}

/// The fused activation against the `narrow + silu + mul` chain it collapses,
/// in every dtype the kernel has a variant for.
#[test]
fn fused_silu_mul_matches_the_op_chain() -> Result<()> {
    let Some(dev) = device() else {
        return Ok(());
    };
    let (rows, inter) = (7usize, 3072usize);

    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let gate_up = Tensor::randn(0f32, 1.0, (1, rows, 2 * inter), &dev)?.to_dtype(dtype)?;
        let fused = crane_core::ops::fused_silu_mul(&gate_up, inter)?;
        assert_eq!(fused.dims(), &[1, rows, inter]);
        assert_eq!(fused.dtype(), dtype);

        let gate = gate_up.narrow(D::Minus1, 0, inter)?;
        let up = gate_up.narrow(D::Minus1, inter, inter)?;
        let want = (candle_nn::ops::silu(&gate)? * up)?;
        assert!(cos_sim(&fused, &want)? >= 0.9999, "dtype={dtype:?}");
    }
    Ok(())
}

/// Greedy decode reads back one index, not the whole vocabulary.
#[test]
fn gpu_argmax_finds_the_largest_logit() -> Result<()> {
    let Some(dev) = device() else {
        return Ok(());
    };
    let logits = Tensor::randn(0f32, 4.0, 1024, &dev)?;
    let got = crane_core::ops::gpu_argmax(&logits)?;
    let want = logits.to_vec1::<f32>()?;
    let want = want
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap();
    assert_eq!(got, want);
    Ok(())
}
