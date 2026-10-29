// SPDX-License-Identifier: MIT
//! Kernel-level timing of the causal MMA flash-attention entries against the
//! matmul SDPA Qwen3.5 falls back to, CK (`flash-attn-rocm`) and FA2's
//! split-KV path (`flash-attn-ab`, BF16 only). Shape: B=1, 8 Q heads, 2 KV
//! heads, `head_dim` 256, at the long-sequence lengths. Timings are not
//! asserted. A column that errors or returns a non-finite output prints
//! `err` and fails the test once the table is complete.
//!
//! Inputs are prepared once per length. Each table starts after a short
//! warm-up, so the first row isn't measured while the GPU is still leaving
//! its idle clocks. Each number is the median wall-clock time per call over
//! batches of back-to-back calls with one sync per batch, so it includes the
//! wrapper's own allocations and uploads. The `mma` columns take Q as F32 and
//! convert it inside the wrapper, so only they pay for that conversion.
//! Outputs drop after every call. If the backend frees synchronously, that
//! adds the same sync to every column.
//!
//! ```text
//! cargo test -p crane-core --no-default-features \
//!     --features "rocm,onnx,flash-attn-rocm" --release --lib fattn_bench \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! On CUDA, `flash-attn-ab` enables the kernels and the `fa2_split` column:
//!
//! ```text
//! cargo test -p crane-core --features "onnx,flash-attn-ab" --release --lib \
//!     fattn_bench -- --ignored --nocapture --test-threads=1
//! ```

use std::time::{Duration, Instant};

use candle_core::{DType, Device, Result, Tensor};

use super::super::fattn::build_additive_mask_f16;
use super::super::fattn::test_support::{
    LONG_SEQ_HKV as HKV, LONG_SEQ_HQ as HQ, LongSeqDist, long_seq_inputs, test_gpu_device,
};
use super::long_seq_tests::{SEQS, round_to};
use super::{
    fattn_mma_bf16_causal, fattn_mma_bf16_causal_with_mask, fattn_mma_causal,
    fattn_mma_causal_with_mask,
};
use crate::models::qwen3_5::grouped_sdpa;

const HEAD_DIM: usize = 256;
const BATCHES: usize = 20;
const CALLS: u32 = 50;
/// How long [`warm_up`] keeps the GPU busy before the first timed row.
const WARM_UP: Duration = Duration::from_secs(1);
/// Table columns, in the order [`bench_table`] fills each row.
const COLUMNS: [&str; 5] = ["mma", "mma_mask", "sdpa", "ck", "fa2_split"];

///// Tests ///////////////////////////////////////////////////////////////

// Prints the BF16 timing table.
#[test]
#[ignore = "needs a CUDA/ROCm GPU, long-running"]
fn fattn_bench_bf16_hd256() {
    bench_table(DType::BF16);
}

// Prints the F16 timing table.
#[test]
#[ignore = "needs a CUDA/ROCm GPU, long-running"]
fn fattn_bench_f16_hd256() {
    bench_table(DType::F16);
}

// The median is the middle value, or the mean of the two middle values.
#[test]
fn median_of_odd_and_even_lengths() {
    assert!((median(&mut [3.0, 1.0, 2.0]) - 2.0).abs() < f64::EPSILON);
    assert!((median(&mut [4.0, 1.0, 3.0, 2.0]) - 2.5).abs() < f64::EPSILON);
}

///// Harness /////////////////////////////////////////////////////////////

/// Prints one row of µs per call for every length in [`SEQS`].
///
/// # Panics
///
/// After the last row, if any column failed. Each failure is reported on
/// stderr with its column and length.
fn bench_table(dtype: DType) {
    let gpu = test_gpu_device();
    // head_dim is 256, exact in an f32.
    #[allow(clippy::cast_precision_loss)]
    let scale = (HEAD_DIM as f32).sqrt().recip();
    warm_up(&gpu).unwrap();
    print!("{dtype:?} hd{HEAD_DIM}, us per call\n{:>6}", "seq");
    for name in COLUMNS {
        print!("{name:>11}");
    }
    println!();
    let mut failures = 0;
    for seq in SEQS {
        let [q_host, k_host, v_host] =
            long_seq_inputs(seq, HEAD_DIM, LongSeqDist::Normal, 0).map(|h| round_to(h, dtype));
        let bshd = |h: &[f32], heads| Tensor::from_slice(h, (1, seq, heads, HEAD_DIM), &gpu);
        // The MMA entries read q as F32. Every other path takes it in dtype.
        let q = bshd(&q_host, HQ).unwrap();
        let q_dt = q.to_dtype(dtype).unwrap();
        let k = bshd(&k_host, HKV).unwrap().to_dtype(dtype).unwrap();
        let v = bshd(&v_host, HKV).unwrap().to_dtype(dtype).unwrap();
        let mask = build_additive_mask_f16(seq, seq, 0, None, 0, &gpu).unwrap();
        // Qwen3.5 builds its mask once per forward, in the attention dtype.
        let sdpa_mask = mask.to_dtype(dtype).unwrap();
        let bhsd = |t: &Tensor| t.transpose(1, 2).unwrap().contiguous().unwrap();
        let (q_h, k_h, v_h) = (bhsd(&q_dt), bhsd(&k), bhsd(&v));
        let bf16 = dtype == DType::BF16;

        let mma = time_us(&gpu, || {
            if bf16 {
                fattn_mma_bf16_causal(&q, &k, &v, scale, 0)
            } else {
                fattn_mma_causal(&q, &k, &v, scale, 0)
            }
        });
        let mma_mask = time_us(&gpu, || {
            if bf16 {
                fattn_mma_bf16_causal_with_mask(&q, &k, &v, scale, 0, &mask)
            } else {
                fattn_mma_causal_with_mask(&q, &k, &v, scale, 0, &mask)
            }
        });
        let sdpa = time_us(&gpu, || {
            grouped_sdpa(
                &q_h,
                &k_h,
                &v_h,
                Some(&sdpa_mask),
                f64::from(scale),
                HQ / HKV,
            )
        });
        #[cfg(feature = "flash-attn-rocm")]
        let ck = Some(time_us(&gpu, || {
            candle_flash_attn_rocm::flash_attn(&q_dt, &k, &v, scale, true)
        }));
        #[cfg(not(feature = "flash-attn-rocm"))]
        let ck = None;
        #[cfg(feature = "flash-attn-ab")]
        let fa2_split = bf16.then(|| {
            time_us(&gpu, || {
                crate::ops::flash_split::causal_bf16_256_auto(&q_dt, &k, &v)
            })
        });
        #[cfg(not(feature = "flash-attn-ab"))]
        let fa2_split = None;

        let cells: [Option<Result<f64>>; COLUMNS.len()] =
            [Some(mma), Some(mma_mask), Some(sdpa), ck, fa2_split];
        print!("{seq:>6}");
        for (name, cell) in COLUMNS.iter().zip(&cells) {
            match cell {
                Some(Ok(t)) => print!("{t:>11.1}"),
                Some(Err(e)) => {
                    failures += 1;
                    eprintln!("{name} seq={seq}: {e}");
                    print!("{:>11}", "err");
                },
                None => print!("{:>11}", "-"),
            }
        }
        println!();
    }
    assert_eq!(failures, 0, "{failures} column(s) failed, see stderr");
}

/// Keeps the GPU busy with large matmuls for [`WARM_UP`]. A GPU coming out
/// of idle runs at low clocks for a while, and without this the first timed
/// row is inflated in every column.
///
/// # Errors
///
/// Returns the first error of the matmul or of a device sync.
fn warm_up(gpu: &Device) -> Result<()> {
    let a = Tensor::ones((2048, 2048), DType::F32, gpu)?;
    let start = Instant::now();
    while start.elapsed() < WARM_UP {
        for _ in 0..10 {
            std::hint::black_box(a.matmul(&a)?);
        }
        gpu.synchronize()?;
    }
    Ok(())
}

/// Median µs per call of `f`, over [`BATCHES`] batches of [`CALLS`]
/// back-to-back calls with one device sync per batch. One untimed batch
/// first absorbs JIT compilation and allocator growth.
///
/// # Errors
///
/// Returns the first error of `f` or of a device sync. Also returns an
/// error if the first output, checked once before timing, is not finite.
fn time_us(gpu: &Device, mut f: impl FnMut() -> Result<Tensor>) -> Result<f64> {
    let sum = f()?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    if !sum.is_finite() {
        candle_core::bail!("non-finite output, sum {sum}");
    }
    let mut batch = || -> Result<()> {
        for _ in 0..CALLS {
            std::hint::black_box(f()?);
        }
        gpu.synchronize()
    };
    batch()?;
    let mut per_call = [0f64; BATCHES];
    for t in &mut per_call {
        let start = Instant::now();
        batch()?;
        *t = start.elapsed().as_secs_f64() * 1e6 / f64::from(CALLS);
    }
    Ok(median(&mut per_call))
}

/// Median of `values`, reordering them.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        f64::midpoint(values[mid - 1], values[mid])
    } else {
        values[mid]
    }
}
