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
//! wrapper's own allocations and uploads. Outputs drop after every call. If
//! the backend frees synchronously, that adds the same sync to every column.
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
    fattn_mma_causal_with_mask, prefill_dispatch, prefill_dispatch_bf16,
};
use crate::models::qwen3_5::grouped_sdpa;
use crate::ops::fused_ops::fattn::{gqa_z_tiles, q_tiles, select_mma_ncols, select_mma_ncols_bf16};

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

// Compares today's auto-selected stream-K grid against a forced
// single-wave (one block per tile, no fixup) grid at the shapes stream-K
// targets, printing a bitwise-match check between the two.
#[test]
#[ignore = "needs a CUDA/ROCm GPU"]
fn forced_grid_split_vs_no_split_hd256() {
    let gpu = test_gpu_device();
    #[allow(clippy::cast_precision_loss)]
    let scale = (HEAD_DIM as f32).sqrt().recip();
    let gqa_ratio = HQ / HKV;
    for seq in [
        231usize, 238, 257, 311, 401, 503, 601, 701, 809, 907, 1009, 1103, 1301, 1511, 1709, 2003,
        2503, 3001, 3511, 4001, 4201, 4507, 5003, 6007, 8009, 10007,
    ] {
        for bf16 in [false, true] {
            let dtype = if bf16 { DType::BF16 } else { DType::F16 };
            let [q_host, k_host, v_host] =
                long_seq_inputs(seq, HEAD_DIM, LongSeqDist::Normal, 0).map(|h| round_to(h, dtype));
            let bshd = |h: &[f32], heads| Tensor::from_slice(h, (1, seq, heads, HEAD_DIM), &gpu);
            let q = bshd(&q_host, HQ).unwrap().to_dtype(dtype).unwrap();
            let k = bshd(&k_host, HKV).unwrap().to_dtype(dtype).unwrap();
            let v = bshd(&v_host, HKV).unwrap().to_dtype(dtype).unwrap();

            let ncols = if bf16 {
                select_mma_ncols_bf16(HEAD_DIM, gqa_ratio, seq, false)
            } else {
                select_mma_ncols(HEAD_DIM, gqa_ratio, seq, false)
            };
            let Some((ncols1, ncols2)) = ncols else {
                eprintln!("[fattn_grid_trace] seq={seq} bf16={bf16}: no kernel variant, skipped");
                continue;
            };
            let ntiles = q_tiles(seq, ncols1) * gqa_z_tiles(gqa_ratio, ncols2) * HKV;

            let mut last_output = Vec::new();
            for (mode, forced_grid) in [("auto", None), ("no_split", Some(ntiles))] {
                eprintln!(
                    "[fattn_grid_trace] seq={seq} bf16={bf16} mode={mode} ntiles={ntiles} ---"
                );
                let dispatch_once = || {
                    if bf16 {
                        prefill_dispatch_bf16(&q, &k, &v, None, None, scale, Some(0), forced_grid)
                    } else {
                        prefill_dispatch(&q, &k, &v, None, None, scale, Some(0), forced_grid)
                    }
                    .unwrap()
                };
                // Warm up (HIPRTC JIT, allocator) before timing.
                for _ in 0..10 {
                    dispatch_once();
                }
                gpu.synchronize().unwrap();
                const CALLS: u32 = 50;
                let start = std::time::Instant::now();
                let mut out = None;
                for _ in 0..CALLS {
                    out = Some(dispatch_once());
                }
                gpu.synchronize().unwrap();
                let per_call_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(CALLS);
                eprintln!(
                    "[fattn_grid_trace] seq={seq} bf16={bf16} mode={mode}: {per_call_us:.1} us/call"
                );
                let vec = out
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                if mode == "auto" {
                    last_output = vec;
                } else {
                    let matches = last_output == vec;
                    eprintln!(
                        "[fattn_grid_trace] seq={seq} bf16={bf16}: auto vs no_split bitwise \
                         match={matches}"
                    );
                }
            }
        }
    }
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
        let q = bshd(&q_host, HQ).unwrap().to_dtype(dtype).unwrap();
        let k = bshd(&k_host, HKV).unwrap().to_dtype(dtype).unwrap();
        let v = bshd(&v_host, HKV).unwrap().to_dtype(dtype).unwrap();
        let mask = build_additive_mask_f16(seq, seq, 0, None, 0, &gpu).unwrap();
        // Qwen3.5 builds its mask once per forward, in the attention dtype.
        let sdpa_mask = mask.to_dtype(dtype).unwrap();
        let bhsd = |t: &Tensor| t.transpose(1, 2).unwrap().contiguous().unwrap();
        let (q_h, k_h, v_h) = (bhsd(&q), bhsd(&k), bhsd(&v));
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
            candle_flash_attn_rocm::flash_attn(&q, &k, &v, scale, true)
        }));
        #[cfg(not(feature = "flash-attn-rocm"))]
        let ck = None;
        #[cfg(feature = "flash-attn-ab")]
        let fa2_split = bf16.then(|| {
            time_us(&gpu, || {
                crate::ops::flash_split::causal_bf16_256_auto(&q, &k, &v)
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
