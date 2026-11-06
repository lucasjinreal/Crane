// SPDX-License-Identifier: MIT
//! Kernel-level timing of the causal MMA flash-attention entries against the
//! matmul SDPA Qwen3.5 falls back to, CK (`flash-attn-rocm`) and FA2's
//! split-KV path (`flash-attn-ab`, BF16 only). Shape: B=1, 8 Q heads, 2 KV
//! heads, `head_dim` 256, at the long-sequence lengths. Timings are not
//! asserted. A column that errors or returns a non-finite output prints
//! `err` and fails the test once the table is complete.
//! `fattn_bench_gqa_shapes` times only the in-kernel causal entries, at the
//! GQA shapes of Qwen3.5-0.8B and Qwen3-4B up to 32k.
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
    LaunchOverride, fattn_mma_bf16_causal, fattn_mma_bf16_causal_with_mask, fattn_mma_causal,
    fattn_mma_causal_with_mask, prefill_dispatch, prefill_dispatch_bf16,
};
use crate::models::qwen3_5::grouped_sdpa;
use crate::ops::fused_ops::fattn::{
    gqa_z_tiles, is_amd_backend, q_tiles, select_mma_ncols, select_mma_ncols_bf16,
};

const HEAD_DIM: usize = 256;
const BATCHES: usize = 20;
const CALLS: u32 = 50;
/// How long [`warm_up`] keeps the GPU busy before the first timed row.
const WARM_UP: Duration = Duration::from_secs(1);
/// Table columns, in the order [`bench_table`] fills each row.
const COLUMNS: [&str; 5] = ["mma", "mma_mask", "sdpa", "ck", "fa2_split"];
/// `(Q heads, KV heads, head_dim)` of Qwen3.5-0.8B and Qwen3-4B.
const GQA_SHAPES: [(usize, usize, usize); 2] = [(8, 2, 256), (32, 8, 128)];
/// From under one wave of tiles up to long-context prefill.
const GQA_SEQS: [usize; 6] = [512, 1024, 4096, 8192, 16384, 32768];
/// Lengths of [`fattn_bench_tile_order`]. For both GQA shapes and batch
/// sizes they reach past [`super::HEAVY_FIRST_MAX_TILES`].
const TILE_ORDER_SEQS: [usize; 5] = [1024, 2048, 4096, 8192, 16384];
/// Timed batches per cell of [`fattn_bench_tile_order`].
const TILE_ORDER_BATCHES: usize = 9;

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

// Prints the in-kernel causal timing table of every shape in
// `GQA_SHAPES`, with each launch's tile count.
#[test]
#[ignore = "needs a CUDA/ROCm GPU, long-running"]
fn fattn_bench_gqa_shapes() {
    let gpu = test_gpu_device();
    warm_up(&gpu).unwrap();
    let mut failures = 0;
    for (hq, hkv, d) in GQA_SHAPES {
        failures += bench_gqa_shape(&gpu, hq, hkv, d);
    }
    assert_eq!(failures, 0, "{failures} cell(s) failed, see stderr");
}

// Prints, for every shape in `GQA_SHAPES` at batch 1 and 2, the µs per
// call of a one-block-per-tile in-kernel causal launch in each tile order:
// stream-K and heavy-first. Rows show the tile count, so the
// `HEAVY_FIRST_MAX_TILES` cutoff can be checked on any GPU.
#[test]
#[ignore = "needs a CUDA/ROCm GPU, long-running"]
fn fattn_bench_tile_order() {
    let gpu = test_gpu_device();
    warm_up(&gpu).unwrap();
    let mut failures = 0;
    for (hq, hkv, d) in GQA_SHAPES {
        for batch in [1, 2] {
            failures += bench_tile_order(&gpu, hq, hkv, d, batch);
        }
    }
    assert_eq!(failures, 0, "{failures} cell(s) failed, see stderr");
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
                let forced = LaunchOverride {
                    grid: forced_grid,
                    heavy_first: None,
                };
                let dispatch_once = || {
                    if bf16 {
                        prefill_dispatch_bf16(&q, &k, &v, None, None, scale, Some(0), forced)
                    } else {
                        prefill_dispatch(&q, &k, &v, None, None, scale, Some(0), forced)
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

/// Prints one table of µs per call of the in-kernel causal entries for
/// `hq` Q heads, `hkv` KV heads and head dim `d`, one row per length in
/// [`GQA_SEQS`]. A dtype with no kernel variant prints `-`.
///
/// Returns the number of failed cells, each reported on stderr.
fn bench_gqa_shape(gpu: &Device, hq: usize, hkv: usize, d: usize) -> usize {
    // long_seq_inputs is fixed at HQ Q and HKV KV heads. With the same
    // ratio, seq * hq / HQ of its rows hold exactly seq rows of hq and hkv
    // heads.
    assert!(
        hq.is_multiple_of(HQ) && hq * HKV == hkv * HQ,
        "{hq}Q/{hkv}KV: not a multiple of {HQ}Q/{HKV}KV"
    );
    let gqa_ratio = hq / hkv;
    let is_amd = is_amd_backend();
    // head_dim is at most 512, exact in an f32.
    #[allow(clippy::cast_precision_loss)]
    let scale = (d as f32).sqrt().recip();
    println!("{hq}Q/{hkv}KV hd{d}, us per call");
    println!(
        "{:>6}{:>10}{:>11}{:>10}{:>11}",
        "seq", "f16 tiles", "f16", "bf16 tiles", "bf16"
    );
    let mut failures = 0;
    for seq in GQA_SEQS {
        let [q_host, k_host, v_host] = long_seq_inputs(seq * hq / HQ, d, LongSeqDist::Normal, 0);
        let bshd = |host: &[f32], heads| Tensor::from_slice(host, (1, seq, heads, d), gpu).unwrap();
        let (q32, k32, v32) = (bshd(&q_host, hq), bshd(&k_host, hkv), bshd(&v_host, hkv));
        print!("{seq:>6}");
        for dtype in [DType::F16, DType::BF16] {
            let bf16 = dtype == DType::BF16;
            let select = if bf16 {
                select_mma_ncols_bf16
            } else {
                select_mma_ncols
            };
            let Some((ncols1, ncols2)) = select(d, gqa_ratio, seq, is_amd) else {
                print!("{:>10}{:>11}", "-", "-");
                continue;
            };
            let ntiles = q_tiles(seq, ncols1) * gqa_z_tiles(gqa_ratio, ncols2) * hkv;
            let [q, k, v] = [&q32, &k32, &v32].map(|src| src.to_dtype(dtype).unwrap());
            let timed = time_us(gpu, || {
                if bf16 {
                    fattn_mma_bf16_causal(&q, &k, &v, scale, 0)
                } else {
                    fattn_mma_causal(&q, &k, &v, scale, 0)
                }
            });
            match timed {
                Ok(us) => print!("{ntiles:>10}{us:>11.1}"),
                Err(err) => {
                    failures += 1;
                    eprintln!("{hq}Q/{hkv}KV hd{d} {dtype:?} seq={seq}: {err}");
                    print!("{ntiles:>10}{:>11}", "err");
                },
            }
        }
        println!();
    }
    failures
}

/// Prints one table of µs per call for `hq` Q heads, `hkv` KV heads, head
/// dim `d` and `batch`, one row per dtype and length in
/// [`TILE_ORDER_SEQS`]. Columns are the stream-K order and heavy-first.
///
/// Returns the number of failed cells, each reported on stderr.
fn bench_tile_order(gpu: &Device, hq: usize, hkv: usize, d: usize, batch: usize) -> usize {
    assert!(
        hq.is_multiple_of(HQ) && hq * HKV == hkv * HQ,
        "{hq}Q/{hkv}KV: not a multiple of {HQ}Q/{HKV}KV"
    );
    let gqa_ratio = hq / hkv;
    let is_amd = is_amd_backend();
    // head_dim is at most 512, exact in an f32.
    #[allow(clippy::cast_precision_loss)]
    let scale = (d as f32).sqrt().recip();
    println!("{hq}Q/{hkv}KV hd{d} batch {batch}, us per call");
    println!(
        "{:>6}{:>7}{:>7}{:>11}{:>12}",
        "dtype", "seq", "tiles", "stream_k", "heavy_first"
    );
    let mut failures = 0;
    for seq in TILE_ORDER_SEQS {
        let rows = seq * batch * hq / HQ;
        let [q_host, k_host, v_host] = long_seq_inputs(rows, d, LongSeqDist::Normal, 0);
        let bshd =
            |host: &[f32], heads| Tensor::from_slice(host, (batch, seq, heads, d), gpu).unwrap();
        let (q32, k32, v32) = (bshd(&q_host, hq), bshd(&k_host, hkv), bshd(&v_host, hkv));
        // Fewer calls at long lengths, where one call already takes long.
        let calls = u32::try_from(2048 * 50 / (seq * batch))
            .unwrap()
            .clamp(3, CALLS);
        for dtype in [DType::F16, DType::BF16] {
            let bf16 = dtype == DType::BF16;
            let select = if bf16 {
                select_mma_ncols_bf16
            } else {
                select_mma_ncols
            };
            let Some((ncols1, ncols2)) = select(d, gqa_ratio, seq, is_amd) else {
                continue;
            };
            let ntiles = q_tiles(seq, ncols1) * gqa_z_tiles(gqa_ratio, ncols2) * hkv * batch;
            let [q, k, v] = [&q32, &k32, &v32].map(|src| src.to_dtype(dtype).unwrap());
            print!("{:>6}{seq:>7}{ntiles:>7}", format!("{dtype:?}"));
            for heavy_first in [false, true] {
                let forced = LaunchOverride {
                    grid: Some(ntiles),
                    heavy_first: Some(heavy_first),
                };
                let timed = time_us_with(gpu, TILE_ORDER_BATCHES, calls, || {
                    if bf16 {
                        prefill_dispatch_bf16(&q, &k, &v, None, None, scale, Some(0), forced)
                    } else {
                        prefill_dispatch(&q, &k, &v, None, None, scale, Some(0), forced)
                    }
                });
                match timed {
                    Ok(us) => print!("{us:>12.1}"),
                    Err(err) => {
                        failures += 1;
                        eprintln!("{dtype:?} seq={seq} heavy_first={heavy_first}: {err}");
                        print!("{:>12}", "err");
                    },
                }
            }
            println!();
        }
    }
    failures
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

/// [`time_us_with`] over [`BATCHES`] batches of [`CALLS`] calls.
///
/// # Errors
///
/// As [`time_us_with`].
fn time_us(gpu: &Device, f: impl FnMut() -> Result<Tensor>) -> Result<f64> {
    time_us_with(gpu, BATCHES, CALLS, f)
}

/// Median µs per call of `f`, over `batches` batches of `calls`
/// back-to-back calls with one device sync per batch. One untimed batch
/// first absorbs JIT compilation and allocator growth.
///
/// # Errors
///
/// Returns the first error of `f` or of a device sync. Also returns an
/// error if the first output, checked once before timing, is not finite.
fn time_us_with(
    gpu: &Device,
    batches: usize,
    calls: u32,
    mut f: impl FnMut() -> Result<Tensor>,
) -> Result<f64> {
    let sum = f()?.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    if !sum.is_finite() {
        candle_core::bail!("non-finite output, sum {sum}");
    }
    let mut batch = || -> Result<()> {
        for _ in 0..calls {
            std::hint::black_box(f()?);
        }
        gpu.synchronize()
    };
    batch()?;
    let mut per_call = vec![0f64; batches];
    for t in &mut per_call {
        let start = Instant::now();
        batch()?;
        *t = start.elapsed().as_secs_f64() * 1e6 / f64::from(calls);
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
