// SPDX-License-Identifier: MIT
//! Kernel-level timing of the causal MMA flash-attention entries against the
//! matmul SDPA Qwen3.5 falls back to, CK (`flash-attn-rocm`) and FA2's
//! split-KV path (`flash-attn-ab`, BF16 only). Shape: B=1, 8 Q heads, 2 KV
//! heads, `head_dim` 256, at the long-sequence lengths. Timings are not
//! asserted. A column that errors or returns a non-finite output prints
//! `err` and fails the test once the table is complete.
//! `fattn_bench_gqa_shapes` times only the in-kernel causal entries, at the
//! GQA shapes of Qwen3.5-0.8B, Qwen3-1.7B and Qwen3-4B up to 64k. Each row
//! times the stream-K and the heavy-first tile order of the same launch, in
//! alternating batches, and prints the order the launch policy picks.
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
    LaunchOverride, TileCost, fattn_mma_bf16_causal, fattn_mma_bf16_causal_with_mask,
    fattn_mma_causal, fattn_mma_causal_with_mask, prefill_dispatch, prefill_dispatch_bf16,
    use_heavy_first,
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
/// `(Q heads, KV heads, head_dim)` of Qwen3.5-0.8B, Qwen3-1.7B and
/// Qwen3-4B. The last two share their KV head count and differ in tiles.
const GQA_SHAPES: [(usize, usize, usize); 3] = [(8, 2, 256), (16, 8, 128), (32, 8, 128)];
/// From under one wave of tiles up to long-context prefill.
const GQA_SEQS: [usize; 10] = [
    512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 65536,
];
/// Lengths of [`fattn_bench_tile_order`]. For both GQA shapes and batch
/// sizes they reach past [`super::HEAVY_FIRST_MAX_TILES`].
const TILE_ORDER_SEQS: [usize; 5] = [1024, 2048, 4096, 8192, 16384];
/// Timed batches per cell of [`fattn_bench_tile_order`].
const TILE_ORDER_BATCHES: usize = 9;
/// Rounds of [`time_pair_us`], each timing one batch of every launch.
const PAIR_ROUNDS: usize = 9;
/// How long [`time_pair_us`] aims to make one batch, in µs.
const PAIR_BATCH_US: f64 = 100_000.0;

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

// Prints, for every shape in `GQA_SHAPES`, the in-kernel causal timing of
// both tile orders with each launch's tile count.
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

/// Prints, for `hq` Q heads, `hkv` KV heads and head dim `d`, one table per
/// dtype of µs per call of the in-kernel causal entry with each tile order,
/// one row per length in [`GQA_SEQS`]. `delta%` is heavy-first against
/// stream-K, negative when it is faster. `gate` is the order the launch
/// policy picks. A length with no kernel variant prints `-`.
///
/// Returns the number of failed rows, each reported on stderr.
fn bench_gqa_shape(gpu: &Device, hq: usize, hkv: usize, d: usize) -> usize {
    let gqa_ratio = hq / hkv;
    let is_amd = is_amd_backend();
    // head_dim is at most 512, exact in an f32.
    #[allow(clippy::cast_precision_loss)]
    let scale = (d as f32).sqrt().recip();
    let mut failures = 0;
    for dtype in [DType::F16, DType::BF16] {
        let bf16 = dtype == DType::BF16;
        let select = if bf16 {
            select_mma_ncols_bf16
        } else {
            select_mma_ncols
        };
        println!("{hq}Q/{hkv}KV hd{d} {dtype:?}, us per call");
        println!(
            "{:>6}{:>8}{:>11}{:>11}{:>8}{:>9}",
            "seq", "tiles", "stream_k", "heavy", "delta%", "gate"
        );
        for seq in GQA_SEQS {
            let Some((ncols1, ncols2)) = select(d, gqa_ratio, seq, is_amd) else {
                println!("{seq:>6}{:>8}", "-");
                continue;
            };
            let ntiles = q_tiles(seq, ncols1) * gqa_z_tiles(gqa_ratio, ncols2) * hkv;
            // Timing does not depend on the values, so skip the host
            // generators and the upload.
            let randn = |heads| {
                Tensor::randn(0f32, 1f32, (1, seq, heads, d), gpu)
                    .and_then(|t| t.to_dtype(dtype))
                    .unwrap()
            };
            let (q, k, v) = (randn(hq), randn(hkv), randn(hkv));
            let launch = |heavy| {
                let forced = LaunchOverride {
                    grid: None,
                    heavy_first: Some(heavy),
                };
                if bf16 {
                    prefill_dispatch_bf16(&q, &k, &v, None, None, scale, Some(0), forced)
                } else {
                    prefill_dispatch(&q, &k, &v, None, None, scale, Some(0), forced)
                }
            };
            match time_pair_us(gpu, || launch(false), || launch(true)) {
                Ok((stream_k, heavy)) => {
                    let gate = if use_heavy_first(TileCost::GrowsWithJt, ntiles, ntiles, None) {
                        "heavy"
                    } else {
                        "stream_k"
                    };
                    println!(
                        "{seq:>6}{ntiles:>8}{stream_k:>11.1}{heavy:>11.1}{:>+8.1}{gate:>9}",
                        (heavy / stream_k - 1.0) * 100.0
                    );
                },
                Err(err) => {
                    failures += 1;
                    eprintln!("{hq}Q/{hkv}KV hd{d} {dtype:?} seq={seq}: {err}");
                    println!("{seq:>6}{ntiles:>8}{:>11}", "err");
                },
            }
        }
    }
    failures
}

/// Prints one table of µs per call for `hq` Q heads, `hkv` KV heads, head
/// dim `d` and `batch`, one row per dtype and length in
/// [`TILE_ORDER_SEQS`]. Columns are the stream-K order and heavy-first.
///
/// Returns the number of failed cells, each reported on stderr.
fn bench_tile_order(gpu: &Device, hq: usize, hkv: usize, d: usize, batch: usize) -> usize {
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
        // Timing does not depend on the values, so skip the host generator
        // and the upload (see bench_gqa_shape): that also sidesteps
        // long_seq_inputs's fixed LONG_SEQ_HQ/LONG_SEQ_HKV head counts,
        // which only reinterpret cleanly into (hq, hkv) when hq/hkv shares
        // LONG_SEQ_HQ/LONG_SEQ_HKV's GQA ratio.
        let randn = |heads| Tensor::randn(0f32, 1f32, (batch, seq, heads, d), gpu).unwrap();
        let (q32, k32, v32) = (randn(hq), randn(hkv), randn(hkv));
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
    ensure_finite(&f()?)?;
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

/// Median µs per call of `a` and of `b`, over [`PAIR_ROUNDS`] rounds that
/// each time one batch of `a` and one of `b`, in alternating order. A clock
/// drift of the GPU then hits both the same. Batches hold as many calls as
/// fill [`PAIR_BATCH_US`], at most [`CALLS`], and at least one. One untimed
/// batch of each first absorbs JIT compilation and allocator growth.
///
/// # Errors
///
/// Returns the first error of `a`, `b` or a device sync. Also returns an
/// error if the first output of either, checked once before timing, is not
/// finite.
fn time_pair_us(
    gpu: &Device,
    mut a: impl FnMut() -> Result<Tensor>,
    mut b: impl FnMut() -> Result<Tensor>,
) -> Result<(f64, f64)> {
    ensure_finite(&a()?)?;
    ensure_finite(&b()?)?;
    let one_call = time_batch(gpu, &mut a, 1)?.max(time_batch(gpu, &mut b, 1)?);
    let calls = (PAIR_BATCH_US / one_call).clamp(1.0, f64::from(CALLS));
    // The clamp bounds the value to 1..=CALLS.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let calls = calls as u32;
    time_batch(gpu, &mut a, calls)?;
    time_batch(gpu, &mut b, calls)?;
    let (mut a_us, mut b_us) = ([0f64; PAIR_ROUNDS], [0f64; PAIR_ROUNDS]);
    for round in 0..PAIR_ROUNDS {
        if round % 2 == 0 {
            a_us[round] = time_batch(gpu, &mut a, calls)?;
            b_us[round] = time_batch(gpu, &mut b, calls)?;
        } else {
            b_us[round] = time_batch(gpu, &mut b, calls)?;
            a_us[round] = time_batch(gpu, &mut a, calls)?;
        }
    }
    Ok((median(&mut a_us), median(&mut b_us)))
}

/// µs per call of `calls` back-to-back calls of `f` with one device sync.
///
/// # Errors
///
/// Returns the first error of `f` or of the sync.
fn time_batch(gpu: &Device, f: &mut impl FnMut() -> Result<Tensor>, calls: u32) -> Result<f64> {
    let start = Instant::now();
    for _ in 0..calls {
        std::hint::black_box(f()?);
    }
    gpu.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e6 / f64::from(calls))
}

/// Fails if the sum of `out` is not finite, which catches a kernel that
/// left its output unwritten or blew up.
///
/// # Errors
///
/// Returns an error for a non-finite sum, or if the reduction fails.
fn ensure_finite(out: &Tensor) -> Result<()> {
    let sum = out.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    if !sum.is_finite() {
        candle_core::bail!("non-finite output, sum {sum}");
    }
    Ok(())
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
