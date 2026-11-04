// SPDX-License-Identifier: MIT
//! Long-sequence regression tests for the MMA flash-attention kernels with
//! 8 Q heads and 2 KV heads, at `head_dim` 256 (Qwen3.5's full-attention
//! shape), 128 and 64.
//!
//! Every row of every head is checked against a GPU F32 matmul reference,
//! and sampled rows against a CPU F64 one. Both take Q already scaled and
//! rounded to the input dtype, as the kernels store it. Each input runs
//! twice and must give bitwise-identical output. Run on a GPU, in release
//! mode:
//!
//! ```text
//! cargo test -p crane-core --no-default-features --features "rocm,onnx" \
//!     --release --lib long_seq -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `CRANE_LONG_SEQ_SEQS` and `CRANE_LONG_SEQ_SEEDS` narrow the sweep, for
//! example for a `compute-sanitizer` run on the test binary:
//!
//! ```text
//! CRANE_LONG_SEQ_SEQS=1024 CRANE_LONG_SEQ_SEEDS=2 compute-sanitizer \
//!     --tool racecheck <test binary> long_seq_bf16_hd128 --ignored --nocapture
//! ```

use std::collections::BTreeMap;

use candle_core::{DType, Device, Tensor};

use super::super::fattn::test_support::{
    LONG_SEQ_HKV as HKV, LONG_SEQ_HQ as HQ, LongSeqDist, MaskMode, attention_rows_f64,
    long_seq_inputs, naive_attention, splitmix64, test_gpu_device,
};
use super::super::fattn::{
    build_additive_mask_f16, is_amd_backend, mma_config, mma_config_bf16, select_mma_ncols,
    select_mma_ncols_bf16,
};
use super::super::fattn_long_seq_host::{
    Comparison, RowFailure, Tolerance, compare, parse_seq_list,
};
use super::{
    fattn_mma_bf16_causal, fattn_mma_bf16_causal_with_mask, fattn_mma_causal,
    fattn_mma_causal_with_mask,
};
use crate::models::modules::gpu_flash_attn::try_causal_prefill;

/// 231/238 are unaligned to `FATTN_KQ_STRIDE`, exercising the kernel's
/// partial-tile bounds checks; 1024 and up are aligned.
pub(super) const SEQS: [usize; 5] = [231, 238, 1024, 2048, 4096];
/// Input distributions of the sweep.
pub(super) const DISTS: [LongSeqDist; 3] =
    [LongSeqDist::Normal, LongSeqDist::Scaled, LongSeqDist::Spike];
/// Seeds per distribution. Distribution `i` uses seeds `i * SEEDS_PER_DIST`
/// and up.
pub(super) const SEEDS_PER_DIST: u64 = 10;
/// Rows per block in the F64 check, which samples each block's first and
/// last row.
const F64_BLOCK: usize = 64;
const F64_RANDOM_ROWS: usize = 16;
const WORST_ROWS: usize = 20;

///// Tests ///////////////////////////////////////////////////////////////

// The native-bf16 kernel matches both references at every length and seed,
// and gives the same bits on every run, at head_dim 256.
#[test]
#[ignore = "needs a CUDA/ROCm GPU, long-running"]
fn long_seq_bf16_hd256() {
    run_long_seq(DType::BF16, 256);
}

// As `long_seq_bf16_hd256`, at head_dim 128.
#[test]
#[ignore = "needs a CUDA/ROCm GPU, long-running"]
fn long_seq_bf16_hd128() {
    run_long_seq(DType::BF16, 128);
}

// As `long_seq_bf16_hd256`, at head_dim 64.
#[test]
#[ignore = "needs a CUDA/ROCm GPU, long-running"]
fn long_seq_bf16_hd64() {
    run_long_seq(DType::BF16, 64);
}

// The f16 kernel matches both references at every length and seed, and
// gives the same bits on every run, at head_dim 256.
#[test]
#[ignore = "needs a CUDA/ROCm GPU, long-running"]
fn long_seq_f16_hd256() {
    run_long_seq(DType::F16, 256);
}

// As `long_seq_f16_hd256`, at head_dim 128.
#[test]
#[ignore = "needs a CUDA/ROCm GPU, long-running"]
fn long_seq_f16_hd128() {
    run_long_seq(DType::F16, 128);
}

// As `long_seq_f16_hd256`, at head_dim 64.
#[test]
#[ignore = "needs a CUDA/ROCm GPU, long-running"]
fn long_seq_f16_hd64() {
    run_long_seq(DType::F16, 64);
}

///// Runner //////////////////////////////////////////////////////////////

/// Checks every length, distribution and seed at `head_dim`, prints one
/// summary line per length, and panics at the end if anything failed.
/// [`sweep_from_env`] narrows the sweep.
fn run_long_seq(dtype: DType, head_dim: usize) {
    let gpu = test_gpu_device();
    let (seqs, seeds_per_dist) = sweep_from_env();
    let mut report = Report::default();
    for seq in seqs {
        let geom = Geometry::new(dtype, head_dim, seq);
        let mask = build_additive_mask_f16(seq, seq, 0, None, 0, &gpu).unwrap();
        let mut summary = Summary::default();
        for (dist_idx, &dist) in DISTS.iter().enumerate() {
            for idx in 0..seeds_per_dist {
                let case = Case {
                    seq,
                    head_dim,
                    dist,
                    dist_idx,
                    seed: dist_idx as u64 * SEEDS_PER_DIST + idx,
                    first_in_dist: idx == 0,
                    dtype,
                    geom,
                };
                check_case(&case, &mask, &gpu, &mut summary, &mut report);
            }
        }
        println!("{dtype:?} hd{head_dim} seq={seq} {geom}: {summary}");
        report.failed |= summary.failed();
    }
    report.finish(dtype, head_dim);
}

/// The lengths and seeds per distribution to run. The defaults are [`SEQS`]
/// and [`SEEDS_PER_DIST`]. A sanitizer run is far too slow for the full
/// sweep, so two environment variables narrow it:
/// `CRANE_LONG_SEQ_SEQS=1024,2048` and `CRANE_LONG_SEQ_SEEDS=2`. Seed
/// numbers do not depend on the override, so a narrowed run replays inputs
/// of the full one.
///
/// # Panics
///
/// Panics on a value that does not parse, or on a seed count outside
/// `1..=SEEDS_PER_DIST`.
fn sweep_from_env() -> (Vec<usize>, u64) {
    let seqs = std::env::var("CRANE_LONG_SEQ_SEQS").map_or_else(
        |_| SEQS.to_vec(),
        |s| parse_seq_list(&s).unwrap_or_else(|e| panic!("CRANE_LONG_SEQ_SEQS={s:?}: {e}")),
    );
    let seeds = std::env::var("CRANE_LONG_SEQ_SEEDS").map_or(SEEDS_PER_DIST, |s| {
        s.trim()
            .parse()
            .unwrap_or_else(|e| panic!("CRANE_LONG_SEQ_SEEDS={s:?}: {e}"))
    });
    assert!(
        (1..=SEEDS_PER_DIST).contains(&seeds),
        "CRANE_LONG_SEQ_SEEDS must be in 1..={SEEDS_PER_DIST}, got {seeds}"
    );
    (seqs, seeds)
}

/// One input of the sweep.
#[derive(Debug, Clone, Copy)]
struct Case {
    seq: usize,
    head_dim: usize,
    dist: LongSeqDist,
    /// Position of `dist` in [`DISTS`].
    dist_idx: usize,
    seed: u64,
    /// Whether this is the distribution's first seed, the one the slow F64
    /// check runs on.
    first_in_dist: bool,
    dtype: DType,
    geom: Geometry,
}

/// Runs both kernel entries and `try_causal_prefill` on one input and
/// folds the results into `summary` and `report`.
fn check_case(
    case: &Case,
    mask: &Tensor,
    gpu: &Device,
    summary: &mut Summary,
    report: &mut Report,
) {
    let inputs = Inputs::new(case, gpu);
    let bshd = |h: &[f32], heads| Tensor::from_slice(h, (1, case.seq, heads, case.head_dim), gpu);
    // kv_offset, window_left and window_right are all 0 for plain causal.
    let want = naive_attention(
        &bshd(&inputs.q_ref, HQ).unwrap(),
        &bshd(&inputs.k_host, HKV).unwrap(),
        &bshd(&inputs.v_host, HKV).unwrap(),
        1.0,
        MaskMode::Causal,
        0,
        0,
        0,
    );
    let got = check_kernel(case, &inputs, mask, &want, summary, report);
    check_prefill(case, &inputs, &got, &want, summary, report);
    // CPU F64 is slow, so it checks the first seed of each distribution.
    if case.first_in_dist {
        check_f64(case, &inputs, &got, &want, summary, report);
    }
}

/// The inputs of one [`Case`], on the GPU as the kernels read them and on
/// the host for the references.
struct Inputs {
    /// BSHD in the input dtype, as the kernels read them.
    q: Tensor,
    k: Tensor,
    v: Tensor,
    /// `q` pre-scaled as the kernels hold it, which both references take.
    q_ref: Vec<f32>,
    k_host: Vec<f32>,
    v_host: Vec<f32>,
    scale: f32,
}

impl Inputs {
    /// Generates `case`'s inputs, rounded to its dtype, and uploads them.
    fn new(case: &Case, gpu: &Device) -> Self {
        let (seq, d, dtype) = (case.seq, case.head_dim, case.dtype);
        // head_dim is at most 256, exact in an f32.
        #[allow(clippy::cast_precision_loss)]
        let scale = (d as f32).sqrt().recip();
        let [q_host, k_host, v_host] =
            long_seq_inputs(seq, d, case.dist, case.seed).map(|h| round_to(h, dtype));
        let bshd = |h: &[f32], heads| Tensor::from_slice(h, (1, seq, heads, d), gpu).unwrap();
        Self {
            q: bshd(&q_host, HQ).to_dtype(dtype).unwrap(),
            k: bshd(&k_host, HKV).to_dtype(dtype).unwrap(),
            v: bshd(&v_host, HKV).to_dtype(dtype).unwrap(),
            q_ref: scaled_q(&q_host, scale, dtype),
            k_host,
            v_host,
            scale,
        }
    }
}

/// Runs the causal kernel entry twice and the `_with_mask` entry once,
/// checks them against `want` and each other, and returns the first run's
/// output.
fn check_kernel(
    case: &Case,
    inputs: &Inputs,
    mask: &Tensor,
    want: &[f32],
    summary: &mut Summary,
    report: &mut Report,
) -> Vec<f32> {
    let Inputs { q, k, v, scale, .. } = inputs;
    let scale = *scale;
    // Both entries take the same F16 mask: the bf16 one casts any mask to
    // F16 and builds an F16 one itself in `fattn_mma_bf16_causal`.
    let (got, again, with_mask) = if case.dtype == DType::BF16 {
        (
            fattn_mma_bf16_causal(q, k, v, scale, 0),
            fattn_mma_bf16_causal(q, k, v, scale, 0),
            fattn_mma_bf16_causal_with_mask(q, k, v, scale, 0, mask),
        )
    } else {
        (
            fattn_mma_causal(q, k, v, scale, 0),
            fattn_mma_causal(q, k, v, scale, 0),
            fattn_mma_causal_with_mask(q, k, v, scale, 0, mask),
        )
    };
    let got = to_host(&got.unwrap());
    let nondeterministic = !bitwise_eq(&got, &to_host(&again.unwrap()));
    let with_mask_mismatch = !bitwise_eq(&got, &to_host(&with_mask.unwrap()));
    let cmp = compare(&got, want, HQ, case.head_dim, tolerance(case.dtype));
    summary.record_kernel(case.dist_idx, &cmp, nondeterministic, with_mask_mismatch);
    report.add(case, "kernel", &cmp.failures);
    got
}

/// Runs `try_causal_prefill` the exact way models do, BHSD in the input
/// dtype, and checks it against `want` and the kernel output `got`.
fn check_prefill(
    case: &Case,
    inputs: &Inputs,
    got: &[f32],
    want: &[f32],
    summary: &mut Summary,
    report: &mut Report,
) {
    let bhsd = |t: &Tensor| {
        t.to_dtype(case.dtype)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
    };
    let prefill = try_causal_prefill(
        &bhsd(&inputs.q),
        &bhsd(&inputs.k),
        &bhsd(&inputs.v),
        inputs.scale,
        None,
    )
    .unwrap()
    .expect("try_causal_prefill declined a supported shape");
    let prefill = to_host(&prefill);
    summary.prefill_mismatch += usize::from(!bitwise_eq(got, &prefill));
    let cmp = compare(&prefill, want, HQ, case.head_dim, tolerance(case.dtype));
    summary.add_max(&cmp);
    summary.prefill_failed += usize::from(!cmp.failures.is_empty());
    report.add(case, "try_causal_prefill", &cmp.failures);
}

/// Checks sampled rows of the kernel output `got`, and of the F32
/// reference `want`, against a CPU F64 reference.
fn check_f64(
    case: &Case,
    inputs: &Inputs,
    got: &[f32],
    want: &[f32],
    summary: &mut Summary,
    report: &mut Report,
) {
    let (d, tol) = (case.head_dim, tolerance(case.dtype));
    let rows = sampled_rows(case.seq, case.seed);
    let exact = attention_rows_f64(&inputs.q_ref, &inputs.k_host, &inputs.v_host, d, 1.0, &rows);
    // Attention outputs are convex combinations of v, a few units at most.
    #[allow(clippy::cast_possible_truncation)]
    let exact: Vec<f32> = exact.iter().map(|&x| x as f32).collect();
    let cmp = compare(&gather_rows(got, &rows, d), &exact, HQ, d, tol);
    summary.f64_checked += 1;
    summary.f64_failed += usize::from(!cmp.failures.is_empty());
    let ref_cmp = compare(&gather_rows(want, &rows, d), &exact, HQ, d, tol);
    summary.ref_vs_f64 = summary.ref_vs_f64.max(ref_cmp.max_err);
    // Failure rows index into `rows`, so map them back.
    let failures: Vec<_> = cmp
        .failures
        .iter()
        .map(|f| RowFailure {
            row: rows[f.row],
            ..*f
        })
        .collect();
    report.add(case, "kernel vs f64", &failures);
}

/// The tolerance for the kernels' output in `dtype`. It sits well above
/// output rounding in the dtype, and well below the errors the maintainer
/// reported. BF16 and AMD F16 accumulate in F32. The NVIDIA F16 kernel keeps
/// its output accumulator in `half2` as llama.cpp does, which costs about 1%
/// on peaked softmax rows, so it gets a looser limit than BF16 despite the
/// longer mantissa. Real bugs have been off by whole units, far outside
/// either.
pub(super) fn tolerance(dtype: DType) -> Tolerance {
    if dtype == DType::BF16 {
        Tolerance {
            atol: 1e-2,
            rtol: 2e-2,
        }
    } else if !is_amd_backend() {
        Tolerance {
            atol: 1e-2,
            rtol: 4e-2,
        }
    } else {
        Tolerance {
            atol: 2e-3,
            rtol: 1e-2,
        }
    }
}

///// Reporting ///////////////////////////////////////////////////////////

/// Kernel tiling at one length, for locating failures relative to tile
/// boundaries.
#[derive(Debug, Clone, Copy)]
struct Geometry {
    ncols1: usize,
    ncols2: usize,
    nbatch_fa: usize,
}

impl Geometry {
    /// The tiling the causal entries pick for `seq` at `head_dim` `d`.
    ///
    /// # Panics
    ///
    /// Panics if no MMA kernel covers the shape.
    fn new(dtype: DType, d: usize, seq: usize) -> Self {
        const NO_KERNEL: &str = "no MMA kernel for this shape";
        let is_amd = is_amd_backend();
        let gqa = HQ / HKV;
        let (ncols1, ncols2, nbatch_fa) = if dtype == DType::BF16 {
            let (n1, n2) = select_mma_ncols_bf16(d, gqa, seq, is_amd).expect(NO_KERNEL);
            let cfg = mma_config_bf16(d, n1 * n2, is_amd).expect(NO_KERNEL);
            (n1, n2, cfg.nbatch_fa)
        } else {
            let (n1, n2) = select_mma_ncols(d, gqa, seq, is_amd).expect(NO_KERNEL);
            let cfg = mma_config(d, n1 * n2, is_amd).expect(NO_KERNEL);
            (n1, n2, cfg.nbatch_fa)
        };
        Self {
            ncols1,
            ncols2,
            nbatch_fa,
        }
    }
}

impl std::fmt::Display for Geometry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ncols1={} ncols2={} nbatch_fa={}",
            self.ncols1, self.ncols2, self.nbatch_fa
        )
    }
}

/// Per-length counters. The `*_failed` counts are seeds with at least one
/// element out of tolerance, the maintainer's "x/30 checks" unit.
#[derive(Debug, Default)]
struct Summary {
    /// Inputs checked, which is the denominator of the failure counts.
    cases: usize,
    kernel_failed: usize,
    /// `kernel_failed` and `nondeterministic` split by [`DISTS`] entry.
    kernel_failed_by_dist: [usize; DISTS.len()],
    nondeterministic_by_dist: [usize; DISTS.len()],
    prefill_failed: usize,
    nondeterministic: usize,
    with_mask_mismatch: usize,
    prefill_mismatch: usize,
    f64_checked: usize,
    f64_failed: usize,
    max_err: f32,
    max_ratio: f32,
    ref_vs_f64: f32,
}

impl Summary {
    /// Counts one input's kernel results, for the [`DISTS`] entry at
    /// `dist_idx`.
    fn record_kernel(
        &mut self,
        dist_idx: usize,
        cmp: &Comparison,
        nondeterministic: bool,
        with_mask_mismatch: bool,
    ) {
        let failed = usize::from(!cmp.failures.is_empty());
        self.cases += 1;
        self.nondeterministic += usize::from(nondeterministic);
        self.nondeterministic_by_dist[dist_idx] += usize::from(nondeterministic);
        self.with_mask_mismatch += usize::from(with_mask_mismatch);
        self.kernel_failed += failed;
        self.kernel_failed_by_dist[dist_idx] += failed;
        self.add_max(cmp);
    }

    /// Folds `cmp`'s largest error and error ratio into the maxima.
    fn add_max(&mut self, cmp: &Comparison) {
        self.max_err = self.max_err.max(cmp.max_err);
        self.max_ratio = self.max_ratio.max(cmp.max_ratio);
    }

    /// Whether any check at this length failed.
    fn failed(&self) -> bool {
        self.kernel_failed
            + self.prefill_failed
            + self.nondeterministic
            + self.with_mask_mismatch
            + self.prefill_mismatch
            + self.f64_failed
            > 0
    }
}

impl std::fmt::Display for Summary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let seeds = self.cases;
        write!(
            f,
            "kernel {}/{seeds} ({}), try_causal_prefill {}/{seeds}, \
             nondeterministic {} ({}), with_mask mismatch {}, \
             try_causal_prefill mismatch {}, kernel vs f64 {}/{}, \
             max err {:.2e} ({:.0}% of tol), f32 ref vs f64 {:.1e}",
            self.kernel_failed,
            by_dist(&self.kernel_failed_by_dist),
            self.prefill_failed,
            self.nondeterministic,
            by_dist(&self.nondeterministic_by_dist),
            self.with_mask_mismatch,
            self.prefill_mismatch,
            self.f64_failed,
            self.f64_checked,
            self.max_err,
            self.max_ratio * 100.0,
            self.ref_vs_f64,
        )
    }
}

/// `Normal 0, Scaled 6, Spike 10`.
fn by_dist(counts: &[usize; DISTS.len()]) -> String {
    DISTS
        .iter()
        .zip(counts)
        .map(|(d, n)| format!("{d:?} {n}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The worst failing rows across the whole sweep, plus failure counts by
/// position in the Q tile and by head.
#[derive(Debug, Default)]
struct Report {
    failed: bool,
    worst: Vec<(Case, &'static str, RowFailure)>,
    by_row_in_tile: BTreeMap<usize, usize>,
    by_head: BTreeMap<usize, usize>,
}

impl Report {
    /// Records `case`'s failing rows from the check named `entry`.
    fn add(&mut self, case: &Case, entry: &'static str, failures: &[RowFailure]) {
        if failures.is_empty() {
            return;
        }
        for f in failures {
            *self
                .by_row_in_tile
                .entry(f.row % case.geom.ncols1)
                .or_default() += 1;
            *self.by_head.entry(f.head).or_default() += 1;
            self.worst.push((*case, entry, *f));
        }
        // NaN sorts as the largest error under `total_cmp`.
        self.worst
            .sort_by(|a, b| b.2.max_err.total_cmp(&a.2.max_err));
        self.worst.truncate(WORST_ROWS);
    }

    /// Prints the failure details and panics, if anything failed.
    fn finish(&self, dtype: DType, head_dim: usize) {
        if !self.failed {
            return;
        }
        let label = format!("{dtype:?} hd{head_dim}");
        println!("{label}: worst {} failing rows:", self.worst.len());
        for (case, entry, f) in &self.worst {
            println!(
                "  {entry}: seq={} {:?} seed={} head={} row={} \
                 row%ncols1={} row/nbatch_fa={} max err {:.3e} ({} elems), \
                 worst elem {}: got {:.5} want {:.5}",
                case.seq,
                case.dist,
                case.seed,
                f.head,
                f.row,
                f.row % case.geom.ncols1,
                f.row / case.geom.nbatch_fa,
                f.max_err,
                f.elems,
                f.elem,
                f.got,
                f.want,
            );
        }
        println!(
            "{label}: failing rows by row % ncols1: {:?}",
            self.by_row_in_tile
        );
        println!("{label}: failing rows by head: {:?}", self.by_head);
        panic!("{label}: long-sequence check failed, see the report above");
    }
}

///// Helpers /////////////////////////////////////////////////////////////

/// `q * scale` as the kernels hold it. They fold the softmax scale into Q
/// and store the product in `dtype`, rounding both the scale and the
/// product (`fattn_mma_bf16.cuh`'s `scale_bf16 * q`, `fattn_mma_f16.cuh`'s
/// `scale_h2 * q`). Unless the scale is a power of two (`head_dim` 64 and
/// 256), that is one more rounding of Q, and on peaked rows it costs more
/// than the tolerance. The product of two 16-bit values is exact in an f32,
/// so rounding it once reproduces the kernels' result.
pub(super) fn scaled_q(q: &[f32], scale: f32, dtype: DType) -> Vec<f32> {
    let scale = round_to(vec![scale], dtype)[0];
    round_to(q.iter().map(|x| x * scale).collect(), dtype)
}

/// Rounds host values to `dtype` and back, so every path sees the same
/// inputs.
pub(super) fn round_to(host: Vec<f32>, dtype: DType) -> Vec<f32> {
    let n = host.len();
    Tensor::from_vec(host, n, &Device::Cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Copies a GPU result of any float dtype to the host as flat F32.
pub(super) fn to_host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Whether `a` and `b` hold the same bits, NaN payloads included.
pub(super) fn bitwise_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Rows the F64 check covers: the first and last row of every
/// [`F64_BLOCK`]-row block, which includes rows 0 and `seq - 1`, plus
/// [`F64_RANDOM_ROWS`] seeded random rows.
fn sampled_rows(seq: usize, seed: u64) -> Vec<usize> {
    let mut rows: Vec<usize> = (0..seq)
        .step_by(F64_BLOCK)
        .flat_map(|s| [s, (s + F64_BLOCK - 1).min(seq - 1)])
        .collect();
    let mut state = seed;
    // The modulus keeps the value below seq, which fits a usize.
    #[allow(clippy::cast_possible_truncation)]
    rows.extend((0..F64_RANDOM_ROWS).map(|_| (splitmix64(&mut state) % seq as u64) as usize));
    rows.sort_unstable();
    rows.dedup();
    rows
}

/// The `[HQ, d]` slices of flat BSHD `t` at `rows`, concatenated.
fn gather_rows(t: &[f32], rows: &[usize], d: usize) -> Vec<f32> {
    rows.iter()
        .flat_map(|&r| &t[r * HQ * d..][..HQ * d])
        .copied()
        .collect()
}
