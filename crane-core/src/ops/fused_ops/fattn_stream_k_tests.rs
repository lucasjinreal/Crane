// SPDX-License-Identifier: MIT
//! Stream-K tests for the MMA flash-attention kernels, with 8 Q heads and 2
//! KV heads. Forced grid sizes put 1, 2 and 3 seams inside each output
//! tile, give blocks segments that lie wholly past the causal bound, and
//! make blocks finish one tile and start the next. Every run must match
//! the one-block-per-tile launch and an F32 reference within the
//! long-sequence tolerance, and give the same bits twice. The heavy-first
//! tile order of a one-block-per-tile launch must match the stream-K order
//! bit for bit, also with batch 2 and with 16 Q and 4 KV heads.

use candle_core::{DType, Device, Tensor};

use super::super::fattn::test_support::{
    LONG_SEQ_HKV as HKV, LONG_SEQ_HQ as HQ, LongSeqDist, MaskMode, long_seq_inputs,
    naive_attention, test_gpu_device,
};
use super::super::fattn::{
    FATTN_KQ_STRIDE, build_additive_mask_f16, build_analytic_kv_max, is_amd_backend, mma_config,
    mma_config_bf16, mma_config_rdna_long_kv, select_mma_ncols, select_mma_ncols_bf16,
    stream_k_needs_fixup,
};
use super::super::fattn_long_seq_host::compare;
use super::long_seq_tests::{bitwise_eq, round_to, scaled_q, to_host, tolerance};
use super::{
    LaunchOverride, LaunchShape, fattn_mma_bf16_causal, prefill_dispatch, prefill_dispatch_bf16,
    tile_counts,
};

/// `head_dim` 512 has no MMA kernel on AMD.
#[cfg(feature = "cuda")]
const HEAD_DIMS: [usize; 4] = [64, 128, 256, 512];
#[cfg(not(feature = "cuda"))]
const HEAD_DIMS: [usize; 3] = [64, 128, 256];
/// One unaligned length (the kernel's partial-tile bounds-check path) and
/// one aligned one.
const SEQS: [usize; 2] = [231, 256];
const DTYPES: [DType; 2] = [DType::BF16, DType::F16];
/// The default layout, a second batch element, and twice the heads.
const LAYOUTS: [Layout; 3] = [
    Layout::DEFAULT,
    Layout {
        batch: 2,
        hq: HQ,
        hkv: HKV,
    },
    Layout {
        batch: 1,
        hq: 2 * HQ,
        hkv: 2 * HKV,
    },
];

///// Tests ///////////////////////////////////////////////////////////////

// In-kernel causal masking with every forced grid matches the plain launch
// and the reference, deterministically, at every head_dim and length.
#[test]
fn forced_grids_match_plain_launch_causal() {
    run_forced_grids(Mode::Causal);
}

// As `forced_grids_match_plain_launch_causal`, through the host mask and
// `KV_max` of the `_with_mask` entries.
#[test]
fn forced_grids_match_plain_launch_causal_with_mask() {
    run_forced_grids(Mode::CausalWithMask);
}

// As `forced_grids_match_plain_launch_causal`, with no masking at all.
#[test]
fn forced_grids_match_plain_launch_full() {
    run_forced_grids(Mode::Full);
}

// The long-KV kernel set of the RDNA hd256 64-column tile, in its own
// module with its own fixup kernels, merges split tiles like the default
// set does, at an unaligned and an aligned length.
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
#[test]
fn forced_grids_match_plain_launch_long_kv_set() {
    let gpu = test_gpu_device();
    for dtype in DTYPES {
        for seq in SEQS {
            check_forced_grids(&Case::long_kv(dtype, 256, seq, Mode::Causal, &gpu));
        }
    }
}

// Every tile is computed exactly once whatever order the blocks take
// them in. One block per tile, forced heavy-first and forced stream-K
// order, give the same bits for every mode, layout, dtype, head_dim and
// length. The bits cannot show which order ran, only that neither skips
// or repeats a tile.
#[test]
fn heavy_first_order_matches_stream_k_order_bitwise() {
    let gpu = test_gpu_device();
    let modes = [
        Mode::Causal,
        Mode::CausalWithMask,
        Mode::Full,
        Mode::FullWithKvMax,
    ];
    for (mode, layout) in modes.into_iter().flat_map(|m| LAYOUTS.map(|l| (m, l))) {
        for dtype in DTYPES {
            for d in HEAD_DIMS {
                for seq in SEQS {
                    let case = Case::with_layout(dtype, d, seq, mode, layout, &gpu);
                    let ordered = |heavy_first| {
                        case.run_with(LaunchOverride {
                            grid: Some(case.ntiles),
                            heavy_first: Some(heavy_first),
                            ..LaunchOverride::NONE
                        })
                    };
                    assert!(
                        bitwise_eq(&ordered(true), &ordered(false)),
                        "{case:?}: heavy-first order differs from stream-K order"
                    );
                }
            }
        }
    }
}

// At seq 231 the policy's own grid choice gives a correct, deterministic
// result. `head_dim` 256 always resolves to one block per tile now (see
// `LaunchGeometry::new`); the forced-grid tests above exercise the fixup
// merge directly regardless.
#[test]
fn natural_grid_at_231_is_correct() {
    let gpu = test_gpu_device();
    for dtype in DTYPES {
        let case = Case::new(dtype, 256, 231, Mode::Causal, &gpu);
        let got = case.run(None);
        assert!(
            bitwise_eq(&got, &case.run(None)),
            "{dtype:?}: natural grid not deterministic"
        );
        case.check(&got, "natural grid");
    }
}

// `head_dim` 64 on AMD always engages the stream-K split, regardless of
// wave efficiency (llama.cpp's own rule for that shape). At long sequences
// this puts more than one output tile in a single block's `kbc` range,
// which previously returned different bits across repeated launches of
// the identical input: `process_tile`'s final combine-stage barrier was
// skipped whenever `np == 1`, so a block's next tile could start
// overwriting `tile_Q`/`tile_mask` while other warps were still reading
// them for the tile just finished. Fixed by making that barrier
// unconditional (see `fattn_mma_f16.cuh`/`fattn_mma_bf16.cuh`).
#[test]
fn hd64_long_seq_stream_k_is_deterministic() {
    let gpu = test_gpu_device();
    let (seq, d) = (4096usize, 64usize);
    let scale = (d as f32).sqrt().recip();
    let mut mismatches = 0;
    let mut calls = 0;
    for dist in [LongSeqDist::Normal, LongSeqDist::Scaled, LongSeqDist::Spike] {
        for seed in 0..5u64 {
            let [q_h, k_h, v_h] =
                long_seq_inputs(seq, d, dist, seed).map(|h| round_to(h, DType::BF16));
            let bshd = |h: &[f32], heads| {
                Tensor::from_slice(h, (1, seq, heads, d), &gpu)
                    .unwrap()
                    .to_dtype(DType::BF16)
                    .unwrap()
            };
            let (q, k, v) = (bshd(&q_h, HQ), bshd(&k_h, HKV), bshd(&v_h, HKV));
            let base = to_host(&fattn_mma_bf16_causal(&q, &k, &v, scale, 0).unwrap());
            for _ in 0..30 {
                calls += 1;
                let got = to_host(&fattn_mma_bf16_causal(&q, &k, &v, scale, 0).unwrap());
                if !bitwise_eq(&base, &got) {
                    mismatches += 1;
                }
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "{mismatches} nondeterministic repeat(s) out of {calls} calls"
    );
}

///// Runner //////////////////////////////////////////////////////////////

/// The masking path a case launches through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// No mask, the kernel masks the diagonal (`causal_kv_offset = 0`).
    Causal,
    /// A host causal mask plus `KV_max`, `causal_kv_offset = -1`.
    CausalWithMask,
    /// An all-zero mask, no `KV_max`.
    Full,
    /// An all-zero mask plus a `KV_max` that is the same for every tile.
    FullWithKvMax,
}

/// The batch size and head counts of a case.
#[derive(Debug, Clone, Copy)]
struct Layout {
    /// Batch size.
    batch: usize,
    /// Number of Q heads.
    hq: usize,
    /// Number of KV heads.
    hkv: usize,
}

impl Layout {
    /// Batch 1 with the long-sequence head counts.
    const DEFAULT: Self = Self {
        batch: 1,
        hq: HQ,
        hkv: HKV,
    };
}

/// Runs every forced grid of [`forced_grids`] for every dtype, `head_dim`
/// and length in `mode`, and checks each against the plain launch and the
/// reference.
fn run_forced_grids(mode: Mode) {
    let gpu = test_gpu_device();
    for dtype in DTYPES {
        for d in HEAD_DIMS {
            for seq in SEQS {
                check_forced_grids(&Case::new(dtype, d, seq, mode, &gpu));
            }
        }
    }
}

/// Every forced grid of `case` matches its one-block-per-tile launch and
/// the reference, and gives the same bits twice.
fn check_forced_grids(case: &Case) {
    let plain = case.run(Some(case.ntiles));
    case.check(&plain, "plain grid");
    for (grid, seams) in forced_grids(case.ntiles, case.iter_k) {
        let what = format!("grid {grid} ({seams} seams)");
        let total = case.ntiles * case.iter_k;
        assert!(
            stream_k_needs_fixup(total, grid, case.iter_k).unwrap(),
            "{case:?} {what}: no seam inside a tile"
        );
        assert_eq!(
            max_seams_in_a_tile(total, grid, case.iter_k),
            seams,
            "{case:?} {what}: wrong seam count"
        );
        let got = case.run(Some(grid));
        assert!(
            bitwise_eq(&got, &case.run(Some(grid))),
            "{case:?} {what}: not deterministic"
        );
        // The kernel must keep the stream-K order on a split grid, whatever
        // order the host asks for.
        let heavy_first = case.run_with(LaunchOverride {
            grid: Some(grid),
            heavy_first: Some(true),
            ..LaunchOverride::NONE
        });
        assert!(
            bitwise_eq(&got, &heavy_first),
            "{case:?} {what}: heavy-first changed a split grid"
        );
        let cmp = compare(&got, &plain, case.layout.hq, case.d, tolerance(case.dtype));
        assert!(
            cmp.failures.is_empty(),
            "{case:?} {what}: {} rows off the plain launch, max err {:.2e}",
            cmp.failures.len(),
            cmp.max_err
        );
        case.check(&got, &what);
    }
}

/// `(grid, most seams inside one tile)` pairs for `ntiles` tiles of
/// `iter_k` KV blocks. Uniform splits into 2, 3 and 4 blocks per tile put
/// 1, 2 and 3 seams in every tile. At 4, early causal tiles get segments
/// past their KV bound. `ntiles + 1` and `ntiles - 1` give blocks that end
/// one tile and start the next.
fn forced_grids(ntiles: usize, iter_k: usize) -> Vec<(usize, usize)> {
    assert!(
        iter_k >= 4,
        "iter_k {iter_k} too small for 3 seams in a tile"
    );
    let mut grids = vec![(2 * ntiles, 1), (3 * ntiles, 2), (4 * ntiles, 3)];
    grids.push((ntiles + 1, 1));
    if ntiles > 2 {
        grids.push((ntiles - 1, 1));
    }
    grids
}

/// The largest number of block boundaries falling strictly inside one
/// tile, with the kernel's block ranges `b * total / grid`.
fn max_seams_in_a_tile(total: usize, grid: usize, iter_k: usize) -> usize {
    let mut per_tile = vec![0; total / iter_k];
    for b in 1..grid {
        let kbc = b * total / grid;
        if !kbc.is_multiple_of(iter_k) {
            per_tile[kbc / iter_k] += 1;
        }
    }
    per_tile.into_iter().max().unwrap_or(0)
}

/// One dtype, `head_dim`, length, [`Mode`] and [`Layout`], with the launch
/// inputs prepared the way the public entries prepare them.
struct Case {
    dtype: DType,
    d: usize,
    seq: usize,
    mode: Mode,
    layout: Layout,
    /// Whether every launch forces the long-KV kernel set.
    long_kv: bool,
    q: Tensor,
    /// Padded for GQA batching where the entries pad.
    k: Tensor,
    v: Tensor,
    mask: Option<Tensor>,
    kv_max: Option<Tensor>,
    scale: f32,
    ncols1: usize,
    ncols2: usize,
    ntiles: usize,
    iter_k: usize,
    /// The F32 reference, from Q pre-scaled as the kernels hold it.
    want: Vec<f32>,
}

impl std::fmt::Debug for Case {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} hd{} seq={} {:?} b{} {}Q/{}KV ncols=({}, {}) ntiles={} iter_k={} long_kv={}",
            self.dtype,
            self.d,
            self.seq,
            self.mode,
            self.layout.batch,
            self.layout.hq,
            self.layout.hkv,
            self.ncols1,
            self.ncols2,
            self.ntiles,
            self.iter_k,
            self.long_kv
        )
    }
}

impl Case {
    /// [`Case::with_layout`] with [`Layout::DEFAULT`].
    fn new(dtype: DType, d: usize, seq: usize, mode: Mode, gpu: &Device) -> Self {
        Self::with_layout(dtype, d, seq, mode, Layout::DEFAULT, gpu)
    }

    /// [`Case::new`] on the long-KV kernel set, whatever `seq` the policy
    /// would pick it at.
    fn long_kv(dtype: DType, d: usize, seq: usize, mode: Mode, gpu: &Device) -> Self {
        Self::with_set(dtype, d, seq, mode, (Layout::DEFAULT, true), gpu)
    }

    /// [`Case::with_set`] on the default kernel set.
    fn with_layout(
        dtype: DType,
        d: usize,
        seq: usize,
        mode: Mode,
        layout: Layout,
        gpu: &Device,
    ) -> Self {
        Self::with_set(dtype, d, seq, mode, (layout, false), gpu)
    }

    /// Builds Spike-distribution inputs (a running max that moves deep into
    /// the KV loop) and the reference.
    ///
    /// # Panics
    ///
    /// If no MMA kernel covers the shape, or `layout` does not keep the
    /// long-sequence GQA ratio.
    #[allow(clippy::too_many_lines)] // one case's worth of setup, not separable
    fn with_set(
        dtype: DType,
        d: usize,
        seq: usize,
        mode: Mode,
        (layout, long_kv): (Layout, bool),
        gpu: &Device,
    ) -> Self {
        let Layout { batch, hq, hkv } = layout;
        // long_seq_inputs is fixed at HQ Q and HKV KV heads. With the same
        // ratio, seq * batch * hq / HQ of its rows hold exactly batch
        // elements of seq rows of hq and hkv heads.
        assert!(
            (batch * hq).is_multiple_of(HQ) && hq * HKV == hkv * HQ,
            "{layout:?}: not a multiple of {HQ}Q/{HKV}KV"
        );
        let bf16 = dtype == DType::BF16;
        let is_amd = is_amd_backend();
        let gqa_ratio = hq / hkv;
        // head_dim is at most 512, exact in an f32.
        #[allow(clippy::cast_precision_loss)]
        let scale = (d as f32).sqrt().recip();
        let [q_host, k_host, v_host] =
            long_seq_inputs(seq * batch * hq / HQ, d, LongSeqDist::Spike, 0)
                .map(|h| round_to(h, dtype));
        let bshd = |h: &[f32], heads| Tensor::from_slice(h, (batch, seq, heads, d), gpu).unwrap();
        let q = bshd(&q_host, hq).to_dtype(dtype).unwrap();
        let k = bshd(&k_host, hkv).to_dtype(dtype).unwrap();
        let v = bshd(&v_host, hkv).to_dtype(dtype).unwrap();
        let full = matches!(mode, Mode::Full | Mode::FullWithKvMax);
        let mask_mode = if full {
            MaskMode::Full
        } else {
            MaskMode::Causal
        };
        let want = naive_attention(
            &bshd(&scaled_q(&q_host, scale, dtype), hq),
            &bshd(&k_host, hkv),
            &bshd(&v_host, hkv),
            1.0,
            mask_mode,
            0,
            0,
            0,
        );

        let (k, v, mask) = match mode {
            Mode::Causal => (k, v, None),
            Mode::CausalWithMask | Mode::Full | Mode::FullWithKvMax => {
                let mask = if full {
                    Tensor::zeros((1, 1, seq, seq), DType::F16, gpu).unwrap()
                } else {
                    build_additive_mask_f16(seq, seq, 0, None, 0, gpu).unwrap()
                };
                (k, v, Some(mask))
            },
        };
        let seq_kv = k.dims4().unwrap().1;
        let select = if bf16 {
            select_mma_ncols_bf16
        } else {
            select_mma_ncols
        };
        let (ncols1, ncols2) = select(d, gqa_ratio, seq, is_amd)
            .unwrap_or_else(|| panic!("{dtype:?} hd{d} seq={seq}: no MMA kernel"));
        let cfg = if long_kv {
            mma_config_rdna_long_kv(d, ncols1 * ncols2)
                .unwrap_or_else(|| panic!("hd{d} {ncols1}x{ncols2}: no long-KV set"))
        } else if bf16 {
            mma_config_bf16(d, ncols1 * ncols2, is_amd).unwrap()
        } else {
            mma_config(d, ncols1 * ncols2, is_amd).unwrap()
        };
        let shape = LaunchShape {
            seq_q: seq,
            seq_kv,
            h_kv: hkv,
            batch,
            gqa_ratio,
        };
        let (ntiles, iter_k) = tile_counts(&cfg, ncols1, ncols2, &shape).unwrap();
        let window_right = match mode {
            Mode::CausalWithMask => Some(0),
            // Reaching past seq_kv clamps every tile's bound to the cap.
            Mode::FullWithKvMax => Some(seq_kv),
            Mode::Causal | Mode::Full => None,
        };
        let kv_max = window_right.map(|window_right| {
            build_analytic_kv_max(
                batch,
                seq,
                ncols1,
                0,
                seq_kv,
                window_right,
                seq_kv.next_multiple_of(FATTN_KQ_STRIDE),
                gpu,
            )
            .unwrap()
        });
        Self {
            dtype,
            d,
            seq,
            mode,
            layout,
            long_kv,
            q,
            k,
            v,
            mask,
            kv_max,
            scale,
            ncols1,
            ncols2,
            ntiles,
            iter_k,
            want,
        }
    }

    /// Launches with `grid` blocks, or the policy's grid for `None`, and
    /// returns the output on the host.
    fn run(&self, grid: Option<usize>) -> Vec<f32> {
        self.run_with(LaunchOverride {
            grid,
            ..LaunchOverride::NONE
        })
    }

    /// Launches with the launch policy overridden by `forced`, and returns
    /// the output on the host.
    fn run_with(&self, mut forced: LaunchOverride) -> Vec<f32> {
        if self.long_kv {
            forced.long_kv = Some(true);
        }
        // The allocator hands the next launch the block freed last. Fill a
        // block of the output's size with NaN and free it, so a launch that
        // leaves part of its output unwritten cannot pass on the stale
        // result of an earlier run.
        drop(
            Tensor::full(f32::NAN, self.q.shape(), self.q.device())
                .and_then(|t| t.to_dtype(self.dtype))
                .unwrap(),
        );
        let offset = (self.mode == Mode::Causal).then_some(0);
        let dispatch = if self.dtype == DType::BF16 {
            prefill_dispatch_bf16
        } else {
            prefill_dispatch
        };
        let out = dispatch(
            &self.q,
            &self.k,
            &self.v,
            self.mask.as_ref(),
            self.kv_max.as_ref(),
            self.scale,
            offset,
            forced,
        )
        .unwrap_or_else(|e| panic!("{self:?} {forced:?}: {e}"));
        to_host(&out)
    }

    /// Asserts `got` is within the long-sequence tolerance of the reference.
    fn check(&self, got: &[f32], what: &str) {
        let cmp = compare(
            got,
            &self.want,
            self.layout.hq,
            self.d,
            tolerance(self.dtype),
        );
        assert!(
            cmp.failures.is_empty(),
            "{self:?} {what}: {} rows off the reference, max err {:.2e}",
            cmp.failures.len(),
            cmp.max_err
        );
    }
}
