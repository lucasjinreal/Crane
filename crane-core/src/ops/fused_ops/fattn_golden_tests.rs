// SPDX-License-Identifier: MIT
//! Bit-exact goldens for the MMA flash-attention kernels. A hash of each
//! case's output bits is compared against
//! `tests/fixtures/fattn_goldens_{rocm,cuda}.txt`, recorded on one GPU
//! architecture per backend. `tests/fixtures/README.md` describes how to
//! check and record them.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};

use super::long_seq_tests::{DISTS, SEEDS_PER_DIST, SEQS, round_to, to_host};
use super::{
    LaunchOverride, WindowBounds, fattn_mma_bf16_causal, fattn_mma_bf16_full,
    fattn_mma_bf16_windowed, fattn_mma_causal, fattn_mma_full, fattn_mma_windowed,
    prefill_dispatch, prefill_dispatch_bf16,
};
use crate::ops::fused_ops::fattn::is_amd_backend;
use crate::ops::fused_ops::fattn::test_support::{
    LONG_SEQ_HKV as HKV, LONG_SEQ_HQ as HQ, LongSeqDist, long_seq_inputs, test_gpu_device,
};

const HEAD_DIMS: [usize; 3] = [64, 128, 256];
const DTYPES: [DType; 2] = [DType::BF16, DType::F16];
/// Lengths of the non-causal variants, one unaligned and one aligned.
const VARIANT_SEQS: [usize; 2] = [238, 2048];
/// Left window of [`Variant::Windowed`], not a multiple of any tile size.
const WINDOW_LEFT: usize = 100;

///// Tests ///////////////////////////////////////////////////////////////

// The MMA entries give the recorded bits for every case of the sweep, on
// the GPU architecture the goldens were recorded on. With
// `CRANE_FATTN_GOLDEN=record` it rewrites the file instead and prints how
// many hashes changed.
#[test]
#[ignore = "needs a CUDA/ROCm GPU, long-running"]
fn kernel_output_matches_goldens() {
    let mode = Mode::parse(std::env::var("CRANE_FATTN_GOLDEN").ok().as_deref());
    let gpu = test_gpu_device();
    let current = Goldens {
        arch: Some(device_arch(&gpu)),
        hashes: compute_all(&gpu),
    };
    let path = golden_path();
    if mode == Mode::Record {
        record(&path, &current);
        return;
    }
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e}. Record them with CRANE_FATTN_GOLDEN=record.",
            path.display()
        )
    });
    let want = parse_goldens(&text);
    if let Err(e) = check_arch(want.arch.as_deref(), current.arch.as_deref().unwrap()) {
        panic!("{}: {e}", path.display());
    }
    let diffs = diff_goldens(&want.hashes, &current.hashes);
    assert!(
        diffs.is_empty(),
        "{} of {} cases changed:\n{}",
        diffs.len(),
        current.hashes.len(),
        diffs
            .iter()
            .take(20)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

// The hash is FNV-1a 64 over the little-endian bytes of the f32 bits.
#[test]
fn fnv1a_matches_the_reference_vectors() {
    // Empty input leaves the offset basis.
    assert_eq!(hash_bits(&[]), 0xcbf2_9ce4_8422_2325);
    // 0.0f32 is four zero bytes: FNV-1a 64 of "\0\0\0\0".
    assert_eq!(hash_bits(&[0.0]), 0x4d25_767f_9dce_13f5);
    // Hashing sees the bits, so 0.0 and -0.0 differ.
    assert_ne!(hash_bits(&[0.0]), hash_bits(&[-0.0]));
}

// The file format round-trips, arch header included, and ignores comment
// lines.
#[test]
fn golden_file_round_trips() {
    let goldens = Goldens {
        arch: Some("cc 11.0".to_owned()),
        hashes: BTreeMap::from([
            (
                key(DType::BF16, 256, 231, 0, Variant::Causal),
                0x0123_4567_89ab_cdef,
            ),
            (key(DType::F16, 64, 4096, 29, Variant::Offset), 1),
        ]),
    };
    let text = format_goldens(&goldens);
    assert!(text.starts_with("# arch: cc 11.0\n"));
    assert_eq!(parse_goldens(&text), goldens);
    assert_eq!(parse_goldens("# comment\n\nk 1\n").arch, None);
}

// A changed, missing and unexpected case are each reported.
#[test]
fn diff_reports_changed_missing_and_extra_cases() {
    let want = BTreeMap::from([("a".to_owned(), 1), ("b".to_owned(), 2)]);
    let got = BTreeMap::from([("a".to_owned(), 9), ("c".to_owned(), 3)]);
    let diffs = diff_goldens(&want, &got);
    assert_eq!(diffs.len(), 3);
    assert!(diffs.iter().any(|d| d.starts_with("a: changed")));
    assert!(diffs.iter().any(|d| d.starts_with("b: missing")));
    assert!(diffs.iter().any(|d| d.starts_with("c: not in")));
    assert_eq!(diff_goldens(&want, &want), Vec::<String>::new());
}

// Goldens from another architecture, or without one, are rejected.
#[test]
fn check_fails_on_arch_mismatch() {
    assert!(check_arch(Some("cc 11.0"), "cc 11.0").is_ok());
    assert!(check_arch(Some("cc 11.0"), "cc 12.0").is_err());
    assert!(check_arch(None, "cc 11.0").is_err());
}

// Unset and empty check and `record` records.
#[test]
fn mode_parses_unset_and_record() {
    assert_eq!(Mode::parse(None), Mode::Check);
    assert_eq!(Mode::parse(Some("")), Mode::Check);
    assert_eq!(Mode::parse(Some("record")), Mode::Record);
}

// A misspelled mode fails instead of silently checking.
#[test]
#[should_panic(expected = "CRANE_FATTN_GOLDEN")]
fn mode_rejects_unknown_values() {
    Mode::parse(Some("Record"));
}

///// Sweep ///////////////////////////////////////////////////////////////

/// What [`kernel_output_matches_goldens`] does, set by
/// `CRANE_FATTN_GOLDEN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Check,
    Record,
}

impl Mode {
    /// Unset or empty checks and `record` records. Anything else panics, so
    /// a typo does not silently check.
    fn parse(value: Option<&str>) -> Self {
        match value {
            None | Some("") => Self::Check,
            Some("record") => Self::Record,
            Some(v) => panic!("CRANE_FATTN_GOLDEN={v:?}: expected \"record\" or unset"),
        }
    }
}

/// Entry and input shape a case runs with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Variant {
    /// Plain causal over the full sweep.
    Causal,
    /// Causal over a cached prefix: the last quarter of the rows as q,
    /// against all of k/v.
    Offset,
    /// Sliding window of [`WINDOW_LEFT`] keys.
    Windowed,
    /// No mask.
    Full,
    /// Causal with each KV head repeated twice, a GQA ratio of 2. Ratio 1
    /// has no kernel on AMD.
    Gqa2,
    /// Plain causal on the long-KV kernel set of the RDNA hd256 64-column
    /// tile, forced at every length. Only exists for `head_dim` 256 on `ROCm`.
    LongKv,
}

impl Variant {
    /// The variants run at [`VARIANT_SEQS`] only.
    const EXTRA: [Self; 4] = [Self::Offset, Self::Windowed, Self::Full, Self::Gqa2];

    /// Key suffix. Causal has none.
    fn suffix(self) -> &'static str {
        match self {
            Self::Causal => "",
            Self::Offset => " offset",
            Self::Windowed => " windowed",
            Self::Full => " full",
            Self::Gqa2 => " gqa2",
            Self::LongKv => " longkv",
        }
    }
}

/// Hashes every case of the sweep, keyed by [`key`].
fn compute_all(gpu: &Device) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    for head_dim in HEAD_DIMS {
        let sweep: &[Variant] = if head_dim == 256 && is_amd_backend() {
            &[Variant::Causal, Variant::LongKv]
        } else {
            &[Variant::Causal]
        };
        for seq in SEQS {
            for (seed, dist) in seeds(SEEDS_PER_DIST) {
                hash_cases(gpu, &mut out, sweep, head_dim, seq, dist, seed);
            }
        }
        for seq in VARIANT_SEQS {
            for (seed, dist) in seeds(1) {
                hash_cases(gpu, &mut out, &Variant::EXTRA, head_dim, seq, dist, seed);
            }
        }
        println!("hd{head_dim} done");
    }
    out
}

/// The first `per_dist` seeds of each distribution, numbered as in the
/// long-sequence tests.
fn seeds(per_dist: u64) -> impl Iterator<Item = (u64, LongSeqDist)> {
    (0u64..).zip(DISTS).flat_map(move |(dist_idx, dist)| {
        (0..per_dist).map(move |idx| (dist_idx * SEEDS_PER_DIST + idx, dist))
    })
}

/// Runs `variants` for every dtype on one sweep input and records the
/// output hashes. The host inputs are built once and rounded per dtype.
fn hash_cases(
    gpu: &Device,
    out: &mut BTreeMap<String, u64>,
    variants: &[Variant],
    d: usize,
    seq: usize,
    dist: LongSeqDist,
    seed: u64,
) {
    let host = long_seq_inputs(seq, d, dist, seed);
    for dtype in DTYPES {
        let rounded = host.each_ref().map(|h| round_to(h.clone(), dtype));
        for &variant in variants {
            let hash = hash_bits(&to_host(&run_variant(
                gpu, variant, dtype, d, seq, &rounded,
            )));
            out.insert(key(dtype, d, seq, seed, variant), hash);
        }
    }
}

/// Runs one variant's MMA entry on host inputs already rounded to `dtype`.
fn run_variant(
    gpu: &Device,
    variant: Variant,
    dtype: DType,
    d: usize,
    seq: usize,
    [q_host, k_host, v_host]: &[Vec<f32>; 3],
) -> Tensor {
    // head_dim is at most 256, exact in an f32.
    #[allow(clippy::cast_precision_loss)]
    let scale = (d as f32).sqrt().recip();
    let gqa2;
    let (k_host, v_host, hkv): (&[f32], &[f32], usize) = if variant == Variant::Gqa2 {
        gqa2 = [repeat_heads(k_host, d), repeat_heads(v_host, d)];
        (&gqa2[0], &gqa2[1], 2 * HKV)
    } else {
        (k_host, v_host, HKV)
    };
    let bshd = |h: &[f32], rows, heads| Tensor::from_slice(h, (1, rows, heads, d), gpu).unwrap();
    let q = bshd(q_host, seq, HQ).to_dtype(dtype).unwrap();
    let k = bshd(k_host, seq, hkv).to_dtype(dtype).unwrap();
    let v = bshd(v_host, seq, hkv).to_dtype(dtype).unwrap();
    let bf16 = dtype == DType::BF16;
    let out = match variant {
        Variant::Causal | Variant::Gqa2 if bf16 => fattn_mma_bf16_causal(&q, &k, &v, scale, 0),
        Variant::Causal | Variant::Gqa2 => fattn_mma_causal(&q, &k, &v, scale, 0),
        Variant::Offset => {
            let seq_q = seq / 4;
            let offset = seq - seq_q;
            let q = bshd(&q_host[offset * HQ * d..], seq_q, HQ)
                .to_dtype(dtype)
                .unwrap();
            if bf16 {
                fattn_mma_bf16_causal(&q, &k, &v, scale, offset)
            } else {
                fattn_mma_causal(&q, &k, &v, scale, offset)
            }
        },
        Variant::Windowed if bf16 => fattn_mma_bf16_windowed(&q, &k, &v, scale, 0, WINDOW_LEFT, 0),
        Variant::Windowed => fattn_mma_windowed(&q, &k, &v, scale, 0, WINDOW_LEFT, 0),
        Variant::Full if bf16 => fattn_mma_bf16_full(&q, &k, &v, scale),
        Variant::Full => fattn_mma_full(&q, &k, &v, scale),
        Variant::LongKv => {
            let forced = LaunchOverride {
                long_kv: Some(true),
                ..LaunchOverride::NONE
            };
            let dispatch = if bf16 {
                prefill_dispatch_bf16
            } else {
                prefill_dispatch
            };
            dispatch(
                &q,
                &k,
                &v,
                None,
                None,
                scale,
                Some(0),
                WindowBounds::NONE,
                forced,
            )
        },
    };
    out.unwrap()
}

/// Repeats each KV head of flat BSHD `kv` twice in place, so every q head
/// meets the same values as with the original [`HKV`] heads.
fn repeat_heads(kv: &[f32], d: usize) -> Vec<f32> {
    kv.chunks_exact(d)
        .flat_map(|head| std::iter::repeat_n(head, 2).flatten().copied())
        .collect()
}

/// Compute capability of the test GPU, for example `cc 8.9`. Kernels
/// compile to different code per architecture, so their output bits may
/// differ. HIP reports every gfx110x part as 11.0, so it cannot tell those
/// apart.
#[cfg(feature = "cuda")]
fn device_arch(gpu: &Device) -> String {
    use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;
    let stream = gpu.as_cuda_device().unwrap().cuda_stream();
    let attr = |a| stream.context().attribute(a).unwrap();
    format!(
        "cc {}.{}",
        attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
        attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
    )
}

/// See [`device_arch`].
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
fn device_arch(_gpu: &Device) -> String {
    // test_gpu_device opens ordinal 0.
    let props = candle_core::rocm_backend::rocm_rs::hip::get_device_properties(0).unwrap();
    format!("cc {}.{}", props.major, props.minor)
}

///// Golden file /////////////////////////////////////////////////////////

/// Contents of a golden file.
#[derive(Debug, PartialEq, Eq)]
struct Goldens {
    /// [`device_arch`] of the recording GPU, from the `# arch:` header.
    arch: Option<String>,
    /// Output hash per [`key`].
    hashes: BTreeMap<String, u64>,
}

/// Key of one case, for example `bf16 hd256 seq231 seed0 offset`.
fn key(dtype: DType, head_dim: usize, seq: usize, seed: u64, variant: Variant) -> String {
    let dtype = if dtype == DType::BF16 { "bf16" } else { "f16" };
    format!(
        "{dtype} hd{head_dim} seq{seq} seed{seed}{}",
        variant.suffix()
    )
}

/// Path of this backend's golden file.
fn golden_path() -> PathBuf {
    let backend = if is_amd_backend() { "rocm" } else { "cuda" };
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("fattn_goldens_{backend}.txt"))
}

/// Writes `new` to `path`, through a temporary file so an interrupted run
/// leaves the old file intact. Prints how much differs from the old file
/// first.
fn record(path: &Path, new: &Goldens) {
    if let Ok(text) = std::fs::read_to_string(path) {
        let old = parse_goldens(&text);
        println!(
            "arch {} -> {}, {} of {} hashes differ",
            old.arch.as_deref().unwrap_or("none"),
            new.arch.as_deref().unwrap_or("none"),
            diff_goldens(&old.hashes, &new.hashes).len(),
            new.hashes.len()
        );
    }
    let tmp = path.with_extension("txt.tmp");
    std::fs::write(&tmp, format_goldens(new)).unwrap();
    std::fs::rename(&tmp, path).unwrap();
    println!(
        "recorded {} goldens to {}",
        new.hashes.len(),
        path.display()
    );
}

/// Fails unless the goldens were recorded on `device`'s architecture.
fn check_arch(recorded: Option<&str>, device: &str) -> Result<(), String> {
    match recorded {
        Some(arch) if arch == device => Ok(()),
        Some(arch) => Err(format!(
            "goldens were recorded on {arch}, this device is {device}. Record \
             them locally and do not commit the result."
        )),
        None => Err("no `# arch:` header".to_owned()),
    }
}

/// FNV-1a 64 over the little-endian bytes of each value's bits.
fn hash_bits(values: &[f32]) -> u64 {
    values
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

/// Renders the golden file: the arch header, a comment, then `<key>
/// <hash>` lines.
fn format_goldens(goldens: &Goldens) -> String {
    let mut text = String::new();
    if let Some(arch) = &goldens.arch {
        writeln!(text, "# arch: {arch}").unwrap();
    }
    text.push_str(
        "# Output hashes of the MMA flash-attention entries. See\n\
         # tests/fixtures/README.md. Rewrite only when a kernel change is\n\
         # meant to alter output bits.\n",
    );
    for (key, hash) in &goldens.hashes {
        writeln!(text, "{key} {hash:016x}").unwrap();
    }
    text
}

/// Parses [`format_goldens`] output, skipping comments and blank lines.
fn parse_goldens(text: &str) -> Goldens {
    let mut arch = None;
    let mut hashes = BTreeMap::new();
    for line in text.lines() {
        if let Some(a) = line.strip_prefix("# arch: ") {
            arch = Some(a.to_owned());
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (key, hash) = line
            .rsplit_once(' ')
            .unwrap_or_else(|| panic!("bad line {line:?}"));
        let hash = u64::from_str_radix(hash, 16).unwrap_or_else(|e| panic!("{line:?}: {e}"));
        hashes.insert(key.to_owned(), hash);
    }
    Goldens { arch, hashes }
}

/// One message per case that differs between `want` and `got`.
fn diff_goldens(want: &BTreeMap<String, u64>, got: &BTreeMap<String, u64>) -> Vec<String> {
    let mut diffs = Vec::new();
    for (key, w) in want {
        match got.get(key) {
            Some(g) if g == w => {},
            Some(g) => diffs.push(format!("{key}: changed {w:016x} -> {g:016x}")),
            None => diffs.push(format!("{key}: missing from this run")),
        }
    }
    diffs.extend(
        got.keys()
            .filter(|k| !want.contains_key(*k))
            .map(|k| format!("{k}: not in the golden file")),
    );
    diffs
}
