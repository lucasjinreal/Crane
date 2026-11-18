// SPDX-License-Identifier: MIT
//! Tiled scaled dot-product attention with online softmax.
//!
//! The standard SDPA path (materialize `Q @ K^T`, softmax, matmul `V`)
//! allocates a `[B, H, seq_len, kv_len]` score tensor whose size scales with
//! `kv_len` — how far into the context a request already is. At long context
//! lengths this single allocation can exceed available GPU memory. This
//! module replaces the single-shot computation with a loop over fixed-size
//! `kv_len` tiles, maintaining a running max and an unnormalized output
//! accumulator per query position (the online-softmax recurrence), which
//! caps the materialized score tile at [`TILE_CHUNK_BYTES`] regardless of
//! `kv_len`. The result is mathematically identical to single-shot softmax
//! — this is an exact algorithm, not an approximation.
//!
//! Two independent knobs control when this module tiles at all and how big
//! each tile is when it does:
//!
//! - [`single_shot_threshold`] (set via [`set_single_shot_threshold`]) is
//!   the full-score-matrix size, in bytes, below which [`tiled_sdpa`] skips
//!   tiling entirely and calls the plain single-shot path. This scales with
//!   real VRAM headroom (see `crane-serve`'s `run()`), since a roomy card
//!   can afford to materialize the whole score matrix for most requests.
//! - [`TILE_CHUNK_BYTES`] is the fixed per-tile size used whenever tiling
//!   *does* happen. It is intentionally not derived from headroom: a 512
//!   MiB tile benchmarked slower than a 256 MiB one (likely GPU occupancy
//!   or allocator effects, not a VRAM constraint), so a roomy headroom
//!   should widen the single-shot threshold, not the tile size.

use candle_core::{D, DType, Result, Tensor};
use std::sync::OnceLock;

/// Fixed per-tile score-chunk size used whenever [`tiled_sdpa`] actually
/// tiles. Benchmarked: 512 MiB tiles measured slower than 256 MiB ones, so
/// this is a performance-tuned constant, not a VRAM budget — it does not
/// scale with [`single_shot_threshold`] or available headroom.
const TILE_CHUNK_BYTES: usize = 256 * 1024 * 1024;

/// Fallback single-shot threshold used until [`set_single_shot_threshold`]
/// is called — the value that fixed the original ROCm OOM this module was
/// added for (`kv_len=14336` on a 16GB card). Also what callers get when
/// nothing ever calls [`set_single_shot_threshold`] (e.g. `crane-core` used
/// standalone, without `crane-serve`).
const DEFAULT_SINGLE_SHOT_THRESHOLD: usize = 256 * 1024 * 1024;

static SINGLE_SHOT_THRESHOLD: OnceLock<usize> = OnceLock::new();

/// Set the full-score-matrix size, in bytes, below which [`tiled_sdpa`]
/// skips tiling and uses the plain single-shot path.
///
/// Intended to be called once, at server startup, with the real per-process
/// VRAM headroom (the configured/physical ceiling minus the model's own
/// baseline usage and any MoE-offload reservation) — see
/// `crane-serve`'s `run()`. Only the first call takes effect; this is a
/// one-shot startup configuration, not a runtime knob. Does not affect
/// [`TILE_CHUNK_BYTES`], the size used when tiling does happen.
pub fn set_single_shot_threshold(bytes: usize) {
    let _ = SINGLE_SHOT_THRESHOLD.set(bytes);
}

/// Current single-shot threshold: whatever [`set_single_shot_threshold`]
/// set, or [`DEFAULT_SINGLE_SHOT_THRESHOLD`] if it was never called.
fn single_shot_threshold() -> usize {
    *SINGLE_SHOT_THRESHOLD.get_or_init(|| DEFAULT_SINGLE_SHOT_THRESHOLD)
}

/// Tiled scaled dot-product attention with online softmax accumulation.
///
/// Computes `softmax(Q @ K^T * scale + mask) @ V`. Takes the plain
/// single-shot path when the full `[B, H, seq_len, kv_len]` score matrix
/// fits under [`single_shot_threshold`]; otherwise tiles along `kv_len` in
/// fixed [`TILE_CHUNK_BYTES`] chunks.
///
/// # Arguments
///
/// * `q` - Query tensor, shape `[B, H, seq_len, D]`, contiguous. Any GQA
///   expansion must already be applied by the caller.
/// * `k` - Key tensor, shape `[B, H, kv_len, D]`.
/// * `v` - Value tensor, shape `[B, H, kv_len, D]`.
/// * `scale` - Attention scale factor, typically `1.0 / sqrt(head_dim)`.
/// * `mask` - Optional additive attention mask broadcastable to
///   `[B, H, seq_len, kv_len]` in every dimension except the last, which
///   must equal `kv_len` exactly (tiling narrows along it). Blocked
///   positions should hold a large negative value (e.g. `-1e9`).
///
/// # Returns
///
/// Output tensor of shape `[B, H, seq_len, D]` in `q`'s dtype.
///
/// # Errors
///
/// Returns an error if any tensor op fails (shape mismatch, device error).
#[allow(clippy::many_single_char_names)]
pub fn tiled_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (b, h, seq_len, _d) = q.dims4()?;
    let kv_len = k.dim(2)?;
    // The score tile is immediately upcast to F32 (`s_j`) and a same-shape
    // F32 `p_j` coexists with it, regardless of `q`'s input dtype — budget
    // against F32, not `q.dtype()`, or f16/bf16 models blow past the cap.
    let bytes_per_kv_col = b * h * seq_len * std::mem::size_of::<f32>();
    if bytes_per_kv_col.saturating_mul(kv_len) <= single_shot_threshold() {
        return standard_sdpa(q, k, v, scale, mask);
    }
    let tile_kv = TILE_CHUNK_BYTES
        .checked_div(bytes_per_kv_col)
        .unwrap_or(kv_len)
        .clamp(1, kv_len.max(1));
    tiled_sdpa_inner(q, k, v, scale, mask, tile_kv)
}

/// Core implementation of [`tiled_sdpa`], parameterized on tile size so
/// tests can force tiling without allocating context-length-scale tensors.
#[allow(clippy::many_single_char_names)]
pub(crate) fn tiled_sdpa_inner(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    tile_kv: usize,
) -> Result<Tensor> {
    let kv_len = k.dim(2)?;
    if tile_kv >= kv_len {
        return standard_sdpa(q, k, v, scale, mask);
    }

    let (b, h, seq_len, d) = q.dims4()?;
    let device = q.device();
    let input_dtype = q.dtype();

    let mut o_acc = Tensor::zeros((b, h, seq_len, d), DType::F32, device)?;
    // f32::MIN (not NEG_INFINITY): if a row's first tile is fully masked,
    // `m_j` is -Inf and `m_prev.broadcast_maximum(&m_j)` would stay -Inf,
    // making `correction = exp(-Inf - -Inf) = exp(NaN) = NaN` and
    // permanently contaminating that row. With a finite sentinel,
    // `max(MIN, -Inf) = MIN` gives `correction = exp(0) = 1` and
    // `p_j = exp(-Inf - MIN) = 0`, matching the fully-masked semantics
    // without NaN. On the normal path `exp(MIN - m_j)` still underflows to
    // 0.0, identical to the NEG_INFINITY behavior.
    let mut m_prev = Tensor::full(f32::MIN, (b, h, seq_len, 1), device)?;
    let mut l_prev = Tensor::zeros((b, h, seq_len, 1), DType::F32, device)?;

    let mut kv_start = 0;
    while kv_start < kv_len {
        let tile_len = tile_kv.min(kv_len - kv_start);

        let k_j = k.narrow(2, kv_start, tile_len)?;
        let v_j = v.narrow(2, kv_start, tile_len)?.to_dtype(DType::F32)?;

        let s_j = (q.matmul(&k_j.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let s_j = match mask {
            Some(m) => s_j.broadcast_add(&m.narrow(D::Minus1, kv_start, tile_len)?)?,
            None => s_j,
        };
        let s_j = s_j.to_dtype(DType::F32)?;

        let m_j = s_j.max_keepdim(D::Minus1)?;
        let m_new = m_prev.broadcast_maximum(&m_j)?;
        let correction = (&m_prev - &m_new)?.exp()?;
        let p_j = s_j.broadcast_sub(&m_new)?.exp()?;

        l_prev = (l_prev.broadcast_mul(&correction)? + p_j.sum_keepdim(D::Minus1)?)?;
        o_acc = (o_acc.broadcast_mul(&correction)? + p_j.matmul(&v_j)?)?;
        m_prev = m_new;

        kv_start += tile_len;
    }

    o_acc.broadcast_div(&l_prev)?.to_dtype(input_dtype)
}

/// Single-shot SDPA: the standard three-pass path, used when the full score
/// matrix already fits within [`single_shot_threshold`].
fn standard_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let input_dtype = q.dtype();
    let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
    let attn_weights = match mask {
        Some(m) => attn_weights.broadcast_add(m)?,
        None => attn_weights,
    };
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32)?)?
        .to_dtype(input_dtype)?;
    attn_weights.matmul(v)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    /// `[B, H, S, D]` tensor filled with `arange(0..n) * scale`, deterministic
    /// for reproducible comparisons against the reference implementation.
    fn bhsd(b: usize, h: usize, s: usize, d: usize, scale: f32, device: &Device) -> Tensor {
        (Tensor::arange(0f32, (b * h * s * d) as f32, device)
            .unwrap()
            .reshape((b, h, s, d))
            .unwrap()
            * f64::from(scale))
        .unwrap()
    }

    /// Additive causal mask of shape `[1, 1, seq_len, kv_len]`, matching the
    /// shape built in `Qwen3Model::decode`.
    fn causal_mask(seq_len: usize, kv_len: usize, kv_offset: usize, device: &Device) -> Tensor {
        let mut data = vec![0f32; seq_len * kv_len];
        for i in 0..seq_len {
            for j in 0..kv_len {
                if j > kv_offset + i {
                    data[i * kv_len + j] = -1e9;
                }
            }
        }
        Tensor::from_vec(data, (1, 1, seq_len, kv_len), device).unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn tiled_matches_standard_with_mask() {
        let device = Device::Cpu;
        let (b, h, seq_len, d, kv_len) = (1, 4, 8, 4, 32);
        let q = bhsd(b, h, seq_len, d, 0.037, &device);
        let k = bhsd(b, h, kv_len, d, 0.021, &device);
        let v = bhsd(b, h, kv_len, d, 0.013, &device);
        let scale = 1.0 / (d as f64).sqrt();
        let mask = causal_mask(seq_len, kv_len, kv_len - seq_len, &device);

        let expected = standard_sdpa(&q, &k, &v, scale, Some(&mask)).unwrap();
        let actual = tiled_sdpa_inner(&q, &k, &v, scale, Some(&mask), 8).unwrap();

        assert!(max_abs_diff(&expected, &actual) < 1e-4);
    }

    #[test]
    fn tiled_matches_standard_no_mask() {
        let device = Device::Cpu;
        let (b, h, seq_len, d, kv_len) = (1, 4, 8, 4, 32);
        let q = bhsd(b, h, seq_len, d, 0.037, &device);
        let k = bhsd(b, h, kv_len, d, 0.021, &device);
        let v = bhsd(b, h, kv_len, d, 0.013, &device);
        let scale = 1.0 / (d as f64).sqrt();

        let expected = standard_sdpa(&q, &k, &v, scale, None).unwrap();
        let actual = tiled_sdpa_inner(&q, &k, &v, scale, None, 8).unwrap();

        assert!(max_abs_diff(&expected, &actual) < 1e-4);
    }

    #[test]
    fn single_tile_matches_standard() {
        let device = Device::Cpu;
        let (b, h, seq_len, d, kv_len) = (1, 2, 4, 4, 8);
        let q = bhsd(b, h, seq_len, d, 0.05, &device);
        let k = bhsd(b, h, kv_len, d, 0.03, &device);
        let v = bhsd(b, h, kv_len, d, 0.02, &device);
        let scale = 1.0 / (d as f64).sqrt();
        let mask = causal_mask(seq_len, kv_len, kv_len - seq_len, &device);

        let expected = standard_sdpa(&q, &k, &v, scale, Some(&mask)).unwrap();
        // tile_kv >= kv_len takes the single-tile fast path directly.
        let actual = tiled_sdpa_inner(&q, &k, &v, scale, Some(&mask), kv_len).unwrap();

        assert!(max_abs_diff(&expected, &actual) < 1e-6);
    }

    #[test]
    fn uneven_last_tile() {
        let device = Device::Cpu;
        let (b, h, seq_len, d, kv_len) = (1, 2, 4, 4, 13);
        let q = bhsd(b, h, seq_len, d, 0.05, &device);
        let k = bhsd(b, h, kv_len, d, 0.03, &device);
        let v = bhsd(b, h, kv_len, d, 0.02, &device);
        let scale = 1.0 / (d as f64).sqrt();
        let mask = causal_mask(seq_len, kv_len, kv_len - seq_len, &device);

        let expected = standard_sdpa(&q, &k, &v, scale, Some(&mask)).unwrap();
        // tile_kv=4 over kv_len=13 gives tiles of [4, 4, 4, 1].
        let actual = tiled_sdpa_inner(&q, &k, &v, scale, Some(&mask), 4).unwrap();

        assert!(max_abs_diff(&expected, &actual) < 1e-4);
    }

    #[test]
    fn output_dtype_and_shape_match_input() {
        let device = Device::Cpu;
        let (b, h, seq_len, d, kv_len) = (1, 2, 4, 4, 16);
        let q = bhsd(b, h, seq_len, d, 0.05, &device);
        let k = bhsd(b, h, kv_len, d, 0.03, &device);
        let v = bhsd(b, h, kv_len, d, 0.02, &device);
        let scale = 1.0 / (d as f64).sqrt();

        let out = tiled_sdpa_inner(&q, &k, &v, scale, None, 4).unwrap();

        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(out.dims(), &[b, h, seq_len, d]);
    }

    #[test]
    fn fully_masked_first_tile_no_nan() {
        let device = Device::Cpu;
        let (b, h, seq_len, d, kv_len) = (1, 2, 2, 4, 8);
        let q = bhsd(b, h, seq_len, d, 0.05, &device);
        let k = bhsd(b, h, kv_len, d, 0.03, &device);
        let v = bhsd(b, h, kv_len, d, 0.02, &device);
        let scale = 1.0 / (d as f64).sqrt();

        // First tile (kv 0..4) is -Inf for every row, second tile (kv
        // 4..8) is fully open. With tile_kv=4 the first processed tile is
        // fully masked, exercising the m_prev == f32::MIN path: a naive
        // NEG_INFINITY sentinel would produce exp(-Inf - -Inf) = NaN here.
        let mut mask_data = vec![0f32; seq_len * kv_len];
        for i in 0..seq_len {
            for j in 0..4 {
                mask_data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_vec(mask_data, (1, 1, seq_len, kv_len), &device).unwrap();

        let expected = standard_sdpa(&q, &k, &v, scale, Some(&mask)).unwrap();
        let actual = tiled_sdpa_inner(&q, &k, &v, scale, Some(&mask), 4).unwrap();

        assert!(max_abs_diff(&expected, &actual) < 1e-4);
        let has_nan = actual
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .any(|x| x.is_nan());
        assert!(
            !has_nan,
            "tiled_sdpa produced NaN for fully-masked first tile"
        );
    }

    #[test]
    fn decode_shape_always_single_tile() {
        // seq_len == 1 (decode): bytes_per_kv_col is tiny, so the computed tile_kv
        // comfortably exceeds realistic kv_len values and tiled_sdpa falls
        // through to the single-tile fast path with zero bookkeeping
        // overhead. kv_len=4096 here is just large enough to exercise that
        // arithmetic (tile_kv works out to over 8M for this shape) without
        // allocating model-scale tensors.
        let device = Device::Cpu;
        let (b, h, seq_len, d, kv_len) = (1, 32, 1, 128, 4096);
        let q = bhsd(b, h, seq_len, d, 0.001, &device);
        let k = bhsd(b, h, kv_len, d, 0.001, &device);
        let v = bhsd(b, h, kv_len, d, 0.001, &device);
        let scale = 1.0 / (d as f64).sqrt();

        let expected = standard_sdpa(&q, &k, &v, scale, None).unwrap();
        let actual = tiled_sdpa(&q, &k, &v, scale, None).unwrap();

        assert!(max_abs_diff(&expected, &actual) < 1e-3);
    }
}
