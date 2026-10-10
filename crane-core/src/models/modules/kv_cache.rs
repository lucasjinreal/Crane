//! Shared KV cache with pre-allocated buffers and optional quantization.
//!
//! # Backends behind one contract
//!
//! [`KvCacheBackend`] is the seam: a backend stores the cache however it likes
//! but must, on [`append`](KvCacheBackend::append), take the new post-RoPE
//! `k`/`v` (`[B, num_kv_heads, S, head_dim]`) and return the *full* `k`/`v`
//! spanning all cached positions, **in the compute dtype**, ready for attention.
//! Attention logic never sees the storage representation.
//!
//! - [`FpKvCache`] — lossless f16/bf16 store (default).
//! - [`QuantKvCache`] — per-token symmetric int8 (~2x smaller) or int4 (~4x
//!   smaller), dequantized to the compute dtype on read.
//! - Future: rotation-based codecs (Hadamard + block quantization) for models
//!   whose usable window needs 2–3 bit precision. Each is just another
//!   [`KvCacheBackend`] + enum variant.

use candle_core::{D, DType, Result, Tensor};

/// Headroom (in positions) added when (re)allocating, to amortize growth.
const ROOM: usize = 256;

/// Contract every K/V cache backend honors. See the module docs.
pub trait KvCacheBackend {
    /// Append this step's `k`/`v` and return the full cached `(k, v)` in the
    /// compute dtype (`[B, num_kv_heads, seq_len + S, head_dim]`).
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying tensor operations fail.
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)>;
    /// Drop all cached state (between unrelated requests).
    fn reset(&mut self);
    /// Forget everything past `len`, keeping positions `0..len`.
    ///
    /// The cache is an append-only buffer plus a fill level, so this is just
    /// lowering the fill level: positions `0..len` stay valid and the next
    /// append overwrites the discarded tail. Used to rewind to a prompt
    /// boundary when a later request reuses that prefix.
    fn truncate(&mut self, len: usize);
    /// Number of cached positions.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Bytes currently allocated for this layer's K/V (incl. headroom + scales).
    fn byte_size(&self) -> usize;
}

fn tensor_bytes(t: Option<&Tensor>) -> usize {
    t.map_or(0, |x| x.elem_count() * x.dtype().size_in_bytes())
}

/// Which cache representation to use. Selected once per model load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheKind {
    /// Lossless f16/bf16.
    Fp,
    /// Per-token symmetric int8 (~2x smaller).
    Int8,
    /// Per-token symmetric int4, nibble-packed (~4x smaller).
    Int4,
}

impl KvCacheKind {
    /// Read from `CRANE_KV_QUANT` (`int8` → Int8, `int4` → Int4, else Fp).
    #[must_use]
    pub fn from_env() -> Self {
        match std::env::var("CRANE_KV_QUANT").as_deref() {
            Ok("int8") => Self::Int8,
            Ok("int4") => Self::Int4,
            _ => Self::Fp,
        }
    }
}

/// Per-layer K/V cache. A thin enum dispatcher over the concrete backends so
/// attention modules hold one type regardless of representation.
#[derive(Debug)]
pub enum KvCache {
    Fp(FpKvCache),
    Quant(QuantKvCache),
}

impl KvCache {
    #[must_use]
    pub fn new(kind: KvCacheKind) -> Self {
        match kind {
            KvCacheKind::Fp => Self::Fp(FpKvCache::new()),
            KvCacheKind::Int8 => Self::Quant(QuantKvCache::new(8)),
            KvCacheKind::Int4 => Self::Quant(QuantKvCache::new(4)),
        }
    }

    /// # Errors
    ///
    /// Returns an error if the underlying tensor operations fail.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Fp(c) => c.append(k, v),
            Self::Quant(c) => c.append(k, v),
        }
    }

    pub fn reset(&mut self) {
        match self {
            Self::Fp(c) => c.reset(),
            Self::Quant(c) => c.reset(),
        }
    }

    /// Keep only positions `0..len`; see [`KvCacheBackend::truncate`].
    pub fn truncate(&mut self, len: usize) {
        match self {
            Self::Fp(c) => c.truncate(len),
            Self::Quant(c) => c.truncate(len),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Fp(c) => c.len(),
            Self::Quant(c) => c.len(),
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Fp(c) => c.byte_size(),
            Self::Quant(c) => c.byte_size(),
        }
    }

    /// Full cached `(k, v)` view narrowed to the valid span, in the compute
    /// dtype (dequantized for the `Quant` backend). `None` if empty.
    ///
    /// Used by cross-layer KV sharing and batch-decode orchestration, which
    /// operate on the raw tensors above the cache backend.
    ///
    /// # Errors
    ///
    /// Returns an error if narrowing or dequantizing the underlying tensors fails.
    pub fn current_kv(&self) -> Result<Option<(Tensor, Tensor)>> {
        match self {
            Self::Fp(c) => c.current_kv(),
            Self::Quant(c) => c.current_kv(),
        }
    }

    /// Build an [`FpKvCache`]-backed cache directly from pre-built tensors
    /// (e.g. padded/stacked batch-decode buffers). `k`/`v` may hold more than
    /// `seq_len` positions as pre-allocated headroom for subsequent `append`
    /// calls. Batch-decode orchestration operates on raw FP tensors above the
    /// cache backend, so this bypasses quantization.
    #[must_use]
    pub fn from_fp(k: Tensor, v: Tensor, seq_len: usize) -> Self {
        Self::Fp(FpKvCache::from_tensors(k, v, seq_len))
    }
}

impl Default for KvCache {
    fn default() -> Self {
        Self::new(KvCacheKind::Fp)
    }
}

// ── Growth helper (shared by all backends) ────────────────────────────────

/// Append `new` along the time dim (2) into a pre-allocated buffer, growing
/// with `ROOM` headroom on overflow, and return the filled `[.., filled+S, ..]`
/// view. Works for any rank-4 tensor (codes `[B,H,S,D]` or scales `[B,H,S,1]`).
fn grow_append(buf: &mut Option<Tensor>, new: &Tensor, filled: usize) -> Result<Tensor> {
    let new = new.contiguous()?;
    let add = new.dim(2)?;
    let total = filled + add;
    match buf.take() {
        None => {
            let (b, h, _s, d) = new.dims4()?;
            let store = Tensor::zeros((b, h, add + ROOM, d), new.dtype(), new.device())?;
            store.slice_set(&new, 2, 0)?;
            let view = store.narrow(2, 0, add)?;
            *buf = Some(store);
            Ok(view)
        },
        Some(store) => {
            if total <= store.dim(2)? {
                store.slice_set(&new, 2, filled)?;
                let view = store.narrow(2, 0, total)?;
                *buf = Some(store);
                Ok(view)
            } else {
                let cur = store.narrow(2, 0, filled)?;
                let full = Tensor::cat(&[&cur, &new], 2)?;
                let (b, h, t, d) = full.dims4()?;
                let grown = Tensor::zeros((b, h, t + ROOM, d), new.dtype(), new.device())?;
                grown.slice_set(&full, 2, 0)?;
                *buf = Some(grown);
                Ok(full)
            }
        },
    }
}

// ── Fp backend (lossless) ─────────────────────────────────────────────────

/// Lossless f16/bf16 cache: pre-allocated buffers written with `slice_set`
/// (O(new tokens), not `cat`), grown with fixed headroom on overflow.
#[derive(Debug, Default)]
pub struct FpKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    seq_len: usize,
}

impl FpKvCache {
    /// Creates an empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Full cached `(k, v)` view narrowed to the valid span, or `None` if empty.
    ///
    /// # Errors
    ///
    /// Returns an error if narrowing the underlying tensors fails.
    pub fn current_kv(&self) -> Result<Option<(Tensor, Tensor)>> {
        match (&self.k, &self.v) {
            (Some(k), Some(v)) if self.seq_len > 0 => Ok(Some((
                k.narrow(2, 0, self.seq_len)?,
                v.narrow(2, 0, self.seq_len)?,
            ))),
            _ => Ok(None),
        }
    }

    /// Build a cache directly from pre-built tensors (e.g. padded/stacked
    /// batch-decode buffers). `k`/`v` may hold more than `seq_len` positions
    /// as pre-allocated headroom for subsequent `append` calls.
    #[must_use]
    pub fn from_tensors(k: Tensor, v: Tensor, seq_len: usize) -> Self {
        Self {
            k: Some(k),
            v: Some(v),
            seq_len,
        }
    }
}

impl KvCacheBackend for FpKvCache {
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let add = k.dim(2)?;
        let k_full = grow_append(&mut self.k, k, self.seq_len)?;
        let v_full = grow_append(&mut self.v, v, self.seq_len)?;
        self.seq_len += add;
        Ok((k_full, v_full))
    }

    fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.seq_len = 0;
    }

    fn truncate(&mut self, len: usize) {
        self.seq_len = self.seq_len.min(len);
    }

    fn len(&self) -> usize {
        self.seq_len
    }

    fn byte_size(&self) -> usize {
        tensor_bytes(self.k.as_ref()) + tensor_bytes(self.v.as_ref())
    }
}

// ── Quantized backend (per-token symmetric int8 / int4) ───────────────────

/// Per-token symmetric quantized K/V cache, `bits` ∈ {4, 8}.
///
/// Each `[B,H,S,head_dim]` slice is quantized per token (one f32 scale =
/// `amax / (2^(bits-1)-1)` per position+head) and stored as u8 codes:
/// - 8-bit: one code per element (`[B,H,S,head_dim]`), ~2x smaller than f16.
/// - 4-bit: two nibbles packed per byte (`[B,H,S,head_dim/2]`), ~4x smaller.
///
/// On read the filled span is dequantized to the compute dtype, so attention is
/// unchanged. Read dequantizes the whole filled cache each step — trading
/// decode bandwidth for the memory win that lets long context fit; a fused
/// dequantize-in-attention kernel is the perf follow-up.
#[derive(Debug)]
pub struct QuantKvCache {
    bits: u32,
    k_codes: Option<Tensor>,
    k_scale: Option<Tensor>,
    v_codes: Option<Tensor>,
    v_scale: Option<Tensor>,
    seq_len: usize,
    /// Compute/return dtype (set on first append).
    dtype: Option<DType>,
}

impl QuantKvCache {
    /// Creates an empty cache quantizing to `bits` bits per element.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is not 4 or 8.
    pub fn new(bits: u32) -> Self {
        assert!(bits == 4 || bits == 8, "QuantKvCache supports 4 or 8 bits");
        Self {
            bits,
            k_codes: None,
            k_scale: None,
            v_codes: None,
            v_scale: None,
            seq_len: 0,
            dtype: None,
        }
    }

    /// Full cached `(k, v)` view narrowed to the valid span and dequantized
    /// to the compute dtype, or `None` if empty.
    ///
    /// # Errors
    ///
    /// Returns an error if narrowing or dequantizing the underlying tensors fails.
    pub fn current_kv(&self) -> Result<Option<(Tensor, Tensor)>> {
        let (Some(kc), Some(ks), Some(vc), Some(vs), Some(dtype)) = (
            &self.k_codes,
            &self.k_scale,
            &self.v_codes,
            &self.v_scale,
            self.dtype,
        ) else {
            return Ok(None);
        };
        if self.seq_len == 0 {
            return Ok(None);
        }
        let kc = kc.narrow(2, 0, self.seq_len)?;
        let ks = ks.narrow(2, 0, self.seq_len)?;
        let vc = vc.narrow(2, 0, self.seq_len)?;
        let vs = vs.narrow(2, 0, self.seq_len)?;
        let k = dequantize_per_token(&kc, &ks, self.bits, dtype)?;
        let v = dequantize_per_token(&vc, &vs, self.bits, dtype)?;
        Ok(Some((k, v)))
    }
}

/// Quantize `[B,H,S,D]` per-token (symmetric) to unsigned codes in
/// `[1, 2^bits-1]` plus an f32 per-token scale. `scale = amax/qmax (+eps)`
/// guarantees `|x/scale| <= qmax`, so no clamp is needed. For 4-bit the codes
/// are nibble-packed into `[B,H,S,D/2]` (requires even D).
fn quantize_per_token(x: &Tensor, bits: u32) -> Result<(Tensor, Tensor)> {
    let qmax = f64::from((1u32 << (bits - 1)) - 1); // 127 or 7
    let offset = f64::from(1u32 << (bits - 1)); // 128 or 8
    let x = x.to_dtype(DType::F32)?;
    let amax = x.abs()?.max_keepdim(D::Minus1)?; // [B,H,S,1]
    let scale = amax.affine(1.0 / qmax, 1e-8)?;
    let codes = x.broadcast_div(&scale)?.round()?.affine(1.0, offset)?; // q + offset, in [1, 2*qmax+1]
    let codes = if bits == 8 {
        codes.to_dtype(DType::U8)?
    } else {
        pack_nibbles(&codes)? // f32 in [1,15] -> u8 [B,H,S,D/2]
    };
    Ok((codes, scale))
}

/// Inverse of [`quantize_per_token`] into `dtype`.
fn dequantize_per_token(codes: &Tensor, scale: &Tensor, bits: u32, dtype: DType) -> Result<Tensor> {
    let offset = f64::from(1u32 << (bits - 1));
    let q = if bits == 8 {
        codes.to_dtype(DType::F32)?
    } else {
        unpack_nibbles(codes)? // u8 [.., D/2] -> f32 [.., D] in [1,15]
    };
    q.affine(1.0, -offset)? // codes - offset
        .broadcast_mul(scale)?
        .to_dtype(dtype)
}

/// Pack an even-length last dim of f32 nibbles `[1,15]` into u8 `[.., D/2]`:
/// `byte = lo + hi*16` for adjacent (even, odd) pairs.
fn pack_nibbles(codes: &Tensor) -> Result<Tensor> {
    let dims = codes.dims4()?;
    let (b, h, s, d) = dims;
    debug_assert!(d % 2 == 0, "int4 packing needs even head_dim");
    let pairs = codes.reshape((b, h, s, d / 2, 2))?;
    let lo = pairs.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
    let hi = pairs.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
    (lo + hi.affine(16.0, 0.0)?)?.to_dtype(DType::U8)
}

/// Inverse of [`pack_nibbles`]: u8 `[.., D/2]` -> f32 `[.., D]` of nibbles.
fn unpack_nibbles(codes: &Tensor) -> Result<Tensor> {
    let (b, h, s, d2) = codes.dims4()?;
    let byte = codes.to_dtype(DType::F32)?;
    let hi = byte.affine(1.0 / 16.0, 0.0)?.floor()?;
    let lo = (byte - hi.affine(16.0, 0.0)?)?;
    // Interleave back: stack on a new last axis -> [.., D/2, 2] -> [.., D].
    Tensor::stack(&[&lo, &hi], D::Minus1)?.reshape((b, h, s, d2 * 2))
}

impl KvCacheBackend for QuantKvCache {
    // kc_full/ks_full/vc_full/vs_full/k_full/v_full are the natural names for
    // the six code/scale/dequantized buffers this function threads through,
    // not a typo risk.
    #[allow(clippy::similar_names)]
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let dtype = *self.dtype.get_or_insert(k.dtype());
        let add = k.dim(2)?;
        let filled = self.seq_len;

        let (kc, ks) = quantize_per_token(k, self.bits)?;
        let (vc, vs) = quantize_per_token(v, self.bits)?;

        let kc_full = grow_append(&mut self.k_codes, &kc, filled)?;
        let ks_full = grow_append(&mut self.k_scale, &ks, filled)?;
        let vc_full = grow_append(&mut self.v_codes, &vc, filled)?;
        let vs_full = grow_append(&mut self.v_scale, &vs, filled)?;
        self.seq_len += add;

        let k_full = dequantize_per_token(&kc_full, &ks_full, self.bits, dtype)?;
        let v_full = dequantize_per_token(&vc_full, &vs_full, self.bits, dtype)?;
        Ok((k_full, v_full))
    }

    fn truncate(&mut self, len: usize) {
        self.seq_len = self.seq_len.min(len);
    }

    fn reset(&mut self) {
        self.k_codes = None;
        self.k_scale = None;
        self.v_codes = None;
        self.v_scale = None;
        self.seq_len = 0;
        self.dtype = None;
    }

    fn len(&self) -> usize {
        self.seq_len
    }

    fn byte_size(&self) -> usize {
        tensor_bytes(self.k_codes.as_ref())
            + tensor_bytes(self.k_scale.as_ref())
            + tensor_bytes(self.v_codes.as_ref())
            + tensor_bytes(self.v_scale.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar::<f32>()
            .expect("to_scalar")
    }

    /// `[B,H,S,D]` F32 tensor with distinct, monotonically increasing values.
    fn varying_kv(batch: usize, heads: usize, seq: usize, dim: usize) -> Tensor {
        let data: Vec<f32> = (0..batch * heads * seq * dim)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect();
        Tensor::from_vec(data, (batch, heads, seq, dim), &Device::Cpu).expect("varying_kv")
    }

    // ── grow_append ────────────────────────────────────────────────────────

    #[test]
    fn grow_append_first_call_allocates_with_room() {
        let mut buf: Option<Tensor> = None;
        let new = varying_kv(1, 4, 8, 64);
        let view = grow_append(&mut buf, &new, 0).expect("grow_append");
        assert_eq!(view.dims(), &[1, 4, 8, 64]);
        let store = buf.expect("buf allocated");
        assert_eq!(store.dims(), &[1, 4, 8 + ROOM, 64]);
        assert!(max_abs_diff(&view, &new) < 1e-6);
    }

    #[test]
    fn grow_append_fits_in_existing_buffer() {
        let mut buf: Option<Tensor> = None;
        let new1 = varying_kv(1, 4, 8, 64);
        grow_append(&mut buf, &new1, 0).expect("first append");
        let new2 = varying_kv(1, 4, 1, 64);
        let view = grow_append(&mut buf, &new2, 8).expect("second append");
        assert_eq!(view.dims(), &[1, 4, 9, 64]);
        let store = buf.expect("buf still allocated");
        assert_eq!(store.dims(), &[1, 4, 8 + ROOM, 64]);
        let first8 = view.narrow(2, 0, 8).expect("narrow");
        assert!(max_abs_diff(&first8, &new1) < 1e-6);
        let last = view.narrow(2, 8, 1).expect("narrow");
        assert!(max_abs_diff(&last, &new2) < 1e-6);
    }

    #[test]
    fn grow_append_overflow_reallocates() {
        let mut buf: Option<Tensor> = None;
        let new1 = varying_kv(1, 4, 8, 64);
        grow_append(&mut buf, &new1, 0).expect("first append"); // buffer = 264
        let new2 = varying_kv(1, 4, 260, 64); // 8 + 260 = 268 > 264
        let view = grow_append(&mut buf, &new2, 8).expect("overflow append");
        assert_eq!(view.dims(), &[1, 4, 268, 64]);
        let store = buf.expect("buf reallocated");
        assert_eq!(store.dims(), &[1, 4, 268 + ROOM, 64]);
        let first8 = view.narrow(2, 0, 8).expect("narrow");
        assert!(max_abs_diff(&first8, &new1) < 1e-6);
        let rest = view.narrow(2, 8, 260).expect("narrow");
        assert!(max_abs_diff(&rest, &new2) < 1e-6);
    }

    #[test]
    fn grow_append_exactly_fills_then_overflows() {
        let mut buf: Option<Tensor> = None;
        let new1 = varying_kv(1, 4, 8, 64);
        grow_append(&mut buf, &new1, 0).expect("first append"); // buffer = 264
        let mut filled = 8usize;
        for _ in 0..256 {
            let step = varying_kv(1, 4, 1, 64);
            let view = grow_append(&mut buf, &step, filled).expect("fill step");
            filled += 1;
            assert_eq!(view.dim(2).expect("dim"), filled);
        }
        let store = buf.as_ref().expect("buf").clone();
        assert_eq!(store.dims(), &[1, 4, 264, 64]); // exactly full, no realloc yet

        let overflow_step = varying_kv(1, 4, 1, 64);
        let view = grow_append(&mut buf, &overflow_step, filled).expect("overflow step");
        assert_eq!(view.dims(), &[1, 4, 265, 64]);
        let store = buf.expect("buf reallocated");
        assert_eq!(store.dims(), &[1, 4, 265 + ROOM, 64]);
    }

    // ── pack_nibbles / unpack_nibbles ─────────────────────────────────────

    #[test]
    fn pack_unpack_nibbles_round_trip() {
        let n: usize = 2 * 3 * 8;
        let data: Vec<f32> = (0..n).map(|i| ((i % 15) + 1) as f32).collect();
        let x = Tensor::from_vec(data, (1, 2, 3, 8), &Device::Cpu).expect("from_vec");
        let packed = pack_nibbles(&x).expect("pack");
        assert_eq!(packed.dims(), &[1, 2, 3, 4]);
        assert_eq!(packed.dtype(), DType::U8);
        let unpacked = unpack_nibbles(&packed).expect("unpack");
        assert_eq!(unpacked.dims(), &[1, 2, 3, 8]);
        assert!(max_abs_diff(&unpacked, &x) < 1e-6);
    }

    #[test]
    fn pack_nibbles_known_values() {
        let x = Tensor::from_vec(vec![3.0f32, 12.0, 1.0, 15.0], (1, 1, 1, 4), &Device::Cpu)
            .expect("from_vec");
        let packed = pack_nibbles(&x).expect("pack");
        assert_eq!(packed.dims(), &[1, 1, 1, 2]);
        assert_eq!(packed.dtype(), DType::U8);
        let bytes = packed
            .flatten_all()
            .expect("flatten")
            .to_vec1::<u8>()
            .expect("to_vec1");
        assert_eq!(bytes, vec![3 + 12 * 16, 1 + 15 * 16]);
    }

    // ── quantize_per_token / dequantize_per_token ──────────────────────────

    #[test]
    fn quantize_dequantize_round_trip_int8() {
        let x = varying_kv(1, 4, 3, 64);
        let (codes, scale) = quantize_per_token(&x, 8).expect("quantize");
        assert_eq!(codes.dims(), &[1, 4, 3, 64]);
        assert_eq!(codes.dtype(), DType::U8);
        assert_eq!(scale.dims(), &[1, 4, 3, 1]);
        assert_eq!(scale.dtype(), DType::F32);
        let recon = dequantize_per_token(&codes, &scale, 8, DType::F32).expect("dequantize");
        let diff = max_abs_diff(&recon, &x);
        assert!(diff < 0.04, "int8 round-trip error too large: {diff}");
    }

    #[test]
    fn quantize_dequantize_round_trip_int4() {
        let x = varying_kv(1, 4, 3, 64);
        let (codes, scale) = quantize_per_token(&x, 4).expect("quantize");
        assert_eq!(codes.dims(), &[1, 4, 3, 32]);
        assert_eq!(codes.dtype(), DType::U8);
        assert_eq!(scale.dims(), &[1, 4, 3, 1]);
        let recon = dequantize_per_token(&codes, &scale, 4, DType::F32).expect("dequantize");
        let diff = max_abs_diff(&recon, &x);
        assert!(diff < 0.6, "int4 round-trip error too large: {diff}");
    }

    #[test]
    fn quantize_dequantize_dtype_preserved() {
        let x = varying_kv(1, 2, 3, 64)
            .to_dtype(DType::F16)
            .expect("to_dtype");
        let (codes, scale) = quantize_per_token(&x, 8).expect("quantize");
        let recon = dequantize_per_token(&codes, &scale, 8, DType::F16).expect("dequantize");
        assert_eq!(recon.dtype(), DType::F16);
        assert_eq!(recon.dims(), &[1, 2, 3, 64]);
    }

    // ── FpKvCache ───────────────────────────────────────────────────────────

    #[test]
    fn fp_cache_new_is_empty() {
        let cache = FpKvCache::new();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.byte_size(), 0);
        assert!(cache.current_kv().expect("current_kv").is_none());
    }

    #[test]
    fn fp_cache_append_single_token() {
        let mut cache = FpKvCache::new();
        let k = varying_kv(1, 4, 1, 64);
        let v = (varying_kv(1, 4, 1, 64) + 100.0).expect("shift");
        let (k_full, v_full) = cache.append(&k, &v).expect("append");
        assert_eq!(k_full.dims(), &[1, 4, 1, 64]);
        assert_eq!(v_full.dims(), &[1, 4, 1, 64]);
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
        assert!(max_abs_diff(&k_full, &k) < 1e-6);
        assert!(max_abs_diff(&v_full, &v) < 1e-6);
        assert!(cache.byte_size() > 0);
    }

    #[test]
    fn fp_cache_append_prefill_then_decode() {
        let mut cache = FpKvCache::new();
        let k1 = varying_kv(1, 4, 10, 64);
        let v1 = varying_kv(1, 4, 10, 64);
        let (k_full, _) = cache.append(&k1, &v1).expect("prefill");
        assert_eq!(k_full.dims(), &[1, 4, 10, 64]);
        assert_eq!(cache.len(), 10);

        let k2 = varying_kv(1, 4, 1, 64);
        let v2 = varying_kv(1, 4, 1, 64);
        let (k_full, _) = cache.append(&k2, &v2).expect("decode");
        assert_eq!(k_full.dims(), &[1, 4, 11, 64]);
        assert_eq!(cache.len(), 11);

        let first10 = k_full.narrow(2, 0, 10).expect("narrow");
        assert!(max_abs_diff(&first10, &k1) < 1e-6);
        let last = k_full.narrow(2, 10, 1).expect("narrow");
        assert!(max_abs_diff(&last, &k2) < 1e-6);
    }

    #[test]
    fn fp_cache_append_batch_gt_one() {
        let mut cache = FpKvCache::new();
        let k = varying_kv(2, 4, 3, 64);
        let v = varying_kv(2, 4, 3, 64);
        let (k_full, v_full) = cache.append(&k, &v).expect("append");
        assert_eq!(k_full.dims(), &[2, 4, 3, 64]);
        assert_eq!(cache.len(), 3);
        assert!(max_abs_diff(&k_full, &k) < 1e-6);
        assert!(max_abs_diff(&v_full, &v) < 1e-6);
    }

    #[test]
    fn fp_cache_current_kv_narrows_to_valid_span() {
        let mut cache = FpKvCache::new();
        let k = varying_kv(1, 4, 5, 64);
        let v = varying_kv(1, 4, 5, 64);
        cache.append(&k, &v).expect("append");
        let (ck, cv) = cache.current_kv().expect("current_kv").expect("some");
        assert_eq!(ck.dims(), &[1, 4, 5, 64]);
        assert_eq!(cv.dims(), &[1, 4, 5, 64]);
        assert!(max_abs_diff(&ck, &k) < 1e-6);
    }

    #[test]
    fn fp_cache_from_tensors_narrows_and_extends() {
        let k = varying_kv(1, 4, 20, 64);
        let v = varying_kv(1, 4, 20, 64);
        let cache = FpKvCache::from_tensors(k.clone(), v.clone(), 12);
        assert_eq!(cache.len(), 12);
        let (ck, _) = cache.current_kv().expect("current_kv").expect("some");
        assert_eq!(ck.dims(), &[1, 4, 12, 64]);
        let expected = k.narrow(2, 0, 12).expect("narrow");
        assert!(max_abs_diff(&ck, &expected) < 1e-6);
        assert!(cache.byte_size() > 0);
    }

    #[test]
    fn fp_cache_reset_and_truncate() {
        let mut cache = FpKvCache::new();
        let k = varying_kv(1, 4, 10, 64);
        let v = varying_kv(1, 4, 10, 64);
        cache.append(&k, &v).expect("append");

        cache.truncate(5);
        assert_eq!(cache.len(), 5);
        cache.truncate(100); // beyond len -> no-op
        assert_eq!(cache.len(), 5);

        let k2 = varying_kv(1, 4, 2, 64);
        let v2 = varying_kv(1, 4, 2, 64);
        let (k_full, _) = cache.append(&k2, &v2).expect("append after truncate");
        assert_eq!(k_full.dims(), &[1, 4, 7, 64]);
        assert_eq!(cache.len(), 7);
        let first5 = k_full.narrow(2, 0, 5).expect("narrow");
        let expected_first5 = k.narrow(2, 0, 5).expect("narrow");
        assert!(max_abs_diff(&first5, &expected_first5) < 1e-6);

        cache.reset();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.byte_size(), 0);
        assert!(cache.current_kv().expect("current_kv").is_none());

        let k3 = varying_kv(1, 4, 3, 64);
        let v3 = varying_kv(1, 4, 3, 64);
        let (k_full, _) = cache.append(&k3, &v3).expect("append after reset");
        assert_eq!(k_full.dims(), &[1, 4, 3, 64]);
        assert_eq!(cache.len(), 3);
    }

    // ── QuantKvCache ────────────────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "QuantKvCache supports 4 or 8 bits")]
    fn quant_cache_panics_on_invalid_bits() {
        let _ = QuantKvCache::new(16);
    }

    #[test]
    fn quant_cache_int8_new_is_empty() {
        let cache = QuantKvCache::new(8);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.byte_size(), 0);
        assert!(cache.current_kv().expect("current_kv").is_none());
    }

    #[test]
    fn quant_cache_int8_append_and_accuracy() {
        let mut cache = QuantKvCache::new(8);
        let k = varying_kv(1, 4, 1, 64)
            .to_dtype(DType::F16)
            .expect("to_dtype");
        let v = varying_kv(1, 4, 1, 64)
            .to_dtype(DType::F16)
            .expect("to_dtype");
        let (k_full, v_full) = cache.append(&k, &v).expect("append");
        assert_eq!(k_full.dims(), &[1, 4, 1, 64]);
        assert_eq!(k_full.dtype(), DType::F16);
        assert_eq!(cache.len(), 1);
        let diff = max_abs_diff(
            &k_full.to_dtype(DType::F32).expect("to_f32"),
            &k.to_dtype(DType::F32).expect("to_f32"),
        );
        assert!(diff < 0.05, "int8 reconstruction error too large: {diff}");
        assert_eq!(v_full.dims(), &[1, 4, 1, 64]);
    }

    #[test]
    fn quant_cache_int4_append_and_accuracy() {
        let mut cache = QuantKvCache::new(4);
        let k = varying_kv(1, 4, 1, 64)
            .to_dtype(DType::F16)
            .expect("to_dtype");
        let v = varying_kv(1, 4, 1, 64)
            .to_dtype(DType::F16)
            .expect("to_dtype");
        let (k_full, _) = cache.append(&k, &v).expect("append");
        assert_eq!(k_full.dims(), &[1, 4, 1, 64]);
        assert_eq!(k_full.dtype(), DType::F16);
        let diff = max_abs_diff(
            &k_full.to_dtype(DType::F32).expect("to_f32"),
            &k.to_dtype(DType::F32).expect("to_f32"),
        );
        assert!(diff < 0.6, "int4 reconstruction error too large: {diff}");
    }

    #[test]
    fn quant_cache_int8_prefill_then_decode() {
        let mut cache = QuantKvCache::new(8);
        let k1 = varying_kv(1, 4, 10, 64)
            .to_dtype(DType::F16)
            .expect("to_dtype");
        let v1 = varying_kv(1, 4, 10, 64)
            .to_dtype(DType::F16)
            .expect("to_dtype");
        let (k_full, _) = cache.append(&k1, &v1).expect("prefill");
        assert_eq!(k_full.dims(), &[1, 4, 10, 64]);
        assert_eq!(cache.len(), 10);

        let k2 = varying_kv(1, 4, 1, 64)
            .to_dtype(DType::F16)
            .expect("to_dtype");
        let v2 = varying_kv(1, 4, 1, 64)
            .to_dtype(DType::F16)
            .expect("to_dtype");
        let (k_full, _) = cache.append(&k2, &v2).expect("decode");
        assert_eq!(k_full.dims(), &[1, 4, 11, 64]);
        assert_eq!(k_full.dtype(), DType::F16);
        assert_eq!(cache.len(), 11);
    }

    #[test]
    fn quant_cache_int8_current_kv_matches_append() {
        let mut cache = QuantKvCache::new(8);
        let k = varying_kv(1, 4, 5, 64);
        let v = varying_kv(1, 4, 5, 64);
        let (k_append, v_append) = cache.append(&k, &v).expect("append");
        let (k_current, v_current) = cache.current_kv().expect("current_kv").expect("some");
        assert_eq!(k_current.dims(), k_append.dims());
        assert!(max_abs_diff(&k_current, &k_append) < 1e-6);
        assert!(max_abs_diff(&v_current, &v_append) < 1e-6);
    }

    #[test]
    fn quant_cache_int8_reset_and_truncate() {
        let mut cache = QuantKvCache::new(8);
        let k = varying_kv(1, 4, 10, 64);
        let v = varying_kv(1, 4, 10, 64);
        cache.append(&k, &v).expect("append");

        cache.truncate(5);
        assert_eq!(cache.len(), 5);

        let k2 = varying_kv(1, 4, 2, 64);
        let v2 = varying_kv(1, 4, 2, 64);
        let (k_full, _) = cache.append(&k2, &v2).expect("append after truncate");
        assert_eq!(k_full.dims(), &[1, 4, 7, 64]);
        assert_eq!(cache.len(), 7);

        cache.reset();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.byte_size(), 0);
        assert!(cache.current_kv().expect("current_kv").is_none());

        let k3 = varying_kv(1, 4, 3, 64);
        let v3 = varying_kv(1, 4, 3, 64);
        let (k_full, _) = cache.append(&k3, &v3).expect("append after reset");
        assert_eq!(k_full.dims(), &[1, 4, 3, 64]);
    }

    #[test]
    fn quant_cache_int8_byte_size_smaller_than_fp() {
        let mut fp_cache = FpKvCache::new();
        let mut quant_cache = QuantKvCache::new(8);
        let k = varying_kv(1, 4, 32, 64)
            .to_dtype(DType::F16)
            .expect("to_dtype");
        let v = varying_kv(1, 4, 32, 64)
            .to_dtype(DType::F16)
            .expect("to_dtype");
        fp_cache.append(&k, &v).expect("fp append");
        quant_cache.append(&k, &v).expect("quant append");
        assert!(quant_cache.byte_size() < fp_cache.byte_size());
    }

    // ── KvCache enum dispatcher ────────────────────────────────────────────

    #[test]
    fn kv_cache_fp_dispatch_is_lossless() {
        let mut cache = KvCache::new(KvCacheKind::Fp);
        let k = varying_kv(1, 4, 3, 64);
        let v = varying_kv(1, 4, 3, 64);
        let (k_full, v_full) = cache.append(&k, &v).expect("append");
        assert_eq!(cache.len(), 3);
        assert!(max_abs_diff(&k_full, &k) < 1e-6);
        assert!(max_abs_diff(&v_full, &v) < 1e-6);
    }

    #[test]
    fn kv_cache_int8_dispatch_is_lossy_within_tolerance() {
        let mut cache = KvCache::new(KvCacheKind::Int8);
        let k = varying_kv(1, 4, 3, 64);
        let v = varying_kv(1, 4, 3, 64);
        let (k_full, _) = cache.append(&k, &v).expect("append");
        assert_eq!(cache.len(), 3);
        assert!(max_abs_diff(&k_full, &k) < 0.05);
    }

    #[test]
    fn kv_cache_int4_dispatch_is_lossy_within_tolerance() {
        let mut cache = KvCache::new(KvCacheKind::Int4);
        let k = varying_kv(1, 4, 3, 64);
        let v = varying_kv(1, 4, 3, 64);
        let (k_full, _) = cache.append(&k, &v).expect("append");
        assert_eq!(cache.len(), 3);
        assert!(max_abs_diff(&k_full, &k) < 0.6);
    }

    #[test]
    fn kv_cache_default_is_fp() {
        let mut cache = KvCache::default();
        let k = varying_kv(1, 4, 2, 64);
        let v = varying_kv(1, 4, 2, 64);
        let (k_full, v_full) = cache.append(&k, &v).expect("append");
        assert!(max_abs_diff(&k_full, &k) < 1e-6);
        assert!(max_abs_diff(&v_full, &v) < 1e-6);
    }

    #[test]
    fn kv_cache_from_fp_narrows_and_extends() {
        let k = varying_kv(1, 4, 20, 64);
        let v = varying_kv(1, 4, 20, 64);
        let mut cache = KvCache::from_fp(k.clone(), v.clone(), 8);
        assert_eq!(cache.len(), 8);
        let (ck, _) = cache.current_kv().expect("current_kv").expect("some");
        assert_eq!(ck.dims(), &[1, 4, 8, 64]);

        let k2 = varying_kv(1, 4, 1, 64);
        let v2 = varying_kv(1, 4, 1, 64);
        let (k_full, _) = cache.append(&k2, &v2).expect("append");
        assert_eq!(k_full.dims(), &[1, 4, 9, 64]);
        assert_eq!(cache.len(), 9);
    }

    // ── KvCacheKind::from_env ───────────────────────────────────────────────

    #[test]
    fn kv_cache_kind_from_env() {
        // SAFETY: no other test in this crate reads or writes CRANE_KV_QUANT,
        // and all cases below run sequentially within this single test.
        unsafe {
            std::env::set_var("CRANE_KV_QUANT", "int8");
            assert_eq!(KvCacheKind::from_env(), KvCacheKind::Int8);

            std::env::set_var("CRANE_KV_QUANT", "int4");
            assert_eq!(KvCacheKind::from_env(), KvCacheKind::Int4);

            std::env::set_var("CRANE_KV_QUANT", "garbage");
            assert_eq!(KvCacheKind::from_env(), KvCacheKind::Fp);

            std::env::remove_var("CRANE_KV_QUANT");
            assert_eq!(KvCacheKind::from_env(), KvCacheKind::Fp);
        }
    }

    // ── Quant vs Fp accuracy ────────────────────────────────────────────────

    #[test]
    fn quant_int8_close_to_fp() {
        let mut fp_cache = FpKvCache::new();
        let mut quant_cache = QuantKvCache::new(8);
        let k = varying_kv(1, 4, 3, 64);
        let v = varying_kv(1, 4, 3, 64);
        let (k_fp, v_fp) = fp_cache.append(&k, &v).expect("fp append");
        let (k_quant, v_quant) = quant_cache.append(&k, &v).expect("quant append");
        assert!(max_abs_diff(&k_quant, &k_fp) < 0.04);
        assert!(max_abs_diff(&v_quant, &v_fp) < 0.04);
    }

    #[test]
    fn quant_int4_close_to_fp() {
        let mut fp_cache = FpKvCache::new();
        let mut quant_cache = QuantKvCache::new(4);
        let k = varying_kv(1, 4, 3, 64);
        let v = varying_kv(1, 4, 3, 64);
        let (k_fp, v_fp) = fp_cache.append(&k, &v).expect("fp append");
        let (k_quant, v_quant) = quant_cache.append(&k, &v).expect("quant append");
        assert!(max_abs_diff(&k_quant, &k_fp) < 0.6);
        assert!(max_abs_diff(&v_quant, &v_fp) < 0.6);
    }

    // ── Dtype edge cases ────────────────────────────────────────────────────

    #[test]
    fn fp_cache_preserves_f16_dtype() {
        let mut cache = FpKvCache::new();
        let k = varying_kv(1, 4, 2, 64)
            .to_dtype(DType::F16)
            .expect("to_dtype");
        let v = varying_kv(1, 4, 2, 64)
            .to_dtype(DType::F16)
            .expect("to_dtype");
        let (k_full, v_full) = cache.append(&k, &v).expect("append");
        assert_eq!(k_full.dtype(), DType::F16);
        assert_eq!(v_full.dtype(), DType::F16);
    }

    #[test]
    fn quant_cache_records_and_returns_input_dtype() {
        let mut cache = QuantKvCache::new(8);
        let k_f16 = varying_kv(1, 4, 2, 64)
            .to_dtype(DType::F16)
            .expect("to_dtype");
        let v_f16 = varying_kv(1, 4, 2, 64)
            .to_dtype(DType::F16)
            .expect("to_dtype");
        let (k_full, _) = cache.append(&k_f16, &v_f16).expect("append f16");
        assert_eq!(k_full.dtype(), DType::F16);

        cache.reset();
        let k_f32 = varying_kv(1, 4, 2, 64);
        let v_f32 = varying_kv(1, 4, 2, 64);
        let (k_full, _) = cache.append(&k_f32, &v_f32).expect("append f32");
        assert_eq!(k_full.dtype(), DType::F32);
    }

    // ── Decode loop simulation ──────────────────────────────────────────────

    #[test]
    fn quant_int8_twenty_decode_steps() {
        let mut cache = QuantKvCache::new(8);
        for _ in 0..20 {
            let k = varying_kv(1, 4, 1, 64);
            let v = varying_kv(1, 4, 1, 64);
            cache.append(&k, &v).expect("decode step");
        }
        assert_eq!(cache.len(), 20);
        let (k, v) = cache.current_kv().expect("current_kv").expect("some");
        assert_eq!(k.dims(), &[1, 4, 20, 64]);
        assert_eq!(v.dims(), &[1, 4, 20, 64]);
    }
}
