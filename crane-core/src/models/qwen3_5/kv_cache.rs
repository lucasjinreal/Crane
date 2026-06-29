//! K/V cache for the full-attention layers of Qwen 3.5 / Ornith.
//!
//! The hybrid model only needs this for the 1-in-4 full-attention blocks; the
//! linear-attention (GDN) blocks carry a constant-size recurrent state instead
//! (see [`crate::ops::gdn::GdnLayerCache`]), so the context-growing part of the
//! cache lives in just these layers. At long context that K/V dominates memory,
//! which is why quantizing it lets a single agent hold much more context
//! locally (e.g. Ornith-9B's full 262K window on a 24 GB GPU).
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
//! - [`Int8KvCache`] — per-token symmetric int8 (~2x smaller), dequantized to
//!   the compute dtype on read.
//! - Future: int4-packed (~4x), and rotation-based codecs (rotorquant-style)
//!   for models whose usable window is much larger (≈1M tokens) where 2-3 bit
//!   needs the rotation to stay accurate. Each is just another
//!   `KvCacheBackend` + enum variant.

use candle_core::{DType, Result, Tensor, D};

/// Headroom (in positions) added when (re)allocating, to amortize growth.
const ROOM: usize = 256;

/// Contract every K/V cache backend honors. See the module docs.
pub trait KvCacheBackend {
    /// Append this step's `k`/`v` and return the full cached `(k, v)` in the
    /// compute dtype (`[B, num_kv_heads, seq_len + S, head_dim]`).
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)>;
    /// Drop all cached state (between unrelated requests).
    fn reset(&mut self);
    /// Number of cached positions.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Bytes currently allocated for this layer's K/V (incl. headroom + scales).
    fn byte_size(&self) -> usize;
}

fn tensor_bytes(t: &Option<Tensor>) -> usize {
    t.as_ref()
        .map(|x| x.elem_count() * x.dtype().size_in_bytes())
        .unwrap_or(0)
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
    pub fn from_env() -> Self {
        match std::env::var("CRANE_KV_QUANT").as_deref() {
            Ok("int8") => Self::Int8,
            Ok("int4") => Self::Int4,
            _ => Self::Fp,
        }
    }
}

/// Per-layer K/V cache. A thin enum dispatcher over the concrete backends so
/// `FullAttention` holds one type regardless of representation.
#[derive(Debug)]
pub enum KvCache {
    Fp(FpKvCache),
    Quant(QuantKvCache),
}

impl KvCache {
    pub fn new(kind: KvCacheKind) -> Self {
        match kind {
            KvCacheKind::Fp => Self::Fp(FpKvCache::new()),
            KvCacheKind::Int8 => Self::Quant(QuantKvCache::new(8)),
            KvCacheKind::Int4 => Self::Quant(QuantKvCache::new(4)),
        }
    }

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

    pub fn len(&self) -> usize {
        match self {
            Self::Fp(c) => c.len(),
            Self::Quant(c) => c.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn byte_size(&self) -> usize {
        match self {
            Self::Fp(c) => c.byte_size(),
            Self::Quant(c) => c.byte_size(),
        }
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
        }
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
        }
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
    pub fn new() -> Self {
        Self::default()
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

    fn len(&self) -> usize {
        self.seq_len
    }

    fn byte_size(&self) -> usize {
        tensor_bytes(&self.k) + tensor_bytes(&self.v)
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
}

/// Quantize `[B,H,S,D]` per-token (symmetric) to unsigned codes in
/// `[1, 2^bits-1]` plus an f32 per-token scale. `scale = amax/qmax (+eps)`
/// guarantees `|x/scale| <= qmax`, so no clamp is needed. For 4-bit the codes
/// are nibble-packed into `[B,H,S,D/2]` (requires even D).
fn quantize_per_token(x: &Tensor, bits: u32) -> Result<(Tensor, Tensor)> {
    let qmax = ((1u32 << (bits - 1)) - 1) as f64; // 127 or 7
    let offset = (1u32 << (bits - 1)) as f64; // 128 or 8
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
    let offset = (1u32 << (bits - 1)) as f64;
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
        tensor_bytes(&self.k_codes)
            + tensor_bytes(&self.k_scale)
            + tensor_bytes(&self.v_codes)
            + tensor_bytes(&self.v_scale)
    }
}
