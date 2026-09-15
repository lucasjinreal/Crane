//! Optionally-quantized K/V cache, shared by Qwen 3 and Qwen 3.5 / Ornith.
//!
//! On Qwen 3.5, this only covers the 1-in-4 full-attention blocks; the
//! linear-attention (GDN) blocks carry a constant-size recurrent state instead
//! (see [`crate::ops::gdn::GdnLayerCache`]), so the context-growing part of the
//! cache lives in just these layers. On plain Qwen 3, every layer will use
//! this cache. At long context that K/V dominates memory, which is why
//! quantizing shrinks the *stored* K/V codes. [`QuantKvCache`] also keeps a
//! full-size dequantized scratch buffer alongside those codes (see its
//! struct doc) for callers that can't use the fused kernel — CPU/Metal, or
//! CUDA/ROCm whenever [`KvCache::try_quantized_append`] declines (e.g.
//! `CRANE_QUANT_ATTN_FUSED=0`) — so peak memory isn't reduced in those
//! cases. The fused dequantize-in-attention kernel
//! (`crate::ops::fused_ops::quant_attn`) drops the scratch buffer entirely
//! for CUDA/ROCm decode and prefill, which is where the actual peak-memory
//! reduction comes from.
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

use candle_core::{D, DType, Result, Tensor};

use crate::utils::DeviceExt;

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
    /// Number of cached positions.
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Bytes currently allocated for this layer's K/V (incl. headroom + scales).
    fn byte_size(&self) -> usize;
    /// Extract the current valid-length cache state (`None` if empty), for
    /// KV-swap (preemption/resume) or batch-decode padding/stacking. Unlike
    /// [`append`](Self::append), this never dequantizes — the returned
    /// [`KvCacheState`] preserves whatever representation this backend
    /// stores internally.
    ///
    /// # Errors
    ///
    /// Returns an error if narrowing the underlying tensors fails.
    fn extract(&self) -> Result<Option<KvCacheState>>;
    /// Replace this cache's contents with `state`, discarding whatever was
    /// there before.
    ///
    /// # Errors
    ///
    /// Returns an error if `state`'s representation doesn't match this
    /// backend (e.g. installing a `Quant` state into an `FpKvCache`), or its
    /// bit width doesn't match.
    fn install(&mut self, state: KvCacheState) -> Result<()>;
}

/// A per-layer cache's extracted state: either a plain `(K, V)` pair or a
/// quantized 4-tensor representation. This is the seam that lets KV-swap and
/// batch-decode padding/stacking (`crane-core/src/models/qwen3/modeling.rs`)
/// and the engine's KV-swap/eviction bookkeeping (`crane-serve`) operate on
/// either representation without special-casing which one is active.
#[derive(Debug, Clone)]
pub enum KvCacheState {
    /// Lossless cache: `(K, V)`, each `[B, num_kv_heads, S, head_dim]`.
    Fp(Tensor, Tensor),
    /// Quantized cache: per-token codes + scales for K and V (see
    /// [`QuantKvCache`]'s doc comment for the exact layout), plus the bit
    /// width they were quantized at.
    ///
    /// `bits` must be 4 or 8; [`Self::kind`] panics otherwise.
    Quant {
        /// Quantized K codes, `[B, num_kv_heads, S, head_dim]` (8-bit) or
        /// `[B, num_kv_heads, S, head_dim/2]` (4-bit, nibble-packed).
        k_codes: Tensor,
        /// Per-token f32 dequantization scale for `k_codes`.
        k_scale: Tensor,
        /// Quantized V codes, same layout as `k_codes`.
        v_codes: Tensor,
        /// Per-token f32 dequantization scale for `v_codes`.
        v_scale: Tensor,
        /// Bit width `k_codes`/`v_codes` were quantized at: 4 or 8.
        bits: u32,
    },
}

impl KvCacheState {
    /// Number of cached positions (dim 2 of any constituent tensor — they
    /// all grow together).
    ///
    /// # Errors
    ///
    /// Returns an error if the constituent tensors aren't rank-4.
    pub fn seq_len(&self) -> Result<usize> {
        self.buffers()[0].dim(2)
    }

    /// Total bytes occupied by every constituent tensor.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        self.buffers()
            .into_iter()
            .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
            .sum()
    }

    /// The [`KvCacheKind`] this state represents.
    ///
    /// # Panics
    ///
    /// Panics if a `Quant` variant's `bits` is neither 4 nor 8.
    #[must_use]
    pub fn kind(&self) -> KvCacheKind {
        match self {
            Self::Fp(..) => KvCacheKind::Fp,
            Self::Quant { bits: 8, .. } => KvCacheKind::Int8,
            Self::Quant { bits: 4, .. } => KvCacheKind::Int4,
            Self::Quant { bits, .. } => panic!("KvCacheState::Quant: unsupported bits {bits}"),
        }
    }

    /// Dequantize to a plain `(K, V)` pair in `dtype`. A no-op cast for
    /// `Fp` (already plain); for `Quant`, this applies
    /// [`dequantize_per_token`] to the full cache in one call — unlike
    /// [`QuantKvCache::append`], which dequantizes only newly appended
    /// tokens and keeps the rest in a persistent scratch buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying tensor operations fail.
    pub fn to_fp_pair(&self, dtype: DType) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Fp(k, v) => Ok((k.to_dtype(dtype)?, v.to_dtype(dtype)?)),
            Self::Quant {
                k_codes,
                k_scale,
                v_codes,
                v_scale,
                bits,
            } => Ok((
                dequantize_per_token(k_codes, k_scale, *bits, dtype)?,
                dequantize_per_token(v_codes, v_scale, *bits, dtype)?,
            )),
        }
    }

    /// Inverse of [`Self::to_fp_pair`]: wrap (or quantize) a complete `(K, V)`
    /// pair into `kind`'s representation. Unlike [`KvCacheBackend::append`]'s
    /// incremental per-token quantization, this quantizes the whole tensor
    /// in one pass — used to re-quantize a sequence's state after a
    /// batch-decode round, where the batch itself always runs in plain
    /// compute dtype regardless of storage representation.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying tensor operations fail.
    pub fn from_fp_pair(k: &Tensor, v: &Tensor, kind: KvCacheKind) -> Result<Self> {
        match kind {
            KvCacheKind::Fp => Ok(Self::Fp(k.clone(), v.clone())),
            KvCacheKind::Int8 | KvCacheKind::Int4 => {
                let bits = if kind == KvCacheKind::Int8 { 8 } else { 4 };
                let (k_codes, k_scale) = quantize_per_token(k, bits)?;
                let (v_codes, v_scale) = quantize_per_token(v, bits)?;
                Ok(Self::Quant {
                    k_codes,
                    k_scale,
                    v_codes,
                    v_scale,
                    bits,
                })
            },
        }
    }

    /// This state's constituent tensors, in a fixed order matching
    /// [`Self::from_buffers`]'s expectations: `(K, V)` for `Fp`;
    /// `(k_codes, k_scale, v_codes, v_scale)` for `Quant`. Lets batch-decode
    /// padding/stacking (which pads/stacks/narrows along dim 2) treat both
    /// representations identically, by number of buffers rather than by name.
    #[must_use]
    pub fn buffers(&self) -> Vec<&Tensor> {
        match self {
            Self::Fp(k, v) => vec![k, v],
            Self::Quant {
                k_codes,
                k_scale,
                v_codes,
                v_scale,
                ..
            } => vec![k_codes, k_scale, v_codes, v_scale],
        }
    }

    /// Inverse of [`Self::buffers`]: reassemble a state from an ordered list
    /// of tensors and the kind they belong to.
    ///
    /// # Errors
    ///
    /// Returns an error if `tensors.len()` doesn't match what `kind` expects
    /// (2 for `Fp`, 4 for `Int8`/`Int4`).
    // k_codes/k_scale/v_codes/v_scale are the natural names for the four
    // buffers this function reassembles, not a typo risk.
    #[allow(clippy::similar_names)]
    pub fn from_buffers(tensors: Vec<Tensor>, kind: KvCacheKind) -> Result<Self> {
        match kind {
            KvCacheKind::Fp => {
                let [k, v]: [Tensor; 2] = tensors.try_into().map_err(|v: Vec<Tensor>| {
                    candle_core::Error::Msg(format!(
                        "KvCacheState::from_buffers: Fp expects 2 tensors, got {}",
                        v.len()
                    ))
                })?;
                Ok(Self::Fp(k, v))
            },
            KvCacheKind::Int8 | KvCacheKind::Int4 => {
                let bits = if kind == KvCacheKind::Int8 { 8 } else { 4 };
                let [k_codes, k_scale, v_codes, v_scale]: [Tensor; 4] =
                    tensors.try_into().map_err(|v: Vec<Tensor>| {
                        candle_core::Error::Msg(format!(
                            "KvCacheState::from_buffers: Quant expects 4 tensors, got {}",
                            v.len()
                        ))
                    })?;
                Ok(Self::Quant {
                    k_codes,
                    k_scale,
                    v_codes,
                    v_scale,
                    bits,
                })
            },
        }
    }
}

fn tensor_bytes(t: Option<&Tensor>) -> usize {
    t.map_or(0, |x| x.elem_count() * x.dtype().size_in_bytes())
}

/// Zero-copy borrow of a [`QuantKvCache`]'s raw codes/scales for the current
/// valid region, for fused-kernel attention
/// (`crate::ops::fused_ops::quant_attn`) that dequantizes on the fly instead
/// of reading a persistent dequantized scratch buffer. Returned by
/// [`QuantKvCache::quantized_append`]/[`KvCache::try_quantized_append`].
pub struct QuantizedKvRef {
    /// Quantized K codes, `[B, num_kv_heads, S, head_dim]` (8-bit) or
    /// `[B, num_kv_heads, S, head_dim/2]` (4-bit, nibble-packed).
    pub k_codes: Tensor,
    /// Per-token f32 dequantization scale for `k_codes`.
    pub k_scale: Tensor,
    /// Quantized V codes, same layout as `k_codes`.
    pub v_codes: Tensor,
    /// Per-token f32 dequantization scale for `v_codes`.
    pub v_scale: Tensor,
    /// Bit width `k_codes`/`v_codes` were quantized at: 4 or 8.
    pub bits: u32,
    /// Number of valid cached positions (dim 2 of the code tensors).
    pub seq_len: usize,
}

impl QuantizedKvRef {
    /// Dequantize `k_codes`/`k_scale` to `dtype`. CPU-only fallback path for
    /// callers that can't run the fused kernel (see
    /// `crate::ops::fused_ops::quant_attn`'s module doc).
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying tensor operations fail.
    pub fn dequantize_k(&self, dtype: DType) -> Result<Tensor> {
        dequantize_per_token(&self.k_codes, &self.k_scale, self.bits, dtype)
    }

    /// Dequantize `v_codes`/`v_scale` to `dtype`. See [`Self::dequantize_k`].
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying tensor operations fail.
    pub fn dequantize_v(&self, dtype: DType) -> Result<Tensor> {
        dequantize_per_token(&self.v_codes, &self.v_scale, self.bits, dtype)
    }
}

/// Bytes of KV cache one sequence consumes per generated token when K/V
/// are stored as `bits`-wide (4 or 8) quantized codes plus a per-token
/// f32 scale per head (see [`QuantKvCache`]).
///
/// `num_kv_layers` is the number of layers that carry a growing K/V cache
/// (all layers for Qwen 3, only the full-attention layers for Qwen 3.5's
/// hybrid architecture).
///
/// # Panics
///
/// Panics if `bits` is neither 4 nor 8.
#[must_use]
pub fn quantized_kv_bytes_per_token(
    bits: u32,
    num_kv_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> u64 {
    const SCALE_BYTES_PER_TOKEN_PER_HEAD: u64 = 4;
    assert!(
        bits == 4 || bits == 8,
        "quantized_kv_bytes_per_token: bits must be 4 or 8"
    );
    let head_dim = head_dim as u64;
    let code_bytes = if bits == 8 {
        head_dim
    } else {
        head_dim.div_ceil(2)
    };
    2 * num_kv_layers as u64 * num_kv_heads as u64 * (code_bytes + SCALE_BYTES_PER_TOKEN_PER_HEAD)
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
        std::env::var("CRANE_KV_QUANT")
            .ok()
            .and_then(|s| Self::parse(&s))
            .unwrap_or(Self::Fp)
    }

    /// Parse an explicit value (e.g. from a `--kv-quant` CLI flag),
    /// trimmed and lowercased. `None` for anything other than
    /// `"int8"`/`"int4"` — unlike [`Self::from_env`], which silently falls
    /// back to `Fp` for an unset or unrecognized env var, an explicit CLI
    /// value should fail loudly on a typo rather than silently do nothing.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "int8" => Some(Self::Int8),
            "int4" => Some(Self::Int4),
            _ => None,
        }
    }

    /// Resolve the effective per-token KV byte cost for VRAM pricing.
    ///
    /// Returns [`quantized_kv_bytes_per_token`] when this is an `Int8`/`Int4`
    /// cache and the fused dequantize-in-attention kernel
    /// (`crate::ops::fused_ops::quant_attn`) covers the sequence's entire
    /// lifetime. Returns compute-dtype pricing otherwise, since unfused
    /// decode/prefill or batch-decode's `to_fp_pair()`/`from_fp_pair()`
    /// round-trip dequantize to the compute dtype.
    ///
    /// `num_kv_layers` is the number of layers that carry a growing K/V
    /// cache (all layers for Qwen 3, only the full-attention layers for
    /// Qwen 3.5).
    #[must_use]
    pub fn effective_kv_bytes_per_token(
        self,
        fused_covers_full_lifetime: bool,
        num_kv_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype_bytes: usize,
    ) -> u64 {
        match self {
            Self::Int8 if fused_covers_full_lifetime => {
                quantized_kv_bytes_per_token(8, num_kv_layers, num_kv_heads, head_dim)
            },
            Self::Int4 if fused_covers_full_lifetime => {
                quantized_kv_bytes_per_token(4, num_kv_layers, num_kv_heads, head_dim)
            },
            Self::Fp | Self::Int8 | Self::Int4 => {
                2 * num_kv_layers as u64
                    * num_kv_heads as u64
                    * head_dim as u64
                    * dtype_bytes as u64
            },
        }
    }

    /// Human-readable description for startup logging, so it's always
    /// possible to tell which mode actually ended up active (an explicit
    /// `--kv-quant`/`new_with_kv_kind` request vs. the `CRANE_KV_QUANT` env
    /// var vs. neither) without re-deriving it from the request itself.
    ///
    /// The fused dequantize-in-attention kernel
    /// (`crate::ops::fused_ops::quant_attn`) now covers both decode and
    /// prefill on CUDA/ROCm, so peak VRAM is actually reduced there. On
    /// CPU/Metal (no fused kernel), or for batch-decode's
    /// `to_fp_pair()`/`from_fp_pair()` round-trip, the persistent dequant
    /// scratch buffer still makes an unfused step cost as much as `Fp`, not
    /// less — this description doesn't distinguish those cases from the
    /// caller's actual device/config, since it's a static per-`KvCacheKind`
    /// string.
    #[must_use]
    pub fn describe(&self) -> &'static str {
        match self {
            Self::Fp => "fp16/bf16 (unquantized)",
            Self::Int8 => {
                "int8 (quantized; stored codes ~2x smaller than fp16; peak VRAM is reduced on \
                 CUDA/ROCm's fused attention path, but not yet on CPU/Metal or for batch decode, \
                 and not if CRANE_QUANT_ATTN_FUSED=0 is set)"
            },
            Self::Int4 => {
                "int4 (quantized; stored codes ~4x smaller than fp16; peak VRAM is reduced on \
                 CUDA/ROCm's fused attention path, but not yet on CPU/Metal or for batch decode, \
                 and not if CRANE_QUANT_ATTN_FUSED=0 is set)"
            },
        }
    }
}

/// Per-layer K/V cache. A thin enum dispatcher over the concrete backends so
/// `FullAttention` holds one type regardless of representation.
#[derive(Debug)]
pub enum KvCache {
    /// Lossless f16/bf16 storage.
    Fp(FpKvCache),
    /// Quantized int8/int4 storage.
    Quant(QuantKvCache),
}

impl KvCache {
    /// Construct an empty cache of the given representation.
    #[must_use]
    pub fn new(kind: KvCacheKind) -> Self {
        match kind {
            KvCacheKind::Fp => Self::Fp(FpKvCache::new()),
            KvCacheKind::Int8 => Self::Quant(QuantKvCache::new(8)),
            KvCacheKind::Int4 => Self::Quant(QuantKvCache::new(4)),
        }
    }

    /// See [`KvCacheBackend::append`].
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying tensor operations fail.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Fp(c) => c.append(k, v),
            Self::Quant(c) => c.append(k, v),
        }
    }

    /// Try the fused-kernel append path: quantize and store `k`/`v` without
    /// maintaining the dequantized scratch buffer, for GPU decode and
    /// prefill attention that dequantizes on the fly
    /// (`crate::ops::fused_ops::quant_attn`).
    ///
    /// Returns `Ok(None)` — meaning the caller should fall back to
    /// [`Self::append`] instead — when this cache is [`Self::Fp`] (nothing to
    /// fuse), `k` is on a device without a fused kernel (only CUDA and `ROCm`
    /// have one — notably not Metal, which would otherwise silently hit the
    /// `quant_attn` module's O(total cached) CPU fallback), `k`'s dtype isn't
    /// one the fused kernels support, or the `CRANE_QUANT_ATTN_FUSED` env var
    /// is set to `"0"`.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying tensor operations fail.
    pub fn try_quantized_append(
        &mut self,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<Option<QuantizedKvRef>> {
        let has_fused_kernel = k.device().is_cuda() || k.device().is_rocm();
        let dtype_supported = crate::ops::fused_ops::quant_attn::supports_compute_dtype(k.dtype());
        if !has_fused_kernel
            || !dtype_supported
            || crate::ops::fused_ops::quant_attn::fused_disabled()
        {
            return Ok(None);
        }
        match self {
            Self::Fp(_) => Ok(None),
            Self::Quant(c) => c.quantized_append(k, v).map(Some),
        }
    }

    /// See [`KvCacheBackend::reset`].
    pub fn reset(&mut self) {
        match self {
            Self::Fp(c) => c.reset(),
            Self::Quant(c) => c.reset(),
        }
    }

    /// See [`KvCacheBackend::len`].
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Fp(c) => c.len(),
            Self::Quant(c) => c.len(),
        }
    }

    /// Whether the cache holds any cached positions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// See [`KvCacheBackend::byte_size`].
    #[must_use]
    pub fn byte_size(&self) -> usize {
        match self {
            Self::Fp(c) => c.byte_size(),
            Self::Quant(c) => c.byte_size(),
        }
    }

    /// # Errors
    ///
    /// Returns an error if narrowing the underlying tensors fails.
    pub fn extract(&self) -> Result<Option<KvCacheState>> {
        match self {
            Self::Fp(c) => c.extract(),
            Self::Quant(c) => c.extract(),
        }
    }

    /// # Errors
    ///
    /// Returns an error if `state`'s representation doesn't match this
    /// cache's own kind.
    pub fn install(&mut self, state: KvCacheState) -> Result<()> {
        match self {
            Self::Fp(c) => c.install(state),
            Self::Quant(c) => c.install(state),
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
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Install a pre-built buffer whose first `valid_len` positions along
    /// dim 2 are real cached content, with anything beyond that treated as
    /// pre-allocated headroom (as [`grow_append`] itself would leave). Used
    /// by batch-decode setup, which pre-sizes the batched buffer to fit a
    /// caller-chosen number of extra decode rounds without reallocating —
    /// unlike [`KvCacheBackend::install`], which assumes the given tensor's
    /// length *is* the valid length exactly.
    pub fn install_with_headroom(&mut self, k: Tensor, v: Tensor, valid_len: usize) {
        self.k = Some(k);
        self.v = Some(v);
        self.seq_len = valid_len;
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
        tensor_bytes(self.k.as_ref()) + tensor_bytes(self.v.as_ref())
    }

    fn extract(&self) -> Result<Option<KvCacheState>> {
        if self.seq_len == 0 {
            return Ok(None);
        }
        let (Some(k), Some(v)) = (self.k.as_ref(), self.v.as_ref()) else {
            return Ok(None);
        };
        Ok(Some(KvCacheState::Fp(
            k.narrow(2, 0, self.seq_len)?,
            v.narrow(2, 0, self.seq_len)?,
        )))
    }

    fn install(&mut self, state: KvCacheState) -> Result<()> {
        let KvCacheState::Fp(k, v) = state else {
            candle_core::bail!("FpKvCache::install: expected an Fp state");
        };
        self.seq_len = k.dim(2)?;
        self.k = Some(k);
        self.v = Some(v);
        Ok(())
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
/// On read via [`KvCacheBackend::append`] the filled span is dequantized to
/// the compute dtype, so attention is unchanged. Only newly appended tokens
/// are dequantized each step — the result is kept in a persistent scratch
/// buffer alongside the quantized codes, so a step costs O(new tokens), not
/// O(total cached). This path's scratch buffer holds the full dequantized
/// history, so peak memory is not reduced versus [`FpKvCache`] for it —
/// quantization only shrinks the stored codes, not what attention actually
/// reads from.
///
/// [`Self::quantized_append`] is the alternative, scratch-buffer-free path:
/// GPU decode and prefill on CUDA/ROCm use it via
/// [`KvCache::try_quantized_append`], reading codes/scales directly inside
/// the fused dequantize-in-attention kernels
/// (`crate::ops::fused_ops::quant_attn`) instead of dequantizing into the
/// scratch buffer at all — that's where the actual peak-memory reduction
/// comes from.
#[derive(Debug)]
pub struct QuantKvCache {
    bits: u32,
    k_codes: Option<Tensor>,
    k_scale: Option<Tensor>,
    v_codes: Option<Tensor>,
    v_scale: Option<Tensor>,
    /// Persistent dequantized key scratch, grown the same way as the codes —
    /// avoids re-dequantizing already-cached positions on every step.
    k_dequant: Option<Tensor>,
    /// Persistent dequantized value scratch, grown the same way as the codes —
    /// avoids re-dequantizing already-cached positions on every step.
    v_dequant: Option<Tensor>,
    seq_len: usize,
    /// Compute/return dtype (set on first append).
    dtype: Option<DType>,
}

impl QuantKvCache {
    /// # Panics
    ///
    /// Panics if `bits` is neither 4 nor 8.
    #[must_use]
    pub fn new(bits: u32) -> Self {
        assert!(bits == 4 || bits == 8, "QuantKvCache supports 4 or 8 bits");
        Self {
            bits,
            k_codes: None,
            k_scale: None,
            v_codes: None,
            v_scale: None,
            k_dequant: None,
            v_dequant: None,
            seq_len: 0,
            dtype: None,
        }
    }

    /// Quantize and store `k`/`v`, without maintaining the dequantized
    /// scratch buffer — for GPU decode and prefill attention that
    /// dequantizes on the fly (`crate::ops::fused_ops::quant_attn`) instead
    /// of reading it back.
    ///
    /// Invalidates the scratch buffer (`k_dequant`/`v_dequant` become
    /// `None`); a subsequent [`KvCacheBackend::append`] call lazily rebuilds
    /// it from the stored codes/scales, the same way it already does after
    /// [`KvCacheBackend::install`].
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying tensor operations fail.
    // k_codes/k_scale/v_codes/v_scale are the natural names for the four
    // buffers this function threads through, not a typo risk.
    #[allow(clippy::similar_names)]
    pub fn quantized_append(&mut self, k: &Tensor, v: &Tensor) -> Result<QuantizedKvRef> {
        let add = k.dim(2)?;
        let filled = self.seq_len;

        let (kc, ks) = quantize_per_token(k, self.bits)?;
        let (vc, vs) = quantize_per_token(v, self.bits)?;

        let k_codes = grow_append(&mut self.k_codes, &kc, filled)?;
        let k_scale = grow_append(&mut self.k_scale, &ks, filled)?;
        let v_codes = grow_append(&mut self.v_codes, &vc, filled)?;
        let v_scale = grow_append(&mut self.v_scale, &vs, filled)?;

        self.k_dequant = None;
        self.v_dequant = None;
        self.seq_len += add;

        Ok(QuantizedKvRef {
            k_codes,
            k_scale,
            v_codes,
            v_scale,
            bits: self.bits,
            seq_len: self.seq_len,
        })
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

/// Inverse of [`quantize_per_token`] into `dtype`. `pub(crate)` so
/// `crate::ops::fused_ops::quant_attn`'s CPU fallback can reuse it directly.
pub(crate) fn dequantize_per_token(
    codes: &Tensor,
    scale: &Tensor,
    bits: u32,
    dtype: DType,
) -> Result<Tensor> {
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
///
/// # Panics
///
/// Panics if the last dim (`head_dim`) is odd.
fn pack_nibbles(codes: &Tensor) -> Result<Tensor> {
    let dims = codes.dims4()?;
    let (b, h, s, d) = dims;
    assert!(d % 2 == 0, "int4 packing needs even head_dim");
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
    // kc/ks/vc/vs/kd_new/vd_new are the natural names for the code/scale/
    // dequantized-new-token buffers this function threads through, not a
    // typo risk.
    #[allow(clippy::similar_names)]
    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let dtype = *self.dtype.get_or_insert(k.dtype());
        let add = k.dim(2)?;
        let filled = self.seq_len;

        let (kc, ks) = quantize_per_token(k, self.bits)?;
        let (vc, vs) = quantize_per_token(v, self.bits)?;

        grow_append(&mut self.k_codes, &kc, filled)?;
        grow_append(&mut self.k_scale, &ks, filled)?;
        grow_append(&mut self.v_codes, &vc, filled)?;
        grow_append(&mut self.v_scale, &vs, filled)?;

        // Scratch is absent right after `install` (which can't build it
        // without knowing the compute dtype) — seed it once from the
        // installed codes/scales before appending the new tokens below.
        if self.k_dequant.is_none() && filled > 0 {
            let (Some(kc_buf), Some(ks_buf), Some(vc_buf), Some(vs_buf)) = (
                self.k_codes.as_ref(),
                self.k_scale.as_ref(),
                self.v_codes.as_ref(),
                self.v_scale.as_ref(),
            ) else {
                candle_core::bail!("QuantKvCache::append: filled cache missing codes/scales");
            };
            let k_old = dequantize_per_token(
                &kc_buf.narrow(2, 0, filled)?,
                &ks_buf.narrow(2, 0, filled)?,
                self.bits,
                dtype,
            )?;
            let v_old = dequantize_per_token(
                &vc_buf.narrow(2, 0, filled)?,
                &vs_buf.narrow(2, 0, filled)?,
                self.bits,
                dtype,
            )?;
            grow_append(&mut self.k_dequant, &k_old, 0)?;
            grow_append(&mut self.v_dequant, &v_old, 0)?;
        }

        let kd_new = dequantize_per_token(&kc, &ks, self.bits, dtype)?;
        let vd_new = dequantize_per_token(&vc, &vs, self.bits, dtype)?;
        let k_full = grow_append(&mut self.k_dequant, &kd_new, filled)?;
        let v_full = grow_append(&mut self.v_dequant, &vd_new, filled)?;
        self.seq_len += add;

        Ok((k_full, v_full))
    }

    fn reset(&mut self) {
        self.k_codes = None;
        self.k_scale = None;
        self.v_codes = None;
        self.v_scale = None;
        self.k_dequant = None;
        self.v_dequant = None;
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
            + tensor_bytes(self.k_dequant.as_ref())
            + tensor_bytes(self.v_dequant.as_ref())
    }

    // k_codes/k_scale/v_codes/v_scale are the natural names for the four
    // buffers this function reads, not a typo risk.
    #[allow(clippy::similar_names)]
    fn extract(&self) -> Result<Option<KvCacheState>> {
        if self.seq_len == 0 {
            return Ok(None);
        }
        let (Some(k_codes), Some(k_scale), Some(v_codes), Some(v_scale)) = (
            self.k_codes.as_ref(),
            self.k_scale.as_ref(),
            self.v_codes.as_ref(),
            self.v_scale.as_ref(),
        ) else {
            return Ok(None);
        };
        Ok(Some(KvCacheState::Quant {
            k_codes: k_codes.narrow(2, 0, self.seq_len)?,
            k_scale: k_scale.narrow(2, 0, self.seq_len)?,
            v_codes: v_codes.narrow(2, 0, self.seq_len)?,
            v_scale: v_scale.narrow(2, 0, self.seq_len)?,
            bits: self.bits,
        }))
    }

    // k_codes/k_scale/v_codes/v_scale are the natural names for the four
    // buffers this function installs, not a typo risk.
    //
    // Clears the dequantized scratch buffers (rebuilt lazily on the next
    // `append`, once the compute dtype is known again) — that next `append`
    // pays a one-time O(filled) reseed cost before returning to the usual
    // O(new tokens) steady state.
    #[allow(clippy::similar_names)]
    fn install(&mut self, state: KvCacheState) -> Result<()> {
        let KvCacheState::Quant {
            k_codes,
            k_scale,
            v_codes,
            v_scale,
            bits,
        } = state
        else {
            candle_core::bail!("QuantKvCache::install: expected a Quant state");
        };
        if bits != self.bits {
            candle_core::bail!(
                "QuantKvCache::install: bit-width mismatch ({bits} vs {})",
                self.bits
            );
        }
        self.seq_len = k_codes.dim(2)?;
        self.k_codes = Some(k_codes);
        self.k_scale = Some(k_scale);
        self.v_codes = Some(v_codes);
        self.v_scale = Some(v_scale);
        // Rebuilt lazily on the next `append`, once the compute dtype is known.
        self.k_dequant = None;
        self.v_dequant = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn rand_kv(b: usize, h: usize, s: usize, d: usize) -> Tensor {
        // Scaled up so quantization error isn't dominated by eps.
        (Tensor::randn(0f32, 1f32, (b, h, s, d), &Device::Cpu).unwrap() * 4.0).unwrap()
    }

    // int8: 1 byte/code + 4 byte f32 scale per head -> 2*48*4*(128+4) bytes/token.
    #[test]
    fn quantized_kv_bytes_per_token_int8() {
        assert_eq!(
            quantized_kv_bytes_per_token(8, 48, 4, 128),
            2 * 48 * 4 * (128 + 4)
        );
    }

    // int4: nibble-packed codes (head_dim/2 bytes) + 4 byte f32 scale per head.
    #[test]
    fn quantized_kv_bytes_per_token_int4() {
        assert_eq!(
            quantized_kv_bytes_per_token(4, 48, 4, 128),
            2 * 48 * 4 * (64 + 4)
        );
    }

    #[test]
    #[should_panic(expected = "bits must be 4 or 8")]
    fn quantized_kv_bytes_per_token_rejects_invalid_bits() {
        let _ = quantized_kv_bytes_per_token(16, 48, 4, 128);
    }

    // Fused-not-covering-lifetime must fall back to compute-dtype pricing
    // even for a quantized KvCacheKind, regardless of bits.
    #[test]
    fn effective_kv_bytes_per_token_falls_back_to_compute_dtype_when_not_fused() {
        let fp16_bytes = 2 * 48 * 4 * 128 * 2;
        assert_eq!(
            KvCacheKind::Int8.effective_kv_bytes_per_token(false, 48, 4, 128, 2),
            fp16_bytes
        );
        assert_eq!(
            KvCacheKind::Fp.effective_kv_bytes_per_token(true, 48, 4, 128, 2),
            fp16_bytes
        );
    }

    // Fused-covers-lifetime must use the smaller quantized price for Int8/Int4.
    #[test]
    fn effective_kv_bytes_per_token_uses_quantized_price_when_fused() {
        assert_eq!(
            KvCacheKind::Int8.effective_kv_bytes_per_token(true, 48, 4, 128, 2),
            quantized_kv_bytes_per_token(8, 48, 4, 128)
        );
        assert_eq!(
            KvCacheKind::Int4.effective_kv_bytes_per_token(true, 48, 4, 128, 2),
            quantized_kv_bytes_per_token(4, 48, 4, 128)
        );
    }

    #[test]
    fn kv_cache_kind_parse_recognizes_int8_and_int4() {
        assert_eq!(KvCacheKind::parse("int8"), Some(KvCacheKind::Int8));
        assert_eq!(KvCacheKind::parse("int4"), Some(KvCacheKind::Int4));
    }

    #[test]
    fn kv_cache_kind_parse_rejects_unknown_values() {
        assert_eq!(KvCacheKind::parse("fp"), None);
        assert_eq!(KvCacheKind::parse(""), None);
    }

    #[test]
    fn kv_cache_kind_parse_normalizes_case_and_whitespace() {
        assert_eq!(KvCacheKind::parse("INT8"), Some(KvCacheKind::Int8));
        assert_eq!(KvCacheKind::parse(" int4 "), Some(KvCacheKind::Int4));
    }

    // Per-token symmetric quantization guarantees |x/scale| <= qmax, so the
    // max round-trip error is bounded by one code's worth of scale.
    #[test]
    fn quantize_dequantize_int8_within_bound() {
        let x = rand_kv(1, 4, 8, 128);
        let (codes, scale) = quantize_per_token(&x, 8).unwrap();
        let y = dequantize_per_token(&codes, &scale, 8, DType::F32).unwrap();
        let amax = x.abs().unwrap().max_keepdim(D::Minus1).unwrap();
        let bound = (amax.affine(1.0 / 127.0, 1e-6).unwrap())
            .broadcast_as(x.shape())
            .unwrap();
        let err = (x - y).unwrap().abs().unwrap();
        let over = err
            .gt(&bound)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<u8>()
            .unwrap();
        assert_eq!(
            over, 0,
            "int8 round-trip error exceeded the per-token scale bound"
        );
    }

    #[test]
    fn quantize_dequantize_int4_within_bound() {
        let x = rand_kv(1, 4, 8, 128);
        let (codes, scale) = quantize_per_token(&x, 4).unwrap();
        let y = dequantize_per_token(&codes, &scale, 4, DType::F32).unwrap();
        let amax = x.abs().unwrap().max_keepdim(D::Minus1).unwrap();
        let bound = (amax.affine(1.0 / 7.0, 1e-6).unwrap())
            .broadcast_as(x.shape())
            .unwrap();
        let err = (x - y).unwrap().abs().unwrap();
        let over = err
            .gt(&bound)
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<u8>()
            .unwrap();
        assert_eq!(
            over, 0,
            "int4 round-trip error exceeded the per-token scale bound"
        );
    }

    // int4 packing requires an even head_dim (asserted in pack_nibbles);
    // Qwen3's head_dim is 128, always even, so this documents the constraint
    // rather than exercising a real model config.
    #[test]
    #[should_panic(expected = "int4 packing needs even head_dim")]
    fn pack_nibbles_odd_head_dim_panics() {
        let x = rand_kv(1, 1, 1, 3);
        let _ = pack_nibbles(&x);
    }

    #[test]
    fn pack_unpack_nibbles_round_trip() {
        // Values in [0, 15]: valid nibble range.
        let x = Tensor::new(&[[[[1f32, 2., 3., 4., 5., 6., 7., 8.]]]], &Device::Cpu).unwrap();
        let packed = pack_nibbles(&x).unwrap();
        assert_eq!(packed.dims(), &[1, 1, 1, 4]);
        let unpacked = unpack_nibbles(&packed).unwrap();
        assert_eq!(
            unpacked.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1., 2., 3., 4., 5., 6., 7., 8.]
        );
    }

    // Appending 3 tokens then 1 more must reproduce the same first-3-token
    // values as appending all 4 at once — growth/overflow handling must not
    // corrupt already-cached positions.
    #[test]
    fn quant_kv_cache_incremental_append_matches_bulk() {
        let k1 = rand_kv(1, 2, 3, 8);
        let v1 = rand_kv(1, 2, 3, 8);
        let k2 = rand_kv(1, 2, 1, 8);
        let v2 = rand_kv(1, 2, 1, 8);

        let mut incremental = QuantKvCache::new(8);
        incremental.append(&k1, &v1).unwrap();
        let (k_after_first, _) = incremental.append(&k2, &v2).unwrap();

        let mut bulk = QuantKvCache::new(8);
        let k_all = Tensor::cat(&[&k1, &k2], 2).unwrap();
        let v_all = Tensor::cat(&[&v1, &v2], 2).unwrap();
        let (k_bulk, _) = bulk.append(&k_all, &v_all).unwrap();

        let first_three_incremental = k_after_first.narrow(2, 0, 3).unwrap();
        let first_three_bulk = k_bulk.narrow(2, 0, 3).unwrap();
        let diff = (first_three_incremental - first_three_bulk)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-4,
            "incremental append diverged from bulk append: {diff}"
        );
    }

    // Appending many small batches that together cross the ROOM=256 headroom
    // boundary must still match a single bulk append — exercises the
    // grow_append reallocation branch for k_dequant/v_dequant that the
    // single-digit-token tests above never reach.
    #[test]
    fn quant_scratch_buffer_growth_across_room_boundary() {
        let chunk_s = 4;
        let num_chunks = 66; // 66 * 4 = 264 tokens, past the initial 4+ROOM=260 capacity.
        let ks: Vec<Tensor> = (0..num_chunks).map(|_| rand_kv(1, 2, chunk_s, 8)).collect();
        let vs: Vec<Tensor> = (0..num_chunks).map(|_| rand_kv(1, 2, chunk_s, 8)).collect();

        let mut incremental = QuantKvCache::new(8);
        let (mut k_incremental, mut v_incremental) = (None, None);
        for (k, v) in ks.iter().zip(vs.iter()) {
            let (k_full, v_full) = incremental.append(k, v).unwrap();
            k_incremental = Some(k_full);
            v_incremental = Some(v_full);
        }
        let k_incremental = k_incremental.unwrap();
        let v_incremental = v_incremental.unwrap();
        assert_eq!(
            k_incremental.dim(2).unwrap(),
            chunk_s * num_chunks,
            "seq len must reflect all appended chunks after growth"
        );

        let mut bulk = QuantKvCache::new(8);
        let k_refs: Vec<&Tensor> = ks.iter().collect();
        let v_refs: Vec<&Tensor> = vs.iter().collect();
        let k_all = Tensor::cat(&k_refs, 2).unwrap();
        let v_all = Tensor::cat(&v_refs, 2).unwrap();
        let (k_bulk, v_bulk) = bulk.append(&k_all, &v_all).unwrap();

        let diff_k = (&k_incremental - &k_bulk)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let diff_v = (&v_incremental - &v_bulk)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff_k < 1e-4,
            "K diverged after growth across ROOM boundary: {diff_k}"
        );
        assert!(
            diff_v < 1e-4,
            "V diverged after growth across ROOM boundary: {diff_v}"
        );
    }

    // `quantized_append` must store the same codes/scales `append` would
    // (both call `quantize_per_token`), just without maintaining the
    // dequantized scratch buffer.
    #[test]
    fn quantized_append_returns_correct_shapes() {
        for bits in [8u32, 4] {
            let k = rand_kv(1, 2, 3, 8);
            let v = rand_kv(1, 2, 3, 8);
            let mut cache = QuantKvCache::new(bits);
            let kv_ref = cache.quantized_append(&k, &v).unwrap();

            assert_eq!(kv_ref.bits, bits);
            assert_eq!(kv_ref.seq_len, 3);
            let expected_code_dim = if bits == 8 { 8 } else { 4 };
            assert_eq!(kv_ref.k_codes.dims(), &[1, 2, 3, expected_code_dim]);
            assert_eq!(kv_ref.k_scale.dims(), &[1, 2, 3, 1]);
            assert_eq!(kv_ref.v_codes.dims(), &[1, 2, 3, expected_code_dim]);
            assert_eq!(kv_ref.v_scale.dims(), &[1, 2, 3, 1]);
        }
    }

    // `seq_len` on the returned ref, and the cache's own `len()`, must both
    // advance across repeated `quantized_append` calls.
    #[test]
    fn quantized_append_advances_seq_len() {
        let mut cache = QuantKvCache::new(8);
        let kv_ref = cache
            .quantized_append(&rand_kv(1, 2, 3, 8), &rand_kv(1, 2, 3, 8))
            .unwrap();
        assert_eq!(kv_ref.seq_len, 3);
        assert_eq!(cache.len(), 3);

        let kv_ref = cache
            .quantized_append(&rand_kv(1, 2, 2, 8), &rand_kv(1, 2, 2, 8))
            .unwrap();
        assert_eq!(kv_ref.seq_len, 5);
        assert_eq!(cache.len(), 5);
    }

    // After `quantized_append` (which skips the scratch buffer), a
    // subsequent regular `append` must lazily rebuild the scratch from the
    // stored codes/scales — same rebuild path `install` already relies on —
    // and produce the same result as an uninterrupted `append`-only sequence.
    #[test]
    fn quantized_append_then_regular_append_matches_uninterrupted() {
        let k0 = rand_kv(1, 2, 3, 8);
        let v0 = rand_kv(1, 2, 3, 8);
        let k1 = rand_kv(1, 2, 1, 8);
        let v1 = rand_kv(1, 2, 1, 8);

        let mut reference = QuantKvCache::new(8);
        reference.append(&k0, &v0).unwrap();
        let (ref_k, ref_v) = reference.append(&k1, &v1).unwrap();

        let mut cache = QuantKvCache::new(8);
        cache.quantized_append(&k0, &v0).unwrap();
        let (got_k, got_v) = cache.append(&k1, &v1).unwrap();

        assert_eq!(got_k.dims(), ref_k.dims());
        let diff_k = (&got_k - &ref_k)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let diff_v = (&got_v - &ref_v)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff_k < 1e-4,
            "K diverged after quantized_append + append: {diff_k}"
        );
        assert!(
            diff_v < 1e-4,
            "V diverged after quantized_append + append: {diff_v}"
        );
    }

    // `try_quantized_append` must decline (return `None`) for an `Fp` cache
    // and for a CPU device, since the fused kernels are GPU-only and there's
    // nothing to fuse for an unquantized cache.
    #[test]
    fn try_quantized_append_declines_fp_cache() {
        let mut cache = KvCache::new(KvCacheKind::Fp);
        let got = cache
            .try_quantized_append(&rand_kv(1, 2, 3, 8), &rand_kv(1, 2, 3, 8))
            .unwrap();
        assert!(got.is_none());
    }

    // See [`try_quantized_append_declines_fp_cache`]: CPU device case.
    #[test]
    fn try_quantized_append_declines_cpu_device() {
        let mut cache = KvCache::new(KvCacheKind::Int8);
        let got = cache
            .try_quantized_append(&rand_kv(1, 2, 3, 8), &rand_kv(1, 2, 3, 8))
            .unwrap();
        assert!(got.is_none(), "CPU tensors must not take the fused path");
    }

    #[test]
    fn reset_clears_state() {
        let mut cache = QuantKvCache::new(8);
        cache
            .append(&rand_kv(1, 2, 3, 8), &rand_kv(1, 2, 3, 8))
            .unwrap();
        assert!(!cache.is_empty());
        assert!(cache.byte_size() > 0);

        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.byte_size(), 0);
        assert!(cache.extract().unwrap().is_none());
    }

    #[test]
    fn fp_extract_install_round_trip() {
        let k0 = rand_kv(1, 2, 5, 8);
        let v0 = rand_kv(1, 2, 5, 8);
        let mut a = FpKvCache::new();
        a.append(&k0, &v0).unwrap();
        let state = a.extract().unwrap().expect("non-empty cache");
        assert_eq!(state.seq_len().unwrap(), 5);

        let mut b = FpKvCache::new();
        b.install(state).unwrap();
        assert_eq!(b.len(), 5);

        // Installed content must match what was appended, not just its shape.
        let KvCacheState::Fp(k, v) = b.extract().unwrap().expect("non-empty cache") else {
            panic!("expected an Fp state");
        };
        assert_eq!(
            k.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            k0.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            v.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            v0.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        // Appending after install must extend from position 5, not overwrite.
        let (k_full, _) = b
            .append(&rand_kv(1, 2, 1, 8), &rand_kv(1, 2, 1, 8))
            .unwrap();
        assert_eq!(k_full.dim(2).unwrap(), 6);
    }

    #[test]
    fn quant_extract_install_round_trip() {
        let mut a = QuantKvCache::new(8);
        a.append(&rand_kv(1, 2, 5, 8), &rand_kv(1, 2, 5, 8))
            .unwrap();
        let state = a.extract().unwrap().expect("non-empty cache");
        assert_eq!(state.seq_len().unwrap(), 5);
        assert_eq!(state.kind(), KvCacheKind::Int8);

        let mut b = QuantKvCache::new(8);
        b.install(state.clone()).unwrap();
        assert_eq!(b.len(), 5);

        // Installed codes/scales must match what was extracted exactly (no
        // requantization on install), not just match in shape.
        let installed = b.extract().unwrap().expect("non-empty cache");
        for (orig, got) in state.buffers().into_iter().zip(installed.buffers()) {
            let orig = orig
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let got = got
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(orig, got);
        }
    }

    // Appending after install must exercise the lazy scratch-buffer seed
    // (dequantized-from-install path) and still match an uninterrupted
    // append sequence over the same tokens.
    #[test]
    fn quant_install_then_append_matches_uninterrupted() {
        let k0 = rand_kv(1, 2, 3, 8);
        let v0 = rand_kv(1, 2, 3, 8);
        let k1 = rand_kv(1, 2, 1, 8);
        let v1 = rand_kv(1, 2, 1, 8);

        let mut reference = QuantKvCache::new(8);
        reference.append(&k0, &v0).unwrap();
        let (ref_k, ref_v) = reference.append(&k1, &v1).unwrap();

        let mut a = QuantKvCache::new(8);
        a.append(&k0, &v0).unwrap();
        let state = a.extract().unwrap().expect("non-empty cache");

        let mut b = QuantKvCache::new(8);
        b.install(state).unwrap();
        assert_eq!(b.len(), 3);
        let (got_k, got_v) = b.append(&k1, &v1).unwrap();

        assert_eq!(got_k.dims(), ref_k.dims());
        assert_eq!(got_v.dims(), ref_v.dims());
        let diff_k = (got_k - ref_k)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let diff_v = (got_v - ref_v)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff_k < 1e-4, "K diverged after install+append: {diff_k}");
        assert!(diff_v < 1e-4, "V diverged after install+append: {diff_v}");
    }

    #[test]
    fn install_rejects_mismatched_representation() {
        let fp_state = {
            let mut c = FpKvCache::new();
            c.append(&rand_kv(1, 2, 2, 8), &rand_kv(1, 2, 2, 8))
                .unwrap();
            c.extract().unwrap().unwrap()
        };
        let mut quant = QuantKvCache::new(8);
        assert!(quant.install(fp_state).is_err());

        let quant_state = {
            let mut c = QuantKvCache::new(4);
            c.append(&rand_kv(1, 2, 2, 8), &rand_kv(1, 2, 2, 8))
                .unwrap();
            c.extract().unwrap().unwrap()
        };
        let mut fp = FpKvCache::new();
        assert!(fp.install(quant_state.clone()).is_err());

        // Bit-width mismatch (state is 4-bit, cache expects 8-bit).
        let mut wrong_bits = QuantKvCache::new(8);
        assert!(wrong_bits.install(quant_state).is_err());
    }

    #[test]
    fn kv_cache_state_buffers_round_trip() {
        let fp = KvCacheState::Fp(rand_kv(1, 2, 3, 8), rand_kv(1, 2, 3, 8));
        let rebuilt =
            KvCacheState::from_buffers(fp.buffers().into_iter().cloned().collect(), fp.kind())
                .unwrap();
        assert_eq!(rebuilt.seq_len().unwrap(), 3);

        let quant = KvCacheState::Quant {
            k_codes: rand_kv(1, 2, 3, 8),
            k_scale: rand_kv(1, 2, 3, 1),
            v_codes: rand_kv(1, 2, 3, 8),
            v_scale: rand_kv(1, 2, 3, 1),
            bits: 8,
        };
        let rebuilt = KvCacheState::from_buffers(
            quant.buffers().into_iter().cloned().collect(),
            quant.kind(),
        )
        .unwrap();
        assert_eq!(rebuilt.seq_len().unwrap(), 3);
    }

    #[test]
    fn kv_cache_state_byte_size_matches_manual_sum() {
        let fp = KvCacheState::Fp(rand_kv(1, 2, 3, 8), rand_kv(1, 2, 3, 8));
        // F32: 1*2*3*8 * 4 bytes, times 2 tensors.
        assert_eq!(fp.byte_size(), (1 * 2 * 3 * 8 * 4 * 2) as u64);
    }

    // `to_fp_pair` on a `Quant` state must match dequantize_per_token exactly
    // (it's a thin wrapper, not a separate math path).
    #[test]
    fn to_fp_pair_dequantizes_quant_state() {
        let k = rand_kv(1, 2, 4, 8);
        let v = rand_kv(1, 2, 4, 8);
        let (k_codes, k_scale) = quantize_per_token(&k, 8).unwrap();
        let (v_codes, v_scale) = quantize_per_token(&v, 8).unwrap();
        let state = KvCacheState::Quant {
            k_codes: k_codes.clone(),
            k_scale: k_scale.clone(),
            v_codes: v_codes.clone(),
            v_scale: v_scale.clone(),
            bits: 8,
        };

        let (k_out, v_out) = state.to_fp_pair(DType::F32).unwrap();
        let k_expected = dequantize_per_token(&k_codes, &k_scale, 8, DType::F32).unwrap();
        let v_expected = dequantize_per_token(&v_codes, &v_scale, 8, DType::F32).unwrap();

        assert_eq!(
            k_out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            k_expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            v_out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            v_expected.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    // `to_fp_pair` on an `Fp` state must be a lossless passthrough (no
    // quantization involved).
    #[test]
    fn to_fp_pair_is_identity_for_fp_state() {
        let k = rand_kv(1, 2, 4, 8);
        let v = rand_kv(1, 2, 4, 8);
        let state = KvCacheState::from_fp_pair(&k, &v, KvCacheKind::Fp).unwrap();
        let (k_out, v_out) = state.to_fp_pair(DType::F32).unwrap();
        assert_eq!(
            k_out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            k.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(
            v_out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            v.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    // Round-tripping a plain (K, V) pair through from_fp_pair -> to_fp_pair
    // at Int8 must stay within the same per-token scale bound as direct
    // quantize/dequantize (it's the same math, just wrapped).
    #[test]
    fn from_fp_pair_round_trips_within_quantization_bound() {
        let k = rand_kv(1, 2, 4, 8);
        let v = rand_kv(1, 2, 4, 8);
        let state = KvCacheState::from_fp_pair(&k, &v, KvCacheKind::Int8).unwrap();
        assert_eq!(state.kind(), KvCacheKind::Int8);

        let (k_out, v_out) = state.to_fp_pair(DType::F32).unwrap();

        for (orig, out) in [(&k, &k_out), (&v, &v_out)] {
            let amax = orig.abs().unwrap().max_keepdim(D::Minus1).unwrap();
            let bound = amax
                .affine(1.0 / 127.0, 1e-6)
                .unwrap()
                .broadcast_as(orig.shape())
                .unwrap();
            let err = (orig - out).unwrap().abs().unwrap();
            let over = err
                .gt(&bound)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<u8>()
                .unwrap();
            assert_eq!(
                over, 0,
                "from_fp_pair/to_fp_pair round-trip exceeded the per-token scale bound"
            );
        }
    }
}
