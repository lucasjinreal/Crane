//! Shared utilities: `repeat_kv`, `repeat_penalty`, causal mask.

use candle_core::{DType, Device, Result, Tensor};

/// Build a causal attention mask of shape `(seq_len, kv_len)` where
/// `kv_len = index_pos + seq_len`.
///
/// `mask[i][j] = 1` means query `i` must **not** attend to key `j`.
///
/// - `index_pos == 0`: classic square `(seq_len, seq_len)` mask.
/// - `index_pos > 0`: rectangular mask for prefix KV caching — the first
///   `index_pos` columns are all-zero (every query attends to all cached prefix
///   keys) and the last `seq_len` columns form the standard causal triangle.
///
/// All models that maintain a KV cache should use this function so that
/// batched user-turn prefill works correctly after prefix restoration.
///
/// # Errors
///
/// Returns an error if allocating the mask tensor on `device` fails.
pub fn build_causal_mask(seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
    let kv_len = index_pos + seq_len;
    let mask: Vec<u8> = (0..seq_len)
        .flat_map(|i| (0..kv_len).map(move |j| u8::from(j > index_pos + i)))
        .collect();
    Tensor::from_slice(&mask, (seq_len, kv_len), device)
}

/// Build an additive causal attention mask of shape `[1, 1, q_len, kv_len]`.
///
/// Position `(i, j)` is `0.0` when `j <= i + kv_offset` (query `i` may attend
/// to key `j`) and `f32::NEG_INFINITY` otherwise. The mask is built in F32 and
/// cast to `dtype`; both sentinels are exactly representable in every float
/// format, so this cast is lossless.
///
/// - Square, no-offset case (prefill from scratch): `q_len == kv_len`,
///   `kv_offset == 0`.
/// - Continuation prefill against an existing KV cache:
///   `kv_offset = kv_len - q_len`.
///
/// Allocates and fills a `q_len * kv_len` buffer on every call. A caller
/// sharing the same mask across multiple layers in one forward pass should
/// build it once and reuse it rather than calling this per layer.
///
/// # Errors
///
/// Returns an error if allocating the mask tensor on `device` or casting to
/// `dtype` fails.
pub fn build_additive_causal_mask(
    q_len: usize,
    kv_len: usize,
    kv_offset: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut data = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        for j in 0..kv_len {
            if j > i + kv_offset {
                data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, q_len, kv_len), device)?.to_dtype(dtype)
}

/// Pre-built additive causal mask encoding exactly the shifted-diagonal
/// pattern `j <= i + kv_offset`. Only constructible through [`CausalMask::new`],
/// which enforces that pattern at the type level — a caller can't pass a
/// padding or sliding-window mask to a causal-only call site by accident,
/// the exact mistake a plain `&Tensor` mask parameter invites.
#[derive(Clone)]
pub struct CausalMask(Tensor);

impl CausalMask {
    /// Builds the mask for `(q_len, kv_len, kv_offset)`: position `(i, j)` is
    /// `0` when `j <= i + kv_offset` and `f32::NEG_INFINITY` otherwise, cast
    /// to `dtype`. See [`build_additive_causal_mask`] for the exact pattern.
    ///
    /// # Errors
    ///
    /// Returns a candle error if the underlying tensor construction fails.
    pub fn new(
        q_len: usize,
        kv_len: usize,
        kv_offset: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        build_additive_causal_mask(q_len, kv_len, kv_offset, dtype, device).map(Self)
    }

    /// Borrows the inner additive mask tensor. `pub`, not `pub(crate)`: the
    /// invariant `CausalMask` protects is on construction (the mask's
    /// pattern), not on reading an already-built one.
    #[must_use]
    pub fn as_tensor(&self) -> &Tensor {
        &self.0
    }
}

/// # Errors
///
/// Returns an error if `logits` can't be converted to a flat `f32` vector
/// (e.g. unexpected dtype or shape), or if rebuilding the output tensor fails.
pub fn apply_repeat_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> Result<Tensor> {
    let device = logits.device();
    let mut logits = logits.to_dtype(candle_core::DType::F32)?.to_vec1::<f32>()?;
    let mut already_seen = std::collections::HashSet::new();
    for token_id in context {
        if already_seen.contains(token_id) {
            continue;
        }
        already_seen.insert(token_id);
        if let Some(logit) = logits.get_mut(*token_id as usize) {
            if *logit >= 0. {
                *logit /= penalty
            } else {
                *logit *= penalty
            }
        }
    }
    let logits_len = logits.len();
    Tensor::from_vec(logits, logits_len, device)
}

/// Repeats a key or value tensor for grouped query attention
/// The input tensor should have a shape `(batch, num_kv_heads, seq_len, head_dim)`,
///
/// # Errors
///
/// Returns an error if `xs` doesn't have exactly 4 dimensions, or if
/// concatenating and reshaping it fails.
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        // Using cat is faster than a broadcast as it avoids going through a potentially
        // strided copy.
        // https://github.com/huggingface/candle/pull/2043
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

/// Drains candle's Metal staging-buffer pool; call periodically while
/// loading to avoid quadratic allocation cost on large checkpoints.
/// No-op on non-Metal devices.
pub fn release_load_staging(device: &Device) {
    if device.is_metal() {
        let _ = device.synchronize();
    }
}

/// This process's physical memory footprint in bytes (macOS's
/// `ri_phys_footprint`), excluding reclaimable file-backed pages like a
/// mmaped checkpoint. Returns `None` off macOS or on syscall failure.
pub fn phys_footprint_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        // struct rusage_info_v2 from <sys/resource.h>, truncated to the
        // fields we need plus padding for the kernel to fill.
        #[repr(C)]
        #[derive(Default)]
        struct RUsageInfoV2 {
            ri_uuid: [u8; 16],
            ri_user_time: u64,
            ri_system_time: u64,
            ri_pkg_idle_wkups: u64,
            ri_interrupt_wkups: u64,
            ri_pageins: u64,
            ri_wired_size: u64,
            ri_resident_size: u64,
            ri_phys_footprint: u64,
            ri_proc_start_abstime: u64,
            ri_proc_exit_abstime: u64,
            ri_child_user_time: u64,
            ri_child_system_time: u64,
            ri_child_pkg_idle_wkups: u64,
            ri_child_interrupt_wkups: u64,
            ri_child_pageins: u64,
            ri_child_elapsed_abstime: u64,
            ri_diskio_bytesread: u64,
            ri_diskio_byteswritten: u64,
        }

        unsafe extern "C" {
            fn proc_pid_rusage(pid: i32, flavor: i32, buffer: *mut core::ffi::c_void) -> i32;
        }

        const RUSAGE_INFO_V2: i32 = 2;
        let mut info = RUsageInfoV2::default();
        let pid = std::process::id() as i32;
        // SAFETY: `info` matches the RUSAGE_INFO_V2 flavor passed below.
        let rc = unsafe {
            proc_pid_rusage(
                pid,
                RUSAGE_INFO_V2,
                std::ptr::from_mut(&mut info).cast::<core::ffi::c_void>(),
            )
        };
        (rc == 0).then_some(info.ri_phys_footprint)
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}
