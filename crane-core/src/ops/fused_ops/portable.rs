//! Device-independent implementations of the fused ops.
//!
//! Used directly when neither `cuda` nor `rocm` is enabled, and as the
//! non-GPU-device branch of [`super::rocm_impl`] — a `rocm` build still runs on
//! CPU when no AMD device is present, so its entry points cannot assume one.

use candle_core::{Device, Result, Tensor};

/// Argmax over a flattened logits tensor.
///
/// On a GPU device this is a device-side reduction plus a 4-byte DtoH, not a
/// full-vocab copy, so there is nothing for a custom kernel to save here.
///
/// # Errors
///
/// Returns an error if a tensor operation fails.
pub fn gpu_argmax(logits: &Tensor) -> Result<u32> {
    let logits = logits.flatten_all()?;
    logits.argmax(0)?.to_scalar::<u32>()
}

/// Order two `(value, index)` candidates: value descending, index ascending.
///
/// A *total* order, which is what makes top-k reproducible: with ties broken
/// only by value, `select_nth_unstable_by` may return any of the equal
/// candidates, so two runs — or the host and the GPU kernel — can legitimately
/// disagree. The GPU kernel encodes exactly this order in its sort key.
fn topk_order(a: &(f32, u32), b: &(f32, u32)) -> std::cmp::Ordering {
    b.0.partial_cmp(&a.0)
        .unwrap_or(std::cmp::Ordering::Greater)
        .then(a.1.cmp(&b.1))
}

/// Indices of the `k` largest values of a 1-D tensor.
///
/// Ordered by value descending and, where values are equal, by index
/// ascending.
///
/// Copies the whole vector to the host. That is ~1 MB per call at a 250 K
/// vocabulary, which is why the GPU builds override this.
///
/// # Errors
///
/// Returns an error if `logits` is not 1-D, if `k` is zero or larger than the
/// vector, or if a tensor operation fails.
pub fn topk_indices(logits: &Tensor, k: usize) -> Result<Tensor> {
    if logits.rank() != 1 {
        candle_core::bail!("topk_indices expects a 1D tensor");
    }
    let n = logits.dims1()?;
    if k == 0 || k > n {
        candle_core::bail!("topk_indices: invalid k");
    }
    let vals = logits.to_vec1::<f32>()?;
    let mut pairs: Vec<(f32, u32)> = vals
        .into_iter()
        .enumerate()
        .map(|(i, v)| (v, i as u32))
        .collect();
    let kth = k.saturating_sub(1);
    pairs.select_nth_unstable_by(kth, topk_order);
    pairs.truncate(k);
    pairs.sort_by(topk_order);
    let out: Vec<u32> = pairs.into_iter().map(|(_, i)| i).collect();
    Tensor::new(out.as_slice(), logits.device())
}

/// Upload a u32 slice, returning a new 1-D U32 tensor on `device`.
///
/// # Errors
///
/// Returns an error if allocation on `device` fails.
pub fn copy_from_slice_u32(src: &[u32], device: &Device) -> Result<Tensor> {
    Tensor::new(src, device)
}

/// Contiguous copy of an f32 tensor, on the device it already lives on.
///
/// # Errors
///
/// Returns an error if the copy fails.
pub fn copy_from_tensor_f32(src: &Tensor) -> Result<Tensor> {
    src.contiguous()
}
