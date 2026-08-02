//! ROCm/HIP implementations of the fused ops.
//!
//! The kernels are the same `kernels/cuda/fused_ops.cu` the CUDA build compiles to
//! PTX; here candle runs it through `hipcc` on first use and caches the code
//! object on disk (see [`crate::ops::rocm`]).
//!
//! Every entry point falls back to [`super::portable`] (or to the op chain it
//! replaces) off a ROCm device: a `rocm` build still runs on CPU when no AMD
//! GPU is present.

use candle_core::{DType, Result, Tensor};

use crate::ops::rocm::{self, arg};

use super::portable;

pub use portable::{copy_from_slice_u32, copy_from_tensor_f32, gpu_argmax};

const MODULE_NAME: &str = "crane_fused_ops";
const SOURCE: &str = include_str!("../../../kernels/cuda/fused_ops.cu");

const TOPK_MODULE: &str = "crane_topk";
const TOPK_SOURCE: &str = include_str!("../../../kernels/cuda/topk.cu");

/// Threads per top-k block. Must equal `TOPK_THREADS` in `kernels/cuda/topk.cu`:
/// the kernel sizes its flush trigger from it.
const TOPK_BLOCK: u32 = 256;

/// Elements one block consumes between two flush checks — `TOPK_THREADS *
/// TOPK_UNROLL` in the kernel.
const TOPK_STEP: usize = 256 * 4;

/// Largest `k` the kernel accepts — `TOPK_MAX_K` in the kernel. Above it a
/// block's staging area could not absorb an iteration's pushes, so the
/// launcher routes to the host sort instead of truncating.
const MAX_TOPK: usize = 2048 / 4;

/// Whether `CRANE_TOPK_HOST=1` asks for the host sort instead of the kernel.
///
/// An A/B switch, in the same family as the sampler's `CRANE_FORCE_GPU_TOPK`.
/// Read once: this sits on the per-token decode path.
fn host_topk_forced() -> bool {
    static FORCED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| std::env::var("CRANE_TOPK_HOST").as_deref() == Ok("1"))
}

/// Blocks in the stage-1 grid, and the span each of them scans.
///
/// The grid is sized first and the span derived from it: sizing the span first
/// and clamping the grid afterwards would leave the tail of the row unvisited,
/// because a block only reads `blockIdx.x * items_per_block ..`.
///
/// Four steps per block is the balance point on a 60-CU RDNA3 part — it puts
/// roughly one block on each CU at a 250 K vocabulary while keeping enough
/// elements per block that the first flush pays for itself.
fn topk_geometry(n: usize) -> (usize, usize) {
    let grid = n.div_ceil(TOPK_STEP * 4).clamp(1, 256);
    (grid, n.div_ceil(grid))
}

/// Indices of the `k` largest values of a 1-D f32 tensor.
///
/// Ordered by value descending and, where values are equal, by index ascending
/// — a total order, so the result is exact and reproducible even with ties.
///
/// Two-stage block reduction on the device. The result is a `[k]` U32 tensor,
/// so the sampler's host transfer drops from the whole vocabulary (~1 MB at
/// 250 K tokens, on every decoded token) to `4 * k` bytes.
///
/// `k` above [`MAX_TOPK`] falls back to the host sort rather than truncating;
/// so does a non-ROCm device, and so does `CRANE_TOPK_HOST=1`.
///
/// # Errors
///
/// Returns an error if `logits` is not a contiguous 1-D f32 tensor, if `k` is
/// zero or larger than the vector, or if the launch fails.
pub fn topk_indices(logits: &Tensor, k: usize) -> Result<Tensor> {
    if !logits.device().is_rocm() || k > MAX_TOPK || host_topk_forced() {
        return portable::topk_indices(logits, k);
    }
    if logits.rank() != 1 {
        candle_core::bail!("topk_indices expects a 1D tensor");
    }
    if !logits.is_contiguous() {
        candle_core::bail!("topk_indices requires contiguous input");
    }
    let n = logits.dims1()?;
    if k == 0 || k > n {
        candle_core::bail!("topk_indices expects 0 < k <= n");
    }

    let dev = logits.device().as_rocm_device()?.clone();
    let (grid, items_per_block) = topk_geometry(n);

    #[allow(clippy::cast_possible_truncation)]
    let (n_u32, k_u32, grid_u32, items_u32) =
        (n as u32, k as u32, grid as u32, items_per_block as u32);

    // Stage 1 hands stage 2 packed sort keys, not (value, index) pairs: the
    // ordering is entirely inside the key, so the merge never looks at a float.
    let tmp_keys = dev.alloc::<u64>(grid * k)?;
    let out_idx = dev.alloc::<u32>(k)?;

    let (storage, layout) = logits.storage_and_layout();
    let x_ptr = rocm::device_ptr(&storage, layout, DType::F32, "topk_indices logits")?;

    let tmp_keys_ptr = tmp_keys.as_ptr();
    let out_ptr = out_idx.as_ptr();
    let m = grid_u32 * k_u32;

    // SAFETY: the argument lists match `topk_stage1_f32` / `topk_stage2_u64` in
    // `kernels/cuda/topk.cu`; both kernels bound their own writes by `n`/`m`, both
    // use only static shared memory, and every `arg` operand outlives the
    // launches.
    unsafe {
        rocm::launch(
            &dev,
            TOPK_MODULE,
            "topk_stage1_f32",
            TOPK_SOURCE,
            grid_u32,
            TOPK_BLOCK,
            0,
            &mut [
                arg(&x_ptr),
                arg(&n_u32),
                arg(&k_u32),
                arg(&items_u32),
                arg(&tmp_keys_ptr),
            ],
        )?;
        rocm::launch(
            &dev,
            TOPK_MODULE,
            "topk_stage2_u64",
            TOPK_SOURCE,
            1,
            TOPK_BLOCK,
            0,
            &mut [arg(&tmp_keys_ptr), arg(&m), arg(&k_u32), arg(&out_ptr)],
        )?;
    }

    // The scratch buffers are freed here, i.e. after the launches are queued
    // but before they have necessarily run. That is safe: the allocator is
    // ordered against this device's stream, so a freed block is only handed out
    // again once the work already queued on that stream has retired.
    Ok(rocm::wrap_u32(out_idx, &dev, k))
}

/// Fused `silu(gate) * up` over a `[..., 2 * intermediate_size]` tensor.
///
/// Replaces the `narrow + silu + mul` chain (3 launches, 2 intermediates) with
/// one launch. Falls back to that chain off a ROCm device.
///
/// # Errors
///
/// Returns an error if the last dimension is not `2 * intermediate_size`, if
/// `gate_up` is not contiguous, if its dtype is not f32/f16/bf16, or if the
/// launch fails.
pub fn fused_silu_mul(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    let dims = gate_up.dims();
    let last = *dims
        .last()
        .ok_or_else(|| candle_core::Error::Msg("fused_silu_mul: scalar input".into()))?;
    if last != 2 * intermediate_size {
        candle_core::bail!(
            "fused_silu_mul: last dim {last} != 2*intermediate_size {}",
            2 * intermediate_size
        );
    }
    if !gate_up.device().is_rocm() {
        return silu_mul_ops(gate_up, intermediate_size);
    }
    if !gate_up.is_contiguous() {
        candle_core::bail!("fused_silu_mul: input must be contiguous");
    }

    let dtype = gate_up.dtype();
    let kernel = match dtype {
        DType::BF16 => "fused_silu_mul_bf16",
        DType::F16 => "fused_silu_mul_f16",
        DType::F32 => "fused_silu_mul_f32",
        dt => candle_core::bail!("fused_silu_mul: unsupported dtype {dt:?}"),
    };

    let dev = gate_up.device().as_rocm_device()?.clone();
    let n_rows = gate_up.elem_count() / last;
    let out_el = n_rows * intermediate_size;

    let (storage, layout) = gate_up.storage_and_layout();
    let src_ptr = rocm::device_ptr(&storage, layout, dtype, "fused_silu_mul input")?;

    #[allow(clippy::cast_possible_truncation)]
    let (grid, block) = (n_rows as u32, intermediate_size.clamp(32, 256) as u32);
    let size_i32 = i32::try_from(intermediate_size).map_err(|_| {
        candle_core::Error::Msg("fused_silu_mul: intermediate_size overflows i32".into())
    })?;

    let mut out_dims = dims.to_vec();
    if let Some(dim) = out_dims.last_mut() {
        *dim = intermediate_size;
    }

    // The buffer's element type has to match the dtype the kernel writes, so
    // the allocation, the launch and the wrap are one arm per dtype.
    macro_rules! run {
        ($ty:ty, $wrap:path) => {{
            let dst = dev.alloc::<$ty>(out_el)?;
            let dst_ptr = dst.as_ptr();
            // SAFETY: matches the kernel's `(const T *, T *, int)` signature;
            // the grid is one block per row and the inner loop is strided, so
            // every output element is written exactly once.
            unsafe {
                rocm::launch(
                    &dev,
                    MODULE_NAME,
                    kernel,
                    SOURCE,
                    grid,
                    block,
                    0,
                    &mut [arg(&src_ptr), arg(&dst_ptr), arg(&size_i32)],
                )?;
            }
            $wrap(dst, &dev, out_dims)
        }};
    }

    Ok(match dtype {
        DType::BF16 => run!(half::bf16, rocm::wrap_bf16),
        DType::F16 => run!(half::f16, rocm::wrap_f16),
        DType::F32 => run!(f32, rocm::wrap_f32),
        dt => candle_core::bail!("fused_silu_mul: unsupported dtype {dt:?}"),
    })
}

/// The op chain [`fused_silu_mul`] replaces, for devices without the kernel.
fn silu_mul_ops(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    use candle_core::D;
    let gate = gate_up.narrow(D::Minus1, 0, intermediate_size)?;
    let up = gate_up.narrow(D::Minus1, intermediate_size, intermediate_size)?;
    candle_nn::ops::silu(&gate)? * up
}
