//! CUDA launcher for the exact GPU top-k in `crane-core/kernels/cuda/topk.cu`.
//!
//! Its own module, mirroring the kernel's own translation unit: top-k shares
//! nothing with the fused elementwise kernels, and the whole thing — kernel
//! plus this launcher plus [`super::rocm_impl`] — is meant to move into candle
//! as a standalone op. The launch geometry is derived here from the row length
//! and `k` alone, so nothing Crane-specific leaks into it.

use candle_core::cuda_backend::CudaStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::{Result, Shape, Tensor};

use super::cuda_impl::ptx;

const TOPK_MODULE: &str = "crane_topk";

/// Threads per top-k block — `TOPK_THREADS` in `kernels/cuda/topk.cu`.
const TOPK_BLOCK: u32 = 256;

/// Elements one block consumes between two flush checks — `TOPK_THREADS *
/// TOPK_UNROLL` in the kernel.
const TOPK_STEP: usize = 256 * 4;

/// Largest `k` the kernel accepts — `TOPK_MAX_K` in the kernel. Above it a
/// block's staging area could not absorb an iteration's pushes, so the launcher
/// routes to the host sort instead of truncating.
const MAX_TOPK: usize = 2048 / 4;

/// Scratch for one `(device, k)`, kept across steps to avoid a malloc per token.
struct TopkTmpBufs {
    keys: candle_core::cuda_backend::cudarc::driver::CudaSlice<u64>,
    cap_elems: usize,
}

thread_local! {
    static TOPK_TMP: std::cell::RefCell<
        std::collections::HashMap<(candle_core::cuda_backend::DeviceId, usize), TopkTmpBufs>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// GPU top-k indices for 1-D f32 tensors.
///
/// Ordered by value descending and, where values are equal, by index ascending
/// — a total order, so the result is exact and reproducible even with ties.
///
/// Two-stage block reduction. `k` above [`MAX_TOPK`] falls back to the host
/// sort rather than truncating.
///
/// Returns a `[k]` U32 tensor of the indices of the k largest values.
///
/// # Errors
///
/// Returns an error if `logits` is not a contiguous 1-D f32 tensor, if `k` is
/// zero or larger than the vector, or if the launch fails.
pub fn topk_indices(logits: &Tensor, k: usize) -> Result<Tensor> {
    if k > MAX_TOPK {
        return super::portable::topk_indices(logits, k);
    }
    if !logits.is_contiguous() {
        candle_core::bail!("topk_indices requires contiguous input");
    }
    if logits.rank() != 1 {
        candle_core::bail!("topk_indices expects a 1D tensor");
    }
    let n = logits.dims1()?;
    if k == 0 || k > n {
        candle_core::bail!("topk_indices expects 0 < k <= n");
    }
    logits.apply_op1_no_bwd(&TopKIndicesOp { k })
}

struct TopKIndicesOp {
    k: usize,
}

impl candle_core::CustomOp1 for TopKIndicesOp {
    fn name(&self) -> &'static str {
        "topk_indices"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            candle_core::bail!("topk_indices requires contiguous layout");
        }
        let k = self.k;
        let n = layout.shape().elem_count();
        let start = layout.start_offset();
        let end = start + n;

        let mut pairs: Vec<(f32, u32)> = match storage {
            candle_core::CpuStorage::F32(vs) => vs[start..end]
                .iter()
                .enumerate()
                .map(|(i, &v)| (v, i as u32))
                .collect(),
            _ => candle_core::bail!("topk_indices only supports f32"),
        };

        // Same total order the kernel's sort key encodes: value descending,
        // then index ascending, so ties cannot make the two paths disagree.
        let order = |a: &(f32, u32), b: &(f32, u32)| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Greater)
                .then(a.1.cmp(&b.1))
        };
        let kth = k.saturating_sub(1);
        pairs.select_nth_unstable_by(kth, order);
        pairs.truncate(k);
        pairs.sort_by(order);

        let out: Vec<u32> = pairs.into_iter().map(|(_, i)| i).collect();
        Ok((candle_core::CpuStorage::U32(out), Shape::from_dims(&[k])))
    }

    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &candle_core::Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::{CudaDType, WrapErr};

        if !layout.is_contiguous() {
            candle_core::bail!("topk_indices requires contiguous layout");
        }
        let k = self.k;
        let k_u32 = k as u32;
        let n = layout.shape().elem_count();
        let n_u32 = n as u32;
        let dev = &storage.device;

        let x = storage.as_cuda_slice::<f32>()?;
        let (o1, o2) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle_core::Error::Msg("topk: need contiguous offsets".into()))?;
        let x = x.slice(o1..o2);

        // Grid first, then the span each block scans: sizing the span first and
        // clamping the grid afterwards would leave the tail of the row
        // unvisited. Both kernels use only static shared memory.
        let grid = n.div_ceil(TOPK_STEP * 4).clamp(1, 256);
        let grid_dim = grid as u32;
        let items_per_block = n.div_ceil(grid) as u32;

        let cap_elems = grid * k;
        let dev_id = dev.id();
        let tmp_keys = TOPK_TMP.with(|cell| -> Result<_> {
            let mut map = cell.borrow_mut();
            match map.get_mut(&(dev_id, k)) {
                Some(bufs) if bufs.cap_elems >= cap_elems => Ok(bufs.keys.clone()),
                _ => {
                    let keys = unsafe { dev.alloc::<u64>(cap_elems)? };
                    map.insert(
                        (dev_id, k),
                        TopkTmpBufs {
                            keys: keys.clone(),
                            cap_elems,
                        },
                    );
                    Ok(keys)
                }
            }
        })?;

        let out_idx = unsafe { dev.alloc::<u32>(k)? };

        // Stage 1 hands stage 2 packed sort keys, not (value, index) pairs: the
        // ordering is entirely inside the key, so the merge never sees a float.
        let f1 = dev.get_or_load_custom_func("topk_stage1_f32", TOPK_MODULE, ptx::TOPK)?;
        {
            let mut builder = f1.builder();
            builder.arg(&x);
            builder.arg(&n_u32);
            builder.arg(&k_u32);
            builder.arg(&items_per_block);
            builder.arg(&tmp_keys);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (grid_dim, 1, 1),
                    block_dim: (TOPK_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .w()?;
        }

        let m = grid_dim * k_u32;
        let f2 = dev.get_or_load_custom_func("topk_stage2_u64", TOPK_MODULE, ptx::TOPK)?;
        {
            let mut builder = f2.builder();
            builder.arg(&tmp_keys);
            builder.arg(&m);
            builder.arg(&k_u32);
            builder.arg(&out_idx);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (TOPK_BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .w()?;
        }

        let dst = CudaStorage::wrap_cuda_slice(out_idx, dev.clone());
        Ok((dst, Shape::from_dims(&[k])))
    }
}
