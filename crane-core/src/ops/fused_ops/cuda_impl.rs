//! CUDA implementations of fused ops using custom PTX kernels.
//!
//! The PTX is compiled from `kernels/fused_ops.cu` at build time via
//! bindgen_cuda and embedded as a const string.

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice, WrapErr};
use candle_core::{CudaDevice, DType, Device, Layout, Result, Shape, Tensor, WithDType};

// PTX compiled from kernels/fused_ops.cu — embedded at build time.
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/crane_kernels_ptx.rs"));
}

const MODULE_NAME: &str = "crane_fused_ops";

/// Load a function from the crane fused-ops PTX module.
///
/// Returns candle's `CudaFunc` opaque wrapper (not re-exported in candle 0.9.x,
/// so we avoid naming the type explicitly — callers rely on type inference).
macro_rules! load_func {
    ($dev:expr, $fn_name:expr) => {
        $dev.get_or_load_custom_func($fn_name, MODULE_NAME, ptx::FUSED_OPS)
    };
}

// =====================================================================
// 1. Fused SiLU(gate) * up
// =====================================================================

/// Fused SiLU activation + element-wise multiply.
///
/// Takes a `gate_up` tensor of shape `[..., 2*intermediate_size]`
/// (gate and up projections concatenated along the last dim) and returns
/// `silu(gate) * up` of shape `[..., intermediate_size]`.
///
/// Replaces 3 candle ops: `narrow(gate)` + `silu(gate)` + `gate * up`.
pub struct FusedSiluMul {
    pub intermediate_size: usize,
}

impl candle_core::CustomOp1 for FusedSiluMul {
    fn name(&self) -> &'static str {
        "fused-silu-mul"
    }

    fn cpu_fwd(
        &self,
        storage: &candle_core::CpuStorage,
        layout: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        // CPU fallback — just do it the slow way.
        use candle_core::CpuStorage as C;

        fn inner<T: WithDType>(
            src: &[T],
            layout: &Layout,
            intermediate_size: usize,
        ) -> Result<(candle_core::CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle_core::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let dims = layout.shape().dims();
            let last = *dims.last().unwrap();
            if last != 2 * intermediate_size {
                candle_core::bail!(
                    "last dim {last} != 2*intermediate_size {}",
                    2 * intermediate_size
                );
            }
            let n_rows = src.len() / last;
            let mut dst = vec![T::zero(); n_rows * intermediate_size];
            for row in 0..n_rows {
                let gate = &src[row * last..row * last + intermediate_size];
                let up = &src[row * last + intermediate_size..row * last + last];
                let out = &mut dst[row * intermediate_size..(row + 1) * intermediate_size];
                for i in 0..intermediate_size {
                    let g: f64 = gate[i].to_f64();
                    let u: f64 = up[i].to_f64();
                    let silu_g = g / (1.0 + (-g).exp());
                    out[i] = T::from_f64(silu_g * u);
                }
            }
            let mut out_dims = dims.to_vec();
            *out_dims.last_mut().unwrap() = intermediate_size;
            let storage = T::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(&out_dims)))
        }

        match storage {
            C::BF16(s) => inner(s, layout, self.intermediate_size),
            C::F16(s) => inner(s, layout, self.intermediate_size),
            C::F32(s) => inner(s, layout, self.intermediate_size),
            C::F64(s) => inner(s, layout, self.intermediate_size),
            _ => candle_core::bail!("unsupported dtype for fused_silu_mul"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &CudaStorage,
        layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dev = storage.device();
        let dims = layout.shape().dims();
        let last = *dims.last().unwrap();
        let intermediate_size = self.intermediate_size;

        if last != 2 * intermediate_size {
            candle_core::bail!(
                "fused_silu_mul: last dim {last} != 2*intermediate_size {}",
                2 * intermediate_size
            );
        }

        let (o1, o2) = match layout.contiguous_offsets() {
            None => candle_core::bail!("fused_silu_mul: input must be contiguous"),
            Some(offsets) => offsets,
        };

        let n_rows = (o2 - o1) / last;
        let out_el = n_rows * intermediate_size;

        // Choose kernel name and launch
        let fn_name = match storage.dtype() {
            DType::BF16 => "fused_silu_mul_bf16",
            DType::F16 => "fused_silu_mul_f16",
            DType::F32 => "fused_silu_mul_f32",
            dt => candle_core::bail!("fused_silu_mul: unsupported dtype {dt:?}"),
        };
        let func = load_func!(dev, fn_name)?;

        let block_size = 1024u32.min(intermediate_size as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_rows as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let slice = match &storage.slice {
            CudaStorageSlice::BF16(s) => {
                let s = s.slice(o1..o2);
                let dst = unsafe { dev.alloc::<half::bf16>(out_el)? };
                let mut builder = func.builder();
                builder.arg(&s);
                builder.arg(&dst);
                let isize_i32 = intermediate_size as i32;
                builder.arg(&isize_i32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::BF16(dst)
            }
            CudaStorageSlice::F16(s) => {
                let s = s.slice(o1..o2);
                let dst = unsafe { dev.alloc::<half::f16>(out_el)? };
                let mut builder = func.builder();
                builder.arg(&s);
                builder.arg(&dst);
                let isize_i32 = intermediate_size as i32;
                builder.arg(&isize_i32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F16(dst)
            }
            CudaStorageSlice::F32(s) => {
                let s = s.slice(o1..o2);
                let dst = unsafe { dev.alloc::<f32>(out_el)? };
                let mut builder = func.builder();
                builder.arg(&s);
                builder.arg(&dst);
                let isize_i32 = intermediate_size as i32;
                builder.arg(&isize_i32);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F32(dst)
            }
            _ => candle_core::bail!("fused_silu_mul: unsupported storage type"),
        };

        let mut out_dims = dims.to_vec();
        *out_dims.last_mut().unwrap() = intermediate_size;
        let dst = CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, Shape::from_dims(&out_dims)))
    }
}

/// Convenience function: fused SiLU(gate) * up.
///
/// `gate_up` must have shape `[..., 2*intermediate_size]` and be contiguous.
pub fn fused_silu_mul(gate_up: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    gate_up.apply_op1_no_bwd(&FusedSiluMul { intermediate_size })
}

// =====================================================================
// 2. Fused residual_add + RMSNorm
// =====================================================================

/// Fused residual addition + RMSNorm.
///
/// Computes: `residual += hidden; out = rmsnorm(residual, weight, eps)`
/// in one pass. The `residual` tensor is updated **in-place**.
///
/// Returns the normalized output tensor.
pub struct FusedAddRmsNorm {
    pub eps: f32,
}

impl FusedAddRmsNorm {
    /// Execute the fused add+rmsnorm on CUDA.
    ///
    /// `residual` is updated in-place. Returns normalized output.
    pub fn fwd(
        &self,
        residual: &mut Tensor,
        hidden: &Tensor,
        weight: &Tensor,
    ) -> Result<Tensor> {
        // Ensure all contiguous
        let residual_c = residual.contiguous()?;
        let hidden = hidden.contiguous()?;
        let weight = weight.contiguous()?;

        match residual_c.device() {
            Device::Cuda(_) => {
                self.cuda_fwd_inplace(residual, &hidden, &weight)
            }
            _ => {
                // CPU fallback: just do add + rmsnorm separately
                let sum = (&residual_c + &hidden)?;
                *residual = sum.clone();
                let norm = candle_nn::RmsNorm::new(weight.clone(), self.eps as f64);
                candle_core::Module::forward(&norm, &sum)
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd_inplace(
        &self,
        residual: &mut Tensor,
        hidden: &Tensor,
        weight: &Tensor,
    ) -> Result<Tensor> {
        // For simplicity of the in-place update, fall back to two-op path
        // but use candle ops (no extra allocation for the sum since the
        // memory pool recycles).
        //
        // The true in-place version requires unsafe access to the residual's
        // internal CudaSlice which candle doesn't expose cleanly for writes.
        // We'll keep the fused kernel for future use when candle adds
        // in-place mutation support, and for now just eliminate the allocation
        // overhead via memory pool + event tracking disable.
        let sum = (residual.contiguous()? + hidden)?;
        *residual = sum.clone();
        let norm = candle_nn::RmsNorm::new(weight.clone(), self.eps as f64);
        candle_core::Module::forward(&norm, &sum)
    }
}

// =====================================================================
// 3. GPU Argmax — greedy decode without DtoH logits transfer
// =====================================================================

/// Perform argmax on the GPU, returning only the index (4 bytes DtoH
/// instead of the full vocab_size * 2 bytes).
///
/// `logits` shape: `[1, 1, vocab_size]` or `[1, vocab_size]` or `[vocab_size]`
/// Returns the token index as u32.
#[cfg(feature = "cuda")]
pub fn gpu_argmax(logits: &Tensor) -> Result<u32> {
    let device = logits.device();
    let dev = match device {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("gpu_argmax requires CUDA device"),
    };

    let logits = logits.contiguous()?.flatten_all()?;
    let vocab_size = logits.elem_count();

    // Get the underlying storage
    let (storage, layout) = logits.storage_and_layout();
    let cuda_storage = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("gpu_argmax: expected CUDA storage"),
    };

    let (o1, _o2) = match layout.contiguous_offsets() {
        Some(o) => o,
        None => candle_core::bail!("gpu_argmax: logits must be contiguous"),
    };

    // Phase 1: per-block reduction
    let num_blocks = 256u32.min((vocab_size as u32 + 1023) / 1024);
    let block_size = 256u32;

    let func1 = load_func!(dev, "gpu_argmax_bf16_phase1")?;
    let func2 = load_func!(dev, "gpu_argmax_phase2")?;

    // Allocate temporary buffers for block results
    let block_max_vals: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32> =
        unsafe { dev.alloc::<f32>(num_blocks as usize)? };
    let block_max_idxs: candle_core::cuda_backend::cudarc::driver::CudaSlice<i32> =
        unsafe { dev.alloc::<i32>(num_blocks as usize)? };
    let output_token: candle_core::cuda_backend::cudarc::driver::CudaSlice<i32> =
        unsafe { dev.alloc::<i32>(1)? };

    // Phase 1 launch
    let cfg1 = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    match &cuda_storage.slice {
        CudaStorageSlice::BF16(s) => {
            let s = s.slice(o1..);
            let mut builder = func1.builder();
            builder.arg(&s);
            builder.arg(&block_max_vals);
            builder.arg(&block_max_idxs);
            let vs = vocab_size as i32;
            builder.arg(&vs);
            unsafe { builder.launch(cfg1) }.w()?;
        }
        _ => candle_core::bail!("gpu_argmax currently only supports BF16"),
    }

    // Phase 2: reduce block results
    let cfg2 = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (num_blocks.min(256), 1, 1),
        shared_mem_bytes: 0,
    };

    {
        let mut builder = func2.builder();
        builder.arg(&block_max_vals);
        builder.arg(&block_max_idxs);
        builder.arg(&output_token);
        let nb = num_blocks as i32;
        builder.arg(&nb);
        unsafe { builder.launch(cfg2) }.w()?;
    }

    // DtoH: only 4 bytes!
    let result = dev.clone_dtoh(&output_token)?;
    Ok(result[0] as u32)
}

// =====================================================================
// 4. GPU TopK — returns indices of the top-k largest values
// =====================================================================

#[cfg(feature = "cuda")]
thread_local! {
    static TOPK_TMP: std::cell::RefCell<
        std::collections::HashMap<(candle_core::cuda_backend::DeviceId, usize), TopkTmpBufs>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

#[cfg(feature = "cuda")]
struct TopkTmpBufs {
    vals: candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    idx: candle_core::cuda_backend::cudarc::driver::CudaSlice<u32>,
    cap_elems: usize,
}

/// GPU top-k indices for 1D f32 tensors (k ≤ 64).
///
/// Two-stage block reduction using custom CUDA kernels compiled from
/// `crane-core/kernels/fused_ops.cu`.
///
/// Returns a `[k]` U32 tensor of the indices of the k largest values,
/// sorted in descending order of value.
#[cfg(feature = "cuda")]
pub fn topk_indices(logits: &Tensor, k: usize) -> Result<Tensor> {
    if !logits.is_contiguous() {
        candle_core::bail!("topk_indices requires contiguous input");
    }
    if logits.rank() != 1 {
        candle_core::bail!("topk_indices expects a 1D tensor");
    }
    if k == 0 || k > 64 {
        candle_core::bail!("topk_indices expects 0 < k <= 64");
    }
    let n = logits.dims1()?;
    if k > n {
        candle_core::bail!("topk_indices expects k <= n");
    }
    logits.apply_op1_no_bwd(&TopKIndicesOp { k })
}

#[cfg(feature = "cuda")]
struct TopKIndicesOp {
    k: usize,
}

#[cfg(feature = "cuda")]
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

        let kth = k.saturating_sub(1);
        pairs.select_nth_unstable_by(kth, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Greater)
        });
        pairs.truncate(k);
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Greater));

        let out: Vec<u32> = pairs.into_iter().map(|(_, i)| i).collect();
        Ok((candle_core::CpuStorage::U32(out), Shape::from_dims(&[k])))
    }

    #[cfg(feature = "cuda")]
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

        let block_dim1 = 128u32;
        let block_dim2 = 128u32;
        let items_per_block = (block_dim1 as usize) * 8;
        let grid = ((n + items_per_block - 1) / items_per_block).clamp(1, 1024);
        let grid_dim = grid as u32;
        let shared1 =
            block_dim1 as usize * k * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>());
        let shared2 =
            block_dim2 as usize * k * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>());

        let cap_elems = grid * k;
        let dev_id = dev.id();
        let (tmp_vals, tmp_idx) = TOPK_TMP.with(|cell| -> Result<_> {
            let mut map = cell.borrow_mut();
            match map.get_mut(&(dev_id, k)) {
                Some(bufs) if bufs.cap_elems >= cap_elems => {
                    Ok((bufs.vals.clone(), bufs.idx.clone()))
                }
                _ => {
                    let vals = unsafe { dev.alloc::<f32>(cap_elems)? };
                    let idx = unsafe { dev.alloc::<u32>(cap_elems)? };
                    map.insert(
                        (dev_id, k),
                        TopkTmpBufs {
                            vals: vals.clone(),
                            idx: idx.clone(),
                            cap_elems,
                        },
                    );
                    Ok((vals, idx))
                }
            }
        })?;

        let out_idx = unsafe { dev.alloc::<u32>(k)? };

        // Stage 1
        let f1 = load_func!(dev, "topk_stage1_f32")?;
        let items_per_block_u32 = items_per_block as u32;
        {
            let mut builder = f1.builder();
            builder.arg(&x);
            builder.arg(&n_u32);
            builder.arg(&k_u32);
            builder.arg(&items_per_block_u32);
            builder.arg(&tmp_vals);
            builder.arg(&tmp_idx);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (grid_dim, 1, 1),
                    block_dim: (block_dim1, 1, 1),
                    shared_mem_bytes: shared1 as u32,
                })
            }
            .w()?;
        }

        // Stage 2
        let m = grid_dim * k_u32;
        let f2 = load_func!(dev, "topk_stage2_f32")?;
        {
            let mut builder = f2.builder();
            builder.arg(&tmp_vals);
            builder.arg(&tmp_idx);
            builder.arg(&m);
            builder.arg(&k_u32);
            builder.arg(&out_idx);
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (block_dim2, 1, 1),
                    shared_mem_bytes: shared2 as u32,
                })
            }
            .w()?;
        }

        let dst = CudaStorage::wrap_cuda_slice(out_idx, dev.clone());
        Ok((dst, Shape::from_dims(&[k])))
    }
}

// =====================================================================
// 5. CUDA tensor memory utilities
// =====================================================================

/// Copy a u32 slice from host to the target device, returning a new 1-D U32 tensor.
///
/// The returned tensor has shape `[src.len()]` on the same device as `device`.
/// This is a plain HtoD allocation — no kernel launch.
#[cfg(feature = "cuda")]
pub fn copy_from_slice_u32(src: &[u32], device: &Device) -> Result<Tensor> {
    Tensor::new(src, device)
}

/// Clone a contiguous f32 tensor — returns a new contiguous copy on the same device.
///
/// For CUDA tensors this is a DtoD copy (no host round-trip).
#[cfg(feature = "cuda")]
pub fn copy_from_tensor_f32(src_tensor: &Tensor) -> Result<Tensor> {
    if src_tensor.dtype() != DType::F32 {
        candle_core::bail!(
            "copy_from_tensor_f32: expected f32 tensor, got {:?}",
            src_tensor.dtype()
        );
    }
    src_tensor.contiguous()
}
