//! Shared plumbing for Crane's own ROCm/HIP kernels.
//!
//! Crane owns two kernel sources (`kernels/cuda/gdn.cu`, `kernels/cuda/fused_ops.cu`).
//! On CUDA they are compiled to PTX at build time; on ROCm candle compiles the
//! *same* sources with `hipcc` on first use and caches the code object on disk
//! (`RocmDevice::get_or_load_custom_func`). What each launcher then needs is
//! the same three things — a device pointer into a tensor, a kernel launch, and
//! a way to hand an output buffer back as a `Tensor` — so they live here rather
//! than being written twice.
//!
//! Raw pointers, not typed slices: candle's ROCm storage exposes
//! `SendSyncDeviceMemory<T>` with a `ptr_at`, and `hipModuleLaunchKernel` takes
//! a `*mut c_void` per argument, so there is no equivalent of cudarc's typed
//! `builder.arg(&slice)`.

use std::ffi::c_void;

use candle_core::op::BackpropOp;
use candle_core::rocm_backend::{
    RocmDevice, RocmStorage, RocmStorageSlice, SendSyncDeviceMemory, rocm_rs,
};
use candle_core::{DType, Layout, Result, Shape, Storage, Tensor};

/// The ROCm slice behind a tensor's storage, or an error naming the operand.
///
/// # Errors
///
/// Returns an error if `storage` is not ROCm storage.
pub fn rocm_slice<'a>(storage: &'a Storage, what: &str) -> Result<&'a RocmStorageSlice> {
    match storage {
        Storage::Rocm(s) => Ok(&s.slice),
        _ => candle_core::bail!("{what} must be a rocm tensor"),
    }
}

/// Device pointer to the first element `layout` addresses.
///
/// Callers must keep the storage guard `storage` borrows from alive until after
/// the launch — the pointer is only valid while the tensor is.
///
/// # Errors
///
/// Returns an error if `storage` is not ROCm storage, if `layout` is not
/// contiguous, or if the slice's dtype is not `dtype`.
pub fn device_ptr(
    storage: &Storage,
    layout: &Layout,
    dtype: DType,
    what: &str,
) -> Result<*mut c_void> {
    slice_ptr(rocm_slice(storage, what)?, layout, dtype, what)
}

/// [`device_ptr`] for a slice already unwrapped from its `Storage`/`RocmStorage`
/// wrapper — what `CustomOp2::rocm_fwd` implementations receive directly.
///
/// # Errors
///
/// Returns an error if `layout` is not contiguous, or if `slice`'s dtype is not
/// `dtype`.
pub fn slice_ptr(
    slice: &RocmStorageSlice,
    layout: &Layout,
    dtype: DType,
    what: &str,
) -> Result<*mut c_void> {
    if !layout.is_contiguous() {
        candle_core::bail!("{what} must be contiguous");
    }
    if slice.dtype() != dtype {
        candle_core::bail!("{what} must be {dtype:?}, got {:?}", slice.dtype());
    }
    let offset = layout.start_offset();
    // SAFETY: `offset` is a `Layout::start_offset`, i.e. an element index the
    // tensor's own allocation covers.
    let ptr = unsafe {
        match slice {
            RocmStorageSlice::U8(m) => m.ptr_at(offset),
            RocmStorageSlice::F8E4M3(m) => m.ptr_at(offset),
            RocmStorageSlice::U32(m) => m.ptr_at(offset),
            RocmStorageSlice::I16(m) => m.ptr_at(offset),
            RocmStorageSlice::I32(m) => m.ptr_at(offset),
            RocmStorageSlice::I64(m) => m.ptr_at(offset),
            RocmStorageSlice::BF16(m) => m.ptr_at(offset),
            RocmStorageSlice::F16(m) => m.ptr_at(offset),
            RocmStorageSlice::F32(m) => m.ptr_at(offset),
            RocmStorageSlice::F64(m) => m.ptr_at(offset),
        }
    };
    Ok(ptr)
}

/// One entry of a kernel argument list.
///
/// `hipModuleLaunchKernel` takes an array of pointers *to* the arguments, so
/// every scalar and every device pointer is passed by address. `value` must
/// outlive the launch.
pub fn arg<T>(value: &T) -> *mut c_void {
    std::ptr::from_ref(value).cast_mut().cast()
}

/// Launch `kernel` from `source`, compiling the module on first use.
///
/// `module` names the on-disk cache entry and must be unique across the custom
/// modules of the process; candle keeps custom modules in their own namespace,
/// so it cannot collide with one of candle's own.
///
/// # Safety
///
/// `args` must match the kernel's parameter list exactly, and `grid` must cover
/// every element the kernel writes.
///
/// # Errors
///
/// Returns an error if `hipcc` fails, if `kernel` is not in `source`, or if the
/// launch is rejected (an oversized `shared_mem_bytes` or block, typically).
// The parameter list is the launch geometry `hipModuleLaunchKernel` takes;
// bundling it into a struct would just move the same fields.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch(
    dev: &RocmDevice,
    module: &str,
    kernel: &str,
    source: &str,
    grid: u32,
    block: u32,
    shared_mem_bytes: u32,
    args: &mut [*mut c_void],
) -> Result<()> {
    let func = dev.get_or_load_custom_func(kernel, module, source)?;
    func.launch(
        rocm_rs::hip::Dim3::from(grid),
        rocm_rs::hip::Dim3::from(block),
        shared_mem_bytes,
        Some(dev.stream()),
        args,
    )
    .map_err(|e| candle_core::Error::Msg(format!("{kernel} launch failed: {e}")))
}

/// A dtype whose device buffer can be wrapped straight into a
/// [`RocmStorageSlice`], so [`launch_binary_elementwise`] can allocate,
/// launch and wrap generically instead of once per dtype.
pub trait RocmElem: Sized {
    /// Wrap a freshly-allocated buffer of `Self` as the matching
    /// `RocmStorageSlice` variant.
    fn wrap_slice(buf: SendSyncDeviceMemory<Self>) -> RocmStorageSlice;
}

impl RocmElem for half::bf16 {
    fn wrap_slice(buf: SendSyncDeviceMemory<Self>) -> RocmStorageSlice {
        RocmStorageSlice::BF16(buf)
    }
}

impl RocmElem for half::f16 {
    fn wrap_slice(buf: SendSyncDeviceMemory<Self>) -> RocmStorageSlice {
        RocmStorageSlice::F16(buf)
    }
}

impl RocmElem for f32 {
    fn wrap_slice(buf: SendSyncDeviceMemory<Self>) -> RocmStorageSlice {
        RocmStorageSlice::F32(buf)
    }
}

/// Launch a flat binary elementwise kernel of signature `(const T*, const T*,
/// T*, uint32_t n)`, allocating the `T`-typed output buffer and wrapping it
/// into a [`RocmStorageSlice`].
///
/// Called by [`binary_elementwise_fwd`], which adds shape validation and
/// dtype dispatch on top. Individual `CustomOp2::rocm_fwd` implementations
/// should prefer that higher-level helper.
///
/// # Safety
///
/// `kernel` (found in `source`) must have exactly the signature described
/// above for dtype `T`. `a_ptr` and `b_ptr` must be valid ROCm device
/// pointers on `dev`, of dtype `T`, each addressing at least `n` contiguous
/// elements, and must remain valid until the launched kernel completes on
/// `dev`'s stream — the same caller obligation [`device_ptr`] documents for
/// the storage guard behind the pointer it returns.
///
/// # Errors
///
/// Returns an error if allocation fails or the launch is rejected (see
/// [`launch`]).
pub unsafe fn launch_binary_elementwise<T: RocmElem>(
    dev: &RocmDevice,
    module: &str,
    kernel: &str,
    source: &str,
    a_ptr: *mut c_void,
    b_ptr: *mut c_void,
    n: usize,
) -> Result<RocmStorageSlice> {
    let dst = dev.alloc::<T>(n)?;
    let dst_ptr = dst.as_ptr();
    // `n` is a tensor's total element count; real workloads never approach
    // u32::MAX (~4 billion) elements.
    #[allow(clippy::cast_possible_truncation)]
    let n_u32 = n as u32;
    let block = 256u32;
    let grid = n_u32.div_ceil(block);
    let mut args = [arg(&a_ptr), arg(&b_ptr), arg(&dst_ptr), arg(&n_u32)];
    // SAFETY: forwarded from this function's own safety contract.
    unsafe { launch(dev, module, kernel, source, grid, block, 0, &mut args) }?;
    Ok(T::wrap_slice(dst))
}

/// Full ROCm forward pass for a binary elementwise `CustomOp2`.
///
/// Handles shape validation, dtype dispatch, pointer extraction, kernel
/// launch and result wrapping — the entire `rocm_fwd` body that every binary
/// elementwise fused op (`snake`, `atan2`, `swiglu`) would otherwise
/// duplicate. Each op calls this from its `CustomOp2::rocm_fwd`, passing its
/// own module name, kernel prefix and `.cu` source; the launched kernel name
/// is `{kernel_prefix}_{bf16,f16,f32}` depending on the operands' dtype.
///
/// # Safety
///
/// For dtype `T` in `{BF16, F16, F32}`, the kernel named
/// `{kernel_prefix}_{bf16,f16,f32}` in `source` must have exactly the
/// signature `(const T*, const T*, T*, uint32_t n)` — the same contract
/// [`launch_binary_elementwise`] requires, forwarded here.
///
/// # Errors
///
/// Returns an error if `l_lhs`/`l_rhs` have different shapes, if either
/// operand is not contiguous, if the dtype is not `BF16`/`F16`/`F32`, or if
/// the kernel launch fails.
#[allow(clippy::too_many_arguments)]
pub unsafe fn binary_elementwise_fwd(
    s_lhs: &RocmStorage,
    l_lhs: &Layout,
    s_rhs: &RocmStorage,
    l_rhs: &Layout,
    module: &str,
    kernel_prefix: &str,
    source: &str,
) -> Result<(RocmStorage, Shape)> {
    if l_lhs.shape() != l_rhs.shape() {
        candle_core::bail!("{kernel_prefix}: lhs and rhs must have the same shape");
    }

    let dev = s_lhs.device.clone();
    let n = l_lhs.shape().elem_count();
    let dtype = s_lhs.slice.dtype();
    let kernel_name = match dtype {
        DType::BF16 => format!("{kernel_prefix}_bf16"),
        DType::F16 => format!("{kernel_prefix}_f16"),
        DType::F32 => format!("{kernel_prefix}_f32"),
        dt => candle_core::bail!("{kernel_prefix}: unsupported dtype {dt:?}"),
    };

    let lhs_ptr = slice_ptr(&s_lhs.slice, l_lhs, dtype, &format!("{kernel_prefix} lhs"))?;
    let rhs_ptr = slice_ptr(&s_rhs.slice, l_rhs, dtype, &format!("{kernel_prefix} rhs"))?;

    // SAFETY: forwarded from this function's own safety contract.
    let slice = unsafe {
        match dtype {
            DType::BF16 => launch_binary_elementwise::<half::bf16>(
                &dev,
                module,
                &kernel_name,
                source,
                lhs_ptr,
                rhs_ptr,
                n,
            ),
            DType::F16 => launch_binary_elementwise::<half::f16>(
                &dev,
                module,
                &kernel_name,
                source,
                lhs_ptr,
                rhs_ptr,
                n,
            ),
            DType::F32 => launch_binary_elementwise::<f32>(
                &dev,
                module,
                &kernel_name,
                source,
                lhs_ptr,
                rhs_ptr,
                n,
            ),
            _ => unreachable!("dtype already validated above"),
        }
    }?;

    Ok((RocmStorage { slice, device: dev }, l_lhs.shape().clone()))
}

/// Hand an f32 output buffer back as a `Tensor` of `shape`, without a copy.
pub fn wrap_f32<S: Into<Shape>>(
    buf: SendSyncDeviceMemory<f32>,
    dev: &RocmDevice,
    shape: S,
) -> Tensor {
    wrap(RocmStorageSlice::F32(buf), dev, shape)
}

/// [`wrap_f32`] for a u32 buffer.
pub fn wrap_u32<S: Into<Shape>>(
    buf: SendSyncDeviceMemory<u32>,
    dev: &RocmDevice,
    shape: S,
) -> Tensor {
    wrap(RocmStorageSlice::U32(buf), dev, shape)
}

/// [`wrap_f32`] for an f16 buffer.
pub fn wrap_f16<S: Into<Shape>>(
    buf: SendSyncDeviceMemory<half::f16>,
    dev: &RocmDevice,
    shape: S,
) -> Tensor {
    wrap(RocmStorageSlice::F16(buf), dev, shape)
}

/// [`wrap_f32`] for a bf16 buffer.
pub fn wrap_bf16<S: Into<Shape>>(
    buf: SendSyncDeviceMemory<half::bf16>,
    dev: &RocmDevice,
    shape: S,
) -> Tensor {
    wrap(RocmStorageSlice::BF16(buf), dev, shape)
}

fn wrap<S: Into<Shape>>(slice: RocmStorageSlice, dev: &RocmDevice, shape: S) -> Tensor {
    Tensor::from_storage(
        Storage::Rocm(RocmStorage {
            slice,
            device: dev.clone(),
        }),
        shape,
        BackpropOp::none(),
        false,
    )
}
