//! Torch-compatible BF16 biased GEMM: same 1 MiB heuristic workspace.
use candle_core::cuda_backend::cudarc::{
    cublaslt::{CudaBlasLT, MatmulShared, result as lt, sys},
    driver::{CudaSlice, CudaView, DevicePtr, DevicePtrMut, DeviceRepr},
};
use candle_core::{Error, Result};
use std::{cell::RefCell, collections::HashMap, mem::size_of, sync::Arc};
struct Resource<T: Copy>(T, unsafe fn(T) -> std::result::Result<(), lt::CublasError>);
impl<T: Copy> Drop for Resource<T> {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.1)(self.0);
        }
    }
}
fn own<T: Copy>(
    value: std::result::Result<T, lt::CublasError>,
    free: unsafe fn(T) -> std::result::Result<(), lt::CublasError>,
) -> Result<Resource<T>> {
    Ok(Resource(value.map_err(Error::wrap)?, free))
}
unsafe fn attr<T>(
    desc: sys::cublasLtMatmulDesc_t,
    key: sys::cublasLtMatmulDescAttributes_t,
    value: &T,
) -> Result<()> {
    unsafe { lt::set_matmul_desc_attribute(desc, key, (value as *const T).cast(), size_of::<T>()) }
        .map_err(Error::wrap)
}
unsafe fn pref<T>(
    desc: sys::cublasLtMatmulPreference_t,
    key: sys::cublasLtMatmulPreferenceAttributes_t,
    value: &T,
) -> Result<()> {
    unsafe { lt::set_matmul_pref_attribute(desc, key, (value as *const T).cast(), size_of::<T>()) }
        .map_err(Error::wrap)
}
pub(super) fn matmul<T: DeviceRepr>(
    blas: &CudaBlasLT,
    w: &CudaView<'_, T>,
    x: &CudaView<'_, T>,
    bias: &CudaView<'_, T>,
    out: &mut CudaSlice<T>,
    m: usize,
    n: usize,
    k: usize,
    tf32: bool,
) -> Result<()> {
    use sys::cublasLtMatmulDescAttributes_t as A;
    use sys::cublasLtMatmulPreferenceAttributes_t as P;
    thread_local! {static WORK:RefCell<HashMap<usize,CudaSlice<u8>>>=RefCell::new(HashMap::new());}
    let dtype = if tf32 {
        sys::cudaDataType_t::CUDA_R_32F
    } else {
        sys::cudaDataType_t::CUDA_R_16BF
    };
    let compute = if tf32 {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
    } else {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    };
    let stream = blas.stream();
    let key = Arc::as_ptr(stream) as usize;
    WORK.with(|work| -> Result<()> {
        let mut work = work.borrow_mut();
        if !work.contains_key(&key) {
            work.insert(
                key,
                unsafe { stream.alloc::<u8>(1024 * 1024) }.map_err(Error::wrap)?,
            );
        }
        let workspace = work.get_mut(&key).expect("workspace was initialized");
        let (wp, _wg) = w.device_ptr(stream);
        let (xp, _xg) = x.device_ptr(stream);
        let (bp, _bg) = bias.device_ptr(stream);
        let (op, _og) = out.device_ptr_mut(stream);
        let (scratch, _sg) = workspace.device_ptr_mut(stream);
        let desc = own(
            lt::create_matmul_desc(compute, sys::cudaDataType_t::CUDA_R_32F),
            lt::destroy_matmul_desc,
        )?;
        let a = own(
            lt::create_matrix_layout(dtype, k as u64, m as u64, k as i64),
            lt::destroy_matrix_layout,
        )?;
        let b = own(
            lt::create_matrix_layout(dtype, k as u64, n as u64, k as i64),
            lt::destroy_matrix_layout,
        )?;
        let c = own(
            lt::create_matrix_layout(dtype, m as u64, n as u64, m as i64),
            lt::destroy_matrix_layout,
        )?;
        let preference = own(lt::create_matmul_pref(), lt::destroy_matmul_pref)?;
        // SAFETY: all descriptors and device allocation guards stay live
        // through the launch; the shared workspace is used on one stream.
        unsafe {
            attr(desc.0, A::CUBLASLT_MATMUL_DESC_TRANSA, &1i32)?;
            attr(desc.0, A::CUBLASLT_MATMUL_DESC_TRANSB, &0i32)?;
            attr(
                desc.0,
                A::CUBLASLT_MATMUL_DESC_EPILOGUE,
                &sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
            )?;
            attr(desc.0, A::CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp)?;
            pref(
                preference.0,
                P::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &(1024 * 1024usize),
            )?;
            for (attribute, ptr) in [
                (P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, wp),
                (P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, xp),
                (P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, op),
                (P::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, bp),
            ] {
                let alignment = (1u64 << ptr.trailing_zeros().min(8)) as u32;
                pref(preference.0, attribute, &alignment)?;
            }
            let heuristic = lt::get_matmul_algo_heuristic(
                *blas.handle(),
                desc.0,
                a.0,
                b.0,
                c.0,
                c.0,
                preference.0,
            )
            .map_err(Error::wrap)?;
            lt::matmul(
                *blas.handle(),
                desc.0,
                (&1f32 as *const f32).cast(),
                (&0f32 as *const f32).cast(),
                wp as *const _,
                a.0,
                xp as *const _,
                b.0,
                op as *const _,
                c.0,
                op as *mut _,
                c.0,
                &heuristic.algo,
                scratch as *mut _,
                1024 * 1024,
                stream.cu_stream() as _,
            )
            .map_err(Error::wrap)?;
        }
        Ok(())
    })
}
