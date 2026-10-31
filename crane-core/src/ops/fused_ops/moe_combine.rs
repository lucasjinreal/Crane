// SPDX-License-Identifier: MIT
//! Fused `MoE` combine: each token's routed expert outputs, weighted by the
//! router and summed, in one pass.
//!
//! Replaces the gather / cast / weight / sum / cast chain that a routed
//! output `[pairs, hidden]` otherwise takes back to `[tokens, hidden]`: at
//! prefill sizes each step of that chain is a full pass over a `[pairs,
//! hidden]` tensor (134 MB in F32 for a 2048-token chunk of a top-8, 2048-wide
//! model), and at decode each is a launch. SYCL has a kernel
//! (`kernels/sycl/fused_ops.cpp`); every other device runs the chain itself.

use candle_core::{D, DType, Result, Tensor};

/// `out[t] = sum_k weights[t, k] * y[row(t, k)]`, `[tokens, hidden]` in
/// `out_dtype`.
///
/// `y` is `[n, hidden]` (F32 or F16); `rows` (`U32`, `tokens * top_k`) maps
/// pair `t * top_k + k` to its row of `y`, or `None` when `y` is already in
/// pair order (`n == tokens * top_k`). `weights` is `[tokens, top_k]` F32.
/// The sum runs over `k` in order, in F32.
///
/// # Errors
///
/// Returns an error if the shapes disagree or a kernel fails.
pub fn moe_combine(
    y: &Tensor,
    rows: Option<&Tensor>,
    weights: &Tensor,
    out_dtype: DType,
) -> Result<Tensor> {
    let (tokens, top_k) = weights.dims2()?;
    let hidden = y.dim(1)?;
    if rows.map_or(y.dim(0)?, Tensor::elem_count) != tokens * top_k {
        candle_core::bail!(
            "moe_combine: {} routed rows for {tokens} tokens x top-{top_k}",
            rows.map_or(y.dim(0)?, Tensor::elem_count)
        )
    }

    #[cfg(feature = "sycl")]
    if y.device().is_sycl()
        && matches!(y.dtype(), DType::F32 | DType::F16)
        && matches!(out_dtype, DType::F32 | DType::F16)
    {
        return sycl_impl::moe_combine(y, rows, weights, out_dtype, [tokens, top_k, hidden]);
    }

    // The chain this replaces.
    let y = match rows {
        Some(rows) => y.index_select(&rows.flatten_all()?, 0)?,
        None => y.clone(),
    };
    y.to_dtype(DType::F32)?
        .reshape((tokens, top_k, hidden))?
        .broadcast_mul(&weights.to_dtype(DType::F32)?.unsqueeze(D::Minus1)?)?
        .sum(1)?
        .to_dtype(out_dtype)
}

#[cfg(feature = "sycl")]
mod sycl_impl {
    //! Launcher for `crane_moe_combine_sycl` (`kernels/sycl/fused_ops.cpp`).

    use std::ffi::c_void;

    use candle_core::op::BackpropOp;
    use candle_core::{DType, Result, Storage, SyclStorage, Tensor};

    // libcrane_gdn_sycl.so — linked by build.rs when `--features sycl`.
    unsafe extern "C" {
        fn crane_moe_combine_sycl(
            queue: *mut c_void,
            y: *const c_void,
            y_f16: i32,
            rows: *const u32,
            w: *const f32,
            out: *mut c_void,
            out_f16: i32,
            tokens: i32,
            top_k: i32,
            hidden: i32,
        ) -> i32;
    }

    fn ptr(t: &Storage, offset_bytes: usize, name: &str) -> Result<*const c_void> {
        match t {
            Storage::Sycl(st) => Ok(unsafe {
                st.buf()
                    .as_ptr()
                    .cast::<u8>()
                    .add(offset_bytes)
                    .cast::<c_void>()
            }),
            _ => candle_core::bail!("moe_combine: {name} must be a sycl tensor"),
        }
    }

    pub(super) fn moe_combine(
        y: &Tensor,
        rows: Option<&Tensor>,
        weights: &Tensor,
        out_dtype: DType,
        [tokens, top_k, hidden]: [usize; 3],
    ) -> Result<Tensor> {
        let int = |n: usize| {
            i32::try_from(n).map_err(|_| candle_core::Error::Msg(format!("{n} exceeds i32")))
        };
        let y = y.contiguous()?;
        let rows = rows
            .map(|r| r.flatten_all()?.to_dtype(DType::U32)?.contiguous())
            .transpose()?;
        let weights = weights.to_dtype(DType::F32)?.contiguous()?;
        let dev = y.device().as_sycl_device()?.clone();

        let (y_s, y_l) = y.storage_and_layout();
        let (w_s, w_l) = weights.storage_and_layout();
        let rows_s = rows.as_ref().map(Tensor::storage_and_layout);
        let rows_ptr = match &rows_s {
            Some((s, l)) => ptr(s, l.start_offset() * 4, "rows")?.cast::<u32>(),
            None => std::ptr::null(),
        };
        let n = tokens * hidden;
        let out = dev.alloc_bytes(n * out_dtype.size_in_bytes())?;
        let status = unsafe {
            crane_moe_combine_sycl(
                dev.queue().native_ptr(),
                ptr(&y_s, y_l.start_offset() * y.dtype().size_in_bytes(), "y")?,
                i32::from(y.dtype() == DType::F16),
                rows_ptr,
                ptr(&w_s, w_l.start_offset() * 4, "weights")?.cast::<f32>(),
                out.as_mut_ptr(),
                i32::from(out_dtype == DType::F16),
                int(tokens)?,
                int(top_k)?,
                int(hidden)?,
            )
        };
        drop((y_s, w_s, rows_s));
        if status != 0 {
            candle_core::bail!("crane_moe_combine_sycl failed (status {status})");
        }
        Ok(Tensor::from_storage(
            Storage::Sycl(SyclStorage::from_buffer(&dev, out, out_dtype, n)),
            (tokens, hidden),
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "sycl")]
    use super::*;
    #[cfg(feature = "sycl")]
    use candle_core::Device;

    /// The SYCL kernel against the portable chain, with and without a row
    /// map, in each supported dtype combination.
    #[cfg(feature = "sycl")]
    #[test]
    fn sycl_matches_portable() -> Result<()> {
        if !candle_core::utils::sycl_is_available() {
            return Ok(());
        }
        let sycl = Device::new_sycl(0)?;
        let cpu = Device::Cpu;
        let (tokens, top_k, hidden) = (37usize, 8usize, 96usize);
        let pairs = tokens * top_k;
        // A permutation of the pairs into a larger buffer, like a GEMM plan's.
        let rows: Vec<u32> = (0..pairs)
            .map(|p| ((p * 131) % (pairs + 5)) as u32)
            .collect();
        let rows = Tensor::new(rows.as_slice(), &cpu)?;
        let weights = Tensor::rand(0f32, 1.0, (tokens, top_k), &cpu)?;
        for (y_dtype, out_dtype) in [
            (DType::F32, DType::F32),
            (DType::F16, DType::F16),
            (DType::F16, DType::F32),
        ] {
            for map in [false, true] {
                let n = if map { pairs + 5 } else { pairs };
                let y = Tensor::randn(0f32, 1.0, (n, hidden), &cpu)?.to_dtype(y_dtype)?;
                let r = map.then_some(&rows);
                let want = moe_combine(&y, r, &weights, out_dtype)?.to_dtype(DType::F32)?;
                let got = moe_combine(
                    &y.to_device(&sycl)?,
                    r.map(|r| r.to_device(&sycl)).transpose()?.as_ref(),
                    &weights.to_device(&sycl)?,
                    out_dtype,
                )?;
                assert_eq!(got.dtype(), out_dtype);
                let got = got.to_device(&cpu)?.to_dtype(DType::F32)?;
                let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;
                let diff = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
                let tol = if out_dtype == DType::F16 { 2e-3 } else { 1e-5 };
                assert!(
                    diff / scale < tol,
                    "{y_dtype:?}->{out_dtype:?} map={map}: rel diff {}",
                    diff / scale
                );
            }
        }
        Ok(())
    }
}
