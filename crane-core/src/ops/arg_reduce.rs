// SPDX-License-Identifier: MIT

//! `argmax` / `argmin` along any dim, correct on Metal.
//!
//! candle 0.11's Metal arg-reductions return wrong indices when the reduced
//! dim is not the last one and the dims after it all have size 1, or when the
//! reduced dim itself has size 1: `[k, 1].argmax(0)` is 0 whatever the values,
//! `[1, k].argmax(0)` is `0..k` instead of zeros. Value reductions (sum, max,
//! ...) are unaffected. See `data/docs/candle-metal-argmax.md` and
//! `tests/metal_arg_reduce.rs`.
//!
//! These helpers move the reduced dim last before reducing on Metal (the case
//! candle gets right) and call candle directly everywhere else.

use candle_core::{Result, Tensor};

fn along(t: &Tensor, dim: usize, keepdim: bool, max: bool) -> Result<Tensor> {
    let reduce = |t: &Tensor, dim: usize| {
        if max {
            t.argmax_keepdim(dim)
        } else {
            t.argmin_keepdim(dim)
        }
    };
    let last = t.rank().saturating_sub(1);
    let out = if t.device().is_metal() && dim != last {
        // `[.., d, .., n]` -> `[.., n, .., d]`, reduce the (contiguous) last dim
        // to 1, swap back: the keepdim result, in the original dim order.
        reduce(&t.transpose(dim, last)?.contiguous()?, last)?.transpose(dim, last)?
    } else {
        reduce(t, dim)?
    };
    if keepdim { Ok(out) } else { out.squeeze(dim) }
}

/// [`Tensor::argmax`] that is also correct on Metal.
///
/// # Errors
///
/// Returns an error if `dim` is out of range or the reduction fails.
pub fn argmax(t: &Tensor, dim: usize) -> Result<Tensor> {
    along(t, dim, false, true)
}

/// [`Tensor::argmax_keepdim`] that is also correct on Metal.
///
/// # Errors
///
/// Returns an error if `dim` is out of range or the reduction fails.
pub fn argmax_keepdim(t: &Tensor, dim: usize) -> Result<Tensor> {
    along(t, dim, true, true)
}

/// [`Tensor::argmin`] that is also correct on Metal.
///
/// # Errors
///
/// Returns an error if `dim` is out of range or the reduction fails.
pub fn argmin(t: &Tensor, dim: usize) -> Result<Tensor> {
    along(t, dim, false, false)
}

/// [`Tensor::argmin_keepdim`] that is also correct on Metal.
///
/// # Errors
///
/// Returns an error if `dim` is out of range or the reduction fails.
pub fn argmin_keepdim(t: &Tensor, dim: usize) -> Result<Tensor> {
    along(t, dim, true, false)
}
