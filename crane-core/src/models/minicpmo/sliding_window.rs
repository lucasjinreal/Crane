//! Basic-mode sliding-window KV cache eviction (phase 6b) — cache-length-
//! triggered, drop the oldest whole "units", realign RoPE for the
//! survivors. Ported from `utils.py`'s `drop_tokens_from_cache`/
//! `realign_rotary_suffix`/`get_rotary_cos_sin` (see the plan doc for the
//! real reference source).
//!
//! Cached keys are stored **post-RoPE** (RoPE is applied before caching in
//! `qwen3::modeling::Attention::forward`, same as the real HF
//! `DynamicCache`), so dropping a range from the middle of the cache
//! leaves the surviving suffix's rotation pointing at now-stale absolute
//! positions — the survivors need to be "un-rotated" from their old
//! position and "re-rotated" at their new (shifted) position. This is the
//! single highest numerical-risk piece of the whole duplex phase: a silent
//! realignment bug degrades generation quality gradually rather than
//! crashing, so this module is validated with hand-computed-value unit
//! tests (cross-checked against `candle_nn::rotary_emb::rope`, the
//! already-trusted fused kernel `Qwen3Model` itself uses) rather than
//! trusted on the strength of a live session alone.

use candle_core::{Result, Tensor, D};

use crate::models::modules::rotary::RotaryEmbedding;

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d - d / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// Full-width (duplicated, `[len, head_dim]`) cos/sin for positions
/// `[start, start+len)`, cast to `dtype` and reshaped to broadcast against
/// a `[B, H, len, head_dim]` key tensor — mirrors `get_rotary_cos_sin`.
fn full_width_cos_sin(rotary: &RotaryEmbedding, start: usize, len: usize, dtype: candle_core::DType) -> Result<(Tensor, Tensor)> {
    let (cos, sin) = rotary.forward(start, len)?;
    let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?.to_dtype(dtype)?.reshape((1, 1, len, ()))?;
    let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?.to_dtype(dtype)?.reshape((1, 1, len, ()))?;
    Ok((cos, sin))
}

/// Re-derive RoPE for `keys` (shape `[B, H, len, D]`, currently rotated as
/// if positioned at `old_start..old_start+len`) so they read as if
/// originally rotated at `new_start..new_start+len` instead — undo the old
/// rotation, then reapply at the new positions. Mirrors
/// `realign_rotary_suffix` in `utils.py`.
///
/// # Errors
///
/// Returns an error if the tensor ops fail (e.g. shape mismatch).
pub(crate) fn realign_rotary_suffix(rotary: &RotaryEmbedding, keys: &Tensor, old_start: usize, new_start: usize, len: usize) -> Result<Tensor> {
    if len == 0 {
        return Ok(keys.clone());
    }
    let dtype = keys.dtype();
    let (cos_old, sin_old) = full_width_cos_sin(rotary, old_start, len, dtype)?;
    let base = (keys.broadcast_mul(&cos_old)? - rotate_half(keys)?.broadcast_mul(&sin_old)?)?;
    let (cos_new, sin_new) = full_width_cos_sin(rotary, new_start, len, dtype)?;
    base.broadcast_mul(&cos_new)? + rotate_half(&base)?.broadcast_mul(&sin_new)?
}

/// Drop `length` cached tokens starting right after the first `preserve`
/// tokens (protecting e.g. the system prompt), realigning the survivors'
/// RoPE so they read as a contiguous, un-gapped sequence. Mirrors
/// `drop_tokens_from_cache` in `utils.py`. Operates on the `(K, V)` pairs
/// from `MiniCpmOLlm::get_kv_caches`; pass the result to
/// `MiniCpmOLlm::set_kv_caches` afterward.
///
/// Returns `Ok(false)` (no-op, cache left untouched) if there's nothing
/// valid to drop — matches Python's own "cannot drop: only N available"
/// guard rather than erroring.
///
/// # Errors
///
/// Returns an error if the tensor ops fail (e.g. shape mismatch).
pub(crate) fn drop_tokens_from_cache(caches: &mut [Option<(Tensor, Tensor)>], length: usize, preserve: usize, rotary: &RotaryEmbedding) -> Result<bool> {
    if length == 0 {
        return Ok(false);
    }
    let total_len = match caches.iter().find_map(|c| c.as_ref()) {
        Some((k, _)) => k.dim(2)?,
        None => return Ok(false),
    };
    if total_len == 0 {
        return Ok(false);
    }

    let preserve = preserve.min(total_len);
    let available = total_len - preserve;
    if available < length {
        return Ok(false);
    }

    let suffix_len = total_len - preserve - length;
    let suffix_old_start = preserve + length;
    let suffix_new_start = preserve;

    for cache in &mut *caches {
        let Some((k, v)) = cache else { continue };
        let prefix_k = k.narrow(2, 0, preserve)?;
        let prefix_v = v.narrow(2, 0, preserve)?;
        let (new_k, new_v) = if suffix_len > 0 {
            let suffix_k = k.narrow(2, suffix_old_start, suffix_len)?;
            let suffix_v = v.narrow(2, suffix_old_start, suffix_len)?;
            let suffix_k = realign_rotary_suffix(rotary, &suffix_k, suffix_old_start, suffix_new_start, suffix_len)?;
            (Tensor::cat(&[&prefix_k, &suffix_k], 2)?.contiguous()?, Tensor::cat(&[&prefix_v, &suffix_v], 2)?.contiguous()?)
        } else {
            (prefix_k.contiguous()?, prefix_v.contiguous()?)
        };
        *k = new_k;
        *v = new_v;
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    const HEAD_DIM: usize = 8;
    const MAX_POS: usize = 64;

    fn rotary() -> RotaryEmbedding {
        RotaryEmbedding::new(HEAD_DIM, MAX_POS, 10_000.0, &Device::Cpu).unwrap()
    }

    /// Ground truth: encode `raw` at `start..start+len` using
    /// `candle_nn::rotary_emb::rope` — the same already-trusted fused
    /// kernel `Qwen3Model`'s own attention uses (via `rope_thd`, same
    /// math, different tensor layout). Used to build test fixtures and as
    /// an independent check on this module's hand-rolled cos/sin math, so
    /// the test isn't just checking the implementation against itself.
    fn encode(raw: &Tensor, start: usize, len: usize) -> Tensor {
        let (cos, sin) = rotary().forward(start, len).unwrap();
        let cos = cos.to_dtype(raw.dtype()).unwrap();
        let sin = sin.to_dtype(raw.dtype()).unwrap();
        candle_nn::rotary_emb::rope(&raw.contiguous().unwrap(), &cos, &sin).unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
    }

    /// One distinct raw (pre-RoPE) key vector per position, `[1, 1, n, D]`.
    fn distinct_raw_keys(n: usize) -> Tensor {
        let data: Vec<f32> = (0..n * HEAD_DIM).map(|i| ((i as f32) * 0.037).sin()).collect();
        Tensor::from_vec(data, (1, 1, n, HEAD_DIM), &Device::Cpu).unwrap()
    }

    #[test]
    fn realign_to_same_position_is_identity() {
        let raw = distinct_raw_keys(5);
        let encoded = encode(&raw, 7, 5);
        let realigned = realign_rotary_suffix(&rotary(), &encoded, 7, 7, 5).unwrap();
        assert!(max_abs_diff(&encoded, &realigned) < 1e-5, "realigning to the same start position must be a no-op");
    }

    #[test]
    fn realign_matches_direct_encode_at_new_position() {
        // The real correctness property `realign_rotary_suffix` claims:
        // un-rotating from old_start then re-rotating at new_start must
        // equal directly encoding the same raw content at new_start.
        let raw = distinct_raw_keys(5);
        let old_start = 20;
        let new_start = 3;
        let old_encoded = encode(&raw, old_start, 5);
        let realigned = realign_rotary_suffix(&rotary(), &old_encoded, old_start, new_start, 5).unwrap();
        let expected = encode(&raw, new_start, 5);
        let diff = max_abs_diff(&realigned, &expected);
        assert!(diff < 1e-4, "realigned keys should match direct re-encoding at the new position, max abs diff {diff}");
    }

    #[test]
    fn realign_round_trip_returns_to_original() {
        // realign(realign(x, a->b), b->a) should return to x.
        let raw = distinct_raw_keys(6);
        let encoded = encode(&raw, 12, 6);
        let there = realign_rotary_suffix(&rotary(), &encoded, 12, 40, 6).unwrap();
        let back = realign_rotary_suffix(&rotary(), &there, 40, 12, 6).unwrap();
        let diff = max_abs_diff(&encoded, &back);
        assert!(diff < 1e-4, "a round-trip realignment should return to the original, max abs diff {diff}");
    }

    #[test]
    fn realign_empty_suffix_is_noop() {
        let raw = distinct_raw_keys(3);
        let encoded = encode(&raw, 0, 3);
        let realigned = realign_rotary_suffix(&rotary(), &encoded, 5, 2, 0).unwrap();
        assert!(max_abs_diff(&encoded, &realigned) < 1e-6, "zero-length realignment should return input unchanged");
    }

    #[test]
    fn drop_tokens_shrinks_cache_and_preserves_prefix() {
        // Layout: 6 cached positions, preserve the first 2 (system
        // prompt), drop 2 (the oldest unit), 2 remain (the newest unit) ->
        // final length 4.
        let raw = distinct_raw_keys(6);
        let k = encode(&raw, 0, 6);
        let v = distinct_raw_keys(6); // values aren't RoPE'd; arbitrary content is fine
        let mut caches: Vec<Option<(Tensor, Tensor)>> = vec![Some((k.clone(), v.clone())), Some((k.clone(), v.clone()))];

        let dropped = drop_tokens_from_cache(&mut caches, 2, 2, &rotary()).unwrap();
        assert!(dropped, "should report a successful drop");

        for cache in &caches {
            let (new_k, new_v) = cache.as_ref().unwrap();
            assert_eq!(new_k.dims(), &[1, 1, 4, HEAD_DIM], "K should shrink from 6 to 4 positions");
            assert_eq!(new_v.dims(), &[1, 1, 4, HEAD_DIM], "V should shrink from 6 to 4 positions");

            // Preserved prefix (original positions 0..2) must be untouched.
            let prefix_before = k.narrow(2, 0, 2).unwrap();
            let prefix_after = new_k.narrow(2, 0, 2).unwrap();
            assert!(max_abs_diff(&prefix_before, &prefix_after) < 1e-6, "preserved prefix keys must be unchanged");
            let v_prefix_before = v.narrow(2, 0, 2).unwrap();
            let v_prefix_after = new_v.narrow(2, 0, 2).unwrap();
            assert!(max_abs_diff(&v_prefix_before, &v_prefix_after) < 1e-6, "preserved prefix values must be unchanged");

            // Realigned suffix (originally positions 4..6, now 2..4) must
            // match directly encoding the same raw content at the new
            // positions.
            let raw_suffix = raw.narrow(2, 4, 2).unwrap();
            let expected_suffix_k = encode(&raw_suffix, 2, 2);
            let actual_suffix_k = new_k.narrow(2, 2, 2).unwrap();
            let diff = max_abs_diff(&actual_suffix_k, &expected_suffix_k);
            assert!(diff < 1e-4, "realigned suffix keys should match direct re-encoding, max abs diff {diff}");

            // Values are just carried over positionally, no RoPE involved.
            let v_suffix_before = v.narrow(2, 4, 2).unwrap();
            let v_suffix_after = new_v.narrow(2, 2, 2).unwrap();
            assert!(max_abs_diff(&v_suffix_before, &v_suffix_after) < 1e-6, "suffix values should be carried over unchanged (not RoPE'd)");
        }
    }

    #[test]
    fn drop_tokens_refuses_when_not_enough_available() {
        let k = distinct_raw_keys(4);
        let v = distinct_raw_keys(4);
        let mut caches: Vec<Option<(Tensor, Tensor)>> = vec![Some((k.clone(), v.clone()))];
        // Only 4 - preserve(1) = 3 available, asking to drop 10 should refuse.
        let dropped = drop_tokens_from_cache(&mut caches, 10, 1, &rotary()).unwrap();
        assert!(!dropped, "should refuse to drop more than is available");
        assert_eq!(caches[0].as_ref().unwrap().0.dim(2).unwrap(), 4, "cache must be left untouched on refusal");
    }

    #[test]
    fn drop_tokens_zero_length_is_noop() {
        let k = distinct_raw_keys(4);
        let v = distinct_raw_keys(4);
        let mut caches: Vec<Option<(Tensor, Tensor)>> = vec![Some((k, v))];
        let dropped = drop_tokens_from_cache(&mut caches, 0, 1, &rotary()).unwrap();
        assert!(!dropped);
    }

    #[test]
    fn drop_tokens_empty_suffix_still_shrinks() {
        // Drop exactly the tail (nothing survives after preserve+drop).
        let raw = distinct_raw_keys(4);
        let k = encode(&raw, 0, 4);
        let v = distinct_raw_keys(4);
        let mut caches: Vec<Option<(Tensor, Tensor)>> = vec![Some((k.clone(), v.clone()))];
        let dropped = drop_tokens_from_cache(&mut caches, 2, 2, &rotary()).unwrap();
        assert!(dropped);
        let (new_k, new_v) = caches[0].as_ref().unwrap();
        assert_eq!(new_k.dim(2).unwrap(), 2);
        let diff = max_abs_diff(new_k, &k.narrow(2, 0, 2).unwrap());
        assert!(diff < 1e-6, "with an empty suffix the result should be exactly the preserved prefix");
        let vdiff = max_abs_diff(new_v, &v.narrow(2, 0, 2).unwrap());
        assert!(vdiff < 1e-6);
    }

    #[test]
    fn rotate_half_matches_hand_computed_value() {
        // x = [1,2,3,4] -> split [1,2] | [3,4] -> rotate_half = [-3,-4, 1, 2]
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 1, 1, 4), &Device::Cpu).unwrap();
        let out = rotate_half(&x).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(out, vec![-3.0, -4.0, 1.0, 2.0]);
    }

    #[test]
    fn full_width_cos_sin_has_expected_shape_and_dtype() {
        let (cos, sin) = full_width_cos_sin(&rotary(), 0, 3, DType::F32).unwrap();
        assert_eq!(cos.dims(), &[1, 1, 3, HEAD_DIM]);
        assert_eq!(sin.dims(), &[1, 1, 3, HEAD_DIM]);
    }
}
