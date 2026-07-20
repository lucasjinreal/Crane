//! Pre-allocated KV cache buffer shared by [`super::attention::GqaAttention`]
//! and the qwen3-specific `Attention`.
//!
//! Backs the cache with a buffer sized `seq_len + ROOM` positions and writes
//! new K/V in place via `slice_set` — `O(new_seq_len)` per decode step instead
//! of an `O(cache_len)` `Tensor::cat` on every call. Falls back to `cat` +
//! reallocate only once the buffer's room is exhausted.

use candle_core::{Result, Tensor};

/// Headroom (in positions) added when (re)allocating, to amortize growth.
const ROOM: usize = 256;

/// Result of [`update_kv_cache`].
pub(crate) struct KvCacheUpdate {
    /// The (possibly reallocated) `(buf_k, buf_v)` buffer to store back.
    pub buffer: (Tensor, Tensor),
    /// Number of valid (filled) positions now in `buffer`.
    pub seq_len: usize,
    /// View over all cached K positions in the compute dtype.
    pub k: Tensor,
    /// View over all cached V positions in the compute dtype.
    pub v: Tensor,
}

/// Append `k`/`v` to a pre-allocated KV cache buffer.
///
/// `kv_cache` is the caller's current `(buf_k, buf_v)` buffer (`None` on the
/// first call) and `cache_seq_len` is the number of valid positions already
/// written into it. Returns the (possibly reallocated) buffer to store back,
/// the new valid length, and views over the full `k`/`v` spanning all cached
/// positions in the compute dtype.
///
/// # Errors
///
/// Returns a candle error if any tensor operation (`contiguous`, `slice_set`,
/// `narrow`, `cat`, `zeros`) fails.
pub(crate) fn update_kv_cache(
    kv_cache: Option<(Tensor, Tensor)>,
    cache_seq_len: usize,
    k: &Tensor,
    v: &Tensor,
) -> Result<KvCacheUpdate> {
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let new_seq_len = k.dim(2)?;

    let Some((buf_k, buf_v)) = kv_cache else {
        // First use — allocate buffer with extra room.
        let (batch, heads, seq, dim) = k.dims4()?;
        let buf_k = Tensor::zeros((batch, heads, seq + ROOM, dim), k.dtype(), k.device())?;
        let buf_v = Tensor::zeros((batch, heads, seq + ROOM, dim), v.dtype(), v.device())?;
        buf_k.slice_set(&k, 2, 0)?;
        buf_v.slice_set(&v, 2, 0)?;
        return Ok(KvCacheUpdate {
            buffer: (buf_k, buf_v),
            seq_len: seq,
            k,
            v,
        });
    };

    let buf_len = buf_k.dim(2)?;
    let new_total = cache_seq_len + new_seq_len;

    if new_total <= buf_len {
        // In-place write — O(new_seq_len).
        buf_k.slice_set(&k, 2, cache_seq_len)?;
        buf_v.slice_set(&v, 2, cache_seq_len)?;
        let k_view = buf_k.narrow(2, 0, new_total)?;
        let v_view = buf_v.narrow(2, 0, new_total)?;
        Ok(KvCacheUpdate {
            buffer: (buf_k, buf_v),
            seq_len: new_total,
            k: k_view,
            v: v_view,
        })
    } else {
        // Buffer overflow — grow with extra room.
        let cur_k = buf_k.narrow(2, 0, cache_seq_len)?;
        let cur_v = buf_v.narrow(2, 0, cache_seq_len)?;
        drop(buf_k);
        drop(buf_v);
        let full_k = Tensor::cat(&[&cur_k, &k], 2)?;
        let full_v = Tensor::cat(&[&cur_v, &v], 2)?;
        drop(cur_k);
        drop(cur_v);
        let total = full_k.dim(2)?;
        let (batch, heads, _, dim) = full_k.dims4()?;
        let new_buf_k = Tensor::zeros((batch, heads, total + ROOM, dim), k.dtype(), k.device())?;
        let new_buf_v = Tensor::zeros((batch, heads, total + ROOM, dim), v.dtype(), v.device())?;
        new_buf_k.slice_set(&full_k, 2, 0)?;
        new_buf_v.slice_set(&full_v, 2, 0)?;
        Ok(KvCacheUpdate {
            buffer: (new_buf_k, new_buf_v),
            seq_len: total,
            k: full_k,
            v: full_v,
        })
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    /// Builds a tensor whose values are `offset + [0, 1, 2, ...]` — pass a
    /// distinct `offset` per call so tests can tell which call's data landed
    /// where, instead of every same-shape call producing identical content.
    fn bhsd(batch: usize, heads: usize, seq: usize, dim: usize, offset: f32) -> Tensor {
        let data: Vec<f32> = (0..batch * heads * seq * dim)
            .map(|i| i as f32 + offset)
            .collect();
        Tensor::from_vec(data, (batch, heads, seq, dim), &Device::Cpu).expect("bhsd tensor")
    }

    #[test]
    fn first_call_allocates_buffer_with_room() {
        let (batch, heads, seq, dim) = (1, 2, 3, 4);
        let k = bhsd(batch, heads, seq, dim, 0.0);
        let v = bhsd(batch, heads, seq, dim, 0.0);

        let update = update_kv_cache(None, 0, &k, &v).expect("first call");

        assert_eq!(update.seq_len, seq);
        assert_eq!(update.buffer.0.dims(), &[batch, heads, seq + ROOM, dim]);
        assert_eq!(update.buffer.1.dims(), &[batch, heads, seq + ROOM, dim]);
        assert_eq!(update.k.dims(), k.dims());
        assert_eq!(update.v.dims(), v.dims());
    }

    #[test]
    fn in_place_write_appends_within_room() {
        let (batch, heads, dim) = (1, 2, 4);
        let prefill = bhsd(batch, heads, 3, dim, 0.0);
        let prefill_update =
            update_kv_cache(None, 0, &prefill, &prefill).expect("prefill");

        let step = bhsd(batch, heads, 1, dim, 1000.0);
        let step_update = update_kv_cache(
            Some(prefill_update.buffer),
            prefill_update.seq_len,
            &step,
            &step,
        )
        .expect("decode step");

        assert_eq!(step_update.seq_len, prefill_update.seq_len + 1);
        // Same buffer identity (no reallocation): dims unchanged.
        assert_eq!(step_update.buffer.0.dims(), &[batch, heads, 3 + ROOM, dim]);
        assert_eq!(
            step_update.k.dims(),
            &[batch, heads, step_update.seq_len, dim]
        );
        // The appended step must be readable back from the cache view.
        let appended = step_update
            .k
            .narrow(2, prefill_update.seq_len, 1)
            .expect("narrow appended");
        let diff = (&appended - &step)
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar::<f32>()
            .expect("scalar");
        assert!(diff < 1e-6, "appended slot must equal the written step");
    }

    #[test]
    fn overflow_reallocates_and_preserves_content() {
        let (batch, heads, dim) = (1, 1, 2);
        let prefill = bhsd(batch, heads, 1, dim, 0.0);
        let mut update = update_kv_cache(None, 0, &prefill, &prefill).expect("prefill");

        // Exhaust the buffer's room (allocated as 1 + ROOM), then push one more
        // to force the overflow/reallocation branch. Each step gets a distinct
        // offset so the final assertion can tell them apart.
        for i in 0..ROOM {
            let offset = (i as f32 + 1.0) * 1000.0;
            let step = bhsd(batch, heads, 1, dim, offset);
            update = update_kv_cache(Some(update.buffer), update.seq_len, &step, &step)
                .expect("fill room");
        }
        assert_eq!(update.seq_len, 1 + ROOM);

        let overflow_step = bhsd(batch, heads, 1, dim, (ROOM as f32 + 1.0) * 1000.0);
        let final_update = update_kv_cache(
            Some(update.buffer),
            update.seq_len,
            &overflow_step,
            &overflow_step,
        )
        .expect("overflow step");

        assert_eq!(final_update.seq_len, update.seq_len + 1);
        assert_eq!(
            final_update.buffer.0.dims(),
            &[batch, heads, final_update.seq_len + ROOM, dim]
        );
        assert_eq!(
            final_update.k.dims(),
            &[batch, heads, final_update.seq_len, dim]
        );
        // The original prefill value must survive the reallocation, at position 0.
        let first = final_update.k.narrow(2, 0, 1).expect("narrow first");
        let diff = (&first - &prefill)
            .expect("sub")
            .abs()
            .expect("abs")
            .max_all()
            .expect("max_all")
            .to_scalar::<f32>()
            .expect("scalar");
        assert!(diff < 1e-6, "prefill content must survive reallocation");
    }
}
