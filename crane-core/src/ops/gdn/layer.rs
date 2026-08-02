//! Top-level `GatedDeltaNet` layer: orchestrates input projection, causal
//! Conv1D, gated delta rule recurrence, gated RMSNorm, and output projection.

use candle_core::quantized::GgmlDType;
use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

use super::backend::{apply_recurrence, compute_beta_g, l2_norm};
use super::conv::causal_conv1d;
use super::cache::GdnLayerCache;
use super::config::{GdnConfig, GdnDims, VHeadOrder};
use super::norm::RmsNormGated;
use super::projection::{GdnInputProjection, GdnInputProjectionKind};
use crate::ops::linear::{linear_layer, LinearLayer};

/// One linear-attention layer as used by Qwen 3.5 (and similar hybrid models).
///
/// Field names follow the HF safetensors layout (`linear_attn.in_proj_qkv.weight`,
/// `linear_attn.conv1d.weight`, `linear_attn.A_log`, …).
pub struct GatedDeltaNet {
    pub input_proj: GdnInputProjection,
    pub conv1d_weight: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
    pub norm: RmsNormGated,
    pub out_proj: LinearLayer,
}

impl GatedDeltaNet {
/// Load weights from `vb.pp("linear_attn")`.
///
/// `cfg` supplies the dimensions; `projection_kind` selects the QKV/Z/B/A
/// weight layout (Qwen 3.5 uses [`GdnInputProjectionKind::Split`]).
/// `quant` requests in-situ quantization of the large projections
/// (conv1d / dt_bias / A_log / norm always stay in full precision).
pub fn load(
        vb: VarBuilder,
        cfg: &dyn GdnConfig,
        projection_kind: GdnInputProjectionKind,
        quant: Option<GgmlDType>,
) -> Result<Self> {
        let dims = GdnDims::new(cfg);
        let vb_la = vb.pp("linear_attn");

        let input_proj = GdnInputProjection::load(vb_la.clone(), &dims, projection_kind, quant)?;
        let conv1d_weight = vb_la.get(
            (dims.conv_dim, 1, dims.conv_kernel_size),
            "conv1d.weight",
        )?;
        let dt_bias = vb_la.get(dims.num_v_heads, "dt_bias")?;
        let a_log = vb_la.get(dims.num_v_heads, "A_log")?;

        let norm = RmsNormGated::new(dims.head_v_dim, cfg.rms_norm_eps(), vb_la.pp("norm"))?;
        let out_proj = linear_layer(dims.value_dim, dims.hidden_size, vb_la.pp("out_proj"), quant)?;

        Ok(Self {
            input_proj,
            conv1d_weight,
            dt_bias,
            a_log,
            norm,
            out_proj,
        })
    }

    /// Forward pass over a sequence: a whole prompt, one prefill chunk, or a
    /// single decode step — all continue from `cache`, which is updated in
    /// place.
    ///
    /// `x: [B, S, hidden_size]`. Returns `[B, S, hidden_size]`.
    pub fn forward(
        &self,
        x: &Tensor,
        dims: &GdnDims,
        cache: &mut GdnLayerCache,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();

        // 1. Input projections → Q, K, V, Z, B, A.
        let projected = self.input_proj.forward(x, dims, batch_size, seq_len)?;

        // 2. Causal Conv1D over [Q|K|V].
        let mixed_qkv = projected.conv_input(dims, batch_size, seq_len)?;
        let mixed_qkv = causal_conv1d(&mixed_qkv, &self.conv1d_weight, dims, cache)?;

        // 3. Split → per-head Q, K, V; expand K from num_k_heads → num_v_heads.
        let (q, k, v) = split_qkv(&mixed_qkv, dims, batch_size, seq_len)?;
        let (q, k) = repeat_kv_heads(&q, &k, dims)?;

        // 4. L2-normalize Q and K (Qwen 3.5 uses QK-norm = L2).
        let q = l2_norm(&q, 1e-6)?;
        let k = l2_norm(&k, 1e-6)?;

        // 5. Compute β (write strength) and g (decay) from B, A, A_log, dt_bias.
        let (beta, g) =
            compute_beta_g(&projected.b, &projected.a, &self.a_log, &self.dt_bias, dtype)?;

        // 6. Run the gated delta rule recurrence.
        let y = apply_recurrence(
            &q, &k, &v, &g, &beta, dims, batch_size, seq_len, cache, dtype,
        )?;

        // 7. Gated RMSNorm + output projection.
        self.finish_forward(y, projected.z, batch_size, seq_len)
    }

    fn finish_forward(
        &self,
        y: Tensor,
        z: Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        // y comes out of the recurrence as `[B, S, num_v_heads, head_v_dim]`.
        // Flatten the head + head_v dims, gated-RMSNorm along head_v, then
        // reshape back to `[B, S, value_dim]`.
        let head_v_dim = self.norm_head_dim();
        let value_dim = y.dim(2)? * head_v_dim;
        let y_2d = y.reshape(((), head_v_dim))?;
        let z_2d = z.reshape(((), head_v_dim))?;
        let y_2d = self.norm.forward(&y_2d, &z_2d)?;
        let y = y_2d.reshape((batch_size, seq_len, value_dim))?;
        self.out_proj.forward(&y)
    }

    fn norm_head_dim(&self) -> usize {
        // out_proj.weight shape is [value_dim, hidden_size]; head_v_dim is
        // stored indirectly as value_dim / num_v_heads. We recover it from
        // the norm's RmsNorm weight length.
        self.norm.weight_len()
    }
}

/// Split the conv1d output `[B, S, conv_dim]` into Q, K, V along the last
/// dim, then reshape each into `[B, S, num_heads, head_dim]`.
fn split_qkv(
    mixed_qkv: &Tensor,
    dims: &GdnDims,
    batch_size: usize,
    seq_len: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    use candle_core::D;
    let q = mixed_qkv.narrow(D::Minus1, 0, dims.key_dim)?;
    let k = mixed_qkv.narrow(D::Minus1, dims.key_dim, dims.key_dim)?;
    let v = mixed_qkv.narrow(D::Minus1, dims.key_dim * 2, dims.value_dim)?;
    let q = q.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?;
    let k = k.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?;
    let v = v.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?;
    Ok((q, k, v))
}

/// Expand K (and Q, for symmetry) from `num_k_heads` to `num_v_heads` by
/// repeating each key head `v_per_group` times (GQA-style expansion for
/// linear attention).
///
/// The expansion must place each key head's replicas at the same indices the
/// checkpoint's value heads occupy, so it follows `dims.v_head_order`:
/// `Interleaved` (HF) repeats along a trailing axis, putting a key head's
/// replicas next to each other; `Chunked` (llama.cpp GGUF) repeats along a
/// leading axis, laying out one full pass of key heads per replica. Both
/// produce the same multiset of rows — only the pairing differs.
///
/// When `num_k_heads == num_v_heads` (`v_per_group == 1`) this is a no-op and
/// the two orders coincide.
fn repeat_kv_heads(q: &Tensor, k: &Tensor, dims: &GdnDims) -> Result<(Tensor, Tensor)> {
    if dims.v_per_group == 1 {
        return Ok((q.clone(), k.clone()));
    }
    let expand = |t: &Tensor| -> Result<Tensor> {
        let (b, s) = (t.dim(0)?, t.dim(1)?);
        let repeated = match dims.v_head_order {
            VHeadOrder::Interleaved => t.unsqueeze(3)?.repeat((1, 1, 1, dims.v_per_group, 1))?,
            VHeadOrder::Chunked => t.unsqueeze(2)?.repeat((1, 1, dims.v_per_group, 1, 1))?,
        };
        repeated
            .contiguous()?
            .reshape((b, s, dims.num_v_heads, dims.head_k_dim))
    };
    Ok((expand(q)?, expand(k)?))
}

// ─────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// `num_k_heads=2, v_per_group=2` -> 4 value heads. Each key head must end
    /// up at exactly the indices its value heads occupy, which differs per
    /// order. Tagging each key head with a distinct value makes the resulting
    /// pairing directly readable.
    fn dims_2x2(order: VHeadOrder) -> GdnDims {
        GdnDims {
            hidden_size: 8,
            num_k_heads: 2,
            num_v_heads: 4,
            head_k_dim: 1,
            head_v_dim: 1,
            conv_kernel_size: 4,
            key_dim: 2,
            value_dim: 4,
            conv_dim: 8,
            v_per_group: 2,
            v_head_order: order,
        }
    }

    fn expanded(order: VHeadOrder) -> Vec<f32> {
        // [B=1, S=1, num_k_heads=2, head_k_dim=1]: key head 0 -> 10, head 1 -> 20
        let t = Tensor::new(&[[[[10.0f32], [20.0f32]]]], &Device::Cpu).unwrap();
        let dims = dims_2x2(order);
        let (q, k) = repeat_kv_heads(&t, &t, &dims).unwrap();
        assert_eq!(q.dims(), &[1, 1, 4, 1]);
        assert_eq!(k.dims(), &[1, 1, 4, 1]);
        q.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// HF: a key head's replicas are adjacent.
    #[test]
    fn repeat_kv_heads_interleaved_groups_replicas_together() {
        assert_eq!(expanded(VHeadOrder::Interleaved), vec![10.0, 10.0, 20.0, 20.0]);
    }

    /// llama.cpp GGUF: one full pass over key heads per replica. Mixing this
    /// up with the interleaved order is the Qwen3.5-GGUF value-head bug.
    #[test]
    fn repeat_kv_heads_chunked_repeats_whole_key_head_pass() {
        assert_eq!(expanded(VHeadOrder::Chunked), vec![10.0, 20.0, 10.0, 20.0]);
    }

    /// Ratios other than 2 must follow the same rule. Qwen3.5-27B is
    /// `16 key / 48 value` heads (v_per_group == 3) and the 122B/397B are
    /// `16 / 64` (== 4), so the formulas must not assume 2 or a power of two.
    /// Modelled here at `2 key / 6 value` heads.
    #[test]
    fn repeat_kv_heads_generalizes_beyond_v_per_group_2() {
        let t = Tensor::new(&[[[[10.0f32], [20.0f32]]]], &Device::Cpu).unwrap();
        let base = GdnDims {
            num_v_heads: 6,
            value_dim: 6,
            v_per_group: 3,
            ..dims_2x2(VHeadOrder::Interleaved)
        };

        // key_head * v_per_group + replica
        let (q, _) = repeat_kv_heads(&t, &t, &base).unwrap();
        assert_eq!(
            q.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
        );

        // replica * num_k_heads + key_head
        let chunked = GdnDims { v_head_order: VHeadOrder::Chunked, ..base };
        let (q, _) = repeat_kv_heads(&t, &t, &chunked).unwrap();
        assert_eq!(
            q.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0]
        );
    }

    /// With `v_per_group == 1` the two orders coincide and expansion is a no-op.
    #[test]
    fn repeat_kv_heads_is_noop_without_grouping() {
        let t = Tensor::new(&[[[[10.0f32], [20.0f32]]]], &Device::Cpu).unwrap();
        for order in [VHeadOrder::Interleaved, VHeadOrder::Chunked] {
            let mut dims = dims_2x2(order);
            dims.num_v_heads = 2;
            dims.value_dim = 2;
            dims.v_per_group = 1;
            let (q, _) = repeat_kv_heads(&t, &t, &dims).unwrap();
            assert_eq!(q.flatten_all().unwrap().to_vec1::<f32>().unwrap(), vec![10.0, 20.0]);
        }
    }
}
