//! Input projections for a Gated Delta Net layer.
//!
//! Two linear projections from hidden_size:
//! 1. `in_proj_qkv` projects to `[Q | K | V]` (concatenated, length `conv_dim`).
//! 2. `in_proj_z` projects to the gate `Z` (length `value_dim`).
//! 3. `in_proj_b` projects to the per-head write strength `β` (length
//!    `num_v_heads`).
//! 4. `in_proj_a` projects to the per-head A-log (length `num_v_heads`).
//!
//! Qwen 3.5 ships the weights as four separate matrices (`Split`); other hybrid
//! architectures may fuse Q/K/V/Z into one matrix and B/A into another
//! (`Grouped`). Both layouts are supported; the dispatch is in
//! [`GdnInputProjection::forward`].

use candle_core::quantized::GgmlDType;
use candle_core::{Module, Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::ops::linear::{linear_layer, LinearLayer};

/// Whether the source checkpoint stores projections as 4 separate matrices
/// (`Split`, used by Qwen 3.5) or 2 fused matrices (`Grouped`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GdnInputProjectionKind {
    /// `in_proj_qkvz` and `in_proj_ba` — two fused matrices.
    Grouped,
    /// `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` — four matrices.
    Split,
}

/// Holds the four (or two, in `Grouped` mode) linear projections.
pub enum GdnInputProjection {
    Grouped {
        in_proj_qkvz: LinearLayer,
        in_proj_ba: LinearLayer,
    },
    Split {
        in_proj_qkv: LinearLayer,
        in_proj_z: LinearLayer,
        in_proj_b: LinearLayer,
        in_proj_a: LinearLayer,
    },
}

impl GdnInputProjection {
    pub fn load(
        vb: VarBuilder,
        dims: &super::config::GdnDims,
        kind: GdnInputProjectionKind,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        match kind {
            GdnInputProjectionKind::Grouped => {
                let in_proj_qkvz = linear_layer(
                    dims.hidden_size,
                    dims.qkvz_out_dim(),
                    vb.pp("in_proj_qkvz"),
                    quant,
                )?;
                // The fused B/A matrix is tiny and feeds the numerically
                // sensitive β/decay path — never quantized (same as `Split`).
                let in_proj_ba = linear_layer(
                    dims.hidden_size,
                    dims.ba_out_dim(),
                    vb.pp("in_proj_ba"),
                    None,
                )?;
                Ok(Self::Grouped {
                    in_proj_qkvz,
                    in_proj_ba,
                })
            }
            GdnInputProjectionKind::Split => {
                let in_proj_qkv =
                    linear_layer(dims.hidden_size, dims.conv_dim, vb.pp("in_proj_qkv"), quant)?;
                let in_proj_z =
                    linear_layer(dims.hidden_size, dims.value_dim, vb.pp("in_proj_z"), quant)?;
                // B and A are tiny (`hidden → num_v_heads`) and feed the
                // numerically sensitive β/decay path — never quantized.
                let in_proj_b =
                    linear_layer(dims.hidden_size, dims.num_v_heads, vb.pp("in_proj_b"), None)?;
                let in_proj_a =
                    linear_layer(dims.hidden_size, dims.num_v_heads, vb.pp("in_proj_a"), None)?;
                Ok(Self::Split {
                    in_proj_qkv,
                    in_proj_z,
                    in_proj_b,
                    in_proj_a,
                })
            }
        }
    }

    /// Project `x: [B, S, H]` through all four projections and slice into the
    /// per-tensor Q, K, V, Z, B, A views needed by the recurrence.
    pub fn forward(
        &self,
        x: &Tensor,
        dims: &super::config::GdnDims,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<GdnProjection> {
        match self {
            Self::Grouped {
                in_proj_qkvz,
                in_proj_ba,
            } => GdnProjection::from_grouped(
                in_proj_qkvz.forward(x)?,
                in_proj_ba.forward(x)?,
                dims,
                batch_size,
                seq_len,
            ),
            Self::Split {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            } => GdnProjection::from_split(
                in_proj_qkv.forward(x)?,
                in_proj_z.forward(x)?,
                in_proj_b.forward(x)?,
                in_proj_a.forward(x)?,
                dims,
                batch_size,
                seq_len,
            ),
        }
    }
}

/// Sliced views produced by [`GdnInputProjection::forward`].
///
/// Shapes:
/// - `q, k`: `[B, S, num_k_heads, head_k_dim]`
/// - `v, z`: `[B, S, num_v_heads, head_v_dim]`
/// - `b, a`: `[B, S, num_v_heads]`
pub struct GdnProjection {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub z: Tensor,
    pub b: Tensor,
    pub a: Tensor,
}

impl GdnProjection {
    fn from_grouped(
        mixed_qkvz: Tensor,
        mixed_ba: Tensor,
        dims: &super::config::GdnDims,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let group_size_qkvz = 2 * dims.head_k_dim + 2 * dims.v_per_group * dims.head_v_dim;
        let mixed_qkvz = mixed_qkvz.reshape((batch_size, seq_len, dims.num_k_heads, group_size_qkvz))?;
        let mixed_ba = mixed_ba.reshape((batch_size, seq_len, dims.num_k_heads, 2 * dims.v_per_group))?;

        let mut offset = 0;
        let q = mixed_qkvz.narrow(D::Minus1, offset, dims.head_k_dim)?;
        offset += dims.head_k_dim;
        let k = mixed_qkvz.narrow(D::Minus1, offset, dims.head_k_dim)?;
        offset += dims.head_k_dim;
        let v = mixed_qkvz.narrow(D::Minus1, offset, dims.v_per_group * dims.head_v_dim)?;
        offset += dims.v_per_group * dims.head_v_dim;
        let z = mixed_qkvz.narrow(D::Minus1, offset, dims.v_per_group * dims.head_v_dim)?;

        let b = mixed_ba.narrow(D::Minus1, 0, dims.v_per_group)?;
        let a = mixed_ba.narrow(D::Minus1, dims.v_per_group, dims.v_per_group)?;

        Ok(Self {
            q,
            k,
            v: v.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?,
            z: z.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?,
            b: b.reshape((batch_size, seq_len, dims.num_v_heads))?,
            a: a.reshape((batch_size, seq_len, dims.num_v_heads))?,
        })
    }

    fn from_split(
        mixed_qkv: Tensor,
        mixed_z: Tensor,
        mixed_b: Tensor,
        mixed_a: Tensor,
        dims: &super::config::GdnDims,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let q = mixed_qkv.narrow(D::Minus1, 0, dims.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, dims.key_dim, dims.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, dims.key_dim * 2, dims.value_dim)?;

        Ok(Self {
            q: q.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?,
            k: k.reshape((batch_size, seq_len, dims.num_k_heads, dims.head_k_dim))?,
            v: v.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?,
            z: mixed_z.reshape((batch_size, seq_len, dims.num_v_heads, dims.head_v_dim))?,
            b: mixed_b.reshape((batch_size, seq_len, dims.num_v_heads))?,
            a: mixed_a.reshape((batch_size, seq_len, dims.num_v_heads))?,
        })
    }

    /// Reassemble the Q|K|V channels into a single `[B, S, conv_dim]` tensor —
    /// the input the causal Conv1D expects.
    pub fn conv_input(
        &self,
        dims: &super::config::GdnDims,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let q = self.q.reshape((batch_size, seq_len, dims.key_dim))?;
        let k = self.k.reshape((batch_size, seq_len, dims.key_dim))?;
        let v = self.v.reshape((batch_size, seq_len, dims.value_dim))?;
        Tensor::cat(&[&q, &k, &v], D::Minus1)
    }
}