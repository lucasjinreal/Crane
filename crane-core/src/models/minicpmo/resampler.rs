//! MiniCPM-o-4.5's perceiver-style `Resampler` — compresses a variable
//! number of vision-tower patch embeddings down to a fixed `num_queries`
//! (64) tokens via cross-attention against learnable query tokens.
//!
//! Ported from `modeling_minicpmo.py`'s `Resampler` (`nn.MultiheadAttention`
//! + 2D sincos position embeddings on the key/value side). This is a
//! genuinely different compression strategy than MiniCPM-V-4.6's
//! `crate::models::minicpm_v::merger` (concat + MLP downsampling, no
//! attention, no learnable queries) — no reuse there. The cross-attention
//! primitive here (separate query/key-value sources, no causal mask, no
//! GQA) also has no counterpart in `crate::models::modules::attention`
//! (self-attention only), so it's implemented directly against
//! `nn.MultiheadAttention`'s packed `in_proj_weight` layout.

use std::io::{Read, Seek};

use candle_core::{DType, Result, Tensor};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};

use crate::models::hunyuan_dense::modeling::Gguf;

/// `nn.MultiheadAttention` equivalent: query and key/value come from
/// different tensors (both already projected to `embed_dim`), packed
/// `in_proj_weight`/`in_proj_bias` of shape `[3*embed_dim, embed_dim]` /
/// `[3*embed_dim]` split into equal q/k/v thirds.
struct CrossAttention {
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    scale: f64,
}

impl CrossAttention {
    fn new(embed_dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let in_proj_weight = vb.get((3 * embed_dim, embed_dim), "in_proj_weight")?;
        let in_proj_bias = vb.get(3 * embed_dim, "in_proj_bias")?;
        let out_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("out_proj"))?;
        let head_dim = embed_dim / num_heads;
        Ok(Self {
            in_proj_weight,
            in_proj_bias,
            out_proj,
            num_heads,
            head_dim,
            embed_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// GGUF equivalent of [`Self::new`] — llama.cpp's `clip.cpp` exports
    /// `nn.MultiheadAttention`'s packed `in_proj_weight`/`in_proj_bias` as
    /// three **separate** `resampler.attn.{q,k,v}.{weight,bias}` tensors
    /// instead; concatenated back into the packed layout this struct's
    /// `proj()` expects, so nothing else about `CrossAttention` needs to
    /// change for the GGUF path.
    fn from_gguf<R: Read + Seek>(gg: &mut Gguf<R>, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let q_w = gg.dequant_tensor("resampler.attn.q.weight")?;
        let k_w = gg.dequant_tensor("resampler.attn.k.weight")?;
        let v_w = gg.dequant_tensor("resampler.attn.v.weight")?;
        let in_proj_weight = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;
        let q_b = gg.dequant_tensor("resampler.attn.q.bias")?.flatten_all()?;
        let k_b = gg.dequant_tensor("resampler.attn.k.bias")?.flatten_all()?;
        let v_b = gg.dequant_tensor("resampler.attn.v.bias")?.flatten_all()?;
        let in_proj_bias = Tensor::cat(&[&q_b, &k_b, &v_b], 0)?;
        let out_proj = Linear::new(gg.dequant_tensor("resampler.attn.out.weight")?, Some(gg.dequant_tensor("resampler.attn.out.bias")?.flatten_all()?));
        let head_dim = embed_dim / num_heads;
        Ok(Self { in_proj_weight, in_proj_bias, out_proj, num_heads, head_dim, embed_dim, scale: 1.0 / (head_dim as f64).sqrt() })
    }

    fn proj(&self, xs: &Tensor, third: usize) -> Result<Tensor> {
        let w = self.in_proj_weight.narrow(0, third * self.embed_dim, self.embed_dim)?;
        let b = self.in_proj_bias.narrow(0, third * self.embed_dim, self.embed_dim)?;
        Linear::new(w, Some(b)).forward(xs)
    }

    /// `query`: `[q_len, batch, embed_dim]`. `key`/`value`: `[kv_len, batch, embed_dim]`
    /// (query and key differ by the caller's added position embedding;
    /// value doesn't get one — matches `self.attn(q, x + pos_embed, x, ...)`).
    /// `key_padding_mask`: `[batch, kv_len]`, `true` at padded (masked-out) positions.
    ///
    /// Returns `[q_len, batch, embed_dim]`.
    fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor, key_padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let (q_len, batch, _) = query.dims3()?;
        let (kv_len, _, _) = key.dims3()?;

        let q = self.proj(query, 0)?; // [q_len, batch, embed_dim]
        let k = self.proj(key, 1)?;
        let v = self.proj(value, 2)?;

        // [L, B, embed_dim] -> [B, heads, L, head_dim]
        let reshape_heads = |t: &Tensor, len: usize| -> Result<Tensor> {
            t.reshape((len, batch * self.num_heads, self.head_dim))?
                .transpose(0, 1)? // [B*heads, len, head_dim]
                .contiguous()
        };
        let q = reshape_heads(&q, q_len)?;
        let k = reshape_heads(&k, kv_len)?;
        let v = reshape_heads(&v, kv_len)?;

        let attn_weights = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * self.scale)?; // [B*heads, q_len, kv_len]

        let attn_weights = match key_padding_mask {
            Some(mask) => {
                // [B, kv_len] bool -> additive [B, 1, 1, kv_len] -> broadcast over heads/q_len,
                // then reshape to match the flattened B*heads batch dim.
                let mask = mask
                    .to_dtype(DType::F32)?
                    .affine(f64::from(f32::MIN), 0.0)? // true(1.0) -> f32::MIN, false(0.0) -> 0.0
                    .reshape((batch, 1, 1, kv_len))?
                    .broadcast_as((batch, self.num_heads, 1, kv_len))?
                    .reshape((batch * self.num_heads, 1, kv_len))?
                    .to_dtype(attn_weights.dtype())?;
                attn_weights.broadcast_add(&mask)?
            }
            None => attn_weights,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32)?)?.to_dtype(q.dtype())?;
        let attn_output = attn_weights.matmul(&v)?; // [B*heads, q_len, head_dim]

        let attn_output = attn_output
            .transpose(0, 1)? // [q_len, B*heads, head_dim]
            .contiguous()?
            .reshape((q_len, batch, self.embed_dim))?;
        self.out_proj.forward(&attn_output)
    }
}

/// 2D sincos position embedding for a `(h, w)` patch grid, `[h*w, embed_dim]`.
///
/// Ported exactly from `get_2d_sincos_pos_embed`/`_from_grid`/
/// `get_1d_sincos_pos_embed_from_grid_new` in `modeling_minicpmo.py`,
/// including its (seemingly swapped but load-bearing) axis naming: the
/// first `embed_dim/2` channels encode the **w** coordinate, the second
/// half the **h** coordinate — `np.meshgrid(grid_w, grid_h)` puts the
/// w-index in `grid[0]` and the code feeds `grid[0]` into what it calls
/// `emb_h`.
fn sincos_pos_embed_2d(embed_dim: usize, h: usize, w: usize, device: &candle_core::Device) -> Result<Tensor> {
    let quarter = embed_dim / 4;
    let mut omega = vec![0f32; quarter];
    for (i, o) in omega.iter_mut().enumerate() {
        let e = i as f32 / quarter as f32;
        *o = 1.0 / 10000f32.powf(e);
    }

    let sincos_1d = |pos: f32| -> Vec<f32> {
        let mut out = Vec::with_capacity(2 * quarter);
        for &o in &omega {
            out.push((pos * o).sin());
        }
        for &o in &omega {
            out.push((pos * o).cos());
        }
        out
    };

    let mut data = Vec::with_capacity(h * w * embed_dim);
    for hi in 0..h {
        for wi in 0..w {
            // "emb_h" half built from the w-coordinate, "emb_w" half from h — see doc comment.
            data.extend(sincos_1d(wi as f32));
            data.extend(sincos_1d(hi as f32));
        }
    }
    Tensor::from_vec(data, (h * w, embed_dim), device)
}

pub struct Resampler {
    query: Tensor,
    kv_proj: Option<Linear>,
    attn: CrossAttention,
    ln_q: LayerNorm,
    ln_kv: LayerNorm,
    ln_post: LayerNorm,
    proj: Tensor,
    embed_dim: usize,
}

impl Resampler {
    pub fn new(num_queries: usize, embed_dim: usize, num_heads: usize, kv_dim: usize, vb: VarBuilder) -> Result<Self> {
        let query = vb.get((num_queries, embed_dim), "query")?;
        let kv_proj = if kv_dim != embed_dim {
            Some(candle_nn::linear_no_bias(kv_dim, embed_dim, vb.pp("kv_proj"))?)
        } else {
            None
        };
        let attn = CrossAttention::new(embed_dim, num_heads, vb.pp("attn"))?;
        let ln_q = layer_norm(embed_dim, 1e-6, vb.pp("ln_q"))?;
        let ln_kv = layer_norm(embed_dim, 1e-6, vb.pp("ln_kv"))?;
        let ln_post = layer_norm(embed_dim, 1e-6, vb.pp("ln_post"))?;
        let proj = vb.get((embed_dim, embed_dim), "proj")?;
        Ok(Self { query, kv_proj, attn, ln_q, ln_kv, ln_post, proj, embed_dim })
    }

    /// GGUF equivalent of [`Self::new`] — loads from the same standalone
    /// `MiniCPM-o-4_5-vision-F16.gguf`-style file as
    /// [`super::vision::VisionModel::from_gguf`] (the resampler's tensors
    /// are bundled in the same file), so this takes the same already-open
    /// `Gguf` reader rather than a separate file path.
    ///
    /// Deliberately **not** read from GGUF: `resampler.pos_embed_k`, a
    /// precomputed 2D sincos position-embedding table baked in at
    /// conversion time — this port already computes the identical formula
    /// analytically at each forward call ([`sincos_pos_embed_2d`], already
    /// HF-diff validated), so the baked table is redundant rather than
    /// something this loader needs to consume.
    ///
    /// `resampler.proj.weight` needs an explicit transpose that nothing
    /// else in this file does — confirmed by directly diffing the raw
    /// dequantized GGUF tensor against the known-correct safetensors
    /// parameter (`proj` is a bare `nn.Parameter` used as `x @ proj` in
    /// the original Python, not a real `nn.Linear`, so it was never in the
    /// `(out_features, in_features)` convention the GGUF exporter
    /// otherwise normalizes everything to) — transposed, the two tensors'
    /// values match to float32 rounding (~3e-8 max abs diff); untransposed
    /// they differ substantially (~0.37).
    ///
    /// # Errors
    ///
    /// Returns an error if a required tensor is missing or has an
    /// unexpected shape.
    pub fn from_gguf<R: Read + Seek>(gg: &mut Gguf<R>, num_heads: usize, kv_dim: usize, embed_dim: usize) -> Result<Self> {
        let query = gg.dequant_tensor("resampler.query")?;
        let kv_proj = if kv_dim != embed_dim { Some(Linear::new(gg.dequant_tensor("resampler.kv.weight")?, None)) } else { None };
        let attn = CrossAttention::from_gguf(gg, embed_dim, num_heads)?;
        let ln_q = LayerNorm::new(gg.dequant_tensor("resampler.ln_q.weight")?, gg.dequant_tensor("resampler.ln_q.bias")?, 1e-6);
        let ln_kv = LayerNorm::new(gg.dequant_tensor("resampler.ln_kv.weight")?, gg.dequant_tensor("resampler.ln_kv.bias")?, 1e-6);
        let ln_post = LayerNorm::new(gg.dequant_tensor("resampler.ln_post.weight")?, gg.dequant_tensor("resampler.ln_post.bias")?, 1e-6);
        let proj = gg.dequant_tensor("resampler.proj.weight")?.t()?.contiguous()?;
        Ok(Self { query, kv_proj, attn, ln_q, ln_kv, ln_post, proj, embed_dim })
    }

    /// `x`: `[batch, max_patches, kv_dim]` (padded vision-tower output).
    /// `tgt_sizes`: true `(h, w)` patch-grid dims per batch item.
    ///
    /// Returns `[batch, num_queries, embed_dim]`.
    pub fn forward(&self, x: &Tensor, tgt_sizes: &[(usize, usize)]) -> Result<Tensor> {
        let (batch, max_patches, _kv_dim) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();

        let x = match &self.kv_proj {
            Some(p) => p.forward(x)?,
            None => x.clone(),
        };
        let x = self.ln_kv.forward(&x)?; // [batch, max_patches, embed_dim]

        // Per-item 2D sincos position embedding, padded to max_patches with
        // zeros (padding positions get masked out of attention anyway).
        let mut pos_embed_data = vec![0f32; batch * max_patches * self.embed_dim];
        for (b, &(h, w)) in tgt_sizes.iter().enumerate() {
            let pe = sincos_pos_embed_2d(self.embed_dim, h, w, device)?.to_dtype(DType::F32)?;
            let pe: Vec<f32> = pe.flatten_all()?.to_vec1()?;
            let dst_start = b * max_patches * self.embed_dim;
            pos_embed_data[dst_start..dst_start + h * w * self.embed_dim].copy_from_slice(&pe);
        }
        let pos_embed = Tensor::from_vec(pos_embed_data, (batch, max_patches, self.embed_dim), device)?.to_dtype(dtype)?;

        let key_padding_mask = if tgt_sizes.iter().any(|&(h, w)| h * w != max_patches) {
            let mut mask_data = vec![0u8; batch * max_patches];
            for (b, &(h, w)) in tgt_sizes.iter().enumerate() {
                for p in (h * w)..max_patches {
                    mask_data[b * max_patches + p] = 1;
                }
            }
            Some(Tensor::from_vec(mask_data, (batch, max_patches), device)?)
        } else {
            None
        };

        // [batch, L, D] -> [L, batch, D] (nn.MultiheadAttention default layout).
        let kv = x.transpose(0, 1)?.contiguous()?;
        let kv_with_pos = (&kv + &pos_embed.transpose(0, 1)?.contiguous()?)?;

        let q = self.ln_q.forward(&self.query)?; // [num_queries, embed_dim]
        let q = q.unsqueeze(1)?.broadcast_as((q.dim(0)?, batch, self.embed_dim))?.contiguous()?;

        let out = self.attn.forward(&q, &kv_with_pos, &kv, key_padding_mask.as_ref())?; // [num_queries, batch, embed_dim]
        let out = out.transpose(0, 1)?.contiguous()?; // [batch, num_queries, embed_dim]

        let out = self.ln_post.forward(&out)?;
        out.broadcast_matmul(&self.proj)
    }
}
