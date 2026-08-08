//! MiniCPM-V-4.6 vision tower.
//!
//! Ported from HF's native `transformers.models.minicpmv4_6.modeling_minicpmv4_6`
//! (`MiniCPMV4_6VisionEmbeddings`, `MiniCPMV4_6VisionEncoder`,
//! `MiniCPMV4_6ViTWindowAttentionMerger`, `MiniCPMV4_6VisionModel`).
//!
//! # Key departures from a "plain SigLIP ViT"
//!
//! - **NaViT packing.** All patches from every image/slice in a request are
//!   concatenated into one `[1, C, patch_size, total_patches]` "pixel_values"
//!   tensor (built by [`super::preprocess`]) and patch-embedded in a single
//!   `Conv2d(kernel=patch_size, stride=patch_size)` pass — the unusual shape
//!   makes each patch occupy its own `patch_size`-wide column, so the conv's
//!   single vertical slide computes every patch's embedding at once.
//! - **Position embeddings are NaViT-bucketized, not resized/interpolated.**
//!   Each image's `(h, w)` patch-grid is mapped onto the tower's native
//!   `num_patches_per_side` grid via `floor(fractional_coord * n_side)`
//!   bucketing (HF's `torch.bucketize(..., right=True)` on evenly-spaced
//!   boundaries reduces to exactly this — see [`VisionEmbeddings::forward`]).
//! - **Attention is blocked per image, always** (not just in the window
//!   merger) — every encoder layer runs full non-causal self-attention
//!   independently within each image/slice's contiguous span of the packed
//!   sequence, never across images. See [`blocked_self_attention`].
//! - **A window-attention merger runs mid-stack** (after encoder layer
//!   `insert_layer_id`, default 6): 2x2 patch windows get their own blocked
//!   self-attention pass, then a real spatial merge (concat the 4 window
//!   tokens -> MLP -> + mean-pooled residual) cuts the sequence to 1/4
//!   length. The remaining encoder layers then run on the shorter sequence.

use candle_core::{Module, Result, Tensor};
use candle_nn::{conv2d, embedding, layer_norm, linear, Activation, Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, VarBuilder};

use super::config::VisionConfig;

// ── Shared attention primitive ──────────────────────────────────────────

/// Standard (non-causal) multi-head self-attention with bias on every
/// projection — matches `MiniCPMV4_6VisionAttention`. Shared by
/// [`EncoderLayer`] and [`WindowAttentionMerger`] (same weight shapes,
/// different callers / different chunk boundaries at call time).
struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VisionAttention {
    fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            q_proj: linear(dim, dim, vb.pp("q_proj"))?,
            k_proj: linear(dim, dim, vb.pp("k_proj"))?,
            v_proj: linear(dim, dim, vb.pp("v_proj"))?,
            out_proj: linear(dim, dim, vb.pp("out_proj"))?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// Full (non-causal) self-attention over `xs` `[chunk_len, dim]` — a
    /// single block. No batch dim: callers operate at `B=1` throughout (NaViT
    /// packing puts everything in one sequence) and slice out chunks
    /// themselves; see [`blocked_self_attention`].
    fn forward_block(&self, xs: &Tensor) -> Result<Tensor> {
        let (seq_len, _dim) = xs.dims2()?;
        let q = self.q_proj.forward(xs)?
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .transpose(0, 1)?; // [heads, seq, head_dim]
        let k = self.k_proj.forward(xs)?
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .transpose(0, 1)?;
        let v = self.v_proj.forward(xs)?
            .reshape((seq_len, self.num_heads, self.head_dim))?
            .transpose(0, 1)?;

        let attn_weights = (q.contiguous()?.matmul(&k.transpose(1, 2)?.contiguous()?)? * self.scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?; // [heads, seq, head_dim]

        let attn_output = attn_output
            .transpose(0, 1)?
            .contiguous()?
            .reshape((seq_len, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&attn_output)
    }
}

/// Run `attn` independently within each `[start, end)` span of `xs`
/// `[total_len, dim]` (no batch dim), concatenating the results back in
/// order. Mirrors HF's "process each chunk separately" fallback in
/// `MiniCPMV4_6VisionAttention.forward` — attention never crosses a chunk
/// boundary, whether chunks are whole images (regular encoder layers) or
/// 2x2 windows (the window-attention merger).
fn blocked_self_attention(attn: &VisionAttention, xs: &Tensor, boundaries: &[usize]) -> Result<Tensor> {
    if boundaries.len() == 2 {
        // Single chunk spanning the whole sequence — skip the split/cat.
        return attn.forward_block(xs);
    }
    let mut outputs = Vec::with_capacity(boundaries.len() - 1);
    for w in boundaries.windows(2) {
        let (start, end) = (w[0], w[1]);
        let chunk = xs.narrow(0, start, end - start)?;
        outputs.push(attn.forward_block(&chunk)?);
    }
    Tensor::cat(&outputs, 0)
}

// ── MLP ──────────────────────────────────────────────────────────────────

struct Mlp {
    fc1: Linear,
    fc2: Linear,
    activation: Activation,
}

impl Mlp {
    fn new(dim: usize, intermediate: usize, activation: Activation, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(dim, intermediate, vb.pp("fc1"))?,
            fc2: linear(intermediate, dim, vb.pp("fc2"))?,
            activation,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = self.activation.forward(&xs)?;
        self.fc2.forward(&xs)
    }
}

// ── Embeddings ───────────────────────────────────────────────────────────

pub struct VisionEmbeddings {
    patch_embedding: Conv2d,
    position_embedding: Embedding,
    num_patches_per_side: usize,
}

impl VisionEmbeddings {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_embedding = conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            vb.pp("patch_embedding"),
        )?;
        let position_embedding = embedding(cfg.num_patches(), cfg.hidden_size, vb.pp("position_embedding"))?;
        Ok(Self {
            patch_embedding,
            position_embedding,
            num_patches_per_side: cfg.num_patches_per_side(),
        })
    }

    /// `pixel_values`: NaViT-packed `[1, C, patch_size, total_patches]`.
    /// `target_sizes`: `(h, w)` patch-grid dims for each image/slice packed
    /// into `pixel_values`, in order; `sum(h*w) == total_patches`.
    ///
    /// Returns `[1, total_patches, hidden_size]`.
    pub fn forward(&self, pixel_values: &Tensor, target_sizes: &[(usize, usize)]) -> Result<Tensor> {
        let patch_embeds = self.patch_embedding.forward(pixel_values)?; // [1, hidden, 1, total_patches]
        let (b, hidden, _one, total_patches) = patch_embeds.dims4()?;
        let patch_embeds = patch_embeds
            .reshape((b, hidden, total_patches))?
            .transpose(1, 2)?; // [1, total_patches, hidden]

        let n_side = self.num_patches_per_side;
        let mut pos_ids: Vec<u32> = Vec::with_capacity(total_patches);
        for &(nb_h, nb_w) in target_sizes {
            // floor(k/n * n_side) for k in 0..n, matching HF's
            // torch.bucketize(arange(0,1,1/n), boundaries, right=True) on
            // evenly-spaced boundaries at i/n_side.
            let bucket = |k: usize, n: usize| -> usize {
                let frac = k as f64 / n as f64;
                ((frac * n_side as f64).floor() as usize).min(n_side - 1)
            };
            for h in 0..nb_h {
                let bh = bucket(h, nb_h);
                for w in 0..nb_w {
                    let bw = bucket(w, nb_w);
                    pos_ids.push((bh * n_side + bw) as u32);
                }
            }
        }
        debug_assert_eq!(pos_ids.len(), total_patches);
        let pos_ids = Tensor::from_vec(pos_ids, total_patches, patch_embeds.device())?;
        let pos_embeds = self
            .position_embedding
            .forward(&pos_ids)?
            .unsqueeze(0)?
            .to_dtype(patch_embeds.dtype())?;

        patch_embeds.add(&pos_embeds)
    }
}

// ── Encoder layer ────────────────────────────────────────────────────────

pub struct EncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: VisionAttention,
    layer_norm2: LayerNorm,
    mlp: Mlp,
}

impl EncoderLayer {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            layer_norm1: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm1"))?,
            self_attn: VisionAttention::new(cfg.hidden_size, cfg.num_attention_heads, vb.pp("self_attn"))?,
            layer_norm2: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm2"))?,
            mlp: Mlp::new(cfg.hidden_size, cfg.intermediate_size, Activation::GeluPytorchTanh, vb.pp("mlp"))?,
        })
    }

    /// `xs`: `[total_patches, hidden]` (no batch dim). `boundaries`: cumulative
    /// per-image chunk boundaries (`[0, len_0, len_0+len_1, ...]`) — attention
    /// runs independently within each `[boundaries[i], boundaries[i+1])` span.
    fn forward(&self, xs: &Tensor, boundaries: &[usize]) -> Result<Tensor> {
        let residual = xs;
        let hidden = self.layer_norm1.forward(xs)?;
        let hidden = blocked_self_attention(&self.self_attn, &hidden, boundaries)?;
        let hidden = (residual + hidden)?;

        let residual = &hidden;
        let mlp_in = self.layer_norm2.forward(&hidden)?;
        let mlp_out = self.mlp.forward(&mlp_in)?;
        residual + mlp_out
    }
}

// ── Window-attention merger ─────────────────────────────────────────────

/// Windowed self-attention + 2x2 spatial merge, run once mid-stack.
/// Ported from `MiniCPMV4_6ViTWindowAttentionMerger` (`get_window_index` +
/// `forward`).
pub struct WindowAttentionMerger {
    layer_norm1: LayerNorm,
    self_attn: VisionAttention,
    pre_norm: LayerNorm,
    linear_1: Linear,
    linear_2: Linear,
    window_h: usize,
    window_w: usize,
}

impl WindowAttentionMerger {
    pub fn new(cfg: &VisionConfig, window_kernel_size: (usize, usize), vb: VarBuilder) -> Result<Self> {
        let (window_h, window_w) = window_kernel_size;
        let window_hidden_size = cfg.hidden_size * window_h * window_w;
        let window_intermediate_size = cfg.intermediate_size * window_h * window_w;
        Ok(Self {
            layer_norm1: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm1"))?,
            self_attn: VisionAttention::new(cfg.hidden_size, cfg.num_attention_heads, vb.pp("self_attn"))?,
            pre_norm: layer_norm(window_hidden_size, cfg.layer_norm_eps, vb.pp("pre_norm"))?,
            linear_1: linear(window_hidden_size, window_intermediate_size, vb.pp("linear_1"))?,
            linear_2: linear(window_intermediate_size, cfg.hidden_size, vb.pp("linear_2"))?,
            window_h,
            window_w,
        })
    }

    /// Global (packed-sequence) window index + per-window chunk boundaries.
    /// Pure integer arithmetic — direct port of `get_window_index`.
    ///
    /// Returns `(window_index, window_boundaries)`: `window_index[i]` is the
    /// original packed-sequence position that should occupy slot `i` after
    /// gathering (so each window's `window_h*window_w` slots are contiguous);
    /// `window_boundaries` are the resulting chunk boundaries, all spaced
    /// `window_h*window_w` apart.
    fn window_index(&self, target_sizes: &[(usize, usize)]) -> (Vec<u32>, Vec<usize>) {
        let (wh, ww) = (self.window_h, self.window_w);
        let mut window_index = Vec::new();
        let mut boundaries = vec![0usize];
        let mut token_offset = 0usize;

        for &(height, width) in target_sizes {
            let num_windows_h = height / wh;
            let num_windows_w = width / ww;
            for wh_i in 0..num_windows_h {
                for ww_i in 0..num_windows_w {
                    for wh_j in 0..wh {
                        for ww_j in 0..ww {
                            let orig = (wh_i * wh + wh_j) * width + (ww_i * ww + ww_j);
                            window_index.push((orig + token_offset) as u32);
                        }
                    }
                    boundaries.push(boundaries.last().unwrap() + wh * ww);
                }
            }
            token_offset += height * width;
        }
        (window_index, boundaries)
    }

    /// `xs`: `[total_patches, hidden]`. `target_sizes`: per-image patch-grid
    /// dims (must all be divisible by the window size). `image_boundaries`:
    /// per-image cumulative boundaries in the ORIGINAL (unwindowed) order.
    ///
    /// Returns `[total_patches/4, hidden]` and the halved `target_sizes` for
    /// the caller to use downstream (image dims are exactly `/2` in each
    /// spatial axis after a 2x2 merge).
    pub fn forward(
        &self,
        xs: &Tensor,
        target_sizes: &[(usize, usize)],
        image_boundaries: &[usize],
    ) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let device = xs.device();
        let residual = xs;
        let hidden = self.layer_norm1.forward(xs)?;

        let (window_index, window_boundaries) = self.window_index(target_sizes);
        let window_index_t = Tensor::from_vec(window_index.clone(), window_index.len(), device)?;
        let windowed = hidden.index_select(&window_index_t, 0)?.contiguous()?;
        let attended = blocked_self_attention(&self.self_attn, &windowed, &window_boundaries)?;

        // Undo the windowing gather: scatter `attended[i]` back to its
        // original position `window_index[i]` (argsort(window_index)).
        let mut inverse = vec![0u32; window_index.len()];
        for (slot, &orig) in window_index.iter().enumerate() {
            inverse[orig as usize] = slot as u32;
        }
        let inverse_t = Tensor::from_vec(inverse, window_index.len(), device)?;
        let unwindowed = attended.index_select(&inverse_t, 0)?;

        let hidden = (residual + unwindowed)?.contiguous()?;

        // Per-image 2x2 spatial merge (pure reshape/permute — no attention).
        let (wh, ww) = (self.window_h, self.window_w);
        let embed_dim = hidden.dim(1)?;
        let mut merged_chunks = Vec::with_capacity(target_sizes.len());
        let mut new_target_sizes = Vec::with_capacity(target_sizes.len());
        for (img_idx, &(height, width)) in target_sizes.iter().enumerate() {
            let start = image_boundaries[img_idx];
            let end = image_boundaries[img_idx + 1];
            let patch = hidden.narrow(0, start, end - start)?; // [h*w, embed_dim]

            let merged_h = height / wh;
            let merged_w = width / ww;
            // [h, w, e] -> [merged_h, wh, merged_w, ww, e] -> permute(0,2,1,3,4)
            let patch_5d = patch
                .reshape((merged_h, wh, merged_w, ww, embed_dim))?
                .permute((0, 2, 1, 3, 4))?
                .contiguous()?; // [merged_h, merged_w, wh, ww, e]

            let flat = patch_5d.reshape((merged_h * merged_w, wh * ww * embed_dim))?;
            let pooled = patch_5d
                .reshape((merged_h * merged_w, wh * ww, embed_dim))?
                .mean(1)?; // [num_windows, e]

            let hidden_state = self.pre_norm.forward(&flat)?;
            let hidden_state = self.linear_1.forward(&hidden_state)?;
            // `Activation::GeluPytorchTanh` maps to `Tensor::gelu()` (tanh
            // approximation) in candle_nn; `gelu_erf()` is the exact variant
            // used by `Activation::Gelu` (see `merger::DownsampleMLP`, which
            // needs exact GELU instead — HF uses plain `nn.GELU()` there).
            let hidden_state = hidden_state.gelu()?;
            let hidden_state = self.linear_2.forward(&hidden_state)?;
            merged_chunks.push((hidden_state + pooled)?);
            new_target_sizes.push((merged_h, merged_w));
        }

        Ok((Tensor::cat(&merged_chunks, 0)?, new_target_sizes))
    }
}

// ── Vision model ─────────────────────────────────────────────────────────

pub struct VisionModel {
    embeddings: VisionEmbeddings,
    layers: Vec<EncoderLayer>,
    vit_merger: WindowAttentionMerger,
    post_layernorm: LayerNorm,
    insert_layer_id: usize,
}

impl VisionModel {
    pub fn new(cfg: &VisionConfig, insert_layer_id: usize, window_kernel_size: (usize, usize), vb: VarBuilder) -> Result<Self> {
        let embeddings = VisionEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("encoder").pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::new(cfg, vb_layers.pp(i))?);
        }
        let vit_merger = WindowAttentionMerger::new(cfg, window_kernel_size, vb.pp("vit_merger"))?;
        let post_layernorm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm"))?;
        Ok(Self {
            embeddings,
            layers,
            vit_merger,
            post_layernorm,
            insert_layer_id,
        })
    }

    fn image_boundaries(target_sizes: &[(usize, usize)]) -> Vec<usize> {
        let mut boundaries = vec![0usize];
        for &(h, w) in target_sizes {
            boundaries.push(boundaries.last().unwrap() + h * w);
        }
        boundaries
    }

    /// `pixel_values`: NaViT-packed `[1, C, patch_size, total_patches]`.
    /// `target_sizes`: per-image/-slice `(h, w)` patch-grid dims.
    ///
    /// Returns `(last_hidden_state, downsampled_target_sizes)` — the second
    /// element accounts for the 2x2 merge inside `vit_merger` (each axis
    /// halved), needed by the caller ([`super::merger::Merger`]) to know the
    /// new per-image chunk boundaries.
    pub fn forward(&self, pixel_values: &Tensor, target_sizes: &[(usize, usize)]) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let embeds = self.embeddings.forward(pixel_values, target_sizes)?;
        let (_b, _s, hidden) = embeds.dims3()?;
        let mut xs = embeds.reshape(((), hidden))?; // drop batch dim: [total_patches, hidden]

        let mut boundaries = Self::image_boundaries(target_sizes);
        let mut cur_target_sizes = target_sizes.to_vec();

        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs, &boundaries)?;
            if i == self.insert_layer_id {
                let (merged, new_sizes) = self.vit_merger.forward(&xs, &cur_target_sizes, &boundaries)?;
                xs = merged;
                cur_target_sizes = new_sizes;
                boundaries = Self::image_boundaries(&cur_target_sizes);
            }
        }

        let xs = self.post_layernorm.forward(&xs)?;
        Ok((xs, cur_target_sizes))
    }
}
