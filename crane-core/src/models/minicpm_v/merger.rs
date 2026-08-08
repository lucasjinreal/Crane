//! Hierarchical downsample merger — projects the vision tower's output into
//! the LLM's embedding space.
//!
//! Ported from HF's `MiniCPMV4_6DownsampleMLP` / `MiniCPMV4_6Merger`. Runs
//! `merger_times` rounds of "concat a 2x2 patch neighborhood -> MLP"; only
//! the last round projects to the LLM's hidden size, earlier rounds (if any)
//! stay at the vision tower's hidden size. Combined with the vision tower's
//! own mid-stack window merger (`vision::WindowAttentionMerger`, a 4x
//! reduction), `merger_times=1` here gives the checkpoint's default 16x
//! total token compression (`downsample_mode: "16x"`).

use candle_core::{Module, Result, Tensor};
use candle_nn::{layer_norm, linear, Activation, LayerNorm, Linear, VarBuilder};

use super::config::Config;

struct DownsampleMlp {
    pre_norm: LayerNorm,
    linear_1: Linear,
    linear_2: Linear,
}

impl DownsampleMlp {
    /// `in_hidden`: the (single-token, pre-concat) hidden size entering this
    /// round; the module concatenates a `merge_h*merge_w` neighborhood, so
    /// its actual input width is `in_hidden * merge_h * merge_w`.
    fn new(in_hidden: usize, merge_h: usize, merge_w: usize, out_hidden: usize, vb: VarBuilder) -> Result<Self> {
        let merged = in_hidden * merge_h * merge_w;
        Ok(Self {
            pre_norm: layer_norm(merged, 1e-6, vb.pp("pre_norm"))?,
            linear_1: linear(merged, merged, vb.pp("linear_1"))?,
            linear_2: linear(merged, out_hidden, vb.pp("linear_2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.pre_norm.forward(xs)?;
        let xs = self.linear_1.forward(&xs)?;
        // HF uses plain `nn.GELU()` here (exact erf-based), unlike the vision
        // tower's `gelu_pytorch_tanh` — `Activation::Gelu` maps to
        // `Tensor::gelu_erf()` in candle_nn.
        let xs = Activation::Gelu.forward(&xs)?;
        self.linear_2.forward(&xs)
    }
}

pub struct Merger {
    mlp: Vec<DownsampleMlp>,
    merge_h: usize,
    merge_w: usize,
    merger_times: usize,
}

impl Merger {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let (merge_h, merge_w) = cfg.merge_kernel_size;
        let vision_hidden = cfg.vision_config.hidden_size;
        let llm_hidden = cfg.text_config.hidden_size;
        let merger_times = cfg.merger_times.max(1);

        let vb_mlp = vb.pp("mlp");
        let mut mlp = Vec::with_capacity(merger_times);
        for i in 0..merger_times - 1 {
            mlp.push(DownsampleMlp::new(vision_hidden, merge_h, merge_w, vision_hidden, vb_mlp.pp(i))?);
        }
        mlp.push(DownsampleMlp::new(
            vision_hidden,
            merge_h,
            merge_w,
            llm_hidden,
            vb_mlp.pp(merger_times - 1),
        )?);

        Ok(Self { mlp, merge_h, merge_w, merger_times })
    }

    /// `xs`: `[total_patches, vision_hidden]` (vision tower output, no batch
    /// dim). `target_sizes`: per-image patch-grid dims at the vision tower's
    /// output resolution (already accounts for the mid-stack window merger's
    /// 4x reduction).
    ///
    /// Returns the flat, fully-merged image token sequence
    /// `[total_image_tokens, llm_hidden]`, concatenated across images in
    /// order — ready to splice into the LLM's input embeddings.
    pub fn forward(&self, xs: &Tensor, target_sizes: &[(usize, usize)]) -> Result<Tensor> {
        let embed_dim = xs.dim(1)?;
        let mut start = 0usize;
        let mut per_image = Vec::with_capacity(target_sizes.len());

        for &(height, width) in target_sizes {
            let num_patches = height * width;
            let patch = xs.narrow(0, start, num_patches)?;
            start += num_patches;

            let (mut cur_h, mut cur_w) = (height, width);
            let mut hidden = self.spatial_concat(&patch, cur_h, cur_w, embed_dim)?;
            hidden = self.mlp[0].forward(&hidden)?;
            cur_h /= self.merge_h;
            cur_w /= self.merge_w;

            for round in &self.mlp[1..self.merger_times] {
                let inner_dim = hidden.dim(1)?;
                hidden = self.spatial_concat(&hidden, cur_h, cur_w, inner_dim)?;
                hidden = round.forward(&hidden)?;
                cur_h /= self.merge_h;
                cur_w /= self.merge_w;
            }

            per_image.push(hidden);
        }

        Tensor::cat(&per_image, 0)
    }

    /// `[h*w, dim]` -> `[(h/mh)*(w/mw), mh*mw*dim]`: concat each `mh x mw`
    /// spatial neighborhood into one row, row-major over (h/mh, w/mw).
    fn spatial_concat(&self, xs: &Tensor, h: usize, w: usize, dim: usize) -> Result<Tensor> {
        let (mh, mw) = (self.merge_h, self.merge_w);
        let (merged_h, merged_w) = (h / mh, w / mw);
        xs.reshape((merged_h, mh, merged_w, mw, dim))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .reshape((merged_h * merged_w, mh * mw * dim))
    }
}
