use std::f64;

use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{
    Activation, Embedding, LayerNorm, LayerNormConfig, Module, VarBuilder, embedding, layer_norm,
};

use super::config::VisionConfig;

// Preserve the original BF16 biased GEMM epilogue and reduction algorithm.
// The CPU/F32 paths retain Candle's portable linear implementation.
type Linear = crate::ops::biased_linear::BiasedLinear;
fn linear(input: usize, output: usize, vb: VarBuilder) -> Result<Linear> {
    Linear::load(input, output, vb)
}
struct PatchEmbed {
    proj: Tensor,
    bias: Tensor,
    in_channels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    hidden_size: usize,
}

impl PatchEmbed {
    fn new(cfg: &VisionConfig, vb: &VarBuilder) -> Result<Self> {
        let proj_vb = vb.pp("proj");
        // Each incoming row is exactly one non-overlapping 3D patch.
        // A single GEMM preserves Conv3d's F32 accumulation across both
        // temporal slices, avoiding two independently rounded BF16 Conv2ds.
        let proj = proj_vb
            .get(
                (
                    cfg.hidden_size,
                    cfg.in_channels,
                    cfg.temporal_patch_size,
                    cfg.patch_size,
                    cfg.patch_size,
                ),
                "weight",
            )?
            .reshape((cfg.hidden_size, ()))?;
        let bias = proj_vb.get(cfg.hidden_size, "bias")?;
        Ok(Self {
            proj,
            bias,
            in_channels: cfg.in_channels,
            patch_size: cfg.patch_size,
            temporal_patch_size: cfg.temporal_patch_size,
            hidden_size: cfg.hidden_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let width = self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size;
        let dtype = self.bias.dtype();
        let xs = xs.reshape(((), width))?.to_dtype(dtype)?;
        let xs = xs.matmul(&self.proj.t()?)?;
        // ATen adds Conv3d bias after cuDNN rounds the convolution output.
        xs.to_dtype(dtype)?.broadcast_add(&self.bias.unsqueeze(0)?)
    }
}

struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
}

impl VisionMlp {
    fn new(dim: usize, hidden_dim: usize, act: Activation, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(dim, hidden_dim, vb.pp("linear_fc1"))?,
            fc2: linear(hidden_dim, dim, vb.pp("linear_fc2"))?,
            act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = xs
            .to_dtype(DType::F32)?
            .apply(&self.act)?
            .to_dtype(xs.dtype())?;
        self.fc2.forward(&xs)
    }
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cos = cos.unsqueeze(D::Minus2)?;
    let sin = sin.unsqueeze(D::Minus2)?;

    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

struct VisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(dim: usize, num_heads: usize, vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: linear(dim, dim * 3, vb.pp("qkv"))?,
            proj: linear(dim, dim, vb.pp("proj"))?,
            num_heads,
            head_dim: dim / num_heads,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        let hidden_states = self.qkv.forward(xs)?;
        let qkv = hidden_states
            .reshape((seq_len, 3, self.num_heads, self.head_dim))?
            .permute((1, 0, 2, 3))?;
        let mut q = qkv.i(0)?.squeeze(0)?;
        let mut k = qkv.i(1)?.squeeze(0)?;
        let mut v = qkv.i(2)?.squeeze(0)?;

        let cos = cos.to_dtype(DType::F32)?;
        let sin = sin.to_dtype(DType::F32)?;
        q = q.to_dtype(DType::F32)?;
        k = k.to_dtype(DType::F32)?;
        v = v.to_dtype(DType::F32)?;
        (q, k) = apply_rotary_pos_emb_vision(&q, &k, &cos, &sin)?;
        // HF rotates in F32, then rounds Q/K back before attention.
        q = q.to_dtype(xs.dtype())?.to_dtype(DType::F32)?;
        k = k.to_dtype(xs.dtype())?.to_dtype(DType::F32)?;

        let mut outputs = Vec::new();
        for window in cu_seqlens.windows(2) {
            let start = window[0];
            let end = window[1];
            if end <= start {
                continue;
            }
            let len = end - start;
            let q_chunk = q.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let k_chunk = k.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let v_chunk = v.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;

            let mut chunk_out = {
                let q = q_chunk.unsqueeze(0)?;
                let k = k_chunk.unsqueeze(0)?;
                let v = v_chunk.unsqueeze(0)?;
                let scale = 1. / (self.head_dim as f64).sqrt();
                #[cfg(feature = "flash-attn")]
                let flash =
                    if xs.device().is_cuda() && matches!(xs.dtype(), DType::BF16 | DType::F16) {
                        Some(
                            candle_flash_attn::flash_attn(
                                &q.to_dtype(xs.dtype())?.transpose(1, 2)?.contiguous()?,
                                &k.to_dtype(xs.dtype())?.transpose(1, 2)?.contiguous()?,
                                &v.to_dtype(xs.dtype())?.transpose(1, 2)?.contiguous()?,
                                scale as f32,
                                false,
                            )?
                            .transpose(1, 2)?
                            .to_dtype(DType::F32)?,
                        )
                    } else {
                        None
                    };
                #[cfg(not(feature = "flash-attn"))]
                let flash: Option<Tensor> = None;
                match flash {
                    Some(output) => output,
                    None => {
                        let weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
                        candle_nn::ops::softmax_last_dim(&weights)?.matmul(&v)?
                    },
                }
            };
            chunk_out = chunk_out.squeeze(0)?.transpose(0, 1)?;

            // Tensor dependencies are ordered on Candle's CUDA stream.
            // Synchronize only at the caller's measurement/response boundary.
            chunk_out = chunk_out.reshape((len, self.num_heads * self.head_dim))?;
            outputs.push(chunk_out.to_dtype(xs.dtype())?);
        }
        let attn_output = Tensor::cat(&outputs, 0)?;
        self.proj.forward(&attn_output)
    }
}

struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn new(cfg: &VisionConfig, vb: &VarBuilder) -> Result<Self> {
        let norm_cfg = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        let norm1 = layer_norm(cfg.hidden_size, norm_cfg, vb.pp("norm1"))?;
        let norm2 = layer_norm(cfg.hidden_size, norm_cfg, vb.pp("norm2"))?;
        let attn = VisionAttention::new(cfg.hidden_size, cfg.num_heads, &vb.pp("attn"))?;
        let mlp = VisionMlp::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.hidden_act.to_activation(),
            &vb.pp("mlp"),
        )?;
        Ok(Self {
            norm1,
            norm2,
            attn,
            mlp,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let normed = self.norm1.forward(xs)?;
        let attn_out = self.attn.forward(&normed, cu_seqlens, cos, sin)?;
        let xs_att = xs.add(&attn_out)?;
        let mlp_out = self.mlp.forward(&self.norm2.forward(&xs_att)?)?;
        xs_att.add(&mlp_out)
    }
}

struct PatchMerger {
    norm: LayerNorm,
    use_postshuffle_norm: bool,
    spatial_merge_unit: usize,
    merged_hidden_size: usize,
    fc1: Linear,
    fc2: Linear,
}

impl PatchMerger {
    fn new(cfg: &VisionConfig, use_postshuffle_norm: bool, vb: &VarBuilder) -> Result<Self> {
        let merged_hidden_size = cfg.hidden_size * cfg.spatial_merge_size.pow(2);
        let norm_dim = if use_postshuffle_norm {
            merged_hidden_size
        } else {
            cfg.hidden_size
        };
        let norm_cfg = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        Ok(Self {
            norm: layer_norm(norm_dim, norm_cfg, vb.pp("norm"))?,
            use_postshuffle_norm,
            spatial_merge_unit: cfg.spatial_merge_size.pow(2),
            merged_hidden_size,
            fc1: linear(merged_hidden_size, merged_hidden_size, vb.pp("linear_fc1"))?,
            fc2: linear(merged_hidden_size, cfg.out_hidden_size, vb.pp("linear_fc2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        if seq_len % self.spatial_merge_unit != 0 {
            candle_core::bail!(
                "Sequence length {} is not divisible by spatial merge unit {}",
                seq_len,
                self.spatial_merge_unit
            );
        }
        let grouped = seq_len / self.spatial_merge_unit;
        let norm_input = if self.use_postshuffle_norm {
            xs.reshape((grouped, self.merged_hidden_size))?
        } else {
            xs.clone()
        };
        let normed = self.norm.forward(&norm_input)?;
        let reshaped = if self.use_postshuffle_norm {
            normed
        } else {
            normed.reshape((grouped, self.merged_hidden_size))?
        };
        let xs = self.fc1.forward(&reshaped)?;
        // HF PatchMerger uses nn.GELU() (erf), independently of the
        // configurable tanh GELU used by the vision transformer MLPs.
        let xs = xs.to_dtype(DType::F32)?.gelu_erf()?.to_dtype(xs.dtype())?;
        self.fc2.forward(&xs)
    }
}

struct VisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl VisionRotaryEmbedding {
    const THETA: f32 = 10000.;

    fn new(dim: usize, device: &Device) -> Result<Self> {
        let dim_f32 = dim as f32;
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| {
                let i_f32 = i as f32;
                1f32 / Self::THETA.powf(i_f32 / dim_f32)
            })
            .collect::<Vec<_>>();
        let inv_freq_len = inv_freq.len();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?,
        })
    }

    fn make_embeds(&self, seqlen: usize) -> Result<Tensor> {
        let seqlen_f32 = seqlen as f32;
        let seq = Tensor::arange(0f32, seqlen_f32, self.inv_freq.device())?
            .to_dtype(self.inv_freq.dtype())?
            .unsqueeze(D::Minus1)?;
        seq.broadcast_matmul(&self.inv_freq)
    }
}

pub struct Qwen3_5VisionModel {
    patch_embed: PatchEmbed,
    pos_embed: Embedding,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
    deepstack_mergers: Vec<PatchMerger>,
    deepstack_lookup: Vec<Option<usize>>,
    rotary_pos_emb: VisionRotaryEmbedding,
    spatial_merge_size: usize,
    num_grid_per_side: usize,
    hidden_size: usize,
}

impl Qwen3_5VisionModel {
    /// # Errors
    ///
    /// Returns an error if a required weight tensor is missing or has an
    /// unexpected shape, or if `num_position_embeddings` is not a perfect square.
    pub fn new(cfg: &VisionConfig, vb: &VarBuilder) -> Result<Self> {
        let patch_embed = PatchEmbed::new(cfg, &vb.pp("patch_embed"))?;
        let pos_embed = embedding(
            cfg.num_position_embeddings,
            cfg.hidden_size,
            vb.pp("pos_embed"),
        )?;

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(VisionBlock::new(cfg, &vb.pp(format!("blocks.{i}")))?);
        }

        let merger = PatchMerger::new(cfg, false, &vb.pp("merger"))?;
        let deepstack_mergers = cfg
            .deepstack_visual_indexes
            .iter()
            .enumerate()
            .map(|(i, _)| PatchMerger::new(cfg, true, &vb.pp(format!("deepstack_merger_list.{i}"))))
            .collect::<Result<Vec<_>>>()?;

        let mut deepstack_lookup = vec![None; cfg.depth];
        for (idx, &layer_idx) in cfg.deepstack_visual_indexes.iter().enumerate() {
            if layer_idx < cfg.depth {
                deepstack_lookup[layer_idx] = Some(idx);
            }
        }

        let head_dim = cfg.hidden_size / cfg.num_heads;
        let rotary_pos_emb = VisionRotaryEmbedding::new(head_dim / 2, vb.device())?;

        // num_position_embeddings is a small ViT grid size (e.g. 32*32), so this
        // round-trip through f64 to take an integer sqrt is exact and non-negative.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let num_grid_per_side = (cfg.num_position_embeddings as f64).sqrt().round() as usize;
        if num_grid_per_side * num_grid_per_side != cfg.num_position_embeddings {
            candle_core::bail!(
                "num_position_embeddings {} is not a perfect square",
                cfg.num_position_embeddings
            );
        }

        Ok(Self {
            patch_embed,
            pos_embed,
            blocks,
            merger,
            deepstack_mergers,
            deepstack_lookup,
            rotary_pos_emb,
            spatial_merge_size: cfg.spatial_merge_size,
            num_grid_per_side,
            hidden_size: cfg.hidden_size,
        })
    }

    fn linspace_points(&self, steps: usize) -> Vec<f32> {
        if steps == 1 {
            return vec![0.0];
        }
        let max_val = (self.num_grid_per_side - 1) as f32;
        let step = max_val / (steps.saturating_sub(1)) as f32;
        (0..steps)
            .map(|i| {
                if i < steps / 2 {
                    i as f32 * step
                } else {
                    // CUDA linspace works backwards from the endpoint in the
                    // upper half and contracts the multiply/subtract.
                    (-step).mul_add((steps - 1 - i) as f32, max_val)
                }
            })
            .collect()
    }

    // This function's length comes from building four separate bilinear-
    // interpolation corners (floor/ceil x h/w) with their per-corner bounded
    // casts; splitting it up would scatter that construction logic across
    // several small functions without simplifying the interpolation itself.
    #[allow(clippy::too_many_lines)]
    fn fast_pos_embed_interpolate(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let device = self.pos_embed.embeddings().device();
        let dtype = self.pos_embed.embeddings().dtype();
        let grid = grid_thw.to_vec2::<u32>()?;

        let mut idx_lists: [Vec<i64>; 4] = Default::default();
        let mut weight_lists: [Vec<f32>; 4] = Default::default();
        let mut hw_lengths = Vec::with_capacity(grid.len());

        for g in &grid {
            let h = g[1] as usize;
            let w = g[2] as usize;
            hw_lengths.push(h * w);

            let h_vals = self.linspace_points(h);
            let w_vals = self.linspace_points(w);

            // h_vals/w_vals are non-negative linspace points bounded by
            // num_grid_per_side, so floor()/ceil() as usize never truncates
            // a fractional integer part or flips sign.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let h_floor: Vec<usize> = h_vals.iter().map(|v| v.floor() as usize).collect();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let w_floor: Vec<usize> = w_vals.iter().map(|v| v.floor() as usize).collect();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let h_ceil: Vec<usize> = h_vals
                .iter()
                .map(|v| (v.ceil() as usize).min(self.num_grid_per_side - 1))
                .collect();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let w_ceil: Vec<usize> = w_vals
                .iter()
                .map(|v| (v.ceil() as usize).min(self.num_grid_per_side - 1))
                .collect();
            let dh: Vec<f32> = h_vals
                .iter()
                .zip(&h_floor)
                .map(|(v, f)| v - *f as f32)
                .collect();
            let dw: Vec<f32> = w_vals
                .iter()
                .zip(&w_floor)
                .map(|(v, f)| v - *f as f32)
                .collect();

            for ((&hf, &hc), &dh_val) in h_floor.iter().zip(&h_ceil).zip(&dh) {
                for ((&wf, &wc), &dw_val) in w_floor.iter().zip(&w_ceil).zip(&dw) {
                    // Grid indices are bounded by num_grid_per_side^2, a small ViT
                    // grid size, so this cast to i64 (needed for index_select) cannot wrap.
                    #[allow(clippy::cast_possible_wrap)]
                    let base00 = (hf * self.num_grid_per_side + wf) as i64;
                    #[allow(clippy::cast_possible_wrap)]
                    let base01 = (hf * self.num_grid_per_side + wc) as i64;
                    #[allow(clippy::cast_possible_wrap)]
                    let base10 = (hc * self.num_grid_per_side + wf) as i64;
                    #[allow(clippy::cast_possible_wrap)]
                    let base11 = (hc * self.num_grid_per_side + wc) as i64;

                    let w00 = (1.0 - dh_val) * (1.0 - dw_val);
                    let w01 = (1.0 - dh_val) * dw_val;
                    let w10 = dh_val * (1.0 - dw_val);
                    let w11 = dh_val * dw_val;

                    idx_lists[0].push(base00);
                    idx_lists[1].push(base01);
                    idx_lists[2].push(base10);
                    idx_lists[3].push(base11);

                    weight_lists[0].push(w00);
                    weight_lists[1].push(w01);
                    weight_lists[2].push(w10);
                    weight_lists[3].push(w11);
                }
            }
        }

        let idx_tensors = idx_lists
            .iter()
            .map(|idxs| Tensor::from_vec(idxs.clone(), (idxs.len(),), device))
            .collect::<Result<Vec<_>>>()?;
        let idx_tensor = Tensor::stack(&idx_tensors, 0)?;

        let weight_tensors = weight_lists
            .iter()
            .map(|weights| Tensor::from_vec(weights.clone(), (weights.len(),), device))
            .collect::<Result<Vec<_>>>()?;
        let weight_tensor = Tensor::stack(&weight_tensors, 0)?;

        let pos_embeds = self.pos_embed.forward(&idx_tensor)?.to_dtype(DType::F32)?;
        let pos_embeds = pos_embeds.broadcast_mul(&weight_tensor.unsqueeze(D::Minus1)?)?;
        let pos_embeds = pos_embeds.sum(0)?;

        let mut splits = Vec::with_capacity(hw_lengths.len());
        let mut start = 0;
        for len in hw_lengths {
            splits.push(pos_embeds.narrow(0, start, len)?);
            start += len;
        }

        let mut permuted = Vec::with_capacity(grid.len());
        for (pos_embed, g) in splits.into_iter().zip(&grid) {
            let t = g[0] as usize;
            let h = g[1] as usize;
            let w = g[2] as usize;
            let pos_embed = pos_embed.repeat((t, 1))?;
            let pos_embed = pos_embed.reshape((
                t,
                h / self.spatial_merge_size,
                self.spatial_merge_size,
                w / self.spatial_merge_size,
                self.spatial_merge_size,
                self.hidden_size,
            ))?;
            let pos_embed = pos_embed
                .permute((0, 1, 3, 2, 4, 5))?
                .reshape((t * h * w, self.hidden_size))?;
            permuted.push(pos_embed);
        }

        Tensor::cat(&permuted, 0)?.to_dtype(dtype)
    }

    fn rot_pos_emb(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let device = self.rotary_pos_emb.inv_freq.device();
        let grid = grid_thw.to_vec2::<u32>()?;
        let max_hw = grid
            .iter()
            .flat_map(|v| v[1..3].iter())
            .copied()
            .max()
            .unwrap_or(0) as usize;
        let freq_table = self.rotary_pos_emb.make_embeds(max_hw)?;

        let mut coords: Vec<(i64, i64)> = Vec::new();
        for g in &grid {
            let h = g[1] as usize;
            let w = g[2] as usize;
            let merged_h = h / self.spatial_merge_size;
            let merged_w = w / self.spatial_merge_size;

            let mut base_coords: Vec<(i64, i64)> = Vec::with_capacity(h * w);
            for br in 0..merged_h {
                for bc in 0..merged_w {
                    for ir in 0..self.spatial_merge_size {
                        for ic in 0..self.spatial_merge_size {
                            // Grid row/col indices are bounded by the image/video
                            // grid size, so this cast to i64 (needed for
                            // index_select) cannot wrap.
                            #[allow(clippy::cast_possible_wrap)]
                            let row = (br * self.spatial_merge_size + ir) as i64;
                            #[allow(clippy::cast_possible_wrap)]
                            let col = (bc * self.spatial_merge_size + ic) as i64;
                            base_coords.push((row, col));
                        }
                    }
                }
            }

            for _ in 0..(g[0] as usize) {
                coords.extend(base_coords.iter().copied());
            }
        }

        let total_tokens = coords.len();
        let mut rows = Vec::with_capacity(total_tokens);
        let mut cols = Vec::with_capacity(total_tokens);
        for &(r, c) in &coords {
            rows.push(r);
            cols.push(c);
        }
        let rows = Tensor::from_vec(rows, (total_tokens,), device)?;
        let cols = Tensor::from_vec(cols, (total_tokens,), device)?;
        let row_embeds = freq_table.index_select(&rows, 0)?;
        let col_embeds = freq_table.index_select(&cols, 0)?;
        Tensor::stack(&[row_embeds, col_embeds], D::Minus2)?
            .reshape((total_tokens, freq_table.dim(D::Minus1)? * 2))
    }

    fn build_cu_seqlens(grid_thw: &Tensor) -> Result<Vec<usize>> {
        let grid = grid_thw.to_vec2::<u32>()?;
        let mut cu = Vec::with_capacity(grid.iter().map(|v| v[0] as usize).sum::<usize>() + 1);
        cu.push(0usize);
        let mut acc = 0usize;
        for g in &grid {
            let area = (g[1] * g[2]) as usize;
            for _ in 0..(g[0] as usize) {
                acc += area;
                cu.push(acc);
            }
        }
        Ok(cu)
    }

    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward(&self, xs: &Tensor, grid_thw: &Tensor) -> Result<(Tensor, Vec<Tensor>)> {
        self.forward_with_callback(xs, grid_thw, |_, _| Ok(()))
    }

    pub fn set_rotary_buffer_dtype(&mut self, dtype: DType) -> Result<()> {
        self.rotary_pos_emb.inv_freq = self.rotary_pos_emb.inv_freq.to_dtype(dtype)?;
        Ok(())
    }

    pub fn forward_with_callback(
        &self,
        xs: &Tensor,
        grid_thw: &Tensor,
        mut capture: impl FnMut(usize, &Tensor) -> Result<()>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let dtype = self.pos_embed.embeddings().dtype();
        let xs = self.patch_embed.forward(&xs.to_dtype(dtype)?)?;
        capture(0, &xs)?;
        let pos_embeds = self.fast_pos_embed_interpolate(grid_thw)?;
        let mut hidden_states = xs.add(&pos_embeds)?;
        capture(1, &hidden_states)?;

        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let seq_len = hidden_states.dim(0)?;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;
        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], D::Minus1)?;
        let cos = emb.cos()?.to_dtype(DType::F32)?;
        let sin = emb.sin()?.to_dtype(DType::F32)?;

        let cu_seqlens = Self::build_cu_seqlens(grid_thw)?;

        let mut deepstack_features = Vec::new();
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            hidden_states = block.forward(&hidden_states, &cu_seqlens, &cos, &sin)?;
            capture(layer_idx + 2, &hidden_states)?;
            if let Some(merger_idx) = self.deepstack_lookup[layer_idx] {
                let feat = self.deepstack_mergers[merger_idx].forward(&hidden_states)?;
                deepstack_features.push(feat);
            }
        }

        let hidden_states = self.merger.forward(&hidden_states)?;
        capture(self.blocks.len() + 2, &hidden_states)?;
        Ok((hidden_states, deepstack_features))
    }
}
