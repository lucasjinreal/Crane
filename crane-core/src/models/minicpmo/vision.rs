//! MiniCPM-o-4.5 vision tower.
//!
//! Ported from `modeling_navit_siglip.py`'s `SiglipVisionTransformer` — a
//! **padded-batch** SigLIP (per-image `pixel_values` padded to a common
//! `[B, C, max_H, max_W]` canvas, with a boolean `patch_attention_mask`
//! marking real vs. padding patches), *not* the NaViT single-packed-sequence
//! variant `crate::models::minicpm_v::vision` implements for MiniCPM-V-4.6.
//! The per-patch math (bucketized position embedding, pre-norm MHA, GELU-tanh
//! MLP) is the same shape of problem, but the batching/masking mechanics
//! differ enough that reusing those structs directly would force an awkward
//! fit — this is a fresh, small port instead. No merger stage: MiniCPM-o's
//! `Resampler` (`super::resampler`) does the vision-token compression, so
//! every one of the tower's `num_hidden_layers` runs in full.

use std::io::{Read, Seek};

use candle_core::{DType, Module, Result, Tensor};
use candle_nn::{conv2d, embedding, layer_norm, linear, Activation, Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, VarBuilder};

use crate::models::hunyuan_dense::modeling::Gguf;
use crate::models::minicpm_v::config::VisionConfig;

// ── Attention ────────────────────────────────────────────────────────────

struct SiglipAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl SiglipAttention {
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

    /// GGUF equivalent of [`Self::new`] — `prefix` is this block's
    /// tensor-name prefix (e.g. `v.blk.0`), matching llama.cpp's own
    /// `clip.cpp`/mtmd convention for `clip.has_minicpmv_projector`
    /// vision towers (real reference: the checkpoint's own GGUF export,
    /// `openbmb/MiniCPM-o-4_5-gguf`'s `MiniCPM-o-4_5-vision-F16.gguf`).
    fn from_gguf<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str, dim: usize, num_heads: usize) -> Result<Self> {
        let head_dim = dim / num_heads;
        let q_proj = Linear::new(gg.dequant_tensor(&format!("{prefix}.attn_q.weight"))?, Some(gg.dequant_tensor(&format!("{prefix}.attn_q.bias"))?.flatten_all()?));
        let k_proj = Linear::new(gg.dequant_tensor(&format!("{prefix}.attn_k.weight"))?, Some(gg.dequant_tensor(&format!("{prefix}.attn_k.bias"))?.flatten_all()?));
        let v_proj = Linear::new(gg.dequant_tensor(&format!("{prefix}.attn_v.weight"))?, Some(gg.dequant_tensor(&format!("{prefix}.attn_v.bias"))?.flatten_all()?));
        let out_proj = Linear::new(gg.dequant_tensor(&format!("{prefix}.attn_out.weight"))?, Some(gg.dequant_tensor(&format!("{prefix}.attn_out.bias"))?.flatten_all()?));
        Ok(Self { q_proj, k_proj, v_proj, out_proj, num_heads, head_dim, scale: 1.0 / (head_dim as f64).sqrt() })
    }

    /// `xs`: `[batch, seq, dim]`. `attn_mask`: additive `[batch, 1, seq, seq]`
    /// (large negative at padded key positions), or `None` when nothing in
    /// the batch is padded.
    fn forward(&self, xs: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq, _dim) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.k_proj.forward(xs)?.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.v_proj.forward(xs)?.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        let attn_weights = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?;
        let attn_weights = match attn_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32)?)?.to_dtype(xs.dtype())?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?; // [b, heads, seq, head_dim]

        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((b, seq, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&attn_output)
    }
}

// ── MLP ──────────────────────────────────────────────────────────────────

struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(dim: usize, intermediate: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(dim, intermediate, vb.pp("fc1"))?,
            fc2: linear(intermediate, dim, vb.pp("fc2"))?,
        })
    }
}

impl Mlp {
    /// GGUF equivalent of [`Self::new`] — every tensor here is standard
    /// PyTorch `(out_features, in_features)`, **but** llama.cpp's
    /// `clip.cpp` exporter's `ffn_up`/`ffn_down` names are swapped
    /// relative to what they'd naively suggest: `ffn_down`'s bias has
    /// `intermediate_size` elements (it's actually the `hidden ->
    /// intermediate` *expansion*, i.e. this port's `fc1`) and `ffn_up`'s
    /// bias has `hidden_size` elements (the `intermediate -> hidden`
    /// *contraction*, this port's `fc2`) — confirmed by directly
    /// inspecting both tensors' shapes against each other, not just
    /// trusting the names.
    fn from_gguf<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let fc1 = Linear::new(gg.dequant_tensor(&format!("{prefix}.ffn_down.weight"))?, Some(gg.dequant_tensor(&format!("{prefix}.ffn_down.bias"))?.flatten_all()?));
        let fc2 = Linear::new(gg.dequant_tensor(&format!("{prefix}.ffn_up.weight"))?, Some(gg.dequant_tensor(&format!("{prefix}.ffn_up.bias"))?.flatten_all()?));
        Ok(Self { fc1, fc2 })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = Activation::GeluPytorchTanh.forward(&xs)?;
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
        let conv_cfg = Conv2dConfig { stride: cfg.patch_size, ..Default::default() };
        let patch_embedding = conv2d(cfg.num_channels, cfg.hidden_size, cfg.patch_size, conv_cfg, vb.pp("patch_embedding"))?;
        let position_embedding = embedding(cfg.num_patches(), cfg.hidden_size, vb.pp("position_embedding"))?;
        Ok(Self { patch_embedding, position_embedding, num_patches_per_side: cfg.num_patches_per_side() })
    }

    /// GGUF equivalent of [`Self::new`] — `v.patch_embd.*`/`v.position_embd.weight`
    /// in llama.cpp's `clip.cpp` naming.
    pub fn from_gguf<R: Read + Seek>(gg: &mut Gguf<R>, cfg: &VisionConfig) -> Result<Self> {
        let conv_cfg = Conv2dConfig { stride: cfg.patch_size, ..Default::default() };
        let patch_embedding = Conv2d::new(gg.dequant_tensor("v.patch_embd.weight")?, Some(gg.dequant_tensor("v.patch_embd.bias")?.flatten_all()?), conv_cfg);
        let position_embedding = Embedding::new(gg.dequant_tensor("v.position_embd.weight")?, cfg.hidden_size);
        Ok(Self { patch_embedding, position_embedding, num_patches_per_side: cfg.num_patches_per_side() })
    }

    /// `pixel_values`: `[B, C, max_H, max_W]`, zero-padded to a common
    /// canvas. `tgt_sizes`: true `(h, w)` patch-grid dims per batch item
    /// (`h*w <= (max_H/patch)*(max_W/patch)`).
    ///
    /// Returns `[B, max_nb_patches, hidden]` — position embeddings are only
    /// meaningful (and only added) at each item's real `h*w` positions;
    /// padding positions get position id 0 (matching Python's
    /// `torch.full(..., fill_value=0)` default before scatter), which is
    /// harmless since the padded positions are masked out of attention.
    pub fn forward(&self, pixel_values: &Tensor, tgt_sizes: &[(usize, usize)]) -> Result<Tensor> {
        let patch_embeds = self.patch_embedding.forward(pixel_values)?; // [B, hidden, H/p, W/p]
        let (b, hidden, gh, gw) = patch_embeds.dims4()?;
        let max_nb_patches = gh * gw;
        let patch_embeds = patch_embeds.reshape((b, hidden, max_nb_patches))?.transpose(1, 2)?; // [B, max_nb, hidden]

        let n_side = self.num_patches_per_side;
        let bucket = |k: usize, n: usize| -> usize {
            let frac = k as f64 / n as f64;
            ((frac * n_side as f64).floor() as usize).min(n_side - 1)
        };

        let mut pos_ids = vec![0u32; b * max_nb_patches];
        for (batch_idx, &(nb_h, nb_w)) in tgt_sizes.iter().enumerate() {
            for h in 0..nb_h {
                let bh = bucket(h, nb_h);
                for w in 0..nb_w {
                    let bw = bucket(w, nb_w);
                    pos_ids[batch_idx * max_nb_patches + h * nb_w + w] = (bh * n_side + bw) as u32;
                }
            }
        }
        let pos_ids = Tensor::from_vec(pos_ids, (b, max_nb_patches), patch_embeds.device())?;
        let pos_embeds = self.position_embedding.forward(&pos_ids)?.to_dtype(patch_embeds.dtype())?;

        patch_embeds.add(&pos_embeds)
    }
}

// ── Encoder layer ────────────────────────────────────────────────────────

struct EncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: SiglipAttention,
    layer_norm2: LayerNorm,
    mlp: Mlp,
}

impl EncoderLayer {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            layer_norm1: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm1"))?,
            self_attn: SiglipAttention::new(cfg.hidden_size, cfg.num_attention_heads, vb.pp("self_attn"))?,
            layer_norm2: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm2"))?,
            mlp: Mlp::new(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?,
        })
    }

    /// GGUF equivalent of [`Self::new`] — `layer_idx`'s tensor-name prefix
    /// is `v.blk.{layer_idx}`.
    fn from_gguf<R: Read + Seek>(gg: &mut Gguf<R>, cfg: &VisionConfig, layer_idx: usize) -> Result<Self> {
        let prefix = format!("v.blk.{layer_idx}");
        let layer_norm1 = LayerNorm::new(gg.dequant_tensor(&format!("{prefix}.ln1.weight"))?, gg.dequant_tensor(&format!("{prefix}.ln1.bias"))?, cfg.layer_norm_eps);
        let self_attn = SiglipAttention::from_gguf(gg, &prefix, cfg.hidden_size, cfg.num_attention_heads)?;
        let layer_norm2 = LayerNorm::new(gg.dequant_tensor(&format!("{prefix}.ln2.weight"))?, gg.dequant_tensor(&format!("{prefix}.ln2.bias"))?, cfg.layer_norm_eps);
        let mlp = Mlp::from_gguf(gg, &prefix)?;
        Ok(Self { layer_norm1, self_attn, layer_norm2, mlp })
    }

    fn forward(&self, xs: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let hidden = self.layer_norm1.forward(xs)?;
        let hidden = self.self_attn.forward(&hidden, attn_mask)?;
        let hidden = (residual + hidden)?;

        let residual = &hidden;
        let mlp_in = self.layer_norm2.forward(&hidden)?;
        let mlp_out = self.mlp.forward(&mlp_in)?;
        residual + mlp_out
    }
}

// ── Vision transformer ──────────────────────────────────────────────────

pub struct VisionModel {
    embeddings: VisionEmbeddings,
    layers: Vec<EncoderLayer>,
    post_layernorm: LayerNorm,
}

impl VisionModel {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = VisionEmbeddings::new(cfg, vb.pp("embeddings"))?;
        let vb_layers = vb.pp("encoder").pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::new(cfg, vb_layers.pp(i))?);
        }
        let post_layernorm = layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm"))?;
        Ok(Self { embeddings, layers, post_layernorm })
    }

    /// GGUF equivalent of [`Self::new`] — loads from a standalone
    /// `MiniCPM-o-4_5-vision-F16.gguf`-style file (`general.architecture =
    /// clip`, `clip.has_minicpmv_projector = 1` — llama.cpp's own standard
    /// convention for MiniCPM-V-style vision towers) instead of the
    /// checkpoint's own safetensors weights. `cfg` still comes from the
    /// checkpoint's `config.json` as usual. Note this GGUF file's tensors
    /// are F16/F32 only — no actual quantization — so this delivers no
    /// VRAM savings over the safetensors path; it exists for pure-GGUF
    /// deployment convenience.
    ///
    /// # Errors
    ///
    /// Returns an error if a required tensor is missing or has an
    /// unexpected shape.
    pub fn from_gguf<R: Read + Seek>(gg: &mut Gguf<R>, cfg: &VisionConfig) -> Result<Self> {
        let embeddings = VisionEmbeddings::from_gguf(gg, cfg)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::from_gguf(gg, cfg, i)?);
        }
        let post_layernorm = LayerNorm::new(gg.dequant_tensor("v.post_ln.weight")?, gg.dequant_tensor("v.post_ln.bias")?, cfg.layer_norm_eps);
        Ok(Self { embeddings, layers, post_layernorm })
    }

    /// `pixel_values`: `[B, C, max_H, max_W]`. `tgt_sizes`: true `(h, w)`
    /// patch-grid dims per batch item.
    ///
    /// Returns `[B, max_nb_patches, hidden]` (still padded — the caller,
    /// [`super::resampler::Resampler`], masks padding via its own
    /// `key_padding_mask` built from the same `tgt_sizes`).
    pub fn forward(&self, pixel_values: &Tensor, tgt_sizes: &[(usize, usize)]) -> Result<Tensor> {
        let mut xs = self.embeddings.forward(pixel_values, tgt_sizes)?;
        let (b, max_nb_patches, _hidden) = xs.dims3()?;

        // Skip building a mask entirely when every item fills the full
        // padded canvas (the common single-image case) — matches Python's
        // `if not torch.any(~patch_attention_mask): attention_mask = None`.
        let needs_mask = tgt_sizes.iter().any(|&(h, w)| h * w != max_nb_patches);
        let attn_mask = if needs_mask {
            let mut mask_data = vec![0f32; b * max_nb_patches];
            for (batch_idx, &(h, w)) in tgt_sizes.iter().enumerate() {
                for pos in (h * w)..max_nb_patches {
                    mask_data[batch_idx * max_nb_patches + pos] = f32::MIN;
                }
            }
            let mask = Tensor::from_vec(mask_data, (b, 1, 1, max_nb_patches), xs.device())?.to_dtype(xs.dtype())?;
            Some(mask)
        } else {
            None
        };

        for layer in &self.layers {
            xs = layer.forward(&xs, attn_mask.as_ref())?;
        }
        self.post_layernorm.forward(&xs)
    }
}

// ── HF-diff validation ─────────────────────────────────────────────────
//
// Compares `VisionModel` + `super::resampler::Resampler` together against
// a real Python forward pass through `modeling_navit_siglip.py`'s
// `SiglipVisionTransformer` + `modeling_minicpmo.py`'s `Resampler` (real
// `vpm.*`/`resampler.*` checkpoint weights only — see
// `/home/hahihula/.claude/jobs/02aa7312/tmp/minicpmo_diff/dump_vision.py`),
// on a real test image (`assets/fossil.png`). The dump also saves the
// already-preprocessed `pixel_values`/`tgt_sizes` — Rust doesn't have its
// own image-preprocessing port yet, so this test feeds Python's exact
// preprocessed tensor in, isolating the neural net math from that (still
// unported) concern.
#[cfg(test)]
mod hf_diff {
    use super::*;
    use crate::models::minicpmo::config::load_config;
    use crate::models::minicpmo::resampler::Resampler;
    use candle_core::Device;
    use candle_nn::VarBuilder;

    // Run with:
    // `cargo test -p crane-core --release --features cuda -- --ignored --nocapture minicpmo_vision_matches_python`
    #[test]
    #[ignore = "needs the real MiniCPM-o-4.5 checkpoint + a matching HF dump"]
    fn minicpmo_vision_matches_python() {
        let model_path = "/home/hahihula/mywork/ai/additional_models/MiniCPM-o-4_5";
        let diff_dir = "/home/hahihula/.claude/jobs/02aa7312/tmp/minicpmo_diff";

        #[cfg(feature = "cuda")]
        let (device, dtype) = (Device::new_cuda(0).unwrap(), DType::BF16);
        #[cfg(not(feature = "cuda"))]
        let (device, dtype) = (Device::Cpu, DType::F32);

        let config = load_config(&format!("{model_path}/config.json")).expect("load config");

        let filenames = crate::utils::utils::get_safetensors_files(model_path).expect("safetensors files");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device) }.expect("varbuilder");

        let vpm = VisionModel::new(&config.vision_config, vb.pp("vpm")).expect("load vpm");
        let vision_dim = config.vision_config.hidden_size;
        let embed_dim = config.llm.hidden_size;
        let resampler = Resampler::new(config.query_num, embed_dim, embed_dim / 128, vision_dim, vb.pp("resampler"))
            .expect("load resampler");

        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{diff_dir}/vision_meta.json")).unwrap()).unwrap();
        let tgt_sizes: Vec<(usize, usize)> = meta["tgt_sizes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| {
                let pair = pair.as_array().unwrap();
                (pair[0].as_u64().unwrap() as usize, pair[1].as_u64().unwrap() as usize)
            })
            .collect();
        let pv_shape: Vec<usize> =
            meta["pixel_values_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();

        let pv_bytes = std::fs::read(format!("{diff_dir}/pixel_values.bin")).unwrap();
        let pv_data: Vec<f32> =
            pv_bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let pixel_values = Tensor::from_vec(pv_data, pv_shape.as_slice(), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();

        let vision_out = vpm.forward(&pixel_values, &tgt_sizes).expect("vision forward");
        let resampled = resampler.forward(&vision_out, &tgt_sizes).expect("resampler forward");
        let resampled = resampled.to_dtype(DType::F32).unwrap();

        let rust_shape = resampled.dims().to_vec();
        let expected_shape: Vec<usize> =
            meta["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        assert_eq!(rust_shape, expected_shape);

        let rust_flat: Vec<f32> = resampled.flatten_all().unwrap().to_vec1().unwrap();

        let py_bytes = std::fs::read(format!("{diff_dir}/vision_resampled.bin")).unwrap();
        let py_flat: Vec<f32> =
            py_bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        assert_eq!(rust_flat.len(), py_flat.len());

        let dot: f64 = rust_flat.iter().zip(&py_flat).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
        let norm_a: f64 = rust_flat.iter().map(|a| f64::from(*a) * f64::from(*a)).sum::<f64>().sqrt();
        let norm_b: f64 = py_flat.iter().map(|b| f64::from(*b) * f64::from(*b)).sum::<f64>().sqrt();
        let cosine = dot / (norm_a * norm_b);
        let max_abs_diff = rust_flat.iter().zip(&py_flat).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);

        println!("cosine similarity: {cosine}, max abs diff: {max_abs_diff}");
        assert!(cosine > 0.99, "cosine similarity too low: {cosine}");
    }
}

// ── GGUF cross-check ────────────────────────────────────────────────────
//
// No fresh Python dump needed (the architecture was already HF-diff
// validated above) — a direct Rust-vs-Rust comparison against the
// already-validated safetensors path catches tensor-mapping bugs in the
// new `from_gguf` paths just as well.
#[cfg(test)]
mod gguf_cross_check {
    use super::*;
    use crate::models::minicpmo::config::load_config;
    use crate::models::minicpmo::resampler::Resampler;
    use candle_core::Device;

    // Run with:
    // `cargo test -p crane-core --release --features cuda -- --ignored --nocapture minicpmo_vision_gguf_matches_safetensors`
    #[test]
    #[ignore = "needs the real MiniCPM-o-4.5 checkpoint + the standalone vision GGUF file"]
    fn minicpmo_vision_gguf_matches_safetensors() {
        let model_path = std::env::var("CRANE_MINICPMO_DIR").unwrap_or_else(|_| "/home/hahihula/mywork/ai/additional_models/MiniCPM-o-4_5".to_string());
        let gguf_path = std::env::var("CRANE_MINICPMO_VISION_GGUF")
            .unwrap_or_else(|_| "/home/hahihula/mywork/ai/additional_models/MiniCPM-o-4_5-vision-F16.gguf".to_string());
        let diff_dir = "/home/hahihula/.claude/jobs/02aa7312/tmp/minicpmo_diff";

        #[cfg(feature = "cuda")]
        let (device, dtype) = (Device::new_cuda(0).unwrap(), DType::BF16);
        #[cfg(not(feature = "cuda"))]
        let (device, dtype) = (Device::Cpu, DType::F32);

        let config = load_config(&format!("{model_path}/config.json")).expect("load config");
        let vision_dim = config.vision_config.hidden_size;
        let embed_dim = config.llm.hidden_size;

        // ── safetensors path (already HF-diff validated above) ──
        let filenames = crate::utils::utils::get_safetensors_files(&model_path).expect("safetensors files");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device) }.expect("varbuilder");
        let st_vpm = VisionModel::new(&config.vision_config, vb.pp("vpm")).expect("load vpm");
        let st_resampler = Resampler::new(config.query_num, embed_dim, embed_dim / 128, vision_dim, vb.pp("resampler")).expect("load resampler");

        // ── GGUF path (new) ──
        let mut gguf_file = std::fs::File::open(&gguf_path).unwrap_or_else(|e| panic!("open {gguf_path}: {e}"));
        let ct = candle_core::quantized::gguf_file::Content::read(&mut gguf_file).expect("parse gguf");
        let mut gg = crate::models::hunyuan_dense::modeling::Gguf::new(ct, &mut gguf_file, device.clone(), dtype);
        let gguf_vpm = VisionModel::from_gguf(&mut gg, &config.vision_config).expect("load vision model from gguf");
        let gguf_resampler = Resampler::from_gguf(&mut gg, embed_dim / 128, vision_dim, embed_dim).expect("load resampler from gguf");

        // ── same real input through both ──
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{diff_dir}/vision_meta.json")).unwrap()).unwrap();
        let tgt_sizes: Vec<(usize, usize)> = meta["tgt_sizes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| {
                let pair = pair.as_array().unwrap();
                (pair[0].as_u64().unwrap() as usize, pair[1].as_u64().unwrap() as usize)
            })
            .collect();
        let pv_shape: Vec<usize> = meta["pixel_values_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let pv_bytes = std::fs::read(format!("{diff_dir}/pixel_values.bin")).unwrap();
        let pv_data: Vec<f32> = pv_bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let pixel_values = Tensor::from_vec(pv_data, pv_shape.as_slice(), &device).unwrap().to_dtype(dtype).unwrap();

        let st_out = st_resampler.forward(&st_vpm.forward(&pixel_values, &tgt_sizes).expect("st vision forward"), &tgt_sizes).expect("st resampler forward").to_dtype(DType::F32).unwrap();
        let gguf_out = gguf_resampler.forward(&gguf_vpm.forward(&pixel_values, &tgt_sizes).expect("gguf vision forward"), &tgt_sizes).expect("gguf resampler forward").to_dtype(DType::F32).unwrap();

        assert_eq!(st_out.dims(), gguf_out.dims(), "GGUF and safetensors paths produced different output shapes");

        let st_flat: Vec<f32> = st_out.flatten_all().unwrap().to_vec1().unwrap();
        let gguf_flat: Vec<f32> = gguf_out.flatten_all().unwrap().to_vec1().unwrap();

        let dot: f64 = st_flat.iter().zip(&gguf_flat).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
        let norm_a: f64 = st_flat.iter().map(|a| f64::from(*a) * f64::from(*a)).sum::<f64>().sqrt();
        let norm_b: f64 = gguf_flat.iter().map(|b| f64::from(*b) * f64::from(*b)).sum::<f64>().sqrt();
        let cosine = dot / (norm_a * norm_b);
        let max_abs_diff = st_flat.iter().zip(&gguf_flat).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);

        println!("GGUF vs safetensors cosine similarity: {cosine}, max abs diff: {max_abs_diff}");
        assert!(cosine > 0.999, "GGUF and safetensors vision outputs diverge too much: cosine {cosine}");
    }
}
