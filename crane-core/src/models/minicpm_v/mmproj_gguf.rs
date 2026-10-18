//! Loader for llama.cpp `mmproj-*.gguf` vision projectors (MiniCPM-V 4.x).
//!
//! The mmproj holds the same weights as the HF checkpoint's `vision_tower.*` +
//! `merger.*` blocks under llama.cpp names (`v.blk.N.ln1`, `mm.up`, ...).
//! Rather than a second copy of the vision code, this renames the tensors back
//! to their HF names and hands [`VisionModel`](super::vision::VisionModel) /
//! [`Merger`](super::merger::Merger) a `VarBuilder`. Tensor *shapes* need no
//! fix-up: GGUF stores dims reversed, which candle un-reverses on read, so
//! `mm.down.weight` (`ne = [4608, 1024]`) already arrives as the HF
//! `linear_2.weight` `[1024, 4608]`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::config::VisionConfig;
use crate::quantized::gguf_file::Gguf;
use crate::quantized::gguf_metadata::GgufMetadata;

/// Everything a mmproj tells us that `config.json` would otherwise provide.
pub struct Mmproj {
    pub vb: VarBuilder<'static>,
    pub vision_config: VisionConfig,
    /// 0-based vision layer after which the window merger runs
    /// (`clip.vision.wa_layer_indexes[0]`).
    pub insert_layer_id: usize,
    /// `clip.vision.projector.scale_factor`: 4 = `"16x"` (full pipeline),
    /// 2 = `"4x"` (no `vit_merger`).
    pub scale_factor: usize,
    /// LLM hidden size the projector outputs (`clip.vision.projection_dim`).
    pub projection_dim: usize,
    /// `clip.projector_type`, e.g. `minicpmv4_7`.
    pub projector_type: String,
    pub max_slice_nums: Option<usize>,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
}

/// llama.cpp tensor name -> HF name (relative to the checkpoint root, with the
/// `model.` prefix `VisionModel`/`Merger` loading expects). `None` = not a
/// weight we use (e.g. `v.tok_embd_sep`, llama.cpp's baked marker embeddings).
fn hf_name(name: &str) -> Option<String> {
    if let Some(rest) = name.strip_prefix("v.blk.") {
        let (idx, field) = rest.split_once('.')?;
        let mapped = match field.split_once('.')? {
            ("ln1", p) => format!("layer_norm1.{p}"),
            ("ln2", p) => format!("layer_norm2.{p}"),
            ("ffn_up", p) => format!("mlp.fc1.{p}"),
            ("ffn_down", p) => format!("mlp.fc2.{p}"),
            ("attn_q", p) => format!("self_attn.q_proj.{p}"),
            ("attn_k", p) => format!("self_attn.k_proj.{p}"),
            ("attn_v", p) => format!("self_attn.v_proj.{p}"),
            ("attn_out", p) => format!("self_attn.out_proj.{p}"),
            _ => return None,
        };
        return Some(format!("model.vision_tower.encoder.layers.{idx}.{mapped}"));
    }
    if let Some(field) = name.strip_prefix("v.vit_merger.") {
        let mapped = match field.split_once('.')? {
            ("ln1", p) => format!("layer_norm1.{p}"),
            ("ds_ln", p) => format!("pre_norm.{p}"),
            ("ds_ffn_up", p) => format!("linear_1.{p}"),
            ("ds_ffn_down", p) => format!("linear_2.{p}"),
            ("attn_q", p) => format!("self_attn.q_proj.{p}"),
            ("attn_k", p) => format!("self_attn.k_proj.{p}"),
            ("attn_v", p) => format!("self_attn.v_proj.{p}"),
            ("attn_out", p) => format!("self_attn.out_proj.{p}"),
            _ => return None,
        };
        return Some(format!("model.vision_tower.vit_merger.{mapped}"));
    }
    Some(
        match name {
            "v.patch_embd.weight" => "model.vision_tower.embeddings.patch_embedding.weight",
            "v.patch_embd.bias" => "model.vision_tower.embeddings.patch_embedding.bias",
            "v.position_embd.weight" => "model.vision_tower.embeddings.position_embedding.weight",
            "v.post_ln.weight" => "model.vision_tower.post_layernorm.weight",
            "v.post_ln.bias" => "model.vision_tower.post_layernorm.bias",
            "mm.input_norm.weight" => "model.merger.mlp.0.pre_norm.weight",
            "mm.input_norm.bias" => "model.merger.mlp.0.pre_norm.bias",
            "mm.up.weight" => "model.merger.mlp.0.linear_1.weight",
            "mm.up.bias" => "model.merger.mlp.0.linear_1.bias",
            "mm.down.weight" => "model.merger.mlp.0.linear_2.weight",
            "mm.down.bias" => "model.merger.mlp.0.linear_2.bias",
            _ => return None,
        }
        .to_string(),
    )
}

/// Load a MiniCPM-V mmproj GGUF, dequantizing/casting every tensor to `dtype`.
///
/// # Errors
///
/// Returns an error if the file isn't a `clip` mmproj, metadata is missing, or
/// a tensor can't be read.
pub fn load_mmproj(path: &Path, device: &Device, dtype: DType) -> Result<Mmproj> {
    let mmap = crate::quantized::gguf_file::mmap_gguf_file(path)
        .with_context(|| format!("mmap mmproj {}", path.display()))?;
    let (ct, _) = crate::quantized::extended_gguf::read_content(mmap.as_ref())?;
    let mut cursor = std::io::Cursor::new(mmap.as_ref());
    let mut gg = Gguf::new(ct, &mut cursor, device.clone(), dtype);

    let meta = gg.metadata().clone();
    let md = GgufMetadata::new(&meta);
    let arch = md.opt_string("general.architecture").unwrap_or_default();
    if arch != "clip" {
        bail!(
            "{} is not an mmproj (general.architecture = {arch:?})",
            path.display()
        );
    }
    let projector_type = md.string("clip.projector_type")?;
    let patch_size = md.usize("clip.vision.patch_size")?;
    let hidden_size = md.usize("clip.vision.embedding_length")?;
    let wa_layers = md.usizes("clip.vision.wa_layer_indexes")?;
    let floats = |key: &str, default: [f32; 3]| -> Vec<f32> {
        md.0.get(key)
            .and_then(|v| v.to_vec().ok())
            .map(|vals| {
                vals.iter()
                    .filter_map(|v| v.to_f32().ok())
                    .collect::<Vec<_>>()
            })
            .filter(|v| v.len() == 3)
            .unwrap_or_else(|| default.to_vec())
    };
    let image_mean = floats("clip.vision.image_mean", [0.5; 3]);
    let image_std = floats("clip.vision.image_std", [0.5; 3]);
    let layer_norm_eps = md
        .f32("clip.vision.attention.layer_norm_epsilon")
        .map_or(1e-6, f64::from);

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let names: Vec<String> = gg.ct.tensor_infos.keys().cloned().collect();
    for name in names {
        if let Some(hf) = hf_name(&name) {
            tensors.insert(hf, gg.dequant_tensor(&name)?);
        }
    }

    let pos = tensors
        .get("model.vision_tower.embeddings.position_embedding.weight")
        .context("mmproj missing v.position_embd.weight")?
        .dim(0)?;
    let side = (pos as f64).sqrt().round() as usize;
    if side * side != pos {
        bail!("position table has {pos} entries, not a square grid");
    }
    let vision_config = VisionConfig {
        hidden_size,
        intermediate_size: md.usize("clip.vision.feed_forward_length")?,
        num_hidden_layers: md.usize("clip.vision.block_count")?,
        num_attention_heads: md.usize("clip.vision.attention.head_count")?,
        num_channels: 3,
        image_size: side * patch_size,
        patch_size,
        layer_norm_eps,
        attention_dropout: 0.0,
    };
    Ok(Mmproj {
        vb: VarBuilder::from_tensors(tensors, dtype, device),
        vision_config,
        insert_layer_id: *wa_layers.first().context("empty wa_layer_indexes")?,
        scale_factor: md.usize("clip.vision.projector.scale_factor")?,
        projection_dim: md.usize("clip.vision.projection_dim")?,
        projector_type,
        max_slice_nums: md.usize("clip.vision.max_slice_nums").ok(),
        image_mean,
        image_std,
    })
}

#[cfg(test)]
mod tests {
    use super::hf_name;

    #[test]
    fn maps_encoder_block_tensors() {
        assert_eq!(
            hf_name("v.blk.3.ffn_up.weight").unwrap(),
            "model.vision_tower.encoder.layers.3.mlp.fc1.weight"
        );
        assert_eq!(
            hf_name("v.blk.26.attn_out.bias").unwrap(),
            "model.vision_tower.encoder.layers.26.self_attn.out_proj.bias"
        );
    }

    #[test]
    fn maps_merger_and_projector() {
        assert_eq!(
            hf_name("v.vit_merger.ds_ffn_down.weight").unwrap(),
            "model.vision_tower.vit_merger.linear_2.weight"
        );
        assert_eq!(
            hf_name("mm.input_norm.bias").unwrap(),
            "model.merger.mlp.0.pre_norm.bias"
        );
        assert_eq!(
            hf_name("mm.down.weight").unwrap(),
            "model.merger.mlp.0.linear_2.weight"
        );
    }

    #[test]
    fn ignores_baked_marker_embeddings() {
        assert!(hf_name("v.tok_embd_sep").is_none());
    }
}
