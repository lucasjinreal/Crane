//! MiniCPM-o-4.5 image preprocessing.
//!
//! Same slicing algorithm as MiniCPM-V-4.6 (`crate::models::minicpm_v::preprocess`,
//! whose pure-arithmetic/image-op helpers this module reuses directly — see
//! `find_best_resize`'s doc comment for the one real difference: no `*4`
//! divisibility constraint, since MiniCPM-o's `Resampler` compresses via
//! cross-attention rather than size-halving merges), but a different
//! placeholder-token scheme: every image *and every slice* gets exactly
//! `image_feature_size` (`<unk>`) tokens, since `Resampler` always emits a
//! fixed `num_queries` tokens regardless of the input patch-grid size —
//! unlike MiniCPM-V-4.6's `Merger`, whose per-slice token count scales with
//! the slice's resolution. Ported from `processing_minicpmo.py`'s
//! `MiniCPMVImageProcessor` (`slice_image`, `get_sliced_images`,
//! `reshape_by_patch`) and `get_slice_image_placeholder`/`get_grid_placeholder`.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use image::DynamicImage;
use serde::Deserialize;

use crate::models::minicpm_v::preprocess::{
    find_best_resize, get_refine_size, get_sliced_grid, reshape_by_patch, resize_bicubic, to_normalized_chw,
};

/// Mirror of MiniCPM-o-4.5's `preprocessor_config.json` (note the field
/// names — `norm_mean`/`norm_std`, not `image_mean`/`image_std`).
#[derive(Debug, Clone, Deserialize)]
pub struct PreprocessorConfig {
    #[serde(default = "default_max_slice_nums")]
    pub max_slice_nums: usize,
    #[serde(default = "default_scale_resolution")]
    pub scale_resolution: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_true")]
    pub slice_mode: bool,
    #[serde(default = "default_true")]
    pub use_image_id: bool,
    #[serde(default = "default_image_feature_size")]
    pub image_feature_size: usize,
    #[serde(default = "default_mean_std")]
    pub norm_mean: Vec<f32>,
    #[serde(default = "default_mean_std")]
    pub norm_std: Vec<f32>,
    #[serde(default = "default_im_start")]
    pub im_start: String,
    #[serde(default = "default_im_end")]
    pub im_end: String,
    #[serde(default = "default_slice_start")]
    pub slice_start: String,
    #[serde(default = "default_slice_end")]
    pub slice_end: String,
    #[serde(default = "default_unk")]
    pub unk: String,
    #[serde(default = "default_im_id_start")]
    pub im_id_start: String,
    #[serde(default = "default_im_id_end")]
    pub im_id_end: String,
}

fn default_max_slice_nums() -> usize {
    9
}
fn default_scale_resolution() -> usize {
    448
}
fn default_patch_size() -> usize {
    14
}
fn default_true() -> bool {
    true
}
fn default_image_feature_size() -> usize {
    64
}
fn default_mean_std() -> Vec<f32> {
    vec![0.5, 0.5, 0.5]
}
fn default_im_start() -> String {
    "<image>".to_string()
}
fn default_im_end() -> String {
    "</image>".to_string()
}
fn default_slice_start() -> String {
    "<slice>".to_string()
}
fn default_slice_end() -> String {
    "</slice>".to_string()
}
fn default_unk() -> String {
    "<unk>".to_string()
}
fn default_im_id_start() -> String {
    "<image_id>".to_string()
}
fn default_im_id_end() -> String {
    "</image_id>".to_string()
}

pub fn load_preprocessor_config(model_dir: &str) -> Result<PreprocessorConfig> {
    let path = std::path::Path::new(model_dir).join("preprocessor_config.json");
    let data = std::fs::read(&path).with_context(|| format!("read preprocessor_config.json at {}", path.display()))?;
    Ok(serde_json::from_slice(&data).with_context(|| format!("parse {}", path.display()))?)
}

/// One preprocessed image: its packed pixel tensor, per-slice target sizes,
/// and the slice grid (needed to build the placeholder text).
pub struct ProcessedImage {
    /// `[C, patch_size, total_patches*patch_size]` — overview + slices,
    /// packed along the last dim in the same order as `target_sizes`.
    pub pixel_values: Tensor,
    /// `(h_patches, w_patches)` per packed unit: `[overview, slice_0, slice_1, ...]`.
    pub target_sizes: Vec<(usize, usize)>,
    /// `Some((grid_h, grid_w))` if the image was sliced, `None` otherwise.
    pub grid: Option<(usize, usize)>,
}

/// Preprocess one image per MiniCPM-o's slicing algorithm (identical search
/// to MiniCPM-V-4.6's, `unit = patch_size` rather than `patch_size * 4`).
pub fn process_image(image: &DynamicImage, cfg: &PreprocessorConfig, device: &Device, dtype: DType) -> Result<ProcessedImage> {
    use image::GenericImageView;

    let (w, h) = image.dimensions();
    let (height, width) = (h as usize, w as usize);
    let rgb = DynamicImage::ImageRgb8(image.to_rgb8());

    let grid = if cfg.slice_mode {
        get_sliced_grid(height, width, cfg.max_slice_nums, cfg.scale_resolution)
    } else {
        None
    };

    let (source_h, source_w) = find_best_resize(height, width, cfg.scale_resolution, cfg.patch_size, grid.is_none());
    let source_img = resize_bicubic(&rgb, source_h, source_w);
    let source_chw = to_normalized_chw(&source_img, &cfg.norm_mean, &cfg.norm_std, device)?.to_dtype(dtype)?;

    let mut packed = vec![reshape_by_patch(&source_chw, cfg.patch_size)?];
    let mut target_sizes = vec![(source_h / cfg.patch_size, source_w / cfg.patch_size)];

    if let Some((grid_h, grid_w)) = grid {
        let (refine_h, refine_w) = get_refine_size(height, width, grid_h, grid_w, cfg.scale_resolution, cfg.patch_size, true);
        let refine_img = resize_bicubic(&rgb, refine_h, refine_w);
        let (tile_h, tile_w) = (refine_h / grid_h, refine_w / grid_w);

        for row in 0..grid_h {
            for col in 0..grid_w {
                let tile = refine_img.crop_imm((col * tile_w) as u32, (row * tile_h) as u32, tile_w as u32, tile_h as u32);
                let tile_chw = to_normalized_chw(&tile, &cfg.norm_mean, &cfg.norm_std, device)?.to_dtype(dtype)?;
                packed.push(reshape_by_patch(&tile_chw, cfg.patch_size)?);
                target_sizes.push((tile_h / cfg.patch_size, tile_w / cfg.patch_size));
            }
        }
    }

    let pixel_values = Tensor::cat(&packed, 2)?;
    Ok(ProcessedImage { pixel_values, target_sizes, grid })
}

/// Pack multiple already-processed images into one NaViT batch (`[1, C,
/// patch_size, total_patches*patch_size]`) plus the flat, in-order
/// `target_sizes` list `VisionModel::forward` expects.
pub fn pack_images(images: &[ProcessedImage]) -> Result<(Tensor, Vec<(usize, usize)>)> {
    let pv: Vec<&Tensor> = images.iter().map(|im| &im.pixel_values).collect();
    let pixel_values = Tensor::cat(&pv, 2)?.unsqueeze(0)?;
    let target_sizes = images.iter().flat_map(|im| im.target_sizes.iter().copied()).collect();
    Ok((pixel_values, target_sizes))
}

/// Build the placeholder string for one image, to be inserted directly into
/// the prompt (unlike MiniCPM-V-4.6, there's no separate marker token to
/// expand afterward). Every image/slice gets exactly `cfg.image_feature_size`
/// placeholder tokens, regardless of its patch-grid size — direct port of
/// `get_slice_image_placeholder`/`get_grid_placeholder`.
pub fn build_placeholder(image: &ProcessedImage, cfg: &PreprocessorConfig, image_idx: usize) -> String {
    let mut placeholder = format!("{}{}{}", cfg.im_start, cfg.unk.repeat(cfg.image_feature_size), cfg.im_end);
    if cfg.use_image_id {
        placeholder = format!("{}{}{}{}", cfg.im_id_start, image_idx, cfg.im_id_end, placeholder);
    }

    if let Some((grid_h, grid_w)) = image.grid {
        let slice_placeholder = format!("{}{}{}", cfg.slice_start, cfg.unk.repeat(cfg.image_feature_size), cfg.slice_end);
        let rows: Vec<String> = (0..grid_h).map(|_| slice_placeholder.repeat(grid_w)).collect();
        placeholder.push('\n');
        placeholder.push_str(&rows.join("\n"));
    }

    placeholder
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_placeholder_no_slices_is_fixed_length() {
        let cfg = PreprocessorConfig {
            max_slice_nums: 9,
            scale_resolution: 448,
            patch_size: 14,
            slice_mode: true,
            use_image_id: true,
            image_feature_size: 64,
            norm_mean: default_mean_std(),
            norm_std: default_mean_std(),
            im_start: default_im_start(),
            im_end: default_im_end(),
            slice_start: default_slice_start(),
            slice_end: default_slice_end(),
            unk: default_unk(),
            im_id_start: default_im_id_start(),
            im_id_end: default_im_id_end(),
        };
        let image = ProcessedImage {
            pixel_values: Tensor::zeros((3, 14, 14), DType::F32, &Device::Cpu).unwrap(),
            target_sizes: vec![(26, 40)],
            grid: None,
        };
        let placeholder = build_placeholder(&image, &cfg, 0);
        assert_eq!(placeholder.matches("<unk>").count(), 64);
        assert!(placeholder.starts_with("<image_id>0</image_id><image>"));
        assert!(placeholder.ends_with("</image>"));
    }
}
