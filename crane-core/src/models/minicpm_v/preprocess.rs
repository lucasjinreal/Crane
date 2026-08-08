//! MiniCPM-V-4.6 image preprocessing: the classic MiniCPM-V slicing algorithm
//! (pick a slice grid, resize an "overview" + per-slice tiles, patchify) plus
//! the placeholder-token expansion that must match it token-for-token.
//!
//! Ported from HF's `image_processing_minicpmv4_6.py` (`find_best_resize`,
//! `get_refine_size`, `get_sliced_grid`, `reshape_by_patch`) and
//! `processing_minicpmv4_6.py` (`MiniCPMV4_6Processor.__call__`'s image
//! placeholder expansion).

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use serde::Deserialize;

use super::config::Config;

/// Mirror of HF `preprocessor_config.json`.
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
    #[serde(default = "default_image_mean")]
    pub image_mean: Vec<f32>,
    #[serde(default = "default_image_std")]
    pub image_std: Vec<f32>,
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
fn default_image_mean() -> Vec<f32> {
    vec![0.5, 0.5, 0.5]
}
fn default_image_std() -> Vec<f32> {
    vec![0.5, 0.5, 0.5]
}

pub fn load_preprocessor_config(model_dir: &str) -> Result<PreprocessorConfig> {
    let path = std::path::Path::new(model_dir).join("preprocessor_config.json");
    let data = std::fs::read(&path)
        .with_context(|| format!("read preprocessor_config.json at {}", path.display()))?;
    let cfg: PreprocessorConfig = serde_json::from_slice(&data)
        .with_context(|| format!("parse {}", path.display()))?;
    Ok(cfg)
}

// ── Slicing arithmetic (pure, no image ops) ─────────────────────────────

pub(crate) fn ensure_divide(length: f64, divisor: usize) -> usize {
    ((length / divisor as f64).round() as usize).max(1) * divisor
}

/// Resize `(height, width)` to fit `scale_resolution` while keeping the
/// aspect ratio, then round to a multiple of `unit`. Direct port of
/// `find_best_resize` — `unit` is `patch_size * 4` for MiniCPM-V-4.6 (4 = the
/// two successive 2x2 merges: the vision tower's window merger + the
/// hierarchical `Merger`) but plain `patch_size` for MiniCPM-o (its
/// `Resampler` compresses via cross-attention, not size-halving, so no extra
/// divisibility constraint) — see `minicpmo::preprocess`.
pub(crate) fn find_best_resize(height: usize, width: usize, scale_resolution: usize, unit: usize, allow_upscale: bool) -> (usize, usize) {
    let (mut h, mut w) = (height as f64, width as f64);
    if h * w > (scale_resolution * scale_resolution) as f64 || allow_upscale {
        let aspect_ratio = w / h;
        h = scale_resolution as f64 / aspect_ratio.sqrt();
        w = h * aspect_ratio;
    }
    let best_width = ensure_divide(w, unit);
    let best_height = ensure_divide(h, unit);
    (best_height, best_width)
}

/// Resolution for the "refined" (to-be-sliced) version of the source image,
/// such that it divides evenly into `grid_h x grid_w` tiles each sized per
/// [`find_best_resize`]. Direct port of `get_refine_size`. `unit`: see
/// [`find_best_resize`].
pub(crate) fn get_refine_size(height: usize, width: usize, grid_h: usize, grid_w: usize, scale_resolution: usize, unit: usize, allow_upscale: bool) -> (usize, usize) {
    let refine_width = ensure_divide(width as f64, grid_w);
    let refine_height = ensure_divide(height as f64, grid_h);
    let (best_h, best_w) = find_best_resize(
        (refine_height as f64 / grid_h as f64).round() as usize,
        (refine_width as f64 / grid_w as f64).round() as usize,
        scale_resolution,
        unit,
        allow_upscale,
    );
    (best_h * grid_h, best_w * grid_w)
}

/// Pick a `(grid_h, grid_w)` slice grid minimizing aspect-ratio distortion,
/// or `None` if the image is small enough not to need slicing. Direct port
/// of `get_sliced_grid` — note HF's local variable names there (`num_rows`,
/// `num_cols`) are swapped from their visual meaning; this port uses
/// `grid_h`/`grid_w` (rows/cols) throughout instead, matching how the result
/// is actually consumed downstream (`get_refine_size`, and the Processor's
/// `num_rows, num_cols = image_grids[...]`). Reused as-is by `minicpmo::preprocess`
/// (the search algorithm doesn't depend on the vision tower's merge strategy).
pub(crate) fn get_sliced_grid(height: usize, width: usize, max_slice_nums: usize, scale_resolution: usize) -> Option<(usize, usize)> {
    let log_ratio = (width as f64 / height as f64).ln();
    let ratio = (width * height) as f64 / (scale_resolution * scale_resolution) as f64;
    let multiple = (ratio.ceil() as usize).min(max_slice_nums);
    if multiple <= 1 {
        return None;
    }

    let mut best = (1usize, 1usize);
    let mut min_error = f64::INFINITY;
    for num_slices in [multiple.saturating_sub(1), multiple, multiple + 1] {
        if num_slices == 1 || num_slices > max_slice_nums || num_slices == 0 {
            continue;
        }
        for grid_w in 1..=num_slices {
            if num_slices % grid_w != 0 {
                continue;
            }
            let grid_h = num_slices / grid_w;
            let error = (log_ratio - (grid_w as f64 / grid_h as f64).ln()).abs();
            if error < min_error {
                best = (grid_h, grid_w);
                min_error = error;
            }
        }
    }
    Some(best)
}

// ── Image ops ────────────────────────────────────────────────────────────

pub(crate) fn resize_bicubic(image: &DynamicImage, height: usize, width: usize) -> DynamicImage {
    let (w, h) = image.dimensions();
    if (h as usize, w as usize) == (height, width) {
        return image.clone();
    }
    // PIL BICUBIC (resample=3)'s kernel (a=-0.5) is Catmull-Rom, not Lanczos —
    // same convention already established in `qwen3_5/processor.rs`.
    image.resize_exact(width as u32, height as u32, FilterType::CatmullRom)
}

/// RGB `DynamicImage` -> normalized `[C, H, W]` tensor (`(pixel/255 - mean) /
/// std`, per-channel).
pub(crate) fn to_normalized_chw(image: &DynamicImage, mean: &[f32], std: &[f32], device: &Device) -> Result<Tensor> {
    let rgb = image.to_rgb8();
    let (w, h) = rgb.dimensions();
    let (w, h) = (w as usize, h as usize);
    let mut chw = vec![0f32; 3 * w * h];
    for (x, y, pixel) in rgb.enumerate_pixels() {
        for c in 0..3 {
            let v = pixel[c as usize] as f32 / 255.0;
            chw[c * w * h + y as usize * w + x as usize] = (v - mean[c]) / std[c];
        }
    }
    Ok(Tensor::from_vec(chw, (3, h, w), device)?)
}

/// `[C, H, W]` (H = h_patches*patch, W = w_patches*patch) -> NaViT-packed
/// `[C, patch, h_patches*w_patches*patch]`, patches enumerated row-major.
/// Direct port of `reshape_by_patch` (re-derived via reshape/permute instead
/// of literally reimplementing `F.unfold`, since candle has no unfold op —
/// see the module doc for the derivation).
pub(crate) fn reshape_by_patch(chw: &Tensor, patch_size: usize) -> Result<Tensor> {
    let (c, h, w) = chw.dims3()?;
    let (hp, wp) = (h / patch_size, w / patch_size);
    Ok(chw
        .reshape((c, hp, patch_size, wp, patch_size))?
        .permute((0, 2, 1, 3, 4))?
        .contiguous()?
        .reshape((c, patch_size, hp * wp * patch_size))?)
}

/// One preprocessed image: its packed pixel tensor, per-slice target sizes
/// (in patches), and the grid info needed to build the placeholder text.
pub struct ProcessedImage {
    /// `[C, patch_size, total_patches*patch_size]` — this image's overview +
    /// slices, packed along the last dim in the same order as `target_sizes`.
    pub pixel_values: Tensor,
    /// `(h_patches, w_patches)` per packed unit: `[overview, slice_0, slice_1, ...]`.
    pub target_sizes: Vec<(usize, usize)>,
    /// `Some((grid_h, grid_w))` if the image was sliced, `None` otherwise.
    pub grid: Option<(usize, usize)>,
}

/// Preprocess one image per MiniCPM-V-4.6's slicing algorithm.
pub fn process_image(image: &DynamicImage, cfg: &PreprocessorConfig, device: &Device, dtype: DType) -> Result<ProcessedImage> {
    let (w, h) = image.dimensions();
    let (height, width) = (h as usize, w as usize);
    let rgb = DynamicImage::ImageRgb8(image.to_rgb8());

    let grid = if cfg.slice_mode {
        get_sliced_grid(height, width, cfg.max_slice_nums, cfg.scale_resolution)
    } else {
        None
    };

    let (source_h, source_w) = find_best_resize(height, width, cfg.scale_resolution, cfg.patch_size * 4, grid.is_none());
    let source_img = resize_bicubic(&rgb, source_h, source_w);
    let source_chw = to_normalized_chw(&source_img, &cfg.image_mean, &cfg.image_std, device)?.to_dtype(dtype)?;

    let mut packed = vec![reshape_by_patch(&source_chw, cfg.patch_size)?];
    let mut target_sizes = vec![(source_h / cfg.patch_size, source_w / cfg.patch_size)];

    if let Some((grid_h, grid_w)) = grid {
        let (refine_h, refine_w) = get_refine_size(height, width, grid_h, grid_w, cfg.scale_resolution, cfg.patch_size * 4, true);
        let refine_img = resize_bicubic(&rgb, refine_h, refine_w);
        let (tile_h, tile_w) = (refine_h / grid_h, refine_w / grid_w);

        for row in 0..grid_h {
            for col in 0..grid_w {
                let tile = refine_img.crop_imm((col * tile_w) as u32, (row * tile_h) as u32, tile_w as u32, tile_h as u32);
                let tile_chw = to_normalized_chw(&tile, &cfg.image_mean, &cfg.image_std, device)?.to_dtype(dtype)?;
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

// ── Placeholder text expansion ──────────────────────────────────────────

/// Special-token strings the placeholder builder needs. Read from the
/// tokenizer's `extra_special_tokens` (`tokenizer_config.json`) rather than
/// hardcoded, since these aren't guaranteed stable across checkpoints.
pub struct PlaceholderTokens {
    pub image_token: String,
    pub image_start_token: String,
    pub image_end_token: String,
    pub slice_start_token: String,
    pub slice_end_token: String,
    pub image_id_start_token: String,
    pub image_id_end_token: String,
}

/// Build the expanded placeholder string for one image, replacing a single
/// `image_token` occurrence in the prompt. Mirrors
/// `MiniCPMV4_6Processor.__call__`'s per-image expansion exactly (token
/// counts must match what `VisionModel` + `Merger` actually produce: the
/// vision pipeline halves each spatial axis twice — the window merger, then
/// the `Merger` — so `downsample_mode: "16x"` means `h_patches*w_patches/16`
/// tokens per image/slice).
pub fn build_placeholder(image: &ProcessedImage, tokens: &PlaceholderTokens, local_image_index: usize, use_image_id: bool, downsample_divisor: usize) -> String {
    let overview_tokens = (image.target_sizes[0].0 * image.target_sizes[0].1) / downsample_divisor;
    let mut placeholder = format!(
        "{}{}{}",
        tokens.image_start_token,
        tokens.image_token.repeat(overview_tokens),
        tokens.image_end_token
    );
    if use_image_id {
        placeholder = format!(
            "{}{}{}{}",
            tokens.image_id_start_token, local_image_index, tokens.image_id_end_token, placeholder
        );
    }

    if let Some((grid_h, grid_w)) = image.grid {
        let per_slice_tokens = (image.target_sizes[1].0 * image.target_sizes[1].1) / downsample_divisor;
        let slice_placeholder = format!(
            "{}{}{}",
            tokens.slice_start_token,
            tokens.image_token.repeat(per_slice_tokens),
            tokens.slice_end_token
        );
        let rows: Vec<String> = (0..grid_h).map(|_| slice_placeholder.repeat(grid_w)).collect();
        placeholder.push_str(&rows.join("\n"));
    }

    placeholder
}

pub fn downsample_divisor(cfg: &Config) -> usize {
    if cfg.downsample_mode == "4x" {
        4
    } else {
        16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_divide_rounds_to_nearest() {
        assert_eq!(ensure_divide(100.0, 56), 112); // round(100/56)=2 -> 112
        assert_eq!(ensure_divide(10.0, 56), 56); // max(round(10/56),1)=1 -> 56
    }

    #[test]
    fn get_sliced_grid_small_image_returns_none() {
        // 448x448 exactly fills one scale_resolution^2 tile -> ratio=1, multiple<=1.
        assert_eq!(get_sliced_grid(448, 448, 9, 448), None);
    }

    #[test]
    fn get_sliced_grid_wide_image_prefers_more_columns() {
        // A very wide image should end up with grid_w > grid_h.
        let (gh, gw) = get_sliced_grid(448, 448 * 6, 9, 448).expect("should slice");
        assert!(gw >= gh, "wide image should get more columns than rows: ({gh},{gw})");
    }

    #[test]
    fn find_best_resize_divisible_by_patch_times_4() {
        let (h, w) = find_best_resize(1000, 1500, 448, 14 * 4, false);
        assert_eq!(h % (14 * 4), 0);
        assert_eq!(w % (14 * 4), 0);
    }
}
