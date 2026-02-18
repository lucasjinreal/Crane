use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use std::path::Path;

use anyhow::Context;

/// Interpolation mode for resizing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    /// Bilinear (smooth, default for most image tasks)
    Bilinear,
    /// Nearest neighbor (fast, preserves hard edges, pixelated)
    Nearest,
    // Future: Bicubic, Lanczos, etc. (would require custom kernel or different backend)
    Bicubic,
}

/// Loads an image from disk, converts to normalized float tensor, resizes it,
/// and returns a **B×C×H×W** tensor (batch=1, channels=3) ready for model input.
///
/// # Arguments
/// * `path`       - Path to the image file
/// * `target_h`   - Desired output height
/// * `target_w`   - Desired output width
/// * `mode`       - Interpolation method
/// * `device`     - Target device (CUDA/CPU)
///
/// # Returns
/// Tensor of shape `[1, 3, target_h, target_w]` with values ∈ [0.0, 1.0]
pub fn load_and_resize_image_to_tensor(
    path: impl AsRef<std::path::Path>,
    target_h: usize,
    target_w: usize,
    mode: ResizeMode,
    device: &Device,
) -> Result<Tensor> {
    let img = image::ImageReader::open(path.as_ref())
        .with_context(|| format!("Failed to open image: {:?}", path.as_ref()))?
        .decode()
        .context("Image decoding failed")?
        .to_rgb8();

    let (orig_w, orig_h) = img.dimensions();
    let needs_resize = orig_h as usize != target_h || orig_w as usize != target_w;

    let raw = img.into_raw();

    let mut tensor = (Tensor::from_vec(raw, (orig_h as usize, orig_w as usize, 3), device)?
        .to_dtype(DType::F32)?
        .permute((2, 0, 1))?
        / 255.0)?;

    if needs_resize {
        tensor = match mode {
            ResizeMode::Bilinear => {
                let t = tensor.unsqueeze(0)?;
                let t = t.upsample_bilinear2d(target_h, target_w, false)?;
                t.squeeze(0)?
            }
            ResizeMode::Nearest => {
                let t = tensor.unsqueeze(0)?;
                let t = t.upsample_nearest2d(target_h, target_w)?;
                t.squeeze(0)?
            }
            ResizeMode::Bicubic => {
                return Err(E::msg("Bicubic resize not implemented yet"));
            }
        };
    }

    Ok(tensor)
}

pub fn load_image_and_smart_resize(
    path: &Path,
    device: &Device,
    _dtype: DType,
    mode: ResizeMode,
) -> Result<(Tensor, Tensor)> {
    let img = image::ImageReader::open(path)
        .with_context(|| format!("Failed to open image: {:?}", path))?
        .decode()
        .context("Image decoding failed")?
        .to_rgb8();

    let (orig_w, orig_h) = img.dimensions();

    const PATCH_SIZE: usize = 14;
    const SPATIAL_MERGE: usize = 2;
    const FACTOR: usize = PATCH_SIZE * SPATIAL_MERGE; // 28
    const MIN_PIXELS: usize = 147_384;
    // const MAX_PIXELS: usize = 2_822_400;
    // const MAX_PIXELS: usize = 2_073_600;
    const MAX_PIXELS: usize = 1_473_600;

    let (resized_h, resized_w) = smart_resize(
        orig_h as usize,
        orig_w as usize,
        FACTOR,
        MIN_PIXELS,
        MAX_PIXELS,
    )?;

    let raw = img.into_raw();
    let mut tensor = (Tensor::from_vec(raw, (orig_h as usize, orig_w as usize, 3), device)?
        .to_dtype(DType::F32)?
        .permute((2, 0, 1))?
        / 255.0)?;

    tensor = match mode {
        ResizeMode::Bilinear => {
            let t = tensor.unsqueeze(0)?;
            t.upsample_bilinear2d(resized_h, resized_w, false)?
        }
        ResizeMode::Nearest => {
            let t = tensor.unsqueeze(0)?;
            t.upsample_nearest2d(resized_h, resized_w)?
        }
        ResizeMode::Bicubic => {
            return Err(E::msg("Bicubic resize not implemented yet"));
        }
    };

    let h_patches = (resized_h / PATCH_SIZE) as u32;
    let w_patches = (resized_w / PATCH_SIZE) as u32;
    let grid_thw = Tensor::new(&[[1u32, h_patches, w_patches]], device)?;

    Ok((tensor, grid_thw))
}

pub fn smart_resize(
    h: usize,
    w: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    let mut height = h;
    let mut width = w;

    // 避免太小
    if height < factor {
        width = width * factor / height.max(1);
        height = factor;
    }
    if width < factor {
        height = height * factor / width.max(1);
        width = factor;
    }

    let mut h_bar = ((height + factor / 2) / factor) * factor;
    let mut w_bar = ((width + factor / 2) / factor) * factor;

    let pixels = h_bar * w_bar;

    if pixels > max_pixels {
        let scale = (pixels as f64 / max_pixels as f64).sqrt();
        h_bar = ((height as f64 / scale / factor as f64).floor() as usize).max(1) * factor;
        w_bar = ((width as f64 / scale / factor as f64).floor() as usize).max(1) * factor;
    } else if pixels < min_pixels {
        let scale = (min_pixels as f64 / pixels as f64).sqrt();
        h_bar = ((height as f64 * scale / factor as f64).ceil() as usize) * factor;
        w_bar = ((width as f64 * scale / factor as f64).ceil() as usize) * factor;
    }

    if (h_bar as f64 / w_bar as f64).max(w_bar as f64 / h_bar as f64) > 200.0 {
        return Err(E::msg("Aspect ratio too extreme after resize"));
    }

    Ok((h_bar, w_bar))
}
