use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use kornia::image::{Image, ImageSize};
use kornia::imgproc::interpolation::InterpolationMode;
use kornia::imgproc::resize::resize_native;
use kornia::io::functional as F;

use kornia::tensor::CpuAllocator;
use std::path::Path;

pub fn load_image_and_smart_resize(
    path: &Path,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    // 1. 用 kornia 读图（RGB8，HWC，连续内存）
    let img_u8: Image<u8, 3, CpuAllocator> = F::read_image_any_rgb8(path)?;
    let size = img_u8.size();

    let img_f32: Image<f32, 3, CpuAllocator> = img_u8.cast_and_scale(1.0 / 255.0)?;

    const PATCH_SIZE: usize = 14;
    const SPATIAL_MERGE: usize = 2;
    const FACTOR: usize = PATCH_SIZE * SPATIAL_MERGE; // 28
    const MIN_PIXELS: usize = 147_384;
    const MAX_PIXELS: usize = 2_822_400;

    let (width, height) = (size.width as usize, size.height as usize);

    let (resized_h, resized_w) = smart_resize(height, width, FACTOR, MIN_PIXELS, MAX_PIXELS)?;

    // 3. resize（kornia 的强项）
    let mut resized = Image::<f32, 3, CpuAllocator>::from_size_val(
        ImageSize {
            width: resized_w,
            height: resized_h,
        },
        0.0,
        CpuAllocator,
    )?;

    resize_native(&img_f32, &mut resized, InterpolationMode::Bicubic)?;

    let raw: Vec<f32> = resized.to_vec();

    // H * W
    let hw = resized_h * resized_w;

    // CHW buffer
    let mut buf = vec![0f32; 3 * hw];

    // HWC -> CHW + normalize [-1, 1]
    for i in 0..hw {
        let base = i * 3;

        buf[i] = raw[base] * 2.0 - 1.0; // R
        buf[hw + i] = raw[base + 1] * 2.0 - 1.0; // G
        buf[2 * hw + i] = raw[base + 2] * 2.0 - 1.0; // B
    }

    let pixel_values =
        Tensor::from_vec(buf, (1, 3, resized_h, resized_w), device)?.to_dtype(dtype)?;

    let h_patches = (resized_h / PATCH_SIZE) as u32;
    let w_patches = (resized_w / PATCH_SIZE) as u32;
    let grid_thw = Tensor::new(&[[1u32, h_patches, w_patches]], device)?;

    Ok((pixel_values, grid_thw))
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
