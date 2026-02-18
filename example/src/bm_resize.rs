use anyhow::Result;
use crane_core::models::{DType, Device};
use crane_core::utils::image_utils;
use image::{ImageBuffer, Rgb};
use std::{env, path::Path};

fn main() -> Result<()> {
    // Create a sample tensor

    let args: Vec<String> = env::args().collect();
    let image_path = args.get(1).map(|s| s.as_str()).unwrap_or("test.jpg");

    let device = Device::cuda_if_available(0)?;

    let target_h = 512;
    let target_w = 512;

    let resized = image_utils::load_and_resize_image_to_tensor(
        image_path,
        target_h,
        target_w,
        image_utils::ResizeMode::Bilinear,
        &device,
    )
    .unwrap();

    println!("output shape: {:?}", resized.shape());

    // saving resized tensor to image

    let hwc = resized.squeeze(0)?.permute((1, 2, 0))?;
    let pixels = (hwc * 255.0)?
        .clamp(0.0, 255.0)?
        .to_dtype(DType::U8)?
        .flatten_all()?
        .to_vec1::<u8>()?;

    let out_img =
        ImageBuffer::<Rgb<u8>, _>::from_vec(target_h as u32, target_w as u32, pixels).unwrap();

    // Save with _resized suffix
    let out_path = Path::new(&image_path).with_file_name(format!(
        "{}_resized.jpg",
        Path::new(&image_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("image")
    ));

    out_img.save(&out_path)?;
    println!("Saved resized image to: {}", out_path.to_str().unwrap());

    Ok(())
}
