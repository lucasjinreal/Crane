use std::path::Path;

use anyhow::Context;
use anyhow::Result;

pub fn print_candle_build_info() {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );
}

pub fn get_safetensors_files(model_path: &str) -> Result<Vec<std::path::PathBuf>> {
    let model_dir = Path::new(model_path);

    let index_file = model_dir.join("model.safetensors.index.json");
    if index_file.exists() {
        let index_data =
            std::fs::read_to_string(&index_file).context("Failed to read index.json")?;
        let json: serde_json::Value =
            serde_json::from_str(&index_data).context("Invalid index.json format")?;

        let weight_map = json["weight_map"]
            .as_object()
            .context("Missing weight_map in index.json")?;

        let mut files = weight_map
            .values()
            .filter_map(|v| v.as_str())
            .map(|s| model_dir.join(s))
            .collect::<Vec<_>>();

        files.sort_unstable();
        files.dedup();

        for file in &files {
            if !file.exists() {
                return Err(anyhow::anyhow!("Missing shard file: {}", file.display()));
            }
        }
        return Ok(files);
    }

    let single_file = model_dir.join("model.safetensors");
    if single_file.exists() {
        return Ok(vec![single_file]);
    }

    Err(anyhow::anyhow!(
        "No valid model files found in: {}. \
         Need either model.safetensors.index.json or model.safetensors",
        model_dir.display()
    ))
}


