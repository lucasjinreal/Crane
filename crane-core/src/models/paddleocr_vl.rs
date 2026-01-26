use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::paddleocr_vl::{Config, PaddleOCRVLModel};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokenizers::Tokenizer;

use crate::utils::image_utils;

pub struct PaddleOcrVL {
    model: PaddleOCRVLModel,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    config: Config,
    eos_token_id: u32,
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    // video_token_id: u32,  // 目前 PaddleOCR-VL video 支援仍實驗性，先註解
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OcrTask {
    Ocr,
    Table,
    Formula,
    Chart,
}

impl OcrTask {
    pub fn prompt(&self) -> &'static str {
        match self {
            OcrTask::Ocr => "OCR:",
            OcrTask::Table => "Table Recognition:",
            OcrTask::Formula => "Formula Recognition:",
            OcrTask::Chart => "Chart Recognition:",
        }
    }
}

#[derive(Debug, Clone)]
pub struct OcrResult {
    pub text: String,
    pub tokens_generated: usize,
    pub duration_secs: f32,
}

impl PaddleOcrVL {
    pub fn from_pretrained(
        model_id: &str,
        revision: Option<&str>,
        cpu: bool,
        bf16: bool,
    ) -> Result<Self> {
        let device = if cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        let dtype = if bf16 && device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        println!(
            "Loading PaddleOCR-VL from HF: {} @ {}",
            model_id,
            revision.unwrap_or("main")
        );

        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.unwrap_or("main").to_string(),
        ));

        let config_path = repo.get("config.json")?;
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;

        let tokenizer_path = repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| E::msg(format!("Failed to load tokenizer: {}", e)))?;

        let model_file = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))?;

        println!("Loading weights from: {:?}", model_file.display());

        let vb = if model_file.extension().map_or(false, |ext| ext == "bin") {
            VarBuilder::from_pth(&model_file, dtype, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[&model_file], dtype, &device)? }
        };

        let model = PaddleOCRVLModel::new(&config, vb)?;

        let eos_token_id = tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|end_of_sentence|>"))
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .unwrap_or(2);

        Ok(Self {
            model,
            tokenizer,
            device,
            dtype,
            config: config.clone(),
            eos_token_id,
            image_token_id: config.image_token_id,
            vision_start_token_id: config.vision_start_token_id,
            vision_end_token_id: config.vision_end_token_id,
        })
    }

    pub fn from_local(path: impl AsRef<Path>, cpu: bool, bf16: bool) -> Result<Self> {
        let device = if cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };
        let dtype = if bf16 && device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let base = path.as_ref().to_path_buf();
        if !base.is_dir() {
            return Err(E::msg(format!("Not a directory: {:?}", base)));
        }

        let config: Config =
            serde_json::from_str(&std::fs::read_to_string(base.join("config.json"))?)?;
        let tokenizer = Tokenizer::from_file(base.join("tokenizer.json")).map_err(E::msg)?;

        let safetensors_path = base.join("model.safetensors");
        let vb = if safetensors_path.exists() {
            unsafe { VarBuilder::from_mmaped_safetensors(&[&safetensors_path], dtype, &device)? }
        } else {
            let pth_path = base.join("pytorch_model.bin");
            if pth_path.exists() {
                VarBuilder::from_pth(&pth_path, dtype, &device)?
            } else {
                return Err(E::msg(
                    "Neither model.safetensors nor pytorch_model.bin found",
                ));
            }
        };

        let model = PaddleOCRVLModel::new(&config, vb)?;

        let eos_token_id = tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|end_of_sentence|>"))
            .unwrap_or(2);

        Ok(Self {
            model,
            tokenizer,
            device,
            dtype,
            config: config.clone(),
            eos_token_id,
            image_token_id: config.image_token_id,
            vision_start_token_id: config.vision_start_token_id,
            vision_end_token_id: config.vision_end_token_id,
        })
    }

    pub fn recognize(
        &mut self,
        image_path: impl AsRef<Path>,
        task: OcrTask,
        max_new_tokens: usize,
    ) -> Result<OcrResult> {
        let start = Instant::now();

        let (pixel_values, grid_thw) = load_image(image_path.as_ref(), &self.device, self.dtype)?;

        let grid_vec: Vec<Vec<u32>> = grid_thw.to_vec2().map_err(|e| E::msg(e.to_string()))?;
        let g = &grid_vec[0];
        let spatial_merge = self.config.vision_config.spatial_merge_size as usize;
        let num_image_tokens = (g[1] as usize / spatial_merge) * (g[2] as usize / spatial_merge);

        let input_ids = build_input_tokens(
            &self.tokenizer,
            task,
            num_image_tokens,
            self.image_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
            &self.device,
        )?;

        self.model.clear_kv_cache();

        let generated = self.model.generate(
            &input_ids,
            &pixel_values,
            &grid_thw,
            max_new_tokens,
            self.eos_token_id,
        )?;

        let output_tokens: Vec<u32> = generated
            .into_iter()
            .take_while(|&t| t != self.eos_token_id)
            .collect();

        let mut text = self
            .tokenizer
            .decode(&output_tokens, true)
            .map_err(|e| anyhow::anyhow!("Tokenizer decode failed: {}", e))?;
        text = text.trim().to_string();

        if text.is_empty() {
            text = "[no text recognized]".to_string();
        }

        let duration = start.elapsed().as_secs_f32();

        Ok(OcrResult {
            text,
            tokens_generated: output_tokens.len(),
            duration_secs: duration,
        })
    }

    pub fn recognize_and_print(
        &mut self,
        image_path: impl AsRef<Path>,
        task: OcrTask,
        max_new_tokens: usize,
    ) -> Result<()> {
        let result = self.recognize(image_path, task, max_new_tokens)?;
        println!("\n{}", "=".repeat(70));
        println!("Task          : {:?}", task);
        println!("Duration      : {:.2} s", result.duration_secs);
        println!(
            "Tokens        : {} ({:.1} tok/s)",
            result.tokens_generated,
            result.tokens_generated as f32 / result.duration_secs.max(0.01)
        );
        println!("{}", "=".repeat(70));
        println!("{}", result.text);
        println!("{}", "=".repeat(70));
        Ok(())
    }
}

// ======================== Helper functions ========================

pub fn load_image(path: &Path, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
    // let img = image::ImageReader::open(path)?
    //     .decode()
    //     .map_err(|e| E::msg(format!("Cannot decode image {}: {}", path.display(), e)))?
    //     .to_rgb8();

    // let (width, height) = (img.width() as usize, img.height() as usize);

    // let resized = image::imageops::resize(
    //     &img,
    //     new_w as u32,
    //     new_h as u32,
    //     image::imageops::FilterType::CatmullRom,
    // );

    // let mut buf = vec![0f32; 3 * new_h * new_w];
    // for c in 0..3 {
    //     for y in 0..new_h {
    //         for x in 0..new_w {
    //             let idx = c * new_h * new_w + y * new_w + x;
    //             buf[idx] = resized.get_pixel(x as u32, y as u32)[c] as f32 / 255.0 * 2.0 - 1.0;
    //         }
    //     }
    // }

    // let pixel_values = Tensor::from_vec(buf, (1, 3, new_h, new_w), device)?.to_dtype(dtype)?;

    // let h_patches = (new_h / PATCH_SIZE) as u32;
    // let w_patches = (new_w / PATCH_SIZE) as u32;
    // let grid_thw = Tensor::new(&[[1u32, h_patches, w_patches]], device)?;

    // Ok((pixel_values, grid_thw))

    image_utils::load_image_and_smart_resize(path, device, dtype)
}

pub fn build_input_tokens(
    tokenizer: &Tokenizer,
    task: OcrTask,
    num_image_tokens: usize,
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
    device: &Device,
) -> Result<Tensor> {
    let bos_id = tokenizer.token_to_id("<|begin_of_sentence|>").unwrap_or(1);

    let parts = [
        "User: ",
        "<|image_start|>",
        // image placeholders will be inserted here
        "<|image_end|>",
        task.prompt(),
        "\nAssistant: ",
    ];

    let mut tokens = vec![bos_id];

    for &part in &parts[..2] {
        // User: + <|image_start|>
        tokens.extend(tokenizer.encode(part, false)?.get_ids().iter().copied());
    }

    tokens.extend(vec![image_token_id; num_image_tokens]);

    for &part in &parts[2..] {
        // <|image_end|> + prompt + \nAssistant:
        tokens.extend(tokenizer.encode(part, false)?.get_ids().iter().copied());
    }

    Tensor::new(tokens.as_slice(), device)?
        .unsqueeze(0)
        .map_err(|e| E::msg(format!("Tensor new failed: {}", e)))
}
