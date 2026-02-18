use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::paddleocr_vl::{Config, PaddleOCRVLModel};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;

use crate::utils::image_utils;

pub struct PaddleOcrVL {
    model: Arc<PaddleOCRVLModel>,
    tokenizer: Tokenizer,
    pub device: Device,
    dtype: DType,
    config: Config,
    eos_token_id: u32,
    image_token_id: u32,
    vision_start_token_id: u32,
    vision_end_token_id: u32,
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

pub trait PaddleOCRVLGenerateStream {
    fn generate_stream<F>(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        grid_thw: &Tensor,
        max_new_tokens: usize,
        eos_token_id: u32,
        on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32);
}

impl PaddleOCRVLGenerateStream for PaddleOCRVLModel {
    fn generate_stream<F>(
        &mut self,
        input_ids: &Tensor,
        pixel_values: &Tensor,
        grid_thw: &Tensor,
        max_new_tokens: usize,
        eos_token_id: u32,
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32),
    {
        self.clear_kv_cache();

        let mut generated_tokens = Vec::new();
        let mut current_ids = input_ids.clone();

        let logits = self.forward(&current_ids, Some(pixel_values), Some(grid_thw), 0)?;
        let next_token = logits
            .argmax(D::Minus1)?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?[0];

        on_token(next_token);
        generated_tokens.push(next_token);

        if next_token == eos_token_id {
            return Ok(generated_tokens);
        }

        let mut seqlen_offset = current_ids.dim(1)?;
        current_ids = Tensor::new(&[next_token], current_ids.device())?.unsqueeze(0)?;

        for _ in 1..max_new_tokens {
            let logits = self.forward(&current_ids, None, None, seqlen_offset)?;
            let next_token = logits
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?[0];

            on_token(next_token);
            generated_tokens.push(next_token);

            if next_token == eos_token_id {
                break;
            }

            seqlen_offset += 1;
            current_ids = Tensor::new(&[next_token], current_ids.device())?.unsqueeze(0)?;
        }

        Ok(generated_tokens)
    }
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

        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.unwrap_or("main").to_string(),
        ));

        let config: Config =
            serde_json::from_str(&std::fs::read_to_string(repo.get("config.json")?)?)?;

        let tokenizer = Tokenizer::from_file(repo.get("tokenizer.json")?).map_err(E::msg)?;

        let model_file = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))?;

        let vb = if model_file.extension().map_or(false, |e| e == "bin") {
            VarBuilder::from_pth(&model_file, dtype, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[&model_file], dtype, &device)? }
        };

        let model = PaddleOCRVLModel::new(&config, vb)?;

        let eos_token_id = tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|end_of_sentence|>"))
            .or_else(|| tokenizer.token_to_id(""))
            .unwrap_or(2);

        Ok(Self {
            model: Arc::new(model),
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

        let base = path.as_ref();
        let config: Config =
            serde_json::from_str(&std::fs::read_to_string(base.join("config.json"))?)?;
        let tokenizer = Tokenizer::from_file(base.join("tokenizer.json")).map_err(E::msg)?;

        let safetensors = base.join("model.safetensors");
        let vb = if safetensors.exists() {
            unsafe { VarBuilder::from_mmaped_safetensors(&[&safetensors], dtype, &device)? }
        } else {
            VarBuilder::from_pth(&base.join("pytorch_model.bin"), dtype, &device)?
        };

        let model = PaddleOCRVLModel::new(&config, vb)?;

        let eos_token_id = tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|end_of_sentence|>"))
            .unwrap_or(2);

        Ok(Self {
            model: Arc::new(model),
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
        let merge = self.config.vision_config.spatial_merge_size as usize;
        let num_image_tokens = (g[1] as usize / merge) * (g[2] as usize / merge);

        let input_ids = build_input_tokens(
            &self.tokenizer,
            task,
            num_image_tokens,
            self.image_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
            &self.device,
        )?;

        let model = Arc::get_mut(&mut self.model).expect("recognize called while model is shared");
        model.clear_kv_cache();

        let generated = model.generate(
            &input_ids,
            &pixel_values,
            &grid_thw,
            max_new_tokens,
            self.eos_token_id,
        )?;

        let output: Vec<u32> = generated
            .into_iter()
            .take_while(|&t| t != self.eos_token_id)
            .collect();

        let mut text = self
            .tokenizer
            .decode(&output, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?
            .trim()
            .to_string();

        if text.is_empty() {
            text = "[no text recognized]".to_string();
        }

        Ok(OcrResult {
            text,
            tokens_generated: output.len(),
            duration_secs: start.elapsed().as_secs_f32(),
        })
    }

    pub fn recognize_stream<F>(
        &mut self,
        image_path: impl AsRef<Path>,
        task: OcrTask,
        max_new_tokens: usize,
        mut callback: F,
    ) -> Result<OcrResult>
    where
        F: FnMut(&str),
    {
        let start = Instant::now();

        let (pixel_values, grid_thw) = load_image(image_path.as_ref(), &self.device, self.dtype)?;

        let grid: Vec<Vec<u32>> = grid_thw.to_vec2().map_err(|e| E::msg(e.to_string()))?;
        let g = &grid[0];
        let merge = self.config.vision_config.spatial_merge_size as usize;
        let num_image_tokens = (g[1] as usize / merge) * (g[2] as usize / merge);

        let input_ids = build_input_tokens(
            &self.tokenizer,
            task,
            num_image_tokens,
            self.image_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
            &self.device,
        )?;

        let eos_token_id = self.eos_token_id;
        let tokenizer = self.tokenizer.clone();

        let mut text = String::new();
        let mut tokens_generated = 0usize;

        let model =
            Arc::get_mut(&mut self.model).expect("recognize_stream called while model is shared");

        let tokens = model.generate_stream(
            &input_ids,
            &pixel_values,
            &grid_thw,
            max_new_tokens,
            eos_token_id,
            |tok_id| {
                tokens_generated += 1;

                if tok_id == eos_token_id {
                    return;
                }

                if let Ok(s) = tokenizer.decode(&[tok_id], false) {
                    callback(&s);
                    text.push_str(&s);
                }
            },
        )?;

        if text.trim().is_empty() {
            text = "[no text recognized]".to_string();
        }

        Ok(OcrResult {
            text,
            tokens_generated: tokens.len(),
            duration_secs: start.elapsed().as_secs_f32(),
        })
    }
}

pub fn load_image(path: &Path, device: &Device, dtype: DType) -> Result<(Tensor, Tensor)> {
    image_utils::load_image_and_smart_resize(path, device, dtype, image_utils::ResizeMode::Bilinear)
}

pub fn build_input_tokens(
    tokenizer: &Tokenizer,
    task: OcrTask,
    num_image_tokens: usize,
    image_token_id: u32,
    _vision_start_token_id: u32,
    _vision_end_token_id: u32,
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
        tokens.extend(
            tokenizer
                .encode(part, false)
                .map_err(|e| E::msg(format!("Tokenizer encode failed: {}", e)))?
                .get_ids()
                .iter()
                .copied(),
        );
    }

    tokens.extend(vec![image_token_id; num_image_tokens]);

    for &part in &parts[2..] {
        // <|image_end|> + prompt + \nAssistant:
        tokens.extend(
            tokenizer
                .encode(part, false)
                .map_err(|e| E::msg(format!("Tokenizer encode failed: {}", e)))?
                .get_ids()
                .iter()
                .copied(),
        );
    }

    Tensor::new(tokens.as_slice(), device)?
        .unsqueeze(0)
        .map_err(|e| E::msg(format!("Tensor new failed: {}", e)))
}
