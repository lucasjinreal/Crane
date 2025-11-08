#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::io::Write;

use anyhow::{Error as E, Result};

use candle_transformers::models::qwen2::{Config as ConfigBase, ModelForCausalLM as ModelBase};
use candle_transformers::models::qwen2_moe::{Config as ConfigMoe, Model as ModelMoe};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

use crate::generation::based::ModelForCausalLM;
use crate::generation::GenerationConfig;
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::utils;

pub struct TextGeneration {
    pub model: Model,
    pub device: Device,
    pub tokenizer: TokenOutputStream,
    pub logits_processor: LogitsProcessor,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    pub fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.model.clear_kv_cache();
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

pub struct Model {
    pub tokenizer: TokenOutputStream,
    pub device: Device,
    model_typed: ModelTyped,
}

pub enum ModelTyped {
    Base(ModelBase),
    Moe(ModelMoe),
}

impl Model {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        Self::from_pretrained(model_path, device, dtype)
    }

    fn forward(&mut self, xs: &Tensor, s: usize) -> candle_core::Result<Tensor> {
        match self.model_typed {
            ModelTyped::Moe(ref mut m) => m.forward(xs, s),
            ModelTyped::Base(ref mut m) => m.forward(xs, s),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match self.model_typed {
            ModelTyped::Moe(ref mut m) => m.clear_kv_cache(),
            ModelTyped::Base(ref mut m) => m.clear_kv_cache(),
        }
    }

    fn from_pretrained(model_path: &str, device: &Device, dtype: &DType) -> Result<Model> {
        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer not found at {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let filenames = utils::get_safetensors_files(model_path)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, *dtype, device) }?;

        let config_file = std::path::Path::new(model_path).join("config.json");
        let config_data = std::fs::read(config_file)?;
        let config: ConfigBase = serde_json::from_slice(&config_data)?;

        let model_typed = ModelTyped::Base(ModelBase::new(&config, vb)?);

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            model_typed,
        })
    }

    pub fn prepare_inputs(&self, inputs: &str) -> Result<Vec<u32>> {
        let input_ids = self
            .tokenizer
            .tokenizer
            .encode(inputs, true)
            .map_err(E::msg)
            .unwrap()
            .get_ids()
            .to_vec();
        Ok(input_ids)
    }

    pub fn warmup(&mut self) {
        let _ = self
            .generate(
                &[45, 546, 456],
                &GenerationConfig::with_max_tokens(20),
                None,
            )
            .unwrap();
    }
}

impl ModelForCausalLM for Model {
    fn device(&self) -> &Device {
        &self.device
    }

    fn generate(
        &mut self,
        input_ids: &[u32],
        config: &crate::generation::GenerationConfig,
        mut streamer: Option<&mut dyn crate::generation::streamer::TokenStreamer>,
    ) -> Result<Vec<u32>> {
        self.tokenizer.clear();
        self.clear_kv_cache();

        let mut logits_processor = LogitsProcessor::new(1024, config.temperature, config.top_p);

        let mut tokens = input_ids.to_vec();
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };
        let start_gen = std::time::Instant::now();
        for index in 0..config.max_new_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            let logits = self.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if config.repetition_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(config.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    config.repetition_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;

        if config.report_speed {
            println!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)\n",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
        }

        Ok(tokens)
    }
}
