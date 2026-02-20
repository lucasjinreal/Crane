#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::io::Write;

use anyhow::{Error as E, Result};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

use super::modeling::{Config, Qwen3Model};
use crate::generation::based::ModelForCausalLM;
use crate::generation::GenerationConfig;
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::utils;

/// Format of model weights on disk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelFormat {
    /// Auto-detect from path (default).
    Auto,
    /// Standard HuggingFace safetensors.
    Safetensors,
    /// GGUF quantized format.
    Gguf,
}

pub struct Model {
    pub tokenizer: TokenOutputStream,
    pub device: Device,
    pub dtype: DType,
    inner: Qwen3Model,
}

impl Model {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        Self::new_with_format(model_path, device, dtype, ModelFormat::Auto)
    }

    pub fn new_with_format(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        format: ModelFormat,
    ) -> Result<Self> {
        let format = match format {
            ModelFormat::Auto => {
                let p = std::path::Path::new(model_path);
                if p.is_file()
                    && p.extension()
                        .map(|e| e == "gguf")
                        .unwrap_or(false)
                {
                    ModelFormat::Gguf
                } else {
                    ModelFormat::Safetensors
                }
            }
            other => other,
        };

        match format {
            ModelFormat::Gguf | ModelFormat::Auto => Self::from_gguf(model_path, device),
            ModelFormat::Safetensors => Self::from_pretrained(model_path, device, dtype),
        }
    }

    fn forward(&mut self, xs: &Tensor, s: usize) -> candle_core::Result<Tensor> {
        self.inner.forward(xs, s)
    }

    pub fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
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
        let config: Config = serde_json::from_slice(&config_data)?;

        let inner = Qwen3Model::new(&config, vb)?;

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype: *dtype,
            inner,
        })
    }

    /// Load a GGUF quantized model file.
    fn from_gguf(model_path: &str, device: &Device) -> Result<Model> {
        let gguf_path = std::path::Path::new(model_path);

        let tokenizer_path = {
            let same_dir = gguf_path
                .parent()
                .unwrap_or(gguf_path)
                .join("tokenizer.json");
            if same_dir.exists() {
                same_dir
            } else {
                let parent = gguf_path
                    .parent()
                    .and_then(|p| p.parent())
                    .unwrap_or(gguf_path)
                    .join("tokenizer.json");
                if parent.exists() {
                    parent
                } else {
                    anyhow::bail!(
                        "Cannot find tokenizer.json near {}. \
                         Place tokenizer.json in the same directory as the GGUF file.",
                        gguf_path.display()
                    );
                }
            }
        };
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let mut file = std::fs::File::open(gguf_path)?;
        let ct = candle_core::quantized::gguf_file::Content::read(&mut file)?;

        eprintln!(
            "GGUF loaded: {} tensors, {} metadata entries",
            ct.tensor_infos.len(),
            ct.metadata.len(),
        );

        let inner = Qwen3Model::from_gguf(ct, &mut file, device)?;
        let dtype = inner.model_dtype();

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype,
            inner,
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

    /// Run a single forward step, returning raw logits.
    pub fn forward_step(
        &mut self,
        input_ids: &[u32],
        start_pos: usize,
    ) -> candle_core::Result<Tensor> {
        let input = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        self.inner.forward(&input, start_pos)
    }

    // ── KV cache management (for continuous-batching engine) ────────────

    pub fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    pub fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.inner.get_kv_caches()
    }

    pub fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        self.inner.set_kv_caches(caches);
    }

    /// Total bytes held by the model's KV caches (no GPU copies).
    pub fn active_kv_cache_bytes(&self) -> u64 {
        self.inner.active_kv_cache_bytes()
    }

    // ── Batched decode (GPU-efficient concurrent serving) ───────────────

    pub fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        self.inner.setup_batch_decode(seq_kv_caches, extra_room)
    }

    pub fn step_batch_decode(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        let n = positions.len();
        let input = Tensor::new(tokens, &self.device)?.reshape((n, 1))?;
        self.inner
            .step_batch_decode(&input, positions, attention_mask, batch_kv_info)
    }

    pub fn step_batch_decode_with_input_ids(
        &mut self,
        input_ids: &Tensor,
        positions: &[usize],
        attention_mask: Option<&Tensor>,
        batch_kv_info: Option<(&[usize], usize)>,
    ) -> candle_core::Result<Tensor> {
        self.inner
            .step_batch_decode(input_ids, positions, attention_mask, batch_kv_info)
    }

    pub fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        self.inner
            .extract_batch_kv(kv_lens, original_max_kv, rounds_done)
    }

    pub fn warmup(&mut self) {
        if let Err(e) = self.generate(
            &[45, 546, 456],
            &GenerationConfig::with_max_tokens(5),
            None,
        ) {
            eprintln!("warmup failed (non-fatal): {e}");
        }
        self.clear_kv_cache();
    }
}

impl ModelForCausalLM for Model {
    fn device(&self) -> &Device {
        &self.device
    }

    fn generate(
        &mut self,
        input_ids: &[u32],
        config: &GenerationConfig,
        mut streamer: Option<&mut dyn crate::generation::streamer::TokenStreamer>,
    ) -> Result<Vec<u32>> {
        self.tokenizer.clear();
        self.clear_kv_cache();

        let mut logits_processor = LogitsProcessor::new(1024, config.temperature, config.top_p);

        let mut tokens = input_ids.to_vec();
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        // Qwen3 can stop on <|im_end|> (151645) or <|endoftext|> (151643).
        let eos_token: Option<u32> = config
            .eos_token_id
            .or_else(|| self.tokenizer.get_token("<|im_end|>"))
            .or_else(|| self.tokenizer.get_token("<|endoftext|>"));
        let mut streamer_finalized = false;

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

            if eos_token == Some(next_token) {
                if let Some(ref mut s) = streamer {
                    s.finalize()?;
                }
                streamer_finalized = true;
                break;
            }

            if let Some(ref mut s) = streamer {
                s.append(next_token)?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(ref mut s) = streamer {
            if !streamer_finalized {
                s.finalize()?;
            }
        }

        if config.report_speed {
            println!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)\n",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
        }

        Ok(tokens)
    }
}