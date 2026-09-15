#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::io::Write;

use anyhow::{Error as E, Result};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
// TODO(candle-transformers-removal): Generation helpers only; see CANDLE_TRANSFORMERS.md.
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

use super::modeling::{Config, HunYuanDenseV1};
use crate::generation::GenerationConfig;
use crate::generation::based::ModelForCausalLM;
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::tokenizer_utils;
use crate::utils::utils;

/// Format of model weights on disk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelFormat {
    /// Auto-detect from path (default).
    Auto,
    /// Standard `HuggingFace` safetensors.
    Safetensors,
    /// GGUF quantized format.
    Gguf,
}

pub struct Model {
    pub tokenizer: TokenOutputStream,
    pub device: Device,
    pub dtype: DType,
    inner: HunYuanDenseV1,
}

impl Model {
    /// # Errors
    ///
    /// Returns an error if the model or tokenizer cannot be loaded from `model_path`.
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        Self::new_with_format(model_path, device, dtype, ModelFormat::Auto)
    }

    /// # Errors
    ///
    /// Returns an error if the model or tokenizer cannot be loaded from `model_path`.
    pub fn new_with_format(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        format: ModelFormat,
    ) -> Result<Self> {
        let format = match format {
            ModelFormat::Auto => {
                let p = std::path::Path::new(model_path);
                if p.is_file() && p.extension().is_some_and(|e| e == "gguf") {
                    ModelFormat::Gguf
                } else {
                    ModelFormat::Safetensors
                }
            },
            other => other,
        };

        match format {
            ModelFormat::Gguf | ModelFormat::Auto => Self::from_gguf(model_path, device),
            ModelFormat::Safetensors => Self::from_pretrained(model_path, device, *dtype),
        }
    }

    fn forward(&mut self, xs: &Tensor, s: usize) -> candle_core::Result<Tensor> {
        self.inner.forward(xs, s)
    }

    pub fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }

    fn from_pretrained(model_path: &str, device: &Device, dtype: DType) -> Result<Model> {
        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer not found at {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let filenames = utils::get_safetensors_files(model_path)?;

        // Load safetensors with explicit dtype casting.
        // The model weights are stored in BF16, which is unsupported for CPU matmul.
        // We use mmap loading but the VarBuilder's dtype field ensures each tensor
        // is cast to the requested dtype (e.g. F32) when accessed.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, device) }?;

        let config_file = std::path::Path::new(model_path).join("config.json");
        let config_data = std::fs::read(config_file)?;
        let config: Config = serde_json::from_slice(&config_data)?;

        let inner = HunYuanDenseV1::new(&config, &vb)?;

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype,
            inner,
        })
    }

    /// Load a GGUF quantized model file.
    ///
    /// `model_path` should point to a `.gguf` file. The tokenizer is read
    /// from the GGUF metadata if present; a sibling `tokenizer.json` is
    /// only consulted as a fallback.
    fn from_gguf(model_path: &str, device: &Device) -> Result<Model> {
        let gguf_path = std::path::Path::new(model_path);

        // Open and parse GGUF
        let mmap = crate::quantized::gguf_file::mmap_gguf_file(gguf_path)?;
        let mut cursor = std::io::Cursor::new(mmap.as_ref());
        let ct = candle_core::quantized::gguf_file::Content::read(&mut cursor)?;

        eprintln!(
            "GGUF loaded: {} tensors, {} metadata entries",
            ct.tensor_infos.len(),
            ct.metadata.len(),
        );

        let tokenizer = tokenizer_utils::resolve_gguf_tokenizer(&ct, gguf_path)?;

        let inner = HunYuanDenseV1::from_gguf(ct, &mut cursor, device)?;
        let dtype = inner.model_dtype();

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype,
            inner,
        })
    }

    /// # Errors
    ///
    /// Returns an error if tokenization fails.
    pub fn prepare_inputs(&self, inputs: &str) -> Result<Vec<u32>> {
        let input_ids = self
            .tokenizer
            .tokenizer
            .encode(inputs, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        Ok(input_ids)
    }

    /// Format a user message with the Hunyuan chat template.
    /// Returns: `<BOS><User>message<Assistant>`
    pub fn format_chat(&self, user_message: &str) -> String {
        format!(
            "<\u{ff5c}hy_begin\u{2581}of\u{2581}sentence\u{ff5c}>\
             <\u{ff5c}hy_User\u{ff5c}>{user_message}\
             <\u{ff5c}hy_Assistant\u{ff5c}>"
        )
    }

    /// Tokenize a user message with the Hunyuan chat template applied.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization fails.
    pub fn prepare_chat(&self, user_message: &str) -> Result<Vec<u32>> {
        let formatted = self.format_chat(user_message);
        self.prepare_inputs(&formatted)
    }

    /// Run a single forward step, returning raw logits. Caller manages KV cache.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward_step(
        &mut self,
        input_ids: &[u32],
        start_pos: usize,
    ) -> candle_core::Result<Tensor> {
        let input = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        self.inner.forward(&input, start_pos)
    }

    /// Pad per-sequence KV caches and load them into the model's attention layers
    /// as a batched `[N, kv_heads, max_kv_len, head_dim]` tensor.
    ///
    /// Returns `(kv_lens, max_kv_len)` for building masks and later extraction.
    ///
    /// `extra_room`: number of decode tokens to pre-allocate in the KV buffer
    /// (avoids `Tensor::cat` reallocation during multi-round decode).
    ///
    /// # Errors
    ///
    /// Returns an error if padding or stacking the KV caches fails.
    pub fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<(Tensor, Tensor)>>],
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        self.inner.setup_batch_decode(seq_kv_caches, extra_room)
    }

    /// Run one batched decode step using the model's existing batched KV cache.
    ///
    /// Builds the `[N, 1]` input tensor from `tokens` (one per sequence).
    /// Returns logits `[N, 1, vocab]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
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

    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
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

    /// Extract per-sequence KV caches from the model's batched state,
    /// removing padding. Clears model KV cache afterward.
    ///
    /// # Errors
    ///
    /// Returns an error if extracting the KV caches fails.
    pub fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<super::modeling::BatchKvCache> {
        self.inner
            .extract_batch_kv(kv_lens, original_max_kv, rounds_done)
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    /// Extract per-layer KV caches (cheap Arc clone).
    pub fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.inner.get_kv_caches()
    }

    /// Restore per-layer KV caches.
    pub fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        self.inner.set_kv_caches(caches);
    }

    /// Total bytes held by the model's KV caches (no GPU copies).
    pub fn active_kv_cache_bytes(&self) -> u64 {
        self.inner.active_kv_cache_bytes()
    }

    pub fn warmup(&mut self) {
        if let Err(e) = self.generate(&[45, 546, 456], &GenerationConfig::with_max_tokens(5), None)
        {
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
        config: &crate::generation::GenerationConfig,
        mut streamer: Option<&mut dyn crate::generation::streamer::TokenStreamer>,
    ) -> Result<Vec<u32>> {
        self.tokenizer.clear();
        self.clear_kv_cache();

        let mut logits_processor = LogitsProcessor::new(1024, config.temperature, config.top_p);

        let mut tokens = input_ids.to_vec();
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        // Hunyuan uses eos_token_id = 120020
        let eos_token = config.eos_token_id.unwrap_or(120_020);
        let mut streamer_finalized = false;

        let start_gen = std::time::Instant::now();
        for index in 0..config.max_new_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            let logits = self.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?;

            // Greedy + no repetition penalty: GPU argmax avoids D→H logits copy.
            // repetition_penalty is compared against the exact default sentinel 1.0,
            // not a computed value, so exact float equality is intentional here.
            #[allow(clippy::float_cmp)]
            let no_repetition_penalty = config.repetition_penalty == 1.;
            let next_token = if config.temperature.is_none() && no_repetition_penalty {
                crate::ops::gpu_argmax(&logits)?
            } else {
                let logits = logits.to_dtype(DType::F32)?;
                let logits = if no_repetition_penalty {
                    logits
                } else {
                    let start_at = tokens.len().saturating_sub(config.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        config.repetition_penalty,
                        &tokens[start_at..],
                    )?
                };
                logits_processor.sample(&logits)?
            };
            tokens.push(next_token);
            generated_tokens += 1;

            if next_token == eos_token {
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
        if let Some(ref mut s) = streamer
            && !streamer_finalized
        {
            s.finalize()?;
        }

        if config.report_speed {
            // generated_tokens is a small per-request token count; f64 has ample
            // precision for it, this is purely a display metric.
            #[allow(clippy::cast_precision_loss)]
            let tokens_per_sec = generated_tokens as f64 / dt.as_secs_f64();
            println!("\n{generated_tokens} tokens generated ({tokens_per_sec:.2} token/s)\n");
        }

        Ok(tokens)
    }
}
