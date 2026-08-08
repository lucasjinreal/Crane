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

use super::modeling::{Config, MiniCpm5Model};
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

/// Read EOS token id(s) from `generation_config.json` (preferred) then
/// `config.json`. The field may be a single integer or a list; returns an
/// empty vec if absent. MiniCPM5-1B uses multi-id EOS (`[1, 130073]`).
fn read_eos_token_ids(model_path: &str) -> Vec<u32> {
    fn from_value(v: &serde_json::Value) -> Vec<u32> {
        match v {
            serde_json::Value::Number(n) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            serde_json::Value::Array(a) => a
                .iter()
                .filter_map(|e| e.as_u64().map(|x| x as u32))
                .collect(),
            _ => Vec::new(),
        }
    }
    for fname in ["generation_config.json", "config.json"] {
        let path = std::path::Path::new(model_path).join(fname);
        let Ok(data) = std::fs::read(&path) else {
            continue;
        };
        let Ok(json) = serde_json::from_slice::<serde_json::Value>(&data) else {
            continue;
        };
        if let Some(eos) = json.get("eos_token_id") {
            let ids = from_value(eos);
            if !ids.is_empty() {
                return ids;
            }
        }
    }
    Vec::new()
}

/// Wraps [`MiniCpm5Model`] with a tokenizer and generation loop.
///
/// **Dtype note:** the checkpoint is released in BF16 (`torch_dtype:
/// "bfloat16"` in `config.json`) and must be run in BF16 or F32 — running it
/// in F16 produces all-NaN logits from the very first forward pass (verified
/// empirically: F16 prefill on the 1B checkpoint → 130560/130560 NaN logits;
/// BF16/F32 → clean logits in [-11, 15]). F16's narrower exponent range
/// (vs. BF16/F32) overflows somewhere in this model's activations; BF16 has
/// the same exponent range as F32 so it doesn't hit this. `crane-serve`
/// already defaults to BF16 on CUDA for every model, so this only bites
/// callers that force F16 explicitly.
pub struct Model {
    pub tokenizer: TokenOutputStream,
    pub device: Device,
    pub dtype: DType,
    /// Stop tokens read from `generation_config.json` / `config.json`.
    /// Used when the caller's `GenerationConfig` doesn't pin its own
    /// `eos_token_id`.
    eos_token_ids: Vec<u32>,
    inner: MiniCpm5Model,
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

    /// EOS token id(s) read from `generation_config.json` / `config.json`
    /// (safetensors) at load time. Empty for GGUF checkpoints — the GGUF
    /// path relies on the request's `eos_token_id` or a tokenizer lookup.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    fn from_pretrained(model_path: &str, device: &Device, dtype: &DType) -> Result<Model> {
        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer not found at {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let filenames = utils::get_safetensors_files(model_path)?;

        // Load safetensors with explicit dtype casting.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, *dtype, device) }?;

        let config_file = std::path::Path::new(model_path).join("config.json");
        let config_data = std::fs::read(config_file)?;
        let config: Config = serde_json::from_slice(&config_data)?;

        let inner = MiniCpm5Model::new(&config, vb)?;
        let eos_token_ids = read_eos_token_ids(model_path);

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype: *dtype,
            eos_token_ids,
            inner,
        })
    }

    /// Load a GGUF quantized model file.
    ///
    /// `model_path` should point to a `.gguf` file. The tokenizer is read
    /// from the GGUF itself (`tokenizer.ggml.tokens`/`merges`/`token_type`);
    /// a sibling `tokenizer.json` is only consulted as a fallback for older
    /// or third-party quantizers that lack the embedded metadata.
    fn from_gguf(model_path: &str, device: &Device) -> Result<Model> {
        use crate::utils::tokenizer_utils::{
            build_tokenizer_from_gguf_path, gguf_has_embedded_tokenizer,
        };

        let gguf_path = std::path::Path::new(model_path);
        let parent = gguf_path.parent().unwrap_or(gguf_path);

        // Open and parse GGUF
        let mut file = std::fs::File::open(gguf_path)?;
        let ct = candle_core::quantized::gguf_file::Content::read(&mut file)?;

        eprintln!(
            "GGUF loaded: {} tensors, {} metadata entries",
            ct.tensor_infos.len(),
            ct.metadata.len(),
        );

        // Prefer the embedded tokenizer; fall back to a sibling
        // tokenizer.json only when the GGUF lacks the necessary metadata.
        let tokenizer = if gguf_has_embedded_tokenizer(&ct) {
            build_tokenizer_from_gguf_path(gguf_path)?
                .ok_or_else(|| anyhow::anyhow!("GGUF reports embedded tokenizer but build returned None"))?
        } else {
            let same_dir = parent.join("tokenizer.json");
            let tokenizer_path = if same_dir.exists() {
                same_dir
            } else {
                let grandparent = parent.parent().unwrap_or(parent).join("tokenizer.json");
                if grandparent.exists() {
                    grandparent
                } else {
                    anyhow::bail!(
                        "GGUF lacks `tokenizer.ggml.tokens`/`merges` metadata and no sibling \
                         tokenizer.json was found near {}. Place tokenizer.json next to the \
                         GGUF file, or re-export with a current llama.cpp to embed the tokenizer.",
                        gguf_path.display()
                    );
                }
            };
            eprintln!("GGUF has no embedded tokenizer; falling back to {}", tokenizer_path.display());
            Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?
        };

        // EOS: sibling generation_config.json/config.json wins (may hold the
        // full multi-id set, e.g. MiniCPM5's `[1, 130073]` — GGUF metadata's
        // `tokenizer.ggml.eos_token_id` only ever carries one id, `</s>`,
        // missing the chat template's real turn-end `<|im_end|>`). Always
        // union in `<|im_end|>` from the (now tokenizer-embedded-or-sibling)
        // vocabulary too, so a bare GGUF with no sibling config still stops
        // cleanly instead of running to `max_new_tokens`.
        let mut eos_token_ids = read_eos_token_ids(&parent.to_string_lossy());
        if eos_token_ids.is_empty()
            && let Some(id) = ct
                .metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(|v| v.to_u32().ok())
        {
            eos_token_ids.push(id);
        }
        if let Some(im_end) = tokenizer.token_to_id("<|im_end|>")
            && !eos_token_ids.contains(&im_end)
        {
            eos_token_ids.push(im_end);
        }

        let inner = MiniCpm5Model::from_gguf(ct, &mut file, device)?;
        let dtype = if device.is_cuda() { DType::BF16 } else { DType::F32 };

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype,
            eos_token_ids,
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

    /// Run a single forward step, returning raw logits. Caller manages KV cache.
    pub fn forward_step(
        &mut self,
        input_ids: &[u32],
        start_pos: usize,
    ) -> candle_core::Result<Tensor> {
        let input = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        self.inner.forward(&input, start_pos)
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
        config: &crate::generation::GenerationConfig,
        mut streamer: Option<&mut dyn crate::generation::streamer::TokenStreamer>,
    ) -> Result<Vec<u32>> {
        self.tokenizer.clear();
        self.clear_kv_cache();

        let mut logits_processor = LogitsProcessor::new(1024, config.temperature, config.top_p);

        let mut tokens = input_ids.to_vec();
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        // Stop tokens: an explicit `eos_token_id` on the request wins; otherwise
        // use the model's configured EOS set (MiniCPM5 uses multiple, e.g.
        // [1, 130073]); last-resort, look up `<|im_end|>` in the tokenizer.
        let mut stop_ids: Vec<u32> = match config.eos_token_id {
            Some(e) => vec![e],
            None if !self.eos_token_ids.is_empty() => self.eos_token_ids.clone(),
            None => self.tokenizer.get_token("<|im_end|>").into_iter().collect(),
        };
        stop_ids.sort_unstable();
        stop_ids.dedup();

        let start_gen = std::time::Instant::now();
        let mut finalized = false;
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

            if stop_ids.binary_search(&next_token).is_ok() {
                if let Some(ref mut s) = streamer {
                    s.finalize()?;
                    finalized = true;
                }
                break;
            }

            if let Some(ref mut s) = streamer {
                s.append(next_token)?;
            }
        }
        let dt = start_gen.elapsed();
        if !finalized {
            if let Some(ref mut s) = streamer {
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
