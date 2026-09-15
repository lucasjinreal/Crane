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
use ribo::utils::log;
use tokenizers::Tokenizer;

use super::modeling::{BatchKvCache, Config, Qwen3Model};
use crate::device::{DeviceAssignment, GpuBudget};
use crate::generation::GenerationConfig;
use crate::generation::based::ModelForCausalLM;
use crate::models::modules::quant_kv_cache::{KvCacheKind, KvCacheState};
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
    /// From `GpuBudget::max_concurrent` at load time; used by
    /// [`Self::kv_bytes_per_token`] to decide whether quantized KV pricing
    /// is safe (see its doc comment).
    max_concurrent: Option<usize>,
    inner: Qwen3Model,
}

impl Model {
    /// `gpu_budget` constrains `MoE` expert placement; only consumed once
    /// the checkpoint is `MoE` (see [`crate::device::GpuBudget`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the model files cannot be found or loaded.
    pub fn new(
        model_path: &str,
        devices: &DeviceAssignment,
        dtype: &DType,
        gpu_budget: &GpuBudget,
    ) -> Result<Self> {
        Self::new_with_format(model_path, devices, dtype, ModelFormat::Auto, gpu_budget)
    }

    /// `devices.main` holds every weight but `MoE` experts; `devices.expert`
    /// holds `MoE` expert weights, if the checkpoint is `MoE`. Applies to
    /// both GGUF and safetensors checkpoints. `gpu_budget` constrains `MoE`
    /// expert placement; only consumed once the checkpoint is `MoE` (see
    /// [`crate::device::GpuBudget`]).
    ///
    /// # Errors
    ///
    /// Returns an error if the model files cannot be found or loaded.
    pub fn new_with_format(
        model_path: &str,
        devices: &DeviceAssignment,
        dtype: &DType,
        format: ModelFormat,
        gpu_budget: &GpuBudget,
    ) -> Result<Self> {
        Self::new_with_options(
            model_path,
            devices,
            dtype,
            format,
            gpu_budget,
            KvCacheKind::from_env(),
        )
    }

    /// Like [`Self::new_with_format`], but takes an explicit `kv_kind`
    /// (e.g. from a `--kv-quant` CLI flag) instead of reading
    /// `CRANE_KV_QUANT`.
    ///
    /// # Errors
    ///
    /// Returns an error if the model files cannot be found or loaded.
    pub fn new_with_options(
        model_path: &str,
        devices: &DeviceAssignment,
        dtype: &DType,
        format: ModelFormat,
        gpu_budget: &GpuBudget,
        kv_kind: KvCacheKind,
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
            ModelFormat::Gguf | ModelFormat::Auto => {
                Self::from_gguf(model_path, devices, gpu_budget, kv_kind)
            },
            ModelFormat::Safetensors => {
                Self::from_pretrained(model_path, devices, *dtype, gpu_budget, kv_kind)
            },
        }
    }

    fn forward(&mut self, xs: &Tensor, s: usize) -> candle_core::Result<Tensor> {
        self.inner.forward(xs, s)
    }

    pub fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }

    /// Bytes of KV cache one sequence consumes per generated token. See
    /// [`Config::kv_bytes_per_token`]/[`Config::quantized_kv_bytes_per_token`].
    ///
    /// Prices at the smaller quantized storage size only when the fused
    /// dequantize-in-attention kernel (`crate::ops::fused_ops::quant_attn`)
    /// covers the sequence's entire lifetime: CUDA/ROCm, `max_concurrent ==
    /// 1` so batch decode's `to_fp_pair()`/`from_fp_pair()` round-trip
    /// (which always dequantizes to the compute dtype) never applies, and
    /// `CRANE_QUANT_ATTN_FUSED` isn't `0` (which forces every append onto
    /// the unfused path). Otherwise prices at the compute dtype's size,
    /// since KV-swap/preemption and any unfused decode/prefill step fully
    /// dequantize `Int8`/`Int4` caches, and a reservation based on the
    /// smaller quantized size would under-claim VRAM against that worst
    /// case (see git history for why that matters here).
    pub fn kv_bytes_per_token(&self) -> u64 {
        let config = self.inner.config();
        let fused_covers_full_lifetime = (self.device.is_cuda() || self.device.is_rocm())
            && self.max_concurrent == Some(1)
            && !crate::ops::fused_ops::quant_attn::fused_disabled();
        self.inner.kv_kind().effective_kv_bytes_per_token(
            fused_covers_full_lifetime,
            config.num_hidden_layers,
            config.num_key_value_heads,
            config.head_dim(),
            self.dtype.size_in_bytes(),
        )
    }

    fn from_pretrained(
        model_path: &str,
        devices: &DeviceAssignment,
        dtype: DType,
        gpu_budget: &GpuBudget,
        kv_kind: KvCacheKind,
    ) -> Result<Model> {
        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer not found at {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let filenames = utils::get_safetensors_files(model_path)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &devices.main) }?;

        let config_file = std::path::Path::new(model_path).join("config.json");
        let config_data = std::fs::read(config_file)?;
        let config: Config = serde_json::from_slice(&config_data)?;

        let inner =
            Qwen3Model::new_with_kv_kind(&config, vb, &devices.expert, gpu_budget, kv_kind)?;

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: devices.main.clone(),
            dtype,
            max_concurrent: gpu_budget.max_concurrent,
            inner,
        })
    }

    /// Load a GGUF quantized model file.
    fn from_gguf(
        model_path: &str,
        devices: &DeviceAssignment,
        gpu_budget: &GpuBudget,
        kv_kind: KvCacheKind,
    ) -> Result<Model> {
        let gguf_path = std::path::Path::new(model_path);

        let mmap = crate::quantized::gguf_file::mmap_gguf_file(gguf_path)?;
        let mut cursor = std::io::Cursor::new(mmap.as_ref());
        let ct = candle_core::quantized::gguf_file::Content::read(&mut cursor)?;

        log::info!(
            "GGUF header parsed: {} tensors, {} metadata entries",
            ct.tensor_infos.len(),
            ct.metadata.len(),
        );

        let tokenizer = tokenizer_utils::resolve_gguf_tokenizer(&ct, gguf_path)?;

        let inner =
            Qwen3Model::from_gguf_with_kv_kind(ct, &mut cursor, devices, gpu_budget, kv_kind)?;
        let dtype = inner.model_dtype();

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: devices.main.clone(),
            dtype,
            max_concurrent: gpu_budget.max_concurrent,
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

    /// Run a single forward step, returning raw logits.
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

    // ── KV cache management (for continuous-batching engine) ────────────

    pub fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    pub fn get_kv_caches(&self) -> Vec<Option<KvCacheState>> {
        self.inner.get_kv_caches()
    }

    pub fn set_kv_caches(&mut self, caches: Vec<Option<KvCacheState>>) {
        self.inner.set_kv_caches(caches);
    }

    /// Total bytes held by the model's KV caches (no GPU copies).
    pub fn active_kv_cache_bytes(&self) -> u64 {
        self.inner.active_kv_cache_bytes()
    }

    // ── Batched decode (GPU-efficient concurrent serving) ───────────────

    /// # Errors
    ///
    /// Returns an error if the batch decode setup fails.
    pub fn setup_batch_decode(
        &mut self,
        seq_kv_caches: &[Vec<Option<KvCacheState>>],
        extra_room: usize,
    ) -> candle_core::Result<(Vec<usize>, usize)> {
        self.inner.setup_batch_decode(seq_kv_caches, extra_room)
    }

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

    /// # Errors
    ///
    /// Returns an error if extracting the batch KV caches fails.
    pub fn extract_batch_kv(
        &mut self,
        kv_lens: &[usize],
        original_max_kv: usize,
        rounds_done: usize,
    ) -> candle_core::Result<BatchKvCache> {
        self.inner
            .extract_batch_kv(kv_lens, original_max_kv, rounds_done)
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
            let logits = if (config.repetition_penalty - 1.).abs() < f32::EPSILON {
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
        if let Some(ref mut s) = streamer
            && !streamer_finalized
        {
            s.finalize()?;
        }

        if config.report_speed {
            // generated_tokens is bounded by config.max_new_tokens, far below
            // f64's 52-bit mantissa limit.
            #[allow(clippy::cast_precision_loss)]
            let tokens_per_sec = generated_tokens as f64 / dt.as_secs_f64();
            println!("\n{generated_tokens} tokens generated ({tokens_per_sec:.2} token/s)\n");
        }

        Ok(tokens)
    }
}
