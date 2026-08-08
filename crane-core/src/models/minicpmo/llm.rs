//! Standalone wrapper around MiniCPM-o-4.5's `llm.*` tower.
//!
//! The tower itself is a plain dense Qwen3 decoder (confirmed against the
//! real checkpoint: `q_norm`/`k_norm` present, no GDN/gated-delta-net
//! tensors) — this file only handles MiniCPM-o's checkpoint layout
//! (`llm.model.*` / `llm.lm_head.weight`, rather than Qwen3's own
//! `model.*` / `lm_head.weight`) on top of [`crate::models::qwen3::modeling::Qwen3Model`].
//! Text-only, no vision/audio/tts splicing yet — see the `minicpmo` module
//! doc for phase status.

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::generation::LogitsProcessor;
use tokenizers::Tokenizer;

use super::config::{load_config, MiniCpmOConfig};
use crate::generation::based::ModelForCausalLM;
use crate::generation::GenerationConfig;
use crate::models::qwen3::modeling::Qwen3Model;
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::utils;

pub struct MiniCpmOLlm {
    pub tokenizer: TokenOutputStream,
    pub device: Device,
    pub dtype: DType,
    pub config: MiniCpmOConfig,
    inner: Qwen3Model,
}

impl MiniCpmOLlm {
    /// # Errors
    ///
    /// Returns an error if the model files cannot be found or loaded.
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer not found at {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let config_path = std::path::Path::new(model_path).join("config.json");
        let config = load_config(&config_path.to_string_lossy())?;

        let filenames = utils::get_safetensors_files(model_path)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, *dtype, device) }?;

        // MiniCPM-o nests the Qwen3 decoder under `llm.model.*`, with
        // `llm.lm_head.weight` as a sibling (not the `model.*` /
        // `lm_head.weight` layout `Qwen3Model::new` assumes).
        let llm_vb = vb.pp("llm");
        let inner = Qwen3Model::new_from_model_vb(&config.llm, llm_vb.pp("model"), llm_vb)?;

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype: *dtype,
            config,
            inner,
        })
    }

    /// Construct the LLM tower from a standalone quantized GGUF file
    /// (a standard llama.cpp-style Qwen3 conversion — confirmed against the
    /// real `openbmb/MiniCPM-o-4_5-gguf` release: `general.architecture =
    /// qwen3`, plain `blk.N.*`/`token_embd.weight`/`output.weight` naming,
    /// no MiniCPM-o-specific prefix or layout — so this reuses
    /// [`Qwen3Model::from_gguf`] unmodified, the same loader already used
    /// for standalone Qwen3/Qwen3.5 GGUF checkpoints) instead of the real
    /// checkpoint's own safetensors weights. Dramatically cuts VRAM (Q8_0
    /// is roughly half of bf16) — this is what makes a full `DuplexSession`
    /// fit comfortably instead of needing nearly the whole GPU. The other
    /// towers (audio/TTS/Token2wav) are unaffected and still load from
    /// `model_path`'s safetensors as usual; only the LLM's own weight
    /// source changes here.
    ///
    /// `model_path` still must point at a real MiniCPM-o-4.5 checkpoint
    /// directory — `tokenizer.json`/`config.json` are read from there as
    /// usual (the GGUF file's own embedded tokenizer metadata is not used).
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint directory's tokenizer/config, or
    /// the GGUF file itself, can't be loaded.
    pub fn from_gguf(model_path: &str, gguf_path: &str, device: &Device) -> Result<Self> {
        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer not found at {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let config_path = std::path::Path::new(model_path).join("config.json");
        let config = load_config(&config_path.to_string_lossy())?;

        let mut file = std::fs::File::open(gguf_path)
            .map_err(|e| anyhow::anyhow!("failed to open GGUF file {gguf_path}: {e}"))?;
        let ct = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("failed to parse GGUF file {gguf_path}: {e}"))?;
        let inner = Qwen3Model::from_gguf(ct, &mut file, device)?;
        let dtype = inner.model_dtype();

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype,
            config,
            inner,
        })
    }

    pub fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
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

    /// Run a single forward step, returning raw logits. Exposed (rather
    /// than kept private) so HF-diff tests can compare intermediate
    /// forward passes directly against `modeling_minicpmo.py`'s `self.llm`.
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward_step(&mut self, input_ids: &[u32], start_pos: usize) -> candle_core::Result<Tensor> {
        let input = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        self.inner.forward(&input, start_pos)
    }

    /// Token embedding lookup only (no decoder layers) — the hook callers
    /// that splice in non-text embeddings (vision, later audio) need before
    /// running [`Self::forward_embeds`]. `input_ids`: `[1, S]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedding lookup fails.
    pub fn embed_only(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        self.inner.embed_tokens().forward(input_ids)?.to_dtype(self.dtype)
    }

    /// Run the decoder from a caller-supplied embedding sequence (e.g. text
    /// embeddings with vision tokens spliced in) instead of doing the token
    /// lookup internally. Returns logits `[1, V]` (last position only).
    ///
    /// # Errors
    ///
    /// Returns an error if the forward pass fails.
    pub fn forward_embeds(&mut self, inputs_embeds: &Tensor, start_pos: usize) -> candle_core::Result<Tensor> {
        self.inner.forward_embeds(inputs_embeds, start_pos)
    }

    /// Full-sequence post-norm hidden states (`[1, seq_len, hidden]`,
    /// pre-`lm_head`) from the most recent [`Self::forward_embeds`]/
    /// [`Self::forward_step`] call — needed by TTS conditioning
    /// (`_generate_speech_non_streaming`'s `tts_proj_layer=-1`, i.e. the
    /// final layer's output, not an arbitrary intermediate one).
    #[must_use]
    pub fn last_hidden_states(&self) -> Option<&Tensor> {
        self.inner.last_hidden_states()
    }

    /// Number of tokens currently held in the decoder's KV cache — lets
    /// callers (the duplex session) always derive `start_pos` for the next
    /// forward call from the cache's actual length rather than a
    /// separately-tracked counter, which would drift out of sync once
    /// sliding-window eviction can shrink the cache mid-session.
    #[must_use]
    pub fn kv_cache_len(&self) -> usize {
        self.inner.kv_cache_len()
    }

    /// Per-layer `(K, V)` cache tensors, `[1, num_kv_heads, seq_len, head_dim]`
    /// each, post-RoPE (RoPE is applied before caching, same as the real
    /// HF `DynamicCache`) — needed for sliding-window eviction, which must
    /// realign the survivors' RoPE after dropping a range from the middle.
    #[must_use]
    pub fn get_kv_caches(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.inner.get_kv_caches()
    }

    /// Restore per-layer KV caches after out-of-place eviction/realignment.
    pub fn set_kv_caches(&mut self, caches: Vec<Option<(Tensor, Tensor)>>) {
        self.inner.set_kv_caches(caches);
    }

    #[must_use]
    pub fn rope_theta(&self) -> f64 {
        self.inner.config().rope_theta
    }

    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.inner.config().head_dim()
    }

    #[must_use]
    pub fn max_position_embeddings(&self) -> usize {
        self.inner.config().max_position_embeddings
    }
}

impl ModelForCausalLM for MiniCpmOLlm {
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

        // MiniCPM-o's Qwen3 tower stops on <|im_end|> or <|endoftext|>, same
        // special tokens as plain Qwen3.
        let eos_token: Option<u32> = config
            .eos_token_id
            .or_else(|| self.tokenizer.get_token("<|im_end|>"))
            .or_else(|| self.tokenizer.get_token("<|endoftext|>"));
        let mut streamer_finalized = false;

        for index in 0..config.max_new_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];

            let logits = self.forward_step(ctxt, start_pos)?;
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
        if let Some(ref mut s) = streamer
            && !streamer_finalized
        {
            s.finalize()?;
        }

        Ok(tokens)
    }
}

// ── HF-diff validation ─────────────────────────────────────────────────
//
// Compares this module's forward pass against a real Python forward pass
// through `modeling_minicpmo.py`'s `self.llm` (a plain `Qwen3ForCausalLM`
// loaded with only the `llm.*` checkpoint tensors — see
// `/home/hahihula/.claude/jobs/02aa7312/tmp/minicpmo_diff/dump_llm.py`),
// on the fixed prompt "The capital of France is" (5 tokens, ids dumped in
// `meta.json`). `Qwen3Model::decode` only ever returns the last position's
// logits (see its doc comment), so this compares against the last row of
// the Python dump's `[5, vocab]` logits.
#[cfg(test)]
mod hf_diff {
    use super::*;

    // Run with:
    // `cargo test -p crane-core --release --features cuda -- --ignored --nocapture minicpmo_llm_matches_python`
    #[test]
    #[ignore = "needs the real MiniCPM-o-4.5 checkpoint + a matching HF dump"]
    fn minicpmo_llm_matches_python() {
        let model_path = "/home/hahihula/mywork/ai/additional_models/MiniCPM-o-4_5";
        let diff_dir = "/home/hahihula/.claude/jobs/02aa7312/tmp/minicpmo_diff";

        #[cfg(feature = "cuda")]
        let (device, dtype) = (Device::new_cuda(0).unwrap(), DType::BF16);
        #[cfg(not(feature = "cuda"))]
        let (device, dtype) = (Device::Cpu, DType::F32);

        let mut model = MiniCpmOLlm::new(model_path, &device, &dtype).expect("load MiniCPM-o llm tower");

        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{diff_dir}/meta.json")).unwrap()).unwrap();
        let input_ids: Vec<u32> = meta["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let shape: Vec<usize> = meta["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let (seq_len, vocab_size) = (shape[0], shape[1]);
        assert_eq!(input_ids.len(), seq_len);

        let logits = model
            .forward_step(&input_ids, 0)
            .expect("forward")
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let rust_logits: Vec<f32> = logits.to_vec1().unwrap();
        assert_eq!(rust_logits.len(), vocab_size);

        let py_bytes = std::fs::read(format!("{diff_dir}/logits.bin")).unwrap();
        let py_logits_all: Vec<f32> = py_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        // Last row of the [seq_len, vocab] dump.
        let py_last = &py_logits_all[(seq_len - 1) * vocab_size..seq_len * vocab_size];

        let dot: f64 = rust_logits.iter().zip(py_last).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
        let norm_a: f64 = rust_logits.iter().map(|a| f64::from(*a) * f64::from(*a)).sum::<f64>().sqrt();
        let norm_b: f64 = py_last.iter().map(|b| f64::from(*b) * f64::from(*b)).sum::<f64>().sqrt();
        let cosine = dot / (norm_a * norm_b);
        let max_abs_diff = rust_logits
            .iter()
            .zip(py_last)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);

        println!("cosine similarity: {cosine}, max abs diff: {max_abs_diff}");

        let rust_top5 = {
            let mut idx: Vec<usize> = (0..rust_logits.len()).collect();
            idx.sort_unstable_by(|&a, &b| rust_logits[b].partial_cmp(&rust_logits[a]).unwrap());
            idx[..5].to_vec()
        };
        let py_top5: Vec<usize> = meta["top5_last"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        println!("rust top5: {rust_top5:?}, python top5: {py_top5:?}");
        assert_eq!(rust_top5[0], py_top5[0], "top-1 predicted token mismatch");

        assert!(cosine > 0.999, "cosine similarity too low: {cosine}");
    }
}
