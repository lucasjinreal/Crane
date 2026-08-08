//! MiniCPM-V-4.6 vision-language model.
//!
//! Glues [`super::vision::VisionModel`] + [`super::merger::Merger`] onto
//! [`Qwen3_5TextModel`] — MiniCPM-V-4.6's text backbone is Qwen 3.5's hybrid
//! GDN/full-attention decoder, reused here **unmodified** (see
//! `super::config` module docs). Unlike Qwen 3.5-VL, no MRoPE/grid position
//! math is needed: HF's own `MiniCPMV4_6Model.forward` just uses plain
//! sequential positions, so the `[3, S]` position tensor `forward_embeds`
//! expects is simply three identical rows.
//!
//! Forward flow mirrors [`crate::models::qwen3_5::vlm::Qwen3_5VLModel`]:
//! 1. Preprocess the image (`super::preprocess`) -> packed pixel values +
//!    per-slice target sizes.
//! 2. Render the prompt with `<|image_pad|>` expanded to the full structured
//!    placeholder (`<image>...</image><slice>...</slice>...`).
//! 3. Embed `input_ids`, run the vision tower + merger, splice image
//!    features over the placeholder positions
//!    ([`crate::models::qwen3_5::vlm::splice_image_features`]).
//! 4. `Qwen3_5TextModel::forward_embeds` with sequential `[3, S]` positions.
//!
//! Single-sequence only (same constraint as `qwen3_5_vl`).

use anyhow::{Context, Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use super::config::{load_config, Config};
use super::merger::Merger;
use super::preprocess::{
    self, downsample_divisor, load_preprocessor_config, build_placeholder, PlaceholderTokens,
    PreprocessorConfig, ProcessedImage,
};
use super::vision::VisionModel;
use crate::models::qwen3_5::vlm::splice_image_features;
use crate::models::qwen3_5::{Config as Qwen35Config, Qwen3_5TextModel};
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::utils;

#[derive(Debug, Clone)]
pub struct VlGenerationConfig {
    pub max_new_tokens: usize,
    /// Strip a leading `<think> … </think>` block from the returned string
    /// (still generated/streamed, just not included in the final text).
    pub strip_thinking: bool,
}

impl Default for VlGenerationConfig {
    fn default() -> Self {
        Self { max_new_tokens: 512, strip_thinking: true }
    }
}

/// Read `extra_special_tokens` from `tokenizer_config.json`, falling back to
/// MiniCPM-V-4.6's known defaults if a field is absent (older/renamed
/// checkpoints).
fn load_placeholder_tokens(model_path: &str) -> Result<PlaceholderTokens> {
    let path = std::path::Path::new(model_path).join("tokenizer_config.json");
    let data = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_slice(&data)?;
    let extra = json.get("extra_special_tokens").cloned().unwrap_or_default();
    let get = |key: &str, default: &str| -> String {
        extra
            .get(key)
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| default.to_string())
    };
    Ok(PlaceholderTokens {
        image_token: get("image_token", "<|image_pad|>"),
        image_start_token: get("image_start_token", "<image>"),
        image_end_token: get("image_end_token", "</image>"),
        slice_start_token: get("slice_start_token", "<slice>"),
        slice_end_token: get("slice_end_token", "</slice>"),
        image_id_start_token: get("image_id_start_token", "<image_id>"),
        image_id_end_token: get("image_id_end_token", "</image_id>"),
    })
}

pub struct MinicpmV46VLModel {
    pub tokenizer: TokenOutputStream,
    pub device: Device,
    pub dtype: DType,
    pub image_token_id: u32,
    pub preprocessor: PreprocessorConfig,
    placeholder_tokens: PlaceholderTokens,
    downsample_divisor: usize,
    use_image_id: bool,
    eos_token_ids: Vec<u32>,
    vision: VisionModel,
    merger: Merger,
    text: Qwen3_5TextModel,
}

impl MinicpmV46VLModel {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer not found at {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let filenames = utils::get_safetensors_files(model_path)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, *dtype, device) }
            .with_context(|| format!("mmap safetensors at {model_path}"))?;

        let config_path = std::path::Path::new(model_path).join("config.json");
        let cfg: Config = load_config(config_path.to_str().context("non-UTF8 path")?)?;

        eprintln!(
            "[minicpm_v] loading vision tower: hidden={} layers={} insert_layer_id={} window={:?}",
            cfg.vision_config.hidden_size, cfg.vision_config.num_hidden_layers, cfg.insert_layer_id, cfg.window_kernel_size,
        );
        let vision = VisionModel::new(
            &cfg.vision_config,
            cfg.insert_layer_id,
            cfg.window_kernel_size,
            vb.pp("model").pp("vision_tower"),
        )
        .context("build vision tower")?;
        let merger = Merger::new(&cfg, vb.pp("model").pp("merger")).context("build merger")?;

        eprintln!("[minicpm_v] loading text model (Qwen 3.5 hybrid decoder, reused unmodified)");
        let qwen35_cfg = Qwen35Config {
            text_config: cfg.text_config.clone(),
            vision_config: None,
            image_token_id: None,
            video_token_id: None,
            vision_start_token_id: None,
            vision_end_token_id: None,
            tie_word_embeddings: cfg.tie_word_embeddings,
        };
        let text = Qwen3_5TextModel::new(&qwen35_cfg, vb, device, *dtype, None).context("build text model")?;

        let preprocessor = load_preprocessor_config(model_path)?;
        let placeholder_tokens = load_placeholder_tokens(model_path)?;

        let vocab = tokenizer.get_vocab(true);
        let eos_token_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| vocab.get(*t).map(|id| *id as u32))
            .collect();
        if eos_token_ids.is_empty() {
            anyhow::bail!("tokenizer has neither <|im_end|> nor <|endoftext|>");
        }

        let use_image_id = preprocessor.use_image_id;
        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype: *dtype,
            image_token_id: cfg.image_token_id,
            preprocessor,
            placeholder_tokens,
            downsample_divisor: downsample_divisor(&cfg),
            use_image_id,
            eos_token_ids,
            vision,
            merger,
            text,
        })
    }

    /// Run the vision tower + merger on a packed image batch.
    ///
    /// `pixel_values`: `[1, C, patch_size, total_patches*patch_size]`.
    /// `target_sizes`: per-image/-slice `(h, w)` patch-grid dims.
    ///
    /// Returns `[total_image_tokens, llm_hidden]`.
    pub fn encode_images(&self, pixel_values: &Tensor, target_sizes: &[(usize, usize)]) -> Result<Tensor> {
        let (vision_out, downsampled_sizes) = self
            .vision
            .forward(pixel_values, target_sizes)
            .context("vision forward")?;
        self.merger.forward(&vision_out, &downsampled_sizes).context("merger forward")
    }

    /// Forward pass for a vision-language prefill (or a text-only prefill,
    /// when `pixel_values`/`target_sizes` are `None`).
    ///
    /// `input_ids`: `[1, S]`, already containing the expanded placeholder
    /// (see [`Self::render_prompt`]).
    ///
    /// Returns logits `[1, V]` (last position only).
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        target_sizes: Option<&[(usize, usize)]>,
        start_pos: usize,
    ) -> Result<Tensor> {
        let image_embeds = match (pixel_values, target_sizes) {
            (Some(pv), Some(ts)) => Some(self.encode_images(pv, ts)?),
            _ => None,
        };

        let (_b, seq_len) = input_ids.dims2()?;
        let mut hidden_states = self.text.embed_only(input_ids)?;

        if let Some(img_emb) = image_embeds {
            hidden_states = splice_image_features(input_ids, &hidden_states, &img_emb, self.image_token_id)?;
        }

        // No MRoPE grid needed (see module docs) — all three axes share the
        // plain sequential position.
        let flat: Vec<u32> = (0..seq_len)
            .flat_map(|i| std::iter::repeat_n((start_pos + i) as u32, 3))
            .collect();
        let position_ids = Tensor::from_vec(flat, (3, seq_len), &self.device)?;

        self.text.forward_embeds(&hidden_states, &position_ids, start_pos, None)
    }

    /// Decode one new token. The KV cache must already hold the prefill.
    pub fn decode_step(&mut self, token: u32, start_pos: usize) -> Result<Tensor> {
        let input = Tensor::from_vec(vec![token], (1usize, 1usize), &self.device)?;
        let hidden = self.text.embed_only(&input)?;
        let p = start_pos as u32;
        let pos = Tensor::from_vec(vec![p, p, p], (3usize, 1usize), &self.device)?;
        self.text.forward_embeds(&hidden, &pos, start_pos, None)
    }

    pub fn clear_kv_cache(&mut self) {
        self.text.reset_gdn_caches().expect("GDN reset failed");
    }

    /// Preprocess `image` and build the `input_ids` for one user turn,
    /// expanding the chat template's single `<|image_pad|>` marker into the
    /// full structured placeholder (`<image>...</image>` [+ `<slice>...
    /// </slice>` per tile]). Returns `None` for `ProcessedImage`/pixel data
    /// when `image` is `None` (text-only turn).
    pub fn render_prompt(&self, image: Option<&image::DynamicImage>, user_text: &str) -> Result<(Vec<u32>, Option<ProcessedImage>)> {
        let processed = image
            .map(|img| preprocess::process_image(img, &self.preprocessor, &self.device, self.dtype))
            .transpose()?;

        let user_content = match &processed {
            Some(p) => {
                let placeholder = build_placeholder(p, &self.placeholder_tokens, 0, self.use_image_id, self.downsample_divisor);
                format!("{placeholder}\n{user_text}")
            }
            None => user_text.to_string(),
        };
        let rendered = format!("<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n");

        let base = self.tokenizer.tokenizer.encode(rendered, false).map_err(E::msg)?;
        let ids = base.get_ids().to_vec();
        Ok((ids, processed))
    }

    /// Answer `user_text` about `image` (pass `None` for a text-only turn).
    /// Resets the KV cache, so each call is an independent single turn.
    pub fn generate(
        &mut self,
        image: Option<&image::DynamicImage>,
        user_text: &str,
        cfg: &VlGenerationConfig,
        mut on_token: impl FnMut(&str),
    ) -> Result<String> {
        let (input_ids, processed) = self.render_prompt(image, user_text)?;
        let input_tensor = Tensor::from_vec(input_ids.clone(), (1usize, input_ids.len()), &self.device)?;

        let (pixel_values, target_sizes) = match &processed {
            Some(p) => (Some(p.pixel_values.clone().unsqueeze(0)?), Some(p.target_sizes.clone())),
            None => (None, None),
        };

        self.clear_kv_cache();
        let mut logits = self.forward(&input_tensor, pixel_values.as_ref(), target_sizes.as_deref(), 0)?;

        let mut generated: Vec<u32> = Vec::with_capacity(cfg.max_new_tokens);
        let mut cur_pos = input_ids.len();
        for _ in 0..cfg.max_new_tokens {
            let next = logits
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .argmax(candle_core::D::Minus1)?
                .to_scalar::<u32>()?;
            if self.eos_token_ids.contains(&next) {
                break;
            }
            if let Ok(text) = self.tokenizer.tokenizer.decode(&[next], true) {
                on_token(&text);
            }
            generated.push(next);
            logits = self.decode_step(next, cur_pos)?;
            cur_pos += 1;
        }

        let text = self.tokenizer.tokenizer.decode(&generated, true).map_err(E::msg)?;
        Ok(if cfg.strip_thinking { strip_thinking(&text) } else { text })
    }
}

fn strip_thinking(text: &str) -> String {
    match text.split_once("</think>") {
        Some((head, tail)) if head.trim_start().starts_with("<think>") => tail.trim().to_string(),
        _ => text.trim().to_string(),
    }
}
