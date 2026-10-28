//! MiniCPM-V-4.6 / 4.7 vision-language model.
//!
//! Glues [`super::vision::VisionModel`] + [`super::merger::Merger`] onto
//! [`Qwen3_5TextModel`] — MiniCPM-V-4.6's text backbone is Qwen 3.5's hybrid
//! GDN/full-attention decoder, reused here **unmodified** (see
//! `super::config` module docs). 4.6 needs no MRoPE/grid position math: HF's
//! own `MiniCPMV4_6Model.forward` just uses plain sequential positions, so the
//! `[3, S]` position tensor `forward_embeds` expects is simply three identical
//! rows. 4.7 (`mrope_mode: "canvas"`) lays images on a 2D canvas instead — see
//! [`super::canvas_rope`] — and decode positions are `cache_len + rope_delta`.
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

use super::canvas_rope::{CanvasTokenIds, canvas_position_ids};
use super::config::{Config, load_config};
use super::merger::Merger;
use super::mmproj_gguf::load_mmproj;
use super::preprocess::{
    self, PlaceholderTokens, PreprocessorConfig, ProcessedImage, build_placeholder,
    downsample_divisor, load_preprocessor_config,
};
use super::vision::VisionModel;
use crate::models::dspark::DSparkDrafter;
use crate::models::dspark::policy::AdaptivePolicy;
use crate::models::qwen3_5::StateSnapshot;
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
    /// `Some(false)` pre-closes the think block in the generation prompt
    /// (`<think>\n\n</think>\n\n`, the 4.7 instruct default); `Some(true)`
    /// opens it (`<think>\n`); `None` uses the model default (4.7: off,
    /// 4.6: legacy prompt with no think tag at all).
    pub enable_thinking: Option<bool>,
    /// Per-call override of the checkpoint's `downsample_mode` (`"16x"` or
    /// `"4x"`); `None` keeps the config value.
    pub downsample_mode: Option<String>,
    /// Speculative decoding with the loaded DSpark drafter. `None` (default)
    /// uses it whenever one is loaded; `Some(false)` forces plain decoding.
    pub dspark: Option<bool>,
    /// Fall back to plain decoding while speculation is losing (acceptance
    /// below the measured break-even), probing now and then. `None` (default)
    /// is on; `Some(false)` always speculates. `CRANE_DSPARK_ADAPTIVE=0` also
    /// turns it off.
    pub dspark_adaptive: Option<bool>,
    /// Testing aid: `(speculative rounds, plain tokens)` cycle replacing the
    /// cost-based decision, to exercise the mode switches deterministically.
    #[doc(hidden)]
    pub dspark_schedule: Option<(usize, usize)>,
}

/// What the last speculative [`MinicpmV46VLModel::generate`] did.
#[derive(Debug, Clone, Default)]
pub struct SpecStats {
    /// Verification rounds run.
    pub rounds: usize,
    /// Tokens emitted overall (the prefill token, speculative rounds, plain stretches).
    pub tokens: usize,
    /// Of those, tokens produced by speculative rounds.
    pub spec_tokens: usize,
    /// Of those, tokens decoded the plain way while speculation was losing.
    pub plain_tokens: usize,
    /// Switches between speculative and plain decoding (either direction).
    pub switches: usize,
    /// Wall-clock seconds spent in the two switch passes.
    pub switch_secs: f64,
    /// Draft tokens that were accepted.
    pub accepted_drafts: usize,
    /// Wall-clock seconds spent drafting / verifying (device-synchronised).
    pub draft_secs: f64,
    pub verify_secs: f64,
    /// Everything else in the round loop: acceptance, emit, commit, appending
    /// the new context to the drafter.
    pub other_secs: f64,
}

impl SpecStats {
    /// Mean tokens produced per verification round.
    #[must_use]
    pub fn mean_accept_len(&self) -> f64 {
        if self.rounds == 0 {
            0.0
        } else {
            self.spec_tokens as f64 / self.rounds as f64
        }
    }
}

impl Default for VlGenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            strip_thinking: true,
            enable_thinking: None,
            downsample_mode: None,
            dspark: None,
            dspark_adaptive: None,
            dspark_schedule: None,
        }
    }
}

/// Read `extra_special_tokens` from `tokenizer_config.json`, falling back to
/// MiniCPM-V-4.6's known defaults if a field is absent (older/renamed
/// checkpoints).
fn load_placeholder_tokens(model_path: &str) -> Result<PlaceholderTokens> {
    let path = std::path::Path::new(model_path).join("tokenizer_config.json");
    let data = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_slice(&data)?;
    let extra = json
        .get("extra_special_tokens")
        .cloned()
        .unwrap_or_default();
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

/// Serves both MiniCPM-V-4.6 and 4.7 (the name predates 4.7; 4.7 differs only
/// in canvas M-RoPE, the prompt template, and the `"4x"` mode being exercised).
pub struct MinicpmV46VLModel {
    pub tokenizer: TokenOutputStream,
    pub device: Device,
    pub dtype: DType,
    pub image_token_id: u32,
    pub preprocessor: PreprocessorConfig,
    placeholder_tokens: PlaceholderTokens,
    downsample_divisor: usize,
    /// `downsample_divisor` from `config.json`, restored when a call doesn't
    /// override the mode.
    config_divisor: usize,
    use_image_id: bool,
    /// `Some` for 4.7 checkpoints (`mrope_mode: "canvas"`).
    canvas_ids: Option<CanvasTokenIds>,
    /// Offset added to the cache length to get a decode token's position;
    /// set by the last prefill (0 for text-only / sequential-position models).
    rope_delta: i64,
    eos_token_ids: Vec<u32>,
    dspark: Option<DSparkDrafter>,
    last_spec: Option<SpecStats>,
    /// Measured seconds per plain decode step, kept across requests so the
    /// adaptive fallback knows the break-even from the start.
    plain_step_secs: Option<f64>,
    vision: VisionModel,
    merger: Merger,
    text: Qwen3_5TextModel,
}

impl MinicpmV46VLModel {
    /// # Errors
    ///
    /// Returns an error if the tokenizer, safetensors weights, or
    /// `config.json`/`preprocessor_config.json` can't be found or loaded
    /// from `model_path`, if building the vision tower, merger, or text
    /// model fails, or if the tokenizer has neither an `<|im_end|>` nor
    /// `<|endoftext|>` token.
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
            cfg.vision_config.hidden_size,
            cfg.vision_config.num_hidden_layers,
            cfg.insert_layer_id,
            cfg.window_kernel_size,
        );
        let vision = VisionModel::new(
            &cfg.vision_config,
            cfg.insert_layer_id,
            cfg.window_kernel_size,
            &vb.pp("model").pp("vision_tower"),
        )
        .context("build vision tower")?;
        let merger = Merger::new(&cfg, &vb.pp("model").pp("merger")).context("build merger")?;

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
        let text = Qwen3_5TextModel::new(&qwen35_cfg, &vb, device, *dtype, None)
            .context("build text model")?;

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
            config_divisor: downsample_divisor(&cfg),
            use_image_id,
            canvas_ids: cfg.canvas_token_ids(),
            rope_delta: 0,
            eos_token_ids,
            dspark: None,
            last_spec: None,
            plain_step_secs: None,
            vision,
            merger,
            text,
        })
    }

    /// Load from llama.cpp files: a `qwen35`-arch LLM GGUF (any quant) plus
    /// the matching `mmproj-*.gguf` vision projector. No `config.json`,
    /// `tokenizer.json` or safetensors needed — the tokenizer comes from the
    /// LLM GGUF, the marker token ids from its vocab, and the vision config
    /// from the mmproj metadata.
    ///
    /// `dtype` is the vision tower's compute dtype; the text model picks its
    /// own (see `Qwen3_5TextModel::from_gguf`).
    ///
    /// # Errors
    ///
    /// Returns an error if either file can't be read or isn't the expected
    /// kind, if the vocab lacks the image marker tokens, or if building the
    /// vision tower, merger or text model fails.
    pub fn from_gguf(
        llm_gguf: &str,
        mmproj_gguf: &str,
        device: &Device,
        dtype: &DType,
    ) -> Result<Self> {
        let gguf_path = std::path::Path::new(llm_gguf);
        let mmap = crate::quantized::gguf_file::mmap_gguf_file(gguf_path)
            .with_context(|| format!("mmap GGUF file {llm_gguf}"))?;
        let mut cursor = std::io::Cursor::new(mmap.as_ref());
        let (ct, extended) = crate::quantized::extended_gguf::read_content(mmap.as_ref())?;
        let tokenizer = crate::utils::tokenizer_utils::resolve_gguf_tokenizer(&ct, gguf_path)?;

        // Text model first: its load has transient peaks (packed expert
        // fusion) that must not land on top of the resident vision weights.
        // MiniCPM-V-4.7-35B-A3B Q4_K_M + the f16 mmproj fit a 24 GB card in
        // this order (~23 GB) and OOM in the other.
        let text = if extended.is_empty() {
            Qwen3_5TextModel::from_gguf(ct, &mut cursor, device)?
        } else {
            Qwen3_5TextModel::from_gguf_extended(ct, &mut cursor, device, extended)?
        };

        let proj = load_mmproj(std::path::Path::new(mmproj_gguf), device, *dtype)
            .context("load mmproj")?;
        // 4.7's projector uses canvas M-RoPE; 4.6's keeps sequential positions.
        let canvas = proj.projector_type == "minicpmv4_7";
        let divisor = if proj.scale_factor == 2 { 4 } else { 16 };

        let id_of = |tok: &str| -> Result<u32> {
            tokenizer
                .token_to_id(tok)
                .with_context(|| format!("vocab has no {tok:?} token"))
        };
        let placeholder_tokens = PlaceholderTokens {
            image_token: "<|image_pad|>".into(),
            image_start_token: "<image>".into(),
            image_end_token: "</image>".into(),
            slice_start_token: "<slice>".into(),
            slice_end_token: "</slice>".into(),
            image_id_start_token: "<image_id>".into(),
            image_id_end_token: "</image_id>".into(),
        };
        let image_token_id = id_of("<|image_pad|>")?;
        let canvas_ids = if canvas {
            let newline = tokenizer
                .encode("\n", false)
                .map_err(E::msg)?
                .get_ids()
                .first()
                .copied()
                .context("tokenizer produced no id for \\n")?;
            Some(CanvasTokenIds {
                image_start: id_of("<image>")?,
                image_end: id_of("</image>")?,
                slice_start: id_of("<slice>")?,
                slice_end: id_of("</slice>")?,
                newline,
            })
        } else {
            None
        };

        let eos_token_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tokenizer.token_to_id(t))
            .collect();
        if eos_token_ids.is_empty() {
            anyhow::bail!("tokenizer has neither <|im_end|> nor <|endoftext|>");
        }

        eprintln!(
            "[minicpm_v] GGUF: projector={} scale_factor={} insert_layer_id={}",
            proj.projector_type, proj.scale_factor, proj.insert_layer_id
        );
        let vision = VisionModel::new(
            &proj.vision_config,
            proj.insert_layer_id,
            (2, 2),
            &proj.vb.pp("model").pp("vision_tower"),
        )
        .context("build vision tower")?;
        let merger = Merger::with_dims(
            proj.vision_config.hidden_size,
            proj.projection_dim,
            (2, 2),
            1,
            &proj.vb.pp("model").pp("merger"),
        )
        .context("build merger")?;

        let preprocessor = PreprocessorConfig {
            max_slice_nums: proj.max_slice_nums.unwrap_or(9),
            scale_resolution: 448,
            patch_size: proj.vision_config.patch_size,
            slice_mode: true,
            use_image_id: true,
            image_mean: proj.image_mean,
            image_std: proj.image_std,
        };
        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype: *dtype,
            image_token_id,
            preprocessor,
            placeholder_tokens,
            downsample_divisor: divisor,
            config_divisor: divisor,
            use_image_id: true,
            canvas_ids,
            rope_delta: 0,
            eos_token_ids,
            dspark: None,
            last_spec: None,
            plain_step_secs: None,
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
    ///
    /// # Errors
    ///
    /// Returns an error if the vision tower or merger forward pass fails,
    /// e.g. from a `target_sizes`/`pixel_values` mismatch.
    pub fn encode_images(
        &self,
        pixel_values: &Tensor,
        target_sizes: &[(usize, usize)],
    ) -> Result<Tensor> {
        let (vision_out, downsampled_sizes) = self
            .vision
            .forward(pixel_values, target_sizes, self.downsample_divisor == 16)
            .context("vision forward")?;
        self.merger
            .forward(&vision_out, &downsampled_sizes)
            .context("merger forward")
    }

    /// Forward pass for a vision-language prefill (or a text-only prefill,
    /// when `pixel_values`/`target_sizes` are `None`).
    ///
    /// `input_ids`: `[1, S]`, already containing the expanded placeholder
    /// (see [`Self::render_prompt`]).
    ///
    /// Returns logits `[1, V]` (last position only).
    ///
    /// # Errors
    ///
    /// Returns an error if image encoding fails, if splicing image features
    /// into the input embeddings fails (e.g. a placeholder-token count
    /// mismatch), or if the text model's forward pass fails.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        target_sizes: Option<&[(usize, usize)]>,
        start_pos: usize,
    ) -> Result<Tensor> {
        let (hidden_states, position_ids) =
            self.prepare_prefill(input_ids, pixel_values, target_sizes, start_pos)?;
        self.text
            .forward_embeds(&hidden_states, &position_ids, start_pos, None)
    }

    /// Embed `input_ids`, splice in image features and compute `[3, S]`
    /// position ids (recording `rope_delta`): everything a prefill needs
    /// before the decoder runs.
    fn prepare_prefill(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        target_sizes: Option<&[(usize, usize)]>,
        start_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let image_embeds = match (pixel_values, target_sizes) {
            (Some(pv), Some(ts)) => Some(self.encode_images(pv, ts)?),
            _ => None,
        };

        let mut hidden_states = self.text.embed_only(input_ids)?;

        if let Some(img_emb) = image_embeds {
            hidden_states =
                splice_image_features(input_ids, &hidden_states, &img_emb, self.image_token_id)?;
        }

        let position_ids = self.prefill_positions(input_ids, target_sizes, start_pos)?;
        Ok((hidden_states, position_ids))
    }

    /// `[3, S]` position ids for a prefill chunk, recording `rope_delta`.
    ///
    /// Canvas models (4.7) lay images on a 2D canvas on the first chunk;
    /// everything else (4.6, text-only turns) is plain sequential positions.
    fn prefill_positions(
        &mut self,
        input_ids: &Tensor,
        target_sizes: Option<&[(usize, usize)]>,
        start_pos: usize,
    ) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        if let (Some(ids), Some(sizes), 0) = (&self.canvas_ids, target_sizes, start_pos) {
            let tokens = input_ids.flatten_all()?.to_vec1::<u32>()?;
            let merge = if self.downsample_divisor == 4 { 2 } else { 4 };
            let canvas = canvas_position_ids(&tokens, self.image_token_id, sizes, ids, merge);
            self.rope_delta = canvas.rope_deltas;
            let flat: Vec<i64> = canvas.position_ids.into_iter().flatten().collect();
            let flat: Vec<u32> = flat.into_iter().map(|p| p as u32).collect();
            return Ok(Tensor::from_vec(flat, (3, seq_len), &self.device)?);
        }
        if start_pos == 0 {
            self.rope_delta = 0;
        }
        let base = start_pos as i64 + self.rope_delta;
        let flat: Vec<u32> = (0..seq_len)
            .flat_map(|i| std::iter::repeat_n((base + i as i64) as u32, 3))
            .collect();
        Ok(Tensor::from_vec(flat, (3, seq_len), &self.device)?)
    }

    /// Decode one new token. The KV cache must already hold the prefill.
    ///
    /// # Errors
    ///
    /// Returns an error if the text model's embedding lookup or forward pass
    /// fails.
    pub fn decode_step(&mut self, token: u32, start_pos: usize) -> Result<Tensor> {
        let input = Tensor::from_vec(vec![token], (1usize, 1usize), &self.device)?;
        let hidden = self.text.embed_only(&input)?;
        let p = (start_pos as i64 + self.rope_delta) as u32;
        let pos = Tensor::from_vec(vec![p, p, p], (3usize, 1usize), &self.device)?;
        self.text.forward_embeds(&hidden, &pos, start_pos, None)
    }

    /// # Errors
    ///
    /// Returns an error if a cache could not be reset: the reset issues device
    /// memsets, so it is not infallible in-memory bookkeeping.
    pub fn clear_kv_cache(&mut self) -> Result<()> {
        self.text.reset_gdn_caches()
    }

    /// Enable fused causal attention in the text tower; see
    /// [`Qwen3_5TextModel::enable_flash_causal`].
    pub fn enable_flash_causal(&mut self) {
        self.text.enable_flash_causal();
    }

    /// Preprocess `image` and build the `input_ids` for one user turn,
    /// expanding the chat template's single `<|image_pad|>` marker into the
    /// full structured placeholder (`<image>...</image>` [+ `<slice>...
    /// </slice>` per tile]). Returns `None` for `ProcessedImage`/pixel data
    /// when `image` is `None` (text-only turn).
    ///
    /// # Errors
    ///
    /// Returns an error if image preprocessing fails or if the rendered
    /// prompt can't be tokenized.
    pub fn render_prompt(
        &self,
        image: Option<&image::DynamicImage>,
        user_text: &str,
        enable_thinking: Option<bool>,
    ) -> Result<(Vec<u32>, Option<ProcessedImage>)> {
        let processed = image
            .map(|img| preprocess::process_image(img, &self.preprocessor, &self.device, self.dtype))
            .transpose()?;

        let user_content = match &processed {
            Some(p) => {
                let placeholder = build_placeholder(
                    p,
                    &self.placeholder_tokens,
                    0,
                    self.use_image_id,
                    self.downsample_divisor,
                );
                format!("{placeholder}\n{user_text}")
            },
            None => user_text.to_string(),
        };
        // 4.7's template defaults to instruct mode (`enable_thinking=False`):
        // the generation prompt carries a pre-closed empty think block. 4.6's
        // legacy prompt carries no think tag at all.
        let think_tag = match enable_thinking.or(self.canvas_ids.map(|_| false)) {
            Some(false) => "<think>\n\n</think>\n\n",
            Some(true) => "<think>\n",
            None => "",
        };
        let rendered = format!(
            "<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{think_tag}"
        );

        let base = self
            .tokenizer
            .tokenizer
            .encode(rendered, false)
            .map_err(E::msg)?;
        let ids = base.get_ids().to_vec();
        Ok((ids, processed))
    }

    /// Attach a DSpark drafter: a HF directory (`config.json` +
    /// `model.safetensors`) or a llama.cpp `dflash` `.gguf` file. Generation then uses speculative decoding unless
    /// a call sets [`VlGenerationConfig::dspark`] to `Some(false)`.
    ///
    /// # Errors
    ///
    /// Returns an error if the drafter can't be loaded, or doesn't fit this
    /// target (hidden size / vocabulary / tap layers).
    pub fn load_dspark(&mut self, path: &str) -> Result<()> {
        let is_gguf = std::path::Path::new(path)
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
        let drafter = if is_gguf {
            DSparkDrafter::from_gguf(path, &self.device, self.dtype)
        } else {
            DSparkDrafter::from_dir(path, &self.device, self.dtype)
        }
        .with_context(|| format!("load DSpark drafter from {path}"))?;
        self.attach_dspark(drafter)
    }

    /// Attach an already-built drafter; see [`Self::load_dspark`].
    ///
    /// # Errors
    ///
    /// Returns an error if the drafter doesn't fit this target.
    pub fn attach_dspark(&mut self, drafter: DSparkDrafter) -> Result<()> {
        let cfg = drafter.config();
        let layers = self.text.num_layers();
        if cfg.hidden_size != self.text.config().hidden_size
            || cfg.target_layer_ids().iter().any(|&l| l >= layers)
        {
            anyhow::bail!(
                "DSpark drafter (hidden {}, taps {:?}) does not fit this target (hidden {}, {} layers)",
                cfg.hidden_size,
                cfg.target_layer_ids(),
                self.text.config().hidden_size,
                layers
            );
        }
        eprintln!(
            "[minicpm_v] DSpark drafter attached: block {} tokens, taps {:?}",
            cfg.block_size,
            cfg.target_layer_ids()
        );
        self.dspark = Some(drafter);
        Ok(())
    }

    /// Statistics of the last generation if it ran speculatively.
    #[must_use]
    pub fn last_spec_stats(&self) -> Option<&SpecStats> {
        self.last_spec.as_ref()
    }

    /// `[3, n]` position ids `base .. base + n` on all three axes.
    fn flat_positions(&self, base: i64, n: usize) -> Result<Tensor> {
        let row: Vec<u32> = (0..n).map(|i| (base + i as i64) as u32).collect();
        let mut flat = Vec::with_capacity(3 * n);
        for _ in 0..3 {
            flat.extend_from_slice(&row);
        }
        Ok(Tensor::from_vec(flat, (3, n), &self.device)?)
    }

    /// Leave speculation: consume `pending` (the last one is the anchor) in
    /// one pass, extend the drafter's context with their features, and return
    /// the greedy token that follows. Afterwards every layer has consumed
    /// exactly `committed` tokens, with nothing pending.
    fn spec_to_plain(
        &mut self,
        drafter: &mut DSparkDrafter,
        pending: &[u32],
        committed: &mut usize,
        tap_layers: &[usize],
    ) -> Result<u32> {
        let p = pending.len();
        let ids = Tensor::from_vec(pending.to_vec(), (1, p), &self.device)?;
        let hs = self.text.embed_only(&ids)?;
        let pos = self.flat_positions(*committed as i64 + self.rope_delta, p)?;
        let (logits, taps) = self
            .text
            .forward_spec(&hs, &pos, *committed, p, tap_layers, 1, true)?;
        self.text.commit_spec(*committed + p);
        let end = *committed + p;
        let have = drafter.ctx_len();
        if end > have {
            drafter.append_context(&taps.narrow(0, have - *committed, end - have)?)?;
        }
        *committed = end;
        Ok(logits.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?[0])
    }

    /// Resume speculation after a plain stretch. `snapshot` is the model state
    /// when the stretch began (`*committed` tokens consumed, the drafter's
    /// context already covering them); `consumed` are the stretch tokens the
    /// plain decode fed to the model. They are replayed in one batched pass so
    /// the drafter gets their features, leaving the model exactly where the
    /// plain decode was.
    fn plain_to_spec(
        &mut self,
        drafter: &mut DSparkDrafter,
        snapshot: &StateSnapshot,
        consumed: &[u32],
        committed: &mut usize,
        tap_layers: &[usize],
    ) -> Result<()> {
        self.text.restore_state(snapshot)?;
        let n = consumed.len();
        if n == 0 {
            return Ok(());
        }
        let ids = Tensor::from_vec(consumed.to_vec(), (1, n), &self.device)?;
        let hs = self.text.embed_only(&ids)?;
        let pos = self.flat_positions(*committed as i64 + self.rope_delta, n)?;
        let (_, taps) = self
            .text
            .forward_spec(&hs, &pos, *committed, n, tap_layers, 0, true)?;
        self.text.commit_spec(*committed + n);
        debug_assert_eq!(drafter.ctx_len(), *committed);
        drafter.append_context(&taps)?;
        *committed += n;
        Ok(())
    }

    /// Greedy speculative decoding (DSpark). Output is the same token stream
    /// as plain greedy decoding, up to floating-point differences between the
    /// batched verification pass and single-token decoding.
    ///
    /// Round structure: draft `block_size` tokens after the *anchor* (the
    /// latest token, not yet seen by the target), then verify
    /// `[pending tokens] + [drafts]` in one target pass. The GDN layers
    /// checkpoint their state right after the pending tokens, so the rejected
    /// drafts never touch it; the next round re-feeds the newly accepted
    /// tokens as its `pending` (see `Qwen3_5TextModel::forward_spec`).
    fn generate_spec(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        target_sizes: Option<&[(usize, usize)]>,
        cfg: &VlGenerationConfig,
        on_token: &mut impl FnMut(&str),
    ) -> Result<String> {
        const PREFILL_CHUNK: usize = 1024;
        let mut drafter = self.dspark.take().context("no DSpark drafter loaded")?;
        let result = (|| -> Result<String> {
            let tap_layers: Vec<usize> = drafter.config().target_layer_ids().to_vec();
            let block = drafter.config().block_size;
            let mask_id = drafter.config().mask_token_id();
            let text_dtype = self.text.dtype();
            let device = self.device.clone();
            let eos_ids = self.eos_token_ids.clone();
            let sync = || device.synchronize();

            // ── prefill, chunked, collecting target features for the drafter ──
            let (hidden, positions) =
                self.prepare_prefill(input_ids, pixel_values, target_sizes, 0)?;
            let prompt_len = hidden.dim(1)?;
            let mut all_taps = Vec::new();
            let mut first_logits = None;
            for off in (0..prompt_len).step_by(PREFILL_CHUNK) {
                let len = PREFILL_CHUNK.min(prompt_len - off);
                let last = off + len == prompt_len;
                let (logits, taps) = self.text.forward_spec(
                    &hidden.narrow(1, off, len)?,
                    &positions.narrow(1, off, len)?,
                    off,
                    len,
                    &tap_layers,
                    usize::from(last),
                    false,
                )?;
                self.text.commit_spec(off + len);
                all_taps.push(taps);
                if last {
                    first_logits = Some(logits);
                }
            }
            drafter.reset();
            drafter.append_context(&Tensor::cat(&all_taps, 0)?)?;
            drop(all_taps);

            let mut stats = SpecStats::default();
            let adaptive = cfg.dspark_adaptive.unwrap_or(true)
                && std::env::var("CRANE_DSPARK_ADAPTIVE").map_or(true, |v| v != "0");
            let mut policy = AdaptivePolicy::new(adaptive, self.plain_step_secs);
            if let Some((spec_rounds, plain_tokens)) = cfg.dspark_schedule {
                policy = policy.with_schedule(spec_rounds, plain_tokens);
            }
            let mut generated: Vec<u32> = Vec::with_capacity(cfg.max_new_tokens);
            let mut stream = TokenOutputStream::new(self.tokenizer.tokenizer.clone());
            let mut emit =
                |tok: u32, generated: &mut Vec<u32>, stream: &mut TokenOutputStream| -> bool {
                    if eos_ids.contains(&tok) {
                        return false;
                    }
                    if let Ok(Some(text)) = stream.next_token(tok) {
                        on_token(&text);
                    }
                    generated.push(tok);
                    true
                };

            let first = first_logits
                .context("empty prompt")?
                .argmax(candle_core::D::Minus1)?
                .to_vec1::<u32>()?[0];
            let mut alive = emit(first, &mut generated, &mut stream);
            stats.tokens = 1;

            // committed = tokens every layer has consumed; pending ends at the anchor.
            let mut committed = prompt_len;
            let mut pending = vec![first];
            let mut block_ids = vec![mask_id; block];
            let mut round_timed = 0.0f64;
            // The very first request times a plain decode step so the break-even
            // is known from the start (see `AdaptivePolicy::begin_calibration`).
            let mut go_plain = policy.begin_calibration();
            while alive && generated.len() < cfg.max_new_tokens {
                if !go_plain {
                    let anchor = *pending.last().expect("pending is never empty");
                    let round_start = std::time::Instant::now();
                    let t0 = round_start;

                    // ── draft ──
                    block_ids[0] = anchor;
                    let ids = Tensor::from_vec(block_ids.clone(), (1, block), &self.device)?;
                    let embeds = self.text.embed_only(&ids)?.squeeze(0)?;
                    let hid = drafter.block_hidden(&embeds)?;
                    let base = self
                        .text
                        .lm_head_logits(&hid.to_dtype(text_dtype)?)?
                        .to_dtype(DType::F32)?;
                    let draft = drafter.markov_draft(&base, &hid, anchor)?;
                    sync()?;
                    stats.draft_secs += t0.elapsed().as_secs_f64();

                    // ── verify: [pending] + [drafts], state checkpointed after pending ──
                    let t1 = std::time::Instant::now();
                    let p = pending.len();
                    let mut input = pending.clone();
                    input.extend_from_slice(&draft.tokens);
                    let input_t = Tensor::from_vec(input.clone(), (1, input.len()), &self.device)?;
                    let hs = self.text.embed_only(&input_t)?;
                    let pos =
                        self.flat_positions(committed as i64 + self.rope_delta, input.len())?;
                    let (logits, taps) = self.text.forward_spec(
                        &hs,
                        &pos,
                        committed,
                        p,
                        &tap_layers,
                        block + 1,
                        true,
                    )?;
                    let posterior = logits.argmax(candle_core::D::Minus1)?.to_vec1::<u32>()?;
                    sync()?;
                    stats.verify_secs += t1.elapsed().as_secs_f64();

                    let mut accepted = 0;
                    while accepted < block && draft.tokens[accepted] == posterior[accepted] {
                        accepted += 1;
                    }
                    let bonus = posterior[accepted];
                    stats.rounds += 1;
                    stats.accepted_drafts += accepted;

                    // ── emit ──
                    let mut produced = 0;
                    for &tok in draft.tokens[..accepted]
                        .iter()
                        .chain(std::iter::once(&bonus))
                    {
                        if generated.len() >= cfg.max_new_tokens {
                            break;
                        }
                        stats.tokens += 1;
                        stats.spec_tokens += 1;
                        produced += 1;
                        if !emit(tok, &mut generated, &mut stream) {
                            alive = false;
                            break;
                        }
                    }

                    // ── commit: model state = everything before this round's drafts ──
                    self.text.commit_spec(committed + p);
                    // The new anchor sits at committed + p + accepted; features for
                    // every earlier position are now known.
                    let anchor_pos = committed + p + accepted;
                    let have = drafter.ctx_len();
                    if anchor_pos > have {
                        drafter.append_context(&taps.narrow(
                            0,
                            have - committed,
                            anchor_pos - have,
                        )?)?;
                    }
                    committed += p;
                    pending = draft.tokens[..accepted].to_vec();
                    pending.push(bonus);
                    sync()?;
                    let round_secs = round_start.elapsed().as_secs_f64();
                    stats.other_secs +=
                        round_secs - (stats.draft_secs + stats.verify_secs - round_timed);
                    round_timed = stats.draft_secs + stats.verify_secs;

                    // ── is speculation still paying? If not, decode plainly for a while ──
                    go_plain = alive
                        && generated.len() < cfg.max_new_tokens
                        && policy.on_round(produced, round_secs);
                    if !go_plain {
                        continue;
                    }
                }
                go_plain = false;
                stats.switches += 1;
                let switch_start = std::time::Instant::now();
                // Consume the pending tokens in one pass (their features join the
                // drafter's context), then decode one token at a time from a
                // snapshot taken at that point.
                let mut next =
                    self.spec_to_plain(&mut drafter, &pending, &mut committed, &tap_layers)?;
                let snapshot: StateSnapshot = self.text.snapshot_state();
                sync()?;
                let mut switch_secs = switch_start.elapsed().as_secs_f64();
                let base_committed = committed;
                let mut stretch: Vec<u32> = Vec::new();
                let mut step_secs: Vec<f64> = Vec::new();
                let limit = policy.stretch_len();
                while stretch.len() < limit && generated.len() < cfg.max_new_tokens {
                    stats.tokens += 1;
                    stats.plain_tokens += 1;
                    stretch.push(next);
                    if !emit(next, &mut generated, &mut stream) {
                        alive = false;
                        break;
                    }
                    if stretch.len() >= limit || generated.len() >= cfg.max_new_tokens {
                        break;
                    }
                    let t = std::time::Instant::now();
                    let logits = self.decode_step(next, committed)?;
                    committed += 1;
                    next = logits
                        .squeeze(0)?
                        .to_dtype(DType::F32)?
                        .argmax(candle_core::D::Minus1)?
                        .to_scalar::<u32>()?;
                    step_secs.push(t.elapsed().as_secs_f64());
                }
                step_secs.sort_by(f64::total_cmp);
                let median = step_secs.get(step_secs.len() / 2).copied();
                policy.on_plain_done(median);
                self.plain_step_secs = policy.plain_secs();
                if !alive || generated.len() >= cfg.max_new_tokens {
                    break;
                }
                // Back to speculation: replay the stretch (all but its last token,
                // which becomes the anchor) from the snapshot in one batched pass
                // to regenerate the drafter's context.
                stats.switches += 1;
                let back_start = std::time::Instant::now();
                committed = base_committed;
                self.plain_to_spec(
                    &mut drafter,
                    &snapshot,
                    &stretch[..stretch.len() - 1],
                    &mut committed,
                    &tap_layers,
                )?;
                pending = vec![*stretch.last().expect("stretch is never empty")];
                sync()?;
                switch_secs += back_start.elapsed().as_secs_f64();
                stats.switch_secs += switch_secs;
                policy.on_switch_cost(switch_secs);
            }
            if let Ok(Some(text)) = stream.decode_rest() {
                on_token(&text);
            }
            self.last_spec = Some(stats);
            let text = self
                .tokenizer
                .tokenizer
                .decode(&generated, true)
                .map_err(E::msg)?;
            Ok(if cfg.strip_thinking {
                strip_thinking(&text)
            } else {
                text
            })
        })();
        self.dspark = Some(drafter);
        result
    }

    /// Answer `user_text` about `image` (pass `None` for a text-only turn).
    /// Resets the KV cache, so each call is an independent single turn.
    ///
    /// # Errors
    ///
    /// Returns an error if prompt rendering, the prefill/decode forward
    /// passes, the KV cache reset, or final detokenization fails.
    pub fn generate(
        &mut self,
        image: Option<&image::DynamicImage>,
        user_text: &str,
        cfg: &VlGenerationConfig,
        mut on_token: impl FnMut(&str),
    ) -> Result<String> {
        self.downsample_divisor = match cfg.downsample_mode.as_deref() {
            Some("4x") => 4,
            Some(_) => 16,
            None => self.config_divisor,
        };
        let (input_ids, processed) = self.render_prompt(image, user_text, cfg.enable_thinking)?;
        let input_tensor =
            Tensor::from_vec(input_ids.clone(), (1usize, input_ids.len()), &self.device)?;

        let (pixel_values, target_sizes) = match &processed {
            Some(p) => (
                Some(p.pixel_values.clone().unsqueeze(0)?),
                Some(p.target_sizes.clone()),
            ),
            None => (None, None),
        };

        self.clear_kv_cache()?;
        self.last_spec = None;
        if self.dspark.is_some() && cfg.dspark != Some(false) {
            return self.generate_spec(
                &input_tensor,
                pixel_values.as_ref(),
                target_sizes.as_deref(),
                cfg,
                &mut on_token,
            );
        }
        let mut logits = self.forward(
            &input_tensor,
            pixel_values.as_ref(),
            target_sizes.as_deref(),
            0,
        )?;

        let mut generated: Vec<u32> = Vec::with_capacity(cfg.max_new_tokens);
        let mut output_stream = TokenOutputStream::new(self.tokenizer.tokenizer.clone());
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
            if let Ok(Some(text)) = output_stream.next_token(next) {
                on_token(&text);
            }
            generated.push(next);
            logits = self.decode_step(next, cur_pos)?;
            cur_pos += 1;
        }
        if let Ok(Some(text)) = output_stream.decode_rest() {
            on_token(&text);
        }

        let text = self
            .tokenizer
            .tokenizer
            .decode(&generated, true)
            .map_err(E::msg)?;
        Ok(if cfg.strip_thinking {
            strip_thinking(&text)
        } else {
            text
        })
    }
}

fn strip_thinking(text: &str) -> String {
    match text.split_once("</think>") {
        Some((head, tail)) if head.trim_start().starts_with("<think>") => tail.trim().to_string(),
        _ => text.trim().to_string(),
    }
}
