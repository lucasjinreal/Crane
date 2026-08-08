//! Qwen 3.5 vision-language model.
//!
//! Glues the [`vision::Qwen3_5VisionModel`] (ViT + 2×2-spatial-merge MLP)
//! onto [`Qwen3_5TextModel`] (hybrid Mamba/Transformer text decoder).
//!
//! Forward flow:
//! 1. Tokenize the prompt with `<|image_pad|>` × N placeholder tokens at
//!    each image position.
//! 2. Run the vision tower on the pre-processed pixel values → projected
//!    image embeddings `[total_image_tokens, hidden_size]`.
//! 3. Embed the input_ids normally, then **splice** the image embeddings
//!    over the `<|image_pad|>` positions.
//! 4. Build a `[3, S]` position_ids tensor: text tokens advance a single
//!    T/H/W counter, image tokens use the (t, h, w) coordinates derived
//!    from `image_grid_thw`.
//! 5. Call [`Qwen3_5TextModel::forward_embeds`] with the spliced hidden
//!    states + 3D position_ids; MRoPE handles the rotation correctly.
//!
//! Single-sequence only for now (the hybrid cache complexity isn't worth
//! tackling in the MVP). The Qwen 3.5 engine already caps
//! `max_concurrent=1`, so this matches the existing constraint.

use anyhow::{Context, Error as E, Result};
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use super::config::{load_config, VisionConfig};
use super::model::Qwen3_5TextModel;
use super::processor::{load_preprocessor_config, PreprocessorConfig};
use super::vision::Qwen3_5VisionModel;
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::utils;

/// Options for [`Qwen3_5VLModel::generate`]. Decoding is greedy (argmax) —
/// the hybrid GDN cache makes sampled rollback awkward, and every current
/// caller wants a deterministic description.
#[derive(Debug, Clone)]
pub struct VlGenerationConfig {
    pub max_new_tokens: usize,
    /// Strip the `<think> … </think>` preamble from the returned string.
    /// The tokens are still generated (and streamed), just not returned.
    pub strip_thinking: bool,
}

impl Default for VlGenerationConfig {
    fn default() -> Self {
        Self { max_new_tokens: 256, strip_thinking: true }
    }
}

pub struct Qwen3_5VLModel {
    pub tokenizer: TokenOutputStream,
    pub device: Device,
    pub dtype: DType,
    pub image_token_id: u32,
    /// `<|image_pad|>` appears N times per image, where N = h/spatial_merge *
    /// w/spatial_merge (so the text model sees merged patch tokens, not raw
    /// ones). Cached here so the caller can build the input_ids sequence.
    pub spatial_merge_size: usize,
    /// The checkpoint's `preprocessor_config.json`, so callers don't have to
    /// load it separately just to turn a `DynamicImage` into pixel values.
    pub preprocessor: PreprocessorConfig,
    /// Token ids that terminate generation, resolved from the tokenizer vocab
    /// (`<|im_end|>`, `<|endoftext|>`, `<|im_start|>`) rather than hardcoded.
    pub eos_token_ids: Vec<u32>,
    /// MRoPE position the next decoded token should use. Seeded by `forward`;
    /// diverges from the absolute token index whenever the prompt has images.
    next_mrope_pos: u32,
    vision: Qwen3_5VisionModel,
    text: Qwen3_5TextModel,
}

impl Qwen3_5VLModel {
    /// Load a multimodal Qwen 3.5 checkpoint from a HF directory.
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        Self::new_with_options(model_path, device, dtype, None)
    }

    /// `quant` requests in-situ quantization of the text-model linears
    /// (same semantics as [`super::model::Model::new_with_options`]).
    /// Vision tower linears are NOT quantized — they're tiny (depth=12, ~600M
    /// params total) and quantized linears don't play well with bf16 vision
    /// activations.
    pub fn new_with_options(
        model_path: &str,
        device: &Device,
        dtype: &DType,
        quant: Option<GgmlDType>,
    ) -> Result<Self> {
        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!(
                "Tokenizer not found at {}; multimodal Qwen 3.5 still needs tokenizer.json \
                 alongside the safetensors (chat template too, if you want tools)",
                tokenizer_path.display()
            );
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let filenames = utils::get_safetensors_files(model_path)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, *dtype, device) }
            .with_context(|| format!("mmap safetensors at {model_path}"))?;

        let config_path = std::path::Path::new(model_path).join("config.json");
        let cfg = load_config(config_path.to_str().context("non-UTF8 path")?)?;
        let vcfg: &VisionConfig = cfg
            .vision_config
            .as_ref()
            .context("config.json has no vision_config; this model is text-only")?;

        let image_token_id = cfg.image_token_id.unwrap_or(248_056);
        eprintln!(
            "[qwen3_5_vl] loading vision tower: depth={} hidden={} out_hidden={} patch={} merge={}",
            vcfg.depth, vcfg.hidden_size, vcfg.out_hidden_size, vcfg.patch_size, vcfg.spatial_merge_size,
        );
        let vision = Qwen3_5VisionModel::new(vcfg, vb.pp("model").pp("visual"))
            .context("build vision tower")?;

        eprintln!("[qwen3_5_vl] loading text model");
        let text = Qwen3_5TextModel::new(&cfg, vb, device, *dtype, quant)
            .context("build text model")?;

        let preprocessor = load_preprocessor_config(model_path)?;

        // Resolve stop tokens by name. `<|im_start|>` is included because the
        // prompt is hand-rendered (see `render_prompt`) — without the chat
        // template's stop strings the model will happily open a new turn.
        let vocab = tokenizer.get_vocab(true);
        let eos_token_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
            .iter()
            .filter_map(|t| vocab.get(*t).map(|id| *id as u32))
            .collect();
        if eos_token_ids.is_empty() {
            anyhow::bail!("tokenizer has none of <|im_end|> / <|endoftext|> / <|im_start|>");
        }

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype: *dtype,
            image_token_id,
            spatial_merge_size: vcfg.spatial_merge_size,
            preprocessor,
            eos_token_ids,
            next_mrope_pos: 0,
            vision,
            text,
        })
    }

    /// Encode a batch of pre-processed images.
    ///
    /// `pixel_values`: `[total_patches, in_channels * temporal * patch * patch]`
    /// — the flat concatenation of every patch across every image.
    /// `image_grid_thw`: `[num_images, 3]` — `(t, h, w)` patches per image.
    ///
    /// Returns projected embeddings ready to splice into the text token stream:
    /// `[total_image_tokens, hidden_size]`.
    pub fn encode_images(
        &self,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
    ) -> Result<Tensor> {
        let (out, _deepstack) = self
            .vision
            .forward(pixel_values, image_grid_thw)
            .context("vision forward")?;
        Ok(out)
    }

    /// Build the 3D position-ids tensor `[3, S]` for a sequence that has
    /// contiguous `<|image_pad|>` spans (one per image).
    ///
    /// Mirrors HF `Qwen3_5VLModel.get_rope_index`: a single running counter
    /// walks the sequence. Text tokens take `[p, p, p]` and advance it by 1.
    /// An image span of `t × (h/merge) × (w/merge)` tokens takes
    /// `[base + t_i, base + h_i, base + w_i]` in row-major order — the local
    /// grid coordinates **offset by the running counter** — after which the
    /// counter resumes at `max(position in the span) + 1`, i.e.
    /// `base + max(t, h, w)`.
    ///
    /// Returns the position tensor and the counter value the next token after
    /// this sequence should use (needed by [`Self::decode_step`], since with
    /// images that value is much smaller than the absolute token index).
    pub fn build_position_ids(
        &self,
        text_ids: &[u32],
        image_grid_thw: &Tensor,
        start_pos: usize,
    ) -> Result<(Tensor, u32)> {
        let grids: Vec<Vec<u32>> = image_grid_thw.to_vec2::<u32>()?;
        let merge = self.spatial_merge_size as u32;
        let seq_len = text_ids.len();

        let mut positions = vec![[0u32; 3]; seq_len];
        let mut next_pos = start_pos as u32;
        let mut image_idx = 0usize;
        let mut i = 0usize;

        while i < seq_len {
            if text_ids[i] != self.image_token_id {
                positions[i] = [next_pos; 3];
                next_pos += 1;
                i += 1;
                continue;
            }

            let grid = grids
                .get(image_idx)
                .with_context(|| format!("image span {image_idx} has no image_grid_thw entry"))?;
            let (gt, gh, gw) = (grid[0], grid[1] / merge, grid[2] / merge);
            let span = (gt * gh * gw) as usize;
            if i + span > seq_len {
                anyhow::bail!(
                    "image {image_idx} needs {span} placeholder tokens but only {} remain",
                    seq_len - i
                );
            }
            let base = next_pos;
            let hw = gh * gw;
            for k in 0..span {
                let k = k as u32;
                positions[i + k as usize] =
                    [base + k / hw, base + (k % hw) / gw, base + (k % hw) % gw];
            }
            next_pos = base + gt.max(gh).max(gw);
            i += span;
            image_idx += 1;
        }

        let flat: Vec<u32> = positions.into_iter().flatten().collect();
        Ok((Tensor::from_vec(flat, (3, seq_len), &self.device)?, next_pos))
    }

    /// Forward pass for a vision-language prefill.
    ///
    /// `input_ids` already has `<|image_pad|>` placeholder tokens in place
    /// (length = text tokens + sum of merged_image_tokens).
    /// `pixel_values` / `image_grid_thw` come from the image preprocessor.
    /// Returns logits of shape `[B, V]` (last position only — matches the
    /// text-only forward contract).
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        start_pos: usize,
    ) -> Result<Tensor> {
        let image_embeds = match (pixel_values, image_grid_thw) {
            (Some(pv), Some(gt)) => Some(self.encode_images(pv, gt)?),
            _ => None,
        };

        let (_b, seq_len) = input_ids.dims2()?;
        let mut hidden_states = self.text.embed_only(input_ids)?;

        if let Some(img_emb) = image_embeds {
            hidden_states =
                splice_image_features(input_ids, &hidden_states, &img_emb, self.image_token_id)?;
        }

        let position_ids = if image_grid_thw.is_some() {
            let ids: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
            let (pos, next) = self.build_position_ids(&ids, image_grid_thw.unwrap(), start_pos)?;
            self.next_mrope_pos = next;
            pos
        } else {
            let flat: Vec<u32> = (0..seq_len)
                .flat_map(|i| std::iter::repeat((start_pos + i) as u32).take(3))
                .collect();
            self.next_mrope_pos = (start_pos + seq_len) as u32;
            Tensor::from_vec(flat, (3, seq_len), &self.device)?
        };

        self.text
            .forward_embeds(&hidden_states, &position_ids, start_pos, None)
    }

    /// Decode one new token. The KV cache must already hold the prefill
    /// (`start_pos` is the absolute index of the new token, used for the cache
    /// offset). No image embeddings are needed — they're already in the cache.
    ///
    /// The MRoPE position is *not* `start_pos`: image spans consume far fewer
    /// position slots than tokens, so the rope counter is tracked separately
    /// (seeded by the preceding [`Self::forward`]) and advanced by 1 here.
    pub fn decode_step(
        &mut self,
        token: u32,
        start_pos: usize,
    ) -> Result<Tensor> {
        let input = Tensor::from_vec(vec![token], (1usize, 1usize), &self.device)?;
        let hidden = self.text.embed_only(&input)?;
        let p = self.next_mrope_pos;
        let pos = Tensor::from_vec(vec![p, p, p], (3usize, 1usize), &self.device)?;
        self.next_mrope_pos = p + 1;
        self.text.forward_embeds(&hidden, &pos, start_pos, None)
    }

    pub fn clear_kv_cache(&mut self) {
        self.text.reset_gdn_caches().expect("GDN reset failed");
    }

    /// Build the `input_ids` for one user turn, expanding the single
    /// `<|image_pad|>` marker into `n_image_tokens` copies.
    ///
    /// The prompt is rendered by hand rather than through `chat_template.jinja`
    /// on purpose: the shipped template emits a "# Tools" preamble that sends
    /// the model into a loop under greedy decoding.
    fn render_prompt(&self, user_text: &str, n_image_tokens: usize) -> Result<Vec<u32>> {
        let rendered = if n_image_tokens > 0 {
            format!(
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_text}<|im_end|>\n<|im_start|>assistant\n"
            )
        } else {
            format!("<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n")
        };
        let base = self
            .tokenizer
            .tokenizer
            .encode(rendered, false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let mut ids = Vec::with_capacity(base.len() + n_image_tokens);
        let mut spliced = false;
        for &tok in &base {
            if tok == self.image_token_id {
                if spliced {
                    anyhow::bail!("prompt contains more than one <|image_pad|> marker");
                }
                ids.extend(std::iter::repeat_n(tok, n_image_tokens));
                spliced = true;
            } else {
                ids.push(tok);
            }
        }
        if n_image_tokens > 0 && !spliced {
            anyhow::bail!("<|image_pad|> did not survive tokenization");
        }
        Ok(ids)
    }

    /// Answer `user_text` about `image` (pass `None` for a text-only turn).
    ///
    /// Handles preprocessing, prompt construction, prefill and the greedy
    /// decode loop. `on_token` receives each newly decoded token's text as it
    /// is produced; pass `|_| {}` if you don't need streaming.
    ///
    /// This resets the KV cache, so each call is an independent single turn.
    pub fn generate(
        &mut self,
        image: Option<&image::DynamicImage>,
        user_text: &str,
        cfg: &VlGenerationConfig,
        mut on_token: impl FnMut(&str),
    ) -> Result<String> {
        let processed = match image {
            Some(img) => Some(self.preprocessor.process(img, &self.device, self.dtype)?),
            None => None,
        };

        let (pixel_values, grid_tensor, n_image_tokens) = match &processed {
            Some(p) => {
                let (t, h, w) = p.grid_thw;
                let merge = self.preprocessor.merge_size as u32;
                let n = (t * (h / merge) * (w / merge)) as usize;
                let grid = Tensor::from_vec(vec![t, h, w], (1usize, 3usize), &self.device)?;
                (Some(&p.pixel_values), Some(grid), n)
            }
            None => (None, None, 0),
        };

        let input_ids = self.render_prompt(user_text, n_image_tokens)?;
        let input_tensor =
            Tensor::from_vec(input_ids.clone(), (1usize, input_ids.len()), &self.device)?;

        self.clear_kv_cache();
        let mut logits = self.forward(&input_tensor, pixel_values, grid_tensor.as_ref(), 0)?;

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

        let text = self
            .tokenizer
            .tokenizer
            .decode(&generated, true)
            .map_err(E::msg)?;
        Ok(if cfg.strip_thinking { strip_thinking(&text) } else { text })
    }
}

/// Drop a leading `<think> … </think>` block. Qwen 3.5 opens one even when the
/// prompt asks it not to think, and it is noise in a description.
fn strip_thinking(text: &str) -> String {
    match text.split_once("</think>") {
        Some((head, tail)) if head.trim_start().starts_with("<think>") => tail.trim().to_string(),
        _ => text.trim().to_string(),
    }
}

/// Replace positions where `input_ids == image_token_id` with the corresponding
/// row of `image_embeds`. Both `hidden_states` and `image_embeds` share the
/// last dim (text hidden size).
///
/// `pub(crate)`: also reused by `minicpm_v::vlm` — the splice-at-placeholder
/// operation is model-agnostic (Qwen 3.5-VL and MiniCPM-V-4.6 both use a
/// single repeated placeholder token id).
pub(crate) fn splice_image_features(
    input_ids: &Tensor,
    hidden_states: &Tensor,
    image_embeds: &Tensor,
    image_token_id: u32,
) -> Result<Tensor> {
    let (b_sz, seq_len, hidden_size) = hidden_states.dims3()?;
    let ids: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
    let mut result = hidden_states.clone();

    let mut img_idx = 0usize;
    let total_img = image_embeds.dim(0)?;

    for b in 0..b_sz {
        for s in 0..seq_len {
            let token = ids[b * seq_len + s];
            if token == image_token_id {
                if img_idx >= total_img {
                    anyhow::bail!(
                        "more image-placeholder positions ({}) than image embeddings ({})",
                        img_idx + 1,
                        total_img
                    );
                }
                let img_emb = image_embeds
                    .narrow(0, img_idx, 1)?
                    .reshape((1, 1, hidden_size))?
                    .to_dtype(result.dtype())?;
                result = result.slice_assign(
                    &[b..b + 1, s..s + 1, 0..hidden_size],
                    &img_emb,
                )?;
                img_idx += 1;
            }
        }
    }

    Ok(result)
}