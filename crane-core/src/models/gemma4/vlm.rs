//! Gemma 4 vision-language model.
//!
//! Combines the vision encoder with the text decoder for multimodal inference.
//! Image tokens in the input are replaced with projected vision features.

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use super::modeling::{Gemma4Config, Gemma4Model, Gemma4TextConfig};
use super::vision::{Gemma4MultimodalEmbedder, Gemma4VisionConfig, Gemma4VisionModel};
use crate::utils::token_output_stream::TokenOutputStream;
use crate::utils::utils;

/// Full Gemma 4 config.json with both text and vision configs.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma4VLConfig {
    pub text_config: Gemma4TextConfig,
    pub vision_config: Gemma4VisionConfig,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
}

fn default_image_token_id() -> u32 {
    258880
}

pub struct Gemma4VLModel {
    pub tokenizer: TokenOutputStream,
    pub device: Device,
    pub dtype: DType,
    vision_tower: Gemma4VisionModel,
    embed_vision: Gemma4MultimodalEmbedder,
    language_model: Gemma4Model,
    image_token_id: u32,
}

impl Gemma4VLModel {
    pub fn new(model_path: &str, device: &Device, dtype: &DType) -> Result<Self> {
        let tokenizer_path = std::path::Path::new(model_path).join("tokenizer.json");
        if !tokenizer_path.exists() {
            anyhow::bail!("Tokenizer not found at {}", tokenizer_path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        let filenames = utils::get_safetensors_files(model_path)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, *dtype, device) }?;

        let config_file = std::path::Path::new(model_path).join("config.json");
        let config_data = std::fs::read(config_file)?;
        let config: Gemma4VLConfig = serde_json::from_slice(&config_data)?;

        let model_vb = vb.pp("model");

        // Vision tower
        let vision_tower = Gemma4VisionModel::new(
            &config.vision_config,
            model_vb.pp("vision_tower"),
        )?;

        // Vision → text projection
        let embed_vision = Gemma4MultimodalEmbedder::new(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            config.vision_config.rms_norm_eps,
            model_vb.pp("embed_vision"),
        )?;

        // Text decoder (multimodal checkpoint: weights under model.language_model)
        let language_model = Gemma4Model::new(
            &config.text_config,
            vb.clone(),
            true, // is_multimodal
        )?;

        Ok(Self {
            tokenizer: TokenOutputStream::new(tokenizer),
            device: device.clone(),
            dtype: *dtype,
            vision_tower,
            embed_vision,
            language_model,
            image_token_id: config.image_token_id,
        })
    }

    /// Run vision encoder on pixel values and project to text space.
    pub fn encode_image(
        &self,
        pixel_values: &Tensor,
        pixel_position_ids: &Tensor,
        padding_positions: &Tensor,
    ) -> Result<Tensor> {
        let vision_features = self.vision_tower.forward(
            pixel_values,
            pixel_position_ids,
            padding_positions,
        )?;
        let projected = self.embed_vision.forward(&vision_features)?;
        Ok(projected)
    }

    /// Forward pass for VLM: embed text, splice in vision features, run decoder.
    ///
    /// * `input_ids` — token IDs including image placeholder tokens
    /// * `image_embeds` — projected vision features [B, num_image_tokens, hidden_size]
    /// * `start_pos` — KV cache position
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        image_embeds: Option<&Tensor>,
        start_pos: usize,
    ) -> Result<Tensor> {
        // HF replaces image token IDs with PAD before embedding.
        // This ensures the PLE (per-layer embeddings) use PAD's embedding
        // at image positions, not the image placeholder token's embedding.
        let pad_token_id = 0u32; // Gemma's pad_token_id
        let llm_input_ids = if image_embeds.is_some() {
            let ids: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;
            let masked: Vec<u32> = ids.iter().map(|&t| {
                if t == self.image_token_id { pad_token_id } else { t }
            }).collect();
            Tensor::new(masked.as_slice(), input_ids.device())?.reshape(input_ids.shape())?
        } else {
            input_ids.clone()
        };

        // Get text embeddings using masked IDs (PAD at image positions)
        let mut hidden_states = self.language_model.embed(&llm_input_ids)?;

        // Replace image token positions with vision embeddings
        if let Some(img_emb) = image_embeds {
            hidden_states = self.splice_image_features(
                input_ids, // original IDs to find image token positions
                &hidden_states,
                img_emb,
            )?;
        }

        // Run text decoder with masked IDs (for PLE) and modified embeddings
        self.language_model
            .forward_embeds(&llm_input_ids, hidden_states, start_pos)
            .map_err(Into::into)
    }

    /// Replace positions where input_ids == image_token_id with image embeddings.
    fn splice_image_features(
        &self,
        input_ids: &Tensor,
        text_embeds: &Tensor,
        image_embeds: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = text_embeds.dims3()?;
        let ids: Vec<u32> = input_ids.flatten_all()?.to_vec1()?;

        let mut result = text_embeds.clone();
        let mut img_offset = 0usize;

        for b in 0..b_sz {
            for s in 0..seq_len {
                let token = ids[b * seq_len + s];
                if token == self.image_token_id {
                    // Get the image embedding for this position
                    let img_emb = image_embeds
                        .narrow(0, b, 1)?
                        .narrow(1, img_offset, 1)?; // [1, 1, hidden_size]
                    result = result.slice_assign(
                        &[b..b + 1, s..s + 1, 0..hidden_size],
                        &img_emb,
                    )?;
                    img_offset += 1;
                }
            }
            img_offset = 0; // reset per batch
        }

        Ok(result)
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }
}
