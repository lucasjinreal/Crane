//! Multi-modal projector for Qwen3-ASR: maps the audio encoder's `d_model`-
//! wide output to the text decoder's `output_dim`-wide (`hidden_size`)
//! embedding space.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Activation, VarBuilder};

use super::config::AudioConfig;
use crate::models::with_tracing;

/// Two-layer MLP projecting audio encoder output to the text decoder's
/// embedding width: `linear_1` -> activation (GELU on released checkpoints)
/// -> `linear_2`, both linear layers with bias.
///
/// The supplied [`VarBuilder`] must be scoped to the checkpoint's
/// `model.multi_modal_projector` prefix (tensors `linear_1.{weight,bias}`,
/// `linear_2.{weight,bias}` sit directly underneath).
#[derive(Debug, Clone)]
pub struct MultiModalProjector {
    linear_1: with_tracing::Linear,
    linear_2: with_tracing::Linear,
    activation: Activation,
}

impl MultiModalProjector {
    /// Builds the projector from an `audio_config` and a `VarBuilder`
    /// already scoped to `model.multi_modal_projector`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if a required weight tensor is missing or has
    /// an unexpected shape.
    // See `AudioEncoderAttention::new`'s comment on `VarBuilder` by-value.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(config: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let linear_1 = with_tracing::linear(config.d_model, config.d_model, vb.pp("linear_1"))?;
        let linear_2 = with_tracing::linear(config.d_model, config.output_dim, vb.pp("linear_2"))?;
        Ok(Self {
            linear_1,
            linear_2,
            activation: config.activation_function,
        })
    }

    /// Projects `[..., d_model]` audio encoder output to `[..., output_dim]`.
    ///
    /// # Errors
    ///
    /// Returns a candle error if `xs`'s last dimension doesn't match
    /// `d_model`.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.activation.forward(&self.linear_1.forward(xs)?)?;
        self.linear_2.forward(&xs)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device};

    use super::*;

    fn small_audio_config(d_model: usize, output_dim: usize) -> AudioConfig {
        AudioConfig {
            d_model,
            encoder_layers: 2,
            encoder_attention_heads: 2,
            num_key_value_heads: 2,
            encoder_ffn_dim: 16,
            output_dim,
            num_mel_bins: 16,
            downsample_hidden_size: 3,
            max_position_embeddings: 13,
            n_window: 50,
            n_window_infer: 800,
            conv_chunksize: 500,
            activation_function: Activation::Gelu,
            scale_embedding: false,
        }
    }

    fn make_projector_vb(config: &AudioConfig) -> VarBuilder<'static> {
        let device = &Device::Cpu;
        let dtype = DType::F32;
        let d = config.d_model;
        let out = config.output_dim;

        let mut t: HashMap<String, Tensor> = HashMap::new();
        t.insert(
            "linear_1.weight".into(),
            Tensor::zeros((d, d), dtype, device).expect("zeros"),
        );
        t.insert(
            "linear_1.bias".into(),
            Tensor::zeros(d, dtype, device).expect("zeros"),
        );
        t.insert(
            "linear_2.weight".into(),
            Tensor::zeros((out, d), dtype, device).expect("zeros"),
        );
        t.insert(
            "linear_2.bias".into(),
            Tensor::zeros(out, dtype, device).expect("zeros"),
        );

        VarBuilder::from_tensors(t, dtype, device)
    }

    #[test]
    fn projector_output_shape_0_6b() {
        let config = small_audio_config(896, 1024);
        let vb = make_projector_vb(&config);
        let projector = MultiModalProjector::new(&config, vb).expect("new");

        let xs = Tensor::zeros((1, 13, config.d_model), DType::F32, &Device::Cpu).expect("zeros");
        let out = projector.forward(&xs).expect("forward");
        assert_eq!(out.dims(), &[1, 13, config.output_dim]);
    }

    #[test]
    fn projector_output_shape_1_7b() {
        let config = small_audio_config(1024, 2048);
        let vb = make_projector_vb(&config);
        let projector = MultiModalProjector::new(&config, vb).expect("new");

        let xs = Tensor::zeros((1, 13, config.d_model), DType::F32, &Device::Cpu).expect("zeros");
        let out = projector.forward(&xs).expect("forward");
        assert_eq!(out.dims(), &[1, 13, config.output_dim]);
    }
}
