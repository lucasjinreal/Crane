//! `SpeechConnector`: projects a tokenizer's VAE latents into the decoder's
//! hidden space. One instance each for the acoustic and semantic paths
//! (`model.acoustic_connector.*` / `model.semantic_connector.*` in the
//! checkpoint).
//!
//! Port of `kugelaudio_model.py`'s `SpeechConnector`
//! (`fc1 -> RMSNorm(eps=1e-6) -> fc2`, both linears bias=`true`, the
//! `1e-6` epsilon hardcoded in the Python rather than config-driven).

use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::models::with_tracing::{Linear, RmsNorm, linear};

/// Fixed in the Python (`LlamaRMSNorm(output_dim, eps=1e-6)`), not a
/// `config.json` field.
const NORM_EPS: f64 = 1e-6;

pub struct SpeechConnector {
    fc1: Linear,
    norm: RmsNorm,
    fc2: Linear,
}

impl SpeechConnector {
    pub fn load(input_dim: usize, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(input_dim, output_dim, vb.pp("fc1"))?,
            norm: RmsNorm::new(output_dim, NORM_EPS, vb.pp("norm"))?,
            fc2: linear(output_dim, output_dim, vb.pp("fc2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = self.norm.forward(&x)?;
        self.fc2.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    #[test]
    fn forward_shape_and_finite() {
        let device = Device::Cpu;
        let (input_dim, output_dim) = (6usize, 10usize);
        let mut t: HashMap<String, Tensor> = HashMap::new();
        let fill = |shape: &[usize]| -> Tensor {
            let n: usize = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|i| 0.01 * (i as f32 + 1.0)).collect();
            Tensor::from_vec(data, shape, &device).unwrap()
        };
        t.insert("fc1.weight".into(), fill(&[output_dim, input_dim]));
        t.insert("fc1.bias".into(), fill(&[output_dim]));
        t.insert("norm.weight".into(), Tensor::ones(output_dim, DType::F32, &device).unwrap());
        t.insert("fc2.weight".into(), fill(&[output_dim, output_dim]));
        t.insert("fc2.bias".into(), fill(&[output_dim]));
        let vb = VarBuilder::from_tensors(t, DType::F32, &device);

        let connector = SpeechConnector::load(input_dim, output_dim, vb).expect("load");
        let x = Tensor::rand(-1f32, 1f32, (3, input_dim), &device).unwrap();
        let y = connector.forward(&x).expect("forward");
        assert_eq!(y.dims(), &[3, output_dim]);
        let max_abs: f32 = y.abs().unwrap().max_all().unwrap().to_scalar().unwrap();
        assert!(max_abs.is_finite());
    }
}
