//
// inference code of SNAC-24khz model from https://huggingface.co/hubertsiuzdak/snac_24khz
// the onnx we only using decoder at the moment
// mainly for driven Orpheus model

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};

#[derive(Debug)]
pub struct SNAC24DecoderONNX {
    model: candle_onnx::onnx::ModelProto,
}

impl SNAC24DecoderONNX {
    const SNAC_24_DECODER_ONNX_MODEL_PATH: &str = "checkpoints/snac_24khz_sim.onnx";

    pub fn new(model_path: Option<&str>, device: Option<&Device>) -> Result<Self> {
        let target_device = device.unwrap_or(&Device::Cpu);
        let model_path = model_path.unwrap_or(Self::SNAC_24_DECODER_ONNX_MODEL_PATH);
        if !std::path::Path::new(model_path).exists() {
            return Err(anyhow::anyhow!(
                "path not found {}, download from: https://huggingface.co/onnx-community/snac_24khz-ONNX/resolve/main/onnx/decoder_model_fp16.onnx",
                model_path
            ));
        }
        let model = candle_onnx::read_file(model_path)?;
        Ok(Self { model })
    }

    pub fn forward(
        &self,
        audio_code0: &Tensor,
        audio_code1: &Tensor,
        audio_code2: &Tensor,
    ) -> Result<Tensor> {
        let inputs = std::collections::HashMap::from_iter([
            ("c1".to_string(), audio_code0.clone()),
            ("c2".to_string(), audio_code1.clone()),
            ("c3".to_string(), audio_code2.clone()),
        ]);

        let out = candle_onnx::simple_eval(&self.model, inputs).unwrap();
        let out_names = &self.model.graph.as_ref().unwrap().output;

        let output = out.get(&out_names[0].name).unwrap().clone();
        // Extract and return audio_values output
        Ok(output)
    }
}