//
// inference code of SNAC-24khz model from https://huggingface.co/hubertsiuzdak/snac_24khz
// the onnx we only using decoder at the moment
// mainly for driven Orpheus model

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use hound::{SampleFormat, WavSpec, WavWriter};

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

    pub fn save_audio_data_to_file(
        &self,
        audio_values: &Tensor,
        filename: &str,
        sample_rate: Option<u32>,
    ) -> Result<String> {
        let sample_rate = sample_rate.unwrap_or(24000);

        // Convert to f32 and flatten [batch, channels, samples] => [samples]
        let audio_values = audio_values.to_dtype(DType::F32)?.flatten_all()?;

        // Scale to i16 range and clamp values
        let scaled = audio_values
            .affine(32767.0, 0.0)? // Scale to maximum i16 range
            .clamp(-32768.0, 32767.0)? // Prevent overflow
            .round()?; // Properly round to nearest integer

        // Convert to i64 (closest available integer type)
        let audio_i64 = scaled.to_dtype(DType::I64)?;

        // Create WAV specification
        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        // Convert to i16 samples during writing
        let mut writer = WavWriter::create(filename, spec)?;
        for sample in audio_i64.to_vec1::<i64>()? {
            writer.write_sample(sample.clamp(i16::MIN as i64, i16::MAX as i64) as i16)?;
        }
        writer.finalize()?;

        Ok(filename.to_string())
    }
}