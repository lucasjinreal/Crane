use anyhow::{Ok, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_onnx as onnx;
use std::collections::HashMap;

pub struct MoonshineASR {
    pub encoder_path: String,
    pub decoder_path: String,
    pub num_layers: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub decoder_start_token_id: i64,
    pub eos_token_id: i64,
    pub token_rate: usize,
    pub device: Device,
}

impl MoonshineASR {
    pub fn new(
        models_dir: &str,
        model_name: &str,
        token_rate: Option<usize>,
        device: &Device,
    ) -> Result<Self> {
        let encoder_path = format!("{}/encoder_model.onnx", models_dir);
        let decoder_path = format!("{}/decoder_model_merged.onnx", models_dir);

        let (num_layers, num_key_value_heads, head_dim) = if model_name.contains("tiny") {
            (6_usize, 8_usize, 36_usize)
        } else if model_name.contains("base") {
            (8_usize, 8_usize, 52_usize)
        } else {
            return Err(anyhow::anyhow!("Unknown model name"));
        };

        let token_rate = token_rate.unwrap_or_else(|| match model_name {
            name if name.contains("tiny") => 6,
            name if name.contains("base") => 6,
            _ => 6,
        });

        Ok(Self {
            encoder_path,
            decoder_path,
            num_layers,
            num_key_value_heads,
            head_dim,
            decoder_start_token_id: 1,
            eos_token_id: 2,
            token_rate,
            device: device.clone(),
        })
    }

    fn load_onnx_model(&self, path: &str) -> Result<onnx::onnx::ModelProto> {
        let model = onnx::read_file(path)?;
        Ok(model)
    }

    pub fn load_audio(audio_f: String) -> Result<Vec<f32>> {
        let mut reader = hound::WavReader::open(audio_f).expect("Failed to open WAV file");
        let spec = reader.spec();
        if spec.sample_rate != 16000 || spec.channels != 1 || spec.bits_per_sample != 16 {
            panic!("Unsupported audio format: expected 16kHz mono 16-bit");
        }
        let samples: Vec<i16> = reader.samples().map(Result::unwrap).collect();
        let audio: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        Ok(audio)
    }

    pub fn generate_from_audio(&self, audio_file: String) -> Result<Vec<i64>> {
        let audio = Self::load_audio(audio_file)?;
        self.generate(audio.as_slice(), None)
    }

    pub fn generate(&self, audio: &[f32], max_len: Option<usize>) -> Result<Vec<i64>> {
        let max_len = max_len.unwrap_or_else(|| {
            ((audio.len() as f64 / 16_000.0) * (self.token_rate as f64)).ceil() as usize
        });

        let encoder_model = self.load_onnx_model(&self.encoder_path)?;
        let decoder_model = self.load_onnx_model(&self.decoder_path)?;

        let mut encoder_inputs: HashMap<String, Tensor> = HashMap::new();
        encoder_inputs.insert(
            "input_values".to_string(),
            Tensor::from_slice(audio, &[1, audio.len()], &self.device)?,
        );

        let encoder_outputs: HashMap<String, Tensor> =
            onnx::simple_eval(&encoder_model, encoder_inputs)?;
        let last_hidden_state = encoder_outputs
            .get("last_hidden_state")
            .ok_or_else(|| anyhow::anyhow!("encoder returned no last_hidden_state"))?
            .clone();

        let mut past_key_values: HashMap<String, Tensor> = HashMap::new();
        for i in 0..self.num_layers {
            for a in &["decoder", "encoder"] {
                for b in &["key", "value"] {
                    let name = format!("past_key_values.{}.{}.{}", i, a, b);
                    let empty = vec![0f32; 0];
                    let t = Tensor::from_slice(
                        &empty,
                        &[1, self.num_key_value_heads, 0, self.head_dim],
                        &self.device,
                    )?;
                    past_key_values.insert(name, t);
                }
            }
        }

        let mut tokens: Vec<i64> = vec![self.decoder_start_token_id];
        let mut input_ids_tensor =
            Tensor::from_slice(&[self.decoder_start_token_id], &[1, 1], &self.device)?;

        for step in 0..max_len {
            let use_cache_branch = step > 0;
            let mut decoder_inputs: HashMap<String, Tensor> = HashMap::new();
            decoder_inputs.insert("input_ids".to_string(), input_ids_tensor.clone());
            decoder_inputs.insert(
                "encoder_hidden_states".to_string(),
                last_hidden_state.clone(),
            );
            decoder_inputs.insert(
                "use_cache_branch".to_string(),
                Tensor::from_slice(
                    &[if use_cache_branch { 1.0f32 } else { 0.0f32 }],
                    &[1],
                    &self.device,
                )?,
            );

            for (k, v) in &past_key_values {
                decoder_inputs.insert(k.clone(), v.clone());
            }

            let decoder_outputs: HashMap<String, Tensor> =
                onnx::simple_eval(&decoder_model, decoder_inputs)?;
            let logits_tensor = decoder_outputs
                .get("logits")
                .ok_or_else(|| anyhow::anyhow!("decoder returned no logits"))?
                .clone();

            let logits_buf: Vec<f32> = logits_tensor.i((0, 0, ..))?.to_vec1()?;

            let mut max_index = 0usize;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &v) in logits_buf.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    max_index = i;
                }
            }
            let next_token = max_index as i64;
            tokens.push(next_token);
            if next_token == self.eos_token_id {
                break;
            }

            input_ids_tensor = Tensor::from_slice(&[next_token], &[1, 1], &self.device)?;

            // Update past key values assuming output names match input names
            let keys: Vec<String> = past_key_values.keys().cloned().collect();
            for k in &keys {
                if let Some(pv) = decoder_outputs.get(k) {
                    if !use_cache_branch || k.as_str().contains("decoder") {
                        past_key_values.insert(k.clone(), pv.clone());
                    }
                }
            }
        }
        Ok(tokens)
    }
}

// fn main() -> Result<()> {
//     let model = MoonshineASR::new("./models/tiny", "tiny", None)?;
//     let dummy_audio = vec![0f32; 16000 * 3];
//     let tokens = model.generate(&dummy_audio, None)?;
//     println!("tokens: {:?}", tokens);
//     Ok(())
// }
