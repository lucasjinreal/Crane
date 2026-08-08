//! Audio-embedding projection + pooling: takes [`super::audio_encoder::AudioEncoder`]'s
//! `d_model`-wide output and produces LLM-space tokens, mirroring
//! [`super::resampler::Resampler`]'s role for vision. Ported from
//! `modeling_minicpmo.py`'s `MultiModalProjector` + the `audio_avg_pooler`
//! (`nn.AvgPool1d`) used in `get_audio_embedding`.
//!
//! **Order matters**: the projector runs *before* pooling (project
//! `d_model` -> `llm_hidden`, then average-pool `audio_pool_step` adjacent
//! frames together), not the other way around — confirmed against
//! `get_audio_embedding`'s exact call order.

use std::io::{Read, Seek};

use candle_core::{Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::models::hunyuan_dense::modeling::Gguf;

pub struct AudioProjector {
    linear1: Linear,
    linear2: Linear,
    pool_step: usize,
}

impl AudioProjector {
    pub fn new(in_dim: usize, out_dim: usize, pool_step: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear1: linear(in_dim, out_dim, vb.pp("linear1"))?,
            linear2: linear(out_dim, out_dim, vb.pp("linear2"))?,
            pool_step,
        })
    }

    /// GGUF equivalent of [`Self::new`] — the projector's tensors
    /// (`audio_projector.linear1/2`) are bundled in the same
    /// `MiniCPM-o-4_5-audio-F16.gguf` file as [`super::audio_encoder::AudioEncoder`]'s,
    /// so this takes the same already-open `Gguf` reader rather than a
    /// separate file path.
    ///
    /// # Errors
    ///
    /// Returns an error if a required tensor is missing or has an
    /// unexpected shape.
    pub fn from_gguf<R: Read + Seek>(gg: &mut Gguf<R>, pool_step: usize) -> Result<Self> {
        let linear1 = Linear::new(gg.dequant_tensor("audio_projector.linear1.weight")?, Some(gg.dequant_tensor("audio_projector.linear1.bias")?.flatten_all()?));
        let linear2 = Linear::new(gg.dequant_tensor("audio_projector.linear2.weight")?, Some(gg.dequant_tensor("audio_projector.linear2.bias")?.flatten_all()?));
        Ok(Self { linear1, linear2, pool_step })
    }

    /// `audio_states`: `[B, T, d_model]` (encoder output). Returns `[B,
    /// T_pooled, llm_hidden]`, `T_pooled = (T - pool_step) / pool_step + 1`
    /// (`nn.AvgPool1d`'s default — no padding, floor division, trailing
    /// remainder frames dropped).
    pub fn forward(&self, audio_states: &Tensor) -> Result<Tensor> {
        let hidden = self.linear1.forward(audio_states)?;
        let hidden = hidden.relu()?;
        let hidden = self.linear2.forward(&hidden)?; // [B, T, llm_hidden]

        let (b, t, h) = hidden.dims3()?;
        let t_pooled = (t.saturating_sub(self.pool_step)) / self.pool_step + 1;
        let usable = t_pooled * self.pool_step;
        // [B, T, H] -> narrow to a whole number of pool windows -> average each.
        hidden.narrow(1, 0, usable)?.reshape((b, t_pooled, self.pool_step, h))?.mean(2)
    }
}

// ── HF-diff validation ─────────────────────────────────────────────────
//
// Compares `AudioEncoder` + `AudioProjector` together against a real Python
// forward pass through `modeling_minicpmo.py`'s `MiniCPMWhisperEncoder` +
// `MultiModalProjector` + `nn.AvgPool1d` (real `apm.*`/`audio_projection_layer.*`
// checkpoint weights only — see
// `/home/hahihula/.claude/jobs/02aa7312/tmp/minicpmo_diff/dump_audio.py`),
// on a real audio clip resampled to 16kHz mono (dumped as raw f32 PCM, since
// Rust doesn't have its own wav-loading/resampling pipeline in this crate —
// same "feed pre-processed input in" approach as `vision`'s hf_diff test for
// image preprocessing).
#[cfg(test)]
mod hf_diff {
    use super::*;
    use crate::models::minicpmo::config::load_config;
    use crate::models::qwen3_asr::feature_extractor::WhisperFeatureExtractor;
    use candle_core::{DType, Device};

    // Run with:
    // `cargo test -p crane-core --release --features cuda -- --ignored --nocapture minicpmo_audio_matches_python`
    #[test]
    #[ignore = "needs the real MiniCPM-o-4.5 checkpoint + a matching HF dump"]
    fn minicpmo_audio_matches_python() {
        let model_path = "/home/hahihula/mywork/ai/additional_models/MiniCPM-o-4_5";
        let diff_dir = "/home/hahihula/.claude/jobs/02aa7312/tmp/minicpmo_diff";

        #[cfg(feature = "cuda")]
        let (device, dtype) = (Device::new_cuda(0).unwrap(), DType::BF16);
        #[cfg(not(feature = "cuda"))]
        let (device, dtype) = (Device::Cpu, DType::F32);

        let config = load_config(&format!("{model_path}/config.json")).expect("load config");

        let filenames = crate::utils::utils::get_safetensors_files(model_path).expect("safetensors files");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device) }.expect("varbuilder");

        let encoder = super::super::audio_encoder::AudioEncoder::new(&config.audio_config, vb.pp("apm")).expect("load apm");
        let audio_output_dim = config.audio_config.encoder_ffn_dim / 4;
        let embed_dim = config.llm.hidden_size;
        let projector = AudioProjector::new(audio_output_dim, embed_dim, config.audio_pool_step, vb.pp("audio_projection_layer"))
            .expect("load audio_projection_layer");

        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{diff_dir}/audio_meta.json")).unwrap()).unwrap();
        let num_samples = meta["num_samples"].as_u64().unwrap() as usize;

        let sample_bytes = std::fs::read(format!("{diff_dir}/audio_samples_16k.bin")).unwrap();
        let samples: Vec<f32> =
            sample_bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        assert_eq!(samples.len(), num_samples);

        let mel_extractor = WhisperFeatureExtractor::new(config.audio_config.num_mel_bins, &device, dtype).expect("mel extractor");
        let features = mel_extractor.extract(&samples).expect("extract mel features");
        assert_eq!(features.real_frame_count, meta["mel_frames"].as_u64().unwrap() as usize);

        let encoder_out = encoder.forward(&features.input_features).expect("encoder forward");
        let pooled = projector.forward(&encoder_out).expect("projector forward");
        let pooled = pooled.to_dtype(DType::F32).unwrap();

        let rust_shape = pooled.dims().to_vec();
        let expected_shape: Vec<usize> =
            meta["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        assert_eq!(rust_shape, expected_shape);

        let rust_flat: Vec<f32> = pooled.flatten_all().unwrap().to_vec1().unwrap();

        let py_bytes = std::fs::read(format!("{diff_dir}/audio_embeds.bin")).unwrap();
        let py_flat: Vec<f32> =
            py_bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        assert_eq!(rust_flat.len(), py_flat.len());

        let dot: f64 = rust_flat.iter().zip(&py_flat).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
        let norm_a: f64 = rust_flat.iter().map(|a| f64::from(*a) * f64::from(*a)).sum::<f64>().sqrt();
        let norm_b: f64 = py_flat.iter().map(|b| f64::from(*b) * f64::from(*b)).sum::<f64>().sqrt();
        let cosine = dot / (norm_a * norm_b);
        let max_abs_diff = rust_flat.iter().zip(&py_flat).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);

        println!("cosine similarity: {cosine}, max abs diff: {max_abs_diff}");
        assert!(cosine > 0.99, "cosine similarity too low: {cosine}");
    }
}

// ── GGUF cross-check ────────────────────────────────────────────────────
//
// No Python reference needed here (the architecture itself was already
// HF-diff validated above) — this only needs to catch *tensor-mapping*
// bugs in the new `from_gguf` paths, which a direct Rust-vs-Rust
// comparison against the already-validated safetensors path catches just
// as well, without needing a fresh Python dump.
#[cfg(test)]
mod gguf_cross_check {
    use super::*;
    use crate::models::minicpmo::config::load_config;
    use crate::models::qwen3_asr::feature_extractor::WhisperFeatureExtractor;
    use candle_core::{DType, Device};

    // Run with:
    // `cargo test -p crane-core --release --features cuda -- --ignored --nocapture minicpmo_audio_gguf_matches_safetensors`
    #[test]
    #[ignore = "needs the real MiniCPM-o-4.5 checkpoint + the standalone audio GGUF file"]
    fn minicpmo_audio_gguf_matches_safetensors() {
        let model_path = std::env::var("CRANE_MINICPMO_DIR").unwrap_or_else(|_| "/home/hahihula/mywork/ai/additional_models/MiniCPM-o-4_5".to_string());
        let gguf_path = std::env::var("CRANE_MINICPMO_AUDIO_GGUF")
            .unwrap_or_else(|_| "/home/hahihula/mywork/ai/additional_models/MiniCPM-o-4_5-audio-F16.gguf".to_string());
        let diff_dir = "/home/hahihula/.claude/jobs/02aa7312/tmp/minicpmo_diff";

        #[cfg(feature = "cuda")]
        let (device, dtype) = (Device::new_cuda(0).unwrap(), DType::BF16);
        #[cfg(not(feature = "cuda"))]
        let (device, dtype) = (Device::Cpu, DType::F32);

        let config = load_config(&format!("{model_path}/config.json")).expect("load config");
        let audio_output_dim = config.audio_config.encoder_ffn_dim / 4;
        let embed_dim = config.llm.hidden_size;

        // ── safetensors path (already HF-diff validated above) ──
        let filenames = crate::utils::utils::get_safetensors_files(&model_path).expect("safetensors files");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device) }.expect("varbuilder");
        let st_encoder = super::super::audio_encoder::AudioEncoder::new(&config.audio_config, vb.pp("apm")).expect("load apm");
        let st_projector = AudioProjector::new(audio_output_dim, embed_dim, config.audio_pool_step, vb.pp("audio_projection_layer")).expect("load audio_projection_layer");

        // ── GGUF path (new) ──
        let mut gguf_file = std::fs::File::open(&gguf_path).unwrap_or_else(|e| panic!("open {gguf_path}: {e}"));
        let ct = candle_core::quantized::gguf_file::Content::read(&mut gguf_file).expect("parse gguf");
        let mut gg = crate::models::hunyuan_dense::modeling::Gguf::new(ct, &mut gguf_file, device.clone(), dtype);
        let gguf_encoder = super::super::audio_encoder::AudioEncoder::from_gguf(&mut gg, &config.audio_config).expect("load audio encoder from gguf");
        let gguf_projector = AudioProjector::from_gguf(&mut gg, config.audio_pool_step).expect("load audio projector from gguf");

        // ── same real input through both ──
        let sample_bytes = std::fs::read(format!("{diff_dir}/audio_samples_16k.bin")).unwrap();
        let samples: Vec<f32> = sample_bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();

        let mel_extractor = WhisperFeatureExtractor::new(config.audio_config.num_mel_bins, &device, dtype).expect("mel extractor");
        let features = mel_extractor.extract(&samples).expect("extract mel features");

        let st_out = st_projector.forward(&st_encoder.forward(&features.input_features).expect("st encoder forward")).expect("st projector forward").to_dtype(DType::F32).unwrap();
        let gguf_out = gguf_projector.forward(&gguf_encoder.forward(&features.input_features).expect("gguf encoder forward")).expect("gguf projector forward").to_dtype(DType::F32).unwrap();

        assert_eq!(st_out.dims(), gguf_out.dims(), "GGUF and safetensors paths produced different output shapes");

        let st_flat: Vec<f32> = st_out.flatten_all().unwrap().to_vec1().unwrap();
        let gguf_flat: Vec<f32> = gguf_out.flatten_all().unwrap().to_vec1().unwrap();

        let dot: f64 = st_flat.iter().zip(&gguf_flat).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
        let norm_a: f64 = st_flat.iter().map(|a| f64::from(*a) * f64::from(*a)).sum::<f64>().sqrt();
        let norm_b: f64 = gguf_flat.iter().map(|b| f64::from(*b) * f64::from(*b)).sum::<f64>().sqrt();
        let cosine = dot / (norm_a * norm_b);
        let max_abs_diff = st_flat.iter().zip(&gguf_flat).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);

        println!("GGUF vs safetensors cosine similarity: {cosine}, max abs diff: {max_abs_diff}");
        assert!(cosine > 0.999, "GGUF and safetensors audio outputs diverge too much: cosine {cosine}");
    }
}
