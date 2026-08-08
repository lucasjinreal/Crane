//! CosyVoice2's `CausalMaskedDiffWithXvec` — glues the speech-token
//! embedding, [`super::conformer::UpsampleConformerEncoderV2`], and
//! [`super::dit::DiT`]/[`super::cfm`] into the full speech-token-ids ->
//! mel-spectrogram pipeline. Ported from
//! `stepaudio2.cosyvoice2.flow.flow.CausalMaskedDiffWithXvec.inference`
//! (real reference source — the non-streaming, non-cached path only).
//!
//! Architecture constants below are hardcoded from the checkpoint's own
//! `assets/token2wav/flow.yaml` (a fixed release config, not something that
//! varies per user — same posture as other fixed architecture constants
//! elsewhere in this module).

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{embedding, linear, Embedding, Linear, VarBuilder};

use super::cfm::solve_euler;
use super::conformer::UpsampleConformerEncoderV2;
use super::dit::DiT;

const INPUT_SIZE: usize = 512;
const OUTPUT_SIZE: usize = 80; // mel channels
const SPK_EMBED_DIM: usize = 192;
const VOCAB_SIZE: usize = 6561;
const ENCODER_NUM_BLOCKS: usize = 6;
const ENCODER_NUM_UP_BLOCKS: usize = 4;
const ENCODER_UP_STRIDE: usize = 2;
const ENCODER_ATTENTION_HEADS: usize = 8;
const ENCODER_LINEAR_UNITS: usize = 2048;
const ENCODER_PRE_LOOKAHEAD_LEN: usize = 3;
const ENCODER_MAX_LEN: usize = 3000;

const DIT_IN_CHANNELS: usize = 320;
const DIT_HIDDEN_SIZE: usize = 512;
const DIT_DEPTH: usize = 16;
const DIT_NUM_HEADS: usize = 8;
const DIT_HEAD_DIM: usize = 64;
const DIT_MLP_RATIO: f64 = 4.0;

const INFERENCE_CFG_RATE: f64 = 0.7;

pub struct Flow {
    input_embedding: Embedding,
    spk_embed_affine_layer: Linear,
    encoder: UpsampleConformerEncoderV2,
    encoder_proj: Linear,
    estimator: DiT,
    device: Device,
    dtype: DType,
}

impl Flow {
    pub fn new(model_path: &str, device: &Device, dtype: DType) -> Result<Self> {
        let path = format!("{model_path}/assets/token2wav/flow.safetensors");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device) }?;

        let input_embedding = embedding(VOCAB_SIZE, INPUT_SIZE, vb.pp("input_embedding"))?;
        let spk_embed_affine_layer = linear(SPK_EMBED_DIM, OUTPUT_SIZE, vb.pp("spk_embed_affine_layer"))?;
        let encoder = UpsampleConformerEncoderV2::new(
            INPUT_SIZE,
            INPUT_SIZE,
            ENCODER_NUM_BLOCKS,
            ENCODER_NUM_UP_BLOCKS,
            ENCODER_PRE_LOOKAHEAD_LEN,
            ENCODER_UP_STRIDE,
            ENCODER_ATTENTION_HEADS,
            ENCODER_LINEAR_UNITS,
            ENCODER_MAX_LEN,
            vb.pp("encoder"),
        )?;
        let encoder_proj = linear(INPUT_SIZE, OUTPUT_SIZE, vb.pp("encoder_proj"))?;
        let estimator = DiT::new(
            DIT_IN_CHANNELS,
            OUTPUT_SIZE,
            DIT_HIDDEN_SIZE,
            DIT_DEPTH,
            DIT_NUM_HEADS,
            DIT_HEAD_DIM,
            DIT_MLP_RATIO,
            vb.pp("decoder").pp("estimator"),
        )?;

        Ok(Self { input_embedding, spk_embed_affine_layer, encoder, encoder_proj, estimator, device: device.clone(), dtype })
    }

    /// `token`: `[1, gen_len]` generated speech-token ids (i64). `prompt_token`:
    /// `[1, prompt_len]`. `prompt_feat`: `[1, prompt_len*2, 80]`. `spk_emb`:
    /// `[1, 192]` (raw, pre-affine-projection). `noise`: initial CFM noise
    /// `[1, 80, prompt_len*2 + gen_len*2]` — caller-supplied so HF-diff tests
    /// can feed a captured Python tensor; real inference should pass fresh
    /// `Tensor::randn`.
    ///
    /// Returns the newly generated portion's mel spectrogram, `[1, 80, gen_len*2]`.
    ///
    /// # Errors
    ///
    /// Returns an error if any sub-model forward pass fails.
    pub fn inference(
        &self,
        token: &Tensor,
        prompt_token: &Tensor,
        prompt_feat: &Tensor,
        spk_emb: &Tensor,
        noise: &Tensor,
        n_timesteps: usize,
    ) -> Result<Tensor> {
        let embedding = {
            let norm = spk_emb.sqr()?.sum_keepdim(1)?.sqrt()?;
            spk_emb.broadcast_div(&norm)?
        };
        let embedding = self.spk_embed_affine_layer.forward(&embedding)?; // [1, 80]

        let combined = Tensor::cat(&[prompt_token, token], 1)?; // [1, prompt_len+gen_len]
        let combined_embeds = self.input_embedding.forward(&combined)?.to_dtype(self.dtype)?; // [1, T, 512]

        let h = self.encoder.forward(&combined_embeds)?; // [1, T*2, 512]
        let h = self.encoder_proj.forward(&h)?; // [1, T*2, 80]

        let mel_len1 = prompt_feat.dim(1)?;
        let total_len = h.dim(1)?;
        let mel_len2 = total_len - mel_len1;

        let mut conds = vec![0f32; total_len * OUTPUT_SIZE];
        let prompt_feat_flat: Vec<f32> = prompt_feat.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        conds[..mel_len1 * OUTPUT_SIZE].copy_from_slice(&prompt_feat_flat);
        let conds = Tensor::from_vec(conds, (1, total_len, OUTPUT_SIZE), &self.device)?.to_dtype(self.dtype)?;
        let conds = conds.transpose(1, 2)?.contiguous()?; // [1, 80, T]

        let mu = h.transpose(1, 2)?.contiguous()?; // [1, 80, T]

        let feat = solve_euler(&self.estimator, noise, &mu, &embedding, &conds, n_timesteps, INFERENCE_CFG_RATE)?;
        let feat = feat.narrow(2, mel_len1, mel_len2)?;
        Ok(feat)
    }
}

// ── HF-diff validation ─────────────────────────────────────────────────
//
// Compares the full `Flow::inference` pipeline (token embedding ->
// `UpsampleConformerEncoderV2` -> `DiT`/CFM Euler solve) against a real
// Python run of `CausalMaskedDiffWithXvec.inference` (real `flow.pt`
// weights, real system-default-reference prompt assets — see
// `/home/hahihula/.claude/jobs/02aa7312/tmp/minicpmo_diff/dump_flow.py`).
// The dump seeds Python's RNG right before constructing `CausalConditionalCFM`
// so its `rand_noise` buffer is reproducible, then dumps the exact noise
// slice used — this test feeds that same noise in, making the comparison
// fully deterministic despite CFM sampling normally drawing fresh noise.
#[cfg(test)]
mod hf_diff {
    use super::*;
    use crate::models::minicpmo::token2wav::SystemDefaultPrompt;

    // Run with:
    // `cargo test -p crane-core --release -- --ignored --nocapture minicpmo_flow_matches_python`
    //
    // Deliberately F32 (not bf16, even with --features cuda) — this
    // pipeline is a 16-layer DiT run through 10 Euler steps, and the
    // Python dump itself used F32 for the same reason (tighter numeric
    // comparison, matching what was actually validated).
    #[test]
    #[ignore = "needs the real MiniCPM-o-4.5 checkpoint + a matching HF dump"]
    fn minicpmo_flow_matches_python() {
        let model_path = "/home/hahihula/mywork/ai/additional_models/MiniCPM-o-4_5";
        let diff_dir = "/home/hahihula/.claude/jobs/02aa7312/tmp/minicpmo_diff";
        let device = Device::Cpu;
        let dtype = DType::F32;

        let flow = Flow::new(model_path, &device, dtype).expect("load flow");
        let prompt = SystemDefaultPrompt::load(model_path, &device, dtype).expect("load system default prompt");

        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{diff_dir}/flow_meta.json")).unwrap()).unwrap();
        let gen_tokens: Vec<i64> = meta["gen_tokens"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        let token = Tensor::from_vec(gen_tokens.clone(), (1, gen_tokens.len()), &device).unwrap();

        let noise_shape: Vec<usize> = meta["noise_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let noise_bytes = std::fs::read(format!("{diff_dir}/flow_noise.bin")).unwrap();
        let noise_flat: Vec<f32> = noise_bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let noise = Tensor::from_vec(noise_flat, noise_shape.as_slice(), &device).unwrap();

        let n_timesteps = meta["n_timesteps"].as_u64().unwrap() as usize;
        let feat = flow
            .inference(&token, &prompt.prompt_token, &prompt.prompt_feat, &prompt.spk_emb, &noise, n_timesteps)
            .expect("flow inference");

        let expected_shape: Vec<usize> = meta["feat_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        assert_eq!(feat.dims(), expected_shape.as_slice());

        let rust_flat: Vec<f32> = feat.flatten_all().unwrap().to_vec1().unwrap();
        let py_bytes = std::fs::read(format!("{diff_dir}/flow_feat.bin")).unwrap();
        let py_flat: Vec<f32> = py_bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
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
