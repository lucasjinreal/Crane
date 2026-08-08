//! CosyVoice2-style Token2wav vocoder: turns MiniCPM-o's TTS speech-token
//! ids (`super::tts_llm`'s output) into a waveform. Real reference source:
//! `stepaudio2` (bundled in the `minicpmo-utils` PyPI sdist) — see the plan
//! doc for how it was obtained and the architecture summary.
//!
//! **Zero-shot here means "the checkpoint's own baked-in default voice"**,
//! not "no reference audio" — CosyVoice2's flow model is an x-vector
//! (speaker-embedding) conditioned CFM with no no-reference mode. This pass
//! precomputes the system default reference's `prompt_token`/`prompt_feat`/
//! `spk_emb` once (offline, via a Python script using the real `s3tokenizer`
//! + `campplus.onnx`) and ships them as a static asset
//! (`assets/token2wav/system_default_prompt.safetensors`) rather than
//! running either ONNX component at Rust runtime — see `prompt.rs`.

pub mod cfm;
pub mod conformer;
pub mod dit;
pub mod flow;
pub mod hifigan;
pub mod prompt;

pub use conformer::UpsampleConformerEncoderV2;
pub use dit::DiT;
pub use flow::Flow;
pub use hifigan::HiFTGenerator;
pub use prompt::SystemDefaultPrompt;

use candle_core::{DType, Device, Result, Tensor};

/// Speech-token ids -> waveform, end to end: [`Flow`] (conformer encoder +
/// DiT/CFM) produces a mel spectrogram conditioned on the system default
/// voice, [`HiFTGenerator`] turns that into audio.
pub struct Token2Wav {
    flow: Flow,
    hift: HiFTGenerator,
    prompt: SystemDefaultPrompt,
    device: Device,
    dtype: DType,
}

impl Token2Wav {
    /// # Errors
    ///
    /// Returns an error if any checkpoint asset is missing or malformed.
    pub fn new(model_path: &str, device: &Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            flow: Flow::new(model_path, device, dtype)?,
            hift: HiFTGenerator::new(model_path, device, dtype)?,
            prompt: SystemDefaultPrompt::load(model_path, device, dtype)?,
            device: device.clone(),
            dtype,
        })
    }

    /// `speech_tokens`: MiniCPM-o's TTS tower output (`MiniCpmTts::generate`).
    /// `n_timesteps`: CFM Euler steps (10 matches the checkpoint's own default).
    ///
    /// Returns mono 24kHz PCM samples in `[-1, 1]`.
    ///
    /// # Errors
    ///
    /// Returns an error if any sub-model forward pass fails.
    pub fn synthesize(&self, speech_tokens: &[u32], n_timesteps: usize) -> Result<Vec<f32>> {
        let token_ids: Vec<i64> = speech_tokens.iter().map(|&t| i64::from(t)).collect();
        let token = Tensor::from_vec(token_ids.clone(), (1, token_ids.len()), &self.device)?;

        // Noise length must match the *encoder's* output length: the
        // conformer upsamples the combined (prompt+generated) token
        // sequence by 2x uniformly — it's `(prompt_token_len +
        // gen_token_len) * 2`, not `prompt_feat_len + gen_token_len * 2`
        // (prompt_feat_len happens to equal `prompt_token_len * 2` for the
        // system-default reference, but deriving it from the token counts
        // directly is what actually matches `Flow::inference`'s own
        // `total_len = h.dim(1)` computation).
        let prompt_token_len = self.prompt.prompt_token.dim(1)?;
        let total_len = (prompt_token_len + token_ids.len()) * 2;
        let noise = Tensor::randn(0f32, 1f32, (1, 80, total_len), &self.device)?.to_dtype(self.dtype)?;

        let feat = self.flow.inference(&token, &self.prompt.prompt_token, &self.prompt.prompt_feat, &self.prompt.spk_emb, &noise, n_timesteps)?;
        self.hift.forward(&feat)
    }

    /// Output sample rate: `24000`.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        24_000
    }
}
