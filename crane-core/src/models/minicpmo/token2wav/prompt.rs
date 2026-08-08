//! Loads the precomputed system-default reference voice's conditioning
//! tensors — see the `token2wav` module doc on why this is precomputed
//! rather than derived at runtime from the two ONNX components.

use candle_core::{DType, Device, Result, Tensor};

pub struct SystemDefaultPrompt {
    /// `[1, prompt_len]`, `int64` speech-token ids of the reference audio.
    pub prompt_token: Tensor,
    /// `[1, mel_len, 80]`, the reference audio's mel spectrogram (`mel_len == prompt_len * 2`).
    pub prompt_feat: Tensor,
    /// `[1, 192]`, the reference audio's raw (pre-affine-projection) speaker embedding.
    pub spk_emb: Tensor,
}

impl SystemDefaultPrompt {
    /// # Errors
    ///
    /// Returns an error if the asset file is missing or malformed.
    pub fn load(model_path: &str, device: &Device, dtype: DType) -> Result<Self> {
        let path = format!("{model_path}/assets/token2wav/system_default_prompt.safetensors");
        let tensors = candle_core::safetensors::load(path, device)?;
        let get = |name: &str| -> Result<&Tensor> {
            tensors.get(name).ok_or_else(|| candle_core::Error::Msg(format!("system_default_prompt.safetensors missing {name}")))
        };

        let prompt_token = get("prompt_token")?.unsqueeze(0)?;
        let prompt_feat = get("prompt_feat")?.to_dtype(dtype)?.unsqueeze(0)?;
        let spk_emb = get("spk_emb")?.to_dtype(dtype)?.unsqueeze(0)?;

        Ok(Self { prompt_token, prompt_feat, spk_emb })
    }
}
