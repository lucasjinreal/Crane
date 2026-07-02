//! [`Tts`] trait implementation for [`crane_core::models::qwen3_tts::Model`].

use std::path::Path;

use anyhow::Result;
use candle_core::Tensor;
use crane_core::generation::SpeechOptions;
use crane_core::models::qwen3_tts::Model;

use super::tts::{AudioInfo, Tts, VoiceInfo};

impl Tts for Model {
    fn audio_info(&self) -> AudioInfo {
        AudioInfo {
            sample_rate: self.sample_rate(),
            channels: 1,
            bits_per_sample: 16,
        }
    }

    fn voices(&self) -> Vec<VoiceInfo> {
        match self.config.tts_model_type.as_deref().unwrap_or("base") {
            "custom_voice" => self
                .config
                .talker_config
                .spk_id
                .keys()
                .map(|name| VoiceInfo { name: name.clone(), languages: vec![] })
                .collect(),
            _ => vec![],
        }
    }

    fn supports_voice_cloning(&self) -> bool {
        self.config.tts_model_type.as_deref().unwrap_or("base") == "base"
    }

    /// Delegates to the inherent [`Model::generate_speech`], discarding the
    /// sample rate (available via [`Tts::audio_info`] instead).
    fn generate_speech(
        &mut self,
        text: &str,
        language: &str,
        voice: Option<&str>,
        opts: &SpeechOptions,
    ) -> Result<Tensor> {
        let (tensor, _sample_rate) = Model::generate_speech(self, text, language, voice, opts)?;
        Ok(tensor)
    }

    /// Delegates to the inherent [`Model::generate_voice_clone`].
    ///
    /// Converts `ref_audio: &Path` to `&str`; returns an error for non-UTF-8 paths.
    fn generate_voice_clone(
        &mut self,
        text: &str,
        language: &str,
        ref_audio: &Path,
        ref_text: &str,
        opts: &SpeechOptions,
    ) -> Result<Tensor> {
        let ref_audio_str = ref_audio
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("reference audio path is not valid UTF-8"))?;
        let (tensor, _sample_rate) =
            Model::generate_voice_clone(self, text, language, ref_audio_str, ref_text, opts)?;
        Ok(tensor)
    }
}
