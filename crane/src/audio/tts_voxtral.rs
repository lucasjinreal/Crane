//! [`Tts`] trait implementation for [`crane_core::models::voxtral_tts::Model`].

use anyhow::Result;
use candle_core::Tensor;
use crane_core::generation::SpeechOptions;
use crane_core::models::voxtral_tts::Model;

use super::tts::{AudioInfo, Tts, VoiceInfo};

impl Tts for Model {
    fn audio_info(&self) -> AudioInfo {
        AudioInfo {
            sample_rate: self.sample_rate(),
            channels: 1,
            bits_per_sample: 16,
        }
    }

    /// Returns a [`VoiceInfo`] for each voice embedding loaded from disk.
    ///
    /// Languages are left empty because the config does not provide per-voice
    /// language metadata; empty means the voice works with any language.
    fn voices(&self) -> Vec<VoiceInfo> {
        self.available_voices()
            .into_iter()
            .map(|name| VoiceInfo { name: name.to_string(), languages: vec![] })
            .collect()
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
}
