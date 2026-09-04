// SPDX-License-Identifier: MIT
//! [`Tts`] trait implementation for [`crane_core::models::audio8_tts::Model`].

use anyhow::Result;
use candle_core::Tensor;
use crane_core::generation::SpeechOptions;
use crane_core::models::audio8_tts::Model;

use super::pcm::AudioInfo;
use super::tts::{Tts, VoiceInfo};

impl Tts for Model {
    fn audio_info(&self) -> AudioInfo {
        AudioInfo {
            sample_rate: self.sample_rate(),
            channels: 1,
            bits_per_sample: 16,
        }
    }

    /// Audio8-TTS's Preview release ships a single bundled reference voice
    /// with no selection mechanism, so there are no named voices to list.
    fn voices(&self) -> Vec<VoiceInfo> {
        vec![]
    }

    /// Delegates to the inherent [`Model::generate_speech`], discarding the
    /// sample rate (available via [`Tts::audio_info`] instead).
    ///
    /// `language` and `voice` are accepted for `Tts`-trait parity but have
    /// no effect: the ONNX graph takes no language-ID or voice-selection
    /// input, and `Model::generate_speech` never reads either parameter.
    /// Whatever language comes out is whichever language `text` is written
    /// in, since Audio8-TTS's backbone was trained multilingually and
    /// infers the spoken language from the text itself rather than from an
    /// explicit tag.
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
