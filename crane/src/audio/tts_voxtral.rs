//! [`Tts`] trait implementation for [`crane_core::models::voxtral_tts::Model`].

use anyhow::Result;
use candle_core::Tensor;
use crane_core::generation::SpeechOptions;
use crane_core::models::voxtral_tts::Model;

use super::tts::{AudioInfo, Tts, TtsStream, VoiceInfo};

/// ISO 639-1 codes used as voice-name prefixes for non-English presets.
const VOICE_LANGUAGE_PREFIXES: &[&str] = &["ar", "de", "es", "fr", "hi", "it", "nl", "pt"];

/// Derives a voice's language from its name.
///
/// Non-English presets are named `<lang>_<style>` (e.g. `es_female`); the
/// language is the prefix before the first `_`. English presets have no
/// language prefix (`casual_male`, `neutral_female`, `cheerful_female`, ...).
///
/// This is a naming-convention heuristic, not config-derived: a custom voice
/// file whose name happens to start with one of `VOICE_LANGUAGE_PREFIXES`
/// followed by `_` would be misclassified.
fn voice_name_languages(name: &str) -> Vec<String> {
    let prefix = name.split_once('_').map_or(name, |(prefix, _)| prefix);
    let language = if VOICE_LANGUAGE_PREFIXES.contains(&prefix) { prefix } else { "en" };
    vec![language.to_string()]
}

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
    /// The config carries no per-voice language metadata, so the language is
    /// derived from the voice name itself: preset names are prefixed with an
    /// ISO 639-1 code (e.g. `es_female`, `de_male`) for every language except
    /// English, whose presets have no prefix (`casual_male`, `neutral_female`,
    /// `cheerful_female`, ...).
    fn voices(&self) -> Vec<VoiceInfo> {
        self.available_voices()
            .into_iter()
            .map(|name| VoiceInfo { name: name.to_string(), languages: voice_name_languages(name) })
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

    /// Returns a streaming iterator that yields f32 PCM chunks as they are
    /// generated. The first chunk arrives after ~0.4 s; subsequent chunks
    /// arrive every ~2 s.
    fn generate_speech_stream(
        &mut self,
        text: &str,
        language: &str,
        voice: Option<&str>,
        opts: &SpeechOptions,
    ) -> Result<TtsStream<'_>> {
        // Capture audio_info before the mutable borrow from generate_speech_streaming.
        let audio_info = self.audio_info();
        let stream = self.generate_speech_streaming(text, language, voice, opts)?;
        Ok(TtsStream::new(audio_info, stream))
    }
}

#[cfg(test)]
mod tests {
    use super::voice_name_languages;

    #[test]
    fn test_voice_name_languages_prefixed() {
        assert_eq!(voice_name_languages("es_female"), vec!["es"]);
        assert_eq!(voice_name_languages("de_male"), vec!["de"]);
        assert_eq!(voice_name_languages("hi_female"), vec!["hi"]);
    }

    #[test]
    fn test_voice_name_languages_unprefixed_is_english() {
        assert_eq!(voice_name_languages("casual_male"), vec!["en"]);
        assert_eq!(voice_name_languages("neutral_female"), vec!["en"]);
        assert_eq!(voice_name_languages("cheerful_female"), vec!["en"]);
    }
}
