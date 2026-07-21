//! [`Asr`] trait implementation for [`crane_core::models::qwen3_asr::Model`].

use anyhow::Result;
use crane_core::models::qwen3_asr::Model;

use super::asr::{Asr, TranscribeOptions, Transcript};

/// BCP-47 language codes Qwen3-ASR claims to support, per its model card.
/// Sorted alphabetically.
const LANGUAGES: &[&str] = &[
    "ar", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fil", "fr", "hi", "hu", "id", "it",
    "ja", "ko", "mk", "ms", "nl", "pl", "pt", "ro", "ru", "sv", "th", "tr", "vi", "yue", "zh",
];

impl Asr for Model {
    fn input_sample_rate(&self) -> u32 {
        self.sample_rate()
    }

    /// Delegates to the inherent [`Model::transcribe`], wrapping the returned
    /// text in a [`Transcript`] (the model doesn't report a detected
    /// language, so `language` is always `None`).
    fn transcribe(&mut self, audio: &[f32], opts: &TranscribeOptions) -> Result<Transcript> {
        let text = Self::transcribe(self, audio, opts)?;
        Ok(Transcript {
            text,
            language: None,
            is_final: true,
        })
    }

    fn supported_languages(&self) -> Vec<String> {
        LANGUAGES.iter().map(ToString::to_string).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::LANGUAGES;

    #[test]
    fn supported_languages_is_sorted_and_nonempty() {
        assert!(!LANGUAGES.is_empty());
        let mut sorted = LANGUAGES.to_vec();
        sorted.sort_unstable();
        assert_eq!(LANGUAGES, sorted.as_slice());
    }

    #[test]
    fn supported_languages_contains_expected_codes() {
        for code in ["en", "zh", "de"] {
            assert!(LANGUAGES.contains(&code), "missing language code {code}");
        }
    }
}
