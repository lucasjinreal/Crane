//! [`Tts`] trait implementation for [`crane_core::models::qwen3_tts::Model`].

use anyhow::Result;
use candle_core::Tensor;
use crane_core::generation::SpeechOptions;
use crane_core::models::qwen3_tts::Model;
use crane_core::models::qwen3_tts::modeling::TalkerConfig;

use super::tts::{load_wav_f32, AudioInfo, Tts, TtsStream, VoiceInfo};

/// Maps a `codec_language_id` language name to its ISO 639-1 code.
///
/// Names not recognized (e.g. a language added to a future checkpoint) pass
/// through unchanged.
fn language_name_to_code(name: &str) -> &str {
    match name {
        "chinese" => "zh",
        "english" => "en",
        "german" => "de",
        "italian" => "it",
        "portuguese" => "pt",
        "spanish" => "es",
        "japanese" => "ja",
        "korean" => "ko",
        "french" => "fr",
        "russian" => "ru",
        other => other,
    }
}

/// Derives the sorted list of ISO 639-1 language codes a custom-voice talker
/// supports.
///
/// Reads `codec_language_id` keys, excluding the special `"auto"` value and
/// any name that is actually a per-speaker dialect override (listed in
/// `spk_is_dialect`'s values, e.g. `"beijing_dialect"`) rather than a
/// standalone language.
fn talker_languages(cfg: &TalkerConfig) -> Vec<String> {
    let dialects: std::collections::HashSet<&str> =
        cfg.spk_is_dialect.values().filter_map(serde_json::Value::as_str).collect();
    let mut languages: Vec<String> = cfg
        .codec_language_id
        .keys()
        .filter(|name| name.as_str() != "auto" && !dialects.contains(name.as_str()))
        .map(|name| language_name_to_code(name).to_string())
        .collect();
    languages.sort_unstable();
    languages
}

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
            "custom_voice" => {
                let languages = talker_languages(&self.config.talker_config);
                self.config
                    .talker_config
                    .spk_id
                    .keys()
                    .map(|name| VoiceInfo { name: name.clone(), languages: languages.clone() })
                    .collect()
            }
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

    /// Loads `ref_audio` at the speaker encoder's sample rate, then delegates
    /// to the inherent [`Model::generate_voice_clone`].
    fn generate_voice_clone(
        &mut self,
        text: &str,
        language: &str,
        ref_audio: &str,
        ref_text: &str,
        opts: &SpeechOptions,
    ) -> Result<Tensor> {
        let spk_sr = self.speaker_encoder_sample_rate();
        let ref_samples = load_wav_f32(ref_audio, spk_sr)?;
        let (tensor, _sample_rate) =
            Model::generate_voice_clone(self, text, language, &ref_samples, ref_text, opts)?;
        Ok(tensor)
    }

    fn generate_speech_stream(
        &mut self,
        text: &str,
        language: &str,
        voice: Option<&str>,
        opts: &SpeechOptions,
    ) -> Result<TtsStream<'_>> {
        let audio_info = self.audio_info();
        let stream = self.generate_speech_streaming(text, language, voice, opts)?;
        Ok(TtsStream::new(audio_info, stream))
    }
}

#[cfg(test)]
mod tests {
    use super::talker_languages;
    use super::TalkerConfig;

    fn talker_config(codec_language_id: &str, spk_is_dialect: &str) -> TalkerConfig {
        let json = format!(
            r#"{{
                "vocab_size": 1, "hidden_size": 1, "intermediate_size": 1,
                "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
                "code_predictor_config": {{
                    "hidden_size": 1, "intermediate_size": 1,
                    "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1
                }},
                "codec_language_id": {codec_language_id},
                "spk_is_dialect": {spk_is_dialect}
            }}"#
        );
        serde_json::from_str(&json).expect("valid TalkerConfig json")
    }

    #[test]
    fn talker_languages_excludes_auto_and_sorts() {
        let cfg = talker_config(
            r#"{"english": 0, "auto": 1, "chinese": 2}"#,
            "{}",
        );
        assert_eq!(talker_languages(&cfg), vec!["en", "zh"]);
    }

    #[test]
    fn talker_languages_excludes_dialect_overrides() {
        let cfg = talker_config(
            r#"{"chinese": 0, "beijing_dialect": 1, "english": 2}"#,
            r#"{"speaker_a": "beijing_dialect", "speaker_b": false}"#,
        );
        assert_eq!(talker_languages(&cfg), vec!["en", "zh"]);
    }

    #[test]
    fn talker_languages_empty_when_no_language_map() {
        let cfg = talker_config("{}", "{}");
        assert!(talker_languages(&cfg).is_empty());
    }
}
