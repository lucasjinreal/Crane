//! [`Tts`] trait implementation for [`crane_core::models::voxcpm2::VoxCpm2Model`].
//!
//! `generate_speech` is zero-shot. `generate_voice_clone` maps onto
//! VoxCPM2's "Ultimate Cloning" mode (`VoxCpm2Conditioning::Continuation` —
//! reference audio *and* its transcript, audio-continuation style) since
//! that's the only one of VoxCPM2's three real conditioning modes whose
//! shape matches the trait's `ref_audio`+`ref_text` signature (same mapping
//! `tts_qwen3.rs`'s own `generate_voice_clone` uses). The other two modes
//! (transcript-free "Controllable Cloning", and the combined
//! reference-prefix-plus-continuation mode) aren't reachable through this
//! trait's shape — use `VoxCpm2Model::generate_speech_conditioned` directly
//! for those (see `example/src/voxcpm2_simple.rs`).
//!
//! `voices()` still returns no presets — cloning is always driven by a
//! caller-supplied reference clip, never a discrete preset list.
//! `generate_speech_stream` is left at the trait default (single-chunk-wrap)
//! — real streaming generation is a separate, not-yet-implemented pass (see
//! `crane_core::models::voxcpm2`'s module docs).

use anyhow::Result;
use candle_core::Tensor;
use crane_core::generation::SpeechOptions;
use crane_core::models::voxcpm2::{VoxCpm2Conditioning, VoxCpm2GenerationConfig, VoxCpm2Model};

use super::pcm::{load_wav_f32, AudioInfo};
use super::tts::{Tts, VoiceInfo};

impl Tts for VoxCpm2Model {
    fn audio_info(&self) -> AudioInfo {
        AudioInfo { sample_rate: self.sample_rate, channels: 1, bits_per_sample: 16 }
    }

    /// No discrete presets — VoxCPM2 is zero-shot per-utterance, or cloned
    /// from a caller-supplied reference clip.
    fn voices(&self) -> Vec<VoiceInfo> {
        vec![]
    }

    fn supports_voice_cloning(&self) -> bool {
        true
    }

    /// `language`/`voice` are unused: VoxCPM2's zero-shot path infers
    /// prosody/language from the text itself and has no voice selection.
    fn generate_speech(
        &mut self,
        text: &str,
        _language: &str,
        _voice: Option<&str>,
        opts: &SpeechOptions,
    ) -> Result<Tensor> {
        // `max_new_tokens` doc says "codec frames"; VoxCPM2's closest analog
        // is its own generation-step count (each step yields one 4-frame
        // latent patch) — pass through directly as an upper bound rather
        // than inventing an unjustified conversion factor. The model's own
        // stop head almost always ends generation well before this cap.
        let cfg = VoxCpm2GenerationConfig { max_len: opts.max_new_tokens.max(1), ..Default::default() };
        VoxCpm2Model::generate_speech(self, text, &cfg)
    }

    /// Maps onto `VoxCpm2Conditioning::Continuation`: `ref_audio` is treated
    /// as prompt audio to continue from, `ref_text` as its transcript
    /// (concatenated with `text` before tokenizing — matches the reference's
    /// own `prompt_text + target_text` behavior). `language`/`voice` unused,
    /// same as `generate_speech`.
    fn generate_voice_clone(
        &mut self,
        text: &str,
        _language: &str,
        ref_audio: &str,
        ref_text: &str,
        opts: &SpeechOptions,
    ) -> Result<Tensor> {
        let sr = self.encoder_sample_rate();
        let samples = load_wav_f32(ref_audio, sr)?;
        let prompt_feat = self.encode_reference_audio(&samples, true)?;
        let conditioning =
            VoxCpm2Conditioning::Continuation { prompt_text: ref_text.to_string(), prompt_feat };
        let cfg = VoxCpm2GenerationConfig { max_len: opts.max_new_tokens.max(1), ..Default::default() };
        self.generate_speech_conditioned(text, &conditioning, &cfg)
    }
}
