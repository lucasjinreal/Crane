//! [`Asr`] trait implementation for [`crane_core::models::qwen3_asr::Model`].

use anyhow::Result;
use crane_core::models::qwen3_asr::Model;

use super::asr::{Asr, Transcript, TranscribeOptions};

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
}
