//! Shared ASR abstraction: trait, stream, and result/option types.
//!
//! Consumers (crane-serve, ...) hold a `Box<dyn Asr>` and
//! never import model-specific types.

use anyhow::Result;
pub use crane_core::generation::TranscribeOptions;

/// A finished or partial transcription result.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Transcript {
    /// The transcribed text.
    pub text: String,
    /// BCP-47-ish language code the model detected/used, if it reports one.
    pub language: Option<String>,
    /// `false` for a partial/incremental result that may still change;
    /// `true` for a final result.
    pub is_final: bool,
}

/// A stream of incremental transcripts produced during streaming ASR.
///
/// Consumers iterate with `while let Some(t) = stream.next_chunk()? { ... }`.
/// The last yielded [`Transcript`] has `is_final: true`; after that,
/// [`next_chunk`](AsrStream::next_chunk) returns `None`.
pub struct AsrStream<'a> {
    inner: Box<dyn Iterator<Item = Result<Transcript>> + 'a>,
}

impl<'a> AsrStream<'a> {
    /// Create a stream from a transcript iterator.
    pub fn new(iter: impl Iterator<Item = Result<Transcript>> + 'a) -> Self {
        Self {
            inner: Box::new(iter),
        }
    }

    /// Returns the next transcript, or `None` when the stream is complete.
    ///
    /// Returns `Err` if the underlying model encounters a transcription error.
    pub fn next_chunk(&mut self) -> Result<Option<Transcript>> {
        self.inner.next().transpose()
    }

    /// Create a single-item stream from a completed transcript.
    ///
    /// Used by the default [`Asr::transcribe_stream`] implementation for
    /// models without true streaming support.
    pub fn once(transcript: Transcript) -> Self {
        Self::new(std::iter::once(Ok(transcript)))
    }
}

/// Unified ASR interface.
///
/// Implemented by every ASR model in Crane. Consumers (crane-serve, ...)
/// use `Box<dyn Asr>` and never import model-specific types.
pub trait Asr {
    /// Sample rate this model expects input audio at (e.g. 16000).
    fn input_sample_rate(&self) -> u32;

    /// Transcribe a complete utterance.
    ///
    /// `audio` is f32 PCM at [`input_sample_rate`](Asr::input_sample_rate) (mono).
    /// Always returns a [`Transcript`] with `is_final: true`.
    fn transcribe(&mut self, audio: &[f32], opts: &TranscribeOptions) -> Result<Transcript>;

    /// Transcribe audio incrementally as it becomes available.
    ///
    /// The default implementation calls [`transcribe`](Asr::transcribe) once
    /// and wraps the result in a single-item stream; models with true
    /// streaming support override this method.
    fn transcribe_stream(
        &mut self,
        audio: &[f32],
        opts: &TranscribeOptions,
    ) -> Result<AsrStream<'_>> {
        let t = self.transcribe(audio, opts)?;
        Ok(AsrStream::once(t))
    }

    /// BCP-47 language codes this model claims to support (service discovery
    /// only — not used to reject a `TranscribeOptions::language` request).
    ///
    /// Implementations typically allocate on every call; callers that need
    /// the list more than once should cache the result.
    fn supported_languages(&self) -> Vec<String> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Transcript ---

    #[test]
    fn test_transcript_construction() {
        let t = Transcript {
            text: "hello world".to_string(),
            language: Some("en".to_string()),
            is_final: true,
        };
        assert_eq!(t.text, "hello world");
        assert_eq!(t.language.as_deref(), Some("en"));
        assert!(t.is_final);
    }

    #[test]
    fn test_transcript_default() {
        let t = Transcript::default();
        assert_eq!(t.text, "");
        assert_eq!(t.language, None);
        assert!(!t.is_final);
    }

    // --- TranscribeOptions ---

    #[test]
    fn test_transcribe_options_default() {
        let opts = TranscribeOptions::default();
        assert_eq!(opts.max_new_tokens, 8192);
        assert!((opts.temperature - 0.0).abs() < f64::EPSILON);
        assert_eq!(opts.top_p, None);
        assert!((opts.repetition_penalty - 1.05).abs() < f32::EPSILON);
        assert_eq!(opts.language, None);
    }

    // --- AsrStream ---

    #[test]
    fn test_asr_stream_once() -> Result<()> {
        let transcript = Transcript {
            text: "hi".to_string(),
            language: None,
            is_final: true,
        };
        let mut stream = AsrStream::once(transcript);

        let chunk = stream.next_chunk()?.expect("should have one chunk");
        assert_eq!(chunk.text, "hi");

        assert!(stream.next_chunk()?.is_none());
        Ok(())
    }

    #[test]
    fn test_asr_stream_multi_chunk() -> Result<()> {
        let chunks = vec![
            Ok(Transcript {
                text: "one".to_string(),
                language: None,
                is_final: false,
            }),
            Ok(Transcript {
                text: "one two".to_string(),
                language: None,
                is_final: false,
            }),
            Ok(Transcript {
                text: "one two three".to_string(),
                language: None,
                is_final: true,
            }),
        ];
        let mut stream = AsrStream::new(chunks.into_iter());

        assert_eq!(stream.next_chunk()?.expect("chunk 1").text, "one");
        assert_eq!(stream.next_chunk()?.expect("chunk 2").text, "one two");
        let last = stream.next_chunk()?.expect("chunk 3");
        assert_eq!(last.text, "one two three");
        assert!(last.is_final);

        assert!(stream.next_chunk()?.is_none());
        Ok(())
    }

    #[test]
    fn test_asr_stream_error_propagation() {
        let chunks: Vec<Result<Transcript>> = vec![Err(anyhow::anyhow!("transcription failed"))];
        let mut stream = AsrStream::new(chunks.into_iter());

        let result = stream.next_chunk();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("transcription failed")
        );
    }

    #[test]
    fn test_asr_stream_empty() -> Result<()> {
        let chunks: Vec<Result<Transcript>> = vec![];
        let mut stream = AsrStream::new(chunks.into_iter());
        assert!(stream.next_chunk()?.is_none());
        Ok(())
    }

    // --- Asr trait default methods ---

    struct MockAsr;

    impl Asr for MockAsr {
        fn input_sample_rate(&self) -> u32 {
            16000
        }

        fn transcribe(&mut self, _audio: &[f32], _opts: &TranscribeOptions) -> Result<Transcript> {
            Ok(Transcript {
                text: "mock transcript".to_string(),
                language: Some("en".to_string()),
                is_final: true,
            })
        }
    }

    #[test]
    fn test_default_transcribe_stream() -> Result<()> {
        let opts = TranscribeOptions::default();
        let mut model = MockAsr;
        let mut stream = model.transcribe_stream(&[0.0, 0.1], &opts)?;

        let chunk = stream.next_chunk()?.expect("should have one chunk");
        assert_eq!(chunk.text, "mock transcript");
        assert!(chunk.is_final);

        assert!(stream.next_chunk()?.is_none());
        Ok(())
    }

    #[test]
    fn test_input_sample_rate() {
        let model = MockAsr;
        assert_eq!(model.input_sample_rate(), 16000);
    }

    #[test]
    fn test_default_supported_languages_is_empty() {
        let model = MockAsr;
        assert!(model.supported_languages().is_empty());
    }
}
