//! Prompt construction: builds the exact token sequence + voice-prompt
//! splice points `KugelAudioModel::generate` needs, mirroring
//! `kugelaudio_open.processors.kugelaudio_processor.KugelAudioProcessor.__call__`.
//!
//! Text tokenizer note: the checkpoint ships no tokenizer of its own.
//! `KugelAudioProcessor.from_pretrained` loads
//! `KugelAudioTextTokenizer.from_pretrained(language_model_pretrained_name)`,
//! defaulting to `"Qwen/Qwen2.5-1.5B"` — but that default is a red herring:
//! plain Qwen2.5's tokenizer doesn't define `<|vision_start|>` /
//! `<|vision_end|>` / `<|vision_pad|>` at all, so
//! `KugelAudioTextTokenizer._add_speech_special_tokens`'s
//! `add_special_tokens` call would append them at whatever the next free id
//! happens to be — not the fixed `151652`/`151653`/`151654` this module (and
//! `kugelaudio_processor.py`, and the checkpoint's `decoder_config.vocab_size
//! = 152064`) hardcode. Those exact ids match a **Qwen2-VL-family**
//! tokenizer instead (verified against `Qwen/Qwen2-VL-2B-Instruct`'s
//! `tokenizer_config.json`: `<|vision_start|>=151652`,
//! `<|vision_end|>=151653`, `<|vision_pad|>=151654`), which is what callers
//! should actually load and pass in here.

use anyhow::Result;
use tokenizers::Tokenizer;

use super::model::special_tokens::SPEECH_DIFFUSION_ID;

/// Samples of audio per acoustic-latent frame — `1000 * hop_length` (all
/// six `encoder_ratios` multiplied: `8*5*5*4*2*2 = 3200`), i.e. one
/// diffusion-generated latent frame is exactly `3200` audio samples at
/// 24kHz. Called `speech_compression_ratio` in the Python
/// (`KugelAudioProcessor.__init__`'s default of `3200`, not derived from
/// `config.json`).
pub const SPEECH_COMPRESSION_RATIO: usize = 3200;

/// Output of [`build_prompt`]: the full token sequence (voice-prompt frames
/// represented by `speech_diffusion_id` placeholders, matching what the
/// Python's `text_ids`/`input_ids` contains) plus where those placeholders
/// are, so the caller can splice in real voice-prompt embeddings at exactly
/// those positions.
pub struct PromptResult {
    pub token_ids: Vec<u32>,
    /// `true` at every voice-prompt placeholder position (there is at most
    /// one contiguous run — a single optional voice prompt).
    pub speech_input_mask: Vec<bool>,
    /// Number of voice-prompt placeholder frames (`0` if no voice prompt),
    /// i.e. `speech_input_mask.iter().filter(|&&b| b).count()`.
    pub voice_frame_count: usize,
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("kugelaudio prompt: tokenize {text:?}: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

/// Build a KugelAudio prompt for `text`, optionally conditioned on a voice
/// prompt of `voice_audio_num_samples` raw audio samples (pass `None` for
/// zero-shot generation with a built-in voice).
///
/// Does **not** append the trailing `speech_start` token that begins
/// generation — that's appended by [`super::model::KugelAudioModel::generate`]
/// itself, once, right before the autoregressive loop starts (mirroring
/// `KugelAudioProcessor.__call__` appending it, immediately followed by
/// `generate()`'s per-step loop treating it as the first "already generated"
/// token).
pub fn build_prompt(
    tokenizer: &Tokenizer,
    text: &str,
    voice_audio_num_samples: Option<usize>,
) -> Result<PromptResult> {
    const SYSTEM_PROMPT: &str = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n";

    let formatted_text = if text.trim_start().starts_with("Speaker") {
        text.trim().to_string()
    } else {
        format!("Speaker 0: {}", text.trim())
    };

    let mut token_ids = Vec::new();
    let mut speech_input_mask = Vec::new();
    let mut voice_frame_count = 0usize;

    token_ids.extend(encode(tokenizer, SYSTEM_PROMPT)?);
    speech_input_mask.resize(token_ids.len(), false);

    if let Some(num_samples) = voice_audio_num_samples {
        let voice_input_tokens = encode(tokenizer, " Voice input:\n")?;
        token_ids.extend(&voice_input_tokens);
        speech_input_mask.resize(token_ids.len(), false);

        let speaker_prefix = encode(tokenizer, " Speaker 0:")?;
        token_ids.extend(&speaker_prefix);
        speech_input_mask.resize(token_ids.len(), false);

        voice_frame_count = num_samples.div_ceil(SPEECH_COMPRESSION_RATIO);
        token_ids.extend(std::iter::repeat_n(SPEECH_DIFFUSION_ID, voice_frame_count));
        speech_input_mask.extend(std::iter::repeat_n(true, voice_frame_count));

        let newline = encode(tokenizer, "\n")?;
        token_ids.extend(&newline);
        speech_input_mask.resize(token_ids.len(), false);
    }

    token_ids.extend(encode(tokenizer, " Text input:\n")?);
    speech_input_mask.resize(token_ids.len(), false);

    token_ids.extend(encode(tokenizer, &format!(" {formatted_text}\n"))?);
    speech_input_mask.resize(token_ids.len(), false);

    token_ids.extend(encode(tokenizer, " Speech output:\n")?);
    speech_input_mask.resize(token_ids.len(), false);

    debug_assert_eq!(token_ids.len(), speech_input_mask.len());
    Ok(PromptResult {
        token_ids,
        speech_input_mask,
        voice_frame_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tokenizer() -> Tokenizer {
        // A minimal whitespace-split BPE-free tokenizer is enough to
        // exercise `build_prompt`'s structure/mask bookkeeping without
        // needing the real ~7MB Qwen2-VL tokenizer.json in this crate's
        // test fixtures.
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;

        let vocab = "<unk> the quick brown fox jumps over lazy dog \
             Speaker 0: Transform text provided by various speakers into speech output, \
             utilizing distinct voice of each respective speaker. Voice input: Text Speech"
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| {
                (
                    w.trim_matches(|c: char| ".,:".contains(c)).to_string(),
                    i as u32,
                )
            })
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".into())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(Whitespace {}));
        tok
    }

    #[test]
    fn no_voice_prompt_has_no_true_mask_positions() {
        let tok = tiny_tokenizer();
        let result = build_prompt(&tok, "hello there", None).expect("build_prompt");
        assert!(result.speech_input_mask.iter().all(|&b| !b));
        assert_eq!(result.voice_frame_count, 0);
        assert_eq!(result.token_ids.len(), result.speech_input_mask.len());
    }

    #[test]
    fn voice_prompt_mask_run_length_matches_frame_count() {
        let tok = tiny_tokenizer();
        // 3200 * 5 samples -> exactly 5 frames.
        let result = build_prompt(&tok, "hello there", Some(3200 * 5)).expect("build_prompt");
        assert_eq!(result.voice_frame_count, 5);
        let true_count = result.speech_input_mask.iter().filter(|&&b| b).count();
        assert_eq!(true_count, 5);
        // The true run must be contiguous.
        let first_true = result.speech_input_mask.iter().position(|&b| b).unwrap();
        assert!(
            result.speech_input_mask[first_true..first_true + 5]
                .iter()
                .all(|&b| b)
        );
    }

    #[test]
    fn partial_frame_rounds_up() {
        let tok = tiny_tokenizer();
        let result = build_prompt(&tok, "hi", Some(3200 * 2 + 1)).expect("build_prompt");
        assert_eq!(result.voice_frame_count, 3);
    }
}
