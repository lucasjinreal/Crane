// SPDX-License-Identifier: MIT
//! `ChatML` prompt construction for Audio8-TTS: text cleaning and the
//! `codes`-tensor layout the slow AR expects (row 0 = token IDs with the
//! reference voice's semantic tokens spliced in, rows 1..=N = the reference
//! voice's raw codec codebook indices at those same columns).

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

/// Inclusive Unicode code-point ranges treated as CJK for the purposes of
/// whitespace normalization (matches the reference implementation's
/// character classes for Hangul, CJK punctuation/symbols, Hiragana/Katakana,
/// CJK Unified Ideographs and extensions, and CJK compatibility forms).
const CJK_RANGES: &[(u32, u32)] = &[
    (0x1100, 0x11FF),
    (0x2E80, 0x2FDF),
    (0x3000, 0x303F),
    (0x3040, 0x30FF),
    (0x3100, 0x31FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7A3),
    (0xD7B0, 0xD7FF),
    (0xF900, 0xFAFF),
    (0xFE30, 0xFE4F),
    (0xFF01, 0xFF9F),
    (0x20000, 0x2FA1F),
];

fn is_cjk(c: char) -> bool {
    let cp = c as u32;
    CJK_RANGES.iter().any(|&(lo, hi)| cp >= lo && cp <= hi)
}

/// Line-break characters, matching the reference's `[\r\n\v\f\x1c-\x1e\x85  ]`.
fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\r' | '\n' | '\u{0B}' | '\u{0C}' | '\u{1C}'
            ..='\u{1E}' | '\u{85}' | '\u{2028}' | '\u{2029}'
    )
}

/// Approximates Unicode general category `Cc` (C0/C1 control characters).
/// The reference strips every category-`C*` code point (`Cc`/`Cf`/`Co`/`Cs`/`Cn`),
/// but real TTS input text overwhelmingly only ever contains stray `Cc`
/// bytes, so the rarer categories aren't handled here.
fn is_control(c: char) -> bool {
    let cp = c as u32;
    (cp <= 0x1F) || (0x7F..=0x9F).contains(&cp)
}

/// Collapses whitespace runs to a single space, except a run containing a
/// line break flanked by CJK characters on both sides, which is removed
/// entirely (CJK text doesn't use spaces between characters split across a
/// line break). Trims leading/trailing whitespace.
fn normalize_whitespace(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_whitespace() {
            let start = i;
            let mut has_line_break = false;
            while i < chars.len() && chars[i].is_whitespace() {
                has_line_break |= is_line_break(chars[i]);
                i += 1;
            }
            let left_is_cjk = start > 0 && is_cjk(chars[start - 1]);
            let right_is_cjk = i < chars.len() && is_cjk(chars[i]);
            if !(has_line_break && left_is_cjk && right_is_cjk) {
                out.push(' ');
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out.trim().to_string()
}

/// Strips control characters and normalizes whitespace, matching the
/// reference implementation's `clean_text`.
pub(crate) fn clean_text(text: &str) -> String {
    let filtered: String = text
        .chars()
        .filter(|&c| c.is_whitespace() || !is_control(c))
        .collect();
    normalize_whitespace(&filtered)
}

/// Cleans `text`, then prepends `<|speaker:0|>` unless it already contains a
/// `<|speaker:N|>` marker.
pub(crate) fn format_reference_text(text: &str) -> String {
    let cleaned = clean_text(text);
    if has_speaker_marker(&cleaned) {
        cleaned
    } else {
        format!("<|speaker:0|>{cleaned}")
    }
}

/// True if `text` contains a `<|speaker:N|>` marker for some digit sequence `N`.
fn has_speaker_marker(text: &str) -> bool {
    text.match_indices("<|speaker:").any(|(pos, pat)| {
        let after = &text[pos + pat.len()..];
        let digits_len = after.bytes().take_while(u8::is_ascii_digit).count();
        digits_len > 0 && after[digits_len..].starts_with("|>")
    })
}

/// Builds Audio8-TTS's `ChatML` prompt tensor from target text, a reference
/// voice's transcript, and its pre-encoded codec codes.
pub(crate) struct PromptBuilder {
    tokenizer: Tokenizer,
    semantic_begin_id: i64,
    num_codebooks: usize,
}

impl PromptBuilder {
    /// Loads `tokenizer_dir/tokenizer.json`.
    ///
    /// # Errors
    ///
    /// Returns an error if the tokenizer file is missing or malformed.
    pub(crate) fn new(
        tokenizer_dir: &Path,
        semantic_begin_id: i64,
        num_codebooks: usize,
    ) -> Result<Self> {
        let tokenizer_json = tokenizer_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_json)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("loading {}", tokenizer_json.display()))?;
        Ok(Self {
            tokenizer,
            semantic_begin_id,
            num_codebooks,
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("tokenizing {text:?}"))?;
        Ok(encoding.get_ids().iter().map(|&id| i64::from(id)).collect())
    }

    /// Builds the `(1, num_codebooks + 1, total_len)` `i64` prompt tensor:
    /// row 0 holds the `ChatML` token sequence with the reference voice's
    /// codebook-0 codes spliced in as semantic tokens, and rows
    /// `1..=num_codebooks` hold the reference voice's raw codec codes at
    /// those same columns (zero elsewhere).
    ///
    /// # Errors
    ///
    /// Returns an error if `reference_codes` isn't shaped
    /// `[num_codebooks][T > 0]` with every row the same length, or if
    /// tokenization fails.
    pub(crate) fn build(
        &self,
        target_text: &str,
        reference_text: &str,
        reference_codes: &[Vec<i64>],
    ) -> Result<Tensor> {
        validate_reference_codes(reference_codes, self.num_codebooks)?;

        let mut prefix = Vec::new();
        prefix.extend(self.encode("<|im_start|>system\n")?);
        prefix.extend(self.encode(
            "convert the provided text to speech reference to the following:\n\nText:\n",
        )?);
        prefix.extend(self.encode(&format_reference_text(reference_text))?);
        prefix.extend(self.encode("\n\nSpeech:\n")?);

        let semantic_ids: Vec<i64> = reference_codes[0]
            .iter()
            .map(|&code| code + self.semantic_begin_id)
            .collect();

        let mut suffix = Vec::new();
        suffix.extend(self.encode("<|im_end|>\n")?);
        suffix.extend(self.encode("<|im_start|>user\n")?);
        suffix.extend(self.encode(&clean_text(target_text))?);
        suffix.extend(self.encode("<|im_end|>\n")?);
        suffix.extend(self.encode("<|im_start|>assistant\n<|voice|>")?);

        let total_len = prefix.len() + semantic_ids.len() + suffix.len();
        let num_rows = self.num_codebooks + 1;
        let values = splice_prompt_rows(&prefix, &semantic_ids, &suffix, reference_codes);

        Tensor::from_vec(values, (1, num_rows, total_len), &Device::Cpu)
            .map_err(|e| anyhow!("building prompt tensor: {e}"))
    }
}

/// Validates `reference_codes` is shaped `[num_codebooks][T > 0]` with every
/// row the same length, returning `T`.
///
/// # Errors
///
/// Returns an error if the codebook count doesn't match `num_codebooks`, or
/// the rows are empty or of unequal length.
fn validate_reference_codes(reference_codes: &[Vec<i64>], num_codebooks: usize) -> Result<usize> {
    if reference_codes.len() != num_codebooks {
        bail!(
            "reference codes have {} codebooks, expected {}",
            reference_codes.len(),
            num_codebooks
        );
    }
    let ref_len = reference_codes.first().map_or(0, Vec::len);
    if ref_len == 0 || reference_codes.iter().any(|row| row.len() != ref_len) {
        bail!("reference codes must be non-empty and have equal-length rows");
    }
    Ok(ref_len)
}

/// Splices `semantic_ids` between `prefix` and `suffix` into row 0, and
/// `reference_codes` into rows `1..=reference_codes.len()` at the same
/// columns as `semantic_ids` (zero elsewhere), returning the flattened
/// `(reference_codes.len() + 1) * total_len` values for a
/// `(1, reference_codes.len() + 1, total_len)` prompt tensor.
fn splice_prompt_rows(
    prefix: &[i64],
    semantic_ids: &[i64],
    suffix: &[i64],
    reference_codes: &[Vec<i64>],
) -> Vec<i64> {
    let prefix_len = prefix.len();
    let ref_len = semantic_ids.len();
    let mut row0 = Vec::with_capacity(prefix_len + ref_len + suffix.len());
    row0.extend_from_slice(prefix);
    row0.extend_from_slice(semantic_ids);
    row0.extend_from_slice(suffix);
    let total_len = row0.len();

    let num_rows = reference_codes.len() + 1;
    let mut values = vec![0i64; num_rows * total_len];
    values[..total_len].copy_from_slice(&row0);
    for (codebook, row) in reference_codes.iter().enumerate() {
        let dst = (codebook + 1) * total_len + prefix_len;
        values[dst..dst + ref_len].copy_from_slice(row);
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    // A matching, non-empty, equal-length codebook count returns the row length.
    #[test]
    fn validate_reference_codes_accepts_consistent_rows() {
        let codes = vec![vec![1i64, 2, 3], vec![4, 5, 6]];
        assert_eq!(validate_reference_codes(&codes, 2).unwrap(), 3);
    }

    // A codebook count mismatch is an error.
    #[test]
    fn validate_reference_codes_rejects_wrong_codebook_count() {
        let codes = vec![vec![1i64, 2, 3]];
        assert!(validate_reference_codes(&codes, 2).is_err());
    }

    // `num_codebooks == 0` with no reference rows must error, not panic on
    // an out-of-bounds index into an empty `reference_codes`.
    #[test]
    fn validate_reference_codes_rejects_zero_codebooks() {
        let codes: Vec<Vec<i64>> = Vec::new();
        assert!(validate_reference_codes(&codes, 0).is_err());
    }

    // Unequal-length rows are an error.
    #[test]
    fn validate_reference_codes_rejects_unequal_row_lengths() {
        let codes = vec![vec![1i64, 2, 3], vec![4, 5]];
        assert!(validate_reference_codes(&codes, 2).is_err());
    }

    // Row 0 is prefix ++ semantic_ids ++ suffix; reference codebook rows land
    // at the same columns as semantic_ids, zero elsewhere.
    #[test]
    fn splice_prompt_rows_layout() {
        let prefix = [10i64, 11];
        let semantic_ids = [100i64, 101, 102];
        let suffix = [20i64];
        let reference_codes = vec![vec![1i64, 2, 3], vec![4, 5, 6]];

        let values = splice_prompt_rows(&prefix, &semantic_ids, &suffix, &reference_codes);

        let total_len = prefix.len() + semantic_ids.len() + suffix.len();
        assert_eq!(values.len(), (reference_codes.len() + 1) * total_len);
        assert_eq!(&values[..total_len], &[10, 11, 100, 101, 102, 20]);
        assert_eq!(&values[total_len..2 * total_len], &[0, 0, 1, 2, 3, 0]);
        assert_eq!(&values[2 * total_len..3 * total_len], &[0, 0, 4, 5, 6, 0]);
    }

    // Control characters (including DEL) are stripped, whitespace is kept.
    #[test]
    fn clean_text_strips_control_chars() {
        assert_eq!(clean_text("a\x00b\x01c\x7fd"), "abcd");
    }

    // Runs of ordinary whitespace collapse to a single space.
    #[test]
    fn clean_text_normalizes_whitespace() {
        assert_eq!(clean_text("hello   world\t!"), "hello world !");
    }

    // A line break between two CJK characters is removed entirely, not
    // replaced with a space, since CJK text doesn't space words apart.
    #[test]
    fn clean_text_cjk_linebreak_removed() {
        let input = "\u{4e00}\n\u{4e01}";
        assert_eq!(clean_text(input), "\u{4e00}\u{4e01}");
    }

    // A plain space (no line break) between CJK characters is preserved.
    #[test]
    fn clean_text_cjk_space_preserved() {
        let input = "\u{4e00} \u{4e01}";
        assert_eq!(clean_text(input), "\u{4e00} \u{4e01}");
    }

    // Reference text without a speaker marker gets one prepended.
    #[test]
    fn format_reference_text_prepends_speaker() {
        assert_eq!(format_reference_text("hello"), "<|speaker:0|>hello");
    }

    // Reference text that already has a speaker marker is left alone.
    #[test]
    fn format_reference_text_no_double() {
        assert_eq!(
            format_reference_text("<|speaker:0|>hello"),
            "<|speaker:0|>hello"
        );
    }

    // Spot-checks common CJK scripts (Han, Hiragana, Hangul) and one ASCII
    // character that must not be classified as CJK.
    #[test]
    fn is_cjk_identifies_common_ranges() {
        assert!(is_cjk('\u{4e00}')); // CJK Unified Ideographs
        assert!(is_cjk('\u{3040}')); // Hiragana block start (technically unassigned, but in-range)
        assert!(is_cjk('\u{ac00}')); // Hangul syllable
        assert!(!is_cjk('a'));
    }
}
