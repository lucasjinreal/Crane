//! CJK multi-character-token splitting wrapper.
//!
//! VoxCPM2 was trained with every multi-character Chinese *vocab* token
//! split into individual characters, so inference must apply the same
//! transform or Chinese text tokenizes into pieces the model never saw.
//!
//! Ported from the function `voxcpm2.py` actually calls —
//! `voxcpm/model/utils.py`'s `mask_multichar_chinese_tokens` — **not** the
//! checkpoint's own `tokenization_voxcpm2.py` (a separate file for other
//! consumers, e.g. vLLM, with different — and meaningfully different —
//! logic: it checks four CJK Unicode blocks and pre-validates every
//! expansion character against the vocab; the actual model code checks only
//! the main CJK Unified Ideographs block (`U+4E00..=U+9FFF`) and does no
//! such validation, silently falling back to `<unk>` per HF's normal
//! `convert_tokens_to_ids` behavior for any OOV expansion character).
//!
//! Two-step algorithm (faithfully reproduced, not simplified to a
//! single-step check on each token's own text — the two differ whenever a
//! word-initial token like `"▁你好"` has no separately-vocabbed unmarked
//! counterpart `"你好"`, which the two-step form correctly declines to
//! split and a naive one-step form would not):
//! 1. Build a set of vocab token strings that are, **as stored** (before
//!    stripping any `▁` marker), 2+ characters and entirely within the CJK
//!    range — a token with a leading `▁` marker never enters this set
//!    (`▁` itself isn't CJK), only genuinely marker-free multi-char entries do.
//! 2. For every vocab token (marked or not), strip its own `▁` and check
//!    *that* against the set from step 1 — if it matches, this token id
//!    expands to the individual characters' token ids.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use tokenizers::Tokenizer;

/// SentencePiece's word-boundary marker (`▁`, U+2581), stripped before the
/// multichar-CJK check.
const SP_MARKER: char = '\u{2581}';

fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

fn is_multichar_cjk(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else { return false };
    if !is_cjk(first) {
        return false;
    }
    let mut count = 1;
    for c in chars {
        if !is_cjk(c) {
            return false;
        }
        count += 1;
    }
    count >= 2
}

pub struct VoxCpm2Tokenizer {
    inner: Tokenizer,
    /// `token_id -> [char_token_id, ...]` for multi-char CJK vocab entries.
    split_map: HashMap<u32, Vec<u32>>,
}

impl VoxCpm2Tokenizer {
    pub fn new(inner: Tokenizer) -> Self {
        let split_map = Self::build_split_map(&inner);
        Self { inner, split_map }
    }

    pub fn from_file(path: &str) -> Result<Self> {
        let inner = Tokenizer::from_file(path).map_err(anyhow::Error::msg).with_context(|| format!("load tokenizer from {path}"))?;
        Ok(Self::new(inner))
    }

    fn build_split_map(tok: &Tokenizer) -> HashMap<u32, Vec<u32>> {
        let vocab = tok.get_vocab(true);

        // Step 1: raw (unstripped) vocab strings that already qualify.
        let multichar_tokens: HashSet<&str> =
            vocab.keys().filter(|t| is_multichar_cjk(t)).map(String::as_str).collect();

        // Step 2: every vocab entry, stripped, checked against that set.
        let mut split_map = HashMap::new();
        for (token, &id) in &vocab {
            let clean: String = token.chars().filter(|&c| c != SP_MARKER).collect();
            if !multichar_tokens.contains(clean.as_str()) {
                continue;
            }
            let expansion: Vec<u32> = clean
                .chars()
                .map(|c| {
                    let mut buf = [0u8; 4];
                    let s = c.encode_utf8(&mut buf);
                    tok.token_to_id(s).or_else(|| tok.token_to_id("<unk>")).unwrap_or(id)
                })
                .collect();
            split_map.insert(id, expansion);
        }
        split_map
    }

    /// Tokenize `text`, then expand any multi-char CJK token id into its
    /// per-character token ids (`add_special_tokens=false`, matching how
    /// `voxcpm2.py` calls the wrapped tokenizer — it appends control tokens
    /// like `audio_start_token` itself afterward).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, false).map_err(anyhow::Error::msg)?;
        let ids = encoding.get_ids();
        let mut out = Vec::with_capacity(ids.len());
        for &id in ids {
            match self.split_map.get(&id) {
                Some(expansion) => out.extend_from_slice(expansion),
                None => out.push(id),
            }
        }
        Ok(out)
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner.decode(ids, skip_special_tokens).map_err(anyhow::Error::msg)
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_multichar_cjk_requires_two_plus_cjk_chars() {
        assert!(is_multichar_cjk("你好"));
        assert!(!is_multichar_cjk("你")); // single char
        assert!(!is_multichar_cjk("")); // empty
        assert!(!is_multichar_cjk("ab")); // not CJK
        assert!(!is_multichar_cjk("你a")); // mixed
    }

    #[test]
    fn is_multichar_cjk_rejects_marker_prefixed() {
        // The marker itself isn't CJK, so a raw (unstripped) marked token
        // never qualifies directly — matches step 1 of the real algorithm.
        assert!(!is_multichar_cjk("\u{2581}你好"));
    }
}
