// SPDX-License-Identifier: MIT
//! Grammar-constrained decoding: a per-sequence hook that can mask the
//! sampler's logits before a token is chosen, and observe the token that
//! was actually sampled to advance its own state.
//!
//! Scoped today to [`tool_call_skeleton::ToolCallSkeleton`], a hand-rolled
//! constraint for Qwen3-Coder's tool-call XML wrapper (see that module's
//! doc comment for why). The [`GrammarConstraint`] trait is the seam for a
//! more general grammar/PEG engine later: neither `sampling.rs` nor
//! [`super::sequence::Sequence`] need to change to support a different
//! implementation, only the constructor call in `accept_request`.

pub mod tool_call_skeleton;

use candle_core::Tensor;

/// What the sampler is allowed to produce for the next token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenMask {
    /// No constraint. The zero-cost fast path used before a grammar has
    /// triggered and after it has finished.
    Unconstrained,
    /// Only these token ids may be sampled next; every other token's logit
    /// is set to an effectively-`-inf` sentinel before sampling. Does not
    /// include EOS ids even when [`GrammarConstraint::allows_eos`] is
    /// true — the sampler reconciles the two independently, since a
    /// grammar has no reason to know a request's EOS token ids.
    AllowOnly(Vec<u32>),
}

/// A per-sequence constraint on what the sampler may produce next.
///
/// Implementations are stateful: [`advance`](GrammarConstraint::advance)
/// is called once per decode step with the token that was actually
/// sampled, and [`token_mask`](GrammarConstraint::token_mask) is queried
/// before the *next* step's sampling to constrain it.
pub trait GrammarConstraint: Send {
    /// Advance state with the token just sampled and the text it decoded
    /// to (empty if the token produced no new safe text yet, e.g. a
    /// partial UTF-8 sequence).
    fn advance(&mut self, token_id: u32, decoded_text: &str);

    /// The constraint for the token about to be sampled.
    fn token_mask(&self) -> TokenMask;

    /// Whether an EOS token is currently permitted. Checked independently
    /// of [`token_mask`](GrammarConstraint::token_mask) since EOS ids are
    /// request state the grammar itself does not hold.
    fn allows_eos(&self) -> bool;

    /// Once true, the engine drops the grammar (`Sequence::grammar =
    /// None`) so later decode steps pay no per-token overhead.
    fn is_finished(&self) -> bool;
}

/// Token-id → decoded-text lookup for the whole vocabulary, built once per
/// model load and shared (read-only) across every sequence's grammar.
///
/// Needed because forcing a literal (e.g. `</tool_call>`) is a *token-id*
/// operation — masking logits — even though trigger/tag detection
/// elsewhere in the grammar operates on decoded strings.
pub struct VocabByteTable {
    /// Indexed by token id. Empty string for ids with no decodable text
    /// (e.g. some special/control tokens).
    entries: Vec<String>,
}

impl VocabByteTable {
    /// Build the table from `tokenizer`'s full vocabulary.
    #[must_use]
    pub fn build(tokenizer: &tokenizers::Tokenizer) -> Self {
        let vocab_size = tokenizer.get_vocab_size(true);
        let mut entries = vec![String::new(); vocab_size];
        for (id, entry) in entries.iter_mut().enumerate() {
            // Vocab sizes are far below u32::MAX.
            #[allow(clippy::cast_possible_truncation)]
            let id_u32 = id as u32;
            if let Ok(text) = tokenizer.decode(&[id_u32], false) {
                *entry = text;
            }
        }
        Self { entries }
    }

    /// Token ids whose decoded text is a non-empty exact match for a
    /// prefix of `target[offset..]` — i.e. tokens that could validly
    /// extend the forced literal starting at `offset` without overshooting
    /// it.
    ///
    /// Known limitation: a vocab token whose bytes would straddle the end
    /// of `target` (partly the forced literal, partly whatever legally
    /// follows it) is never matched, since "whatever follows" isn't known
    /// here. This can force slightly less efficient tokenization during a
    /// forced span; a future trie-based grammar engine would handle the
    /// general case. If this ever returns empty, [`apply_grammar_mask`]
    /// degrades to leaving the step unconstrained rather than deadlocking.
    #[must_use]
    pub fn tokens_matching_prefix(&self, target: &str, offset: usize) -> Vec<u32> {
        let Some(remaining) = target.get(offset..) else {
            return Vec::new();
        };
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, text)| !text.is_empty() && remaining.starts_with(text.as_str()))
            .map(|(id, _)| {
                // Vocab sizes are far below u32::MAX.
                #[allow(clippy::cast_possible_truncation)]
                let id = id as u32;
                id
            })
            .collect()
    }
}

/// Apply a grammar's [`TokenMask`] to `logits` (already squeezed to shape
/// `[vocab]`), returning the tensor to sample from.
///
/// `Unconstrained` is a no-op returning `logits` unchanged (cheap clone —
/// candle tensors are refcounted handles, not copies). `AllowOnly` builds
/// a fresh `-1e9`-filled tensor and scatters the allowed ids' original
/// logits back in — cheap since the allow-list is always a tiny handful
/// of tokens (forcing a short literal), the same gather/scatter idiom as
/// [`super::sampling::apply_penalties_inplace`], just inverted (keep a
/// few, blank the rest, instead of modify a few, keep the rest).
///
/// An empty allow-list is treated as "nothing to force" and leaves
/// `logits` unconstrained, rather than erroring mid-generation — see
/// [`VocabByteTable::tokens_matching_prefix`]'s doc for when this can
/// happen.
///
/// # Errors
/// Returns an error if a tensor operation fails.
pub fn apply_grammar_mask(logits: &Tensor, mask: &TokenMask) -> candle_core::Result<Tensor> {
    let TokenMask::AllowOnly(allowed_ids) = mask else {
        return Ok(logits.clone());
    };
    if allowed_ids.is_empty() {
        return Ok(logits.clone());
    }
    let vocab = logits.dim(0)?;
    let idx = Tensor::new(allowed_ids.as_slice(), logits.device())?;
    let saved = logits.gather(&idx, candle_core::D::Minus1)?;
    let masked = Tensor::full(-1e9f32, vocab, logits.device())?;
    masked.scatter_set(&idx, &saved, candle_core::D::Minus1)?;
    Ok(masked)
}

/// Suppress EOS tokens in `logits` in-place by setting their logits to an
/// effectively-`-inf` sentinel. Used when a grammar's
/// [`GrammarConstraint::allows_eos`] returns `false`.
///
/// # Errors
/// Returns an error if a tensor operation fails.
pub fn suppress_eos_inplace(logits: &Tensor, eos_token_id: &[u32]) -> candle_core::Result<()> {
    if eos_token_id.is_empty() {
        return Ok(());
    }
    let idx = Tensor::new(eos_token_id, logits.device())?;
    let neg = Tensor::full(-1e9f32, eos_token_id.len(), logits.device())?;
    logits.scatter_set(&idx, &neg, candle_core::D::Minus1)
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    fn logits(values: &[f32]) -> Tensor {
        Tensor::new(values, &Device::Cpu).unwrap()
    }

    #[test]
    // Unconstrained must never touch the logits.
    fn unconstrained_mask_is_noop() {
        let t = logits(&[1.0, 2.0, 3.0]);
        let out = apply_grammar_mask(&t, &TokenMask::Unconstrained).unwrap();
        assert_eq!(out.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    // AllowOnly must preserve the listed ids' original logits and blank
    // every other id.
    fn allow_only_keeps_listed_tokens_blanks_rest() {
        let t = logits(&[1.0, 2.0, 3.0, 4.0]);
        let out = apply_grammar_mask(&t, &TokenMask::AllowOnly(vec![1, 3])).unwrap();
        let out = out.to_vec1::<f32>().unwrap();
        assert_eq!(out[1], 2.0);
        assert_eq!(out[3], 4.0);
        assert!(out[0] < -1e8);
        assert!(out[2] < -1e8);
    }

    #[test]
    // An empty allow-list must degrade to unconstrained, not error.
    fn allow_only_empty_list_is_noop() {
        let t = logits(&[1.0, 2.0, 3.0]);
        let out = apply_grammar_mask(&t, &TokenMask::AllowOnly(vec![])).unwrap();
        assert_eq!(out.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    // Only the given ids get suppressed; everything else is untouched.
    fn suppress_eos_blanks_only_eos_ids() {
        let t = logits(&[1.0, 2.0, 3.0]);
        suppress_eos_inplace(&t, &[1]).unwrap();
        let out = t.to_vec1::<f32>().unwrap();
        assert_eq!(out[0], 1.0);
        assert!(out[1] < -1e8);
        assert_eq!(out[2], 3.0);
    }

    #[test]
    // No eos ids configured must leave logits untouched.
    fn suppress_eos_noop_when_empty() {
        let t = logits(&[1.0, 2.0, 3.0]);
        suppress_eos_inplace(&t, &[]).unwrap();
        assert_eq!(t.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    /// A small synthetic vocab: 0="<", 1="tool", 2="_call", 3=">", 4="\n",
    /// 5="foo". Mirrors what a real BPE vocab looks like closely enough to
    /// exercise prefix matching without needing a real tokenizer.
    fn mock_vocab_table() -> VocabByteTable {
        VocabByteTable {
            entries: vec![
                "<".to_string(),
                "tool".to_string(),
                "_call".to_string(),
                ">".to_string(),
                "\n".to_string(),
                "foo".to_string(),
            ],
        }
    }

    #[test]
    // At offset 0, only the token spelling exactly "<" starts the literal.
    fn tokens_matching_prefix_finds_valid_continuations() {
        let table = mock_vocab_table();
        let mut ids = table.tokens_matching_prefix("<tool_call>", 0);
        ids.sort_unstable();
        assert_eq!(ids, vec![0]);
    }

    #[test]
    // From offset 1 onward, "tool" is the next valid continuation.
    fn tokens_matching_prefix_respects_offset() {
        let table = mock_vocab_table();
        let mut ids = table.tokens_matching_prefix("<tool_call>", 1);
        ids.sort_unstable();
        assert_eq!(ids, vec![1]);
    }

    #[test]
    // A token unrelated to the target literal must never be included.
    fn tokens_matching_prefix_excludes_non_matching() {
        let table = mock_vocab_table();
        let ids = table.tokens_matching_prefix("<tool_call>", 0);
        assert!(!ids.contains(&5));
    }

    #[test]
    // An offset at or past the target's end has nothing left to match.
    fn tokens_matching_prefix_empty_past_end() {
        let table = mock_vocab_table();
        assert!(table.tokens_matching_prefix("<", 1).is_empty());
    }

    #[test]
    // Regression check against a *real* tokenizer, not the synthetic mock
    // above: every literal `tool_call_skeleton::ToolCallSkeleton` forces
    // must have at least one vocab token able to start it, or the grammar
    // silently degrades to unconstrained right when it matters most (see
    // `apply_grammar_mask`'s doc). Set `CRANE_GRAMMAR_TEST_GGUF` to a GGUF
    // model path to run; skipped otherwise since CI has no model file.
    #[ignore]
    fn real_tokenizer_can_start_every_forced_literal() {
        let Ok(path) = std::env::var("CRANE_GRAMMAR_TEST_GGUF") else {
            eprintln!("skipping: CRANE_GRAMMAR_TEST_GGUF not set");
            return;
        };
        let auto = crane_core::autotokenizer::AutoTokenizer::from_gguf(&path)
            .expect("failed to load tokenizer from GGUF");
        let table = VocabByteTable::build(&auto.tokenizer);

        for literal in [
            "\n<function=",   // Idle -> AwaitingFunctionPrefix (after "<tool_call>")
            "\n</tool_call>", // InCallBody -> AwaitingCallClose (after "</function>")
        ] {
            let ids = table.tokens_matching_prefix(literal, 0);
            println!("{literal:?}: {} candidate starting token(s)", ids.len());
            assert!(
                !ids.is_empty(),
                "no vocab token can start forcing {literal:?} — the grammar can never \
                 constrain this transition and will silently no-op instead",
            );
        }
    }
}
