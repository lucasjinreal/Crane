// SPDX-License-Identifier: MIT
//! Grammar constraint for Qwen3-Coder's tool-call XML skeleton:
//!
//! ```text
//! <tool_call>
//! <function=example_function_name>
//! <parameter=example_parameter_1>
//! value_1
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! Qwen3-Coder is documented to sometimes omit the opening `<tool_call>`
//! tag. `crate::tools::parse_output` already recovers from this after the
//! fact; this module instead prevents it at generation time.
//!
//! Trigger detection (deciding when to start constraining at all) is
//! string-level, reusing the same incremental-match style as
//! `crate::reasoning`'s `partial_tag_suffix_len` and `crate::tools`'s
//! `next_tag_pos`. Forcing a specific literal once triggered is
//! token-level, via [`super::VocabByteTable`], since masking logits is
//! inherently a token-id operation.
//!
//! The `<tool_call>` wrapper's leniency applies to *every* call, not just
//! the first: forcing it for calls after the first was tried and measured
//! to backfire. Once the model's preferred (bare) continuation is masked
//! out, the freed probability mass does not reliably flow to the forced
//! wrapper — it can just as easily concentrate on EOS instead, since the
//! grammar only forbids illegal tokens, it doesn't boost legal ones. In a
//! live trace this was decisive, not marginal: EOS logit 21.2 vs. 15.9 for
//! the best compliant continuation, collapsing a 4-call batch to 1 call in
//! every trial. llama.cpp's own grammar has the identical structural
//! exposure (its `zero_or_more(tool_call)` enforces no minimum beyond the
//! first call either) and no mitigation for it at the grammar, sampler, or
//! server level. Only the closing `</tool_call>` is force-guaranteed for
//! every call — that carries no such risk, since EOS is never a
//! competing option while it's being forced.

use super::GrammarConstraint;
use super::TokenMask;
use super::VocabByteTable;
use std::sync::Arc;

/// Longest buffer of un-matched text kept in [`State::Idle`] before
/// trimming, bounding memory for long content-only (no tool call)
/// responses. Comfortably longer than any realistic tool name.
const IDLE_BUF_CAP: usize = 512;

/// A literal string currently being forced, tracked by how many bytes of
/// it have already been consumed by accepted tokens.
#[derive(Debug, Clone)]
struct ForcedLiteral {
    target: String,
    consumed: usize,
}

impl ForcedLiteral {
    fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
            consumed: 0,
        }
    }

    fn remaining(&self) -> &str {
        &self.target[self.consumed..]
    }

    fn is_complete(&self) -> bool {
        self.consumed >= self.target.len()
    }

    /// Advance by `text` if it's consistent with what remains to force.
    /// Returns `false` if `text` diverges from the forced literal — this
    /// should never happen if the sampler respected `token_mask()`, but
    /// is checked defensively rather than assumed.
    fn advance(&mut self, text: &str) -> bool {
        if text.is_empty() {
            return true;
        }
        if self.remaining().starts_with(text) {
            self.consumed += text.len();
            true
        } else {
            false
        }
    }
}

/// States of the tool-call skeleton grammar. See the module doc for the
/// overall shape.
#[derive(Debug, Clone)]
enum State {
    /// No trigger seen yet. Unconstrained; EOS always allowed.
    Idle { buf: String },
    /// A `"<tool_call>"` trigger was seen; forcing the `"<function="`
    /// prefix that must follow it (any call, not just the first).
    AwaitingFunctionPrefix(ForcedLiteral),
    /// The `"<function="` prefix is done; forcing the rest of a name that
    /// is a prefix of at least one of the request's tool names, followed
    /// by `'>'`. Holds the partial name text matched so far.
    AwaitingFunctionName(String),
    /// Inside a call's parameter body — unconstrained, watching for
    /// `"</function>"`.
    InCallBody { buf: String },
    /// `"</function>"` was just seen; forcing `"\n</tool_call>"`.
    AwaitingCallClose(ForcedLiteral),
    /// Terminal fallback, reached defensively if the sampler ever
    /// produced text inconsistent with a forced literal (should not
    /// happen if the mask was applied, but never allowed to wedge a
    /// sequence). Fully unconstrained from here on.
    Finished,
}

/// Grammar constraint enforcing Qwen3-Coder's tool-call XML skeleton for
/// one in-flight sequence. See the module doc for the exact rules.
pub struct ToolCallSkeleton {
    tool_names: Vec<String>,
    /// `"<function={name}>"` for each of `tool_names`, precomputed once so
    /// `step`'s `Idle` arm (run on essentially every decoded token) doesn't
    /// re-`format!` them per call.
    triggers: Vec<String>,
    vocab: Arc<VocabByteTable>,
    state: State,
}

impl ToolCallSkeleton {
    /// Build a fresh, untriggered grammar for a request offering
    /// `tool_names`. `vocab` is shared (via `Arc`) across every sequence,
    /// built once at engine startup.
    #[must_use]
    pub fn new(tool_names: Vec<String>, vocab: Arc<VocabByteTable>) -> Self {
        let triggers = tool_names
            .iter()
            .map(|name| format!("<function={name}>"))
            .collect();
        Self {
            tool_names,
            triggers,
            vocab,
            state: State::Idle { buf: String::new() },
        }
    }

    /// Token ids that could extend `partial` toward a complete
    /// `"NAME>"` for some tool name starting with `partial`, unioned
    /// across every such candidate name.
    fn function_name_mask(&self, partial: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = self
            .tool_names
            .iter()
            .filter(|name| name.starts_with(partial) || partial.starts_with(name.as_str()))
            .flat_map(|name| {
                let target = format!("{name}>");
                self.vocab.tokens_matching_prefix(&target, partial.len())
            })
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Whether `partial` (with a trailing `'>'`) completes exactly one of
    /// the request's tool names.
    fn completes_a_name(partial: &str) -> Option<&str> {
        partial.strip_suffix('>')
    }

    /// Drop `buf`'s leading bytes if it's grown past [`IDLE_BUF_CAP`],
    /// landing on a char boundary. Used by the two unconstrained,
    /// substring-watching states (`Idle`, `InCallBody`) to bound memory on
    /// long spans of text with no trigger/closer in sight.
    fn cap_buf(buf: &mut String) {
        if buf.len() <= IDLE_BUF_CAP {
            return;
        }
        let cut = buf.len() - IDLE_BUF_CAP;
        let cut = (cut..buf.len())
            .find(|&i| buf.is_char_boundary(i))
            .unwrap_or(buf.len());
        buf.drain(..cut);
    }

    /// Apply `text` to `state`, one step of the state machine.
    ///
    /// Returns the resulting state and, when a match/literal completed
    /// mid-token, any trailing bytes of `text` that came *after* the
    /// completion point — e.g. a single token decoding to `">\n"` when
    /// only the `>` was needed to complete `</function>`. That overflow
    /// belongs to the *new* state, not the one that just finished, so it
    /// is handed back rather than silently dropped; `advance` re-feeds it
    /// through `step` again instead of losing it.
    ///
    /// Only `Idle` and `InCallBody` can produce overflow: they detect a
    /// trigger/closer via unconstrained substring search, so a single
    /// token can legally contain bytes beyond it. The `ForcedLiteral`
    /// states never can, since their token mask only ever allows a token
    /// whose full text fits within what remains to be forced.
    fn step(&self, state: State, text: &str) -> (State, Option<String>) {
        match state {
            State::Idle { mut buf } => {
                buf.push_str(text);
                if let Some(pos) = buf.find("<tool_call>") {
                    let rest = buf[pos + "<tool_call>".len()..].to_string();
                    // The template places a newline between "<tool_call>"
                    // and "<function=".
                    (
                        State::AwaitingFunctionPrefix(ForcedLiteral::new("\n<function=")),
                        (!rest.is_empty()).then_some(rest),
                    )
                } else if let Some((pos, trigger_len)) = self
                    .triggers
                    .iter()
                    .filter_map(|trigger| {
                        buf.find(trigger.as_str()).map(|pos| (pos, trigger.len()))
                    })
                    .min_by_key(|(pos, _)| *pos)
                {
                    let rest = buf[pos + trigger_len..].to_string();
                    (
                        State::InCallBody { buf: String::new() },
                        (!rest.is_empty()).then_some(rest),
                    )
                } else {
                    Self::cap_buf(&mut buf);
                    (State::Idle { buf }, None)
                }
            },
            State::AwaitingFunctionPrefix(mut lit) => {
                if !lit.advance(text) {
                    (State::Finished, None)
                } else if lit.is_complete() {
                    (State::AwaitingFunctionName(String::new()), None)
                } else {
                    (State::AwaitingFunctionPrefix(lit), None)
                }
            },
            State::AwaitingFunctionName(mut partial) => {
                partial.push_str(text);
                if let Some(name) = Self::completes_a_name(&partial) {
                    if self.tool_names.iter().any(|n| n == name) {
                        (State::InCallBody { buf: String::new() }, None)
                    } else {
                        // The sampler chose a `>` that doesn't complete any
                        // offered tool name — only possible on a mask
                        // desync. Stop constraining rather than wedge.
                        (State::Finished, None)
                    }
                } else if self
                    .tool_names
                    .iter()
                    .any(|n| n.starts_with(&partial) || partial.starts_with(n.as_str()))
                {
                    (State::AwaitingFunctionName(partial), None)
                } else {
                    (State::Finished, None)
                }
            },
            State::InCallBody { mut buf } => {
                buf.push_str(text);
                if let Some(pos) = buf.find("</function>") {
                    let rest = buf[pos + "</function>".len()..].to_string();
                    (
                        State::AwaitingCallClose(ForcedLiteral::new("\n</tool_call>")),
                        (!rest.is_empty()).then_some(rest),
                    )
                } else {
                    Self::cap_buf(&mut buf);
                    (State::InCallBody { buf }, None)
                }
            },
            State::AwaitingCallClose(mut lit) => {
                if !lit.advance(text) {
                    (State::Finished, None)
                } else if lit.is_complete() {
                    // The call is fully closed. Go back to `Idle` rather
                    // than forcing a fresh wrapper — see the module doc for
                    // why forcing it here backfires.
                    (State::Idle { buf: String::new() }, None)
                } else {
                    (State::AwaitingCallClose(lit), None)
                }
            },
            State::Finished => (State::Finished, None),
        }
    }
}

impl GrammarConstraint for ToolCallSkeleton {
    fn advance(&mut self, _token_id: u32, decoded_text: &str) {
        if decoded_text.is_empty() {
            return;
        }
        let mut state = std::mem::replace(&mut self.state, State::Finished);
        let mut pending = Some(decoded_text.to_string());
        while let Some(text) = pending.take() {
            let (next_state, leftover) = self.step(state, &text);
            state = next_state;
            if matches!(state, State::Finished) {
                break;
            }
            pending = leftover;
        }
        self.state = state;
    }

    fn token_mask(&self) -> TokenMask {
        match &self.state {
            State::Idle { .. } | State::InCallBody { .. } | State::Finished => {
                TokenMask::Unconstrained
            },
            State::AwaitingFunctionPrefix(lit) | State::AwaitingCallClose(lit) => {
                TokenMask::AllowOnly(self.vocab.tokens_matching_prefix(&lit.target, lit.consumed))
            },
            State::AwaitingFunctionName(partial) => {
                TokenMask::AllowOnly(self.function_name_mask(partial))
            },
        }
    }

    fn allows_eos(&self) -> bool {
        matches!(self.state, State::Idle { .. } | State::Finished)
    }

    fn is_finished(&self) -> bool {
        matches!(self.state, State::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> Arc<VocabByteTable> {
        // A tiny synthetic vocab sufficient to spell out every literal
        // this grammar forces, plus two tool names ("glob", "read") and
        // an unrelated content token ("foo"). `entries` is private to
        // `grammar`, but visible here since this is a descendant module.
        let strings = [
            "<",
            "tool",
            "_call",
            ">",
            "\n",
            "<function=",
            "glob",
            "read",
            "</function",
            "</",
            "foo",
        ];
        Arc::new(VocabByteTable {
            entries: strings.iter().map(|s| (*s).to_string()).collect(),
        })
    }

    fn names() -> Vec<String> {
        vec!["glob".to_string(), "read".to_string()]
    }

    fn feed(g: &mut ToolCallSkeleton, text: &str) {
        // Token id is irrelevant to this grammar's own logic (only
        // `decoded_text` drives state); use 0 throughout.
        g.advance(0, text);
    }

    #[test]
    // Before any trigger, everything is unconstrained and EOS is fine.
    fn idle_is_unconstrained() {
        let g = ToolCallSkeleton::new(names(), table());
        assert_eq!(g.token_mask(), TokenMask::Unconstrained);
        assert!(g.allows_eos());
        assert!(!g.is_finished());
    }

    #[test]
    // Plain content that never mentions a tool call must never trigger.
    fn idle_pass_through_never_triggers() {
        let mut g = ToolCallSkeleton::new(names(), table());
        feed(&mut g, "just some ordinary prose, nothing tool-shaped here");
        assert_eq!(g.token_mask(), TokenMask::Unconstrained);
        assert!(g.allows_eos());
    }

    #[test]
    // A "<tool_call>" trigger must force the "<function=" prefix next.
    fn tool_call_trigger_forces_function_prefix() {
        let mut g = ToolCallSkeleton::new(names(), table());
        feed(&mut g, "<tool_call>");
        assert!(!g.allows_eos());
        match g.token_mask() {
            TokenMask::AllowOnly(ids) => assert!(!ids.is_empty()),
            TokenMask::Unconstrained => panic!("expected a forced prefix"),
        }
    }

    #[test]
    // A bare "<function=glob>" (no "<tool_call>" wrapper) must trigger
    // straight into the call body — the documented first-call leniency.
    fn bare_function_trigger_enters_call_body_directly() {
        let mut g = ToolCallSkeleton::new(names(), table());
        feed(&mut g, "<function=glob>");
        assert_eq!(g.token_mask(), TokenMask::Unconstrained);
        assert!(!g.allows_eos(), "must not allow ending mid-call");
    }

    #[test]
    // Once inside a call body, "</function>" must force the closer.
    fn function_close_forces_tool_call_close() {
        let mut g = ToolCallSkeleton::new(names(), table());
        feed(&mut g, "<function=glob>");
        feed(
            &mut g,
            "<parameter=pattern>\n*.rs\n</parameter>\n</function>",
        );
        match g.token_mask() {
            TokenMask::AllowOnly(ids) => assert!(!ids.is_empty()),
            TokenMask::Unconstrained => panic!("expected the closer to be forced"),
        }
        assert!(!g.allows_eos());
    }

    #[test]
    // A single token whose text both completes "</function>" and carries
    // a trailing byte (e.g. ">\n" — the exact pattern seen against the
    // real model) must not lose that trailing byte: it has to count
    // toward the forced "\n</tool_call>" closer, not be silently dropped.
    fn overflow_byte_past_function_close_is_not_lost() {
        let mut g = ToolCallSkeleton::new(names(), table());
        feed(&mut g, "<function=glob>");
        // One token combining the "</function>" match with an extra "\n".
        feed(&mut g, "</function>\n");
        // If the "\n" were lost, "</tool_call>" alone would still be
        // missing its leading newline and fail to complete the literal.
        feed(&mut g, "</tool_call>");
        assert!(
            !g.is_finished(),
            "leftover byte must not desync the grammar"
        );
        assert!(g.allows_eos());
        assert_eq!(g.token_mask(), TokenMask::Unconstrained);
    }

    #[test]
    // Same overflow hazard at the "<tool_call>" trigger: a token
    // combining it with its own trailing newline must feed that newline
    // into the forced "\n<function=" prefix, not drop it.
    fn overflow_bytes_past_tool_call_trigger_are_not_lost() {
        let mut g = ToolCallSkeleton::new(names(), table());
        feed(&mut g, "<tool_call>\n");
        // If the "\n" were lost, this would need to start with "\n" again.
        feed(&mut g, "<function=");
        assert!(!g.is_finished());
        match g.token_mask() {
            TokenMask::AllowOnly(ids) => assert!(!ids.is_empty()),
            TokenMask::Unconstrained => panic!("expected still forcing the function name"),
        }
    }

    #[test]
    // A single token spelling the entire bare opener plus everything up
    // to and including "</function>" must correctly propagate all of it
    // through Idle and InCallBody in one `advance` call, ending up with
    // the closer forced — not stuck in an earlier state.
    fn overflow_propagates_through_multiple_states_in_one_advance() {
        let mut g = ToolCallSkeleton::new(names(), table());
        feed(
            &mut g,
            "<function=glob>\n<parameter=pattern>\n*.rs\n</parameter>\n</function>",
        );
        assert!(!g.is_finished());
        match g.token_mask() {
            TokenMask::AllowOnly(ids) => assert!(!ids.is_empty()),
            TokenMask::Unconstrained => panic!("expected the closer to be forced"),
        }
    }

    #[test]
    // After a call fully closes, the grammar goes back to `Idle`: EOS is
    // allowed (turn may legitimately end there) and nothing is forced —
    // forcing a fresh wrapper here was tried and measured to bias the
    // model toward stopping instead of continuing (see module doc).
    fn after_call_closes_returns_to_idle() {
        let mut g = ToolCallSkeleton::new(names(), table());
        feed(&mut g, "<function=glob>");
        feed(&mut g, "</function>");
        feed(&mut g, "\n</tool_call>");
        assert!(g.allows_eos());
        assert_eq!(g.token_mask(), TokenMask::Unconstrained);
    }

    #[test]
    // A second call gets the exact same bare-opener leniency as the
    // first: no wrapper forced, straight into the call body once its
    // "<function=NAME>" is seen.
    fn subsequent_call_gets_same_leniency_as_first() {
        let mut g = ToolCallSkeleton::new(names(), table());
        feed(&mut g, "<function=glob>");
        feed(&mut g, "</function>");
        feed(&mut g, "\n</tool_call>");
        feed(&mut g, "<function=read>");
        assert_eq!(g.token_mask(), TokenMask::Unconstrained);
        assert!(!g.allows_eos(), "must not allow ending mid-call");
    }

    #[test]
    // A name that matches no offered tool must never appear in the mask.
    fn unknown_function_name_never_allowed() {
        let g = ToolCallSkeleton::new(names(), table());
        let mask = g.function_name_mask("nonexistent");
        assert!(mask.is_empty());
    }

    #[test]
    // Full happy path end to end, chunked the way real tokens would
    // arrive — each chunk stays within one state's forced boundary,
    // mirroring how the sampler's mask would only ever allow a token
    // whose whole text fits within what's currently being forced. Must
    // never fall back to `Finished` (that's the desync escape hatch).
    fn full_single_call_never_desyncs() {
        let mut g = ToolCallSkeleton::new(names(), table());
        for chunk in [
            "<tool_call>", // Idle -> AwaitingFunctionPrefix("\n<function=")
            "\n",          // consumes the leading newline
            "<function=",  // completes the prefix -> AwaitingFunctionName("")
            "glob",        // partial name "glob"
            ">",           // completes "glob>" -> InCallBody
            "\n<parameter=pattern>\n*.rs\n</parameter>\n</function>", // free text, ends the call
            "\n",          // AwaitingCallClose("\n</tool_call>") consumes '\n'
            "</tool_call>", // completes the closer -> back to Idle
        ] {
            feed(&mut g, chunk);
            assert!(!g.is_finished(), "desynced after feeding {chunk:?}");
        }
        assert!(g.allows_eos());
    }
}
