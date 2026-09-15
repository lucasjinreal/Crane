//! Tool (function) calling: parsing what the model emits back into OpenAI shape.
//!
//! The request side needs no code — `tools` is handed to the Jinja chat
//! template verbatim (`tool | tojson`), and the template owns the prompt
//! format. The response side does: the model answers in the template's own
//! syntax, and clients expect `message.tool_calls`.
//!
//! Qwen 3.5/3.6/3.8 and Ornith all specify the same XML-ish grammar, quoted
//! from the Qwen 3.8 template itself:
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
//! Note this is *not* the `{"name": …, "arguments": …}` JSON that older Qwen
//! releases used — parameter values are raw text, newline-delimited, and may
//! span multiple lines.
//!
//! The template also permits prose before a call ("You may provide optional
//! reasoning for your function call in natural language BEFORE the function
//! call, but NOT after"), so surrounding text is preserved as `content`.

use serde_json::{Map, Value};

use crate::openai_api::{FunctionCall, ToolCall};

const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";
const FN_OPEN: &str = "<function=";
const FN_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";

/// Which tag `next_tag_pos` matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TagKind {
    /// `CALL_OPEN` — the standard, well-formed opener.
    Wrapped,
    /// `FN_OPEN` with no `CALL_OPEN` before it.
    Bare,
}

/// Position and kind of whichever tag that could start a tool call comes
/// first in `s`: a well-formed `CALL_OPEN`, or a bare `FN_OPEN` with no
/// `CALL_OPEN` before it. Shared by `parse_output` and `ToolCallStream::push`
/// so the two scans can't drift apart. `CALL_OPEN` and `FN_OPEN` diverge at
/// their second byte (`t` vs `f`), so two matches can never start at the
/// same offset.
fn next_tag_pos(s: &str) -> Option<(TagKind, usize)> {
    let call_pos = s.find(CALL_OPEN);
    let fn_pos = s.find(FN_OPEN);
    match (call_pos, fn_pos) {
        (None, None) => None,
        (Some(cp), None) => Some((TagKind::Wrapped, cp)),
        (None, Some(fp)) => Some((TagKind::Bare, fp)),
        (Some(cp), Some(fp)) if cp < fp => Some((TagKind::Wrapped, cp)),
        (Some(_), Some(fp)) => Some((TagKind::Bare, fp)),
    }
}

/// Everything the model produced, split into prose and calls.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ParsedOutput {
    /// Text outside any `<tool_call>` block, trimmed.
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

impl ParsedOutput {
    pub fn has_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

/// Split `text` into prose and tool calls.
///
/// An unterminated `<tool_call>` is dropped rather than guessed at — that
/// happens when generation hits the token limit mid-call, and a half-parsed
/// call would be executed by the client as if it were complete.
///
/// Also recovers a `<function=...></function>` block that is missing its
/// `<tool_call>` opener — observed in the field where a model emits several
/// chained calls and drops the opening tag on one of them while still
/// closing it. Without this, `</tool_call>` does not contain the substring
/// `<tool_call>` (it has an extra `/`), so the scan for the next opener
/// skips straight past the whole malformed call, including any well-formed
/// calls that were meant to follow inside it, and dumps it all into
/// `content` as literal XML. A well-formed block is unaffected: `CALL_OPEN`
/// always precedes its own `FN_OPEN` within the same block, so this only
/// takes the fallback path for a `<function=` with no `<tool_call>` before it.
/// The fallback also checks that the `</function>` it finds isn't actually
/// the closer of a *later* call — otherwise a stray `<function=` mentioned
/// in prose before a real call would steal that call's closing tag and
/// misattribute its parameters to the wrong function name.
///
/// Unlike the wrapped case, a bare `<function=` with no `</function>`
/// anywhere in the rest of the text is *not* dropped: a coding assistant can
/// legitimately mention the literal string `<function=` in prose with no
/// closing tag ever following, and there is no way to tell that apart from a
/// truncated call. Silently discarding the rest of the message on every such
/// mention would lose real answer text, so the leading `<` is re-emitted as
/// literal content and scanning continues instead.
pub fn parse_output(text: &str) -> ParsedOutput {
    let mut content = String::new();
    let mut tool_calls = Vec::new();
    let mut rest = text;

    loop {
        let (kind, pos) = match next_tag_pos(rest) {
            Some(x) => x,
            None => break,
        };
        content.push_str(&rest[..pos]);

        if kind == TagKind::Bare {
            // A bare `<function=...>` with no `<tool_call>` before it.
            let from_fn = &rest[pos..];
            let real_close = from_fn.find(FN_CLOSE).filter(|&end| {
                // Reject a `</function>` that belongs to a later call: if
                // another opener appears before it, this `<function=` was
                // just mentioned in prose ahead of a real one.
                let between = &from_fn[FN_OPEN.len()..end];
                !between.contains(CALL_OPEN) && !between.contains(FN_OPEN)
            });
            match real_close {
                Some(end) => {
                    let block_end = end + FN_CLOSE.len();
                    if let Some(call) = parse_call(&from_fn[..block_end], tool_calls.len()) {
                        tool_calls.push(call);
                    }
                    let mut remainder = &from_fn[block_end..];
                    // The dropped opener can leave one or more closers
                    // behind; swallow them (and any whitespace before each)
                    // so they don't leak into content. `while`, not `if`: a
                    // stuttering model can emit more than one closer for the
                    // same dropped opener. This can in principle eat literal
                    // prose that happens to start with `</tool_call>` right
                    // after a bare call, but that's far less likely than a
                    // model repeating its own closer.
                    while let Some(after) = remainder.trim_start().strip_prefix(CALL_CLOSE) {
                        remainder = after;
                    }
                    rest = remainder;
                },
                None => {
                    // No closer anywhere in the rest of the text (or the one
                    // found belongs to a later call): this `<function=` was
                    // never a real call (e.g. the model mentioned the tag in
                    // prose). Re-emit the leading `<` as literal content and
                    // keep scanning past it, rather than discarding
                    // everything that follows.
                    content.push('<');
                    rest = &rest[pos + 1..];
                },
            }
        } else {
            let after = &rest[pos + CALL_OPEN.len()..];
            match after.find(CALL_CLOSE) {
                Some(end) => {
                    if let Some(call) = parse_call(&after[..end], tool_calls.len()) {
                        tool_calls.push(call);
                    }
                    rest = &after[end + CALL_CLOSE.len()..];
                },
                None => {
                    // Truncated mid-call: the prose already copied above is kept,
                    // the fragment is dropped, and nothing after it can exist.
                    rest = "";
                    break;
                },
            }
        }
    }
    content.push_str(rest);

    ParsedOutput {
        content: content.trim().to_string(),
        tool_calls,
    }
}

/// Parse one `<function=NAME>…</function>` body into an OpenAI tool call.
fn parse_call(block: &str, index: usize) -> Option<ToolCall> {
    let name_start = block.find(FN_OPEN)? + FN_OPEN.len();
    let name_end = block[name_start..].find('>')? + name_start;
    let name = block[name_start..name_end].trim();
    if name.is_empty() {
        return None;
    }

    let mut args = Map::new();
    let mut rest = &block[name_end..];
    while let Some(p) = rest.find(PARAM_OPEN) {
        let after = &rest[p + PARAM_OPEN.len()..];
        let Some(key_end) = after.find('>') else {
            break;
        };
        let key = after[..key_end].trim().to_string();
        let value_region = &after[key_end + 1..];
        let Some(value_end) = value_region.find(PARAM_CLOSE) else {
            break;
        };
        // The template wraps values in newlines (`>\nVALUE\n</parameter>`),
        // but a value may itself span lines, so only the framing is trimmed.
        args.insert(key, coerce(value_region[..value_end].trim()));
        rest = &value_region[value_end + PARAM_CLOSE.len()..];
    }

    Some(ToolCall {
        // OpenAI requires an id so tool results can be correlated back. The
        // template does not emit one, so synthesize a stable per-message id.
        id: format!("call_{index}"),
        kind: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            // OpenAI carries arguments as a JSON *string*, not an object.
            arguments: Value::Object(args).to_string(),
        },
    })
}

/// Parameter values arrive as text. Recover JSON scalars so a client that
/// deserializes `arguments` sees `{"count": 3}` rather than `{"count": "3"}`,
/// while anything that is not valid JSON stays a string.
fn coerce(raw: &str) -> Value {
    match serde_json::from_str::<Value>(raw) {
        // A bare word like `Paris` is not JSON; `3`, `true`, `[1,2]` are.
        Ok(v) if !v.is_string() => v,
        _ => Value::String(raw.to_string()),
    }
}

/// Streaming filter that keeps tool-call markup out of `content` deltas.
///
/// A tool call cannot be streamed incrementally the way text can: the client
/// needs a complete, parseable call before it can run anything, and OpenAI's
/// own incremental `tool_calls` deltas assume a JSON grammar this template
/// does not use. So text streams normally until `<tool_call>` (or a bare
/// `<function=`, see `parse_output`'s doc) appears, after which everything is
/// buffered and the finished calls are emitted in one delta at the end.
///
/// A `<function=` mention in ordinary prose (with no real call anywhere
/// after it) triggers this buffering too, since there's no way to tell it
/// apart from a dropped opener until more text arrives — already-streamed
/// deltas can't be recalled. That reply's remaining prose is only delivered
/// once generation finishes rather than incrementally; `finish()` calling
/// `parse_output` still recovers the correct content either way.
#[derive(Default)]
pub struct ToolCallStream {
    /// Text held back because it might be the start of `<tool_call>` or
    /// `<function=`.
    pending: String,
    /// Everything from the first `<tool_call>`/`<function=` onward.
    buffered: String,
    in_call: bool,
}

impl ToolCallStream {
    /// Feed a decoded content delta; returns the part safe to stream now.
    pub fn push(&mut self, text: &str) -> String {
        if self.in_call {
            self.buffered.push_str(text);
            return String::new();
        }
        self.pending.push_str(text);

        // Start buffering at whichever tag appears first — a bare
        // `<function=` (a dropped `<tool_call>` opener, see `parse_output`'s
        // doc) must trigger this exactly like a well-formed opener does, or
        // its markup streams straight through as live content.
        if let Some((_, idx)) = next_tag_pos(&self.pending) {
            let emit = self.pending[..idx].to_string();
            self.buffered = self.pending[idx..].to_string();
            self.pending.clear();
            self.in_call = true;
            return emit;
        }
        // Hold back only a suffix that could still become `<tool_call>` or
        // `<function=`.
        let keep = crate::reasoning::partial_tag_suffix_len(&self.pending, CALL_OPEN).max(
            crate::reasoning::partial_tag_suffix_len(&self.pending, FN_OPEN),
        );
        let split = self.pending.len() - keep;
        let emit = self.pending[..split].to_string();
        self.pending = self.pending[split..].to_string();
        emit
    }

    /// Flush: any trailing content, plus the parsed calls.
    pub fn finish(&mut self) -> (String, Vec<ToolCall>) {
        if !self.in_call {
            return (std::mem::take(&mut self.pending), Vec::new());
        }
        let parsed = parse_output(&std::mem::take(&mut self.buffered));
        (parsed.content, parsed.tool_calls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact shape the Qwen 3.8 template instructs the model to produce.
    const CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

    #[test]
    fn parses_a_single_call() {
        let out = parse_output(CALL);
        assert_eq!(out.tool_calls.len(), 1);
        let c = &out.tool_calls[0];
        assert_eq!(c.function.name, "get_weather");
        assert_eq!(c.function.arguments, r#"{"city":"Paris"}"#);
        assert_eq!(c.kind, "function");
        assert_eq!(out.content, "");
    }

    /// The template explicitly allows reasoning before a call, so prose must
    /// survive as content rather than being swallowed.
    #[test]
    fn keeps_prose_around_calls() {
        let out = parse_output(&format!("Let me check that.\n{CALL}"));
        assert_eq!(out.content, "Let me check that.");
        assert_eq!(out.tool_calls.len(), 1);
    }

    #[test]
    fn parses_multiple_calls_with_distinct_ids() {
        let two = format!("{CALL}\n{CALL}");
        let out = parse_output(&two);
        assert_eq!(out.tool_calls.len(), 2);
        assert_ne!(out.tool_calls[0].id, out.tool_calls[1].id);
    }

    #[test]
    fn parses_multiple_parameters() {
        let text = "<tool_call>\n<function=search>\n<parameter=query>\nrust lang\n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>";
        let out = parse_output(text);
        let args: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust lang");
        // Numeric-looking values become JSON numbers, not strings.
        assert_eq!(args["limit"], 5);
    }

    /// Values may span lines — the template's own example says so.
    #[test]
    fn preserves_multiline_parameter_values() {
        let text = "<tool_call>\n<function=write>\n<parameter=body>\nline one\nline two\n</parameter>\n</function>\n</tool_call>";
        let out = parse_output(text);
        let args: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(args["body"], "line one\nline two");
    }

    /// Hitting the token limit mid-call must not yield a half-built call the
    /// client would then execute.
    #[test]
    fn drops_an_unterminated_call() {
        let out =
            parse_output("Checking.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nPar");
        assert!(out.tool_calls.is_empty());
        assert_eq!(out.content, "Checking.");
    }

    #[test]
    fn plain_text_has_no_calls() {
        let out = parse_output("Just an ordinary answer.");
        assert!(!out.has_calls());
        assert_eq!(out.content, "Just an ordinary answer.");
    }

    /// A malformed block (no `<function=`) is skipped, not turned into a call
    /// with an empty name.
    #[test]
    fn skips_a_block_without_a_function_name() {
        let out = parse_output("<tool_call>\ngarbage\n</tool_call>");
        assert!(out.tool_calls.is_empty());
    }

    // ── bare `<function=...>` fallback (missing `<tool_call>` opener) ──

    /// The exact bug reproduced live against the model: a batch of 4 chained
    /// calls where the first is missing its `<tool_call>` opener but still
    /// has a `</tool_call>` closer, and the rest are well-formed. All 4 must
    /// be recovered, in order, with distinct ids.
    #[test]
    fn recovers_bare_function_without_opening_tag() {
        let text = "<function=glob>\n<parameter=pattern>\n*/openai.ts\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=glob>\n<parameter=pattern>\n*/sglang.ts\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=glob>\n<parameter=pattern>\n*/llama.ts\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=glob>\n<parameter=pattern>\n*/mistral.ts\n</parameter>\n</function>\n</tool_call>";
        let out = parse_output(text);
        assert_eq!(out.tool_calls.len(), 4);
        assert_eq!(out.tool_calls[0].function.name, "glob");
        let a0: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(a0["pattern"], "*/openai.ts");
        assert_eq!(out.tool_calls[0].id, "call_0");
        assert_eq!(out.tool_calls[1].id, "call_1");
        assert_eq!(out.tool_calls[3].id, "call_3");
        assert_eq!(out.content, "");
    }

    /// A bare `<function=...>` with no `<tool_call>` wrapper at all — not
    /// even a stray closer.
    #[test]
    fn recovers_bare_function_without_any_wrapper() {
        let out = parse_output("<function=ls>\n<parameter=path>\n/tmp\n</parameter>\n</function>");
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.name, "ls");
        let args: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(args["path"], "/tmp");
        assert_eq!(out.content, "");
    }

    /// Multiple consecutive bare functions without any wrapper tags.
    #[test]
    fn recovers_multiple_consecutive_bare_functions() {
        let text = "<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n<function=b>\n<parameter=y>\n2\n</parameter>\n</function>";
        let out = parse_output(text);
        assert_eq!(out.tool_calls.len(), 2);
        assert_eq!(out.tool_calls[0].function.name, "a");
        assert_eq!(out.tool_calls[1].function.name, "b");
        assert_eq!(out.tool_calls[0].id, "call_0");
        assert_eq!(out.tool_calls[1].id, "call_1");
    }

    /// A bare function followed by prose — the prose must survive as content.
    #[test]
    fn keeps_prose_after_bare_function() {
        let out = parse_output(
            "<function=ls>\n<parameter=path>\n.\n</parameter>\n</function>\nHere are the results.",
        );
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.content, "Here are the results.");
    }

    /// A bare `<function=` with no `</function>` anywhere after it is not
    /// distinguishable from a genuine truncated call, so — unlike a truncated
    /// `<tool_call>` — it is not dropped: it is re-emitted as literal content
    /// so a real mid-sentence mention of the tag isn't silently discarded.
    #[test]
    fn keeps_unclosed_bare_function_as_content() {
        let out = parse_output("Checking.\n<function=ls>\n<parameter=path>\n/tm");
        assert!(out.tool_calls.is_empty());
        assert_eq!(
            out.content,
            "Checking.\n<function=ls>\n<parameter=path>\n/tm"
        );
    }

    /// A bare `<function=` mentioned in prose with no closer anywhere in the
    /// message, and no more tags after it, must not swallow the rest of the
    /// answer — this is the scenario a coding assistant hits when explaining
    /// tool-call syntax without actually invoking one.
    #[test]
    fn bare_function_in_prose_without_closer_keeps_content() {
        let text = "You can write calls like <function=name> in the template.";
        let out = parse_output(text);
        assert!(out.tool_calls.is_empty());
        assert_eq!(out.content, text);
    }

    /// A well-formed `<tool_call>` block followed by a bare `<function=`
    /// block (missing its own opener) — both must be recovered.
    #[test]
    fn recovers_bare_function_after_well_formed_call() {
        let text = "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n<function=b>\n<parameter=y>\n2\n</parameter>\n</function>";
        let out = parse_output(text);
        assert_eq!(out.tool_calls.len(), 2);
        assert_eq!(out.tool_calls[0].function.name, "a");
        assert_eq!(out.tool_calls[1].function.name, "b");
    }

    /// Two orphaned `</tool_call>` closers stacked after one bare block must
    /// both be swallowed, not just the first.
    #[test]
    fn swallows_multiple_orphaned_closers() {
        let text = "<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n</tool_call>\nDone.";
        let out = parse_output(text);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.name, "a");
        assert_eq!(out.content, "Done.");
    }

    // ── streaming ──

    /// Markup must never reach the client as content, even when the opening
    /// tag is split across token boundaries.
    #[test]
    fn stream_withholds_tool_markup() {
        let mut s = ToolCallStream::default();
        let mut streamed = String::new();
        for tok in [
            "Let me ",
            "check.",
            "\n<",
            "tool",
            "_call>",
            "\n<function=get_weather>\n",
            "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
        ] {
            streamed.push_str(&s.push(tok));
        }
        let (tail, calls) = s.finish();
        streamed.push_str(&tail);

        assert_eq!(streamed.trim(), "Let me check.");
        assert!(
            !streamed.contains("tool_call"),
            "markup leaked: {streamed:?}"
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    /// Ordinary replies must still stream token by token — a `<` that never
    /// becomes a tag has to be released, not swallowed.
    #[test]
    fn stream_passes_plain_text_through() {
        let mut s = ToolCallStream::default();
        let mut out = String::new();
        for tok in ["a < b", " and ", "c > d"] {
            out.push_str(&s.push(tok));
        }
        let (tail, calls) = s.finish();
        out.push_str(&tail);
        assert_eq!(out, "a < b and c > d");
        assert!(calls.is_empty());
    }

    /// A bare `<function=` (no `<tool_call>` opener) must trigger buffering
    /// exactly like a well-formed opener does, even split across token
    /// boundaries — otherwise its markup streams straight through as content.
    #[test]
    fn stream_withholds_bare_function_markup() {
        let mut s = ToolCallStream::default();
        let mut streamed = String::new();
        for tok in [
            "Let me ",
            "check.",
            "\n<",
            "func",
            "tion=get_weather>",
            "\n<parameter=city>\nParis\n</parameter>",
            "\n</function>",
        ] {
            streamed.push_str(&s.push(tok));
        }
        let (tail, calls) = s.finish();
        streamed.push_str(&tail);

        assert_eq!(streamed.trim(), "Let me check.");
        assert!(
            !streamed.contains("function"),
            "markup leaked: {streamed:?}"
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
    }

    /// Streaming equivalent of `recovers_bare_function_without_opening_tag`:
    /// the first call in a batch is missing its `<tool_call>` opener, fed in
    /// small chunks to simulate token-by-token delivery.
    #[test]
    fn stream_recovers_bare_first_call_in_batch() {
        let mut s = ToolCallStream::default();
        let mut streamed = String::new();

        let input = "<function=glob>\n<parameter=pattern>\n*/openai.ts\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=glob>\n<parameter=pattern>\n*/sglang.ts\n</parameter>\n</function>\n</tool_call>";

        // Feed in small chunks to simulate token-by-token delivery.
        for tok in input.split_inclusive('\n') {
            streamed.push_str(&s.push(tok));
        }
        let (tail, calls) = s.finish();
        streamed.push_str(&tail);

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "glob");
        assert_eq!(calls[1].function.name, "glob");
        assert!(
            streamed.trim().is_empty(),
            "unexpected content: {streamed:?}"
        );
    }

    // ── bare `<function=` mentioned in prose ahead of a real call ──

    /// A prose mention of `<function=x>` followed by a well-formed
    /// `<tool_call>` block must not let the parser steal that block's
    /// `</function>` for the prose mention — the prose is kept as content
    /// and the real call is parsed with its own name and arguments.
    #[test]
    fn bare_function_in_prose_before_real_call() {
        let text = "See <function=x> for syntax.\n<tool_call>\n<function=real>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>";
        let out = parse_output(text);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.name, "real");
        let args: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(args["a"], 1);
        assert!(
            out.content.contains("<function=x>"),
            "prose mention lost: {:?}",
            out.content
        );
    }

    /// Streaming equivalent of `bare_function_in_prose_before_real_call`.
    #[test]
    fn stream_bare_function_in_prose_before_real_call() {
        let input = "See <function=x> for syntax.\n<tool_call>\n<function=real>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>";
        let mut s = ToolCallStream::default();
        let mut streamed = String::new();
        for tok in input.split_inclusive('\n') {
            streamed.push_str(&s.push(tok));
        }
        let (tail, calls) = s.finish();
        streamed.push_str(&tail);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "real");
        assert!(
            streamed.contains("<function=x>"),
            "prose mention lost: {streamed:?}"
        );
    }

    /// A prose `<function=x>` (no closer) followed by a legitimate bare
    /// `<function=real>...</function>` — the prose is kept and the real
    /// call is still recovered.
    #[test]
    fn bare_function_in_prose_before_another_bare_call() {
        let text = "See <function=x> for syntax.\n<function=real>\n<parameter=a>\n1\n</parameter>\n</function>";
        let out = parse_output(text);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.name, "real");
        let args: Value = serde_json::from_str(&out.tool_calls[0].function.arguments).unwrap();
        assert_eq!(args["a"], 1);
        assert!(
            out.content.contains("<function=x>"),
            "prose mention lost: {:?}",
            out.content
        );
    }
}
