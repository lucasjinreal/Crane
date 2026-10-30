//! Thinking control and `<think>` separation for reasoning models.
//!
//! Two halves of the same feature:
//!
//! * **[`ThinkingOptions`]** — what the *request* asks for, forwarded into the
//!   Jinja chat template as `enable_thinking` / `reasoning_effort`.
//! * **[`ReasoningSplitter`]** — what comes *back*: reasoning models emit their
//!   scratchpad and their answer in one token stream, and the two have to be
//!   separated before the answer reaches the client.
//!
//! The subtlety is that the opening `<think>` usually is **not** in the model's
//! output at all — Qwen 3.5/3.6/3.8 templates end the generation prompt with a
//! dangling `<think>\n`, so the completion opens mid-scratchpad and the first
//! tag the model emits is the *closing* one. Deciding "are we in reasoning?"
//! from the output alone therefore gets it backwards; [`ReasoningSplitter`]
//! reads the rendered prompt instead, which also gets the
//! `enable_thinking: false` case right (those templates pre-close the block, so
//! the completion is pure content).

use serde_json::Value;

pub const THINK_OPEN: &str = "<think>";
pub const THINK_CLOSE: &str = "</think>";

// ─────────────────────────────────────────────────────────────
//  Request-side: what the template is told
// ─────────────────────────────────────────────────────────────

/// Reasoning knobs forwarded to the chat template.
///
/// `None` means "leave the template variable undefined", which is not the same
/// as passing a value — the Qwen templates branch on `is defined` and on
/// `|default('xhigh')`, so an undefined variable selects the model's own
/// default while an explicit one overrides it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ThinkingOptions {
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<String>,
}

impl ThinkingOptions {
    /// Read the knobs from an OpenAI-style request.
    ///
    /// `chat_template_kwargs` is the vLLM/SGLang convention
    /// (`{"chat_template_kwargs": {"enable_thinking": false}}`);
    /// `reasoning_effort` is `OpenAI`'s own top-level field. When both carry an
    /// effort, `chat_template_kwargs` wins — it is the more specific,
    /// template-targeted channel.
    pub fn from_request(
        chat_template_kwargs: Option<&Value>,
        reasoning_effort: Option<&str>,
    ) -> Self {
        let kwargs = chat_template_kwargs.and_then(Value::as_object);
        Self {
            enable_thinking: kwargs
                .and_then(|k| k.get("enable_thinking"))
                .and_then(Value::as_bool),
            reasoning_effort: kwargs
                .and_then(|k| k.get("reasoning_effort"))
                .and_then(Value::as_str)
                .or(reasoning_effort)
                .map(str::to_owned),
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Response-side: splitting the stream
// ─────────────────────────────────────────────────────────────

/// Incrementally splits a completion into reasoning and content.
///
/// Feed it decoded token text with [`push`](Self::push) and flush with
/// [`finish`](Self::finish). Tags are recognized even when split across token
/// boundaries (`</` + `think>`): any trailing text that could still grow into a
/// tag is held back rather than emitted.
pub struct ReasoningSplitter {
    in_reasoning: bool,
    /// Text held back because it may be the start of a tag.
    pending: String,
    seen_reasoning: bool,
    seen_content: bool,
}

impl ReasoningSplitter {
    /// Decide the starting state from the rendered prompt.
    ///
    /// If the prompt's last `<think>` is not followed by a `</think>`, the
    /// template left the block open and the completion begins inside it.
    #[must_use]
    pub fn for_prompt(prompt: &str) -> Self {
        let open = prompt.rfind(THINK_OPEN);
        let close = prompt.rfind(THINK_CLOSE);
        let in_reasoning = match (open, close) {
            (Some(o), Some(c)) => o > c,
            (Some(_), None) => true,
            (None, _) => false,
        };
        Self {
            in_reasoning,
            pending: String::new(),
            seen_reasoning: false,
            seen_content: false,
        }
    }

    /// A splitter for output known to contain no reasoning block.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            in_reasoning: false,
            pending: String::new(),
            seen_reasoning: false,
            seen_content: false,
        }
    }

    /// Whether the completion is currently inside a reasoning block.
    #[must_use]
    pub fn in_reasoning(&self) -> bool {
        self.in_reasoning
    }

    /// Consume `text`, returning `(reasoning_delta, content_delta)`. Either may
    /// be empty; both are empty while a partial tag is buffered.
    pub fn push(&mut self, text: &str) -> (String, String) {
        self.pending.push_str(text);
        let mut reasoning = String::new();
        let mut content = String::new();

        loop {
            let tag = if self.in_reasoning {
                THINK_CLOSE
            } else {
                THINK_OPEN
            };
            if let Some(idx) = self.pending.find(tag) {
                let before: String = self.pending[..idx].into();
                self.emit(&before, &mut reasoning, &mut content);
                self.pending = self.pending[idx + tag.len()..].to_string();
                self.in_reasoning = !self.in_reasoning;
                continue;
            }
            // No complete tag. Hold back only a suffix that could still become
            // one; everything before it is safe to emit.
            let keep = partial_tag_suffix_len(&self.pending, tag);
            let split = self.pending.len() - keep;
            let ready: String = self.pending[..split].into();
            self.emit(&ready, &mut reasoning, &mut content);
            self.pending = self.pending[split..].to_string();
            break;
        }
        (reasoning, content)
    }

    /// Flush whatever is buffered; call once the stream ends.
    pub fn finish(&mut self) -> (String, String) {
        let rest = std::mem::take(&mut self.pending);
        let mut reasoning = String::new();
        let mut content = String::new();
        self.emit(&rest, &mut reasoning, &mut content);
        (reasoning, content)
    }

    /// Route `text` to the active sink, trimming the leading whitespace that
    /// templates leave behind after a tag (`</think>\n\n`).
    fn emit(&mut self, text: &str, reasoning: &mut String, content: &mut String) {
        if text.is_empty() {
            return;
        }
        if self.in_reasoning {
            let text = if self.seen_reasoning {
                text
            } else {
                text.trim_start()
            };
            if !text.is_empty() {
                self.seen_reasoning = true;
                reasoning.push_str(text);
            }
        } else {
            let text = if self.seen_content {
                text
            } else {
                text.trim_start()
            };
            if !text.is_empty() {
                self.seen_content = true;
                content.push_str(text);
            }
        }
    }
}

/// Length of the longest suffix of `haystack` that is a proper prefix of `tag`.
///
/// This is what must stay buffered: `"...</"` could still become `"</think>"`,
/// so it cannot be emitted as content yet.
pub(crate) fn partial_tag_suffix_len(haystack: &str, tag: &str) -> usize {
    let max = tag.len().saturating_sub(1).min(haystack.len());
    (1..=max)
        .rev()
        .find(|&n| {
            haystack.is_char_boundary(haystack.len() - n)
                && tag.starts_with(&haystack[haystack.len() - n..])
        })
        .unwrap_or(0)
}

/// Split a complete (non-streamed) completion.
///
/// Returns `(reasoning_content, content)`; `reasoning_content` is `None` when
/// the completion carried no reasoning block.
#[must_use]
pub fn split_complete(prompt: &str, output: &str) -> (Option<String>, String) {
    let mut splitter = ReasoningSplitter::for_prompt(prompt);
    let (mut reasoning, mut content) = splitter.push(output);
    let (r2, c2) = splitter.finish();
    reasoning.push_str(&r2);
    content.push_str(&c2);

    let reasoning = reasoning.trim_end().to_string();
    let content = content.trim_end().to_string();
    if reasoning.is_empty() {
        (None, content)
    } else {
        (Some(reasoning), content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What Qwen 3.8 actually produces: the prompt ends with a dangling
    /// `<think>`, so the completion opens mid-scratchpad and the only tag in
    /// the output is the closing one.
    #[test]
    fn splits_completion_whose_open_tag_was_in_the_prompt() {
        let prompt = "<|im_start|>assistant\n<think>\n";
        let output = "We need answer user. Simple.\n</think>\n\n4";
        let (reasoning, content) = split_complete(prompt, output);
        assert_eq!(reasoning.as_deref(), Some("We need answer user. Simple."));
        assert_eq!(content, "4");
    }

    /// `enable_thinking: false` pre-closes the block in the prompt, so the
    /// completion is pure content — assuming "starts in reasoning" would file
    /// the whole answer as scratchpad and return empty content.
    #[test]
    fn treats_preclosed_prompt_block_as_content() {
        let prompt = "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let (reasoning, content) = split_complete(prompt, "4");
        assert_eq!(reasoning, None);
        assert_eq!(content, "4");
    }

    /// A model that emits both tags itself must work too.
    #[test]
    fn splits_completion_containing_both_tags() {
        let (reasoning, content) = split_complete("user: hi\n", "<think>hmm</think>hello");
        assert_eq!(reasoning.as_deref(), Some("hmm"));
        assert_eq!(content, "hello");
    }

    #[test]
    fn non_reasoning_output_is_all_content() {
        let (reasoning, content) = split_complete("user: hi\n", "just an answer");
        assert_eq!(reasoning, None);
        assert_eq!(content, "just an answer");
    }

    /// Streaming: the tag arrives split across token boundaries, and no partial
    /// tag may leak into either sink.
    #[test]
    fn buffers_tag_split_across_tokens() {
        let mut s = ReasoningSplitter::for_prompt("<think>\n");
        let mut reasoning = String::new();
        let mut content = String::new();
        for tok in [
            "think", "ing", " a bit", "\n<", "/", "think", ">", "\n\n", "answer",
        ] {
            let (r, c) = s.push(tok);
            reasoning.push_str(&r);
            content.push_str(&c);
        }
        let (r, c) = s.finish();
        reasoning.push_str(&r);
        content.push_str(&c);

        assert_eq!(reasoning, "thinking a bit\n");
        assert_eq!(content, "answer");
    }

    /// A `<` that never becomes a tag must still be emitted, not swallowed.
    #[test]
    fn releases_buffered_text_that_is_not_a_tag() {
        let mut s = ReasoningSplitter::disabled();
        let (_, c1) = s.push("a < b");
        let (_, c2) = s.push(" and c");
        let (_, c3) = s.finish();
        assert_eq!(format!("{c1}{c2}{c3}"), "a < b and c");
    }

    /// Unterminated reasoning (hit the token limit mid-thought) must not be
    /// silently reported as the answer.
    #[test]
    fn unterminated_reasoning_stays_reasoning() {
        let (reasoning, content) = split_complete("<think>\n", "still thinking when cut off");
        assert_eq!(reasoning.as_deref(), Some("still thinking when cut off"));
        assert_eq!(content, "");
    }

    #[test]
    fn multibyte_text_is_not_split_mid_character() {
        let (reasoning, content) = split_complete("<think>\n", "推理中\n</think>\n\n答案は4です");
        assert_eq!(reasoning.as_deref(), Some("推理中"));
        assert_eq!(content, "答案は4です");
    }

    // ── ThinkingOptions ──

    #[test]
    fn thinking_options_read_chat_template_kwargs() {
        let kwargs = serde_json::json!({"enable_thinking": false, "reasoning_effort": "low"});
        let opts = ThinkingOptions::from_request(Some(&kwargs), None);
        assert_eq!(opts.enable_thinking, Some(false));
        assert_eq!(opts.reasoning_effort.as_deref(), Some("low"));
    }

    #[test]
    fn thinking_options_fall_back_to_top_level_effort() {
        let opts = ThinkingOptions::from_request(None, Some("medium"));
        assert_eq!(opts.enable_thinking, None);
        assert_eq!(opts.reasoning_effort.as_deref(), Some("medium"));
    }

    /// The template-targeted channel is the more specific one.
    #[test]
    fn thinking_options_prefer_kwargs_over_top_level_effort() {
        let kwargs = serde_json::json!({"reasoning_effort": "low"});
        let opts = ThinkingOptions::from_request(Some(&kwargs), Some("xhigh"));
        assert_eq!(opts.reasoning_effort.as_deref(), Some("low"));
    }

    /// An absent knob must stay absent, so the template's own default applies.
    #[test]
    fn thinking_options_default_to_undefined() {
        let opts = ThinkingOptions::from_request(None, None);
        assert_eq!(opts, ThinkingOptions::default());
    }
}
