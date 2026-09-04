//! OpenAI-compatible request / response types.
//!
//! Covers:
//! * `/v1/chat/completions`  (chat)
//! * `/v1/completions`       (text completion)
//! * `/v1/audio/speech`      (text-to-speech)
//! * `/v1/models`            (model listing + retrieval)
//! * `/v1/tokenize`          (tokenization)
//! * `/v1/detokenize`        (detokenization)
//!
//! Wire format follows the
//! [OpenAI API reference](https://platform.openai.com/docs/api-reference).

use serde::{Deserialize, Serialize};

// ═════════════════════════════════════════════════════════════
//  Shared helpers
// ═════════════════════════════════════════════════════════════

fn default_max_tokens() -> usize {
    512
}

// ═════════════════════════════════════════════════════════════
//  Chat Completion  (/v1/chat/completions)
// ═════════════════════════════════════════════════════════════

// ── Request ──

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// OpenAI's newer alias for `max_tokens`; takes precedence when present.
    pub max_completion_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    pub stop: Option<StringOrArray>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub n: Option<usize>,
    /// Response format constraint (e.g., `{"type": "json_object"}`).
    pub response_format: Option<ResponseFormat>,
    /// Extra variables for the Jinja chat template (vLLM/SGLang convention).
    /// Reasoning models read `enable_thinking` and `reasoning_effort` from
    /// here; see [`crate::reasoning::ThinkingOptions`].
    pub chat_template_kwargs: Option<serde_json::Value>,
    /// OpenAI's top-level reasoning budget (`low` / `medium` / `xhigh` for the
    /// Qwen 3.6+ templates). `chat_template_kwargs` takes precedence.
    pub reasoning_effort: Option<String>,
    /// Function/tool specs, passed to the chat template verbatim — the
    /// template owns the prompt syntax (`tool | tojson` for the Qwen family).
    pub tools: Option<Vec<Tool>>,
    /// Accepted for OpenAI compatibility. `"none"` suppresses the tool block;
    /// anything else is advisory, since forcing a specific call would require
    /// constrained decoding the engine does not implement.
    pub tool_choice: Option<serde_json::Value>,
}

/// A tool the model may call. Only `type: "function"` exists today, and the
/// body is passed through to the template untouched, so an unusual `parameters`
/// schema needs no support here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type", default = "default_function_type")]
    pub kind: String,
    pub function: FunctionDef,
}

fn default_function_type() -> String {
    "function".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the arguments. Free-form so any schema round-trips.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// One call the model decided to make.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "default_function_type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments, per the OpenAI wire format — a *string*, not an
    /// object, so clients must parse it themselves.
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Assistant messages that only call tools carry `content: null`, which is
    /// why this tolerates both a missing field and an explicit null.
    #[serde(default, deserialize_with = "null_as_default")]
    pub content: ChatMessageContent,
    /// The model's `<think>` scratchpad, separated out of `content` so clients
    /// can display or discard it independently. Only ever set on responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Calls the model chose to make. Set on responses, and echoed back by the
    /// client on the follow-up turn so the template can re-render them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// On a `role: "tool"` message, the `id` of the call being answered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool name on a `role: "tool"` message (pre-`tool_call_id` clients).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Treat an explicit JSON `null` like a missing field. `#[serde(default)]`
/// alone does not: it covers absence, not null.
fn null_as_default<'de, D, T>(d: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(d)?.unwrap_or_default())
}

impl Default for ChatMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl ChatMessage {
    /// An assistant reply with no reasoning block.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: ChatMessageContent::Text(content.into()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// An assistant reply whose scratchpad was split out of the raw output.
    pub fn assistant_with_reasoning(
        content: impl Into<String>,
        reasoning_content: Option<String>,
    ) -> Self {
        Self {
            role: "assistant".into(),
            content: ChatMessageContent::Text(content.into()),
            reasoning_content,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// An assistant reply that calls tools. `content` may be empty — the
    /// template allows prose before a call but does not require it.
    pub fn assistant_with_tools(
        content: impl Into<String>,
        reasoning_content: Option<String>,
        tool_calls: Vec<ToolCall>,
    ) -> Self {
        Self {
            role: "assistant".into(),
            content: ChatMessageContent::Text(content.into()),
            reasoning_content,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            tool_call_id: None,
            name: None,
        }
    }

    /// Extract the plain text content from the message.
    /// For multimodal messages, concatenates all text parts.
    pub fn text_content(&self) -> String {
        match &self.content {
            ChatMessageContent::Text(s) => s.clone(),
            ChatMessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Extract image URLs from multimodal content.
    pub fn image_urls(&self) -> Vec<String> {
        match &self.content {
            ChatMessageContent::Text(_) => vec![],
            ChatMessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::ImageUrl { image_url } => Some(image_url.url.clone()),
                    _ => None,
                })
                .collect(),
        }
    }
}

/// Chat message content — either a plain string or structured multimodal parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatMessageContent {
    /// Plain text content (backward compatible).
    Text(String),
    /// Structured content with text and/or image_url parts.
    Parts(Vec<ContentPart>),
}

/// A single content part in a multimodal message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content.
    #[serde(rename = "text")]
    Text { text: String },
    /// Image URL content.
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    /// Image content (alternative key used by some OpenAI clients).
    #[serde(rename = "image")]
    Image { image_url: Option<ImageUrl> },
}

/// An image URL reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ResponseFormat {
    pub r#type: String,
}

// ── Response ──

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

// ── Streaming ──

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Streamed counterpart of [`ChatMessage::reasoning_content`]: deltas
    /// generated inside a `<think>` block arrive here instead of `content`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Tool calls, emitted once complete rather than incrementally — see
    /// [`crate::tools`] for why a partial call must never reach the client.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ChunkDelta {
    pub fn role(role: impl Into<String>) -> Self {
        Self {
            role: Some(role.into()),
            content: None,
            reasoning_content: None,
            tool_calls: None,
        }
    }

    pub fn content(text: impl Into<String>) -> Self {
        Self {
            role: None,
            content: Some(text.into()),
            reasoning_content: None,
            tool_calls: None,
        }
    }

    /// Empty delta, used by the terminal `finish_reason` chunk.
    pub fn empty() -> Self {
        Self {
            role: None,
            content: None,
            reasoning_content: None,
            tool_calls: None,
        }
    }

    pub fn tool_calls(calls: Vec<ToolCall>) -> Self {
        Self {
            role: None,
            content: None,
            reasoning_content: None,
            tool_calls: Some(calls),
        }
    }
}

// ═════════════════════════════════════════════════════════════
//  Text Completion  (/v1/completions)
// ═════════════════════════════════════════════════════════════

// ── Request ──

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: StringOrArray,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// OpenAI's newer alias for `max_tokens`; takes precedence when present.
    pub max_completion_tokens: Option<usize>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    pub stop: Option<StringOrArray>,
    pub suffix: Option<String>,
    pub echo: Option<bool>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub n: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StringOrArray {
    Single(String),
    Array(Vec<String>),
}

impl StringOrArray {
    pub fn as_string(&self) -> String {
        match self {
            Self::Single(s) => s.clone(),
            Self::Array(arr) => arr.join(""),
        }
    }

    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Array(arr) => arr,
        }
    }
}

// ── Response ──

#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
}

// ── Streaming ──

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunkChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: Option<String>,
}

// ═════════════════════════════════════════════════════════════
//  Tokenize / Detokenize  (/v1/tokenize, /v1/detokenize)
// ═════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizeRequest {
    /// Text to tokenize. Can be a single string or a list of chat messages.
    pub text: Option<String>,
    /// If provided, treat as chat messages and apply the chat template first.
    pub messages: Option<Vec<ChatMessage>>,
    /// Whether to add special tokens (default: true).
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<u32>,
    pub count: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DetokenizeRequest {
    pub tokens: Vec<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DetokenizeResponse {
    pub text: String,
}

// ═════════════════════════════════════════════════════════════
//  Audio Speech  (/v1/audio/speech)
// ═════════════════════════════════════════════════════════════

/// Audio response format.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum AudioResponseFormat {
    #[default]
    Wav,
    Pcm,
    Mp3,
    Opus,
    Aac,
    Flac,
}

fn default_speed() -> f64 {
    1.0
}

fn default_audio_max_tokens() -> usize {
    8192
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct SpeechRequest {
    /// Model ID (e.g. "qwen3-tts").
    pub model: String,
    /// The text to synthesize.
    pub input: String,
    /// Voice name (speaker ID or preset for CustomVoice models).
    #[serde(default)]
    pub voice: Option<String>,
    /// Language hint as an ISO 639-1 code (e.g. "zh", "en", "ja") or "auto".
    #[serde(default)]
    pub language: Option<String>,
    /// Optional system-level instructions.
    #[serde(default)]
    pub instructions: Option<String>,
    /// Response audio format.
    #[serde(default)]
    pub response_format: AudioResponseFormat,
    /// Speaking speed multiplier (currently unused, reserved).
    #[serde(default = "default_speed")]
    pub speed: f64,
    /// Generation temperature.
    pub temperature: Option<f64>,
    /// Nucleus sampling top-p.
    pub top_p: Option<f64>,
    /// Repetition penalty.
    pub repetition_penalty: Option<f32>,
    /// Max codec tokens to generate (controls max duration).
    #[serde(default = "default_audio_max_tokens")]
    pub max_tokens: usize,
    /// VoxCPM2 only: flow-matching (CFM) sampler steps per audio frame.
    /// Omitted keeps the model default (10). Lower = faster generation for a
    /// modest quality cost — the CFM decoder dominates VoxCPM2 latency.
    #[serde(default)]
    pub cfm_steps: Option<usize>,
    /// VoxCPM2 only: classifier-free guidance strength for the CFM sampler.
    /// Omitted keeps the model default (2.0).
    #[serde(default)]
    pub cfg_scale: Option<f64>,
    /// Stream the audio as it is generated (chunked `audio/pcm`, 16-bit LE
    /// mono at the model's sample rate — see the `X-Sample-Rate` response
    /// header). Requires `response_format: "pcm"` and is not supported with
    /// voice cloning. Only VoxCPM2 generates incrementally today; other TTS
    /// models fall back to emitting the whole clip as one chunk.
    #[serde(default)]
    pub stream: bool,

    // ── Voice-clone fields (Base model only) ──────────────────
    /// URL or local path to reference audio for voice cloning.
    /// When set, triggers voice-clone mode (requires Base model).
    #[serde(default)]
    pub reference_audio: Option<String>,
    /// Transcript of the reference audio (required for voice cloning).
    #[serde(default)]
    pub reference_text: Option<String>,
}

// ═════════════════════════════════════════════════════════════
//  Audio Transcriptions  (/v1/audio/transcriptions)
// ═════════════════════════════════════════════════════════════

/// Response body for `POST /v1/audio/transcriptions`.
#[derive(Debug, Clone, Serialize)]
pub struct TranscriptionResponse {
    /// The transcribed text.
    pub text: String,
}

// ═════════════════════════════════════════════════════════════
//  Common types
// ═════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

// ── Model listing ──

#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permission: Option<Vec<serde_json::Value>>,
}

// ── Error ──

#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}
