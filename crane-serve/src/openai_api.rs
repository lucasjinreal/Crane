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
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    pub stop: Option<Vec<String>>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub n: Option<usize>,
    /// Response format constraint (e.g., `{"type": "json_object"}`).
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: ChatMessageContent,
}

impl ChatMessage {
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
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    pub stop: Option<Vec<String>>,
    pub suffix: Option<String>,
    pub echo: Option<bool>,
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
    /// Language hint (e.g. "chinese", "english", "japanese", "auto").
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
