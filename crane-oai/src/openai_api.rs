//! OpenAI-compatible request / response types.
//!
//! Covers:
//! * `/v1/chat/completions`  (chat)
//! * `/v1/completions`       (text completion)
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
    pub content: String,
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
