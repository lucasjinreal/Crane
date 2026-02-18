//! SGLang-compatible native API types.
//!
//! Covers the non-OpenAI endpoints inspired by SGLang's native interface:
//!
//! * `POST /generate`       — native text generation
//! * `GET  /model_info`     — model metadata
//! * `GET  /server_info`    — server configuration + live stats
//! * `POST /flush_cache`    — flush KV caches
//! * `POST /abort_request`  — cancel in-flight request
//! * `GET  /health_generate` — deep health check (runs 1-token generation)

use serde::{Deserialize, Serialize};

// ═════════════════════════════════════════════════════════════
//  /generate — Native generation endpoint
// ═════════════════════════════════════════════════════════════

/// Native generation request.
///
/// Accepts either a text prompt or raw token IDs.
/// Follows SGLang's `GenerateReqInput` pattern with sensible defaults.
#[derive(Debug, Clone, Deserialize)]
pub struct GenerateRequest {
    /// Prompt text. Mutually exclusive with `input_ids`.
    pub text: Option<String>,
    /// Pre-tokenized input. Mutually exclusive with `text`.
    pub input_ids: Option<Vec<u32>>,
    /// Sampling parameters.
    #[serde(default)]
    pub sampling_params: SamplingParams,
    /// Enable SSE streaming.
    #[serde(default)]
    pub stream: bool,
    /// Custom request ID (auto-generated if omitted).
    pub rid: Option<String>,
}

/// Sampling parameters for the `/generate` endpoint.
///
/// Modeled after SGLang's `SamplingParams` with reasonable defaults.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct SamplingParams {
    /// Maximum number of tokens to generate.
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: usize,
    /// Sampling temperature. 0 = greedy.
    pub temperature: Option<f64>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f64>,
    /// Top-k sampling. 0 or negative = disabled.
    pub top_k: Option<usize>,
    /// Repetition penalty. 1.0 = no penalty.
    #[serde(default = "default_one_f32")]
    pub repetition_penalty: f32,
    /// Frequency penalty [-2, 2].
    pub frequency_penalty: Option<f32>,
    /// Presence penalty [-2, 2].
    pub presence_penalty: Option<f32>,
    /// Stop strings — generation stops when any is produced.
    pub stop: Option<StringOrList>,
    /// Stop token IDs — generation stops on these IDs.
    pub stop_token_ids: Option<Vec<u32>>,
    /// Whether to skip special tokens in output.
    #[serde(default = "default_true")]
    pub skip_special_tokens: bool,
    /// Random seed for deterministic sampling.
    pub seed: Option<u64>,
    /// Number of parallel completions per prompt.
    #[serde(default = "default_one")]
    pub n: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_new_tokens: default_max_new_tokens(),
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: 1.0,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stop_token_ids: None,
            skip_special_tokens: true,
            seed: None,
            n: 1,
        }
    }
}

fn default_max_new_tokens() -> usize {
    128
}
fn default_one_f32() -> f32 {
    1.0
}
fn default_one() -> usize {
    1
}
fn default_true() -> bool {
    true
}

/// A string or list of strings (for stop sequences).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
pub enum StringOrList {
    Single(String),
    List(Vec<String>),
}

#[allow(dead_code)]
impl StringOrList {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::List(v) => v,
        }
    }
}

/// Non-streaming generate response.
#[derive(Debug, Clone, Serialize)]
pub struct GenerateResponse {
    /// Generated text.
    pub text: String,
    /// Metadata about the generation.
    pub meta_info: GenerateMetaInfo,
}

/// Metadata attached to generation responses.
#[derive(Debug, Clone, Serialize)]
pub struct GenerateMetaInfo {
    /// Unique request ID.
    pub id: String,
    /// Number of prompt tokens.
    pub prompt_tokens: usize,
    /// Number of generated (completion) tokens.
    pub completion_tokens: usize,
    /// Why generation stopped: "stop", "length", or "abort".
    pub finish_reason: String,
}

/// Streaming chunk for `/generate` (SSE `data:` payload).
#[derive(Debug, Clone, Serialize)]
pub struct GenerateStreamChunk {
    /// Incremental text delta.
    pub text: String,
    /// Metadata (populated on the final chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta_info: Option<GenerateMetaInfo>,
}

// ═════════════════════════════════════════════════════════════
//  /model_info
// ═════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize)]
pub struct ModelInfoResponse {
    pub model_path: String,
    pub model_type: String,
    pub is_generation: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,
}

// ═════════════════════════════════════════════════════════════
//  /server_info
// ═════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize)]
pub struct ServerInfoResponse {
    pub version: String,
    pub model_path: String,
    pub model_type: String,
    pub host: String,
    pub port: u16,
    pub max_concurrent: usize,
    pub decode_tokens_per_seq: usize,
    /// Live engine statistics snapshot.
    pub stats: crate::engine::StatsSnapshot,
}

// ═════════════════════════════════════════════════════════════
//  /flush_cache
// ═════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize)]
pub struct FlushCacheResponse {
    pub success: bool,
    pub message: String,
}

// ═════════════════════════════════════════════════════════════
//  /abort_request
// ═════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Deserialize)]
pub struct AbortRequest {
    /// Request ID to abort.
    pub rid: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AbortResponse {
    pub success: bool,
    pub message: String,
}
