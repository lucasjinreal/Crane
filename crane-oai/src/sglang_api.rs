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
    /// Image URL for vision-language models.
    pub image_url: Option<String>,
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
    pub max_seq_len: usize,
    pub gpu_memory_limit: String,
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── SamplingParams defaults ──

    #[test]
    fn sampling_params_default_values() {
        let sp = SamplingParams::default();
        assert_eq!(sp.max_new_tokens, 128);
        assert!(sp.temperature.is_none());
        assert!(sp.top_p.is_none());
        assert!(sp.top_k.is_none());
        assert_eq!(sp.repetition_penalty, 1.0);
        assert!(sp.frequency_penalty.is_none());
        assert!(sp.presence_penalty.is_none());
        assert!(sp.stop.is_none());
        assert!(sp.stop_token_ids.is_none());
        assert!(sp.skip_special_tokens);
        assert!(sp.seed.is_none());
        assert_eq!(sp.n, 1);
    }

    // ── StringOrList ──

    #[test]
    fn string_or_list_single_into_vec() {
        let s = StringOrList::Single("stop".into());
        assert_eq!(s.into_vec(), vec!["stop".to_string()]);
    }

    #[test]
    fn string_or_list_list_into_vec() {
        let s = StringOrList::List(vec!["a".into(), "b".into()]);
        assert_eq!(s.into_vec(), vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn string_or_list_deserialize_single() {
        let v: StringOrList = serde_json::from_str(r#""stop""#).unwrap();
        assert_eq!(v.into_vec(), vec!["stop".to_string()]);
    }

    #[test]
    fn string_or_list_deserialize_list() {
        let v: StringOrList = serde_json::from_str(r#"["a","b"]"#).unwrap();
        assert_eq!(v.into_vec(), vec!["a".to_string(), "b".to_string()]);
    }

    // ── GenerateRequest deserialization ──

    #[test]
    fn generate_request_with_text() {
        let json = r#"{
            "text": "Hello world",
            "sampling_params": {"max_new_tokens": 64},
            "stream": false
        }"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.text, Some("Hello world".into()));
        assert!(req.input_ids.is_none());
        assert_eq!(req.sampling_params.max_new_tokens, 64);
        assert!(!req.stream);
    }

    #[test]
    fn generate_request_with_input_ids() {
        let json = r#"{"input_ids": [1, 2, 3]}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert!(req.text.is_none());
        assert_eq!(req.input_ids, Some(vec![1, 2, 3]));
    }

    #[test]
    fn generate_request_defaults() {
        let json = r#"{"text": "hi"}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert!(!req.stream);
        assert!(req.rid.is_none());
        // sampling_params should have defaults.
        assert_eq!(req.sampling_params.max_new_tokens, 128);
    }

    #[test]
    fn generate_request_with_rid() {
        let json = r#"{"text": "hi", "rid": "custom-123"}"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.rid, Some("custom-123".into()));
    }

    #[test]
    fn generate_request_with_stop_strings() {
        let json = r#"{
            "text": "hi",
            "sampling_params": {
                "stop": ["<|end|>", "\n\n"]
            }
        }"#;
        let req: GenerateRequest = serde_json::from_str(json).unwrap();
        let stops = req.sampling_params.stop.unwrap().into_vec();
        assert_eq!(stops, vec!["<|end|>", "\n\n"]);
    }

    // ── Response serialization ──

    #[test]
    fn generate_response_serializes() {
        let resp = GenerateResponse {
            text: "Hello!".into(),
            meta_info: GenerateMetaInfo {
                id: "gen-1".into(),
                prompt_tokens: 5,
                completion_tokens: 1,
                finish_reason: "stop".into(),
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"text\":\"Hello!\""));
        assert!(json.contains("\"prompt_tokens\":5"));
    }

    #[test]
    fn generate_stream_chunk_optional_meta() {
        let chunk = GenerateStreamChunk {
            text: "tok".into(),
            meta_info: None,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(!json.contains("meta_info"));

        let chunk_with_meta = GenerateStreamChunk {
            text: "".into(),
            meta_info: Some(GenerateMetaInfo {
                id: "g1".into(),
                prompt_tokens: 3,
                completion_tokens: 10,
                finish_reason: "length".into(),
            }),
        };
        let json = serde_json::to_string(&chunk_with_meta).unwrap();
        assert!(json.contains("\"finish_reason\":\"length\""));
    }

    #[test]
    fn model_info_response_optional_fields() {
        let resp = ModelInfoResponse {
            model_path: "/models/test".into(),
            model_type: "qwen3".into(),
            is_generation: true,
            dtype: None,
            device: None,
            max_model_len: None,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(!json.contains("dtype"));
        assert!(!json.contains("device"));
    }

    #[test]
    fn flush_cache_response_serializes() {
        let resp = FlushCacheResponse {
            success: true,
            message: "done".into(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"success\":true"));
    }

    #[test]
    fn abort_request_deserializes() {
        let json = r#"{"rid": "req-42"}"#;
        let req: AbortRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.rid, "req-42");
    }

    #[test]
    fn abort_response_serializes() {
        let resp = AbortResponse {
            success: true,
            message: "aborted".into(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"success\":true"));
    }
}
