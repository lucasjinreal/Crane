//! SSE stream builders for OpenAI-compatible and native streaming.

use std::convert::Infallible;

use axum::response::sse::Event;
use futures::stream::Stream;
use tokio::sync::mpsc;

use crate::engine::EngineResponse;
use crate::openai_api::*;
use crate::sglang_api::*;
use crate::now_epoch;

// ─────────────────────────────────────────────────────────────
//  Chat completions SSE
// ─────────────────────────────────────────────────────────────

pub fn make_chat_sse_stream(
    request_id: String,
    model_name: String,
    mut rx: mpsc::UnboundedReceiver<EngineResponse>,
    include_usage: bool,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let created = now_epoch();

    async_stream::stream! {
        // Role announcement chunk.
        let first_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk".into(),
            created,
            model: model_name.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: ChunkDelta {
                    role: Some("assistant".into()),
                    content: None,
                },
                finish_reason: None,
            }],
            usage: None,
        };
        yield Ok(Event::default().json_data(&first_chunk).unwrap());

        let mut _prompt_tokens = 0usize;
        let mut _completion_tokens = 0usize;

        while let Some(resp) = rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => {
                    _completion_tokens += 1;
                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: Some(text),
                            },
                            finish_reason: None,
                        }],
                        usage: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());
                }
                EngineResponse::Finished {
                    finish_reason,
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    ..
                } => {
                    _prompt_tokens = pt;
                    _completion_tokens = ct;

                    let chunk = ChatCompletionChunk {
                        id: request_id.clone(),
                        object: "chat.completion.chunk".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: ChunkDelta {
                                role: None,
                                content: None,
                            },
                            finish_reason: Some(finish_reason),
                        }],
                        usage: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());

                    if include_usage {
                        let usage_chunk = ChatCompletionChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk".into(),
                            created,
                            model: model_name.clone(),
                            choices: vec![],
                            usage: Some(Usage {
                                prompt_tokens: _prompt_tokens,
                                completion_tokens: _completion_tokens,
                                total_tokens: _prompt_tokens + _completion_tokens,
                            }),
                        };
                        yield Ok(Event::default().json_data(&usage_chunk).unwrap());
                    }

                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                EngineResponse::Error(e) => {
                    yield Ok(Event::default().data(format!("error: {e}")));
                    break;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Text completions SSE
// ─────────────────────────────────────────────────────────────

pub fn make_completion_sse_stream(
    request_id: String,
    model_name: String,
    mut rx: mpsc::UnboundedReceiver<EngineResponse>,
    include_usage: bool,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let created = now_epoch();

    async_stream::stream! {
        let mut _prompt_tokens = 0usize;
        let mut _completion_tokens = 0usize;

        while let Some(resp) = rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => {
                    _completion_tokens += 1;
                    let chunk = CompletionChunk {
                        id: request_id.clone(),
                        object: "text_completion".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![CompletionChunkChoice {
                            index: 0,
                            text,
                            finish_reason: None,
                        }],
                        usage: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());
                }
                EngineResponse::Finished {
                    finish_reason,
                    prompt_tokens: pt,
                    completion_tokens: ct,
                    ..
                } => {
                    _prompt_tokens = pt;
                    _completion_tokens = ct;

                    let chunk = CompletionChunk {
                        id: request_id.clone(),
                        object: "text_completion".into(),
                        created,
                        model: model_name.clone(),
                        choices: vec![CompletionChunkChoice {
                            index: 0,
                            text: String::new(),
                            finish_reason: Some(finish_reason),
                        }],
                        usage: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());

                    if include_usage {
                        let usage_chunk = CompletionChunk {
                            id: request_id.clone(),
                            object: "text_completion".into(),
                            created,
                            model: model_name.clone(),
                            choices: vec![],
                            usage: Some(Usage {
                                prompt_tokens: _prompt_tokens,
                                completion_tokens: _completion_tokens,
                                total_tokens: _prompt_tokens + _completion_tokens,
                            }),
                        };
                        yield Ok(Event::default().json_data(&usage_chunk).unwrap());
                    }

                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                EngineResponse::Error(e) => {
                    yield Ok(Event::default().data(format!("error: {e}")));
                    break;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Native /generate SSE
// ─────────────────────────────────────────────────────────────

pub fn make_generate_sse_stream(
    request_id: String,
    mut rx: mpsc::UnboundedReceiver<EngineResponse>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        while let Some(resp) = rx.recv().await {
            match resp {
                EngineResponse::Token { text, .. } => {
                    let chunk = GenerateStreamChunk {
                        text,
                        meta_info: None,
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());
                }
                EngineResponse::Finished {
                    prompt_tokens,
                    completion_tokens,
                    finish_reason,
                    ..
                } => {
                    let chunk = GenerateStreamChunk {
                        text: String::new(),
                        meta_info: Some(GenerateMetaInfo {
                            id: request_id.clone(),
                            prompt_tokens,
                            completion_tokens,
                            finish_reason,
                        }),
                    };
                    yield Ok(Event::default().json_data(&chunk).unwrap());
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
                EngineResponse::Error(e) => {
                    yield Ok(Event::default().data(format!("error: {e}")));
                    break;
                }
            }
        }
    }
}
