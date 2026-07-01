//! LLM functionality for the Crane SDK

mod client;
mod types;

pub use client::LlmClient;
pub use types::{GenerationConfig, LlmModelType};
