//! HTTP request handlers, organized by API family.
//!
//! | Module   | Routes                                          |
//! |----------|-------------------------------------------------|
//! | `common` | `/health`, `/v1/stats`                          |
//! | `openai` | `/v1/chat/completions`, `/v1/completions`, etc. |
//! | `sglang` | `/generate`, `/model_info`, `/server_info`, etc.|

pub mod common;
pub mod openai;
pub mod sglang;
pub mod sse;
