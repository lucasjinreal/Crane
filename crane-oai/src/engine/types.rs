//! Public types shared between engine thread and API handlers.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use tokio::sync::mpsc;

use super::stats::EngineStats;

/// A request from an API handler to the engine.
pub struct EngineRequest {
    pub id: String,
    pub tokens: Vec<u32>,
    pub max_tokens: usize,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub eos_token_id: u32,
    pub response_tx: mpsc::UnboundedSender<EngineResponse>,
}

/// A response chunk from the engine to an API handler.
#[derive(Debug, Clone)]
pub enum EngineResponse {
    /// A newly generated text delta (for streaming).
    Token { text: String, token_id: u32 },
    /// Generation finished.
    Finished {
        full_text: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: String,
    },
    /// An error occurred.
    Error(String),
}

/// Handle returned to API handlers for sending requests.
#[derive(Clone)]
pub struct EngineHandle {
    pub(crate) request_tx: mpsc::UnboundedSender<EngineRequest>,
    pub stats: Arc<EngineStats>,
}

impl EngineHandle {
    /// Submit a new generation request. Returns a receiver for response chunks.
    pub fn submit(
        &self,
        id: String,
        tokens: Vec<u32>,
        max_tokens: usize,
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        repetition_penalty: f32,
        eos_token_id: u32,
    ) -> anyhow::Result<mpsc::UnboundedReceiver<EngineResponse>> {
        let (response_tx, response_rx) = mpsc::unbounded_channel();
        self.request_tx
            .send(EngineRequest {
                id,
                tokens,
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                eos_token_id,
                response_tx,
            })
            .map_err(|_| anyhow::anyhow!("Engine thread has shut down"))?;
        Ok(response_rx)
    }

    /// Number of active sequences (waiting + running).
    pub fn active_count(&self) -> u64 {
        self.stats.active_sequences.load(Ordering::Relaxed)
            + self.stats.waiting_sequences.load(Ordering::Relaxed)
    }
}
