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
    pub eos_token_id: Vec<u32>,
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
        eos_token_id: Vec<u32>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    fn make_handle() -> (EngineHandle, mpsc::UnboundedReceiver<EngineRequest>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            EngineHandle {
                request_tx: tx,
                stats: Arc::new(EngineStats::new()),
            },
            rx,
        )
    }

    #[test]
    fn submit_returns_receiver() {
        let (handle, _rx) = make_handle();
        let rx = handle.submit(
            "test-1".into(),
            vec![1, 2, 3],
            10,
            Some(0.8),
            Some(0.95),
            Some(40),
            1.0,
            vec![0],
        );
        assert!(rx.is_ok());
    }

    #[test]
    fn submit_fails_when_engine_dropped() {
        let (tx, rx) = mpsc::unbounded_channel::<EngineRequest>();
        drop(rx); // Simulate engine shutdown.
        let handle = EngineHandle {
            request_tx: tx,
            stats: Arc::new(EngineStats::new()),
        };
        let result = handle.submit(
            "test-2".into(),
            vec![1],
            10,
            None,
            None,
            None,
            1.0,
            vec![0],
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("shut down"));
    }

    #[test]
    fn active_count_reflects_stats() {
        let (handle, _rx) = make_handle();
        assert_eq!(handle.active_count(), 0);

        handle.stats.active_sequences.store(3, Ordering::Relaxed);
        handle.stats.waiting_sequences.store(2, Ordering::Relaxed);
        assert_eq!(handle.active_count(), 5);
    }

    #[test]
    fn engine_response_variants() {
        let token = EngineResponse::Token {
            text: "hello".into(),
            token_id: 42,
        };
        if let EngineResponse::Token { text, token_id } = &token {
            assert_eq!(text, "hello");
            assert_eq!(*token_id, 42);
        } else {
            panic!("Expected Token variant");
        }

        let finished = EngineResponse::Finished {
            full_text: "hello world".into(),
            prompt_tokens: 5,
            completion_tokens: 2,
            finish_reason: "stop".into(),
        };
        if let EngineResponse::Finished { prompt_tokens, completion_tokens, .. } = &finished {
            assert_eq!(*prompt_tokens, 5);
            assert_eq!(*completion_tokens, 2);
        }

        let error = EngineResponse::Error("test error".into());
        if let EngineResponse::Error(msg) = &error {
            assert_eq!(msg, "test error");
        }
    }

    #[test]
    fn engine_response_clone() {
        let resp = EngineResponse::Token {
            text: "hi".into(),
            token_id: 1,
        };
        let cloned = resp.clone();
        if let EngineResponse::Token { text, token_id } = cloned {
            assert_eq!(text, "hi");
            assert_eq!(token_id, 1);
        }
    }

    #[tokio::test]
    async fn submit_sends_request_through_channel() {
        let (tx, mut rx) = mpsc::unbounded_channel::<EngineRequest>();
        let handle = EngineHandle {
            request_tx: tx,
            stats: Arc::new(EngineStats::new()),
        };

        let _resp_rx = handle.submit(
            "req-42".into(),
            vec![10, 20, 30],
            100,
            Some(0.7),
            Some(0.9),
            None,
            1.1,
            vec![2],
        ).unwrap();

        let req = rx.recv().await.unwrap();
        assert_eq!(req.id, "req-42");
        assert_eq!(req.tokens, vec![10, 20, 30]);
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.top_p, Some(0.9));
        assert_eq!(req.top_k, None);
        assert!((req.repetition_penalty - 1.1).abs() < 0.001);
        assert_eq!(req.eos_token_id, vec![2]);
    }
}
