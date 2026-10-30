use anyhow::Result;
use std::sync::mpsc;
use tokenizers::Tokenizer;

use crate::{autotokenizer::AutoTokenizer, utils::token_output_stream::TokenOutputStream};

pub trait TokenStreamer {
    /// # Errors
    ///
    /// Returns an error if decoding `token_id` or emitting the resulting
    /// text (e.g. sending it over a channel) fails.
    fn append(&mut self, token_id: u32) -> Result<()>;
    /// # Errors
    ///
    /// Returns an error if signaling completion to the stream's consumer
    /// fails.
    fn finalize(&mut self) -> Result<()>;
}

pub struct TextStreamer {
    pub tokenizer: AutoTokenizer,
    pub buffer: String,
}

impl TokenStreamer for TextStreamer {
    fn append(&mut self, token_id: u32) -> Result<()> {
        let token = self
            .tokenizer
            .decode(&[token_id], true)
            .expect("decode failed");
        self.buffer.push_str(&token);
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        println!();
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum StreamerMessage {
    Token(String), // Decoded token text
    End,
}

pub struct AsyncTextStreamer {
    decode: Box<dyn Fn(u32) -> Result<String> + Send + Sync>,
    sender: mpsc::Sender<StreamerMessage>,
}

impl AsyncTextStreamer {
    #[must_use]
    pub fn new(
        decode: Box<dyn Fn(u32) -> Result<String> + Send + Sync>,
    ) -> (Self, mpsc::Receiver<StreamerMessage>) {
        let (sender, receiver) = mpsc::channel();
        (Self { decode, sender }, receiver)
    }

    pub fn with_tokenizer<T: TokenDecode + Send + Sync + 'static>(
        tokenizer: T,
    ) -> (Self, mpsc::Receiver<StreamerMessage>) {
        let (sender, receiver) = mpsc::channel();

        let decode = Box::new(move |token_id: u32| tokenizer.decode_token(token_id));

        (Self { decode, sender }, receiver)
    }
}

pub trait TokenDecode {
    /// # Errors
    ///
    /// Returns an error if the underlying tokenizer fails to decode
    /// `token_id`.
    fn decode_token(&self, token_id: u32) -> anyhow::Result<String>;
}

// traits decode_token for relatived types.
impl TokenDecode for Tokenizer {
    fn decode_token(&self, token_id: u32) -> anyhow::Result<String> {
        self.decode(&[token_id], true)
            .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))
    }
}

impl TokenDecode for AutoTokenizer {
    fn decode_token(&self, token_id: u32) -> anyhow::Result<String> {
        self.decode(&[token_id], true)
            .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))
    }
}

impl TokenDecode for TokenOutputStream {
    fn decode_token(&self, token_id: u32) -> anyhow::Result<String> {
        self.tokenizer
            .decode(&[token_id], true)
            .map_err(|e| anyhow::anyhow!("Decode failed: {}", e))
    }
}

impl TokenStreamer for AsyncTextStreamer {
    fn append(&mut self, token_id: u32) -> Result<()> {
        let token = (self.decode)(token_id)?;

        self.sender
            .send(StreamerMessage::Token(token))
            .map_err(|e| anyhow::anyhow!("Channel send failed: {}", e))?;

        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        self.sender
            .send(StreamerMessage::End)
            .map_err(|e| anyhow::anyhow!("Failed to send end message through channel: {}", e))?;
        Ok(())
    }
}
