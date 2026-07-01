use crate::common::config::CommonConfig;
use serde::{Deserialize, Serialize};

/// Configuration for chat functionality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatConfig {
    /// Common configuration
    pub common: CommonConfig,

    /// Generation configuration
    pub generation: crate::llm::GenerationConfig,

    /// Maximum number of conversation turns to keep in history
    pub max_history_turns: usize,

    /// Enable streaming responses
    pub enable_streaming: bool,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            common: CommonConfig::default(),
            generation: crate::llm::GenerationConfig::default(),
            max_history_turns: 10,
            enable_streaming: true,
        }
    }
}

/// Role in a chat conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChatRole {
    User,
    Assistant,
    System,
}

/// A message in a chat conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

/// Chat conversation history
#[derive(Debug, Clone)]
pub struct ChatHistory {
    pub messages: Vec<ChatMessage>,
    pub max_turns: usize,
}

impl ChatHistory {
    pub fn new(max_turns: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_turns,
        }
    }

    pub fn add_message(&mut self, message: ChatMessage) {
        self.messages.push(message);

        // Keep only the most recent messages based on max_turns
        if self.messages.len() > self.max_turns * 2 {
            // *2 because each turn has user + assistant
            self.messages
                .drain(0..(self.messages.len() - self.max_turns * 2));
        }
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }
}
