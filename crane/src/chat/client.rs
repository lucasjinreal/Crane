use crate::common::CraneResult;
use crate::chat::{ChatConfig, ChatHistory, ChatMessage, ChatRole};
use crate::llm::LlmClient;

/// High-level chat client for conversational AI applications
pub struct ChatClient {
    config: ChatConfig,
    history: ChatHistory,
    llm_client: LlmClient,
}

impl ChatClient {
    /// Create a new chat client with the given configuration
    pub fn new(config: ChatConfig) -> CraneResult<Self> {
        let llm_client = LlmClient::new(config.common.clone())?;
        let history = ChatHistory::new(config.max_history_turns);
        
        Ok(Self {
            config,
            history,
            llm_client,
        })
    }
    
    /// Send a message and get a response
    pub fn send_message(&mut self, message: &str) -> CraneResult<String> {
        // Add user message to history
        self.history.add_message(ChatMessage {
            role: ChatRole::User,
            content: message.to_string(),
        });
        
        let messages = self.to_core_messages();
        let response = self.llm_client.generate_chat(&messages, &self.config.generation)?;
        
        // Add assistant response to history
        self.history.add_message(ChatMessage {
            role: ChatRole::Assistant,
            content: response.clone(),
        });
        
        Ok(response)
    }
    
    /// Send a message and get a streaming response
    pub fn send_message_streaming<F>(&mut self, message: &str, callback: F) -> CraneResult<String>
    where
        F: Fn(&str),
    {
        // Add user message to history
        self.history.add_message(ChatMessage {
            role: ChatRole::User,
            content: message.to_string(),
        });
        
        let messages = self.to_core_messages();
        let response = self
            .llm_client
            .generate_chat_streaming(&messages, &self.config.generation, callback)?;
        
        // Add assistant response to history
        self.history.add_message(ChatMessage {
            role: ChatRole::Assistant,
            content: response.clone(),
        });
        
        Ok(response)
    }
    
    /// Get the current conversation history
    pub fn get_history(&self) -> &[ChatMessage] {
        &self.history.messages
    }
    
    /// Clear the conversation history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
    
    /// Format the conversation for the model
    fn to_core_messages(&self) -> Vec<crane_core::chat::Message> {
        self.history
            .messages
            .iter()
            .map(|msg| crane_core::chat::Message {
                role: match msg.role {
                    ChatRole::User => crane_core::chat::Role::User,
                    ChatRole::Assistant => crane_core::chat::Role::Assistant,
                    ChatRole::System => crane_core::chat::Role::System,
                },
                content: msg.content.clone(),
            })
            .collect()
    }
}
