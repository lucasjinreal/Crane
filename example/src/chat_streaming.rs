//! Streaming Chat Example
//! 
//! This example shows how to create a chat application with real-time streaming responses.

use crane::prelude::*;
use crane::common::config::{CommonConfig, DataType, DeviceConfig};
use crane::llm::{GenerationConfig, LlmModelType};

fn main() -> CraneResult<()> {
    // Create a chat configuration
    let config = ChatConfig {
        common: CommonConfig {
            model_path: "checkpoints/Qwen2.5-0.5B-Instruct".to_string(), // Update this path to your model
            model_type: LlmModelType::Qwen25,
            device: DeviceConfig::Cpu, // Use DeviceConfig::Cuda(0) for GPU
            dtype: DataType::F16,
            max_memory: None,
        },
        generation: GenerationConfig {
            max_new_tokens: 150, // Keep responses short for demo
            temperature: Some(0.7),
            ..Default::default()
        },
        max_history_turns: 4,
        enable_streaming: true,
    };
    
    // Create a new chat client
    let mut chat_client = ChatClient::new(config)?;
    
    println!("Asking: Tell me about Rust programming in 2 sentences.");
    
    // Send a message with streaming - you'll see the response appear token by token
    let response = chat_client.send_message_streaming(
        "Tell me about Rust programming in 2 sentences.", 
        |token| print!("{}", token)  // This callback prints each token as it arrives
    )?;
    
    println!("\n\nComplete response: {}", response);
    
    Ok(())
}
