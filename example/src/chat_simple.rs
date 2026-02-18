//! Simple Chat Example
//!
//! This example shows how to create a basic chat application using the Crane SDK.

use crane::prelude::*;
use crane::common::config::{CommonConfig, DataType, DeviceConfig};
use crane::llm::{GenerationConfig, LlmModelType};

fn main() -> CraneResult<()> {
    // Create a simple chat configuration
    let config = ChatConfig {
        common: CommonConfig {
            model_path: "checkpoints/Qwen2.5-0.5B-Instruct".to_string(), // Update this path to your model
            model_type: LlmModelType::Qwen25,
            device: DeviceConfig::Cpu, // Use DeviceConfig::Cuda(0) for GPU
            dtype: DataType::F16,
            max_memory: None,
        },
        generation: GenerationConfig {
            max_new_tokens: 100, // Keep responses short for demo
            temperature: Some(0.7),
            ..Default::default()
        },
        max_history_turns: 4,
        enable_streaming: true, // Enable streaming for real-time responses
    };

    // Create a new chat client
    let mut chat_client = ChatClient::new(config)?;

    // Send a simple message and get a response
    let response = chat_client.send_message("Hello, introduce yourself briefly.")?;
    println!("AI Response: {}", response);

    Ok(())
}
