//! Simple Chat Example
//!
//! This example shows how to create a basic chat application using the Crane SDK.

use crane::common::config::{CommonConfig, DataType, DeviceConfig};
use crane::llm::{GenerationConfig, LlmModelType};
use crane::prelude::*;

fn main() -> CraneResult<()> {
    // Create a simple chat configuration.
    //
    // Qwen 3.5 (hybrid Gated Delta Net + attention) runs on CPU/CUDA/Metal.
    // Picks the best available target: CUDA (F16) when built `--features cuda`,
    // Metal (F16) on macOS, otherwise CPU (F32).
    #[cfg(feature = "cuda")]
    let (device, dtype) = (DeviceConfig::Cuda(0), DataType::F16);
    #[cfg(all(not(feature = "cuda"), target_os = "macos"))]
    let (device, dtype) = (DeviceConfig::Metal, DataType::F16);
    #[cfg(all(not(feature = "cuda"), not(target_os = "macos")))]
    let (device, dtype) = (DeviceConfig::Cpu, DataType::F32);

    let config = ChatConfig {
        common: CommonConfig {
            // Update this path to your local Qwen 3.5 checkpoint.
            model_path: "checkpoints/Qwen3.5-0.8B".to_string(),
            model_type: LlmModelType::Qwen35,
            device,
            dtype,
            max_memory: None,
        },
        generation: GenerationConfig {
            max_new_tokens: 128,
            temperature: Some(0.7),
            ..Default::default()
        },
        max_history_turns: 4,
        enable_streaming: true,
    };

    // Create a new chat client
    let mut chat_client = ChatClient::new(config)?;

    // Send a simple message and get a response
    let response = chat_client.send_message("Tell me a joke about Rust.")?;
    println!("AI Response: {}", response);

    Ok(())
}
