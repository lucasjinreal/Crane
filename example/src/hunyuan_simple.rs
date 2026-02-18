//! Simple Hunyuan Chat Example
//!
//! This example shows how to create a basic chat application using the Crane SDK.

use crane::common::config::{CommonConfig, DataType, DeviceConfig};
use crane::llm::{GenerationConfig, LlmModelType};
use crane::prelude::*;

fn main() -> CraneResult<()> {
    let config = ChatConfig {
        common: CommonConfig {
            model_path: "model/Hunyuan-0.5B-Instruct".to_string(),
            model_type: LlmModelType::HunyuanDense,
            device: DeviceConfig::Cpu,
            dtype: DataType::F32,
            max_memory: None,
        },
        generation: GenerationConfig {
            max_new_tokens: 100,
            temperature: Some(0.7),
            top_p: Some(0.9),
            eos_token_id: Some(120020),
            ..Default::default()
        },
        max_history_turns: 4,
        enable_streaming: true,
    };

    let mut chat_client = ChatClient::new(config)?;

    let response = chat_client.send_message("Hello, introduce yourself briefly.")?;
    println!("AI Response: {}", response);

    Ok(())
}
