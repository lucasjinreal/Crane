# Crane SDK Examples

This directory contains simple, user-friendly examples showing how to use the Crane AI SDK for various AI applications.

## Available Examples

### Chat Examples
- `chat_simple.rs`: Basic chat functionality — send a message and get a response
- `chat_streaming.rs`: Chat with real-time streaming responses — token-by-token output
- `hunyuan_simple.rs`: Hunyuan Dense model inference 

### Audio Examples
- `asr_simple.rs`: Automatic Speech Recognition — transcribe audio to text (requires ONNX feature)
- `tts_simple.rs`: TTS unified entry — auto-detects Base vs CustomVoice model
- `tts_custom_voice.rs`: TTS with predefined speakers (CustomVoice model)
- `tts_voice_clone.rs`: TTS voice cloning from reference audio (Base model)

### Vision Examples
- `vision_simple.rs`: Vision capabilities — image analysis and OCR
- `ocr_simple.rs`: PaddleOCR VL — document OCR

## Running Examples

```bash
# Simple Chat
cargo run --bin chat_simple --release

# Streaming Chat
cargo run --bin chat_streaming --release

# ASR (with ONNX support)
cargo run --features onnx --bin asr_simple --release

# Vision
cargo run --bin vision_simple --release

# OCR
cargo run --bin ocr_simple --release

# Hunyuan Dense
cargo run --bin hunyuan_simple --release

# TTS — CustomVoice (predefined speakers)
cargo run --bin tts_custom_voice --release -- vendor/Qwen3-TTS-12Hz-0.6B-CustomVoice

# TTS — Voice Clone (reference audio)
cargo run --bin tts_voice_clone --release -- vendor/Qwen3-TTS-12Hz-0.6B-Base

# TTS — Auto-detect model type
cargo run --bin tts_simple --release -- vendor/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

TTS examples write generated audio to `data/audio/output`.

## Prerequisites

Before running these examples, make sure you have:

1. Downloaded the required model checkpoints (e.g., Qwen2.5-0.5B-Instruct)
2. Updated the model paths in the examples to match your local paths
3. For ONNX examples, ensure you have the required ONNX models and run with `--features onnx`

## Getting Started

Each example is designed to be simple and self-explanatory. Start with `chat_simple.rs` to see the basic usage pattern:

```rust
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
```

The Crane SDK provides a high-level interface for various AI capabilities:
- **Chat**: Natural conversation with AI models
- **Vision**: OCR, image analysis
- **Audio**: ASR, TTS (with ONNX feature)
- **Multimodal**: Vision-language models

For API server usage (OpenAI / SGLang compatible), see [crane-oai/README.md](../crane-oai/README.md).

For more advanced usage, check the documentation in the main `crane` crate.