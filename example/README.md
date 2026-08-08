# Crane SDK Examples

This directory contains simple, user-friendly examples showing how to use the Crane AI SDK for various AI applications.

## Available Examples

### Chat Examples
- `chat_simple.rs`: Basic chat functionality — send a message and get a response
- `chat_streaming.rs`: Chat with real-time streaming responses — token-by-token output
- `chat_cli.rs`: Interactive multi-turn chat REPL — `--model-type` selects the backend (`qwen25`, `qwen3`, `qwen35` (default), `hunyuan`, `minicpm5`)
- `hunyuan_simple.rs`: Hunyuan Dense model inference
- `ornith_tools.rs`: Ornith (Qwen3.5-arch) tool-calling demo — chat-template `tool_call`/`tool` turns

### Audio Examples
- `asr_simple.rs`: Automatic Speech Recognition — transcribe audio to text (requires ONNX feature)
- `tts_simple.rs`: TTS unified entry — auto-detects Base vs CustomVoice model
- `tts_custom_voice.rs`: TTS with predefined speakers (CustomVoice model)
- `tts_voice_clone.rs`: TTS voice cloning from reference audio (Base model)
- `voxtral_tts_simple.rs`: Voxtral TTS inference
- `voxcpm2_simple.rs`: VoxCPM2 zero-shot TTS (text in, audio out; no voice cloning yet — see `AGENTS.md`)

### Vision Examples
- `vision_simple.rs`: Vision capabilities — image analysis and OCR
- `ocr_simple.rs`: PaddleOCR VL — document OCR
- `qwen3_5_vl_simple.rs`: Qwen3.5-VL multimodal inference (image + text)

### MiniCPM / MiniCPM-o Examples

MiniCPM comes in several forms in this codebase; not all of them have a dedicated example binary yet:

- **MiniCPM5-1B** (plain dense chat model): use `chat_cli.rs` with `--model-type minicpm5` (see below) — no separate binary. Both safetensors and GGUF checkpoints work (`-m` auto-detects by file extension) — a bare `.gguf` with no sibling files loads fine, since the tokenizer/chat-template are read straight from GGUF metadata. See `AGENTS.md`'s "MiniCPM5-1B" section.
- **MiniCPM-o-4.5** (full omni: vision + audio understanding, speech-token TTS, full-duplex live audio chat): `minicpmo_duplex_simple.rs` is the live full-duplex demo — connects to `crane-serve`'s `/v1/audio/duplex` WebSocket, streams mic (or `--wav` file) audio in, prints listen/speak arbitration state, plays back and saves any spoken response. It talks to a running server, so start `crane-serve --model-type minicpmo` first (see below). Turn-based (non-duplex) vision/audio understanding and TTS generation are implemented in `crane-core` but not yet exposed through a dedicated `example/` binary or the `crane` SDK traits — see `AGENTS.md`'s "MiniCPM-o-4.5 (full omni)" section for the current API surface (`MiniCpmOVlModel`, `MiniCpmTts`, `Token2Wav`, direct Rust API only).
- **MiniCPM-V-4.6** (vision-language chat): served via `crane-serve`'s OpenAI-compatible `/v1/chat/completions` with multimodal `image_url` content parts (`--model-type minicpmv4_6` or auto-detected from `config.json`) — no `example/` binary yet either.

**MiniCPM-o-4.5 GGUF options**: `crane-serve --model-type minicpmo` can load any of the four towers (LLM, audio, vision\*, TTS) from standalone GGUF files instead of the checkpoint's safetensors, via `--llm-gguf <path>` (currently the only tower exposed as a server CLI flag — audio/TTS GGUF loaders exist in `crane-core` but aren't wired to `crane-serve`'s CLI yet). The upstream LLM GGUF (e.g. `MiniCPM-o-4_5-Q8_0.gguf`) is genuinely quantized and meaningfully cuts VRAM (~19.8GB → ~11.2GB total); the audio/vision/TTS GGUF releases are F16/F32-only re-exports with no memory benefit, useful only for pure-GGUF deployment convenience. See `AGENTS.md` for full detail.

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

# MiniCPM5-1B chat (interactive REPL) — safetensors dir or a bare .gguf file both work
cargo run --bin chat_cli --release -- -m /path/to/MiniCPM5-1B --model-type minicpm5
cargo run --bin chat_cli --release -- -m /path/to/MiniCPM5-1B-Q8_0.gguf --model-type minicpm5

# MiniCPM-o-4.5 full-duplex live audio chat — start the server first...
cargo run -p crane-serve --features cuda --release --bin crane-serve -- -m /path/to/MiniCPM-o-4_5 --model-type minicpmo
# ...then, in another terminal, either talk live via the mic:
cargo run --bin minicpmo_duplex_simple --release
# ...or feed a WAV file headlessly:
cargo run --bin minicpmo_duplex_simple --release -- --wav path/to/clip.wav --no-playback --output-wav /tmp/response.wav
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

For API server usage (OpenAI / SGLang compatible), see [crane-serve/README.md](../crane-serve/README.md).

For more advanced usage, check the documentation in the main `crane` crate.