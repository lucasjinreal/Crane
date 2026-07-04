# Crane AI SDK

A comprehensive Rust SDK for building AI applications with various capabilities including chat, vision, audio, and multimodal processing.

## Features

- **Chat Applications**: High-level chat interfaces with streaming support
- **Vision Processing**: OCR, image analysis, and computer vision capabilities
- **Audio Processing**: ASR (Automatic Speech Recognition), TTS (Text-to-Speech), and VAD (Voice Activity Detection)
- **Multimodal Models**: Vision-language models for combined image and text processing
- **Easy Setup**: Simple initialization for different AI applications
- **Configurable**: Flexible configuration for different use cases
- **Cross-platform**: Works across different hardware and OS

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
crane = { path = "../crane" }  # Adjust path as needed
```

## Quick Start

### Chat Application

```rust
use crane::prelude::*;

fn main() -> CraneResult<()> {
    // Create a chat configuration
    let config = ChatConfig {
        common: CommonConfig {
            model_path: "checkpoints/Qwen2.5-0.5B-Instruct".to_string(),
            device: DeviceConfig::Cpu,
            dtype: DataType::F16,
            max_memory: None,
        },
        generation: GenerationConfig {
            max_new_tokens: 235,
            temperature: Some(0.67),
            top_p: Some(1.0),
            repetition_penalty: 1.1,
            ..Default::default()
        },
        max_history_turns: 4,
        enable_streaming: true,
    };
    
    // Create a new chat client
    let mut chat_client = ChatClient::new(config)?;
    
    // Send a message and get a response
    let response = chat_client.send_message("Hello, how are you?")?;
    println!("Response: {}", response);
    
    Ok(())
}
```

### ASR (Automatic Speech Recognition)

```rust
use crane::prelude::*;

fn main() -> CraneResult<()> {
    // Create an ASR configuration
    let config = CommonConfig {
        model_path: "checkpoints/moonshine".to_string(), // Adjust path as needed
        device: DeviceConfig::Cpu,
        dtype: DataType::F16,
        max_memory: None,
    };
    
    // Create a new ASR client
    let asr_client = AsrClient::new(config)?;
    
    // Transcribe an audio file
    let transcription = asr_client.transcribe_from_file("data/audio_sample.wav")?;
    println!("Transcription: {}", transcription);
    
    Ok(())
}
```

## Modules

- `chat`: High-level chat interfaces and CLI utilities
- `vision`: OCR, image analysis, computer vision capabilities
- `audio`: ASR, TTS, voice activity detection
- `multimodal`: Vision-language models and multimodal processing
- `llm`: Generic LLM interfaces and provider abstractions
- `common`: Shared utilities, configuration, and error handling
- `engine`: Protocol-independent model runtime (continuous batching engine,
  `ModelRuntime`) shared by `crane-serve` and `crane-wyoming`; see below

## Engine Module

`engine` is Crane's protocol-independent model runtime: the continuous
batching engine, KV cache management, and model lifecycle live here so that
`crane-serve` (HTTP) and `crane-wyoming` can share the same inference core
without either transport leaking into it.

### What lives here

- **`ModelRuntime`** — owns the full model lifecycle (LLM engine, TTS models,
  VLM channels) behind a transport-agnostic API. Consumers load a model once
  and get back handles; no HTTP or Wyoming types appear here.
- **`InferenceEngine`** — the continuous batching loop for LLMs: prefill,
  batched decode, KV cache eviction/recovery, all on one dedicated thread.
- **`TtsCache`** — optional disk cache for TTS responses (behind the
  `tts-cache` feature), keyed by every input that affects the waveform
  (text, voice, language, sampling params).
- **Model factory / auto-detection** — maps a `config.json` (or path hint)
  to a concrete `ModelBackend`.

### Architecture

```
crane-serve ──┐
              ├──► ModelRuntime ──► InferenceEngine (LLM) ──► ModelBackend ──► crane-core
crane-wyoming ┘         │
                         └──► TtsHandle (dedicated thread per TTS model) ──► crane::audio::tts::Tts
```

The LLM engine loop, once started:

```text
API handlers ──(request channel)──► Engine thread
      ◄──(per-request response channel)──┘

Engine loop (each iteration = one "step"):
  1. Drain new requests from channel
  2. Detect & cancel disconnected clients
  3. Scheduler picks next batch (prefill > decode)
  4. Prefill step: run full prompt for ONE new sequence
  5. Decode step: batched or sequential forward for running sequences
  6. If idle → blocking wait for new request
```

TTS and VLM models run on their own dedicated OS thread instead, addressed
through channels — `ModelRuntime` can be shared via `Arc` across async tasks
without locking.

### Module layout

| Module          | Responsibility                                          |
|-----------------|----------------------------------------------------------|
| `runtime`       | `ModelRuntime`, `TtsHandle`, `TtsGenerateRequest` — protocol-independent model lifecycle |
| `mod` (root)    | `InferenceEngine`, `MemoryConfig` — the LLM continuous batching loop |
| `backend`       | `ModelBackend` trait + concrete implementations (Gemma4, Hunyuan Dense, Qwen 2.5/3/3.5) |
| `model_factory` | `ModelType`/`ModelFormat` auto-detection and backend/TTS factory creation |
| `scheduler`     | FIFO scheduler with prefill-priority batching             |
| `sequence`      | Per-request lifecycle state (`Sequence`, `SequenceStatus`) |
| `sampling`      | Token sampling (top-k, top-p, Gumbel-max, repetition penalty) |
| `stats`         | Lock-free counters (`EngineStats`) shared with the API layer |
| `cache`         | `TtsCache` — optional disk cache for TTS responses (behind the `tts-cache` feature) |
| `types`         | Public request/response types + `EngineHandle`             |

### Usage

```rust,ignore
use candle_core::{DType, Device};
use crane::engine::{model_factory, MemoryConfig, ModelRuntime};

let device = Device::Cpu;
let dtype = DType::F16;
let model_type = model_factory::detect_model_type(&model_path);

let mut runtime = ModelRuntime::new(
    model_name, model_type, "f16".into(), "cpu".into(), tokenizer, eos_token_id,
);

if model_type.is_tts() {
    runtime.load_tts(&model_path, &device, &dtype)?;
} else {
    let backend = model_factory::create_backend(model_type, &model_path, &device, &dtype, format)?;
    let memory_config = MemoryConfig::parse(max_seq_len, gpu_memory_limit.as_deref(), &device);
    runtime.load_llm_engine(backend, max_concurrent, decode_tokens_per_seq, memory_config);
}
```

From there, a transport layer (`crane-serve`, `crane-wyoming`) reads
`runtime.engine()` to submit generation requests, or `runtime.tts_handle(name)`
/ `runtime.generate_speech(...)` to synthesize speech. See
`crane-serve/src/lib.rs` for the reference HTTP integration.

### GPU memory control

`MemoryConfig::parse` turns `--gpu-memory-limit` (absolute, e.g. `"8G"`, or
fractional, e.g. `"0.7"`) into an absolute byte budget, recorded relative to
a post-warmup baseline. When the running KV cache would exceed the budget,
the engine evicts the longest-output sequence (preserving its state),
tightens the effective concurrency cap, and resumes the evicted sequence
automatically once load subsides — no OOM, no dropped requests.

### TTS disk cache

`TtsCache` (behind the `tts-cache` feature) is wired in at the
`ModelRuntime` level, not inside `TtsHandle`, so direct handle callers are
unaffected by whether a cache is configured. Cache keys hash every input
that affects the waveform (model, voice, language, text, sampling params)
with blake3; requests with `reference_audio` set (voice cloning) always
bypass the cache, since a reference clip can't be captured in a
fixed-size key.

### Design constraints

The engine sits directly on the hot path (per-token generation, per-step
scheduling, KV cache access). Weigh these before correctness-equivalent
alternatives when planning or reviewing a change here.

When planning a new feature, ask up front: does this copy per request? does
it add a CPU/GPU sync point? does its cost scale with batch size or with
per-request state? Design for the answer rather than retrofitting
performance later.

### Testing

```bash
cargo test -p crane engine::

# Specific modules
cargo test -p crane engine::runtime
cargo test -p crane engine::model_factory
cargo test -p crane engine::cache --features tts-cache
```

## License

MIT