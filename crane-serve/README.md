# crane-serve

crane-serve runs AI models locally on your own machine: large language
models, vision-language models, text-to-speech, and speech recognition. No
cloud account, no per-token billing, no data leaving your hardware.

It exposes an OpenAI-compatible API and an SGLang-compatible API, so
existing OpenAI SDK clients, chat UIs, and tooling work against it
unmodified. It runs on [Crane Local AI](../README.md), a Rust inference
framework built on Candle. A continuous-batching scheduler serves multiple
requests concurrently on CPU, NVIDIA CUDA, or Apple Metal, with no Python
runtime required.

## Features

- **OpenAI-compatible API** — Chat Completions, Text Completions,
  Text-to-Speech, Speech Recognition, Models, Tokenize/Detokenize
- **SGLang native API** — `/generate`, `/model_info`, `/server_info` and
  related endpoints
- **Continuous batching** — Dedicated inference thread with prefill-priority
  scheduling, dynamic KV memory budget, and automatic sequence
  eviction/recovery
- **Multi-model support** — Auto-detects and loads Hunyuan Dense, Qwen 2.5,
  Qwen 3, Qwen 3.5 (hybrid GDN + softmax), **Bonsai 2 Ternary 27B (PTQ1_0/PQ2_0
  GGUF)**
- **Text-to-Speech & speech recognition** — Qwen3-TTS and Voxtral TTS for
  synthesis, Qwen3-ASR for transcription, all through OpenAI-compatible
  endpoints
- **Tool / function calling** — OpenAI-shaped `tools`, `tool_calls` and
  `finish_reason: "tool_calls"`, streaming included; the prompt syntax comes
  from the model's own chat template
- **Reasoning control** — `enable_thinking` / `reasoning_effort` per request,
  with the `<think>` scratchpad separated out of `content` into
  `reasoning_content`
- **Streaming** — SSE (Server-Sent Events) token streaming
- **Cross-platform acceleration** — CPU / CUDA / Apple Metal, selected
  automatically

## Quick Start

### Build

```bash
# CPU only
cargo build -p crane-serve --release

# CUDA (NVIDIA GPU)
cargo build -p crane-serve --release --features cuda
```

### Run

```bash
# Auto-detect model type and device
crane --model-path /path/to/model

# Specify model type and port
crane --model-path /path/to/Qwen2.5-7B-Instruct \
    --model-type qwen25 \
    --port 8000

# GGUF weights
crane --model-path /path/to/model.gguf \
    --format gguf

# Bonsai 2 Ternary 27B — architecture and PTQ1_0/PQ2_0 are auto-detected
./target/release/crane-serve \
    --model-path checkpoints/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    --port 8000

# Force CPU
crane --model-path /path/to/model --cpu
```

### Built-in browser UI

The API server remains headless by default. Add `--ui` to serve the built-in
React interface at the server root, without starting a separate frontend
process:

```bash
crane-serve --model-path /path/to/model --ui
```

The UI selects its view from the loaded model: ASR models show an audio upload
and transcription screen; language models show chat; vision-language models
also expose an image attachment control. The frontend source lives in `ui/`.
After changing it, rebuild its embedded production assets with
`npm run build --prefix crane-serve/ui` before building `crane-serve`.

Microphone transcription requires a secure browser origin: use
`http://localhost:<port>` when opening the UI on the server machine, or HTTPS
when accessing it through a LAN IP or hostname. Browsers intentionally block
microphone access from plain HTTP remote origins.

## Text-to-Speech and Speech Recognition

Crane Local AI supports Qwen3-TTS, Voxtral TTS, and Qwen3-ASR, all exposed
through OpenAI-compatible endpoints. See
[Audio: Text-to-Speech and Speech Recognition](docs/audio.md) for setup,
voice lists, generation parameters, and troubleshooting.

## GPU Deployment

Build with `--features cuda`, control VRAM usage, and use GGUF
quantized models on GPU: see [GPU Deployment](docs/gpu.md).

## Mixture-of-Experts (MoE) Models

Run MoE checkpoints too large to fit entirely in VRAM (e.g. Qwen3-Coder-30B-A3B)
by keeping part of the model on CPU and part on GPU: see
[Mixture-of-Experts (MoE) Models](docs/moe.md).

## CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | *(required)* | Path to model directory or GGUF file |
| `--model-type` | `auto` | Architecture: `auto`, `hunyuan`, `qwen25`, `qwen3`, `qwen3_5`, `qwen3_5_vl`, `qwen3_tts`, `voxtral_tts` (aliases: `voxtral`, `voxtral-tts`, `voxtral_4b`), `qwen3_asr` (alias: `asr`) |
| `--model-name` | directory name | Model name shown in API responses |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8080` | Bind port |
| `--unix-socket` | *(none)* | Serve over a Unix domain socket at this path instead of TCP (Unix only); a stale socket file is removed and the new one created with `0600` permissions |
| `--ui` | `false` | Serve Crane Local AI's built-in browser UI at `/` |
| `--log-level` | *(none)* | Log verbosity filter: a bare level (`debug`, `info`, `warn`) or per-target filters (`info,crane_core=debug`). Overrides `RUST_LOG` when both are set |
| `--cpu` | `false` | Force CPU even when a GPU is available |
| `--max-concurrent` / `-c` | `16` | Maximum number of requests the server generates responses for at the same time. Actual concurrency may be lower when `--gpu-memory-limit` is active. |
| `--decode-tokens-per-seq` | `16` | How many tokens to generate for one request before checking on the others waiting their turn. Higher = the GPU spends more time per switch (slightly more efficient), but a request that just arrived waits longer for its first token. Lower = requests share GPU time more evenly, so new ones start responding sooner. |
| `--format` | `auto` | Weight format: `auto`, `safetensors`, `gguf` |
| `--quant` | *(none)* | Quantize the model on load to reduce memory usage (e.g. `q4k`, `q8_0`). Qwen 3.5 safetensors only |
| `--dtype` | *(auto)* | Inference precision: `f16`, `bf16`, or `f32`. Defaults to `bf16` on NVIDIA GPUs, `f16` on AMD/Apple GPUs, `f32` on CPU |
| `--context` | *(none)* | Max context length (prompt + generation) as a human-readable size, e.g. `128K`, `1M`. Mutually exclusive with `--max-seq-len` |
| `--max-seq-len` | `0` | Max context length as a raw token count; `0` = unlimited. Use `--context` for human-readable sizes instead. If left at `0` and `--context` is also unset while `--gpu-memory-limit` is set, and the loaded model supports it (currently Qwen3), a safe value is auto-derived from measured VRAM headroom at load time — see [Auto-derived context length](docs/gpu.md#auto-derived-context-length). |
| `--gpu-memory-limit` | *(none)* | VRAM cap: absolute (`5G`, `8G`, `5120M`) or fractional (`0.7` = 70% of total) |
| `--offload-experts` | `false` | Mixture-of-Experts models only: keep every expert weight on CPU instead of measuring VRAM and promoting some to GPU. Use this if you already know your GPU has no room for any experts, to skip that measurement step at startup. See [Mixture-of-Experts (MoE) Models](docs/moe.md). |
| `--text-only` | `false` | Qwen 3.5-VL / Ornith only: opt out of the vision tower and load the same checkpoint as a plain text model (skips the ~600M-param ViT entirely — no extra VRAM — and unlocks `--quant`, which the VLM path doesn't support). Vision-capable checkpoints load with vision by default; this flag is the opt-out. |
| `--mmproj` | *(none)* | MiniCPM-V GGUF only: path to the llama.cpp `mmproj-*.gguf` vision projector paired with a GGUF `--model-path`. Unset: a sibling `mmproj*.gguf` in the same directory is used (the full `mmproj-model-f16.gguf` is preferred over `-4x`). |
| `--dspark` | *(none)* | MiniCPM-V only: enable DSpark speculative decoding with this drafter (the `MiniCPM-V-4.7-1B-DSpark` directory or a `dspark-*.gguf`). Greedy output is unchanged. An adaptive fallback decodes plainly while speculation is losing (e.g. prose on GGUF); `CRANE_DSPARK_ADAPTIVE=0` always speculates. See the main README. |
| `--llm-gguf` | *(none)* | MiniCPM-o duplex only: load the language model from a quantized GGUF file to cut its memory usage roughly in half. Other components still load from `--model-path` |
| `--api-key` | *(none)* | API key required for non-exempt endpoints; repeatable to configure multiple valid keys. Pass with no value (`--api-key`) to generate a random key printed to stdout at startup. Also settable via `CRANE_API_KEY` (always requires a value there). Unset means open access. |
| `--api-key-file` | *(none)* | File with one API key per line (`#`-prefixed lines are comments); combines with `--api-key`. Also settable via `CRANE_API_KEY_FILE`. |

### Parameter tuning guide

| Goal | Recommendation |
|------|----------------|
| Constrained VRAM (≤12 GB) | Set `--gpu-memory-limit`; use `--max-concurrent 4–8` as a safety ceiling |
| Maximum throughput | Increase `--decode-tokens-per-seq` to `32` to reduce scheduling round-trips |
| Lowest time-to-first-token | Decrease `--decode-tokens-per-seq` to `4–8` so prefill slots in sooner |
| Long context generation | Set `--context` (e.g. `128K`) to avoid unbounded KV growth |
| Large MoE model on a small GPU | See [Mixture-of-Experts (MoE) Models](docs/moe.md) |

### Authentication

Disabled by default (open access). Configure `--api-key` (repeatable) and/or
`--api-key-file` to require a valid key on every request:

```bash
crane-serve --model-path /path/to/model --api-key sk-mysecretkey

# Generate a random key at startup instead of choosing one
crane-serve --model-path /path/to/model --api-key
```

Once at least one key is configured, requests must present it via either
header:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-mysecretkey" \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-7B-Instruct", "messages": [{"role": "user", "content": "Hi"}]}'

# or
curl http://localhost:8080/v1/chat/completions \
  -H "X-Api-Key: sk-mysecretkey" \
  ...
```

A missing or invalid key returns `401` with an OpenAI-shaped error body.
`/health`, `/v1/stats`, `/` and `/ui/*` stay reachable without a key so
monitoring and the browser UI shell keep working — note the UI's own API
calls still require one, so `--ui` combined with `--api-key` isn't usable
from the browser today. Binding `--host 0.0.0.0` with an API key configured
logs a startup warning, since keys travel in cleartext without TLS in front
of the server.

## API Reference

Full endpoint documentation, request/response fields, and curl examples
for the OpenAI-compatible, SGLang-compatible, and management endpoints:
see [API Reference](docs/api-reference.md).

## Using the OpenAI SDK

crane-serve implements the OpenAI API shape, so the official `openai`
Python package works unmodified. See
[Using the OpenAI SDK](docs/openai-sdk.md) for chat, tool calling, and
TTS examples.

## Using with opencode

[opencode](https://opencode.ai/) can talk to crane-serve as a custom
OpenAI-compatible provider. crane-serve has no auth layer and ignores the
`model` field in requests (it always serves whatever was loaded via
`--model-path`/`--model-name` at startup), so any placeholder API key and
model ID work — the `models` entry below just controls what opencode shows
in its UI and what context/output limits it enforces client-side.

Start crane-serve with the [Qwen3-Coder worked example](docs/moe.md#worked-example-qwen3-coder-30b-a3b-gguf-on-a-16-gb-card), then add this to
`opencode.json` (project root) or `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "autoupdate": false,
  "share": "disabled",
  "clipboard": {
    "linux": {
      "enablePrimaryCopy": true
    }
  },
  "provider": {
    "crane": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Crane Local AI",
      "options": {
        "baseURL": "http://localhost:8080/v1",
        "apiKey": "not-needed"
      },
      "models": {
        "qwen3-coder": {
            "name": "Qwen3 Coder",
            "limit": { "context": 131072, "output": 8192 }
        }
      }
    }
  },
  "model": "crane/qwen3-coder"
}
```

- `"model": "crane/qwen3-coder"` makes this the default model opencode opens
  with, so there's no need to select it manually via `/models` each session.
- `"limit": { "context": 131072, "output": 8192 }` should match whatever
  `--context` you actually started crane-serve with. opencode's config
  wants a raw token count here, not `128K` shorthand, so convert it
  yourself: multiply by 1024 (`K` means x1024, not x1000) — `128 * 1024 =
  131072`. If you instead start crane-serve with `--context 64K`, use
  `65536` (`64 * 1024`) here. This number is only used by opencode, to know
  when to start trimming old messages from the conversation — the server
  isn't told about it, so a mismatch doesn't crash anything, it just means
  opencode trims too early or too late.
- `autoupdate`, `share`, and `clipboard` are general opencode settings
  unrelated to Crane; keep, drop, or change them independently.

Restart opencode and it opens directly on `Crane Local AI`.

## Source Structure

```
crane-serve/src/
├── main.rs              # CLI entry point, AppState, route registration
├── openai_api.rs        # OpenAI request/response types (incl. SpeechRequest)
├── sglang_api.rs        # SGLang native API types
├── chat_template.rs     # Chat template rendering (Jinja / Hunyuan hard-coded)
├── reasoning.rs         # Thinking control + <think> / content separation
├── tools.rs             # Tool-call parsing (incl. streaming filter)
├── handlers/
│   ├── common.rs        # /health, /v1/stats
│   ├── openai.rs        # OpenAI endpoint handlers
│   ├── sglang.rs        # SGLang endpoint handlers
│   ├── tts.rs           # /v1/audio/speech handler (Qwen3-TTS, Voxtral TTS)
│   ├── asr.rs           # /v1/audio/transcriptions handler (Qwen3-ASR)
│   ├── vlm.rs           # VLM handler (PaddleOCR-VL)
│   └── sse.rs           # SSE stream builder
└── engine/
    ├── mod.rs           # InferenceEngine core loop (continuous batching)
    ├── types.rs         # EngineRequest / EngineResponse / EngineHandle
    ├── stats.rs         # Lock-free atomic counters
    ├── sampling.rs      # Token sampling (top-k/p, Gumbel-max, repetition penalty)
    ├── scheduler.rs     # Prefill-priority scheduler (dynamic effective_max_running cap)
    ├── sequence.rs      # Sequence lifecycle management
    ├── backend.rs       # ModelBackend trait and per-model implementations
    └── model_factory.rs # Model auto-detection and factory (incl. Qwen3TTS, VoxtralTTS)
```

## Model Backend Support

| Model | Batch decode | KV Swap | Formats | Notes |
|-------|-------------|---------|---------|-------|
| Hunyuan Dense | ✅ | ✅ | Safetensors / GGUF | KV pre-alloc, GQA 4D matmul, RoPE cache growth |
| Qwen 3 | ✅ | ✅ | Safetensors / GGUF | + QK Norm 4D, GGUF quantization |
| Qwen 3.5 / 3.6 / 3.8 | ❌ | ❌ | Safetensors / GGUF | Hybrid GDN + softmax attention; CUDA fused recurrence kernel; `max_concurrent=1`. Qwen 3.6-27B and 3.8-27B are the same architecture scaled up and load on this path (they declare `model_type: "qwen3_5"`). Tool calling and reasoning control supported. |
| Qwen 2.5 | sequential | ❌ | Safetensors | — |
| **Qwen3-TTS** | N/A | N/A | Safetensors (ONNX fallback optional) | Dedicated thread; no continuous batching; voice cloning supported |
| **Voxtral TTS** | N/A | N/A | Safetensors | Dedicated thread; 37-codebook codec; no voice cloning |
| **Qwen3-ASR** | N/A | N/A | Safetensors | Dedicated thread; no continuous batching |

Model type is auto-detected from `config.json` / `params.json` (`model_type` / `architectures`), from a `.gguf` header's `general.architecture`, or can be set explicitly with `--model-type`.

Tool calling and reasoning control are **template-driven**, not model-type-driven: any checkpoint whose chat template defines a tool protocol and/or a `<think>` block gets them, and a template that defines neither simply ignores the corresponding request fields.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CRANE_SAMPLE_TRACE` | `0` | Verbose sampling timing logs |
| `CRANE_KV_QUANT` | unset | Qwen 3 and Qwen 3.5 family K/V cache: `int8` (~2x smaller) or `int4` (~4x smaller). Auto-derived context length conservatively assumes ~2x for both. For Qwen 3, `--kv-quant` does the same thing as a CLI flag and takes precedence over this variable. |
| `CRANE_EMBED_DENSE` | `0` | GGUF: dequantize the whole embedding table at load instead of gathering rows (pre-optimization behaviour; costs ~1.7 GiB on Qwen 3.8-27B) |
| `CRANE_PROF` | `0` | Per-forward-pass profiler: splits kernel *submission* time from wall time after a device sync |

GPU-specific environment variables (`CRANE_FORCE_GPU_TOPK`,
`CRANE_TOPP_FALLBACK_TOPK`, `CRANE_TOPK_SAMPLE_ON_CPU`) are documented in
[GPU Deployment](docs/gpu.md).

## Notes

- **API key authentication is opt-in** — unset by default (open access); see
  [Authentication](#authentication) for `--api-key`/`--api-key-file`.
- **KV eviction is lossless** — Evicted sequences preserve their full state and
  resume automatically; in-flight requests are not dropped or errored. Eviction
  only kicks in when a *new* request needs room; one long-running session's own
  growing memory use is never paused this way — see
  [Mixture-of-Experts (MoE) Models](docs/moe.md) for why `--context` is the
  real safety net there.
- **No `--context` (or `--max-seq-len`)** means no limit. On constrained
  hardware, always set an explicit value to avoid runaway memory growth, or
  rely on [auto-derived context length](docs/gpu.md#auto-derived-context-length)
  as a safety net if you don't.
- **`--decode-tokens-per-seq`** controls decode rounds per engine step, not per
  request. Requests always complete fully regardless of this value.

## Testing

```bash
# All unit tests
cargo test -p crane-serve
cargo test -p crane-core

# Specific modules
cargo test -p crane-serve engine::scheduler
cargo test -p crane-serve openai_api::tests
cargo test -p crane-serve sglang_api::tests
cargo test -p crane-serve tools          # tool-call parsing + streaming filter
cargo test -p crane-serve reasoning      # <think> separation, thinking options
cargo test -p crane-core autotokenizer
```

Tool calling and reasoning control are additionally checked against a real
checkpoint's own chat template. These need a local model, so they are
`#[ignore]`d and read it from an env var:

```bash
CRANE_QWEN38_MODEL=/path/to/Qwen3.8-27B-Q4_K_M.gguf \
  cargo test -p crane-serve --test tool_calling -- --ignored --nocapture
CRANE_QWEN38_MODEL=/path/to/Qwen3.8-27B-Q4_K_M.gguf \
  cargo test -p crane-serve --test thinking_control -- --ignored --nocapture
```

`tool_calling` renders a full agentic turn (tools out, assistant `tool_calls`
replayed, `<tool_response>` back in) through the model's template, which is
what catches a template contract change that unit tests with a mock cannot.

## License

MIT
