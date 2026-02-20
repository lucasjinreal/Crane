# crane-oai

An OpenAI & SGLang compatible inference API server built on the [Crane](../README.md) framework, with continuous batching support.

## Features

- **OpenAI-compatible API** — Chat Completions, Text Completions, Models, Tokenize/Detokenize
- **SGLang native API** — `/generate`, `/model_info`, `/server_info` and related endpoints
- **Continuous batching** — Dedicated inference thread with prefill-priority scheduling, dynamic KV memory budget, and automatic sequence eviction/recovery
- **Multi-model support** — Auto-detects and loads Hunyuan Dense, Qwen 2.5, Qwen 3 architectures
- **Streaming** — SSE (Server-Sent Events) token streaming
- **Cross-platform acceleration** — CPU / CUDA / Apple Metal, selected automatically

## Quick Start

### Build

```bash
# CPU only
cargo build -p crane-oai --release

# CUDA (NVIDIA GPU)
cargo build -p crane-oai --release --features cuda
```

### Run

```bash
# Auto-detect model type and device
crane-oai --model-path /path/to/model

# Specify model type and port
crane-oai --model-path /path/to/Qwen2.5-7B-Instruct \
    --model-type qwen25 \
    --port 8000

# GGUF weights
crane-oai --model-path /path/to/model.gguf \
    --format gguf

# Force CPU
crane-oai --model-path /path/to/model --cpu
```

## CUDA Usage

> **Note:** CUDA support requires the `cuda` feature flag at build time (see above). The server automatically uses the first available CUDA device.

### Basic CUDA inference

```bash
crane-oai --model-path /path/to/Qwen3-8B-Instruct
```

On CUDA, `model_info` will report the device as `Cuda(0)` (or `Cuda(1)`, etc.).

### GPU memory control

GPU memory grows as KV caches accumulate. Use `--gpu-memory-limit` to keep usage bounded:

```bash
# Hard cap at 8 GB — recommended starting point for a 12 GB GPU
crane-oai --model-path /path/to/model \
    --gpu-memory-limit 8G \
    --max-seq-len 4096

# Cap at 5 GB for 8 GB VRAM cards
crane-oai --model-path /path/to/model \
    --gpu-memory-limit 5G \
    --max-seq-len 2048 \
    --max-concurrent 4

# Use 75% of total VRAM
crane-oai --model-path /path/to/model \
    --gpu-memory-limit 0.75
```

When the KV memory budget is exceeded, the engine evicts the longest-output sequence (preserving its state), tightens the concurrency cap, and resumes that sequence automatically once load subsides. This avoids OOM without crashing the server.

**Recommended values by GPU size:**

| GPU VRAM | `--gpu-memory-limit` | `--max-seq-len` |
|----------|---------------------|----------------|
| 8 GB     | `6G` or `0.7`       | `2048`         |
| 12 GB    | `8G` or `0.7`       | `4096`         |
| 24 GB    | `20G` or `0.8`      | `8192`         |
| 48 GB+   | *(omit)*            | *(omit)*       |

### GGUF quantized models on CUDA

GGUF quantization roughly halves VRAM usage compared to FP16:

```bash
crane-oai --model-path /path/to/Qwen3-8B-Q4_K_M.gguf \
    --format gguf \
    --gpu-memory-limit 8G
```

### Multi-GPU note

Currently crane-oai runs on a single CUDA device (device 0). Multi-GPU tensor parallelism is not yet supported.

## CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | *(required)* | Path to model directory or GGUF file |
| `--model-type` | `auto` | Architecture: `auto`, `hunyuan`, `qwen25`, `qwen3` |
| `--model-name` | directory name | Model name shown in API responses |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8080` | Bind port |
| `--cpu` | `false` | Force CPU even when a GPU is available |
| `--max-concurrent` | `16` | Hard cap on concurrently decoding sequences. Actual concurrency may be lower when `--gpu-memory-limit` is active. |
| `--decode-tokens-per-seq` | `16` | Max decode rounds per scheduling step. Higher = less scheduling overhead, higher TTFT for queued requests. |
| `--format` | `auto` | Weight format: `auto`, `safetensors`, `gguf` |
| `--max-seq-len` | `0` | Max sequence length (prompt + generation); `0` = unlimited |
| `--gpu-memory-limit` | *(none)* | VRAM cap: absolute (`5G`, `8G`, `5120M`) or fractional (`0.7` = 70% of total) |

### Parameter tuning guide

| Goal | Recommendation |
|------|----------------|
| Constrained VRAM (≤12 GB) | Set `--gpu-memory-limit`; use `--max-concurrent 4–8` as a safety ceiling |
| Maximum throughput | Increase `--decode-tokens-per-seq` to `32` to reduce scheduling round-trips |
| Lowest time-to-first-token | Decrease `--decode-tokens-per-seq` to `4–8` so prefill slots in sooner |
| Long context generation | Set `--max-seq-len` to avoid unbounded KV growth |

## API Reference

### OpenAI-compatible

#### `POST /v1/chat/completions`

```bash
# Non-streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'

# Streaming (SSE)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

**Request fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | — | Model name |
| `messages` | array | — | `[{role, content}]` |
| `max_tokens` | int | `512` | Max tokens to generate |
| `temperature` | float | `0.8` | Sampling temperature; `0` = greedy |
| `top_p` | float | `0.95` | Nucleus sampling threshold |
| `top_k` | int | `40` | Top-k sampling |
| `repetition_penalty` | float | `1.05` | Repetition penalty |
| `stream` | bool | `false` | Enable SSE streaming |
| `stream_options` | object | — | `{"include_usage": true}` to include token counts in the final chunk |
| `seed` | int | — | Random seed for reproducibility |

#### `POST /v1/completions`

Raw text completion (no chat template applied).

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-7B-Instruct", "prompt": "The capital of France is", "max_tokens": 64}'
```

`prompt` accepts a single string or an array of strings (concatenated).

#### `GET /v1/models` · `GET /v1/models/:model_id`

List available models or fetch metadata for a specific one.

#### `POST /v1/tokenize` · `POST /v1/detokenize`

```bash
# Text → token IDs
curl http://localhost:8080/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'

# Token IDs → text
curl http://localhost:8080/v1/detokenize \
  -H "Content-Type: application/json" \
  -d '{"tokens": [9707, 1917]}'
```

### SGLang-compatible

#### `POST /generate`

```bash
curl http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The meaning of life is",
    "sampling_params": {"max_new_tokens": 128, "temperature": 0.8, "top_p": 0.95}
  }'
```

**`sampling_params` fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_new_tokens` | int | `128` | Max tokens to generate |
| `temperature` | float | `0.8` | Sampling temperature |
| `top_p` | float | `0.95` | Nucleus sampling |
| `top_k` | int | `20` | Top-k sampling |
| `repetition_penalty` | float | `1.0` | Repetition penalty |
| `stop` | string/array | — | Stop string(s) |
| `stop_token_ids` | array | — | Stop token IDs |
| `seed` | int | — | Random seed |
| `n` | int | `1` | Number of parallel completions |

#### `GET /model_info`

Returns model metadata including device (`Cuda(0)`, `Metal(0)`, `Cpu`).

#### `GET /server_info`

Returns server config and live engine stats.

```json
{
  "version": "0.1.0",
  "model_path": "/models/Qwen2.5-7B-Instruct",
  "max_concurrent": 16,
  "decode_tokens_per_seq": 16,
  "stats": {
    "total_requests": 42,
    "completed_requests": 40,
    "avg_decode_tokens_per_sec": 35.2,
    "active_sequences": 2,
    "waiting_sequences": 0
  }
}
```

#### `GET /health_generate`

Deep health check — runs a 1-token inference probe with a 30-second timeout.

#### `POST /abort_request`

Cancel an in-flight request by ID.

```bash
curl http://localhost:8080/abort_request \
  -H "Content-Type: application/json" \
  -d '{"rid": "gen-xxxx-xxxx"}'
```

### Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check, returns `{"status": "ok"}` |
| `/v1/stats` | GET | Engine stats snapshot (requests, throughput, active sequences) |
| `/flush_cache` | POST | KV cache flush (reserved for compatibility; caches are managed automatically) |

## Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",  # crane-oai does not require an API key
)

response = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Source Structure

```
crane-oai/src/
├── main.rs              # CLI entry point, AppState, route registration
├── openai_api.rs        # OpenAI request/response types
├── sglang_api.rs        # SGLang native API types
├── chat_template.rs     # Chat template rendering (Jinja / Hunyuan hard-coded)
├── handlers/
│   ├── common.rs        # /health, /v1/stats
│   ├── openai.rs        # OpenAI endpoint handlers
│   ├── sglang.rs        # SGLang endpoint handlers
│   └── sse.rs           # SSE stream builder
└── engine/
    ├── mod.rs           # InferenceEngine core loop (continuous batching)
    ├── types.rs         # EngineRequest / EngineResponse / EngineHandle
    ├── stats.rs         # Lock-free atomic counters
    ├── sampling.rs      # Token sampling (top-k/p, Gumbel-max, repetition penalty)
    ├── scheduler.rs     # Prefill-priority scheduler (dynamic effective_max_running cap)
    ├── sequence.rs      # Sequence lifecycle management
    ├── backend.rs       # ModelBackend trait and per-model implementations
    └── model_factory.rs # Model auto-detection and factory
```

## Model Backend Support

| Model | Batch decode | KV Swap | Formats | Notes |
|-------|-------------|---------|---------|-------|
| Hunyuan Dense | ✅ | ✅ | Safetensors / GGUF | KV pre-alloc, GQA 4D matmul, RoPE cache growth |
| Qwen 3 | ✅ | ✅ | Safetensors / GGUF | + QK Norm 4D, GGUF quantization |
| Qwen 2.5 | sequential | ❌ | Safetensors | — |

Model type is auto-detected from `config.json` (`model_type` / `architectures`) or can be set explicitly with `--model-type`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CRANE_FORCE_GPU_TOPK` | `0` | Force GPU top-k even for large vocabularies |
| `CRANE_TOPP_FALLBACK_TOPK` | `64` | k value for GPU top-k fallback |
| `CRANE_TOPK_SAMPLE_ON_CPU` | `0` | Sample on CPU after GPU top-k |
| `CRANE_SAMPLE_TRACE` | `0` | Verbose sampling timing logs |

## Notes

- **No API key required** — crane-oai does not authenticate requests.
- **Single CUDA device** — The server uses CUDA device 0. Multi-GPU tensor parallelism is not yet supported.
- **KV eviction is lossless** — Evicted sequences preserve their full state and resume automatically; in-flight requests are not dropped or errored.
- **`--max-seq-len 0`** means no limit. On constrained hardware, always set an explicit value to avoid runaway memory growth.
- **GGUF quantization** is supported for Hunyuan Dense and Qwen 3. Qwen 2.5 requires Safetensors format.
- **`--decode-tokens-per-seq`** controls decode rounds per engine step, not per request. Requests always complete fully regardless of this value.
- **Log diagnostics** — The startup log prints `kv_bytes` and `kv_budget`. Monitor these to validate your `--gpu-memory-limit` headroom.

## Testing

```bash
# All unit tests (125 crane-oai + 11 crane-core)
cargo test -p crane-oai
cargo test -p crane-core

# Specific modules
cargo test -p crane-oai engine::scheduler
cargo test -p crane-oai openai_api::tests
cargo test -p crane-oai sglang_api::tests
cargo test -p crane-core autotokenizer
```

## License

MIT
