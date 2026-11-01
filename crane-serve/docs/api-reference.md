# API Reference

crane-serve exposes an OpenAI-compatible API, an SGLang-compatible API, and a
small set of management endpoints. All endpoints accept and return JSON
unless noted otherwise.

## OpenAI-compatible

### `POST /v1/audio/speech`

Synthesizes speech from text using Qwen3-TTS or Voxtral TTS. Returns audio
bytes. See [Audio](audio.md) for a setup guide, voice lists, and
troubleshooting.

```bash
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS",
    "input": "Hello! This is Crane, an ultra-fast inference framework written in Rust.",
    "voice": "Chelsie",
    "language": "english"
  }' \
  --output speech.wav
```

**Request fields:**

| Field | Type | Default | Qwen3-TTS | Voxtral TTS | Description |
|-------|------|---------|-----------|-------------|-------------|
| `model` | string | — | ✅ | ✅ | Model name (e.g. `"Qwen3-TTS"`, `"voxtral"`) |
| `input` | string | — | ✅ | ✅ | Text to synthesize. UTF-8, up to a few thousand characters |
| `voice` | string | `null` | ✅ speaker name | ✅ embedding name | Qwen3: speaker from `config.json` (e.g. `"Serena"`). Voxtral: voice embedding (e.g. `"de_female"`). `null` uses the default |
| `language` | string | `"auto"` | ✅ used | accepted | Language hint: `"english"`, `"german"`, `"french"`, `"chinese"`, `"japanese"`, etc. |
| `instructions` | string | `null` | ✅ | accepted | Optional system-level prompt to guide speaking style |
| `response_format` | string | `"wav"` | ✅ | ✅ | Output format: `"wav"` or `"pcm"` (raw 16-bit LE at 24 kHz). `"mp3"`, `"opus"`, `"aac"`, `"flac"` return `400` |
| `speed` | float | `1.0` | reserved | reserved | Speaking speed multiplier. Not yet applied |
| `temperature` | float | `0.9` | ✅ used | accepted, no effect | Sampling temperature. Lower means more deterministic. Qwen3 only |
| `top_p` | float | `null` | ✅ used | accepted, no effect | Nucleus sampling threshold. `null` or `1.0` disables filtering. Qwen3 only |
| `repetition_penalty` | float | `1.05` | ✅ used | accepted, no effect | Repetition penalty for codec token generation. Qwen3 only |
| `max_tokens` | int | `8192` | ✅ | ✅ | Max codec tokens. Qwen3: ~83 ms/token at 12 Hz. Voxtral: ~80 ms/frame at 12.5 Hz |
| `reference_audio` | string | `null` | ✅ Base model only | ❌ not supported | Local path to a reference WAV for voice cloning |
| `reference_text` | string | `null` | ✅ Base model only | ❌ not supported | Transcript of the reference audio. Required with `reference_audio` |

**Response:** binary audio with `Content-Type: audio/wav` or `audio/pcm`,
depending on `response_format`.

**Approximate duration cap:** `max_tokens / 12` seconds (e.g. `8192` tokens ≈
683 seconds, `2048` ≈ 171 seconds).

### `POST /v1/audio/transcriptions`

Transcribes audio to text using Qwen3-ASR. Multipart upload, OpenAI-compatible.
See [Audio](audio.md#speech-recognition-asr) for setup.

```bash
curl http://localhost:8080/v1/audio/transcriptions \
  -F file=@speech.wav \
  -F language=english
```

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | Audio file to transcribe (required, 25 MiB limit) |
| `language` | string | Optional language hint |
| `temperature` | number | Optional sampling temperature override |

**Response:** `{"text": "..."}`.

### `GET /v1/audio/duplex` (WebSocket)

Real-time, full-duplex voice conversation. Experimental, and supported only
with MiniCPM-o-4.5. See
[Audio: real-time duplex](audio.md#real-time-duplex-audio-experimental) for
the wire protocol.

### `POST /v1/chat/completions`

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

Beyond the standard OpenAI fields, the chat endpoint accepts:

| Field | Meaning |
|:------|:--------|
| `tools` | Function specs. Passed to the model's chat template verbatim, so the prompt syntax is whatever that model was trained on. |
| `tool_choice` | `"none"` renders no tool block. Other values are accepted but advisory: forcing a particular call would need constrained decoding, which the engine does not implement. |
| `chat_template_kwargs` | Extra template variables (vLLM/SGLang convention), e.g. `{"enable_thinking": false}`. |
| `reasoning_effort` | OpenAI's top-level budget. `chat_template_kwargs` wins if both set it. |

#### Tool / function calling

Supported for any model whose chat template defines a tool protocol. The
Qwen 3.5 / 3.6 / 3.8 and Ornith families all do, and share one syntax
(`<tool_call><function=NAME><parameter=KEY>…`). crane-serve parses that back
into OpenAI's `tool_calls`, so clients need no model-specific handling.

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'
```

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "id": "call_0",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

Run the tool, then send the result back as a `tool` message. Echo the
assistant's `tool_calls` turn back too. The template re-renders it into the
transcript. Without it, the model cannot see that it already called the tool
and will call it again:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "user", "content": "What is the weather in Paris right now?"},
      {"role": "assistant", "content": null, "tool_calls": [
        {"id": "call_0", "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}
      ]},
      {"role": "tool", "tool_call_id": "call_0", "name": "get_weather",
       "content": "{\"temperature_c\": 18, \"conditions\": \"cloudy\"}"}
    ],
    "tools": [{"type": "function", "function": {"name": "get_weather",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
      "required": ["city"]}}}]
  }'
# → "The current weather in Paris is: Temperature 18°C, Conditions: Cloudy"
```

Notes:

- `arguments` is a JSON-encoded string, per the OpenAI wire format. Parse it
  client-side. Values that look like JSON scalars are recovered as such, so
  a numeric argument arrives as `{"limit": 5}` rather than `{"limit": "5"}`.
- Call `id`s are synthesized (`call_0`, `call_1`, …) because the template
  does not emit them; they are stable within one response, which is all
  `tool_call_id` correlation needs.
- **Streaming**: tool-call markup never appears in `content` deltas. A
  complete call is emitted as a single `tool_calls` delta before the
  terminal chunk, because a partially-streamed call is not something a
  client can safely run.
- If generation stops mid-call (token limit), the fragment is **discarded**
  rather than half-parsed, and `finish_reason` stays `length`.

#### Reasoning control (Qwen 3.5 / 3.6 / 3.8)

Reasoning models emit a `<think>` scratchpad. crane-serve splits it out of
`content` into `reasoning_content` (streaming too), so the answer alone is
what a client displays.

```bash
# Thinking on (the family default) — answer in content, scratchpad separate
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "What is 2+2?"}]}'
# → {"message": {"content": "4", "reasoning_content": "Two plus two is four."}}

# Shorter thinking — much lower latency
  -d '{..., "reasoning_effort": "low"}'

# Off entirely
  -d '{..., "chat_template_kwargs": {"enable_thinking": false}}'
```

`reasoning_effort` takes `low`, `medium` or `xhigh`. Qwen 3.8 defaults to
`xhigh`, the longest budget, so a modest `max_tokens` can be consumed
entirely by the scratchpad. `medium` is the neutral baseline and injects no
instruction; only `low` and `xhigh` add one.

#### Multimodal / Vision (PaddleOCR-VL-1.5)

For VLM requests, use an array in `content` to provide the image URL and the
prompt text:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "paddleocr_vl-1.5",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://i0.hdslb.com/bfs/new_dyn/1824ac967aca31d7ac9da4fdda678c4639471072.png"}},
          {"type": "text", "text": "OCR:"}
        ]
      }
    ],
    "max_tokens": 1024
  }'
```

**Request fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | — | Model name |
| `messages` | array | — | `[{role, content}]`. `content` can be a string, or an array of `{"type": "text", "text": "..."}` and `{"type": "image_url", "image_url": {"url": "..."}}` items for VLM models. |
| `max_tokens` | int | `512` | Max tokens to generate |
| `temperature` | float | `0.8` | Sampling temperature; `0` = greedy |
| `top_p` | float | `0.95` | Nucleus sampling threshold |
| `top_k` | int | `40` | Top-k sampling |
| `repetition_penalty` | float | `1.05` | Repetition penalty |
| `stream` | bool | `false` | Enable SSE streaming |
| `stream_options` | object | — | `{"include_usage": true}` to include token counts in the final chunk |
| `seed` | int | — | Random seed for reproducibility |

### `POST /v1/completions`

Raw text completion, no chat template applied.

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen2.5-7B-Instruct", "prompt": "The capital of France is", "max_tokens": 64}'
```

`prompt` accepts a single string or an array of strings (concatenated).

### `GET /v1/models` · `GET /v1/models/:model_id`

List available models or fetch metadata for a specific one.

### `POST /v1/tokenize` · `POST /v1/detokenize`

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

`POST /tokenize` and `POST /detokenize` (no `/v1` prefix) are SGLang-compat
aliases for the same two endpoints.

## SGLang-compatible

### `POST /generate`

```bash
curl http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The meaning of life is",
    "sampling_params": {"max_new_tokens": 128, "temperature": 0.8, "top_p": 0.95}
  }'
```

#### Multimodal / Vision (PaddleOCR-VL-1.5)

Include the `image_url` parameter to run a multimodal request:

```bash
curl http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "OCR:",
    "image_url": "https://i0.hdslb.com/bfs/new_dyn/1824ac967aca31d7ac9da4fdda678c4639471072.png",
    "sampling_params": {"max_new_tokens": 1024}
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

### `GET /model_info`

Returns model metadata including device (`Cuda(0)`, `Metal(0)`, `Cpu`).

### `GET /server_info`

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

### `GET /health_generate`

Deep health check. Runs a 1-token inference probe with a 30-second timeout.

### `POST /abort_request`

Cancel an in-flight request by ID.

```bash
curl http://localhost:8080/abort_request \
  -H "Content-Type: application/json" \
  -d '{"rid": "gen-xxxx-xxxx"}'
```

## Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check, returns `{"status": "ok"}` |
| `/v1/stats` | GET | Engine stats snapshot: request counts, throughput, `avg_ttft_ms` (from submit), `avg_queue_ms`, `avg_tpot_ms`, `avg_decode_batch_size`, `total_preemptions` |
| `/flush_cache` | GET, POST | KV cache flush (reserved for compatibility; caches are managed automatically) |
