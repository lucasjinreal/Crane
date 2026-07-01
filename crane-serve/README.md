# crane-serve

An OpenAI & SGLang compatible inference API server built on the [Crane](../README.md) framework, with continuous batching support.

## Features

- **OpenAI-compatible API** — Chat Completions, Text Completions, Text-to-Speech, Models, Tokenize/Detokenize
- **SGLang native API** — `/generate`, `/model_info`, `/server_info` and related endpoints
- **Continuous batching** — Dedicated inference thread with prefill-priority scheduling, dynamic KV memory budget, and automatic sequence eviction/recovery
- **Multi-model support** — Auto-detects and loads Hunyuan Dense, Qwen 2.5, Qwen 3, Qwen 3.5 (hybrid GDN + softmax), Qwen3-TTS, Voxtral TTS
- **Qwen3-TTS** — Full two-level TTS inference (Talker + Code Predictor) with native Candle speech-tokenizer decoder (ONNX optional fallback); exposes OpenAI-compatible `/v1/audio/speech`
- **Voxtral TTS** — 4B-parameter Mistral-based TTS with 20 multilingual voice embeddings, flow-matching acoustic model, and codec decoder; exposes OpenAI-compatible `/v1/audio/speech`
- **Streaming** — SSE (Server-Sent Events) token streaming
- **Cross-platform acceleration** — CPU / CUDA / Apple Metal, selected automatically

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

# Force CPU
crane --model-path /path/to/model --cpu
```

### Qwen3-TTS Quick Start

> Native speech-tokenizer decoding is enabled by default. ONNX export is optional as a compatibility fallback.

**Step 1 — Download the model**

```bash
mkdir -p checkpoints/
# Base model (multi-speaker, 12 Hz)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --local-dir checkpoints/Qwen3-TTS-12Hz-0.6B-Base

# Custom-voice model (add your own speaker embedding)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --local-dir checkpoints/Qwen3-TTS-12Hz-0.6B-CustomVoice
```

**Step 2 — (Optional) Export the ONNX speech-tokenizer decoder fallback**

```bash
# Install the Qwen3-TTS Python package first (needed for export only)
pip install -e vendor/Qwen3-TTS

python scripts/export_qwen_tts_tokenizer_onnx.py \
    checkpoints/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer \
    checkpoints/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer/speech_tokenizer_decoder.onnx
```

**Step 3 — Build and start the server**

```bash
# CPU (always available)
cargo build -p crane-serve --release

# CUDA
cargo build -p crane-serve --release --features "cuda"

# macOS Metal (auto-enabled by target)
cargo build -p crane-serve --release

# Start the TTS server (auto-detected from config.json)
./target/release/crane \
    --model-path checkpoints/Qwen3-TTS-12Hz-0.6B-Base \
    --model-type qwen3_tts \
    --port 8080
```

**Step 4 — Synthesize speech**

```bash
# Generate WAV audio from text (CustomVoice model — predefined speakers)
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS",
    "input": "今天天气真好，我们去公园吧！",
    "voice": "Serena",
    "language": "chinese"
  }' \
  --output speech.wav

# English
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS",
    "input": "Hello, this is a test of the Qwen3-TTS model.",
    "voice": "Ryan",
    "language": "english"
  }' \
  --output speech.wav

# Voice cloning (Base model — reference audio + transcript)
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS",
    "input": "そんな何もない今日が 少しだけでもいい日になったと思えたら",
    "language": "japanese",
    "reference_audio": "/path/to/reference.wav",
    "reference_text": "Reference audio transcript goes here"
  }' \
  --output voice_clone.wav
```

### Voxtral TTS Quick Start

**Step 1 — Download the model**

```bash
mkdir -p checkpoints/
hf download mistralai/Voxtral-4B-TTS-2603 \
    --local-dir checkpoints/Voxtral-4B-TTS-2603
```

**Step 2 — Build and start the server**

```bash
cargo build -p crane-serve --release

./target/release/crane-serve \
    --model-path checkpoints/Voxtral-4B-TTS-2603 \
    --model-type voxtral_tts \
    --port 8080
```

The model type is auto-detected from `params.json` or from the path containing `"voxtral"` — `--model-type voxtral_tts` is optional.

**Step 3 — Synthesize speech**

```bash
# English (neutral female voice)
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxtral",
    "input": "Hello, how are you today?",
    "voice": "neutral_female",
    "response_format": "wav"
  }' \
  --output speech.wav

# German female voice
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxtral",
    "input": "Hallo, wie geht es dir heute?",
    "voice": "de_female",
    "language": "german",
    "response_format": "wav"
  }' \
  --output speech.wav

# French male voice
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxtral",
    "input": "Bonjour, comment allez-vous?",
    "voice": "fr_male",
    "language": "french",
    "response_format": "wav"
  }' \
  --output speech.wav
```

## Text-to-Speech (TTS)

Crane supports two TTS model families. Both expose the same `POST /v1/audio/speech` endpoint and share the same request format.

### Voxtral TTS

Crane implements a **Voxtral TTS** inference pipeline in pure Rust + Candle.

#### Architecture overview

```
Text input
  │
  ▼
Tekkenizer (tekken.json)
  │
  ▼
┌──────────────────────────────────────────────────┐
│  LLM Backbone  (26-layer Mistral, GQA, RoPE)    │
│  - Embeds text tokens + voice embedding          │
│  - Autoregressively generates semantic code      │
└──────────────────────────────────────────────────┘
  │  hidden state per frame
  ▼
┌──────────────────────────────────────────────────┐
│  Acoustic Transformer  (3-layer, flow matching)  │
│  - Predicts 36 acoustic codes from hidden state  │
└──────────────────────────────────────────────────┘
  │  [T, 37] codes (1 semantic + 36 acoustic)
  ▼
┌──────────────────────────────────────────────────┐
│  Codec Decoder  (4-stage conv + transformer)     │
│  - Converts [1, T, 37] codes → [1, 1, S] audio  │
└──────────────────────────────────────────────────┘
  │
  ▼
WAV audio (24 kHz, 16-bit PCM)
```

#### Setup

**1. Download:**

```bash
huggingface-cli download mistralai/Voxtral-4B-TTS-2603 \
    --local-dir checkpoints/Voxtral-4B-TTS-2603
```

**2. Build:**

```bash
# CPU
cargo build -p crane-serve --release

# CUDA
cargo build -p crane-serve --release --features "cuda"
```

**3. Start the server:**

```bash
./target/release/crane-serve \
    --model-path checkpoints/Voxtral-4B-TTS-2603 \
    --model-type voxtral_tts \
    --port 8080
```

#### Available voices

Voice embeddings are stored as `.pt` files in the `voice_embedding/` subdirectory of the checkpoint. Pass the filename without extension as the `voice` field.

| Voice | Gender | Language / Style |
|-------|--------|-----------------|
| `neutral_female` | Female | English (neutral) |
| `neutral_male` | Male | English (neutral) |
| `casual_female` | Female | English (casual) |
| `casual_male` | Male | English (casual) |
| `cheerful_female` | Female | English (cheerful) |
| `de_female` | Female | German |
| `de_male` | Male | German |
| `fr_female` | Female | French |
| `fr_male` | Male | French |
| `es_female` | Female | Spanish |
| `es_male` | Male | Spanish |
| `it_female` | Female | Italian |
| `it_male` | Male | Italian |
| `pt_female` | Female | Portuguese |
| `pt_male` | Male | Portuguese |
| `nl_female` | Female | Dutch |
| `nl_male` | Male | Dutch |
| `hi_female` | Female | Hindi |
| `hi_male` | Male | Hindi |
| `ar_male` | Male | Arabic |

If `voice` is omitted, the first loaded voice is used. The list of available voices can differ between checkpoint releases — check `voice_embedding/` in your checkpoint directory.

#### Notes

- Voxtral uses **greedy decoding**. The `temperature`, `top_p`, and `repetition_penalty` fields are accepted for API compatibility but do not affect the output.
- Voxtral **does not support voice cloning** (`reference_audio` / `reference_text` return an error).
- On CPU with F32 weights, generation is slow (~14 transformer passes per audio frame). Use CUDA or Metal for practical inference.

### Qwen3-TTS

Crane implements a full **Qwen3-TTS** inference pipeline in pure Rust + Candle, making it the first non-Python framework to support this model.

#### Architecture overview

```
Text input
  │
  ▼
Tokenizer (Qwen chat tokenizer: tokenizer.json OR vocab.json + merges.txt)
  │
  ▼
┌──────────────────────────────────────────────────┐
│  Talker  (28-layer transformer, MRoPE)           │
│  - Embeds text via text_projection MLP           │
│  - Autoregressively generates codec group 0      │
└──────────────────────────────────────────────────┘
  │  hidden state + group-0 token
  ▼
┌──────────────────────────────────────────────────┐
│  Code Predictor  (5-layer transformer)           │
│  - Predicts codec groups 1–15 from hidden state  │
└──────────────────────────────────────────────────┘
  │  [T, 16] codec tokens
  ▼
┌──────────────────────────────────────────────────┐
│  Speech Tokenizer Decoder  (native Candle)       │
│  - 12 Hz, 24 kHz, 16 RVQ quantizers             │
│  - Converts [1, 16, T] tokens → [1, 1, S] audio │
└──────────────────────────────────────────────────┘
  │
  ▼
WAV audio (24 kHz, 16-bit PCM)
```

#### Setup

**1. (Optional) Export the speech tokenizer ONNX fallback** (one-time step):

```bash
# Install required Python package (for export only — not needed at runtime)
pip install -e vendor/Qwen3-TTS

# Base model
python scripts/export_qwen_tts_tokenizer_onnx.py \
    checkpoints/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer \
    checkpoints/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer/speech_tokenizer_decoder.onnx

# CustomVoice model
python scripts/export_qwen_tts_tokenizer_onnx.py \
    checkpoints/Qwen3-TTS-12Hz-0.6B-CustomVoice/speech_tokenizer \
    checkpoints/Qwen3-TTS-12Hz-0.6B-CustomVoice/speech_tokenizer/speech_tokenizer_decoder.onnx
```

**2. Build:**

```bash
# CPU
cargo build -p crane-serve --release

# CUDA
cargo build -p crane-serve --release --features "cuda"

# macOS Metal (auto-enabled by target)
cargo build -p crane-serve --release
```

**3. Start the server:**

```bash
./target/release/crane \
    --model-path checkpoints/Qwen3-TTS-12Hz-0.6B-Base \
    --model-type qwen3_tts \
    --port 8080
```

The model type is auto-detected from `config.json` — `--model-type qwen3_tts` is optional if the path contains `Qwen3-TTS`.

#### Model variants

| Variant | HuggingFace ID | Description | Voice selection |
|---------|---------------|-------------|-----------------|
| CustomVoice | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Predefined speakers | `voice` field (e.g. "Serena") |
| Base | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Voice cloning via reference audio | `reference_audio` + `reference_text` |

#### Voice cloning (Base model)

The Base model supports in-context learning (ICL) voice cloning: given a reference audio clip and its transcript, the model synthesizes new text in the same voice.

```bash
# Voice-clone a Japanese sentence using a reference audio clip
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS",
    "input": "こうして君に直接ありがとうを言える時間をくれたこと それが多分一番私は嬉しい",
    "language": "japanese",
    "reference_audio": "data/audio/kinsenka_3.wav",
    "reference_text": "こうして君に直接ありがとうを言える時間をくれたこと それが多分一番私は嬉しい"
  }' \
  --output voice_clone.wav
```

**Voice-clone fields:**

| Field | Type | Description |
|-------|------|-------------|
| `reference_audio` | string | Local file path to reference WAV audio |
| `reference_text` | string | Transcript of the reference audio (required) |

> **Note:** When `reference_audio` is set, the model runs in voice-clone mode regardless of the `voice` field. The reference audio must be a WAV file. The `language` field should match the target text language.

#### Speaker list (Qwen3-TTS)

The available speakers are defined in `config.json` under `talker_config.spk_id`. Open the file to see all names:

```bash
python3 -c "
import json, sys
c = json.load(open('checkpoints/Qwen3-TTS-12Hz-0.6B-Base/config.json'))
for name in sorted(c['talker_config']['spk_id']):
    print(name)
"
```

Built-in speakers for the CustomVoice model include: `Serena`, `Vivian`, `Uncle_fu`, `Ryan`, `Aiden`, `Ono_anna`, `Sohee`, `Eric`, `Dylan`. Larger model variants may include additional speakers. Check your model's `config.json` for the full list.

### Generation Parameters

Parameters that control sampling during TTS generation. Voxtral TTS uses greedy decoding and ignores the sampling fields.

| Parameter | Default | Qwen3-TTS | Voxtral TTS | Notes |
|-----------|---------|-----------|-------------|-------|
| `temperature` | `0.9` | ✅ used | accepted (no effect) | Lower = more stable prosody |
| `top_p` | `null` | ✅ used | accepted (no effect) | Nucleus sampling; `null` or `1.0` matches reference defaults |
| `repetition_penalty` | `1.05` | ✅ used | accepted (no effect) | Helps avoid repeating codec tokens |
| `max_tokens` | `8192` | ✅ used | ✅ used | Qwen3: ~83 ms/token at 12 Hz; Voxtral: ~80 ms/frame at 12.5 Hz |
| `language` | `"auto"` | ✅ used | accepted (no effect) | Wrong language hint degrades Qwen3 quality |

### Troubleshooting

**`TTS model not loaded`** — The server was started with an LLM model. Restart with `--model-type qwen3_tts` or `--model-type voxtral_tts`.

**Garbled audio / silence (Qwen3-TTS)** — Try lowering `temperature` to `0.5`–`0.7` and ensure `language` matches the input text.

**`native speech tokenizer load failed` (Qwen3-TTS)** — Check `<model_dir>/speech_tokenizer/config.json` and safetensors files are complete. For older checkpoints that miss `quantizer.*._codebook.cluster_usage`, Crane now auto-falls back to a ones vector during load; if loading still fails, optionally export ONNX as a fallback decoder.

**`Speech tokenizer ONNX not found` (Qwen3-TTS fallback path only)** — This appears only when native decoder load fails and ONNX fallback is requested; run export script and place ONNX at `<model_dir>/speech_tokenizer/speech_tokenizer_decoder.onnx`.

**`Voxtral TTS does not support voice cloning`** — The `reference_audio` field is not supported by Voxtral TTS. Use Qwen3-TTS Base for voice cloning.

**Slow generation on CPU (Voxtral)** — Voxtral requires ~14 transformer passes per audio frame. On CPU with F32 weights, short phrases can take several minutes. Use CUDA or Metal.

## Qwen3-TTS

Crane implements a full **Qwen3-TTS** inference pipeline in pure Rust + Candle, making it the first non-Python framework to support this model.

### Setup

**1. (Optional) Export the speech tokenizer ONNX fallback** (one-time step):

```bash
# Install required Python package (for export only — not needed at runtime)
pip install -e vendor/Qwen3-TTS

# Base model
python scripts/export_qwen_tts_tokenizer_onnx.py \
    checkpoints/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer \
    checkpoints/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer/speech_tokenizer_decoder.onnx

# CustomVoice model
python scripts/export_qwen_tts_tokenizer_onnx.py \
    checkpoints/Qwen3-TTS-12Hz-0.6B-CustomVoice/speech_tokenizer \
    checkpoints/Qwen3-TTS-12Hz-0.6B-CustomVoice/speech_tokenizer/speech_tokenizer_decoder.onnx
```

**2. Build:**

```bash
# CPU
cargo build -p crane-serve --release

# CUDA
cargo build -p crane-serve --release --features "cuda"

# macOS Metal (auto-enabled by target)
cargo build -p crane-serve --release
```

**3. Start the server:**

```bash
./target/release/crane \
    --model-path checkpoints/Qwen3-TTS-12Hz-0.6B-Base \
    --model-type qwen3_tts \
    --port 8080
```

The model type is auto-detected from `config.json` — `--model-type qwen3_tts` is optional if the path contains `Qwen3-TTS`.

### Model variants

| Variant | HuggingFace ID | Description | Voice selection |
|---------|---------------|-------------|-----------------|
| CustomVoice | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Predefined speakers | `voice` field (e.g. "Serena") |
| Base | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Voice cloning via reference audio | `reference_audio` + `reference_text` |

#### Voice cloning (Base model)

The Base model supports in-context learning (ICL) voice cloning: given a reference audio clip and its transcript, the model synthesizes new text in the same voice.

```bash
# Voice-clone a Japanese sentence using a reference audio clip
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS",
    "input": "こうして君に直接ありがとうを言える時間をくれたこと それが多分一番私は嬉しい",
    "language": "japanese",
    "reference_audio": "data/audio/kinsenka_3.wav",
    "reference_text": "こうして君に直接ありがとうを言える時間をくれたこと それが多分一番私は嬉しい"
  }' \
  --output voice_clone.wav
```

**Voice-clone fields:**

| Field | Type | Description |
|-------|------|-------------|
| `reference_audio` | string | Local file path to reference WAV audio |
| `reference_text` | string | Transcript of the reference audio (required) |

> **Note:** When `reference_audio` is set, the model runs in voice-clone mode regardless of the `voice` field. The reference audio must be a WAV file. The `language` field should match the target text language.

### Speaker list (Base model)

The available speakers are defined in `config.json` under `talker_config.spk_id`. Open the file to see all names:

```bash
python3 -c "
import json, sys
c = json.load(open('checkpoints/Qwen3-TTS-12Hz-0.6B-Base/config.json'))
for name in sorted(c['talker_config']['spk_id']):
    print(name)
"
```

Built-in speakers for the CustomVoice model include: `Serena`, `Vivian`, `Uncle_fu`, `Ryan`, `Aiden`, `Ono_anna`, `Sohee`, `Eric`, `Dylan`. Larger model variants may include additional speakers. Check your model's `config.json` for the full list.

### Generation parameters

| Parameter | Default | Recommended | Notes |
|-----------|---------|-------------|-------|
| `temperature` | `0.9` | `0.7`–`0.9` | Lower = more stable prosody; higher = more expressive |
| `max_tokens` | `8192` | `2048`–`8192` | 12 tokens ≈ 1 second of audio at 12 Hz |
| `repetition_penalty` | `1.05` | `1.0`–`1.1` | Helps avoid repeating codec tokens |
| `top_p` | `null` | `null` or `1.0` | Nucleus sampling; `1.0` matches reference defaults |
| `language` | `auto` | match input | Wrong language hint degrades quality |

### Troubleshooting

**`native speech tokenizer load failed`** — Check `<model_dir>/speech_tokenizer/config.json` and safetensors files are complete. For older checkpoints that miss `quantizer.*._codebook.cluster_usage`, Crane now auto-falls back to a ones vector during load; if loading still fails, optionally export ONNX as a fallback decoder.

**`TTS model not loaded`** — The server was started with an LLM model (not TTS). Restart with `--model-type qwen3_tts`.

**Garbled audio / silence** — Try lowering `temperature` to `0.5`–`0.7` and ensure `language` matches the input text.

**`Speech tokenizer ONNX not found` (fallback path only)** — This appears only when native decoder load fails and ONNX fallback is requested; run export script and place ONNX at `<model_dir>/speech_tokenizer/speech_tokenizer_decoder.onnx`.

## CUDA Usage

> **Note:** CUDA support requires the `cuda` feature flag at build time (see above). The server automatically uses the first available CUDA device.

### Basic CUDA inference

```bash
crane-serve --model-path /path/to/Qwen3-8B-Instruct
```

On CUDA, `model_info` will report the device as `Cuda(0)` (or `Cuda(1)`, etc.).

### GPU memory control

GPU memory grows as KV caches accumulate. Use `--gpu-memory-limit` to keep usage bounded:

```bash
# Hard cap at 8 GB — recommended starting point for a 12 GB GPU
crane-serve --model-path /path/to/model \
    --gpu-memory-limit 8G \
    --max-seq-len 4096

# Cap at 5 GB for 8 GB VRAM cards
crane-serve --model-path /path/to/model \
    --gpu-memory-limit 5G \
    --max-seq-len 2048 \
    --max-concurrent 4

# Use 75% of total VRAM
crane-serve --model-path /path/to/model \
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
crane-serve --model-path /path/to/Qwen3-8B-Q4_K_M.gguf \
    --format gguf \
    --gpu-memory-limit 8G
```

### Multi-GPU note

Currently crane-serve runs on a single CUDA device (device 0). Multi-GPU tensor parallelism is not yet supported.

## CLI Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | *(required)* | Path to model directory or GGUF file |
| `--model-type` | `auto` | Architecture: `auto`, `hunyuan`, `qwen25`, `qwen3`, `qwen3_5`, `qwen3_tts`, `voxtral_tts` (aliases: `voxtral`, `voxtral-tts`, `voxtral_4b`) |
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

#### `POST /v1/audio/speech` — Text-to-Speech

Synthesizes speech from text using either Qwen3-TTS or Voxtral TTS. Returns a WAV audio file (or raw PCM, depending on `response_format`).

```bash
# Voxtral TTS — German female voice
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxtral",
    "input": "Guten Morgen, wie geht es Ihnen?",
    "voice": "de_female",
    "language": "german",
    "response_format": "wav"
  }' \
  --output speech.wav

# Qwen3-TTS — Chinese
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS",
    "input": "今天天气真好，我们去公园吧。",
    "voice": "Chelsie",
    "language": "chinese"
  }' \
  --output speech.wav

# Qwen3-TTS — English with higher temperature
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS",
    "input": "Hello! This is Crane, an ultra-fast inference framework written in Rust.",
    "voice": "Chelsie",
    "language": "english",
    "temperature": 0.8,
    "max_tokens": 2048
  }' \
  --output hello.wav

# Qwen3-TTS — Voice cloning (Base model)
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS",
    "input": "そんな何もない今日が 少しだけでもいい日になったと思えたら",
    "language": "japanese",
    "reference_audio": "data/audio/kinsenka_3.wav",
    "reference_text": "こうして君に直接ありがとうを言える時間をくれたこと それが多分一番私は嬉しい"
  }' \
  --output voice_clone.wav

# Auto-detect language
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS",
    "input": "This is a bilingual test. 这是一个中英文混合测试。",
    "language": "auto"
  }' \
  --output bilingual.wav
```

**Request fields:**

| Field | Type | Default | Qwen3-TTS | Voxtral TTS | Description |
|-------|------|---------|-----------|-------------|-------------|
| `model` | string | — | ✅ | ✅ | Model name (e.g. `"Qwen3-TTS"`, `"voxtral"`) |
| `input` | string | — | ✅ | ✅ | The text to synthesize. UTF-8, up to a few thousand characters |
| `voice` | string | `null` | ✅ speaker name | ✅ embedding name | Qwen3: speaker from `config.json` (e.g. `"Serena"`). Voxtral: voice embedding (e.g. `"de_female"`). `null` uses default |
| `language` | string | `"auto"` | ✅ used | accepted | Language hint: `"english"`, `"german"`, `"french"`, `"chinese"`, `"japanese"`, etc. |
| `instructions` | string | `null` | ✅ | accepted | Optional system-level prompt to guide speaking style |
| `response_format` | string | `"wav"` | ✅ | ✅ | Output audio format: `"wav"` or `"pcm"` (raw 16-bit LE at 24 kHz). `"mp3"`, `"opus"`, `"aac"`, `"flac"` return `400` |
| `speed` | float | `1.0` | reserved | reserved | Speaking speed multiplier (reserved, not yet applied) |
| `temperature` | float | `0.9` | ✅ used | accepted (no effect) | Sampling temperature. Lower = more deterministic (Qwen3 only) |
| `top_p` | float | `null` | ✅ used | accepted (no effect) | Nucleus sampling threshold; `null` or `1.0` disables filtering (Qwen3 only) |
| `repetition_penalty` | float | `1.05` | ✅ used | accepted (no effect) | Repetition penalty for codec token generation (Qwen3 only) |
| `max_tokens` | int | `8192` | ✅ | ✅ | Max codec tokens. Qwen3: ~83 ms/token at 12 Hz. Voxtral: ~80 ms/frame at 12.5 Hz |
| `reference_audio` | string | `null` | ✅ Base model only | ❌ not supported | Local path to reference WAV for voice cloning |
| `reference_text` | string | `null` | ✅ Base model only | ❌ not supported | Transcript of the reference audio (required with `reference_audio`) |

**Response:** Binary audio bytes with `Content-Type: audio/wav` (`response_format="wav"`) or `audio/pcm` (`response_format="pcm"`).

**Approximate duration cap:** `max_tokens / 12` seconds (e.g. `8192` tokens ≈ 683 seconds, `2048` ≈ 171 seconds).

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

#### Multimodal / Vision (PaddleOCR-VL-1.5)

For VLM requests, use an array in `content` to provide the image URL and the prompt text:

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

#### Multimodal / Vision (PaddleOCR-VL-1.5)

To run a multimodal inference request with a VLM (PaddleOCR-VL), include the `image_url` parameter:

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

### Chat (LLM)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",  # crane-serve does not require an API key
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

### Text-to-Speech

#### Voxtral TTS

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

# German female voice
response = client.audio.speech.create(
    model="voxtral",
    voice="de_female",
    input="Hallo, wie geht es Ihnen heute?",
    extra_body={"language": "german", "response_format": "wav"},
)
response.stream_to_file("output.wav")

# English neutral voice
response = client.audio.speech.create(
    model="voxtral",
    voice="neutral_female",
    input="Hello, this is a test of the Voxtral TTS engine.",
    extra_body={"response_format": "wav"},
)
response.stream_to_file("english.wav")
```

#### Qwen3-TTS

> Start crane-serve with `--model-type qwen3_tts` and a Qwen3-TTS checkpoint.

**CustomVoice model (predefined speakers):**

```python
from openai import OpenAI
import pathlib

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

# --- Basic synthesis (returns WAV bytes) ---
response = client.audio.speech.create(
    model="Qwen3-TTS",
    voice="Chelsie",
    input="今天天气真好，我们去公园吧！",
    extra_body={"language": "chinese"},
)
response.stream_to_file("output.wav")

# --- English ---
response = client.audio.speech.create(
    model="Qwen3-TTS",
    voice="Ethan",
    input="Hello, this is a test of the Crane TTS engine.",
    extra_body={"language": "english", "temperature": 0.7},
)
response.stream_to_file("english.wav")
```

**Base model (voice cloning):**

```python
from openai import OpenAI
import pathlib

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
)

# --- Voice clone: synthesize new text in the reference speaker's voice ---
response = client.audio.speech.create(
    model="Qwen3-TTS",
    voice="clone",  # voice field is ignored in voice-clone mode
    input="そんな何もない今日が 少しだけでもいい日になったと思えたら",
    extra_body={
        "language": "japanese",
        "reference_audio": "data/audio/kinsenka_3.wav",
        "reference_text": "こうして君に直接ありがとうを言える時間をくれたこと それが多分一番私は嬉しい",
    },
)
response.stream_to_file("voice_clone.wav")
```

**Low-level requests (requests library):**

```python
import requests, pathlib

# Voxtral TTS
r = requests.post(
    "http://localhost:8080/v1/audio/speech",
    json={
        "model": "voxtral",
        "input": "Hello, how are you today?",
        "voice": "neutral_female",
        "response_format": "wav",
    },
)
r.raise_for_status()
pathlib.Path("speech.wav").write_bytes(r.content)

# Qwen3-TTS CustomVoice
r = requests.post(
    "http://localhost:8080/v1/audio/speech",
    json={
        "model": "Qwen3-TTS",
        "input": "今天天气真好，我们去公园吧！",
        "voice": "Chelsie",
        "language": "chinese",
        "temperature": 0.7,
        "max_tokens": 2048,
    },
)
r.raise_for_status()
pathlib.Path("speech.wav").write_bytes(r.content)

# Qwen3-TTS voice clone
r = requests.post(
    "http://localhost:8080/v1/audio/speech",
    json={
        "model": "Qwen3-TTS",
        "input": "こんにちは、今日はいい天気ですね。",
        "language": "japanese",
        "reference_audio": "data/audio/kinsenka_3.wav",
        "reference_text": "Reference transcript here",
        "max_tokens": 2048,
    },
)
r.raise_for_status()
pathlib.Path("voice_clone.wav").write_bytes(r.content)
```

## Source Structure

```
crane-serve/src/
├── main.rs              # CLI entry point, AppState, route registration
├── openai_api.rs        # OpenAI request/response types (incl. SpeechRequest)
├── sglang_api.rs        # SGLang native API types
├── chat_template.rs     # Chat template rendering (Jinja / Hunyuan hard-coded)
├── handlers/
│   ├── common.rs        # /health, /v1/stats
│   ├── openai.rs        # OpenAI endpoint handlers
│   ├── sglang.rs        # SGLang endpoint handlers
│   ├── tts.rs           # /v1/audio/speech handler (Qwen3-TTS, Voxtral TTS)
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
| Qwen 3.5 | ❌ | ❌ | Safetensors | Hybrid GDN + softmax attention; CUDA fused recurrence kernel; `max_concurrent=1` |
| Qwen 2.5 | sequential | ❌ | Safetensors | — |
| **Qwen3-TTS** | N/A | N/A | Safetensors (ONNX fallback optional) | Dedicated thread; no continuous batching; voice cloning supported |
| **Voxtral TTS** | N/A | N/A | Safetensors | Dedicated thread; 37-codebook codec; no voice cloning |

Model type is auto-detected from `config.json` / `params.json` (`model_type` / `architectures`) or can be set explicitly with `--model-type`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CRANE_FORCE_GPU_TOPK` | `0` | Force GPU top-k even for large vocabularies |
| `CRANE_TOPP_FALLBACK_TOPK` | `64` | k value for GPU top-k fallback |
| `CRANE_TOPK_SAMPLE_ON_CPU` | `0` | Sample on CPU after GPU top-k |
| `CRANE_SAMPLE_TRACE` | `0` | Verbose sampling timing logs |

## Notes

- **No API key required** — crane-serve does not authenticate requests.
- **Single CUDA device** — The server uses CUDA device 0. Multi-GPU tensor parallelism is not yet supported.
- **KV eviction is lossless** — Evicted sequences preserve their full state and resume automatically; in-flight requests are not dropped or errored.
- **`--max-seq-len 0`** means no limit. On constrained hardware, always set an explicit value to avoid runaway memory growth.
- **GGUF quantization** is supported for Hunyuan Dense and Qwen 3. Qwen 2.5 requires Safetensors format.
- **`--decode-tokens-per-seq`** controls decode rounds per engine step, not per request. Requests always complete fully regardless of this value.
- **Log diagnostics** — The startup log prints `kv_bytes` and `kv_budget`. Monitor these to validate your `--gpu-memory-limit` headroom.
- **Qwen3-TTS and Voxtral TTS run on a dedicated thread** — No continuous batching; each `/v1/audio/speech` request is processed sequentially. Concurrent requests are queued in an unbounded channel.
- **Qwen3-TTS decoder backend** — The speech-tokenizer decoder (codes → waveform) uses native Candle by default. ONNX export is optional as a compatibility fallback.
- **Voxtral TTS uses greedy decoding** — `temperature`, `top_p`, and `repetition_penalty` are accepted for API compatibility but do not affect the output.

## Testing

```bash
# All unit tests
cargo test -p crane-serve
cargo test -p crane-core

# Specific modules
cargo test -p crane-serve engine::scheduler
cargo test -p crane-serve openai_api::tests
cargo test -p crane-serve sglang_api::tests
cargo test -p crane-core autotokenizer
```

## License

MIT
