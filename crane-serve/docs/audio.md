# Audio: Text-to-Speech and Speech Recognition

Crane supports several text-to-speech (TTS) model families and one speech
recognition (ASR) model. They share the server's OpenAI-compatible
endpoints, so any OpenAI SDK or `curl` client works without modification.

## Choosing a TTS model

| | VoxCPM2 | Voxtral TTS | Qwen3-TTS |
|---|---|---|---|
| Languages | Multilingual, including Chinese and English | English, German, French, Spanish, Italian, Portuguese, Dutch, Hindi, Arabic | Chinese, English, Japanese, and more (auto-detected) |
| Voice cloning | Yes | No | Yes (Base model) |
| Predefined voices | Reference audio cached at startup | Yes (20 embeddings) | Yes (CustomVoice model) |
| Streaming | PCM16, including cached voices | PCM16 | PCM16 |
| Decoding | CFM (`cfm_steps`, `cfg_scale`) | Greedy only | Sampling (`temperature`, `top_p`) |

Pick Voxtral TTS for a broad set of predefined voices without cloning. Pick
Qwen3-TTS if you need to clone a voice from a reference clip, or want
sampling control over prosody. Pick VoxCPM2 for multilingual synthesis with
reusable reference-audio voices and fixed-voice streaming.

## Qwen3-TTS

### Setup

Download a checkpoint. Use the CustomVoice model for predefined speakers, or
the Base model for voice cloning.

```bash
mkdir -p checkpoints/
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --local-dir checkpoints/Qwen3-TTS-12Hz-0.6B-CustomVoice

huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --local-dir checkpoints/Qwen3-TTS-12Hz-0.6B-Base
```

Native speech-tokenizer decoding is enabled by default; no extra export step
is needed. ONNX export is available only as a compatibility fallback — see
[Troubleshooting](#troubleshooting).

Build and start the server:

```bash
cargo build -p crane-serve --release            # CPU
cargo build -p crane-serve --release --features "cuda"            # CUDA
cargo build -p crane-serve --release --features "metal,accelerate" # macOS

./target/release/crane-serve \
    --model-path checkpoints/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --port 8080
```

`--model-type qwen3_tts` is auto-detected from `config.json` and optional.

### Synthesize speech

```bash
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS",
    "input": "Hello, this is a test of the Qwen3-TTS model.",
    "voice": "Ryan",
    "language": "english"
  }' \
  --output speech.wav
```

Built-in speakers for the CustomVoice model include `Serena`, `Vivian`,
`Uncle_fu`, `Ryan`, `Aiden`, `Ono_anna`, `Sohee`, `Eric`, `Dylan`. Larger
variants may include more. To list every speaker in your checkpoint:

```bash
python3 -c "
import json
c = json.load(open('checkpoints/Qwen3-TTS-12Hz-0.6B-Base/config.json'))
for name in sorted(c['talker_config']['spk_id']):
    print(name)
"
```

### Voice cloning (Base model)

The Base model clones a voice from a reference clip and its transcript. This
is in-context learning, not fine-tuning — no training step is required.

```bash
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-TTS",
    "input": "Reference audio transcript goes here, but in a new sentence.",
    "language": "english",
    "reference_audio": "/path/to/reference.wav",
    "reference_text": "Reference audio transcript goes here"
  }' \
  --output voice_clone.wav
```

| Field | Type | Description |
|-------|------|-------------|
| `reference_audio` | string | Local file path to a WAV clip |
| `reference_text` | string | Transcript of that clip (required) |

Setting `reference_audio` switches the request to voice-clone mode
regardless of the `voice` field. `reference_audio` must be a WAV file, and
`language` should match the target text.

## VoxCPM2

### Setup and built-in voices

Place the checkpoint under `checkpoints/VoxCPM2/`. It must include the model
files required by `VoxCpm2Model`, including the converted
`audiovae.safetensors`.

Put reusable reference clips in `data/voices/`:

```text
data/voices/
├── voice_preview_adam.wav
└── voice_preview_clyde.wav
```

Start the server with:

```bash
cargo build -p crane-serve --release

./target/release/crane-serve \
  --model-path checkpoints/VoxCPM2/ \
  --model-type voxcpm2 \
  --voice-dir data/voices \
  --port 8080
```

`--model-type voxcpm2` is optional when auto-detection succeeds.
`--voice-dir` defaults to `data/voices`.

On startup, Crane encodes every supported audio file in the voice directory
and saves its conditioning tensor under:

```text
data/voices/.voxcpm2-cache/<source-filename>.safetensors
```

Subsequent starts load the cached tensor. A cache is rebuilt automatically
when its source audio or relevant model files are newer. Supported source
formats are WAV, MP3, FLAC, OGG, M4A, and AAC; WAV is recommended.

The API voice name is the filename without its extension. For example,
`voice_preview_adam.wav` is selected with `"voice": "voice_preview_adam"`.
Restart the server after adding or replacing a voice file.

### Generate with a fixed voice

```bash
curl http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "voxcpm2",
    "input": "你好，这是 VoxCPM2 的固定音色测试。",
    "voice": "voice_preview_adam",
    "response_format": "wav",
    "cfm_steps": 10,
    "cfg_scale": 2.0,
    "max_tokens": 200
  }' \
  --output voxcpm2.wav

afplay voxcpm2.wav
```

If `voice` is omitted, VoxCPM2 uses zero-shot conditioning; the speaker
identity is then not guaranteed to remain fixed between requests.

### Stream with a fixed voice

Streaming requires `response_format: "pcm"`. The response body is headerless,
mono, signed PCM16 little-endian audio. The sample rate is returned in the
`X-Sample-Rate` header and is currently 48 kHz for VoxCPM2.

```bash
curl --no-buffer http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "voxcpm2",
    "input": "你好，这是固定音色的流式语音测试。",
    "voice": "voice_preview_adam",
    "response_format": "pcm",
    "stream": true,
    "cfm_steps": 10,
    "cfg_scale": 2.0,
    "max_tokens": 200
  }' \
  --output voxcpm2.pcm
```

To convert the raw response to a WAV file:

```bash
ffmpeg -f s16le -ar 48000 -ac 1 -i voxcpm2.pcm voxcpm2-stream.wav
afplay voxcpm2-stream.wav
```

Cached built-in voices support streaming. Request-time cloning through
`reference_audio` cannot be combined with `stream: true`; add the reference
clip to `--voice-dir` when that voice must be reused by streaming requests.

### Streaming test client

Install the optional playback dependency in the Python environment used to
run the client:

```bash
python -m pip install -r tests/requirements-voxcpm2-stream.txt
```

Save the streamed response as a valid WAV file:

```bash
python tests/test_voxcpm2_stream.py \
  --voice voice_preview_adam \
  --text '你好，这是固定音色的流式测试。' \
  --output voxcpm2-stream.wav
```

Play through `sounddevice` while receiving, with a jitter buffer separating
network reads from playback:

```bash
python tests/test_voxcpm2_stream.py \
  --play \
  --voice voice_preview_adam \
  --play-buffer-ms 1000 \
  --output voxcpm2-stream.wav
```

On macOS, list CoreAudio devices and select one explicitly if the default
device reports PortAudio error `-9986`:

```bash
python tests/test_voxcpm2_stream.py --list-devices
python tests/test_voxcpm2_stream.py \
  --play --device 3 --play-buffer-ms 1000 \
  --voice voice_preview_adam
```

Important client options:

| Option | Default | Description |
|---|---:|---|
| `--url` | `http://127.0.0.1:8080/v1/audio/speech` | Speech endpoint |
| `--voice` | `voice_preview_adam` | Voice filename without extension |
| `--cfm-steps` | `10` | CFM steps; very small values reduce quality |
| `--cfg-scale` | `2.0` | Classifier-free guidance scale |
| `--max-tokens` | `200` | Maximum generation length |
| `--play` | off | Enable live playback |
| `--device` | system default | sounddevice output name or index |
| `--play-buffer-ms` | `500` | Audio queued before playback begins |
| `--output` | `voxcpm2-stream.wav` | Saved WAV path |

The client prints HTTP readiness, chunk arrival intervals, buffered audio,
underflow count, time to first audio, and the final generation realtime ratio.
Increasing `underflows` means playback is consuming audio faster than the
model produces it. A larger buffer smooths short gaps, but continuous playback
ultimately requires generation near or above `1.0x` realtime.

### Request-time cloning

Non-streaming requests can clone a WAV reference directly:

```bash
curl http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "voxcpm2",
    "input": "这是使用参考音频克隆出来的新句子。",
    "reference_audio": "/absolute/path/to/reference.wav",
    "reference_text": "参考音频中实际说出的文字",
    "response_format": "wav",
    "cfm_steps": 10,
    "cfg_scale": 2.0,
    "max_tokens": 200
  }' \
  --output voxcpm2-clone.wav
```

`reference_audio` takes precedence over `voice` and recomputes conditioning
for each request. Use the built-in voice cache for repeated requests.

### VoxCPM2 model tests

The model integration tests are ignored by default because they require a
local checkpoint:

```bash
CRANE_VOXCPM2_DIR=checkpoints/VoxCPM2 \
  cargo test -p crane-core --test voxcpm2_generate -- --ignored --nocapture

CRANE_VOXCPM2_DIR=checkpoints/VoxCPM2 \
CRANE_VOXCPM2_REF_WAV=data/voices/voice_preview_adam.wav \
  cargo test -p crane-core --test voxcpm2_conditioning -- --ignored --nocapture
```

These cover waveform validity, reference conditioning, prompt-cache round
trips, and streaming/one-shot equivalence for a fixed random seed.

## Voxtral TTS

### Setup

```bash
mkdir -p checkpoints/
hf download mistralai/Voxtral-4B-TTS-2603 \
    --local-dir checkpoints/Voxtral-4B-TTS-2603

cargo build -p crane-serve --release            # CPU
cargo build -p crane-serve --release --features "cuda"

./target/release/crane-serve \
    --model-path checkpoints/Voxtral-4B-TTS-2603 \
    --port 8080
```

`--model-type voxtral_tts` is auto-detected and optional.

### Synthesize speech

Voice embeddings are `.pt` files under `voice_embedding/` in the checkpoint.
Pass the filename without extension as `voice`.

```bash
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
```

| Voice | Gender | Language / Style |
|-------|--------|-----------------|
| `neutral_female` | Female | English (neutral) |
| `neutral_male` | Male | English (neutral) |
| `casual_female` | Female | English (casual) |
| `casual_male` | Male | English (casual) |
| `cheerful_female` | Female | English (cheerful) |
| `de_female` / `de_male` | Female / Male | German |
| `fr_female` / `fr_male` | Female / Male | French |
| `es_female` / `es_male` | Female / Male | Spanish |
| `it_female` / `it_male` | Female / Male | Italian |
| `pt_female` / `pt_male` | Female / Male | Portuguese |
| `nl_female` / `nl_male` | Female / Male | Dutch |
| `hi_female` / `hi_male` | Female / Male | Hindi |
| `ar_male` | Male | Arabic |

If `voice` is omitted, the first loaded voice is used. The available voices
can differ between checkpoint releases — check `voice_embedding/` in your
checkpoint.

Voxtral uses greedy decoding: `temperature`, `top_p`, and
`repetition_penalty` are accepted for API compatibility but have no effect.
Voxtral does not support voice cloning; `reference_audio` returns an error.
On CPU with F32 weights, generation is slow. Use CUDA or Metal.

## Generation parameters

| Parameter | Default | Qwen3-TTS | Voxtral TTS | Notes |
|-----------|---------|-----------|-------------|-------|
| `temperature` | `0.9` | used | accepted, no effect | Lower = more stable prosody |
| `top_p` | `null` | used | accepted, no effect | `null` or `1.0` matches reference defaults |
| `repetition_penalty` | `1.05` | used | accepted, no effect | Reduces repeated codec tokens |
| `max_tokens` | `8192` | used | used | Qwen3: ~83 ms/token at 12 Hz; Voxtral: ~80 ms/frame at 12.5 Hz |
| `language` | `"auto"` | used | accepted, no effect | A wrong language hint degrades Qwen3 quality |

## Speech recognition (ASR)

Crane supports Qwen3-ASR via `--model-type qwen3_asr` (auto-detected from
`config.json`). Start the server the same way as any other model:

```bash
./target/release/crane-serve \
    --model-path checkpoints/Qwen3-ASR \
    --port 8080
```

Transcribe audio with the OpenAI-compatible transcriptions endpoint:

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

The response is `{"text": "..."}`. `--ui` also exposes an audio upload and
transcription screen in the browser for ASR models.

## Real-time duplex audio (experimental)

`GET /v1/audio/duplex` is a WebSocket endpoint for live, full-duplex voice
conversations, supported only with MiniCPM-o-4.5
(`--model-type minicpmo_duplex`). Only one session is allowed system-wide at
a time — a second connection attempt gets `503 Service Unavailable` — since
a resident session keeps all model towers loaded (~18–19 GB).

Protocol: send an optional JSON `{"system_prompt": "..."}` to start the
session, then stream raw 16-bit PCM audio (16 kHz mono) as binary WebSocket
frames. The server replies with JSON events containing recognized text and,
when the model responds, base64-encoded 24 kHz PCM audio.

This endpoint is early and narrow in scope. Treat it as experimental.

## Notes

- Qwen3-TTS and Voxtral TTS each run on a dedicated thread, not the
  continuous-batching engine. Requests are processed sequentially; concurrent
  requests queue.
- Qwen3-TTS's speech-tokenizer decoder (codes → waveform) uses native Candle
  by default. ONNX export is an optional fallback.

## Troubleshooting

**`TTS model not loaded`** — The server was started with a non-TTS model.
Restart with `--model-type qwen3_tts` or `--model-type voxtral_tts`.

**`ASR model not loaded`** — The server was started without an ASR model.
Restart with `--model-type qwen3_asr`.

**Garbled audio or silence (Qwen3-TTS)** — Lower `temperature` to `0.5`–`0.7`
and check that `language` matches the input text.

**`native speech tokenizer load failed` (Qwen3-TTS)** — Check that
`<model_dir>/speech_tokenizer/config.json` and its safetensors files are
complete. Older checkpoints missing
`quantizer.*._codebook.cluster_usage` fall back to a ones vector
automatically. If loading still fails, export the ONNX fallback decoder:

```bash
pip install -e vendor/Qwen3-TTS
python scripts/export_qwen_tts_tokenizer_onnx.py \
    checkpoints/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer \
    checkpoints/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer/speech_tokenizer_decoder.onnx
```

**`Speech tokenizer ONNX not found`** — Appears only when the native decoder
fails and the ONNX fallback is requested. Run the export script above and
place the ONNX file at
`<model_dir>/speech_tokenizer/speech_tokenizer_decoder.onnx`.

**`Voxtral TTS does not support voice cloning`** — Use Qwen3-TTS Base
instead; `reference_audio` is not supported by Voxtral.

**Slow generation on CPU (Voxtral)** — Voxtral needs ~14 transformer passes
per audio frame. On CPU with F32 weights, short phrases can take several
minutes. Use CUDA or Metal.
