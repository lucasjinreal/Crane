# ASR forward-pass benchmarks: Crane vs. PyTorch

Compares Qwen3-ASR **model forward time only** — audio feature extraction,
tokenization, and (for Crane) HTTP overhead are excluded from both sides, so
the numbers reflect candle vs. torch compute, not I/O.

## PyTorch side: `speed_torch_qwen3_asr.py`

Self-contained [uv script](https://docs.astral.sh/uv/guides/scripts/) — no
virtualenv setup needed, `uv` resolves `torch`/`transformers`/`accelerate`/
`librosa` on first run.

```bash
./tests/speed_torch_qwen3_asr.py [model_path] [audio_path]
# or
uv run --script tests/speed_torch_qwen3_asr.py [model_path] [audio_path]
```

Defaults: `model_path=models/Qwen3-ASR-0.6B-hf`, `audio_path=speech.wav`.

Requires a `transformers` build with `Qwen3ASRForConditionalGeneration`
registered (a dev build as of this writing).

The script runs `processor.apply_transcription_request(audio=...)` (feature
extraction + tokenization, and audio loading from the given path) once,
untimed, then times a single `model.generate(...)` call — prefill and decode
combined, since `generate()` doesn't expose a clean split point — bracketed
by `torch.cuda.synchronize()`. Output:

```
Forward time elapsed: 1345.2ms
Total tokens generated: 87
Tokens per second: 64.68
```

## Crane side: server log line

Crane has no standalone ASR benchmark binary; the timing is built into
`Model::transcribe()` (`crane-core/src/models/qwen3_asr/model.rs`) and logged
on every transcription request, so any real call — including through
`crane-serve` — reports it.

1. Build and start the server against the same checkpoint:

   ```bash
   cargo build -p crane-serve --release   # add --features cuda for GPU
   RUST_LOG=info ./target/release/crane-serve --model-path models/Qwen3-ASR-0.6B-hf --port 8000
   ```

   (`--model-type` is auto-detected from `config.json`'s `qwen3_asr` type.)

2. In another terminal, send the same audio file used for the torch run:

   ```bash
   python3 tests/crane_client.py transcribe speech.wav
   ```

3. Read the timing from the server's stdout:

   ```
   qwen3_asr forward: prefill 142.3ms, decode 87 tokens in 1203.5ms (72.3 tok/s)
   ```

`prefill` covers audio encoding + projection + first decoder pass;
`decode` covers the autoregressive loop. Both are wrapped with a device sync
(`Device::synchronize`) so CUDA numbers reflect actual compute, not just
kernel-launch time.

## Comparing the two

| Metric | Torch | Crane |
|---|---|---|
| Combined forward time | `Forward time elapsed` | `prefill` + `decode` time |
| Decode throughput | `Total tokens generated / elapsed` (includes prefill) | `decode` tok/s (prefill excluded — more precise) |

Notes:
- Use the **same audio file, model size, device, and dtype** on both sides —
  the checkpoint's native dtype is `bfloat16`.
- Token counts won't match exactly between the two runs (different sampling
  defaults: Crane's `TranscribeOptions` default is temperature 0.9 vs. the
  torch script's greedy `do_sample=False`), so compare **tokens/sec**, not
  raw elapsed time, when transcript lengths differ.
- Always run a warmup call before timing (both scripts already do this) —
  first-call latency includes CUDA kernel compilation/caching and isn't
  representative.
