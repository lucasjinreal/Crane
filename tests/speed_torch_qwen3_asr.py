#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "torch",
#     "transformers",
#     "accelerate",
#     "librosa",
# ]
# ///

import sys
import time

import torch
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration


def benchmark_asr(model_path, audio_path):
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )

    # Feature extraction + tokenization happen here, outside the timed region.
    inputs = processor.apply_transcription_request(audio=audio_path).to(
        model.device, model.dtype
    )

    # Warmup (not timed).
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=448, do_sample=False)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start_time

    prompt_len = inputs["input_ids"].shape[1]
    total_tokens = output_ids.shape[1] - prompt_len
    tokens_per_sec = total_tokens / elapsed

    print(f"Forward time elapsed: {elapsed * 1000:.1f}ms")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Tokens per second: {tokens_per_sec:.2f}")


if __name__ == "__main__":
    default_model_path = "models/Qwen3-ASR-0.6B-hf"
    default_audio_path = "speech.wav"
    model_path = sys.argv[1] if len(sys.argv) > 1 else default_model_path
    audio_path = sys.argv[2] if len(sys.argv) > 2 else default_audio_path
    benchmark_asr(model_path, audio_path)
