"""
exporting SNAC model into onnx
which used in Orpheus model when decoding output

SNAC seems also used in many TTS model.
"""

import wave
import time
import numpy as np
from snac import SNAC
import torch
from coreai.utils.bins import save_tensors_to_bin
import torchaudio


torch.set_grad_enabled(False)


def test_model():
    from orpheus_tts import OrpheusModel

    model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
    prompt = """Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we're all connected 24/7 but somehow people feel more alone than ever. And don't even get me started on how it's messing with kids' self-esteem and mental health and whatnot."""

    start_time = time.monotonic()
    syn_tokens = model.generate_speech(
        prompt=prompt,
        voice="tara",
    )

    with wave.open("output.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        total_frames = 0
        chunk_counter = 0
        for audio_chunk in syn_tokens:  # output streaming
            chunk_counter += 1
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)
        duration = total_frames / wf.getframerate()

        end_time = time.monotonic()
        print(
            f"It took {end_time - start_time} seconds to generate {duration:.2f} seconds of audio"
        )


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, codes):
        # Handle multiple code inputs if needed
        audio_hat = self.model.decode(codes)
        # audio_slice = audio_hat[:, :, 2048:4096]
        return audio_hat


def export_snac():
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    # 79.5M

    wave_f = "data/zero_shot_prompt.wav"
    audio, sr = torchaudio.load(wave_f)
    print(audio.shape, sr)

    resampler = torchaudio.transforms.Resample(
        orig_freq=sr,  # Use the original sample rate
        new_freq=24000,  # Target sample rate (24 kHz)
    )
    audio = resampler(audio).unsqueeze(0)
    print(audio.shape)

    # audio = torch.randn(1, 1, 429000)  # B, 1, T
    with torch.inference_mode():
        codes = model.encode(audio)
        audio_hat = model.decode(codes)
        print(f"audio_hat {audio_hat.shape}")

        detached_audio = audio_hat.detach().cpu()
        audio_np = detached_audio.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        with wave.open(wave_f.replace(".wav", "_regen.wav"), "wb") as wf:
            wf.setnchannels(1)  # Mono audio (from your audio_hat shape)
            wf.setsampwidth(2)  # 2 bytes = 16-bit samples (int16)
            wf.setframerate(24000)  # Sample rate (adjust if yours differs)
            wf.writeframes(audio_bytes)
    print(codes)
    print([i.shape for i in codes])

    save_tensors_to_bin(codes, "snac_codes.bin")

    snac_device = "cpu"
    model = model.to(snac_device)
    # codes = torch.randn([1, 1, 24001])
    torch.onnx.export(
        DecoderWrapper(model),
        codes,
        "snac_24khz.onnx",
        input_names=["c1", "c2", "c3"],  # input tensor name(s)
        output_names=["audio_hat"],  # output tensor name(s)
        dynamic_axes={
            "c1": {
                0: "B",
                1: "T",
            },
            "c2": {
                0: "B",
                1: "T",
            },
            "c3": {
                0: "B",
                1: "T",
            },
            "audio_hat": {0: "batch_size", 2: "T"},
        },
        opset_version=19,
    )


if __name__ == "__main__":
    # test_model()
    export_snac()
