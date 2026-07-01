# Voxtral-4B-TTS-2603

Voxtral TTS is a hybrid autoregressive + flow-matching text-to-speech model
from Mistral AI, totaling ~4B parameters across three components:

| Component                   | Params | Role                                    |
|-----------------------------|--------|-----------------------------------------|
| LLM decoder backbone        | 3.4B   | Autoregressive semantic token prediction |
| Flow-matching acoustic TX   | 390M   | Acoustic latent generation via flow matching |
| Audio codec (encoder+decoder)| 300M  | Waveform <-> discrete token conversion  |

Base model: Ministral 3B (26-layer Mistral architecture).
License: CC BY-NC 4.0.

Paper: <https://arxiv.org/abs/2603.25551>

## Architecture

The pipeline has three stages:

1. **LLM backbone** — A Mistral decoder-only transformer (26 layers, GQA 32/8
   heads, dim 3072) predicts semantic tokens autoregressively. Each frame
   produces a hidden state that is projected to logits over an 8192-entry
   semantic codebook.

2. **Acoustic transformer** — A bidirectional 3-layer transformer takes the LLM
   hidden state and generates 36 acoustic codes per frame via flow matching
   (8 Euler steps with classifier-free guidance, alpha=1.2). The continuous
   output is quantized to FSQ codes (21 levels each).

3. **Audio codec decoder** — Converts 37 codes per frame (1 semantic + 36
   acoustic) to a 24 kHz waveform. Uses a conv + transformer architecture with
   ALiBi attention, sliding windows, and 8x temporal upsampling (12.5 Hz frame
   rate -> 24 kHz via 240-sample patches).

## Voice conditioning

Pre-computed voice embeddings (`.pt` files, BF16) are injected directly into
the LLM input sequence. 20 preset voices are available across 10 languages.

## Key parameters

- Frame rate: 12.5 Hz
- Sample rate: 24 kHz
- Semantic codebook: 8192 entries (VQ)
- Acoustic codebooks: 36 dimensions, 21 levels each (FSQ)
- Tokenizer: Tekken (BPE, 131K vocab)

## References

- [Paper (arXiv:2603.25551)](https://arxiv.org/abs/2603.25551)
- [Mistral blog post](https://mistral.ai/news/voxtral-tts)
