//! MiniCPM-o-4.5 (OpenBMB): full omni model — vision + audio understanding,
//! speech-token TTS generation, and a CosyVoice2-style vocoder, sharing one
//! Qwen3 LLM backbone.
//!
//! **Work in progress, implemented in phases** (see `AGENTS.md` for current
//! status — do not assume any capability beyond what's listed there):
//!
//! 1. `llm` — the plain Qwen3 LLM tower standalone (text-only). Implemented.
//! 2. `vision`/`resampler`/`preprocess` — SigLIP + perceiver `Resampler` +
//!    image preprocessing. Implemented.
//! 3. `audio_encoder`/`audio`/`vlm` — Whisper-medium encoder + projector,
//!    spliced together with vision into `llm` for live omni (image + audio)
//!    Q&A (`MiniCpmOVlModel`). Implemented; not yet wired into `crane-serve`
//!    or the `crane` SDK (no HTTP access yet, direct Rust API only).
//! 4. `tts_llm` — `MiniCPMTTS` Llama backbone, zero-shot speech-token
//!    generation (`MiniCpmTts::generate`).
//! 5. `token2wav` — CosyVoice2-style flow-matching + HiFi-GAN vocoder
//!    (`Token2Wav::synthesize`), turning phase 4's speech-token ids into a
//!    real waveform (system-default voice only — no reference-audio voice
//!    cloning, see the `token2wav` module doc). Implemented; not yet wired
//!    into `crane::audio::Tts` or `crane-serve` (direct Rust API only).
//! 6. `duplex` — full-duplex live streaming session (`DuplexSession`).
//!    Sub-phased (see `duplex.rs`'s module doc): 6a (WebSocket-ready chunked
//!    listen/speak arbitration, matching upstream's default
//!    `sliding_window_mode: "off"` config), 6b (opt-in basic-mode
//!    sliding-window KV eviction + RoPE cache realignment, `sliding_window.rs`),
//!    6c (opt-in context-preserve sliding-window mode), and 6d (incremental
//!    per-chunk TTS speech-token generation, `MiniCpmTts::generate_chunk`)
//!    are implemented. **Deliberately not ported**: the real streaming
//!    Token2wav vocoder (`Token2wav.stream` — a novel per-denoising-timestep
//!    *and* per-layer KV cache with no analog elsewhere in this codebase);
//!    audio synthesis stays per-turn (see `duplex.rs`'s module doc). No
//!    WebSocket transport wired up either — direct Rust API only so far.

pub mod audio;
pub mod audio_encoder;
pub mod config;
pub mod duplex;
pub mod llm;
pub mod preprocess;
pub mod resampler;
mod sliding_window;
pub mod token2wav;
pub mod tts_llm;
pub mod vision;
pub mod vlm;

pub use audio::AudioProjector;
pub use audio_encoder::AudioEncoder;
pub use config::{load_config, MiniCpmOConfig};
pub use duplex::{ChunkOutput, DuplexConfig, DuplexGgufPaths, DuplexSession, SlidingWindowMode};
pub use llm::MiniCpmOLlm;
pub use resampler::Resampler;
pub use token2wav::Token2Wav;
pub use tts_llm::{MiniCpmTts, TtsGenerationConfig};
pub use vision::VisionModel;
pub use vlm::{MiniCpmOVlModel, VlGenerationConfig};
