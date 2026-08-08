//! Full-duplex live audio chat session — phases 6a+6b+6c+6d: WebSocket-
//! session-ready chunked listen/speak arbitration (6a), two opt-in
//! sliding-window KV cache eviction modes for sessions that run long enough
//! to matter — basic (6b, cache-length-triggered, drop-and-realign) and
//! context-preserve (6c, unit-count-triggered, drop-but-summarize-into-
//! "previous") — and incremental per-chunk TTS speech-token generation
//! (6d, `MiniCpmTts::generate_chunk`, persistent KV cache across chunks
//! within a turn). Matches the real reference implementation's **default
//! configuration** (`sliding_window_mode: "off"` in `DuplexWindowConfig`/
//! `MiniCPMODuplex._default_duplex_params`) when eviction is left off.
//!
//! **Scope simplification (deliberate, not yet ported)**: the real
//! `_generate_waveform_from_tokens` also streams the *vocoder* itself
//! (`Token2wav.stream`, CosyVoice2's genuine chunked CFM + streaming
//! HiFi-GAN with a cached reference-prompt encoding and a lookahead
//! buffer) so audio can start playing mid-turn. That's a novel subsystem
//! with no analog anywhere in this codebase — in particular the DiT/CFM's
//! `forward_chunk` caches attention **per denoising timestep *and* per
//! transformer layer simultaneously**, a `(n_timesteps, depth, batch,
//! heads, seq, dim)` cache with no existing precedent to lean on, and the
//! highest-risk piece of anything in this port (a bug there degrades
//! audio quality silently rather than crashing). This port keeps
//! Token2wav synthesis **per-turn** (phase 5's already-validated
//! non-streaming `Token2Wav::synthesize`, called once on the whole turn's
//! accumulated codes at `end_of_turn`) rather than porting the streaming
//! vocoder — audio still only arrives once a turn completes, same as 6a.
//! What 6d *does* deliver: TTS token generation itself now happens
//! incrementally, one real audio-chunk-worth of conditioning at a time
//! with a persistent KV cache (matching `MiniCPMTTS.generate_chunk`'s
//! actual mechanics and its per-call, not cross-call, repetition-penalty
//! scope) instead of regenerating a whole turn from scratch in one call at
//! the end. **Porting the real streaming vocoder is left for a later
//! session** — see `AGENTS.md`'s "Not implemented yet" note.
//!
//! Ported from `modeling_minicpmo.py`'s `MiniCPMODuplex.streaming_prefill`/
//! `streaming_generate`/`_convert_results_to_tts_input` and `utils.py`'s
//! `StreamDecoder.feed`/`decode`/`register_unit_start`/`register_unit_end`/
//! `enforce_window`/`enforce_window_with_context`/`_update_previous`/
//! `_rebuild_cache_with_previous` (real reference source — see the plan
//! doc). Basic-mode eviction's RoPE realignment
//! (`drop_tokens_from_cache`/`realign_rotary_suffix`) lives in
//! `super::sliding_window`, kept separate so its math can be
//! hand-computed-value unit tested without needing a model loaded — see
//! that module's doc for why this is the highest numerical-risk piece of
//! the *implemented* part of the whole duplex phase; context-preserve mode
//! reuses `realign_rotary_suffix` directly for its own units-segment
//! realignment.
//!
//! **Scope simplification**: unlike upstream (which lets a duplex session's
//! system prompt be split into a `prefix`/`suffix` pair so context-preserve
//! mode can insert accumulated "previous" content *between* them), this
//! port's `prepare()` only takes one system-prompt string, fed entirely as
//! the fixed `prefix` with an always-empty `suffix` — a real, supported
//! upstream configuration (`suffix_system_prompt` is optional there too),
//! just not the split-prompt case. `previous` content still lands correctly
//! between the (whole) system prompt and the live unit history.

use candle_core::{DType, Device, Result, Tensor};
use tokenizers::Tokenizer;

use super::audio::AudioProjector;
use super::audio_encoder::AudioEncoder;
use super::config::MiniCpmOConfig;
use super::llm::MiniCpmOLlm;
use super::sliding_window;
use super::token2wav::Token2Wav;
use super::tts_llm::{MiniCpmTts, TtsGenerationConfig};
use crate::models::modules::rotary::RotaryEmbedding;
use crate::models::qwen3_asr::feature_extractor::WhisperFeatureExtractor;

/// `DuplexWindowConfig.sliding_window_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlidingWindowMode {
    /// No cache eviction — the KV cache grows for the session's duration.
    /// Matches upstream's own default.
    Off,
    /// Cache-length-triggered eviction: once the cache exceeds
    /// `basic_window_high_tokens`, drop the oldest whole units (each
    /// `streaming_prefill`/`streaming_generate` cycle) until it's back at
    /// or under `basic_window_low_tokens`. Matches `DuplexWindowConfig`'s
    /// `"basic"` mode.
    Basic,
    /// Unit-count-triggered eviction: once more than `context_max_units`
    /// units are live, drop the oldest one — but instead of just deleting
    /// its content, fold its generated (non-listen) text into a running
    /// `"previous: "`-prefixed summary that's re-fed right after the
    /// system prompt, truncated to `context_previous_max_tokens`. Matches
    /// `DuplexWindowConfig`'s `"context"` mode.
    Context,
}

/// One completed `streaming_prefill`/`streaming_generate` cycle's
/// footprint — mirrors an entry in `StreamDecoder._unit_history`. Basic
/// mode only reads `length`; context mode also needs `is_listen`/
/// `generated_tokens` to decide what (if anything) to fold into "previous"
/// when this unit gets evicted.
struct UnitRecord {
    /// KV cache length this unit occupies (its own `<unit>...</unit>`
    /// tokens plus whatever it generated).
    length: usize,
    is_listen: bool,
    /// Raw generated token ids for this unit (unfiltered — control tokens
    /// like `<unit>`/`<|listen|>` are filtered out at extraction time, not
    /// here, matching `StreamDecoder._extract_generated_text`).
    generated_tokens: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct DuplexConfig {
    /// Max text/control tokens generated per audio chunk before forcing a
    /// `<|chunk_eos|>` (`streaming_generate`'s default: 20).
    pub max_new_speak_tokens_per_chunk: usize,
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub text_repetition_penalty: f32,
    pub text_repetition_window_size: usize,
    /// Multiplies the listen-token's probability before final sampling
    /// (<1 discourages listening, >1 encourages it). `1.0` = no effect.
    pub listen_prob_scale: f64,
    /// Force `<|listen|>` for this many chunks at session start (lets the
    /// model "warm up" on audio context before it's allowed to speak).
    pub force_listen_count: usize,
    pub tts_cfg: TtsGenerationConfig,
    pub n_timesteps: usize,
    /// Basic-mode sliding-window eviction — off by default, matching
    /// upstream (see the module doc).
    pub sliding_window_mode: SlidingWindowMode,
    /// High watermark: `SlidingWindowMode::Basic` triggers eviction once
    /// the KV cache exceeds this many tokens (`DuplexWindowConfig`'s
    /// default: 8000).
    pub basic_window_high_tokens: usize,
    /// Low watermark: eviction stops once the cache is back at or under
    /// this many tokens (`DuplexWindowConfig`'s default: 6000).
    pub basic_window_low_tokens: usize,
    /// `SlidingWindowMode::Context`'s cap on the accumulated "previous"
    /// summary's content length, not counting the marker
    /// (`DuplexWindowConfig`'s default: 500).
    pub context_previous_max_tokens: usize,
    /// `SlidingWindowMode::Context` triggers eviction once more than this
    /// many units are live (`DuplexWindowConfig`'s default: 24).
    pub context_max_units: usize,
}

impl Default for DuplexConfig {
    fn default() -> Self {
        Self {
            max_new_speak_tokens_per_chunk: 20,
            temperature: 0.7,
            top_k: 100,
            top_p: 0.8,
            text_repetition_penalty: 1.05,
            text_repetition_window_size: 512,
            listen_prob_scale: 1.0,
            force_listen_count: 0,
            tts_cfg: TtsGenerationConfig::default(),
            n_timesteps: 10,
            sliding_window_mode: SlidingWindowMode::Off,
            basic_window_high_tokens: 8000,
            basic_window_low_tokens: 6000,
            context_previous_max_tokens: 500,
            context_max_units: 24,
        }
    }
}

/// Result of one `streaming_generate` call (one audio chunk's worth of
/// listen/speak arbitration).
pub struct ChunkOutput {
    pub is_listen: bool,
    pub text: String,
    /// `Some` only when the turn just ended and speech was generated
    /// (mono 24kHz PCM, `[-1, 1]`).
    pub audio_waveform: Option<Vec<f32>>,
    pub end_of_turn: bool,
}

pub struct DuplexSession {
    llm: MiniCpmOLlm,
    audio_encoder: AudioEncoder,
    audio_projector: AudioProjector,
    mel_extractor: WhisperFeatureExtractor,
    tts: MiniCpmTts,
    token2wav: Token2Wav,
    cfg: DuplexConfig,
    device: Device,
    dtype: DType,

    unit_token_id: u32,
    unit_end_token_id: u32,
    listen_token_id: u32,
    chunk_eos_token_id: u32,
    chunk_tts_eos_token_id: u32,
    turn_eos_token_id: u32,
    /// Forbidden during normal (non-short-circuit) sampling —
    /// `<|tts_pad|>` plus the tokenizer's own `bad_token_ids`, matching
    /// `StreamDecoder.forbidden_token_ids` (`chunk_eos` is *not* forbidden
    /// here — it gets an independent short-circuit check first, same as Python).
    forbidden_token_ids: Vec<u32>,

    /// Table for the sliding-window RoPE realignment
    /// (`super::sliding_window`) — sized to the LLM's own
    /// `max_position_embeddings`/`rope_theta`/`head_dim`, independent of
    /// `MiniCpmOLlm`'s internal rotary table.
    rotary: RotaryEmbedding,
    /// KV cache length right after the system prompt prefill — the
    /// `preserve` boundary basic-mode eviction never drops below, and the
    /// starting point context mode inserts "previous" content right after.
    /// Basic mode: fixed for the session. Context mode: grows as
    /// "previous" grows (`preserve_prefix_length + previous_token_ids.len()`,
    /// this port's `suffix` is always empty — see the module doc). Mirrors
    /// `StreamDecoder._system_preserve_length`.
    system_preserve_length: usize,
    /// Context mode only: the *fixed* (never-changing) system-prompt-only
    /// boundary, i.e. `system_preserve_length` before any "previous"
    /// content has ever been inserted. Mirrors
    /// `StreamDecoder._preserve_prefix_length`.
    preserve_prefix_length: usize,
    /// Context mode only: `"\n\nprevious: "` tokenized once (no special
    /// tokens, matching Python's `add_special_tokens=False`) — prepended
    /// the first time any content is folded into "previous". Mirrors
    /// `StreamDecoder._previous_marker_token_ids`.
    previous_marker_token_ids: Vec<u32>,
    /// Context mode only: the marker (once present) plus all accumulated
    /// summarized content currently baked into the live KV cache, right
    /// after the system prompt. Mirrors `StreamDecoder._previous_token_ids`.
    previous_token_ids: Vec<u32>,
    /// Each completed unit (one `streaming_prefill`/`streaming_generate`
    /// cycle), oldest first — mirrors `StreamDecoder._unit_history`. Basic
    /// mode only reads `.length`; context mode also reads `.is_listen`/
    /// `.generated_tokens`.
    unit_records: std::collections::VecDeque<UnitRecord>,
    /// KV cache length at the start of the unit currently being built —
    /// mirrors `StreamDecoder._pending_unit_start_cache_len`.
    pending_unit_start_len: Option<usize>,

    current_turn_ended: bool,
    force_listen_remaining: usize,
    /// Recent non-special generated tokens, for repetition penalty —
    /// unbounded growth isn't an issue at chat-turn scale, trimmed to
    /// `text_repetition_window_size` before each penalty application,
    /// matching `StreamDecoder.generated_tokens[-window_size:]`.
    generated_tokens: Vec<u32>,
    pending_logits: Option<Tensor>,

    /// `StreamDecoder`-external, `MiniCPMODuplex`-level TTS state, carried
    /// across chunks *within* a turn (reset at turn boundaries) — mirrors
    /// `tts_text_start_pos`/`tts_past_key_values` (the KV cache itself
    /// lives inside `self.tts`, persisted by simply not calling
    /// `clear_kv_cache` between chunks — see [`Self::streaming_generate`]).
    tts_start_pos: usize,
    /// Speech-token codes accumulated so far this turn, synthesized in one
    /// non-streaming `Token2Wav::synthesize` call at `end_of_turn` — see
    /// the module doc's "Scope simplification" note on why this stays
    /// per-turn rather than genuinely streaming (phase 6d).
    turn_codes: Vec<u32>,
}

fn get_token_or_bail(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    tokenizer.token_to_id(token).ok_or_else(|| candle_core::Error::Msg(format!("tokenizer has no {token} token")))
}

/// Per-tower standalone-GGUF-file overrides for [`DuplexSession::new_with_gguf`] —
/// `None` (the default for every field) falls back to the checkpoint's own
/// safetensors weights for that tower. See that constructor's doc for which
/// of these actually save VRAM (only `llm`) versus exist purely for
/// pure-GGUF deployment convenience (`audio`, `tts`).
#[derive(Debug, Clone, Copy, Default)]
pub struct DuplexGgufPaths<'a> {
    pub llm: Option<&'a str>,
    pub audio: Option<&'a str>,
    pub tts: Option<&'a str>,
}

impl DuplexSession {
    pub fn new(model_path: &str, device: &Device, dtype: DType, cfg: DuplexConfig) -> Result<Self> {
        Self::new_with_gguf(model_path, &DuplexGgufPaths::default(), device, dtype, cfg)
    }

    /// Same as [`Self::new`], but loads the LLM tower from a standalone
    /// quantized GGUF file instead of the checkpoint's own bf16/f32
    /// safetensors weights (see [`MiniCpmOLlm::from_gguf`]) — the other
    /// towers still load from `model_path`'s safetensors as usual.
    /// Q8_0 is roughly half the VRAM of bf16, which is the difference
    /// between a full session barely fitting on a 24GB card (near-zero
    /// headroom) and fitting comfortably. Kept as a convenience wrapper
    /// around [`Self::new_with_gguf`] — the LLM swap is the one with a real
    /// VRAM payoff (the other towers' GGUF exports are F16-only, no actual
    /// quantization, so swapping them doesn't save memory — see
    /// [`Self::new_with_gguf`]'s doc).
    ///
    /// # Errors
    ///
    /// Returns an error if the GGUF file or any other tower's weights
    /// can't be loaded.
    pub fn new_with_llm_gguf(model_path: &str, llm_gguf_path: &str, device: &Device, dtype: DType, cfg: DuplexConfig) -> Result<Self> {
        Self::new_with_gguf(model_path, &DuplexGgufPaths { llm: Some(llm_gguf_path), ..DuplexGgufPaths::default() }, device, dtype, cfg)
    }

    /// Loads any combination of towers from standalone GGUF files instead
    /// of the checkpoint's own safetensors weights — `gguf`'s `None` fields
    /// fall back to safetensors as usual. `model_path` still must point at
    /// a real checkpoint directory regardless: `tokenizer.json`/`config.json`
    /// are always read from there, and any tower left as `None` in `gguf`
    /// still loads its weights from there too.
    ///
    /// Only the LLM's GGUF (`gguf.llm`) has a real VRAM payoff (its
    /// upstream GGUF release is genuinely quantized, e.g. Q8_0 at roughly
    /// half of bf16's size). The audio/TTS GGUF releases
    /// (`openbmb/MiniCPM-o-4_5-gguf`'s `/audio`, `/tts` — vision's `/vision`
    /// GGUF is validated too, see `VisionModel::from_gguf`/
    /// `Resampler::from_gguf`, but has no consumer here since the duplex
    /// session doesn't use vision at all) are F16/F32 only — no actual
    /// quantization — so loading them from GGUF instead of safetensors
    /// doesn't reduce memory use; they exist purely for pure-GGUF
    /// deployment convenience (not needing the safetensors checkpoint's
    /// large weight files at all, just small config/tokenizer JSON
    /// alongside the GGUF weights).
    ///
    /// # Errors
    ///
    /// Returns an error if any requested GGUF file, or any safetensors
    /// fallback, can't be loaded.
    pub fn new_with_gguf(model_path: &str, gguf: &DuplexGgufPaths, device: &Device, dtype: DType, cfg: DuplexConfig) -> Result<Self> {
        let llm = match gguf.llm {
            Some(path) => MiniCpmOLlm::from_gguf(model_path, path, device).map_err(|e| candle_core::Error::Msg(e.to_string()))?,
            None => MiniCpmOLlm::new(model_path, device, &dtype).map_err(|e| candle_core::Error::Msg(e.to_string()))?,
        };

        let config_path = format!("{model_path}/config.json");
        let config: MiniCpmOConfig =
            crate::models::minicpmo::config::load_config(&config_path).map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let filenames =
            crate::utils::utils::get_safetensors_files(model_path).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&filenames, dtype, device) }?;

        let audio_output_dim = config.audio_config.encoder_ffn_dim / 4;
        let embed_dim = config.llm.hidden_size;
        let (audio_encoder, audio_projector) = match gguf.audio {
            Some(path) => {
                let mut file = std::fs::File::open(path).map_err(|e| candle_core::Error::Msg(format!("failed to open audio GGUF {path}: {e}")))?;
                let ct = candle_core::quantized::gguf_file::Content::read(&mut file)?;
                let mut gg = crate::models::hunyuan_dense::modeling::Gguf::new(ct, &mut file, device.clone(), dtype);
                let encoder = AudioEncoder::from_gguf(&mut gg, &config.audio_config)?;
                let projector = AudioProjector::from_gguf(&mut gg, config.audio_pool_step)?;
                (encoder, projector)
            }
            None => {
                let encoder = AudioEncoder::new(&config.audio_config, vb.pp("apm"))?;
                let projector = AudioProjector::new(audio_output_dim, embed_dim, config.audio_pool_step, vb.pp("audio_projection_layer"))?;
                (encoder, projector)
            }
        };
        let mel_extractor = WhisperFeatureExtractor::new(config.audio_config.num_mel_bins, device, dtype)?;

        let tts = match gguf.tts {
            Some(path) => {
                let mut file = std::fs::File::open(path).map_err(|e| candle_core::Error::Msg(format!("failed to open tts GGUF {path}: {e}")))?;
                let ct = candle_core::quantized::gguf_file::Content::read(&mut file)?;
                let mut gg = crate::models::hunyuan_dense::modeling::Gguf::new(ct, &mut file, device.clone(), dtype);
                MiniCpmTts::from_gguf(&mut gg, &config.tts_config, device, dtype)?
            }
            None => MiniCpmTts::new(&config.tts_config, vb.pp("tts"), device, dtype)?,
        };
        let token2wav = Token2Wav::new(model_path, device, dtype).map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let tok = &llm.tokenizer.tokenizer;
        let unit_token_id = get_token_or_bail(tok, "<unit>")?;
        let unit_end_token_id = get_token_or_bail(tok, "</unit>")?;
        let listen_token_id = get_token_or_bail(tok, "<|listen|>")?;
        let chunk_eos_token_id = get_token_or_bail(tok, "<|chunk_eos|>")?;
        let chunk_tts_eos_token_id = get_token_or_bail(tok, "<|chunk_tts_eos|>")?;
        let turn_eos_token_id = get_token_or_bail(tok, "<|turn_eos|>")?;
        let tts_pad_token_id = get_token_or_bail(tok, "<|tts_pad|>")?;

        let rotary = RotaryEmbedding::new(llm.head_dim(), llm.max_position_embeddings(), llm.rope_theta(), device)?;

        // Matches Python's `tokenizer.encode(marker, add_special_tokens=False)`.
        let previous_marker_token_ids: Vec<u32> = llm
            .tokenizer
            .tokenizer
            .encode("\n\nprevious: ", false)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?
            .get_ids()
            .to_vec();

        Ok(Self {
            llm,
            audio_encoder,
            audio_projector,
            mel_extractor,
            tts,
            token2wav,
            cfg,
            device: device.clone(),
            dtype,
            unit_token_id,
            unit_end_token_id,
            listen_token_id,
            chunk_eos_token_id,
            chunk_tts_eos_token_id,
            turn_eos_token_id,
            forbidden_token_ids: vec![tts_pad_token_id],
            rotary,
            system_preserve_length: 0,
            preserve_prefix_length: 0,
            previous_marker_token_ids,
            previous_token_ids: Vec::new(),
            unit_records: std::collections::VecDeque::new(),
            pending_unit_start_len: None,
            current_turn_ended: true,
            force_listen_remaining: 0,
            generated_tokens: Vec::new(),
            pending_logits: None,
            tts_start_pos: 0,
            turn_codes: Vec::new(),
        })
    }

    /// Reset session state and optionally prefill a system prompt —
    /// mirrors `MiniCPMODuplex.prepare` (text-only prefix, no reference
    /// audio support in this pass).
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or the prefill forward pass fails.
    pub fn prepare(&mut self, system_prompt: Option<&str>) -> Result<()> {
        self.llm.clear_kv_cache();
        self.current_turn_ended = true;
        self.force_listen_remaining = self.cfg.force_listen_count;
        self.generated_tokens.clear();
        self.pending_logits = None;
        self.tts.clear_kv_cache();
        self.tts_start_pos = 0;
        self.turn_codes.clear();
        self.unit_records.clear();
        self.pending_unit_start_len = None;
        self.previous_token_ids.clear();

        let prompt = system_prompt.unwrap_or("Streaming Omni Conversation.");
        let rendered = format!("<|im_start|>system\n{prompt}<|im_end|>");
        let ids = self.llm.prepare_inputs(&rendered).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        self.feed_tokens(&ids)?;
        // Mirrors `StreamDecoder.register_system_prompt`/
        // `register_system_prompt_with_context`: protect the system prompt
        // from eviction (basic mode) and anchor where "previous" content
        // gets inserted (context mode) — computed unconditionally since
        // it's cheap and mode-agnostic; only whichever mode is actually
        // selected ever reads it.
        self.system_preserve_length = self.llm.kv_cache_len();
        self.preserve_prefix_length = self.system_preserve_length;
        Ok(())
    }

    /// Current LLM KV cache length — exposed for observability/testing
    /// (e.g. confirming basic-mode sliding-window eviction actually
    /// shrinks the cache over a long session).
    #[must_use]
    pub fn kv_cache_len(&self) -> usize {
        self.llm.kv_cache_len()
    }

    fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor> {
        let ids_t = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        self.llm.embed_only(&ids_t).map_err(|e| candle_core::Error::Msg(e.to_string()))
    }

    /// Feed a batch of token ids through the LLM, advancing the KV cache.
    /// No logits/hidden state needed by callers (prefill only) — mirrors
    /// `StreamDecoder.feed` called without `return_logits`.
    ///
    /// `start_pos` is always derived fresh from the cache's *actual*
    /// current length (matching `StreamDecoder.feed`'s own
    /// `past_len = self.get_cache_length()`) rather than a separately
    /// tracked running counter — basic-mode eviction can shrink the cache
    /// mid-session, and a manually tracked position would silently drift
    /// out of sync with it.
    fn feed_tokens(&mut self, ids: &[u32]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let embeds = self.embed_tokens(ids)?;
        let start_pos = self.llm.kv_cache_len();
        self.llm.forward_embeds(&embeds, start_pos).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        Ok(())
    }

    /// Feed a caller-supplied embedding sequence (e.g. audio embeddings),
    /// advancing the KV cache and returning `(logits [vocab], hidden [1, hidden])`
    /// at the last fed position — mirrors `StreamDecoder.feed(..., return_logits=True)`.
    fn feed_embeds(&mut self, embeds: &Tensor) -> Result<(Tensor, Tensor)> {
        let start_pos = self.llm.kv_cache_len();
        let logits = self.llm.forward_embeds(embeds, start_pos).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let hidden = self.llm.last_hidden_states().ok_or_else(|| candle_core::Error::Msg("no hidden states after forward".into()))?;
        let hidden_last = hidden.narrow(1, hidden.dim(1)? - 1, 1)?;
        let logits_flat = logits.squeeze(0)?.squeeze(0)?;
        Ok((logits_flat, hidden_last))
    }

    /// One audio chunk's worth of encoder mel -> Whisper -> projector,
    /// matching `get_audio_embedding` applied to a single ~1s chunk.
    fn encode_audio_chunk(&self, samples: &[f32]) -> Result<Tensor> {
        let features = self.mel_extractor.extract(samples)?;
        let encoder_out = self.audio_encoder.forward(&features.input_features)?;
        self.audio_projector.forward(&encoder_out)
    }

    /// Prefill one audio chunk: `<unit>` token + this chunk's audio
    /// embeddings, then cache the resulting logits for the next
    /// `streaming_generate` call — mirrors `streaming_prefill`'s audio
    /// branch (vision/text input modes aren't ported in this pass).
    ///
    /// # Errors
    ///
    /// Returns an error if any forward pass fails.
    pub fn streaming_prefill(&mut self, audio_chunk: &[f32]) -> Result<()> {
        self.pending_logits = None;

        // Mirrors `StreamDecoder.register_unit_start`, called before
        // feeding anything else in the unit.
        self.pending_unit_start_len = Some(self.llm.kv_cache_len());

        self.feed_tokens(&[self.unit_token_id])?;

        let audio_embeds = self.encode_audio_chunk(audio_chunk)?.to_dtype(self.dtype)?;
        let (logits, _hidden) = self.feed_embeds(&audio_embeds)?;
        self.pending_logits = Some(logits);
        Ok(())
    }

    /// Sampling policy matching `StreamDecoder.decode`: independent
    /// `<|chunk_eos|>` short-circuit check on the *unfiltered* distribution,
    /// then forbidden-token masking + repetition penalty + listen-prob
    /// scaling before final top-k/top-p/temperature sampling.
    fn decode_policy(&self, logits: &Tensor, rng_seed: u64) -> Result<u32> {
        let mut logits: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;

        // 0. Independent chunk_eos check on the untouched distribution.
        {
            let probs = softmax(&logits);
            let sampled = sample_categorical(&probs, rng_seed);
            if sampled == self.chunk_eos_token_id {
                return Ok(self.chunk_eos_token_id);
            }
        }

        // 1. Forbidden tokens (chunk_eos already ruled out above, not masked here).
        for &id in &self.forbidden_token_ids {
            if let Some(l) = logits.get_mut(id as usize) {
                *l = f32::NEG_INFINITY;
            }
        }

        // 2. Repetition penalty — direct port of StreamDecoder.decode's
        // (deliberately naive, sign-unaware) formula: `/= penalty`
        // unconditionally, not the sign-aware standard formula Crane's own
        // `models::utils::apply_repeat_penalty` uses elsewhere — matching
        // this model's actual trained/expected sampling distribution
        // matters more here than using the "more correct" formula.
        if (self.cfg.text_repetition_penalty - 1.0).abs() > f32::EPSILON && !self.generated_tokens.is_empty() {
            let start = self.generated_tokens.len().saturating_sub(self.cfg.text_repetition_window_size);
            let mut seen = std::collections::HashSet::new();
            for &id in &self.generated_tokens[start..] {
                if !seen.insert(id) {
                    continue;
                }
                if let Some(l) = logits.get_mut(id as usize) {
                    *l /= self.cfg.text_repetition_penalty;
                }
            }
        }

        // 3. Listen-probability scaling.
        if (self.cfg.listen_prob_scale - 1.0).abs() > f64::EPSILON {
            if let Some(l) = logits.get_mut(self.listen_token_id as usize) {
                *l *= self.cfg.listen_prob_scale as f32;
            }
        }

        // 4. Final constrained sampling: temperature -> top-k -> top-p.
        for l in &mut logits {
            *l /= self.cfg.temperature as f32;
        }
        top_k_filter(&mut logits, self.cfg.top_k);
        top_p_filter(&mut logits, self.cfg.top_p as f32);
        let probs = softmax(&logits);
        Ok(sample_categorical(&probs, rng_seed.wrapping_add(1)))
    }

    /// One chunk's listen/speak arbitration + (if the turn just ended)
    /// speech synthesis for the completed turn — mirrors
    /// `streaming_generate`.
    ///
    /// # Errors
    ///
    /// Returns an error if any forward/decode/synthesis step fails.
    pub fn streaming_generate(&mut self, rng_seed: u64) -> Result<ChunkOutput> {
        let Some(mut logits) = self.pending_logits.take() else {
            return Ok(ChunkOutput { is_listen: true, text: String::new(), audio_waveform: None, end_of_turn: false });
        };

        let force_listen = self.force_listen_remaining > 0;
        self.force_listen_remaining = self.force_listen_remaining.saturating_sub(1);

        let mut is_listen = false;
        let mut end_of_turn = false;
        let mut unit_ids: Vec<u32> = Vec::new();
        let mut hidden_rows: Vec<Tensor> = Vec::new();

        for j in 0..self.cfg.max_new_speak_tokens_per_chunk {
            if j == self.cfg.max_new_speak_tokens_per_chunk - 1 {
                // Force a chunk boundary (ls_mode="explicit"): feed chunk_eos and stop.
                self.feed_tokens(&[self.chunk_eos_token_id])?;
                break;
            }

            let next_id = if force_listen {
                self.listen_token_id
            } else {
                let sampled = self.decode_policy(&logits, rng_seed.wrapping_add(j as u64))?;
                if sampled == self.listen_token_id && !self.current_turn_ended {
                    // Not allowed to interrupt an ongoing turn with <|listen|>.
                    get_token_or_bail(&self.llm.tokenizer.tokenizer, "<|tts_bos|>")?
                } else {
                    sampled
                }
            };

            if next_id != self.listen_token_id && next_id != self.chunk_eos_token_id && next_id != self.chunk_tts_eos_token_id {
                if !self.forbidden_token_ids.contains(&next_id) {
                    self.generated_tokens.push(next_id);
                }
            }
            is_listen = next_id == self.listen_token_id;

            if next_id == self.listen_token_id || next_id == self.chunk_eos_token_id || next_id == self.chunk_tts_eos_token_id {
                self.feed_tokens(&[next_id])?;
                break;
            }

            self.current_turn_ended = false;
            let (new_logits, hidden) = {
                let embeds = self.embed_tokens(&[next_id])?;
                self.feed_embeds(&embeds)?
            };
            logits = new_logits;

            end_of_turn = next_id == self.turn_eos_token_id;
            if end_of_turn {
                self.current_turn_ended = true;
            }

            if j != 0 {
                unit_ids.push(next_id);
                hidden_rows.push(hidden);
            }
        }

        self.feed_tokens(&[self.unit_end_token_id])?;

        // Mirrors `StreamDecoder.register_unit_end` + `streaming_generate`'s
        // own post-unit mode dispatch (`enforce_window`/
        // `enforce_window_with_context`) — always registered (cheap),
        // eviction only actually runs for the selected mode.
        self.register_unit_end(is_listen, &unit_ids);
        match self.cfg.sliding_window_mode {
            SlidingWindowMode::Off => {}
            SlidingWindowMode::Basic => {
                self.enforce_window()?;
            }
            SlidingWindowMode::Context => {
                self.enforce_window_with_context()?;
            }
        }

        if is_listen {
            return Ok(ChunkOutput { is_listen: true, text: String::new(), audio_waveform: None, end_of_turn: false });
        }

        let text = self.llm.tokenizer.tokenizer.decode(&unit_ids, true).unwrap_or_default();

        // Phase 6d: generate this chunk's TTS speech-token codes
        // incrementally (persistent KV cache across chunks within a turn,
        // matching `MiniCPMTTS.generate_chunk`'s real per-chunk mechanics
        // and repetition-penalty scope) rather than regenerating the whole
        // turn from scratch at the end. Token2wav synthesis itself stays
        // per-turn, not per-chunk — see the module doc's "Scope
        // simplification" note (the real streaming vocoder needs a novel
        // per-denoising-timestep-and-per-layer KV cache with no existing
        // analog in this codebase, deliberately not ported yet).
        let tts_condition = self.build_tts_condition_for_chunk(&unit_ids, &hidden_rows, end_of_turn)?;
        let condition_len = tts_condition.dim(1)?;
        let is_first_chunk_of_turn = self.tts_start_pos == 0;
        let tts_chunk_cfg = TtsGenerationConfig {
            // Force a full ~1s chunk before allowing EOS, except right at
            // the start (allow <1s of audio) or right at the end (allow
            // stopping early) — matches `streaming_generate`'s
            // `min_token_per_chunk`/`max_token_per_chunk` (25 s3tokenizer
            // codes/s + 1).
            min_new_tokens: if end_of_turn || is_first_chunk_of_turn { 0 } else { 26 },
            max_new_tokens: 26,
            // Re-seed per chunk (mixing in `rng_seed` and the running TTS
            // position) so consecutive chunks don't all sample from an
            // identical fresh RNG stream — Python's equivalent advances
            // torch's global RNG naturally across calls instead.
            seed: self.cfg.tts_cfg.seed.wrapping_add(rng_seed).wrapping_add(self.tts_start_pos as u64),
            ..self.cfg.tts_cfg.clone()
        };
        let new_codes = self.tts.generate_chunk(&tts_condition, self.tts_start_pos, &tts_chunk_cfg).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        self.tts_start_pos += condition_len + new_codes.len();
        self.turn_codes.extend(new_codes);

        if !end_of_turn {
            return Ok(ChunkOutput { is_listen: false, text, audio_waveform: None, end_of_turn: false });
        }

        // Turn complete: synthesize the whole turn's accumulated codes in
        // one non-streaming Token2wav call, then reset TTS state for the
        // next turn.
        let audio_waveform = if self.turn_codes.is_empty() { None } else { Some(self.token2wav.synthesize(&self.turn_codes, self.cfg.n_timesteps).map_err(|e| candle_core::Error::Msg(e.to_string()))?) };
        self.tts.clear_kv_cache();
        self.tts_start_pos = 0;
        self.turn_codes.clear();

        Ok(ChunkOutput { is_listen: false, text, audio_waveform, end_of_turn: true })
    }

    /// Builds one chunk's TTS conditioning sequence: `emb_text(token) +
    /// projector_semantic(hidden)` per generated position (skipped
    /// entirely if this chunk generated no response tokens — e.g. a bare
    /// `<|turn_eos|>`), a `text_eos` embed if this is the chunk that ended
    /// the turn, and always a trailing `audio_bos` embed. Mirrors
    /// `_convert_results_to_tts_input`.
    ///
    /// # Errors
    ///
    /// Returns an error if any embedding lookup fails.
    fn build_tts_condition_for_chunk(&self, chunk_tokens: &[u32], chunk_hidden_rows: &[Tensor], end_of_turn: bool) -> Result<Tensor> {
        let mut parts: Vec<Tensor> = Vec::new();
        if !chunk_tokens.is_empty() {
            let hidden_states = Tensor::cat(chunk_hidden_rows, 1)?.squeeze(0)?;
            parts.push(self.tts.build_condition_embeds(chunk_tokens, &hidden_states)?);
        }
        if end_of_turn {
            parts.push(self.tts.embed_special_token(self.tts.config.text_eos_token_id)?);
        }
        parts.push(self.tts.embed_special_token(self.tts.config.audio_bos_token_id)?);
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 1)
    }

    /// Records the just-completed unit's cache-length footprint (plus, for
    /// context mode's later use, whether it was a listen unit and what it
    /// generated) — mirrors `StreamDecoder.register_unit_end`. A no-op if
    /// `streaming_prefill` wasn't called first (matches Python's own
    /// warn-and-return guard, minus the warning).
    fn register_unit_end(&mut self, is_listen: bool, generated_tokens: &[u32]) {
        let Some(start_len) = self.pending_unit_start_len.take() else {
            return;
        };
        let current_len = self.llm.kv_cache_len();
        let unit_len = current_len.saturating_sub(start_len);
        if unit_len > 0 {
            self.unit_records.push_back(UnitRecord { length: unit_len, is_listen, generated_tokens: generated_tokens.to_vec() });
        }
    }

    /// Whether `id` is a control token that should never appear in the
    /// context-mode "previous" summary (this port's audio-only duplex
    /// session never emits image/slice tags mid-turn, unlike upstream's
    /// broader `_all_special_ids`, so this narrower set is sufficient here).
    fn is_control_token(&self, id: u32) -> bool {
        id == self.unit_token_id
            || id == self.unit_end_token_id
            || id == self.listen_token_id
            || id == self.chunk_eos_token_id
            || id == self.chunk_tts_eos_token_id
            || id == self.turn_eos_token_id
            || self.forbidden_token_ids.contains(&id)
    }

    /// Basic-mode sliding window: once the cache exceeds the high
    /// watermark, drop the oldest whole units (never touching the
    /// system-prompt-protected prefix) until it's back at or under the low
    /// watermark, or there's nothing left to drop. Mirrors
    /// `StreamDecoder.enforce_window`.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying cache-eviction tensor ops fail.
    fn enforce_window(&mut self) -> Result<bool> {
        if self.llm.kv_cache_len() <= self.cfg.basic_window_high_tokens {
            return Ok(false);
        }

        let mut dropped_any = false;
        while self.llm.kv_cache_len() > self.cfg.basic_window_low_tokens {
            let Some(record) = self.unit_records.front() else {
                break;
            };
            let length = record.length;
            let mut caches = self.llm.get_kv_caches();
            let dropped = sliding_window::drop_tokens_from_cache(&mut caches, length, self.system_preserve_length, &self.rotary)?;
            if !dropped {
                break;
            }
            self.llm.set_kv_caches(caches);
            self.unit_records.pop_front();
            dropped_any = true;
        }
        Ok(dropped_any)
    }

    /// Context-preserve sliding window: once more than `context_max_units`
    /// units are live, drop the oldest one, folding its generated
    /// (non-listen) content into the "previous" summary instead of just
    /// discarding it. Mirrors `StreamDecoder.enforce_window_with_context`.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying cache-rebuild tensor/forward ops fail.
    fn enforce_window_with_context(&mut self) -> Result<bool> {
        if self.unit_records.len() <= self.cfg.context_max_units {
            return Ok(false);
        }

        let mut dropped_any = false;
        while self.unit_records.len() > self.cfg.context_max_units {
            let Some(record) = self.unit_records.pop_front() else {
                break;
            };
            // Mirrors `StreamDecoder._extract_generated_text`: listen units
            // contribute nothing, and control tokens are stripped from
            // whatever a speaking unit generated.
            let extracted: Vec<u32> = if record.is_listen {
                Vec::new()
            } else {
                record.generated_tokens.into_iter().filter(|id| !self.is_control_token(*id)).collect()
            };
            self.update_previous_and_rebuild(&extracted)?;
            dropped_any = true;
        }
        Ok(dropped_any)
    }

    /// Folds `new_tokens` into the running "previous" summary (adding the
    /// marker on first use, truncating the oldest content once over
    /// `context_previous_max_tokens`), then rebuilds the live KV cache to
    /// match. A no-op append when `new_tokens` is empty (e.g. the dropped
    /// unit was a listen unit) — the cache still gets rebuilt, since the
    /// dropped unit's raw tokens must still be removed. Mirrors
    /// `StreamDecoder._update_previous`.
    ///
    /// # Errors
    ///
    /// Returns an error if the cache-rebuild tensor/forward ops fail.
    fn update_previous_and_rebuild(&mut self, new_tokens: &[u32]) -> Result<()> {
        append_and_truncate_previous(&mut self.previous_token_ids, &self.previous_marker_token_ids, new_tokens, self.cfg.context_previous_max_tokens);
        self.rebuild_cache_with_previous()
    }

    /// Rebuilds the live KV cache as `[fixed prefix][current "previous"
    /// tokens][remaining units]`: truncates the cache back to just the
    /// fixed system-prompt prefix, re-feeds the (possibly just-updated)
    /// "previous" tokens on top of it via an ordinary forward pass (so
    /// their KV is always freshly and correctly derived, not hand-rolled),
    /// then reattaches the surviving units' KV — realigned via
    /// `sliding_window::realign_rotary_suffix` from their old absolute
    /// position to wherever they now land after prefix+previous. Mirrors
    /// `StreamDecoder._rebuild_cache_with_previous`.
    ///
    /// # Errors
    ///
    /// Returns an error if any tensor op or the forward pass fails.
    fn rebuild_cache_with_previous(&mut self) -> Result<()> {
        let total_len = self.llm.kv_cache_len();
        let units_to_keep_len: usize = self.unit_records.iter().map(|r| r.length).sum();
        let old_units_start = total_len - units_to_keep_len;

        let full = self.llm.get_kv_caches();
        let mut prefix_only = Vec::with_capacity(full.len());
        let mut units_cache = Vec::with_capacity(full.len());
        for layer in &full {
            match layer {
                Some((k, v)) => {
                    prefix_only.push(Some((
                        k.narrow(2, 0, self.preserve_prefix_length)?.contiguous()?,
                        v.narrow(2, 0, self.preserve_prefix_length)?.contiguous()?,
                    )));
                    units_cache.push(if units_to_keep_len > 0 {
                        Some((k.narrow(2, old_units_start, units_to_keep_len)?.contiguous()?, v.narrow(2, old_units_start, units_to_keep_len)?.contiguous()?))
                    } else {
                        None
                    });
                }
                None => {
                    prefix_only.push(None);
                    units_cache.push(None);
                }
            }
        }
        drop(full);

        self.llm.set_kv_caches(prefix_only);
        let previous_tokens = self.previous_token_ids.clone();
        self.feed_tokens(&previous_tokens)?;
        let new_system_total = self.llm.kv_cache_len();

        if units_to_keep_len > 0 {
            let mut current = self.llm.get_kv_caches();
            for (layer, unit_layer) in current.iter_mut().zip(units_cache) {
                if let (Some(kv), Some((uk, uv))) = (layer.as_mut(), unit_layer) {
                    let (k, v) = kv;
                    let realigned_k = if old_units_start == new_system_total {
                        uk
                    } else {
                        sliding_window::realign_rotary_suffix(&self.rotary, &uk, old_units_start, new_system_total, units_to_keep_len)?
                    };
                    *k = Tensor::cat(&[&*k, &realigned_k], 2)?.contiguous()?;
                    *v = Tensor::cat(&[&*v, &uv], 2)?.contiguous()?;
                }
            }
            self.llm.set_kv_caches(current);
        }

        self.system_preserve_length = new_system_total;
        Ok(())
    }

}

/// Appends `new_tokens` to `previous` (adding `marker` first if `previous`
/// is currently empty), then truncates from the *left* of the content
/// (i.e. drops the oldest content, keeping the marker) if the content
/// portion (everything after `marker`) now exceeds `max_content_tokens`.
/// Mirrors `StreamDecoder._update_previous`'s append+truncate math,
/// extracted as a pure function (no cache/model access) so it can be unit
/// tested directly — see `mod tests` below.
fn append_and_truncate_previous(previous: &mut Vec<u32>, marker: &[u32], new_tokens: &[u32], max_content_tokens: usize) {
    if new_tokens.is_empty() {
        // Matches Python's `if not new_tokens and not new_text: return` —
        // no content to fold in, `previous` is left completely untouched
        // (not even the marker gets added). The caller still needs to
        // rebuild the cache regardless, since the dropped unit's raw
        // tokens must still be removed from it.
        return;
    }
    if previous.is_empty() {
        previous.extend_from_slice(marker);
    }
    previous.extend_from_slice(new_tokens);

    let marker_len = marker.len();
    let content_len = previous.len() - marker_len;
    if content_len > max_content_tokens {
        let drop = content_len - max_content_tokens;
        previous.drain(marker_len..marker_len + drop);
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|e| e / sum).collect()
}

/// Deterministic-given-seed categorical sample via a simple xorshift PRNG
/// (no new crate dependency for this) — draws a uniform `[0,1)` value and
/// walks the CDF.
fn sample_categorical(probs: &[f32], seed: u64) -> u32 {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    let uniform = (state >> 11) as f64 / (1u64 << 53) as f64;
    let mut acc = 0f64;
    for (i, &p) in probs.iter().enumerate() {
        acc += f64::from(p);
        if uniform < acc {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

fn top_k_filter(logits: &mut [f32], top_k: usize) {
    if top_k == 0 || top_k >= logits.len() {
        return;
    }
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let threshold = sorted[top_k - 1];
    for l in logits.iter_mut() {
        if *l < threshold {
            *l = f32::NEG_INFINITY;
        }
    }
}

fn top_p_filter(logits: &mut [f32], top_p: f32) {
    if top_p <= 0.0 || top_p >= 1.0 {
        return;
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let probs = softmax(logits);
    let mut cumulative = 0f32;
    let mut cutoff = idx.len();
    for (rank, &i) in idx.iter().enumerate() {
        cumulative += probs[i];
        if cumulative > top_p {
            cutoff = rank + 1;
            break;
        }
    }
    for &i in &idx[cutoff..] {
        logits[i] = f32::NEG_INFINITY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These exercise the pure sampling-policy math in isolation — the
    // riskiest new logic in this module, and the one part of phase 6a
    // that doesn't need the (currently VRAM-constrained on this box, see
    // the `minicpmo_duplex_session.rs` test) full model loaded to validate.

    #[test]
    fn softmax_sums_to_one_and_preserves_order() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax should sum to 1, got {sum}");
        assert!(probs[2] > probs[1] && probs[1] > probs[0], "order should be preserved");
    }

    #[test]
    fn softmax_is_shift_invariant() {
        let a = softmax(&[1.0, 2.0, 3.0]);
        let b = softmax(&[1001.0, 1002.0, 1003.0]);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "softmax should be invariant to a constant shift");
        }
    }

    #[test]
    fn top_k_filter_keeps_exactly_k_finite_logits() {
        let mut logits = vec![5.0, 1.0, 4.0, 2.0, 3.0];
        top_k_filter(&mut logits, 2);
        let finite = logits.iter().filter(|l| l.is_finite()).count();
        assert_eq!(finite, 2, "should keep exactly top_k finite entries");
        assert!(logits[0].is_finite() && logits[2].is_finite(), "the two largest (5.0, 4.0) should survive");
    }

    #[test]
    fn top_k_filter_noop_when_k_covers_everything() {
        let mut logits = vec![1.0, 2.0, 3.0];
        top_k_filter(&mut logits, 10);
        assert!(logits.iter().all(|l| l.is_finite()), "top_k >= len should be a no-op");
    }

    #[test]
    fn top_p_filter_keeps_at_least_the_top_token() {
        // A very small top_p should still keep at least the single most
        // likely token (matches Python's `sorted_indices_to_remove[..., 0] = 0`
        // — the top token is never removed regardless of top_p).
        let mut logits = vec![0.0, 0.0, 10.0, 0.0];
        top_p_filter(&mut logits, 0.01);
        assert!(logits[2].is_finite(), "the top token must survive even a tiny top_p");
        let finite_count = logits.iter().filter(|l| l.is_finite()).count();
        assert_eq!(finite_count, 1, "a tiny top_p with one dominant token should keep only that token");
    }

    #[test]
    fn top_p_filter_noop_at_one() {
        let mut logits = vec![1.0, 2.0, 3.0];
        top_p_filter(&mut logits, 1.0);
        assert!(logits.iter().all(|l| l.is_finite()), "top_p >= 1.0 should be a no-op");
    }

    #[test]
    fn sample_categorical_respects_zero_probability_mass() {
        // An all-mass-on-one-token distribution should always return that token.
        let probs = vec![0.0, 0.0, 1.0, 0.0];
        for seed in 0..50u64 {
            assert_eq!(sample_categorical(&probs, seed), 2, "should always sample the only nonzero-probability token");
        }
    }

    #[test]
    fn sample_categorical_is_deterministic_given_seed() {
        let probs = softmax(&[1.0, 2.0, 0.5, 3.0, 0.1]);
        let a = sample_categorical(&probs, 12345);
        let b = sample_categorical(&probs, 12345);
        assert_eq!(a, b, "same seed should produce the same sample");
    }

    #[test]
    fn append_and_truncate_previous_adds_marker_on_first_content() {
        let mut previous = Vec::new();
        append_and_truncate_previous(&mut previous, &[100, 101], &[1, 2, 3], 10);
        assert_eq!(previous, vec![100, 101, 1, 2, 3], "marker should be prepended on first content");
    }

    #[test]
    fn append_and_truncate_previous_appends_without_marker_on_subsequent_calls() {
        let mut previous = vec![100, 101, 1, 2, 3];
        append_and_truncate_previous(&mut previous, &[100, 101], &[4, 5], 10);
        assert_eq!(previous, vec![100, 101, 1, 2, 3, 4, 5], "marker must not be repeated on later appends");
    }

    #[test]
    fn append_and_truncate_previous_drops_oldest_content_keeping_marker() {
        // marker=[100,101], existing content [1,2,3], appending [4,5] makes
        // content [1,2,3,4,5] (5 tokens), which exceeds max_content_tokens=3
        // -> the oldest 2 get dropped, keeping [3,4,5].
        let mut previous = vec![100, 101, 1, 2, 3];
        append_and_truncate_previous(&mut previous, &[100, 101], &[4, 5], 3);
        assert_eq!(previous, vec![100, 101, 3, 4, 5], "should keep the marker and only the newest max_content_tokens");
    }

    #[test]
    fn append_and_truncate_previous_truncates_after_appending_new_content() {
        let mut previous = vec![100, 101, 1, 2, 3];
        append_and_truncate_previous(&mut previous, &[100, 101], &[4, 5, 6], 4);
        // content is now [1,2,3,4,5,6] (6 tokens), max 4 -> keep newest 4: [3,4,5,6]
        assert_eq!(previous, vec![100, 101, 3, 4, 5, 6]);
    }

    #[test]
    fn append_and_truncate_previous_noop_append_when_under_limit() {
        let mut previous = vec![100, 101, 1, 2];
        append_and_truncate_previous(&mut previous, &[100, 101], &[3], 10);
        assert_eq!(previous, vec![100, 101, 1, 2, 3], "no truncation needed when under the limit");
    }

    #[test]
    fn append_and_truncate_previous_empty_new_tokens_is_noop_when_previous_empty() {
        let mut previous = Vec::new();
        append_and_truncate_previous(&mut previous, &[100, 101], &[], 10);
        assert!(previous.is_empty(), "appending nothing to an empty previous should stay empty (no marker added)");
    }

    #[test]
    fn append_and_truncate_previous_empty_new_tokens_leaves_existing_content_untouched() {
        let mut previous = vec![100, 101, 1, 2, 3];
        append_and_truncate_previous(&mut previous, &[100, 101], &[], 10);
        assert_eq!(previous, vec![100, 101, 1, 2, 3], "no new content means previous is left completely unchanged, matching Python's early return");
    }

    #[test]
    fn sample_categorical_covers_the_distribution_over_many_seeds() {
        // Not a statistical rigor test, just a sanity check that varying
        // the seed actually explores more than one outcome for a
        // non-degenerate distribution.
        let probs = softmax(&[1.0, 1.0, 1.0]);
        let mut seen = std::collections::HashSet::new();
        for seed in 0..200u64 {
            seen.insert(sample_categorical(&probs, seed));
        }
        assert!(seen.len() > 1, "varying the seed should explore more than one outcome for a uniform distribution");
    }
}
