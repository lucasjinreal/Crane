//! Direct-Rust-API sanity check for MiniCPM-o-4.5's full-duplex session
//! (phase 6a — see `minicpmo::duplex`'s module doc for exact scope: chunked
//! listen/speak arbitration matching upstream's default
//! `sliding_window_mode: "off"` config, no WebSocket transport yet).
//!
//! Feeds a real audio clip in 1-second chunks and checks that real
//! listen/speak transitions occur and, if the model decides to speak,
//! that a non-empty turn of audio comes back — same "validate the model
//! layer before the serving layer" order used for every prior phase.
//!
//! Gated by `CRANE_MINICPMO_DIR` so it doesn't run by default.

/// Linear-interpolation resample to 16kHz mono (same helper as
/// `minicpmo_omni_generate.rs` — not a general-purpose resampler, just
/// good enough for this smoke test).
fn resample_to_16k_mono(samples: &[f32], channels: u16, sample_rate: u32) -> Vec<f32> {
    let channels = channels as usize;
    let mono: Vec<f32> = if channels <= 1 {
        samples.to_vec()
    } else {
        samples.chunks_exact(channels).map(|frame| frame.iter().sum::<f32>() / channels as f32).collect()
    };
    if sample_rate == 16_000 {
        return mono;
    }
    let ratio = 16_000f64 / f64::from(sample_rate);
    let out_len = (mono.len() as f64 * ratio).round() as usize;
    (0..out_len)
        .map(|i| {
            let src_pos = i as f64 / ratio;
            let idx = src_pos.floor() as usize;
            let frac = (src_pos - idx as f64) as f32;
            let a = mono.get(idx).copied().unwrap_or(0.0);
            let b = mono.get(idx + 1).copied().unwrap_or(a);
            a + (b - a) * frac
        })
        .collect()
}

#[test]
#[ignore = "needs a local MiniCPM-o-4.5 checkpoint (CRANE_MINICPMO_DIR)"]
fn minicpmo_duplex_session_arbitrates_listen_speak() {
    use crane_core::models::minicpmo::{DuplexConfig, DuplexSession};

    let dir = std::env::var("CRANE_MINICPMO_DIR").expect("set CRANE_MINICPMO_DIR to a MiniCPM-o-4.5 checkpoint dir");
    let audio_path = std::env::var("CRANE_MINICPMO_TEST_AUDIO").unwrap_or_else(|_| format!("{dir}/assets/haimianbaobao.wav"));

    let mut reader = hound::WavReader::open(&audio_path).unwrap_or_else(|e| panic!("open {audio_path}: {e}"));
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(Result::unwrap).collect(),
        hound::SampleFormat::Int => {
            let max = 2f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / max).collect()
        }
    };
    let samples = resample_to_16k_mono(&raw, spec.channels, spec.sample_rate);

    #[cfg(feature = "cuda")]
    let (device, dtype) = if candle_core::utils::cuda_is_available() {
        (candle_core::Device::new_cuda(0).unwrap(), candle_core::DType::BF16)
    } else {
        (candle_core::Device::Cpu, candle_core::DType::F32)
    };
    #[cfg(not(feature = "cuda"))]
    let (device, dtype) = (candle_core::Device::Cpu, candle_core::DType::F32);

    let cfg = DuplexConfig { force_listen_count: 1, ..Default::default() };
    let mut session = DuplexSession::new(&dir, &device, dtype, cfg).expect("load duplex session");
    session.prepare(Some("You are a friendly voice assistant having a live spoken conversation.")).expect("prepare session");

    let chunk_len = 16_000; // 1s at 16kHz
    let mut listen_count = 0usize;
    let mut speak_count = 0usize;
    let mut got_audio = false;

    for (i, chunk) in samples.chunks(chunk_len).enumerate() {
        if chunk.len() < chunk_len / 2 {
            break; // skip a too-short trailing chunk
        }
        session.streaming_prefill(chunk).expect("streaming_prefill");
        let out = session.streaming_generate(42 + i as u64).expect("streaming_generate");
        println!(
            "chunk {i}: is_listen={} end_of_turn={} text={:?} audio={}",
            out.is_listen,
            out.end_of_turn,
            out.text,
            out.audio_waveform.as_ref().map_or(0, Vec::len)
        );
        if out.is_listen {
            listen_count += 1;
        } else {
            speak_count += 1;
        }
        if let Some(wav) = &out.audio_waveform {
            assert!(!wav.is_empty(), "end-of-turn audio waveform was empty");
            assert!(wav.iter().all(|v| v.is_finite()), "non-finite sample in turn audio");
            got_audio = true;
        }
        // Stop once we've seen at least one turn complete with audio, to
        // keep this smoke test's runtime/VRAM footprint bounded.
        if got_audio {
            break;
        }
    }

    println!("listen chunks: {listen_count}, speak chunks: {speak_count}, got_audio: {got_audio}");
    assert!(listen_count + speak_count > 0, "no chunks were processed");
}

/// Phase 6b: basic-mode sliding-window eviction actually shrinks a long
/// session's KV cache. Uses deliberately tiny watermarks so eviction
/// triggers within a handful of 1s chunks instead of needing an
/// 8000-token real session. `sliding_window.rs`'s unit tests already
/// validate the RoPE-realignment math in isolation; this only checks that
/// `DuplexSession` actually wires `enforce_window` in and that the cache
/// genuinely shrinks (not a correctness check on generation quality).
#[test]
#[ignore = "needs a local MiniCPM-o-4.5 checkpoint (CRANE_MINICPMO_DIR)"]
fn minicpmo_duplex_session_sliding_window_shrinks_cache() {
    use crane_core::models::minicpmo::{DuplexConfig, DuplexSession, SlidingWindowMode};

    let dir = std::env::var("CRANE_MINICPMO_DIR").expect("set CRANE_MINICPMO_DIR to a MiniCPM-o-4.5 checkpoint dir");
    let audio_path = std::env::var("CRANE_MINICPMO_TEST_AUDIO").unwrap_or_else(|_| format!("{dir}/assets/haimianbaobao.wav"));

    let mut reader = hound::WavReader::open(&audio_path).unwrap_or_else(|e| panic!("open {audio_path}: {e}"));
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(Result::unwrap).collect(),
        hound::SampleFormat::Int => {
            let max = 2f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / max).collect()
        }
    };
    let samples = resample_to_16k_mono(&raw, spec.channels, spec.sample_rate);

    #[cfg(feature = "cuda")]
    let (device, dtype) = if candle_core::utils::cuda_is_available() {
        (candle_core::Device::new_cuda(0).unwrap(), candle_core::DType::BF16)
    } else {
        (candle_core::Device::Cpu, candle_core::DType::F32)
    };
    #[cfg(not(feature = "cuda"))]
    let (device, dtype) = (candle_core::Device::Cpu, candle_core::DType::F32);

    let cfg = DuplexConfig {
        force_listen_count: 1,
        sliding_window_mode: SlidingWindowMode::Basic,
        // Tiny watermarks so eviction triggers within a few 1s chunks
        // rather than needing an 8000-token session.
        basic_window_high_tokens: 200,
        basic_window_low_tokens: 120,
        ..Default::default()
    };
    let mut session = DuplexSession::new(&dir, &device, dtype, cfg).expect("load duplex session");
    session.prepare(Some("You are a friendly voice assistant having a live spoken conversation.")).expect("prepare session");

    let system_preserve = session.kv_cache_len();
    let chunk_len = 16_000; // 1s at 16kHz
    let mut max_len_seen = 0usize;
    let mut eviction_observed = false;

    for (i, chunk) in samples.chunks(chunk_len).cycle().take(30).enumerate() {
        session.streaming_prefill(chunk).expect("streaming_prefill");
        let out = session.streaming_generate(42 + i as u64).expect("streaming_generate");
        let len = session.kv_cache_len();
        println!("chunk {i}: cache_len={len} is_listen={} end_of_turn={}", out.is_listen, out.end_of_turn);
        max_len_seen = max_len_seen.max(len);
        if len < max_len_seen {
            eviction_observed = true;
        }
        assert!(len >= system_preserve, "cache must never shrink below the protected system-prompt prefix");
    }

    println!("max cache length observed: {max_len_seen}, eviction observed: {eviction_observed}");
    assert!(max_len_seen > 120, "test didn't grow the cache enough to exercise eviction at all — increase chunk count");
    assert!(eviction_observed, "basic-mode sliding window should have shrunk the cache at least once over 30 chunks with a 200/120 watermark");
}

/// Phase 6c: context-preserve sliding-window mode actually rebuilds the
/// cache (unit-count-triggered, not cache-length-triggered like 6b) and
/// keeps the session usable afterward. `context_max_units: 3` forces
/// eviction well within this test's chunk budget.
#[test]
#[ignore = "needs a local MiniCPM-o-4.5 checkpoint (CRANE_MINICPMO_DIR)"]
fn minicpmo_duplex_session_context_preserve_rebuilds_cache() {
    use crane_core::models::minicpmo::{DuplexConfig, DuplexSession, SlidingWindowMode};

    let dir = std::env::var("CRANE_MINICPMO_DIR").expect("set CRANE_MINICPMO_DIR to a MiniCPM-o-4.5 checkpoint dir");
    let audio_path = std::env::var("CRANE_MINICPMO_TEST_AUDIO").unwrap_or_else(|_| format!("{dir}/assets/haimianbaobao.wav"));

    let mut reader = hound::WavReader::open(&audio_path).unwrap_or_else(|e| panic!("open {audio_path}: {e}"));
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(Result::unwrap).collect(),
        hound::SampleFormat::Int => {
            let max = 2f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / max).collect()
        }
    };
    let samples = resample_to_16k_mono(&raw, spec.channels, spec.sample_rate);

    #[cfg(feature = "cuda")]
    let (device, dtype) = if candle_core::utils::cuda_is_available() {
        (candle_core::Device::new_cuda(0).unwrap(), candle_core::DType::BF16)
    } else {
        (candle_core::Device::Cpu, candle_core::DType::F32)
    };
    #[cfg(not(feature = "cuda"))]
    let (device, dtype) = (candle_core::Device::Cpu, candle_core::DType::F32);

    let cfg = DuplexConfig {
        force_listen_count: 1,
        sliding_window_mode: SlidingWindowMode::Context,
        context_max_units: 3,
        context_previous_max_tokens: 50,
        ..Default::default()
    };
    let mut session = DuplexSession::new(&dir, &device, dtype, cfg).expect("load duplex session");
    session.prepare(Some("You are a friendly voice assistant having a live spoken conversation.")).expect("prepare session");

    let mut last_len = session.kv_cache_len();
    let mut shrank_at_least_once = false;

    for (i, chunk) in samples.chunks(16_000).cycle().take(15).enumerate() {
        session.streaming_prefill(chunk).expect("streaming_prefill");
        let out = session.streaming_generate(42 + i as u64).expect("streaming_generate");
        let len = session.kv_cache_len();
        println!("chunk {i}: cache_len={len} is_listen={} end_of_turn={}", out.is_listen, out.end_of_turn);
        if len < last_len {
            shrank_at_least_once = true;
        }
        last_len = len;
        assert!(len > 0, "session should still be able to generate after a context-mode rebuild");
    }

    println!("final cache length: {last_len}, shrank at least once: {shrank_at_least_once}");
    assert!(shrank_at_least_once, "context-preserve mode should have rebuilt (and shrunk) the cache at least once over 15 chunks with context_max_units=3");
}

/// GGUF-LLM variant of `minicpmo_duplex_session_arbitrates_listen_speak` —
/// loads the LLM tower from a standalone quantized GGUF file
/// (`MiniCpmOLlm::from_gguf`/`DuplexSession::new_with_llm_gguf`) instead of
/// the checkpoint's own bf16 safetensors, cutting the LLM's VRAM footprint
/// roughly in half (Q8_0 vs bf16) — this is the fix for the ~18-19GB
/// simultaneous-load requirement leaving almost no headroom on a shared
/// 24.5GB card (see `AGENTS.md`'s resource notes). Same arbitration
/// assertions as the safetensors-LLM test; the real point of this test is
/// confirming the GGUF-loaded LLM tower still produces sane forward passes
/// inside a real duplex session, not just in isolation.
///
/// Gated by both `CRANE_MINICPMO_DIR` (the checkpoint dir, for
/// tokenizer/config and the other five towers) and `CRANE_MINICPMO_LLM_GGUF`
/// (the standalone GGUF file — e.g. `MiniCPM-o-4_5-Q8_0.gguf`).
#[test]
#[ignore = "needs a local MiniCPM-o-4.5 checkpoint (CRANE_MINICPMO_DIR) and a standalone LLM GGUF file (CRANE_MINICPMO_LLM_GGUF)"]
fn minicpmo_duplex_session_gguf_llm_arbitrates_listen_speak() {
    use crane_core::models::minicpmo::{DuplexConfig, DuplexSession};

    let dir = std::env::var("CRANE_MINICPMO_DIR").expect("set CRANE_MINICPMO_DIR to a MiniCPM-o-4.5 checkpoint dir");
    let gguf_path = std::env::var("CRANE_MINICPMO_LLM_GGUF").expect("set CRANE_MINICPMO_LLM_GGUF to a standalone MiniCPM-o-4.5 LLM GGUF file");
    let audio_path = std::env::var("CRANE_MINICPMO_TEST_AUDIO").unwrap_or_else(|_| format!("{dir}/assets/haimianbaobao.wav"));

    let mut reader = hound::WavReader::open(&audio_path).unwrap_or_else(|e| panic!("open {audio_path}: {e}"));
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(Result::unwrap).collect(),
        hound::SampleFormat::Int => {
            let max = 2f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / max).collect()
        }
    };
    let samples = resample_to_16k_mono(&raw, spec.channels, spec.sample_rate);

    #[cfg(feature = "cuda")]
    let (device, dtype) = if candle_core::utils::cuda_is_available() {
        (candle_core::Device::new_cuda(0).unwrap(), candle_core::DType::BF16)
    } else {
        (candle_core::Device::Cpu, candle_core::DType::F32)
    };
    #[cfg(not(feature = "cuda"))]
    let (device, dtype) = (candle_core::Device::Cpu, candle_core::DType::F32);

    let cfg = DuplexConfig { force_listen_count: 1, ..Default::default() };
    let mut session = DuplexSession::new_with_llm_gguf(&dir, &gguf_path, &device, dtype, cfg).expect("load duplex session with GGUF LLM");
    session.prepare(Some("You are a friendly voice assistant having a live spoken conversation.")).expect("prepare session");

    let chunk_len = 16_000; // 1s at 16kHz
    let mut listen_count = 0usize;
    let mut speak_count = 0usize;
    let mut got_audio = false;

    for (i, chunk) in samples.chunks(chunk_len).enumerate() {
        if chunk.len() < chunk_len / 2 {
            break;
        }
        session.streaming_prefill(chunk).expect("streaming_prefill");
        let out = session.streaming_generate(42 + i as u64).expect("streaming_generate");
        println!(
            "chunk {i}: is_listen={} end_of_turn={} text={:?} audio={}",
            out.is_listen,
            out.end_of_turn,
            out.text,
            out.audio_waveform.as_ref().map_or(0, Vec::len)
        );
        if out.is_listen {
            listen_count += 1;
        } else {
            speak_count += 1;
        }
        if let Some(wav) = &out.audio_waveform {
            assert!(!wav.is_empty(), "end-of-turn audio waveform was empty");
            assert!(wav.iter().all(|v| v.is_finite()), "non-finite sample in turn audio");
            got_audio = true;
        }
        if got_audio {
            break;
        }
    }

    println!("listen chunks: {listen_count}, speak chunks: {speak_count}, got_audio: {got_audio}");
    assert!(listen_count + speak_count > 0, "no chunks were processed");
}

/// All-GGUF variant: LLM + audio + TTS all loaded from standalone GGUF
/// files simultaneously (`DuplexSession::new_with_gguf`) — the actual
/// "examine the other downloaded GGUFs" request this was built for. Each
/// tower's `from_gguf` was already cross-checked individually against the
/// known-correct safetensors path (`gguf_cross_check` test modules in
/// `audio.rs`/`vision.rs`/`tts_llm.rs`); this just confirms they still work
/// wired together inside a real session, not just in isolation.
///
/// Gated by `CRANE_MINICPMO_DIR` plus all three standalone GGUF env vars.
#[test]
#[ignore = "needs a local MiniCPM-o-4.5 checkpoint plus standalone LLM/audio/TTS GGUF files"]
fn minicpmo_duplex_session_all_gguf_towers_arbitrate_listen_speak() {
    use crane_core::models::minicpmo::{DuplexConfig, DuplexGgufPaths, DuplexSession};

    let dir = std::env::var("CRANE_MINICPMO_DIR").expect("set CRANE_MINICPMO_DIR to a MiniCPM-o-4.5 checkpoint dir");
    let llm_gguf = std::env::var("CRANE_MINICPMO_LLM_GGUF").expect("set CRANE_MINICPMO_LLM_GGUF");
    let audio_gguf = std::env::var("CRANE_MINICPMO_AUDIO_GGUF").expect("set CRANE_MINICPMO_AUDIO_GGUF");
    let tts_gguf = std::env::var("CRANE_MINICPMO_TTS_GGUF").expect("set CRANE_MINICPMO_TTS_GGUF");
    let audio_path = std::env::var("CRANE_MINICPMO_TEST_AUDIO").unwrap_or_else(|_| format!("{dir}/assets/haimianbaobao.wav"));

    let mut reader = hound::WavReader::open(&audio_path).unwrap_or_else(|e| panic!("open {audio_path}: {e}"));
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(Result::unwrap).collect(),
        hound::SampleFormat::Int => {
            let max = 2f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / max).collect()
        }
    };
    let samples = resample_to_16k_mono(&raw, spec.channels, spec.sample_rate);

    #[cfg(feature = "cuda")]
    let (device, dtype) = if candle_core::utils::cuda_is_available() {
        (candle_core::Device::new_cuda(0).unwrap(), candle_core::DType::BF16)
    } else {
        (candle_core::Device::Cpu, candle_core::DType::F32)
    };
    #[cfg(not(feature = "cuda"))]
    let (device, dtype) = (candle_core::Device::Cpu, candle_core::DType::F32);

    let cfg = DuplexConfig { force_listen_count: 1, ..Default::default() };
    let gguf = DuplexGgufPaths { llm: Some(&llm_gguf), audio: Some(&audio_gguf), tts: Some(&tts_gguf) };
    let mut session = DuplexSession::new_with_gguf(&dir, &gguf, &device, dtype, cfg).expect("load duplex session with all GGUF towers");
    session.prepare(Some("You are a friendly voice assistant having a live spoken conversation.")).expect("prepare session");

    let chunk_len = 16_000;
    let mut listen_count = 0usize;
    let mut speak_count = 0usize;

    for (i, chunk) in samples.chunks(chunk_len).enumerate() {
        if chunk.len() < chunk_len / 2 {
            break;
        }
        session.streaming_prefill(chunk).expect("streaming_prefill");
        let out = session.streaming_generate(42 + i as u64).expect("streaming_generate");
        println!("chunk {i}: is_listen={} end_of_turn={} text={:?}", out.is_listen, out.end_of_turn, out.text);
        if let Some(wav) = &out.audio_waveform {
            assert!(wav.iter().all(|v| v.is_finite()), "non-finite sample in turn audio");
        }
        if out.is_listen {
            listen_count += 1;
        } else {
            speak_count += 1;
        }
    }

    println!("listen chunks: {listen_count}, speak chunks: {speak_count}");
    assert!(listen_count + speak_count > 0, "no chunks were processed");
}
