//! Qwen3-TTS Simple Example
//!
//! Demonstrates text-to-speech synthesis using crane-core's Qwen3-TTS model.
//!
//! # Architecture note
//!
//! The pipeline has two independent stages:
//!
//! ```text
//! [safetensors]  Talker (28L transformer) + Code Predictor (5L transformer)
//!                ──→ generates [T × 16] codec tokens
//!
//! [safetensors]  Speech Tokenizer Decoder (Mimi RVQGAN decoder)
//!                ──→ converts codec tokens → 24 kHz waveform
//! ```
//!
//! The **Talker and Code Predictor are always loaded from safetensors** via Candle —
//! and the speech tokenizer decoder is now also loaded from safetensors in
//! native Candle. ONNX export is optional only as a compatibility fallback.
//!
//! # Setup
//!
//! ```bash
//! # Download model
//! huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
//!     --local-dir checkpoints/Qwen3-TTS-12Hz-0.6B-Base
//!
//! # (Optional fallback) export ONNX speech decoder
//! pip install -e vendor/Qwen3-TTS
//! python scripts/export_qwen_tts_tokenizer_onnx.py \
//!     checkpoints/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer \
//!     checkpoints/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer/speech_tokenizer_decoder.onnx
//!
//! # Run: full pipeline (WAV output)
//! cargo run --bin tts_simple --release
//! ```

fn main() -> anyhow::Result<()> {
    use crane_core::models::{DType, Device};

    let model_path = "checkpoints/Qwen3-TTS-12Hz-0.6B-Base";

    let device = {
        #[cfg(target_os = "macos")]
        { Device::new_metal(0).unwrap_or(Device::Cpu) }
        #[cfg(not(target_os = "macos"))]
        { Device::Cpu }
    };
    let dtype = DType::F32;

    println!("Loading Qwen3-TTS from: {model_path}");
    println!("Device: {device:?}  dtype: {dtype:?}");

    let mut model = crane_core::models::qwen3_tts::Model::new(model_path, &device, &dtype)?;

    let examples: &[(&str, &str, Option<&str>, &str)] = &[
        // (text, language, speaker, output_stem)
        ("今天天气真好，我们去公园吧！",                                    "chinese", Some("Chelsie"), "output_tts_zh"),
        ("Hello! I am Crane, an ultra-fast inference engine in Rust.", "english", Some("Ethan"),   "output_tts_en"),
        ("This is bilingual. 这是中英文混合测试。",                         "auto",    None,           "output_tts_auto"),
    ];

    for (i, (text, lang, speaker, stem)) in examples.iter().enumerate() {
        println!("\n[{}/{}] lang={lang}  speaker={}", i + 1, examples.len(), speaker.unwrap_or("default"));
        println!("  Text: {text}");

        // ── Stage 1: Talker + Code Predictor (safetensors, always available) ──
        let codes = model.generate_codes(
            text,
            lang,
            *speaker,
            2048,   // max codec tokens
            0.7,    // temperature
            None,   // top_p
            1.0,    // repetition_penalty
            None,   // system prompt
        )?;

        println!("  Generated {} codec frames ({:.1}s @ 12 Hz)", codes.len(), codes.len() as f32 / 12.0);

        let wav_path = format!("{stem}.wav");
        let pcm = model.codes_to_pcm(&codes)?;
        write_wav(&wav_path, &pcm, model.sample_rate())?;
        println!("  ✓ Saved {wav_path}  ({} bytes)", pcm.len());
    }

    println!("\nDone!");
    Ok(())
}

/// Write a minimal 16-bit mono WAV file from raw PCM bytes (little-endian i16).
fn write_wav(path: &str, pcm: &[u8], sample_rate: u32) -> anyhow::Result<()> {
    use std::io::Write;

    let data_len = pcm.len() as u32;
    let mut f = std::fs::File::create(path)?;

    // RIFF header
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVE")?;
    // fmt  chunk
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;   // chunk size
    f.write_all(&1u16.to_le_bytes())?;    // PCM
    f.write_all(&1u16.to_le_bytes())?;    // mono
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&(sample_rate * 2).to_le_bytes())?; // byte rate
    f.write_all(&2u16.to_le_bytes())?;    // block align
    f.write_all(&16u16.to_le_bytes())?;   // bits per sample
    // data chunk
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    f.write_all(pcm)?;

    Ok(())
}
