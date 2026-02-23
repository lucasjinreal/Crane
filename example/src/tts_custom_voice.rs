//! Qwen3-TTS CustomVoice Example
//!
//! Demonstrates text-to-speech with predefined speakers using
//! the `Qwen3-TTS-12Hz-0.6B-CustomVoice` model.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin tts_custom_voice --release -- vendor/Qwen3-TTS-12Hz-0.6B-CustomVoice
//! ```

fn main() -> anyhow::Result<()> {
    use crane_core::models::{DType, Device};

    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "vendor/Qwen3-TTS-12Hz-0.6B-CustomVoice".into());

    let device = {
        #[cfg(feature = "cuda")]
        { Device::new_cuda(0).unwrap_or(Device::Cpu) }
        #[cfg(all(target_os = "macos", not(feature = "cuda")))]
        { Device::new_metal(0).unwrap_or(Device::Cpu) }
        #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
        { Device::Cpu }
    };
    let dtype = {
        #[cfg(feature = "cuda")]
        { DType::BF16 }
        #[cfg(not(feature = "cuda"))]
        { DType::F32 }
    };

    println!("Loading Qwen3-TTS CustomVoice from: {model_path}");
    println!("Device: {device:?}  dtype: {dtype:?}");

    let mut model = crane_core::models::qwen3_tts::Model::new(&model_path, &device, &dtype)?;

    // List available speakers
    let speakers: Vec<String> = model.config.talker_config.spk_id.keys().cloned().collect();
    println!("Available speakers: {speakers:?}");

    let speaker = speakers.first().cloned();
    println!("Using speaker: {}", speaker.as_deref().unwrap_or("(none)"));

    let examples: &[(&str, &str, &str)] = &[
        (
            "今天天气真好，我们去公园吧！",
            "chinese",
            "output_cv_zh.wav",
        ),
        (
            "Hello! I am Crane, an ultra-fast inference engine written in Rust.",
            "english",
            "output_cv_en.wav",
        ),
        (
            "こんにちは、今日はいい天気ですね。",
            "japanese",
            "output_cv_ja.wav",
        ),
    ];

    for (i, (text, lang, filename)) in examples.iter().enumerate() {
        println!("\n[{}/{}] lang={lang}  speaker={}", i + 1, examples.len(), speaker.as_deref().unwrap_or("(none)"));
        println!("  Text: {text}");

        let start = std::time::Instant::now();
        let (audio, sr) = model.generate_speech(
            text,
            lang,
            speaker.as_deref(),
            2048,      // max codec tokens
            0.9,       // temperature
            Some(1.0), // top_p
            1.05,      // repetition_penalty
        )?;
        let elapsed = start.elapsed();

        let audio_f32 = audio.to_dtype(DType::F32)?.flatten_all()?;
        let samples = audio_f32.to_vec1::<f32>()?;
        let duration = samples.len() as f32 / sr as f32;
        println!("  Generated {duration:.1}s ({} samples @ {sr} Hz) in {elapsed:.1?}", samples.len());

        write_wav(filename, &samples, sr)?;
        println!("  Saved {filename}");
    }

    println!("\nDone!");
    Ok(())
}

/// Write a 16-bit mono WAV file from f32 samples.
fn write_wav(path: &str, samples: &[f32], sample_rate: u32) -> anyhow::Result<()> {
    use std::io::Write;

    let num_samples = samples.len() as u32;
    let data_len = num_samples * 2;
    let mut f = std::fs::File::create(path)?;

    // RIFF header
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVE")?;
    // fmt chunk
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;    // PCM
    f.write_all(&1u16.to_le_bytes())?;    // mono
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&(sample_rate * 2).to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;    // block align
    f.write_all(&16u16.to_le_bytes())?;   // bits per sample
    // data chunk
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;

    for &s in samples {
        let scaled = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        f.write_all(&scaled.to_le_bytes())?;
    }

    Ok(())
}
