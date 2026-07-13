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
    let output_dir = "data/audio/output";
    std::fs::create_dir_all(output_dir)?;

    // List available speakers
    let speakers: Vec<String> = model.config.talker_config.spk_id.keys().cloned().collect();
    println!("Available speakers: {speakers:?}");

    let speaker = speakers.first().cloned();
    println!("Using speaker: {}", speaker.as_deref().unwrap_or("(none)"));

    let examples: &[(&str, &str, &str)] = &[
        (
            "今天天气真好，我们去公园吧！",
            "chinese",
            "custom_voice_zh.wav",
        ),
        (
            "Hello! I am Crane, an ultra-fast inference engine written in Rust.",
            "english",
            "custom_voice_en.wav",
        ),
        (
            "こんにちは、今日はいい天気ですね。",
            "japanese",
            "custom_voice_ja.wav",
        ),
    ];

    for (i, (text, lang, filename)) in examples.iter().enumerate() {
        println!("\n[{}/{}] lang={lang}  speaker={}", i + 1, examples.len(), speaker.as_deref().unwrap_or("(none)"));
        println!("  Text: {text}");

        let start = std::time::Instant::now();
        let output_path = format!("{output_dir}/{filename}");
        let opts = crane_core::generation::SpeechOptions {
            max_new_tokens: 2048,
            temperature: 0.9,
            top_p: Some(1.0),
            repetition_penalty: 1.05,
        };
        let (audio, sr) = model.generate_speech(text, lang, speaker.as_deref(), &opts)?;
        let saved_path = crane::audio::save_wav(&audio, &output_path, sr)?;
        let elapsed = start.elapsed();
        println!("  Saved {saved_path} in {elapsed:.1?}");
    }

    println!("\nDone!");
    Ok(())
}
