//! Qwen3-TTS Simple Example (unified entry point)
//!
//! Auto-detects Base vs CustomVoice model and runs the appropriate demo.
//! For dedicated examples, see:
//! - `tts_custom_voice` — CustomVoice model with predefined speakers
//! - `tts_voice_clone`  — Base model with reference-audio voice cloning
//!
//! # Usage
//!
//! ```bash
//! # Auto-detect model variant:
//! cargo run --bin tts_simple --release -- vendor/Qwen3-TTS-12Hz-0.6B-Base
//! cargo run --bin tts_simple --release -- vendor/Qwen3-TTS-12Hz-0.6B-CustomVoice
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

    println!("Loading Qwen3-TTS from: {model_path}");
    println!("Device: {device:?}  dtype: {dtype:?}");

    let mut model = crane_core::models::qwen3_tts::Model::new(&model_path, &device, &dtype)?;
    let output_dir = "data/audio/output";
    std::fs::create_dir_all(output_dir)?;

    let model_type = model.config.tts_model_type.as_deref().unwrap_or("base").to_string();
    println!("Model type: {model_type}");

    match model_type.as_str() {
        "base" => run_voice_clone(&mut model, output_dir)?,
        "custom_voice" => run_custom_voice(&mut model, output_dir)?,
        other => anyhow::bail!("Unknown model type: {other}"),
    }

    println!("\nDone!");
    Ok(())
}

/// Voice-clone mode: Base model + reference audio.
fn run_voice_clone(model: &mut crane_core::models::qwen3_tts::Model, output_dir: &str) -> anyhow::Result<()> {
    let ref_audio = "data/audio/kinsenka_3.wav";
    let ref_text_path = "data/audio/kinsenka_3.txt";

    let ref_text = std::fs::read_to_string(ref_text_path)
        .unwrap_or_else(|_| {
            eprintln!("Warning: could not read {ref_text_path}, using empty ref_text");
            String::new()
        });
    let ref_text = ref_text.trim();

    println!("Reference audio : {ref_audio}");
    println!("Reference text  : {ref_text}");

    let examples: &[(&str, &str, &str)] = &[
        (
            "こうして君に直接ありがとうを言える時間をくれたこと それが多分一番私は嬉しい",
            "japanese",
            "simple_vc_1.wav",
        ),
        (
            "そんな何もない今日が 少しだけでもいい日になったと思えたら",
            "japanese",
            "simple_vc_2.wav",
        ),
    ];

    for (i, (text, lang, filename)) in examples.iter().enumerate() {
        println!("\n[{}/{}] lang={lang}", i + 1, examples.len());
        println!("  Text: {text}");

        let start = std::time::Instant::now();
        let output_path = format!("{output_dir}/{filename}");
        let saved_path = model.generate_voice_clone_to_file(
            text, lang, ref_audio, ref_text,
            2048, 0.9, Some(1.0), 1.05,
            &output_path,
        )?;
        let elapsed = start.elapsed();
        println!("  Saved {saved_path} in {elapsed:.1?}");
    }
    Ok(())
}

/// CustomVoice mode: predefined speaker.
fn run_custom_voice(model: &mut crane_core::models::qwen3_tts::Model, output_dir: &str) -> anyhow::Result<()> {
    let speaker = model.config.talker_config.spk_id
        .keys()
        .next()
        .cloned();
    println!("Available speakers: {:?}", model.config.talker_config.spk_id.keys().collect::<Vec<_>>());
    println!("Using speaker: {:?}", speaker);

    let examples: &[(&str, &str, &str)] = &[
        ("今天天气真好，我们去公园吧！", "chinese", "simple_tts_zh.wav"),
        ("Hello! I am Crane, an ultra-fast inference engine in Rust.", "english", "simple_tts_en.wav"),
        ("こんにちは、今日はいい天気ですね。", "japanese", "simple_tts_ja.wav"),
    ];

    for (i, (text, lang, filename)) in examples.iter().enumerate() {
        println!("\n[{}/{}] lang={lang}  speaker={}", i + 1, examples.len(), speaker.as_deref().unwrap_or("(none)"));
        println!("  Text: {text}");

        let start = std::time::Instant::now();
        let output_path = format!("{output_dir}/{filename}");
        let saved_path = model.generate_speech_to_file(
            text, lang, speaker.as_deref(), 2048, 0.9, Some(1.0), 1.05,
            &output_path,
        )?;
        let elapsed = start.elapsed();
        println!("  Saved {saved_path} in {elapsed:.1?}");
    }
    Ok(())
}
