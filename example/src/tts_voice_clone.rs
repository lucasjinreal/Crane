//! Qwen3-TTS Voice Clone (Base model) Example
//!
//! Demonstrates voice cloning using the `Qwen3-TTS-12Hz-0.6B-Base` model.
//! The Base model takes a reference audio clip + its transcript, then
//! synthesizes new text in the same voice.
//!
//! # Prerequisites
//!
//! Place a reference audio file and its transcript under `data/audio/`:
//! - `data/audio/kinsenka_3.wav` — reference speaker audio (WAV, any sample rate)
//! - `data/audio/kinsenka_3.txt` — transcript of the reference audio
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin tts_voice_clone --release -- vendor/Qwen3-TTS-12Hz-0.6B-Base
//! ```

fn main() -> anyhow::Result<()> {
    use crane_core::models::{DType, Device};

    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "vendor/Qwen3-TTS-12Hz-0.6B-Base".into());

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

    println!("Loading Qwen3-TTS Base (voice clone) from: {model_path}");
    println!("Device: {device:?}  dtype: {dtype:?}");

    let mut model = crane_core::models::qwen3_tts::Model::new(&model_path, &device, &dtype)?;
    let output_dir = "data/audio/output";
    std::fs::create_dir_all(output_dir)?;

    // ── Reference audio & transcript ───────────────────────────────
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

    // ── Sentences to clone ─────────────────────────────────────────
    let examples: &[(&str, &str, &str)] = &[
        (
            "こうして君に直接ありがとうを言える時間をくれたこと それが多分一番私は嬉しい",
            "japanese",
            "vc1_base.wav",
        ),
        (
            "そんな何もない今日が 少しだけでもいい日になったと思えたら",
            "japanese",
            "vc2_base.wav",
        ),
    ];

    for (i, (text, lang, filename)) in examples.iter().enumerate() {
        println!("\n[{}/{}] lang={lang}", i + 1, examples.len());
        println!("  Text: {text}");

        let start = std::time::Instant::now();
        let output_path = format!("{output_dir}/{filename}");
        let saved_path = model.generate_voice_clone_to_file(
            text,
            lang,
            ref_audio,
            ref_text,
            2048,      // max codec tokens
            0.9,       // temperature
            Some(1.0), // top_p (matching official Python default)
            1.05,      // repetition_penalty (matching official Python default)
            &output_path,
        )?;
        let elapsed = start.elapsed();
        println!("  Saved {saved_path} in {elapsed:.1?}");
    }

    println!("\nDone!");
    Ok(())
}
