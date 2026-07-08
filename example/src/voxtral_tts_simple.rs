//! Voxtral TTS Simple Example
//!
//! Generates speech from text using Voxtral-4B-TTS.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin voxtral_tts_simple --release -- checkpoints/Voxtral-4B-TTS-2603
//! ```

fn main() -> anyhow::Result<()> {
    use crane_core::models::{DType, Device};

    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "checkpoints/Voxtral-4B-TTS-2603".into());

    let device = {
        #[cfg(feature = "cuda")]
        {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        }
        #[cfg(all(target_os = "macos", not(feature = "cuda")))]
        {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        }
        #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
        {
            Device::Cpu
        }
    };
    let dtype = {
        #[cfg(feature = "cuda")]
        {
            DType::BF16
        }
        #[cfg(not(feature = "cuda"))]
        {
            DType::F32
        }
    };

    if matches!(device, Device::Cpu) {
        eprintln!("WARNING: Voxtral TTS on CPU will be extremely slow (14 transformer passes per frame). GPU strongly recommended.");
    }

    println!("Loading Voxtral TTS from: {model_path}");
    println!("Device: {device:?}  dtype: {dtype:?}");

    let mut model = crane_core::models::voxtral_tts::Model::new(&model_path, &device, &dtype)?;

    let voices = model.available_voices();
    println!("Available voices: {voices:?}");
    println!("Sample rate: {} Hz", model.sample_rate());

    let output_dir = "data/audio/output";
    std::fs::create_dir_all(output_dir)?;

    let examples: &[(&str, Option<&str>, &str)] = &[
        (
            "Hello! I am Crane, an ultra-fast inference engine written in Rust.",
            Some("neutral_female"),
            "voxtral_en_female.wav",
        ),
        (
            "Bonjour, je suis un modèle de synthèse vocale.",
            Some("fr_male"),
            "voxtral_fr_male.wav",
        ),
        (
            "Hallo, dies ist ein Test der Sprachsynthese.",
            Some("de_female"),
            "voxtral_de_female.wav",
        ),
    ];

    for (i, (text, voice, filename)) in examples.iter().enumerate() {
        println!(
            "\n[{}/{}] voice={:?}",
            i + 1,
            examples.len(),
            voice.unwrap_or("default")
        );
        println!("  Text: {text}");

        let start = std::time::Instant::now();
        let output_path = format!("{output_dir}/{filename}");
        let opts = crane_core::generation::SpeechOptions {
            max_new_tokens: 2048,
            ..Default::default()
        };
        let saved_path = model.generate_speech_to_file(text, "auto", *voice, &opts, &output_path)?;
        let elapsed = start.elapsed();
        println!("  Saved {saved_path} in {elapsed:.1?}");
    }

    println!("\nDone!");
    Ok(())
}
