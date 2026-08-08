//! VoxCPM2 Simple Example
//!
//! Generates speech from text using VoxCPM2 (OpenBMB): zero-shot, or with
//! reference-audio conditioning / voice cloning (all three real modes —
//! see `crane_core::models::voxcpm2::VoxCpm2Conditioning`). Streaming
//! generation is a separate, not-yet-implemented pass — see the module docs
//! on `crane_core::models::voxcpm2`.
//!
//! Expects `model.safetensors` + a pre-converted `audiovae.safetensors`
//! (converted once from the upstream `audiovae.pth`, see the module docs)
//! alongside `config.json`/`tokenizer.json` in the checkpoint directory.
//!
//! # Usage
//!
//! ```bash
//! # Zero-shot, built-in example sentences
//! cargo run --bin voxcpm2_simple --release -- checkpoints/VoxCPM2
//!
//! # Zero-shot, custom text
//! cargo run --bin voxcpm2_simple --release -- checkpoints/VoxCPM2 "Your custom text here"
//!
//! # "Controllable Cloning" — clone a voice from a clip, no transcript needed
//! cargo run --bin voxcpm2_simple --release -- checkpoints/VoxCPM2 "Text to speak" --ref-wav ref.wav
//!
//! # "Ultimate Cloning" — continue a reference clip, given its transcript
//! cargo run --bin voxcpm2_simple --release -- checkpoints/VoxCPM2 "Text to speak" \
//!     --prompt-wav ref.wav --prompt-text "What the reference clip says"
//!
//! # Combined: isolated reference prefix (timbre) + continuation suffix (content/prosody)
//! cargo run --bin voxcpm2_simple --release -- checkpoints/VoxCPM2 "Text to speak" \
//!     --ref-wav ref.wav --prompt-wav ref.wav --prompt-text "What the reference clip says"
//! ```

use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "VoxCPM2 TTS demo: zero-shot or reference-audio-conditioned (voice cloning)")]
struct Args {
    /// Path to the VoxCPM2 checkpoint directory (must contain
    /// model.safetensors, audiovae.safetensors, config.json, tokenizer.json)
    model_path: String,
    /// Text to synthesize. Falls back to two built-in example sentences in
    /// zero-shot mode if omitted; required for any conditioning mode.
    text: Option<String>,
    /// Reference audio clip for transcript-free voice cloning
    /// ("Controllable Cloning"). Combine with --prompt-wav/--prompt-text for
    /// the combined reference+continuation mode.
    #[arg(long)]
    ref_wav: Option<String>,
    /// Prompt audio clip to continue from ("Ultimate Cloning") — requires
    /// --prompt-text.
    #[arg(long)]
    prompt_wav: Option<String>,
    /// Transcript of --prompt-wav's content, prepended to `text` before
    /// tokenizing (matches the reference's own `prompt_text + target_text`).
    #[arg(long)]
    prompt_text: Option<String>,
    #[arg(long, default_value = "data/audio/output")]
    output_dir: String,
}

fn main() -> anyhow::Result<()> {
    use crane_core::models::voxcpm2::{VoxCpm2Conditioning, VoxCpm2GenerationConfig, VoxCpm2Model};
    use crane_core::models::{DType, Device};

    let args = Args::parse();

    if args.prompt_wav.is_some() != args.prompt_text.is_some() {
        anyhow::bail!("--prompt-wav and --prompt-text must be given together");
    }
    let wants_conditioning = args.ref_wav.is_some() || args.prompt_wav.is_some();
    if wants_conditioning && args.text.is_none() {
        anyhow::bail!("a text argument is required when using --ref-wav/--prompt-wav");
    }

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
        eprintln!(
            "WARNING: VoxCPM2 on CPU will be slow (multiple transformer passes per audio patch). GPU strongly recommended."
        );
    }

    println!("Loading VoxCPM2 from: {}", args.model_path);
    println!("Device: {device:?}  dtype: {dtype:?}");

    let mut model = VoxCpm2Model::new(&args.model_path, &device, &dtype)?;
    println!("Sample rate: {} Hz, encoder sample rate: {} Hz", model.sample_rate, model.encoder_sample_rate());

    std::fs::create_dir_all(&args.output_dir)?;

    let cfg = VoxCpm2GenerationConfig::default();

    if wants_conditioning {
        let text = args.text.as_deref().expect("checked above");

        let ref_feat = args
            .ref_wav
            .as_deref()
            .map(|path| -> anyhow::Result<_> {
                let samples = crane::audio::load_wav_f32(path, model.encoder_sample_rate())?;
                model.encode_reference_audio(&samples, false).map_err(anyhow::Error::from)
            })
            .transpose()?;
        let prompt = args
            .prompt_wav
            .as_deref()
            .map(|path| -> anyhow::Result<_> {
                let samples = crane::audio::load_wav_f32(path, model.encoder_sample_rate())?;
                let feat = model.encode_reference_audio(&samples, true)?;
                Ok((args.prompt_text.clone().expect("checked above"), feat))
            })
            .transpose()?;

        let conditioning = match (ref_feat, prompt) {
            (Some(r), Some((prompt_text, prompt_feat))) => {
                println!("Mode: combined reference + continuation");
                VoxCpm2Conditioning::RefContinuation { ref_feat: r, prompt_text, prompt_feat }
            }
            (Some(r), None) => {
                println!("Mode: reference-only (Controllable Cloning, no transcript)");
                VoxCpm2Conditioning::Reference(r)
            }
            (None, Some((prompt_text, prompt_feat))) => {
                println!("Mode: continuation-only (Ultimate Cloning)");
                VoxCpm2Conditioning::Continuation { prompt_text, prompt_feat }
            }
            (None, None) => unreachable!("wants_conditioning implies at least one of ref_wav/prompt_wav"),
        };

        println!("  Text: {text}");
        let start = std::time::Instant::now();
        let wav = model.generate_speech_conditioned(text, &conditioning, &cfg)?;
        let output_path = format!("{}/voxcpm2_conditioned.wav", args.output_dir);
        let saved_path = crane::audio::save_wav(&wav, &output_path, model.sample_rate)?;
        println!("  Saved {saved_path} in {:.1?}", start.elapsed());
    } else {
        let examples: Vec<(&str, &str)> = if let Some(ref text) = args.text {
            vec![(text.as_str(), "voxcpm2_custom.wav")]
        } else {
            vec![
                ("Hello! I am Crane, an ultra-fast inference engine written in Rust.", "voxcpm2_en.wav"),
                ("VoxCPM2 supports thirty languages with tokenizer-free speech generation.", "voxcpm2_en_2.wav"),
            ]
        };

        for (i, (text, filename)) in examples.iter().enumerate() {
            println!("\n[{}/{}]", i + 1, examples.len());
            println!("  Text: {text}");

            let start = std::time::Instant::now();
            let wav = model.generate_speech(text, &cfg)?;
            let output_path = format!("{}/{filename}", args.output_dir);
            let saved_path = crane::audio::save_wav(&wav, &output_path, model.sample_rate)?;
            println!("  Saved {saved_path} in {:.1?}", start.elapsed());
        }
    }

    println!("\nDone!");
    Ok(())
}
