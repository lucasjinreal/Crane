//! KugelAudio Simple Example
//!
//! Generates speech from text using KugelAudio (`kugelaudio/kugelaudio-0-open`,
//! a post-trained fine-tune of Microsoft's VibeVoice): zero-shot with the
//! model's built-in voice, or conditioned on a reference clip for voice
//! cloning (raw audio, resampled to 24kHz — no prompt transcript needed).
//!
//! Unlike Crane's other TTS checkpoints, KugelAudio ships **no tokenizer of
//! its own** — it needs a Qwen2-VL-family `tokenizer.json` (not plain
//! Qwen2.5's: only the Qwen2-VL vocab defines `<|vision_start/end/pad|>` at
//! the fixed ids this port hardcodes as speech control tokens). Download one
//! once, e.g.:
//!
//! ```bash
//! curl -L -o tokenizer.json \
//!   https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/resolve/main/tokenizer.json
//! ```
//!
//! See `crane_core::models::kugelaudio::prompt` for the full explanation.
//!
//! # Usage
//!
//! ```bash
//! # Zero-shot, built-in example sentence
//! cargo run --bin kugelaudio_simple --release --features cuda -- \
//!     checkpoints/kugelaudio-0-open --tokenizer tokenizer.json
//!
//! # Zero-shot, custom text
//! cargo run --bin kugelaudio_simple --release --features cuda -- \
//!     checkpoints/kugelaudio-0-open --tokenizer tokenizer.json "Your custom text here"
//!
//! # Voice cloning from a reference clip (no transcript needed)
//! cargo run --bin kugelaudio_simple --release --features cuda -- \
//!     checkpoints/kugelaudio-0-open --tokenizer tokenizer.json "Text to speak" --ref-wav ref.wav
//! ```

use clap::Parser;

#[derive(Parser, Debug)]
#[command(about = "KugelAudio TTS demo: zero-shot or reference-audio-conditioned (voice cloning)")]
struct Args {
    /// Path to the KugelAudio checkpoint directory (must contain
    /// config.json + model-*.safetensors / model.safetensors.index.json)
    model_path: String,
    /// Path to a Qwen2-VL-family tokenizer.json — see module doc comment.
    /// Defaults to `<model_path>/tokenizer.json` if present.
    #[arg(long)]
    tokenizer: Option<String>,
    /// Text to synthesize.
    #[arg(default_value = "Hello! I am Crane, an ultra-fast inference engine written in Rust.")]
    text: String,
    /// Reference audio clip for voice cloning (raw audio, resampled to
    /// 24kHz internally — no transcript needed).
    #[arg(long)]
    ref_wav: Option<String>,
    #[arg(long, default_value = "data/audio/output")]
    output_dir: String,
    /// Classifier-free guidance scale (1.0 disables CFG). Matches the
    /// checkpoint's own default.
    #[arg(long, default_value_t = 3.0)]
    cfg_scale: f64,
    #[arg(long, default_value_t = 2048)]
    max_new_tokens: usize,
    /// Sample instead of greedy-decoding the control-token stream.
    #[arg(long)]
    do_sample: bool,
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,
    /// Force CPU/F32 compute instead of the per-device default (BF16 on
    /// CUDA, F16 on Metal).
    #[arg(long)]
    cpu: bool,
    /// Seed for the diffusion-sampling noise. Defaults to a fresh random
    /// value every launch; pass a fixed value for reproducible output.
    #[arg(long)]
    seed: Option<u64>,
}

const SAMPLE_RATE: u32 = 24_000;

fn main() -> anyhow::Result<()> {
    use candle_core::{DType, Device, Tensor};
    use crane_core::models::kugelaudio::{KugelAudioGenerationConfig, KugelAudioModel, build_prompt};
    use std::time::{SystemTime, UNIX_EPOCH};
    use tokenizers::Tokenizer;

    let args = Args::parse();

    let tokenizer_path = args
        .tokenizer
        .clone()
        .unwrap_or_else(|| format!("{}/tokenizer.json", args.model_path));
    if !std::path::Path::new(&tokenizer_path).exists() {
        anyhow::bail!(
            "no tokenizer.json found at {tokenizer_path:?}. KugelAudio ships no tokenizer of its \
             own -- download a Qwen2-VL-family one and pass --tokenizer, e.g.:\n\n\
             curl -L -o tokenizer.json \
             https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/resolve/main/tokenizer.json"
        );
    }

    let (device, dtype) = if args.cpu {
        (Device::Cpu, DType::F32)
    } else {
        #[cfg(feature = "cuda")]
        {
            (Device::new_cuda(0).unwrap_or(Device::Cpu), DType::BF16)
        }
        #[cfg(all(target_os = "macos", not(feature = "cuda")))]
        {
            (Device::new_metal(0).unwrap_or(Device::Cpu), DType::F16)
        }
        #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
        {
            (Device::Cpu, DType::F32)
        }
    };
    if matches!(device, Device::Cpu) {
        eprintln!(
            "WARNING: KugelAudio on CPU will be slow (28-layer decoder + 20-step diffusion per frame). GPU strongly recommended."
        );
    } else {
        // candle's CUDA/Metal backends both default to a fixed RNG seed
        // (299792458), so the diffusion-sampling noise -- the only thing
        // that varies output when --do-sample is off, the default -- would
        // be identical on every launch without this. CPU's RNG is already
        // OS-entropy-seeded per run and errors if you try to reseed it.
        let seed = args.seed.unwrap_or_else(|| {
            let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64;
            nanos ^ (std::process::id() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        });
        println!("Seed: {seed}");
        device.set_seed(seed)?;
    }

    println!("Loading KugelAudio from: {}", args.model_path);
    println!("Device: {device:?}  dtype: {dtype:?}");
    let mut model = KugelAudioModel::from_pretrained(&args.model_path, &device, dtype)?;

    println!("Loading tokenizer from: {tokenizer_path}");
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("load tokenizer.json: {e}"))?;

    let voice_waveform = args
        .ref_wav
        .as_deref()
        .map(|path| -> anyhow::Result<_> {
            let samples = crane::audio::load_wav_f32(path, SAMPLE_RATE)?;
            let n = samples.len();
            let waveform = Tensor::from_vec(samples, (1, 1, n), &device)?;
            Ok((waveform, n))
        })
        .transpose()?;

    println!("Mode: {}", if voice_waveform.is_some() { "voice cloning (--ref-wav)" } else { "zero-shot" });
    println!("Text: {}", args.text);

    let prompt = build_prompt(&tokenizer, &args.text, voice_waveform.as_ref().map(|(_, n)| *n))?;

    let gen_cfg = KugelAudioGenerationConfig {
        cfg_scale: args.cfg_scale,
        max_new_tokens: args.max_new_tokens,
        do_sample: args.do_sample,
        temperature: args.temperature,
    };

    let start = std::time::Instant::now();
    let out = model.generate(&prompt, voice_waveform.as_ref().map(|(w, _)| w), &gen_cfg)?;
    println!(
        "Generated {} control tokens, {:.2}s audio in {:.1?}",
        out.token_ids.len(),
        out.audio.len() as f32 / SAMPLE_RATE as f32,
        start.elapsed()
    );

    if out.audio.is_empty() {
        anyhow::bail!("generation produced no audio (model emitted speech_end/eos before any diffusion token)");
    }

    std::fs::create_dir_all(&args.output_dir)?;
    let output_path = format!("{}/kugelaudio_output.wav", args.output_dir);
    let n = out.audio.len();
    let wav = Tensor::from_vec(out.audio, (1, 1, n), &device)?;
    let saved_path = crane::audio::save_wav(&wav, &output_path, SAMPLE_RATE)?;
    println!("Saved {saved_path}");

    Ok(())
}
