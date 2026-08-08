//! End-to-end sanity check for MiniCPM-o-4.5's audio understanding path:
//! load a real checkpoint, ask a question about a real spoken audio clip,
//! verify the output is non-empty. Covers phase 3 (audio) of the module's
//! staged build — see the `minicpmo` module doc for what's implemented so far.
//!
//! Gated by `CRANE_MINICPMO_DIR` so it doesn't run by default. The audio
//! encoder + projector math was separately HF-diff validated during
//! development (see `minicpmo::audio`'s `hf_diff` test module) — this test
//! covers the mel-extraction + splicing + generation glue on top of that.

/// Linear-interpolation resample to 16kHz mono — good enough for this
/// smoke test's purposes (a real question about real speech should still
/// come through intelligibly); not a general-purpose resampler.
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
fn minicpmo_omni_generate_is_coherent() {
    use crane_core::models::minicpmo::{MiniCpmOVlModel, VlGenerationConfig};

    let dir = std::env::var("CRANE_MINICPMO_DIR").expect("set CRANE_MINICPMO_DIR to a MiniCPM-o-4.5 checkpoint dir");
    let audio_path =
        std::env::var("CRANE_MINICPMO_TEST_AUDIO").unwrap_or_else(|_| format!("{dir}/assets/haimianbaobao.wav"));

    let mut reader = hound::WavReader::open(&audio_path).unwrap_or_else(|e| panic!("open {audio_path}: {e}"));
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(Result::unwrap).collect(),
        hound::SampleFormat::Int => {
            // 2^(bits-1) as the full-scale divisor — computed directly in
            // f32 to avoid signed-integer-shift overflow (bits_per_sample=16
            // would overflow an i16 shift).
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

    let mut model = MiniCpmOVlModel::new(&dir, &device, &dtype).expect("load MiniCPM-o-4.5");

    let cfg = VlGenerationConfig { max_new_tokens: 96 };
    let text = model
        .generate(None, Some(&samples), "What is said in this audio clip? Answer in English.", &cfg, |_| {})
        .expect("generate");

    println!("generated: {text}");
    assert!(!text.trim().is_empty(), "generation produced no text");
}
