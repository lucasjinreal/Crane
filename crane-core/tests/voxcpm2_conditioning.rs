//! End-to-end sanity checks for VoxCPM2's reference-audio conditioning
//! (voice cloning) and prompt-cache paths, added alongside the generalized
//! `_inference`-style prefill (see `crane_core::models::voxcpm2::model`'s
//! `hf_diff` module for the numeric cross-check against a real Python dump —
//! this file covers the full pipeline glue on top of that, same relationship
//! `voxcpm2_generate.rs` has to `cfm.rs`'s own `hf_diff` module).
//!
//! Gated by `CRANE_VOXCPM2_DIR` + `CRANE_VOXCPM2_REF_WAV` so neither runs by
//! default.

use candle_core::{DType, Device};
use crane_core::models::voxcpm2::{VoxCpm2Conditioning, VoxCpm2GenerationConfig, VoxCpm2Model, VoxCpm2PromptCache};

fn device_and_dtype() -> (Device, DType) {
    #[cfg(feature = "cuda")]
    if candle_core::utils::cuda_is_available() {
        return (Device::new_cuda(0).unwrap(), DType::BF16);
    }
    (Device::Cpu, DType::F32)
}

/// Naive linear-interpolation resampler — good enough for a test fixture,
/// same technique already used by this codebase's other offline audio tests.
fn resample_mono(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = f64::from(to_rate) / f64::from(from_rate);
    let out_len = (samples.len() as f64 * ratio).round() as usize;
    (0..out_len)
        .map(|i| {
            let src_pos = i as f64 / ratio;
            let idx = src_pos.floor() as usize;
            let frac = (src_pos - idx as f64) as f32;
            let a = samples.get(idx).copied().unwrap_or(0.0);
            let b = samples.get(idx + 1).copied().unwrap_or(a);
            a + (b - a) * frac
        })
        .collect()
}

fn load_ref_samples(path: &str, target_sr: u32) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let spec = reader.spec();
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(Result::unwrap).collect(),
        hound::SampleFormat::Int => {
            let max = 2f32.powi(i32::from(spec.bits_per_sample) - 1);
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / max).collect()
        }
    };
    let mono: Vec<f32> = if spec.channels > 1 {
        raw.chunks_exact(spec.channels as usize).map(|c| c.iter().sum::<f32>() / f32::from(spec.channels)).collect()
    } else {
        raw
    };
    resample_mono(&mono, spec.sample_rate, target_sr)
}

fn assert_well_formed(wav: &candle_core::Tensor, label: &str) {
    let dims = wav.dims();
    println!("{label} wav shape: {dims:?}");
    assert_eq!(dims[0], 1);
    assert_eq!(dims[1], 1);
    assert!(dims[2] > 0, "{label}: empty audio");
    let samples: Vec<f32> = wav.flatten_all().unwrap().to_vec1().unwrap();
    assert!(samples.iter().all(|v| v.is_finite()), "{label}: non-finite sample");
    assert!(samples.iter().all(|v| (-1.0..=1.0).contains(v)), "{label}: sample outside tanh range");
    let max_abs = samples.iter().fold(0f32, |a, &b| a.max(b.abs()));
    assert!(max_abs > 0.01, "{label}: near-silence (max_abs={max_abs})");
}

#[test]
#[ignore = "needs a local VoxCPM2 checkpoint (CRANE_VOXCPM2_DIR) and a reference wav (CRANE_VOXCPM2_REF_WAV)"]
fn voxcpm2_continuation_mode_voice_clone_is_well_formed() {
    let dir = std::env::var("CRANE_VOXCPM2_DIR").expect("set CRANE_VOXCPM2_DIR to a VoxCPM2 checkpoint dir");
    let ref_wav = std::env::var("CRANE_VOXCPM2_REF_WAV").expect("set CRANE_VOXCPM2_REF_WAV to a reference wav");

    let (device, dtype) = device_and_dtype();
    let mut model = VoxCpm2Model::new(&dir, &device, &dtype).expect("load VoxCPM2");

    let samples = load_ref_samples(&ref_wav, model.encoder_sample_rate());
    let prompt_feat = model.encode_reference_audio(&samples, true).expect("encode_reference_audio");

    let conditioning = VoxCpm2Conditioning::Continuation {
        prompt_text: "This is what the reference clip says.".to_string(),
        prompt_feat,
    };
    let cfg = VoxCpm2GenerationConfig { max_len: 200, ..Default::default() };
    let wav = model
        .generate_speech_conditioned("Now the cloned voice says something new.", &conditioning, &cfg)
        .expect("generate_speech_conditioned (continuation mode)");
    assert_well_formed(&wav, "continuation-mode");
}

#[test]
#[ignore = "needs a local VoxCPM2 checkpoint (CRANE_VOXCPM2_DIR) and a reference wav (CRANE_VOXCPM2_REF_WAV)"]
fn voxcpm2_reference_only_mode_is_well_formed() {
    let dir = std::env::var("CRANE_VOXCPM2_DIR").expect("set CRANE_VOXCPM2_DIR to a VoxCPM2 checkpoint dir");
    let ref_wav = std::env::var("CRANE_VOXCPM2_REF_WAV").expect("set CRANE_VOXCPM2_REF_WAV to a reference wav");

    let (device, dtype) = device_and_dtype();
    let mut model = VoxCpm2Model::new(&dir, &device, &dtype).expect("load VoxCPM2");

    let samples = load_ref_samples(&ref_wav, model.encoder_sample_rate());
    let ref_feat = model.encode_reference_audio(&samples, false).expect("encode_reference_audio");

    let cfg = VoxCpm2GenerationConfig { max_len: 200, ..Default::default() };
    let wav = model
        .generate_speech_conditioned(
            "This voice was cloned without needing any transcript.",
            &VoxCpm2Conditioning::Reference(ref_feat),
            &cfg,
        )
        .expect("generate_speech_conditioned (reference mode)");
    assert_well_formed(&wav, "reference-only-mode");
}

#[test]
#[ignore = "needs a local VoxCPM2 checkpoint (CRANE_VOXCPM2_DIR) and a reference wav (CRANE_VOXCPM2_REF_WAV)"]
fn voxcpm2_prompt_cache_build_merge_generate_round_trips() {
    let dir = std::env::var("CRANE_VOXCPM2_DIR").expect("set CRANE_VOXCPM2_DIR to a VoxCPM2 checkpoint dir");
    let ref_wav = std::env::var("CRANE_VOXCPM2_REF_WAV").expect("set CRANE_VOXCPM2_REF_WAV to a reference wav");

    let (device, dtype) = device_and_dtype();
    let mut model = VoxCpm2Model::new(&dir, &device, &dtype).expect("load VoxCPM2");

    let samples = load_ref_samples(&ref_wav, model.encoder_sample_rate());
    let prompt_feat = model.encode_reference_audio(&samples, true).expect("encode_reference_audio");
    let cache = VoxCpm2PromptCache::build(None, Some(("Reference transcript.".to_string(), prompt_feat)))
        .expect("build_prompt_cache");

    let cfg = VoxCpm2GenerationConfig { max_len: 200, ..Default::default() };
    let turn1_text = "First generated turn.";
    let (wav1, gen_feat1) =
        model.generate_with_prompt_cache(turn1_text, Some(&cache), &cfg).expect("generate_with_prompt_cache turn 1");
    assert_well_formed(&wav1, "prompt-cache turn 1");

    let merged = model.merge_prompt_cache(Some(&cache), turn1_text, &gen_feat1).expect("merge_prompt_cache");
    assert!(merged.prompt_text.ends_with(turn1_text));

    let (wav2, _gen_feat2) = model
        .generate_with_prompt_cache("Second generated turn, from the merged cache.", Some(&merged), &cfg)
        .expect("generate_with_prompt_cache turn 2");
    assert_well_formed(&wav2, "prompt-cache turn 2");
}
