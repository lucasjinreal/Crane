//! MiniCPM-V-4.7 parity with HF transformers: the rendered prompt ids must be
//! identical and greedy generation must reproduce HF's text.
//!
//! Fixtures (`tests/fixtures/minicpm_v47_chart{16x,4x}.json`) come from
//! `ai/scripts47/hf_ref.py` on `data/images/test_chart.png` with the prompt
//! below (bf16, greedy, `enable_thinking=False`, `max_slice_nums=9`).
//! Gated by `CRANE_MINICPM_V47_DIR` (a MiniCPM-V-4.7-1B safetensors dir).

use crane_core::candle_core::{DType, Device};
use crane_core::models::minicpm_v::{MinicpmV46VLModel, VlGenerationConfig};

const PROMPT: &str = "Describe this image in one or two sentences.";

fn device() -> (Device, DType) {
    let f32 = std::env::var("CRANE_TEST_DTYPE").as_deref() == Ok("f32");
    #[cfg(feature = "cuda")]
    if crane_core::candle_core::utils::cuda_is_available() {
        return (
            Device::new_cuda(0).unwrap(),
            if f32 { DType::F32 } else { DType::BF16 },
        );
    }
    #[cfg(feature = "metal")]
    if crane_core::candle_core::utils::metal_is_available() {
        return (
            Device::new_metal(0).unwrap(),
            if f32 { DType::F32 } else { DType::F16 },
        );
    }
    #[cfg(feature = "sycl")]
    if let Ok(dev) = Device::new_sycl(0) {
        return (dev, if f32 { DType::F32 } else { DType::F16 });
    }
    let _ = f32;
    (Device::Cpu, DType::F32)
}

fn run(fixture: &str, mode: &str) {
    let dir = std::env::var("CRANE_MINICPM_V47_DIR").expect("set CRANE_MINICPM_V47_DIR");
    let fx: serde_json::Value = serde_json::from_str(fixture).unwrap();
    let want_ids: Vec<u32> = serde_json::from_value(fx["input_ids"].clone()).unwrap();
    let want_text = fx["gen_text"].as_str().unwrap();

    let image_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/images/test_chart.png");
    let image = image::open(image_path).unwrap();
    let (device, dtype) = device();
    let mut model = MinicpmV46VLModel::new(&dir, &device, &dtype).expect("load MiniCPM-V-4.7");

    // Prompt parity needs the per-call mode applied before rendering.
    let cfg = VlGenerationConfig {
        max_new_tokens: 64,
        strip_thinking: false,
        enable_thinking: Some(false),
        downsample_mode: Some(mode.to_string()),
        dspark: None,
        ..Default::default()
    };
    let text = model
        .generate(Some(&image), PROMPT, &cfg, |_| {})
        .expect("generate");
    let (got_ids, _) = model
        .render_prompt(Some(&image), PROMPT, Some(false))
        .unwrap();
    assert_eq!(got_ids, want_ids, "prompt ids differ from HF ({mode})");

    println!("crane: {text}\n   hf: {want_text}");
    // f32 vs HF-f32 must agree exactly; bf16 only to a common prefix, since
    // HF itself flips near-ties between bf16 and f32.
    if let Some(f32_text) = fx["gen_text_f32"].as_str()
        && std::env::var("CRANE_TEST_DTYPE").as_deref() == Ok("f32")
    {
        assert_eq!(text, f32_text, "f32 output differs from HF f32 ({mode})");
        return;
    }
    let common = text
        .chars()
        .zip(want_text.chars())
        .take_while(|(a, b)| a == b)
        .count();
    assert!(
        common >= 80.min(want_text.len()),
        "greedy output diverges from HF after {common} chars ({mode})"
    );
}

#[test]
#[ignore = "needs CRANE_MINICPM_V47_DIR"]
fn parity_chart_16x() {
    run(include_str!("fixtures/minicpm_v47_chart16x.json"), "16x");
}

#[test]
#[ignore = "needs CRANE_MINICPM_V47_DIR"]
fn parity_chart_4x() {
    run(include_str!("fixtures/minicpm_v47_chart4x.json"), "4x");
}

/// GGUF LLM + mmproj against the same HF fixtures. Quantization moves logits,
/// so only the rendered prompt must match exactly; the generated text must
/// share a long prefix with HF-f32.
fn run_gguf(fixture: &str, mode: &str) {
    let llm = std::env::var("CRANE_MINICPM_V47_GGUF").expect("set CRANE_MINICPM_V47_GGUF");
    let mmproj = std::env::var("CRANE_MINICPM_V47_MMPROJ").expect("set CRANE_MINICPM_V47_MMPROJ");
    let fx: serde_json::Value = serde_json::from_str(fixture).unwrap();
    let want_ids: Vec<u32> = serde_json::from_value(fx["input_ids"].clone()).unwrap();
    let want_text = fx["gen_text_f32"]
        .as_str()
        .or(fx["gen_text"].as_str())
        .unwrap();

    let image_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/images/test_chart.png");
    let image = image::open(image_path).unwrap();
    let (device, dtype) = device();
    let mut model =
        MinicpmV46VLModel::from_gguf(&llm, &mmproj, &device, &dtype).expect("load GGUF");
    let cfg = VlGenerationConfig {
        max_new_tokens: 64,
        strip_thinking: false,
        enable_thinking: Some(false),
        downsample_mode: Some(mode.to_string()),
        dspark: None,
        ..Default::default()
    };
    let text = model
        .generate(Some(&image), PROMPT, &cfg, |_| {})
        .expect("generate");
    let (got_ids, _) = model
        .render_prompt(Some(&image), PROMPT, Some(false))
        .unwrap();
    assert_eq!(got_ids, want_ids, "GGUF prompt ids differ from HF ({mode})");
    println!("gguf: {text}\n  hf: {want_text}");
    let common = text
        .chars()
        .zip(want_text.chars())
        .take_while(|(a, b)| a == b)
        .count();
    // Q4_K_M flips a near-tie around char 54 yet stays correct; Q8_0 tracks
    // HF-bf16 for the full 64 tokens.
    assert!(
        common >= 40,
        "GGUF output diverges from HF after {common} chars ({mode})"
    );
}

#[test]
#[ignore = "needs CRANE_MINICPM_V47_GGUF + CRANE_MINICPM_V47_MMPROJ"]
fn parity_gguf_chart_16x() {
    run_gguf(include_str!("fixtures/minicpm_v47_chart16x.json"), "16x");
}

#[test]
#[ignore = "needs CRANE_MINICPM_V47_GGUF + CRANE_MINICPM_V47_MMPROJ"]
fn parity_gguf_chart_4x() {
    run_gguf(include_str!("fixtures/minicpm_v47_chart4x.json"), "4x");
}
