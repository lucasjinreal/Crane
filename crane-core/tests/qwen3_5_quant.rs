//! End-to-end generation checks for the Qwen 3.5 weight-loading paths
//! (safetensors, in-situ quantized, GGUF).
//!
//! These need local checkpoints, so they are `#[ignore]`d by default and
//! resolve their inputs from env vars:
//!
//! ```bash
//! CRANE_QWEN35_DIR=/path/to/Qwen3.5-0.8B \
//! CRANE_QWEN35_GGUF=/path/to/Qwen3.5-0.8B-Q4_0.gguf \
//!   cargo test -p crane-core --release --test qwen3_5_quant -- --ignored --nocapture
//! ```
//!
//! Greedy decoding (temperature = None) makes runs on the same machine
//! byte-comparable, which is how the LinearLayer refactor and the quantized
//! paths are validated against the bf16 baseline.

use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device};
use crane_core::generation::based::ModelForCausalLM;
use crane_core::generation::GenerationConfig;
use crane_core::models::qwen3_5::{Model, ModelFormat};

const PROMPT: &str = "<|im_start|>user\nBriefly explain what a crane (the bird) looks like.<|im_end|>\n<|im_start|>assistant\n";
const MAX_NEW_TOKENS: usize = 48;

fn device_and_dtype() -> (Device, DType) {
    #[cfg(feature = "cuda")]
    if candle_core::utils::cuda_is_available() {
        return (Device::new_cuda(0).unwrap(), DType::F16);
    }
    if candle_core::utils::metal_is_available() {
        return (Device::new_metal(0).unwrap(), DType::F16);
    }
    (Device::Cpu, DType::F32)
}

fn greedy_config() -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: MAX_NEW_TOKENS,
        temperature: None,
        top_p: None,
        report_speed: true,
        ..Default::default()
    }
}

fn run_greedy(model: &mut Model, label: &str) -> Vec<u32> {
    let input_ids = model.prepare_inputs(PROMPT).expect("tokenize prompt");
    let tokens = model
        .generate(&input_ids, &greedy_config(), None)
        .expect("generation failed");
    let generated = &tokens[input_ids.len()..];
    let text = model
        .tokenizer
        .tokenizer
        .decode(generated, true)
        .unwrap_or_default();
    println!("[{label}] {} new tokens: {generated:?}", generated.len());
    println!("[{label}] text: {text}");
    assert!(!generated.is_empty(), "no tokens generated");
    generated.to_vec()
}

fn model_dir() -> String {
    std::env::var("CRANE_QWEN35_DIR").expect("set CRANE_QWEN35_DIR to a Qwen3.5 checkpoint dir")
}

#[test]
#[ignore = "needs a local Qwen3.5 checkpoint (CRANE_QWEN35_DIR)"]
fn greedy_safetensors() {
    let (device, dtype) = device_and_dtype();
    let mut model = Model::new(&model_dir(), &device, &dtype).expect("load safetensors model");
    run_greedy(&mut model, "safetensors");
}

fn run_isq(quant: GgmlDType, label: &str) -> Vec<u32> {
    let (device, dtype) = device_and_dtype();
    let mut model = Model::new_with_options(
        &model_dir(),
        &device,
        &dtype,
        ModelFormat::Auto,
        Some(quant),
    )
    .expect("load ISQ model");
    run_greedy(&mut model, label)
}

#[test]
#[ignore = "needs a local Qwen3.5 checkpoint (CRANE_QWEN35_DIR)"]
fn greedy_isq_q8_0() {
    run_isq(GgmlDType::Q8_0, "isq-q8_0");
}

#[test]
#[ignore = "needs a local Qwen3.5 checkpoint (CRANE_QWEN35_DIR)"]
fn greedy_isq_q4k() {
    run_isq(GgmlDType::Q4K, "isq-q4k");
}

#[test]
#[ignore = "needs a local Qwen3.5 GGUF (CRANE_QWEN35_GGUF) with tokenizer.json alongside"]
fn greedy_gguf() {
    let path = std::env::var("CRANE_QWEN35_GGUF").expect("set CRANE_QWEN35_GGUF to a .gguf file");
    let (device, dtype) = device_and_dtype();
    let mut model = Model::new_with_options(&path, &device, &dtype, ModelFormat::Auto, None)
        .expect("load GGUF model");
    run_greedy(&mut model, "gguf");
}

/// Diagnostic: print every metadata key and tensor name in a GGUF file.
/// Used to pin down the converter's naming scheme for the hybrid arch.
#[test]
#[ignore = "needs a local Qwen3.5 GGUF (CRANE_QWEN35_GGUF)"]
fn dump_gguf_header() {
    let path = std::env::var("CRANE_QWEN35_GGUF").expect("set CRANE_QWEN35_GGUF to a .gguf file");
    let mut file = std::fs::File::open(&path).expect("open gguf");
    let ct = candle_core::quantized::gguf_file::Content::read(&mut file).expect("parse gguf");

    let mut keys: Vec<_> = ct.metadata.keys().collect();
    keys.sort();
    for k in keys {
        let v = &ct.metadata[k];
        let vs = format!("{v:?}");
        let vs = if vs.len() > 120 { format!("{}…", &vs[..120]) } else { vs };
        println!("meta  {k} = {vs}");
    }
    let mut names: Vec<_> = ct.tensor_infos.iter().collect();
    names.sort_by(|a, b| a.0.cmp(b.0));
    for (name, info) in names {
        println!("tensor  {name}  {:?}  {:?}", info.shape, info.ggml_dtype);
    }

    // Norm means reveal whether the converter folded the Gemma-style `+1`
    // into the stored weights (safetensors norms average ~0.24).
    for name in [
        "blk.0.attn_norm.weight",
        "blk.0.post_attention_norm.weight",
        "blk.0.ssm_norm.weight",
        "blk.3.attn_q_norm.weight",
        "output_norm.weight",
    ] {
        let t = ct
            .tensor(&mut file, name, &Device::Cpu)
            .expect("load norm tensor");
        let mean = t
            .dequantize(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        println!("norm-mean  {name} = {mean:.4}");
    }
}
