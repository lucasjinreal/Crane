//! Simple Gemma 4 text generation example using crane-core directly.
//!
//! Usage:
//!   cargo run --bin gemma4_simple -- /path/to/gemma-4-E2B

use anyhow::Result;
use crane_core::generation::based::ModelForCausalLM;
use crane_core::generation::GenerationConfig;
use crane_core::models::gemma4::Model;
use crane_core::models::DType;

fn main() -> Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "model/gemma-4-E2B".to_string());

    let device = crane_core::models::Device::cuda_if_available(0)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    eprintln!("Loading Gemma 4 from: {model_path}");
    eprintln!("Device: {:?}, dtype: {:?}", device, dtype);

    let mut model = Model::new(&model_path, &device, &dtype)?;

    let prompt = "<start_of_turn>user\nHello, introduce yourself briefly.<end_of_turn>\n<start_of_turn>model\n";
    eprintln!("Prompt: {prompt}");

    let input_ids = model.prepare_inputs(prompt)?;

    let config = GenerationConfig {
        max_new_tokens: 200,
        temperature: Some(0.7),
        top_p: Some(0.9),
        report_speed: true,
        ..Default::default()
    };

    let tokens = model.generate(&input_ids, &config, None)?;

    let output = model
        .tokenizer
        .tokenizer
        .decode(&tokens[input_ids.len()..], true)
        .unwrap_or_default();
    println!("\nGenerated:\n{output}");

    Ok(())
}
