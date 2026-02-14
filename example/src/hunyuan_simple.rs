use anyhow::Result;
use crane_core::generation::based::ModelForCausalLM;
use crane_core::generation::GenerationConfig;
use crane_core::models::hunyuan_dense::Model;

fn main() -> Result<()> {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "model/Hunyuan-0.5B-Instruct".to_string());

    println!("Loading model from: {}", model_path);

    #[cfg(feature = "cuda")]
    let device = crane_core::models::Device::cuda_if_available(0)?;
    #[cfg(not(feature = "cuda"))]
    let device = crane_core::models::Device::Cpu;

    // CPU only supports F32 matmul; use BF16 on CUDA (4090 natively supports BF16)
    #[cfg(feature = "cuda")]
    let dtype = crane_core::models::DType::BF16;
    #[cfg(not(feature = "cuda"))]
    let dtype = crane_core::models::DType::F32;

    println!("Device: {:?}, dtype: {:?}", device, dtype);

    let mut model = Model::new(&model_path, &device, &dtype)?;
    println!("Model loaded successfully!");

    model.warmup();
    println!("Model warmed up.");

    let prompt = "Hello, who are you?";
    // Apply Hunyuan chat template: <BOS><User>message<Assistant>
    let input_ids = model.prepare_chat(prompt)?;
    println!("Prompt: {}", prompt);
    println!("Input IDs: {:?}", &input_ids[..input_ids.len().min(20)]);

    let gen_config = GenerationConfig {
        max_new_tokens: 128,
        temperature: Some(0.7),
        top_p: Some(0.9),
        eos_token_id: Some(120020),
        report_speed: true,
        ..Default::default()
    };

    println!("\nGenerating...");
    let output_ids = model.generate(&input_ids, &gen_config, None)?;

    let decoded = model
        .tokenizer
        .tokenizer
        .decode(&output_ids, true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    println!("\nFull output: {}", decoded);

    Ok(())
}
