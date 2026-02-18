use anyhow::Result;
use crane_core::generation::based::ModelForCausalLM;
use crane_core::generation::streamer::TokenStreamer;
use crane_core::generation::GenerationConfig;
use crane_core::models::hunyuan_dense::Model;
use std::io::Write;

struct StdoutTokenStreamer {
    stream: crane_core::utils::token_output_stream::TokenOutputStream,
}

impl TokenStreamer for StdoutTokenStreamer {
    fn append(&mut self, token_id: u32) -> Result<()> {
        if let Some(text) = self
            .stream
            .next_token(token_id)
            .map_err(|e| anyhow::anyhow!("{e}"))?
        {
            print!("{text}");
            std::io::stdout()
                .flush()
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        if let Some(text) = self
            .stream
            .decode_rest()
            .map_err(|e| anyhow::anyhow!("{e}"))?
        {
            print!("{text}");
        }
        println!();
        Ok(())
    }
}

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

    let (mut streamer, receiver) =
        crane_core::generation::streamer::AsyncTextStreamer::with_tokenizer(
            model.tokenizer.tokenizer.clone(),
        );

    let output_ids = model.generate(&input_ids, &gen_config, Some(&mut streamer))?;

    let mut response_text = String::new();
    for message in receiver {
        match message {
            crane_core::generation::streamer::StreamerMessage::Token(token_text) => {
                print!("{token_text}");
                std::io::stdout()
                    .flush()
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                response_text.push_str(&token_text);
            }
            crane_core::generation::streamer::StreamerMessage::End => {
                println!();
                break;
            }
        }
    }

    Ok(())
}
