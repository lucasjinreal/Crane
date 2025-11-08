use crane_core::models::qwen25::Model;
use crane_core::models::qwen25::TextGeneration;

use crane_core::models::DType;
use crane_core::models::Device;

use clap::Parser;

#[derive(Parser, Debug)]
#[clap(about, version, author)]
struct Args {
    #[clap(short('m'), long, default_value = "checkpoints/Qwen2.5-0.5B-Instruct")]
    model_path: String,

    #[clap(short('p'), long)]
    prompt: Option<String>,
}

fn main() {
    println!("Hello, world!");
    let args = Args::parse();

    // let dtype = DType::F32;
    let dtype = DType::F16;
    let device = Device::Cpu;

    let model = Model::new(&args.model_path, &device, &dtype).unwrap();
    let tokenizer = model.tokenizer.tokenizer.clone();

    crane_core::utils::utils::print_candle_build_info();

    let mut pipe = TextGeneration::new(
        model,
        tokenizer,
        1024,
        Some(0.67),
        Some(1.0),
        1.1,
        1,
        &device,
    );

    pipe.run("who are you?", 235).unwrap();
    // pipe.run("how's your job?", 235).unwrap();
}
