use crane_core::models::{Device, moonshine_asr::MoonshineASR};

fn main() {
    let model_dir = "checkpoints/tiny-zh/float";
    let model_name = "tiny-zh";
    let token_rate = 16000;
    let device = Device::cuda_if_available(0).unwrap();
    let model = MoonshineASR::new(model_dir, model_name, Some(token_rate), &device).unwrap();

    let audio_f = "data/zero_shot_prompt.wav";
    let tokens = model.generate_from_audio(audio_f.to_string()).unwrap();
    println!("{:?}", tokens);
}
