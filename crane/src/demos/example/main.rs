use crane_core::bins::load_tensors;
use crane_core::models::{DType, Device};

fn main() {
    // runing example to verify model result correctness
    // test_siglip2();
    test_snac();
}

fn test_siglip2() {
    use crane_core::models::siglip2;

    let dtype = DType::F16;
    let device = Device::Cpu;

    let model =
        siglip2::Siglip2VisionModel::from_pretrained("checkpoints/siglip2", &device, &dtype)
            .unwrap();

    // tensor pixel_values, pixel_attention_mask, spatial_shapes;
    // model.forward(pixel_values, pixel_attention_mask, spatial_shapes);
}

fn test_snac() {
    use crane_core::models::snac_onnx;

    let model = snac_onnx::SNAC24DecoderONNX::new(None, None).unwrap();

    let inputs = load_tensors("snac_codes.bin").unwrap();
    let codec0 = inputs[0].clone();
    let codec1 = inputs[1].clone();
    let codec2 = inputs[2].clone();

    println!("codec0: {:?}", codec0.shape());
    println!("codec1: {:?}", codec1.shape());
    println!("codec2: {:?}", codec2.shape());
    println!("codec0: {:?}", codec0);

    let audio = model.forward(&codec0, &codec1, &codec2).unwrap();

    // saving audio tensor to wav
    let _ = model.save_audio_data_to_file(&audio, "outputs/output.wav", None);
}
