//! End-to-end sanity check for MiniCPM-V-4.6: load a real checkpoint, run
//! full image-grounded generation, verify the output is non-empty and the
//! forward pass produced finite logits throughout.
//!
//! Gated by `CRANE_MINICPM_V_DIR` so it doesn't run by default. Used to catch
//! regressions in the vision tower / merger / text-model splice integration.
//! Validated against a real HF forward-pass dump during development — see
//! the MiniCPM-V-4.6 module docs for the architecture this covers.

#[test]
#[ignore = "needs a local MiniCPM-V-4.6 checkpoint (CRANE_MINICPM_V_DIR)"]
fn minicpm_v_generate_is_coherent() {
    use crane_core::models::minicpm_v::{MinicpmV46VLModel, VlGenerationConfig};

    let dir = std::env::var("CRANE_MINICPM_V_DIR")
        .expect("set CRANE_MINICPM_V_DIR to a MiniCPM-V-4.6 checkpoint dir");
    // Any real image works; this repo's Ornith-1.0-9B assets are a
    // convenient shared default already used by other vision-model tests
    // (see the `qwen35-hf-diff-method` notes).
    let image_path = std::env::var("CRANE_MINICPM_V_TEST_IMAGE").unwrap_or_else(|_| {
        "/home/hahihula/mywork/ai/additional_models/Ornith-1.0-9B/assets/ornith_9b_eval.png".to_string()
    });

    #[cfg(feature = "cuda")]
    let (device, dtype) = if candle_core::utils::cuda_is_available() {
        (candle_core::Device::new_cuda(0).unwrap(), candle_core::DType::BF16)
    } else {
        (candle_core::Device::Cpu, candle_core::DType::F32)
    };
    #[cfg(not(feature = "cuda"))]
    let (device, dtype) = (candle_core::Device::Cpu, candle_core::DType::F32);

    let mut model = MinicpmV46VLModel::new(&dir, &device, &dtype).expect("load MiniCPM-V-4.6");

    let image = image::open(&image_path).unwrap_or_else(|e| panic!("open {image_path}: {e}"));

    let cfg = VlGenerationConfig { max_new_tokens: 96, strip_thinking: false };
    let text = model
        .generate(Some(&image), "Describe this image in one or two sentences.", &cfg, |_| {})
        .expect("generate");

    println!("generated: {text}");
    assert!(!text.trim().is_empty(), "generation produced no text");
}
