//! End-to-end sanity check for MiniCPM-o-4.5's vision understanding path:
//! load a real checkpoint, run full image-grounded generation, verify the
//! output is non-empty. Covers phase 2 (vision) of the module's staged
//! build — see the `minicpmo` module doc for what's implemented so far.
//!
//! Gated by `CRANE_MINICPMO_DIR` so it doesn't run by default. The vision
//! tower + resampler math was separately HF-diff validated during
//! development (see `minicpmo::vision`'s `hf_diff` test module) — this test
//! covers the preprocessing + splicing + generation glue on top of that.

#[test]
#[ignore = "needs a local MiniCPM-o-4.5 checkpoint (CRANE_MINICPMO_DIR)"]
fn minicpmo_vlm_generate_is_coherent() {
    use crane_core::models::minicpmo::{MiniCpmOVlModel, VlGenerationConfig};

    let dir = std::env::var("CRANE_MINICPMO_DIR").expect("set CRANE_MINICPMO_DIR to a MiniCPM-o-4.5 checkpoint dir");
    // The checkpoint ships its own README example images under assets/.
    let image_path = std::env::var("CRANE_MINICPMO_TEST_IMAGE").unwrap_or_else(|_| format!("{dir}/assets/fossil.png"));

    #[cfg(feature = "cuda")]
    let (device, dtype) = if candle_core::utils::cuda_is_available() {
        (candle_core::Device::new_cuda(0).unwrap(), candle_core::DType::BF16)
    } else {
        (candle_core::Device::Cpu, candle_core::DType::F32)
    };
    #[cfg(not(feature = "cuda"))]
    let (device, dtype) = (candle_core::Device::Cpu, candle_core::DType::F32);

    let mut model = MiniCpmOVlModel::new(&dir, &device, &dtype).expect("load MiniCPM-o-4.5");
    // MiniCPM-o-4.5 is ~9B params (16GB+ resident in bf16); on a shared GPU
    // there may not be enough headroom left for a multi-slice prefill's
    // activations. Force single-slice (whole-image, no tiling) to keep this
    // sanity check's memory footprint small — the neural net math itself
    // (including multi-slice-shaped inputs) is separately HF-diff validated.
    model.preprocessor.max_slice_nums = 1;

    let image = image::open(&image_path).unwrap_or_else(|e| panic!("open {image_path}: {e}"));

    let cfg = VlGenerationConfig { max_new_tokens: 96 };
    let text = model
        .generate(Some(&image), None, "Describe this image in one or two sentences.", &cfg, |_| {})
        .expect("generate");

    println!("generated: {text}");
    assert!(!text.trim().is_empty(), "generation produced no text");
}
