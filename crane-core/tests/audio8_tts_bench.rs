//! Repeatable latency benchmark for Audio8-TTS's ONNX evaluator path.
//! Not a correctness test (only sanity assertions on the output shape) —
//! prints wall-clock/RTF for manual before/after comparison across
//! evaluator changes. Fixed text and `max_new_tokens` so successive runs
//! are directly comparable; `generate_speech` also samples from a fixed
//! internal seed, so the generated trajectory (and therefore the amount of
//! work done) is identical across runs, isolating the timing delta to the
//! evaluator itself rather than sampling variance.
//!
//! Gated by `CRANE_AUDIO8_DIR` (falls back to `models/tts/Audio8-TTS-0.1B-ONNX-INT8`
//! next to the workspace root, matching `audio8_tts::model`'s own test convention).
//!
//! Run with (`--release` matters: a debug build is 3-10x slower and not
//! comparable to prior measurements):
//! `cargo test -p crane-core --features onnx --release --test audio8_tts_bench -- --ignored --nocapture`

use crane_core::generation::SpeechOptions;
use crane_core::models::audio8_tts::Model;
use std::time::Instant;

fn model_dir() -> Option<std::path::PathBuf> {
    let dir = if let Ok(dir) = std::env::var("CRANE_AUDIO8_DIR") {
        std::path::PathBuf::from(dir)
    } else {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .join("models/tts/Audio8-TTS-0.1B-ONNX-INT8")
    };
    dir.is_dir().then_some(dir)
}

#[test]
#[ignore = "requires the downloaded Audio8-TTS-0.1B-ONNX-INT8 package; profiling only, run with --release --nocapture"]
fn audio8_tts_generation_latency() {
    let dir = model_dir().expect("set CRANE_AUDIO8_DIR or download the model");

    let t0 = Instant::now();
    let mut model = Model::new(dir.to_str().unwrap()).expect("load model");
    println!("[load] session construction: {:?}", t0.elapsed());

    let opts = SpeechOptions {
        max_new_tokens: 256,
        ..Default::default()
    };
    let t0 = Instant::now();
    let (audio, sample_rate) = model
        .generate_speech("Hello from Crane.", "en", None, &opts)
        .expect("generate_speech");
    let elapsed = t0.elapsed();

    assert_eq!(sample_rate, 44100);
    assert!(audio.elem_count() > 0);

    let audio_secs = audio.elem_count() as f64 / f64::from(sample_rate);
    let rtf = elapsed.as_secs_f64() / audio_secs;
    println!(
        "[generate] {:.2}s wall clock -> {:.2}s audio (RTF {:.1}, target <= 1.0)",
        elapsed.as_secs_f64(),
        audio_secs,
        rtf,
    );
}
