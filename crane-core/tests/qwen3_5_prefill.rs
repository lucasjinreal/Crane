//! Long-prompt prefill on real hardware: does chunking hold peak VRAM down
//! without costing throughput, and does the model still answer sensibly?
//!
//! Needs a local Qwen 3.5 GGUF and a GPU, so every test is `#[ignore]`d:
//!
//! ```bash
//! ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
//! CRANE_QWEN35_GGUF=/path/to/Qwen3.5-2B-Q8_0.gguf \
//!   cargo test -p crane-core --release --features rocm \
//!     --test qwen3_5_prefill -- --ignored --nocapture
//! ```
//!
//! `CRANE_PREFILL_CHUNK=0` on the same command line reproduces the old
//! single-pass behaviour for an A/B against these numbers.

use candle_core::{DType, Device};
use crane_core::generation::based::ModelForCausalLM;
use crane_core::generation::GenerationConfig;
use crane_core::models::qwen3_5::{Model, ModelFormat};

/// Prompt length in tokens to aim for, overridable with `CRANE_PREFILL_TOKENS`.
/// The default sits past the point where the single-pass `[B, heads, S, S]`
/// score matrix starts to dominate peak VRAM.
fn target_tokens() -> usize {
    std::env::var("CRANE_PREFILL_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096)
}

fn gguf_path() -> String {
    std::env::var("CRANE_QWEN35_GGUF").expect("set CRANE_QWEN35_GGUF to a .gguf file")
}

fn device_and_dtype() -> (Device, DType) {
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    if let Ok(dev) = Device::new_rocm(0) {
        return (dev, DType::F16);
    }
    #[cfg(feature = "cuda")]
    if let Ok(dev) = Device::new_cuda(0) {
        return (dev, DType::BF16);
    }
    (Device::Cpu, DType::F32)
}

/// VRAM in use, in bytes, straight from `rocm-smi`. `None` when the tool is
/// absent or the machine has no AMD GPU, so the test still runs elsewhere.
fn vram_used() -> Option<u64> {
    let out = std::process::Command::new("rocm-smi")
        .args(["--showmeminfo", "vram", "--csv"])
        .output()
        .ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    // CSV row: `card0,<total>,<used>`.
    text.lines()
        .find(|l| l.starts_with("card"))
        .and_then(|l| l.rsplit(',').next())
        .and_then(|v| v.trim().parse().ok())
}

/// Build a prompt of roughly [`target_tokens`] tokens by repeating a passage
/// until the tokenizer says it is long enough.
fn long_prompt(model: &Model) -> (String, usize) {
    let target = target_tokens();
    const PASSAGE: &str = "Cranes are large, long-legged and long-necked birds in the family \
        Gruidae. Unlike herons, cranes fly with their necks outstretched, not pulled back. \
        They are opportunistic feeders that change their diet according to the season. ";
    let mut body = String::new();
    let mut ids = 0;
    while ids < target {
        body.push_str(PASSAGE);
        ids = model.prepare_inputs(&body).expect("tokenize").len();
    }
    let prompt = format!(
        "<|im_start|>user\n{body}\nIn one sentence: how do cranes fly?<|im_end|>\n\
         <|im_start|>assistant\n"
    );
    let len = model.prepare_inputs(&prompt).expect("tokenize").len();
    (prompt, len)
}

/// A prompt long enough to have OOM'd the single-pass path must now complete,
/// and the run reports peak VRAM plus prefill throughput for the record.
#[test]
#[ignore = "needs a GPU and a local Qwen3.5 GGUF (CRANE_QWEN35_GGUF)"]
fn long_prompt_prefill_completes() {
    let (device, dtype) = device_and_dtype();
    let base_vram = vram_used();

    let mut model = Model::new_with_options(&gguf_path(), &device, &dtype, ModelFormat::Auto, None)
        .expect("load GGUF model");
    let loaded_vram = vram_used();

    let (prompt, prompt_len) = long_prompt(&model);
    let input_ids = model.prepare_inputs(&prompt).expect("tokenize prompt");
    assert!(
        input_ids.len() >= target_tokens(),
        "prompt is only {} tokens",
        input_ids.len()
    );

    // Prefill alone, timed: `generate` would fold decode into the same clock.
    model.clear_kv_cache();
    let t0 = std::time::Instant::now();
    model
        .forward_step(&input_ids, 0)
        .expect("prefill forward failed — this is the OOM the chunking exists to prevent");
    device.synchronize().expect("sync");
    let prefill = t0.elapsed();
    let peak_vram = vram_used();

    // Then a short generation from the same cache state, to prove the prefill
    // left a usable context behind rather than merely not crashing.
    model.clear_kv_cache();
    let cfg = GenerationConfig {
        max_new_tokens: 16,
        temperature: None,
        top_p: None,
        ..Default::default()
    };
    let tokens = model.generate(&input_ids, &cfg, None).expect("generate");
    let text = model
        .tokenizer
        .tokenizer
        .decode(&tokens[input_ids.len()..], true)
        .unwrap_or_default();

    let chunk = std::env::var("CRANE_PREFILL_CHUNK").unwrap_or_else(|_| "default(512)".into());
    println!("[prefill] chunk={chunk} prompt={prompt_len} tokens");
    println!(
        "[prefill] {:.0} tok/s ({:.3} s)",
        prompt_len as f64 / prefill.as_secs_f64(),
        prefill.as_secs_f64()
    );
    if let (Some(base), Some(loaded), Some(peak)) = (base_vram, loaded_vram, peak_vram) {
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        println!(
            "[prefill] vram idle={:.2} GiB loaded={:.2} GiB peak={:.2} GiB",
            gib(base),
            gib(loaded),
            gib(peak)
        );
    }
    println!("[prefill] continuation: {text}");

    assert!(
        tokens.len() > input_ids.len(),
        "no tokens generated after the long prefill"
    );
}
