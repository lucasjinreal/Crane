//! DSpark speculative decoding on MiniCPM-V-4.7: the speculative stream must
//! equal plain greedy decoding (up to floating-point noise from the batched
//! verification pass) while taking fewer target passes.
//!
//! Gated by `CRANE_MINICPM_V47_DIR` (or `CRANE_MINICPM_V47_GGUF` +
//! `CRANE_MINICPM_V47_MMPROJ`) + `CRANE_DSPARK_DIR` (a drafter directory or a
//! `dspark-*.gguf`). Runs on CUDA (bf16), Metal (f16) or the CPU (f32).

use crane_core::candle_core::{DType, Device};
use crane_core::models::minicpm_v::{MinicpmV46VLModel, VlGenerationConfig};

/// The first GPU this build supports, with its native activation dtype.
fn device() -> (Device, DType) {
    #[cfg(feature = "cuda")]
    if crane_core::candle_core::utils::cuda_is_available() {
        return (Device::new_cuda(0).unwrap(), DType::BF16);
    }
    #[cfg(feature = "metal")]
    if crane_core::candle_core::utils::metal_is_available() {
        return (Device::new_metal(0).unwrap(), DType::F16);
    }
    #[cfg(feature = "sycl")]
    if let Ok(dev) = Device::new_sycl(0) {
        return (dev, DType::F16);
    }
    (Device::Cpu, DType::F32)
}

/// Target + drafter: safetensors from `CRANE_MINICPM_V47_DIR`, or a GGUF pair
/// from `CRANE_MINICPM_V47_GGUF` / `CRANE_MINICPM_V47_MMPROJ`.
fn load(device: &Device, dtype: DType) -> MinicpmV46VLModel {
    let draft = std::env::var("CRANE_DSPARK_DIR").expect("set CRANE_DSPARK_DIR");
    let mut model = match std::env::var("CRANE_MINICPM_V47_GGUF") {
        Ok(llm) => {
            let mmproj = std::env::var("CRANE_MINICPM_V47_MMPROJ")
                .expect("set CRANE_MINICPM_V47_MMPROJ with CRANE_MINICPM_V47_GGUF");
            MinicpmV46VLModel::from_gguf(&llm, &mmproj, device, &dtype).expect("load GGUF")
        },
        Err(_) => {
            let dir = std::env::var("CRANE_MINICPM_V47_DIR").expect("set CRANE_MINICPM_V47_DIR");
            MinicpmV46VLModel::new(&dir, device, &dtype).expect("load model")
        },
    };
    model.load_dspark(&draft).expect("load drafter");
    model
}

const PROMPTS: [&str; 3] = [
    "Explain how photosynthesis works in plants, in about 150 words.",
    "Solve step by step: if 3x + 7 = 31, what is x? Then compute x^2 + 2x.",
    "Write a Python function that checks whether a string is a palindrome, with a short docstring.",
];

#[test]
#[ignore = "needs CRANE_MINICPM_V47_DIR + CRANE_DSPARK_DIR"]
fn dspark_matches_plain_greedy() {
    let (device, dtype) = device();
    let mut model = load(&device, dtype);
    // First use pays for page-ins and kernel compilation; keep that out of the
    // timings below.
    for dspark in [false, true] {
        let warm = VlGenerationConfig {
            max_new_tokens: 16,
            dspark: Some(dspark),
            ..Default::default()
        };
        model.generate(None, PROMPTS[0], &warm, |_| {}).unwrap();
    }

    let max_new = 128;
    for prompt in PROMPTS {
        let plain_cfg = VlGenerationConfig {
            max_new_tokens: max_new,
            dspark: Some(false),
            ..Default::default()
        };
        let spec_cfg = VlGenerationConfig {
            max_new_tokens: max_new,
            dspark: Some(true),
            ..Default::default()
        };
        let t = std::time::Instant::now();
        let plain = model.generate(None, prompt, &plain_cfg, |_| {}).unwrap();
        let plain_s = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let spec = model.generate(None, prompt, &spec_cfg, |_| {}).unwrap();
        let spec_s = t.elapsed().as_secs_f64();
        let stats = model.last_spec_stats().cloned().expect("spec stats");

        let common = plain
            .chars()
            .zip(spec.chars())
            .take_while(|(a, b)| a == b)
            .count();
        println!(
            "\n[{}] plain {:.2}s | spec {:.2}s ({:.2}x) | rounds {} mean_accept {:.2} | draft {:.2}s verify {:.2}s other {:.2}s | common prefix {}/{} chars",
            &prompt[..24],
            plain_s,
            spec_s,
            plain_s / spec_s,
            stats.rounds,
            stats.mean_accept_len(),
            stats.draft_secs,
            stats.verify_secs,
            stats.other_secs,
            common,
            plain.chars().count().min(spec.chars().count()),
        );
        assert!(
            common >= 60,
            "speculative output diverges from plain greedy after {common} chars:\nplain: {plain}\nspec:  {spec}"
        );
        assert!(
            stats.mean_accept_len() > 1.5,
            "drafter barely helps: {}",
            stats.mean_accept_len()
        );
    }
}

/// The adaptive fallback switches between speculative rounds and plain decode
/// steps, snapshotting and replaying model state at each switch. In f32 the
/// speculative stream is token-for-token identical to plain greedy decoding,
/// so any bookkeeping slip (positions, snapshot, replay, drafter context)
/// shows up as a different token.
#[test]
#[ignore = "needs CRANE_MINICPM_V47_DIR + CRANE_DSPARK_DIR"]
fn mode_switching_is_lossless_in_f32() {
    let (device, _) = device();
    let mut model = load(&device, DType::F32);

    let max_new = 160;
    for prompt in PROMPTS {
        let plain = model
            .generate(
                None,
                prompt,
                &VlGenerationConfig {
                    max_new_tokens: max_new,
                    dspark: Some(false),
                    ..Default::default()
                },
                |_| {},
            )
            .unwrap();
        // (speculative rounds, plain tokens) cycles, from "switch constantly"
        // to long stretches.
        for schedule in [(1usize, 1usize), (1, 3), (2, 5), (3, 40)] {
            let spec = model
                .generate(
                    None,
                    prompt,
                    &VlGenerationConfig {
                        max_new_tokens: max_new,
                        dspark: Some(true),
                        dspark_schedule: Some(schedule),
                        ..Default::default()
                    },
                    |_| {},
                )
                .unwrap();
            let st = model.last_spec_stats().cloned().expect("spec stats");
            println!(
                "[{}] schedule {schedule:?}: rounds {} plain tokens {} switches {} -> {}",
                &prompt[..20],
                st.rounds,
                st.plain_tokens,
                st.switches,
                if spec == plain {
                    "identical"
                } else {
                    "DIFFERENT"
                }
            );
            assert!(
                st.switches > 0 && st.plain_tokens > 0,
                "schedule {schedule:?} never switched"
            );
            assert_eq!(spec, plain, "schedule {schedule:?} changed the output");
        }
    }
}
