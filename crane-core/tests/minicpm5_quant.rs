//! End-to-end generation checks for MiniCPM5's GGUF loading path.
//!
//! Needs local checkpoints, so it's `#[ignore]`d by default and resolves its
//! inputs from env vars:
//!
//! ```bash
//! CRANE_MINICPM5_DIR=/path/to/MiniCPM5-1B \
//! CRANE_MINICPM5_GGUF=/path/to/MiniCPM5-1B-Q8_0.gguf \
//!   cargo test -p crane-core --release --test minicpm5_quant -- --ignored --nocapture
//! ```
//!
//! `CRANE_MINICPM5_GGUF` can point at a bare `.gguf` file with no sibling
//! `tokenizer.json`/`config.json` at all — `openbmb/MiniCPM5-1B-GGUF`'s files
//! embed the full tokenizer (`tokenizer.ggml.tokens`/`merges`/`chat_template`),
//! and `Model::from_gguf` unions `<|im_end|>` (the chat template's real
//! turn-end token) into the stop-id set regardless of whether a sibling
//! config supplies the full multi-EOS list — a real bug found and fixed
//! during this test's own development: GGUF metadata's
//! `tokenizer.ggml.eos_token_id` only ever carries the single id `</s>`,
//! which the model doesn't actually emit at the end of a chat turn, so
//! without this fix generation on a bare GGUF (no sibling config) never hit
//! a stop token and ran to `max_new_tokens` instead of terminating cleanly.

use candle_core::{DType, Device};
use crane_core::chat::{Message, Role};
use crane_core::generation::based::ModelForCausalLM;
use crane_core::generation::GenerationConfig;
use crane_core::models::minicpm5::Model;

const MAX_NEW_TOKENS: usize = 150;

fn device_and_dtype() -> (Device, DType) {
    #[cfg(feature = "cuda")]
    if candle_core::utils::cuda_is_available() {
        return (Device::new_cuda(0).unwrap(), DType::BF16);
    }
    if candle_core::utils::metal_is_available() {
        return (Device::new_metal(0).unwrap(), DType::F16);
    }
    (Device::Cpu, DType::F32)
}

/// Greedy chat-template generation. Returns the decoded response text and
/// whether generation stopped before hitting `MAX_NEW_TOKENS` (i.e. it found
/// a real stop token rather than running out of budget).
fn run_chat(model_path: &str, tokenizer: &crane_core::autotokenizer::AutoTokenizer) -> (String, bool) {
    let (device, dtype) = device_and_dtype();
    let mut model = Model::new(model_path, &device, &dtype).expect("load model");

    let messages = vec![Message {
        role: Role::User,
        content: "What is the capital of France? Answer in one short sentence.".to_string(),
    }];
    let prompt = tokenizer
        .apply_chat_template_with_options(&messages, Option::<&serde_json::Value>::None, true, None)
        .expect("apply chat template");

    let gen_cfg = GenerationConfig { max_new_tokens: MAX_NEW_TOKENS, temperature: None, ..Default::default() };
    let input_ids = model.prepare_inputs(&prompt).expect("tokenize prompt");
    let output_ids = model.generate(&input_ids, &gen_cfg, None).expect("generate");
    let generated = &output_ids[input_ids.len()..];
    let text = tokenizer.decode(generated, true).expect("decode");
    let stopped_early = generated.len() < MAX_NEW_TOKENS;
    (text, stopped_early)
}

#[test]
#[ignore = "needs local MiniCPM5 checkpoints (CRANE_MINICPM5_DIR, CRANE_MINICPM5_GGUF)"]
fn gguf_and_safetensors_both_answer_correctly_and_stop_cleanly() {
    let dir = std::env::var("CRANE_MINICPM5_DIR").expect("set CRANE_MINICPM5_DIR to a MiniCPM5-1B checkpoint dir");
    let gguf = std::env::var("CRANE_MINICPM5_GGUF").expect("set CRANE_MINICPM5_GGUF to a .gguf file");

    // Chat-template tokenizer comes from the safetensors checkpoint; only
    // used to render the prompt / decode output text identically for both
    // paths, not to exercise the GGUF-embedded-tokenizer path itself (that's
    // exercised inside `Model::new` -> `Model::from_gguf` below).
    let tokenizer = crane_core::autotokenizer::AutoTokenizer::from_pretrained(&dir, None).expect("load tokenizer");

    let (st_text, st_stopped) = run_chat(&dir, &tokenizer);
    println!("safetensors: stopped_early={st_stopped} text={st_text:?}");
    assert!(st_text.to_lowercase().contains("paris"), "safetensors didn't answer Paris: {st_text}");
    assert!(st_stopped, "safetensors generation ran to max_new_tokens instead of stopping at EOS");

    let (gguf_text, gguf_stopped) = run_chat(&gguf, &tokenizer);
    println!("gguf: stopped_early={gguf_stopped} text={gguf_text:?}");
    assert!(gguf_text.to_lowercase().contains("paris"), "gguf didn't answer Paris: {gguf_text}");
    assert!(gguf_stopped, "gguf generation ran to max_new_tokens instead of stopping at EOS (the <|im_end|> union bug)");
}
