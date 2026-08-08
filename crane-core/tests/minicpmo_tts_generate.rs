//! End-to-end sanity check for MiniCPM-o-4.5's TTS token generation: load a
//! real checkpoint, have the chat LLM answer a question, feed its response
//! + hidden states into the TTS tower, and verify a plausible sequence of
//! speech-token ids comes out. Covers phase 4 (TTS token generation) of the
//! module's staged build — see the `minicpmo` module doc for what's
//! implemented so far. No audio yet: turning these ids into a waveform is
//! phase 5 (the Token2wav vocoder), not yet implemented.
//!
//! Uses [`MiniCpmOLlm`] (text-only, phase 1) rather than the full
//! [`MiniCpmOVlModel`] (phases 1-3) — TTS conditioning only needs a chat
//! response and its hidden states, neither of which requires vision/audio
//! understanding, and skipping those towers keeps this test's ~17GB+
//! resident-weights footprint a bit further from this checkpoint's already
//! tight headroom on a shared GPU (see the `minicpmo` AGENTS.md note on
//! known resource constraints).
//!
//! Gated by `CRANE_MINICPMO_DIR` so it doesn't run by default. The TTS
//! tower's own math (weight-norm reconstruction, projector, embeddings, the
//! Llama backbone) was separately HF-diff validated during development (see
//! `minicpmo::tts_llm`'s `hf_diff` test module) — this test covers the
//! chat-LLM-hidden-states -> TTS-conditioning -> autoregressive-sampling
//! glue on top of that.

#[test]
#[ignore = "needs a local MiniCPM-o-4.5 checkpoint (CRANE_MINICPMO_DIR)"]
fn minicpmo_tts_generate_produces_plausible_codes() {
    use candle_core::Tensor;
    use crane_core::models::minicpmo::{load_config, MiniCpmOLlm, MiniCpmTts, TtsGenerationConfig};

    let dir = std::env::var("CRANE_MINICPMO_DIR").expect("set CRANE_MINICPMO_DIR to a MiniCPM-o-4.5 checkpoint dir");

    #[cfg(feature = "cuda")]
    let (device, dtype) = if candle_core::utils::cuda_is_available() {
        (candle_core::Device::new_cuda(0).unwrap(), candle_core::DType::BF16)
    } else {
        (candle_core::Device::Cpu, candle_core::DType::F32)
    };
    #[cfg(not(feature = "cuda"))]
    let (device, dtype) = (candle_core::Device::Cpu, candle_core::DType::F32);

    let mut llm = MiniCpmOLlm::new(&dir, &device, &dtype).expect("load MiniCPM-o-4.5 llm tower");
    let config = load_config(&format!("{dir}/config.json")).expect("load config");
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &crane_core::utils::utils::get_safetensors_files(&dir).unwrap(),
            dtype,
            &device,
        )
    }
    .unwrap();
    let mut tts = MiniCpmTts::new(&config.tts_config, vb.pp("tts"), &device, dtype).expect("load tts tower");

    // Same think-block + <|tts_bos|> prompt convention as
    // `MiniCpmOVlModel::render_prompt`/`render_prompt_for_tts`.
    let tts_bos = llm.tokenizer.get_token("<|tts_bos|>").expect("tokenizer has no <|tts_bos|>");
    let im_end = llm.tokenizer.get_token("<|im_end|>").expect("tokenizer has no <|im_end|>");
    let tts_eos = llm.tokenizer.get_token("<|tts_eos|>");
    let prompt = "<|im_start|>user\nSay a short, cheerful greeting.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let mut prompt_ids = llm.prepare_inputs(prompt).expect("tokenize prompt");
    prompt_ids.push(tts_bos);

    llm.clear_kv_cache();
    let mut logits = llm.forward_step(&prompt_ids, 0).expect("prefill");

    let max_new_tokens = 40;
    let mut response_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut hidden_rows: Vec<Tensor> = Vec::with_capacity(max_new_tokens);
    let mut cur_pos = prompt_ids.len();
    for _ in 0..max_new_tokens {
        let next = logits
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap()
            .argmax(candle_core::D::Minus1)
            .unwrap()
            .to_scalar::<u32>()
            .unwrap();
        if next == im_end || Some(next) == tts_eos {
            break;
        }
        response_tokens.push(next);
        logits = llm.forward_step(&[next], cur_pos).expect("decode step");
        hidden_rows.push(llm.last_hidden_states().expect("hidden states").clone());
        cur_pos += 1;
    }
    let response_text = llm.tokenizer.tokenizer.decode(&response_tokens, true).unwrap();
    println!("chat response: {response_text:?} ({} tokens)", response_tokens.len());
    assert!(!response_tokens.is_empty(), "chat produced no response tokens");

    let hidden_states = Tensor::cat(&hidden_rows, 1).unwrap().squeeze(0).unwrap(); // [response_len, hidden]
    let condition_embeds = tts.build_condition_embeds(&response_tokens, &hidden_states).expect("condition embeds");
    let text_eos_embed = tts.embed_special_token(tts.config.text_eos_token_id).unwrap();
    let audio_bos_embed = tts.embed_special_token(tts.config.audio_bos_token_id).unwrap();
    let inputs_embeds = Tensor::cat(&[&condition_embeds, &text_eos_embed, &audio_bos_embed], 1).unwrap();

    let tts_cfg = TtsGenerationConfig { max_new_tokens: 200, ..Default::default() };
    let codes = tts.generate(&inputs_embeds, &tts_cfg).expect("tts generate");

    println!("generated {} speech-token codes", codes.len());
    assert!(!codes.is_empty(), "TTS produced no speech tokens");
    assert!(codes.len() < tts_cfg.max_new_tokens, "TTS ran to max_new_tokens without stopping");
    let num_audio_tokens = config.tts_config.num_audio_tokens as u32;
    assert!(codes.iter().all(|&c| c < num_audio_tokens), "speech-token code out of vocab range");
}
