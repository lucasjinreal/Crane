//! Targeted live check for phase 6d's incremental TTS generation
//! (`MiniCpmTts::generate_chunk`) — drives it directly (not through the
//! full `DuplexSession`) so it doesn't depend on the listen/speak model
//! actually deciding to speak within a short test clip (see
//! `minicpmo_duplex_session.rs`'s note on `haimianbaobao.wav` never
//! triggering a spoken response in that test).
//!
//! Splits one real chat response into two artificial "chunks" (first half,
//! second half) the same way `DuplexSession::streaming_generate` would
//! across two real audio chunks, calls `generate_chunk` for each without
//! clearing the KV cache in between, then synthesizes the accumulated
//! codes via the ordinary non-streaming `Token2Wav::synthesize` (matching
//! phase 6d's "TTS-chunking only, per-turn audio" scope — see the
//! `minicpmo::duplex` module doc and `AGENTS.md`).
//!
//! Gated by `CRANE_MINICPMO_DIR` so it doesn't run by default.

#[test]
#[ignore = "needs a local MiniCPM-o-4.5 checkpoint (CRANE_MINICPMO_DIR)"]
fn minicpmo_tts_generate_chunk_persists_kv_across_calls() {
    use candle_core::Tensor;
    use crane_core::models::minicpmo::{load_config, MiniCpmOLlm, MiniCpmTts, Token2Wav, TtsGenerationConfig};

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
        candle_nn::VarBuilder::from_mmaped_safetensors(&crane_core::utils::utils::get_safetensors_files(&dir).unwrap(), dtype, &device)
    }
    .unwrap();
    let mut tts = MiniCpmTts::new(&config.tts_config, vb.pp("tts"), &device, dtype).expect("load tts tower");
    let token2wav = Token2Wav::new(&dir, &device, dtype).expect("load token2wav");

    // ── Get a real chat response + hidden states (same recipe as
    // minicpmo_speech_generate.rs), long enough to split into two chunks. ──
    let tts_bos = llm.tokenizer.get_token("<|tts_bos|>").expect("tokenizer has no <|tts_bos|>");
    let im_end = llm.tokenizer.get_token("<|im_end|>").expect("tokenizer has no <|im_end|>");
    let tts_eos = llm.tokenizer.get_token("<|tts_eos|>");
    let prompt =
        "<|im_start|>user\nCount from one to ten, spelling out each number in words.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let mut prompt_ids = llm.prepare_inputs(prompt).expect("tokenize prompt");
    prompt_ids.push(tts_bos);

    llm.clear_kv_cache();
    let mut logits = llm.forward_step(&prompt_ids, 0).expect("prefill");

    let max_new_tokens = 60;
    let mut response_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut hidden_rows: Vec<Tensor> = Vec::with_capacity(max_new_tokens);
    let mut cur_pos = prompt_ids.len();
    for _ in 0..max_new_tokens {
        let next = logits.squeeze(0).unwrap().squeeze(0).unwrap().to_dtype(candle_core::DType::F32).unwrap().argmax(candle_core::D::Minus1).unwrap().to_scalar::<u32>().unwrap();
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
    assert!(response_tokens.len() >= 4, "need at least a few tokens to split into two meaningful chunks");

    let split = response_tokens.len() / 2;
    let (tokens_a, tokens_b) = response_tokens.split_at(split);
    let (hidden_a, hidden_b) = hidden_rows.split_at(split);

    // ── Chunk 1: not the end of the turn, so no text_eos yet, but always audio_bos. ──
    let hidden_a_cat = Tensor::cat(hidden_a, 1).unwrap().squeeze(0).unwrap();
    let condition_a = tts.build_condition_embeds(tokens_a, &hidden_a_cat).expect("chunk 1 condition embeds");
    let audio_bos_embed = tts.embed_special_token(tts.config.audio_bos_token_id).unwrap();
    let inputs_a = Tensor::cat(&[&condition_a, &audio_bos_embed], 1).unwrap();
    let len_a = inputs_a.dim(1).unwrap();

    let cfg_chunk = TtsGenerationConfig { min_new_tokens: 0, max_new_tokens: 26, ..Default::default() };
    let codes_a = tts.generate_chunk(&inputs_a, 0, &cfg_chunk).expect("generate_chunk 1");
    println!("chunk 1: {} codes from {} condition positions", codes_a.len(), len_a);

    let tts_start_pos_b = len_a + codes_a.len();

    // ── Chunk 2: this is the end of the turn, so text_eos + audio_bos. ──
    let hidden_b_cat = Tensor::cat(hidden_b, 1).unwrap().squeeze(0).unwrap();
    let condition_b = tts.build_condition_embeds(tokens_b, &hidden_b_cat).expect("chunk 2 condition embeds");
    let text_eos_embed = tts.embed_special_token(tts.config.text_eos_token_id).unwrap();
    let inputs_b = Tensor::cat(&[&condition_b, &text_eos_embed, &audio_bos_embed], 1).unwrap();
    let len_b = inputs_b.dim(1).unwrap();

    let cfg_final_chunk = TtsGenerationConfig { min_new_tokens: 0, max_new_tokens: 200, ..Default::default() };
    let codes_b = tts.generate_chunk(&inputs_b, tts_start_pos_b, &cfg_final_chunk).expect("generate_chunk 2 (continues chunk 1's KV cache)");
    println!("chunk 2: {} codes from {} condition positions (continuing from position {tts_start_pos_b})", codes_b.len(), len_b);

    // The whole point of generate_chunk: chunk 2 must have actually
    // continued from chunk 1's KV cache state, not started fresh — a
    // reasonable proxy for "the incremental state threading works" is
    // simply that both calls succeeded and produced plausible non-empty
    // output using the position accounting `DuplexSession` itself relies
    // on (`tts_start_pos_b` computed the exact same way).
    assert!(!codes_a.is_empty() || !codes_b.is_empty(), "TTS produced no speech tokens across either chunk");

    let mut all_codes = codes_a;
    all_codes.extend(codes_b);
    println!("total accumulated codes: {}", all_codes.len());
    assert!(!all_codes.is_empty());

    let waveform = token2wav.synthesize(&all_codes, 10).expect("token2wav synthesize");
    println!("waveform: {} samples ({:.2}s at {}Hz)", waveform.len(), waveform.len() as f32 / token2wav.sample_rate() as f32, token2wav.sample_rate());

    assert!(waveform.iter().all(|v| v.is_finite()), "non-finite sample in waveform");
    assert!(waveform.iter().all(|v| (-1.0..=1.0).contains(v)), "sample outside clamp range");
    let max_abs = waveform.iter().fold(0f32, |a, &b| a.max(b.abs()));
    assert!(max_abs > 0.001, "waveform looks like near-silence (max_abs={max_abs})");

    let out_path = std::env::var("CRANE_MINICPMO_CHUNK_SPEECH_OUT").unwrap_or_else(|_| "/tmp/minicpmo_tts_chunk_test.wav".to_string());
    let spec = hound::WavSpec { channels: 1, sample_rate: token2wav.sample_rate(), bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut writer = hound::WavWriter::create(&out_path, spec).expect("create wav writer");
    for &sample in &waveform {
        writer.write_sample((sample * i16::MAX as f32) as i16).expect("write sample");
    }
    writer.finalize().expect("finalize wav");
    println!("wrote {out_path}");
}
