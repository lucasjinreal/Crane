//! Ad-hoc latency profiling for MiniCPM-o's turn-based pipeline — informs
//! phase 6 (full-duplex streaming)'s design, per the plan doc ("plan its
//! concrete design in detail once phases 1-5 are working and the real
//! per-chunk latency budget is known from profiling the turn-based path").
//! Not a correctness test (no assertions beyond "it ran") — prints timings
//! for manual inspection.
//!
//! Split into two separate tests (rather than one combined run) because
//! loading the LLM + TTS + Token2wav towers *simultaneously* is tight
//! enough on this checkpoint's VRAM budget (see the `minicpmo` AGENTS.md
//! resource-constraint note) that adding profiling-only overhead (held
//! `Vec<Tensor>` hidden-state rows, extra KV cache growth from longer
//! generations) pushed a combined run over the edge during development —
//! loading each stage's tower(s) in isolation avoids that entirely and is
//! just as informative for per-stage timing.
//!
//! Gated by `CRANE_MINICPMO_DIR`.

#[test]
#[ignore = "needs a local MiniCPM-o-4.5 checkpoint (CRANE_MINICPMO_DIR); profiling only, no assertions"]
fn minicpmo_profile_llm_and_tts_latency() {
    use candle_core::Tensor;
    use crane_core::models::minicpmo::{load_config, MiniCpmOLlm, MiniCpmTts, TtsGenerationConfig};
    use std::time::Instant;

    let dir = std::env::var("CRANE_MINICPMO_DIR").expect("set CRANE_MINICPMO_DIR to a MiniCPM-o-4.5 checkpoint dir");

    #[cfg(feature = "cuda")]
    let (device, dtype) = if candle_core::utils::cuda_is_available() {
        (candle_core::Device::new_cuda(0).unwrap(), candle_core::DType::BF16)
    } else {
        (candle_core::Device::Cpu, candle_core::DType::F32)
    };
    #[cfg(not(feature = "cuda"))]
    let (device, dtype) = (candle_core::Device::Cpu, candle_core::DType::F32);

    let t0 = Instant::now();
    let mut llm = MiniCpmOLlm::new(&dir, &device, &dtype).expect("load llm");
    let config = load_config(&format!("{dir}/config.json")).expect("load config");
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&crane_core::utils::utils::get_safetensors_files(&dir).unwrap(), dtype, &device)
    }
    .unwrap();
    let mut tts = MiniCpmTts::new(&config.tts_config, vb.pp("tts"), &device, dtype).expect("load tts");
    println!("[load] llm + tts: {:?}", t0.elapsed());

    // ── LLM: prefill + per-token decode ──
    let prompt = "<|im_start|>user\nTell me a short story about a robot learning to paint, in about five sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let tts_bos = llm.tokenizer.get_token("<|tts_bos|>").unwrap();
    let im_end = llm.tokenizer.get_token("<|im_end|>").unwrap();
    let tts_eos = llm.tokenizer.get_token("<|tts_eos|>");
    let mut prompt_ids = llm.prepare_inputs(prompt).unwrap();
    prompt_ids.push(tts_bos);

    llm.clear_kv_cache();
    let t0 = Instant::now();
    let mut logits = llm.forward_step(&prompt_ids, 0).unwrap();
    println!("[llm] prefill ({} tokens): {:?}", prompt_ids.len(), t0.elapsed());

    let max_new_tokens = 80;
    let mut response_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens);
    let mut hidden_rows: Vec<Tensor> = Vec::with_capacity(max_new_tokens);
    let mut cur_pos = prompt_ids.len();
    let mut llm_decode_times = Vec::with_capacity(max_new_tokens);
    for _ in 0..max_new_tokens {
        let next = logits.squeeze(0).unwrap().squeeze(0).unwrap().to_dtype(candle_core::DType::F32).unwrap().argmax(candle_core::D::Minus1).unwrap().to_scalar::<u32>().unwrap();
        if next == im_end || Some(next) == tts_eos {
            break;
        }
        response_tokens.push(next);
        let t0 = Instant::now();
        logits = llm.forward_step(&[next], cur_pos).unwrap();
        llm_decode_times.push(t0.elapsed());
        hidden_rows.push(llm.last_hidden_states().unwrap().clone());
        cur_pos += 1;
    }
    let response_text = llm.tokenizer.tokenizer.decode(&response_tokens, true).unwrap();
    let avg_llm_decode: f64 = llm_decode_times.iter().map(std::time::Duration::as_secs_f64).sum::<f64>() / llm_decode_times.len() as f64;
    println!("[llm] response: {response_text:?} ({} tokens)", response_tokens.len());
    println!("[llm] avg decode step: {:.2}ms ({:.1} tok/s)", avg_llm_decode * 1000.0, 1.0 / avg_llm_decode);

    // ── TTS: condition + prefill + per-token decode ──
    let hidden_states = Tensor::cat(&hidden_rows, 1).unwrap().squeeze(0).unwrap();
    let t0 = Instant::now();
    let condition_embeds = tts.build_condition_embeds(&response_tokens, &hidden_states).unwrap();
    let text_eos_embed = tts.embed_special_token(tts.config.text_eos_token_id).unwrap();
    let audio_bos_embed = tts.embed_special_token(tts.config.audio_bos_token_id).unwrap();
    let inputs_embeds = Tensor::cat(&[&condition_embeds, &text_eos_embed, &audio_bos_embed], 1).unwrap();
    println!("[tts] condition build: {:?}", t0.elapsed());

    let t0 = Instant::now();
    let tts_cfg = TtsGenerationConfig { max_new_tokens: 150, ..Default::default() };
    let codes = tts.generate(&inputs_embeds, &tts_cfg).unwrap();
    let tts_elapsed = t0.elapsed();
    println!(
        "[tts] generate: {} codes in {:?} ({:.2}ms/code, {:.1} tok/s)",
        codes.len(),
        tts_elapsed,
        tts_elapsed.as_secs_f64() * 1000.0 / codes.len() as f64,
        codes.len() as f64 / tts_elapsed.as_secs_f64()
    );
    println!("[tts] audio duration represented: {:.2}s (25Hz speech tokens)", codes.len() as f64 / 25.0);
}

#[test]
#[ignore = "needs a local MiniCPM-o-4.5 checkpoint (CRANE_MINICPMO_DIR); profiling only, no assertions"]
fn minicpmo_profile_token2wav_latency() {
    use crane_core::models::minicpmo::Token2Wav;
    use std::time::Instant;

    let dir = std::env::var("CRANE_MINICPMO_DIR").expect("set CRANE_MINICPMO_DIR to a MiniCPM-o-4.5 checkpoint dir");

    #[cfg(feature = "cuda")]
    let (device, dtype) = if candle_core::utils::cuda_is_available() {
        (candle_core::Device::new_cuda(0).unwrap(), candle_core::DType::BF16)
    } else {
        (candle_core::Device::Cpu, candle_core::DType::F32)
    };
    #[cfg(not(feature = "cuda"))]
    let (device, dtype) = (candle_core::Device::Cpu, candle_core::DType::F32);

    let t0 = Instant::now();
    let token2wav = Token2Wav::new(&dir, &device, dtype).expect("load token2wav");
    println!("[load] token2wav: {:?}", t0.elapsed());

    // Arbitrary valid-range speech-token ids (content doesn't matter for
    // timing) at 25 codes/s (s3tokenizer rate) -> chunk sizes covering
    // sub-second to multi-second "utterances so far".
    let all_codes: Vec<u32> = (0..150u32).map(|i| (i * 37) % 6560).collect();

    for &chunk_codes in &[13usize, 25, 50, 100, 150] {
        let subset = &all_codes[..chunk_codes];
        let t0 = Instant::now();
        let waveform = token2wav.synthesize(subset, 10).expect("synthesize");
        let elapsed = t0.elapsed();
        let audio_s = waveform.len() as f64 / f64::from(token2wav.sample_rate());
        println!(
            "[token2wav] {chunk_codes} codes ({:.2}s audio) -> synth in {:?} (realtime factor: {:.2}x, i.e. {:.2}s of compute per 1s of audio)",
            audio_s,
            elapsed,
            elapsed.as_secs_f64() / audio_s,
            elapsed.as_secs_f64() / audio_s,
        );
    }
}
