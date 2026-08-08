//! End-to-end sanity check for MiniCPM-o-4.5's full speech-output pipeline:
//! chat response -> TTS speech-token ids -> Token2wav vocoder -> real audio.
//! Covers phase 5 (Token2wav) of the module's staged build — the first
//! phase where MiniCPM-o can actually talk. See the `minicpmo` module doc
//! for what's implemented so far, and `token2wav`'s module doc for the
//! "system default voice, not real voice cloning" scope boundary.
//!
//! Uses [`MiniCpmOLlm`] (text-only) rather than the full [`MiniCpmOVlModel`]
//! for the chat step, same reasoning as `minicpmo_tts_generate.rs` (TTS
//! conditioning doesn't need vision/audio understanding, and this
//! checkpoint's memory headroom is tight on a shared GPU).
//!
//! Gated by `CRANE_MINICPMO_DIR` so it doesn't run by default. Every
//! Token2wav sub-stage was separately HF-diff/sanity validated during
//! development (`token2wav::flow`'s `hf_diff` test, cosine > 0.9999999;
//! `token2wav::hifigan`'s F0-predictor HF-diff and waveform shape/range
//! tests) — this test covers the full glue end to end and writes a real
//! `.wav` file so the output can be listened to.

#[test]
#[ignore = "needs a local MiniCPM-o-4.5 checkpoint (CRANE_MINICPMO_DIR)"]
fn minicpmo_speech_generate_produces_real_audio() {
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
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &crane_core::utils::utils::get_safetensors_files(&dir).unwrap(),
            dtype,
            &device,
        )
    }
    .unwrap();
    let mut tts = MiniCpmTts::new(&config.tts_config, vb.pp("tts"), &device, dtype).expect("load tts tower");

    // Token2wav's DiT/CFM + HiFi-GAN math was validated in F32 for the
    // tightest possible numeric comparison (see the sub-module HF-diff
    // tests), but loading a *second* full copy of the pipeline in F32 on
    // top of the chat/TTS models' bf16 weights doesn't fit this
    // checkpoint's already-tight headroom on a shared GPU (see the
    // `minicpmo` AGENTS.md note on known resource constraints) — match the
    // rest of the pipeline's dtype here instead.
    let token2wav = Token2Wav::new(&dir, &device, dtype).expect("load token2wav");

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

    let hidden_states = Tensor::cat(&hidden_rows, 1).unwrap().squeeze(0).unwrap();
    let condition_embeds = tts.build_condition_embeds(&response_tokens, &hidden_states).expect("condition embeds");
    let text_eos_embed = tts.embed_special_token(tts.config.text_eos_token_id).unwrap();
    let audio_bos_embed = tts.embed_special_token(tts.config.audio_bos_token_id).unwrap();
    let inputs_embeds = Tensor::cat(&[&condition_embeds, &text_eos_embed, &audio_bos_embed], 1).unwrap();

    let tts_cfg = TtsGenerationConfig { max_new_tokens: 200, ..Default::default() };
    let codes = tts.generate(&inputs_embeds, &tts_cfg).expect("tts generate");
    println!("generated {} speech-token codes", codes.len());
    assert!(!codes.is_empty(), "TTS produced no speech tokens");

    let waveform = token2wav.synthesize(&codes, 10).expect("token2wav synthesize");
    println!("waveform: {} samples ({:.2}s at {}Hz)", waveform.len(), waveform.len() as f32 / token2wav.sample_rate() as f32, token2wav.sample_rate());

    assert!(waveform.iter().all(|v| v.is_finite()), "non-finite sample in waveform");
    assert!(waveform.iter().all(|v| (-1.0..=1.0).contains(v)), "sample outside clamp range");
    let max_abs = waveform.iter().fold(0f32, |a, &b| a.max(b.abs()));
    assert!(max_abs > 0.001, "waveform looks like near-silence (max_abs={max_abs})");

    let out_path = std::env::var("CRANE_MINICPMO_SPEECH_OUT").unwrap_or_else(|_| "/tmp/minicpmo_speech_test.wav".to_string());
    let spec = hound::WavSpec { channels: 1, sample_rate: token2wav.sample_rate(), bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut writer = hound::WavWriter::create(&out_path, spec).expect("create wav writer");
    for &sample in &waveform {
        writer.write_sample((sample * i16::MAX as f32) as i16).expect("write sample");
    }
    writer.finalize().expect("finalize wav");
    println!("wrote {out_path}");
}
