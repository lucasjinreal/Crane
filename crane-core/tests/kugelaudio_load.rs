//! Real-checkpoint load smoke test for KugelAudio: constructs every
//! sub-network from an actual `kugelaudio/kugelaudio-0-open` clone and
//! checks the weight-name/shape wiring in `models/kugelaudio/*.rs` against
//! the real `model.safetensors.index.json` (not just the small synthetic
//! `VarBuilder`s the per-module unit tests use). Also runs one real forward
//! pass (embed → decoder → lm_head) and one real acoustic-tokenizer
//! encode/decode roundtrip on a short random waveform, checking shapes and
//! finiteness — not output *correctness* (no HF reference comparison yet;
//! see `kugelaudio/model.rs`'s doc comment for what's still unvalidated).
//!
//! Gated by `CRANE_KUGELAUDIO_DIR` so it doesn't run by default (needs an
//! ~18.7GB local checkpoint).

#[test]
#[ignore = "needs a local KugelAudio checkpoint (CRANE_KUGELAUDIO_DIR)"]
fn kugelaudio_loads_and_runs_forward_and_tokenizer_roundtrip() {
    use candle_core::{DType, Device, Tensor};
    use crane_core::models::kugelaudio::KugelAudioModel;

    let dir = std::env::var("CRANE_KUGELAUDIO_DIR")
        .expect("set CRANE_KUGELAUDIO_DIR to a KugelAudio checkpoint dir");

    // CUDA → CUDA BF16 (the checkpoint's native `torch_dtype`); macOS →
    // Metal F16; everything else → CPU F32. Same pattern as
    // `voxcpm2_generate.rs` for another diffusion-head TTS model.
    #[cfg(feature = "cuda")]
    let (device, dtype) = if candle_core::utils::cuda_is_available() {
        (Device::new_cuda(0).unwrap(), DType::BF16)
    } else {
        (Device::Cpu, DType::F32)
    };
    #[cfg(all(target_os = "macos", not(feature = "cuda")))]
    let (device, dtype) = (Device::new_metal(0).unwrap_or(Device::Cpu), DType::F16);
    #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
    let (device, dtype) = (Device::Cpu, DType::F32);

    let mut model =
        KugelAudioModel::from_pretrained(&dir, &device, dtype).expect("KugelAudioModel::from_pretrained");

    let hidden_size = model.config.decoder_config.hidden_size;
    let vocab_size = model.config.decoder_config.vocab_size;
    assert_eq!(hidden_size, 3584);
    assert_eq!(vocab_size, 152_064);

    // A tiny fake "prompt": a handful of token ids run through the real
    // embedding table and decoder stack.
    let input_ids = Tensor::from_vec(vec![100u32, 200, 300, 400], (1, 4), &device).unwrap();
    let embeds = model.embed_text_tokens(&input_ids).expect("embed_text_tokens");
    assert_eq!(embeds.dims(), &[1, 4, hidden_size]);

    let (hidden, logits) = model.forward(&embeds, 0).expect("forward");
    assert_eq!(hidden.dims(), &[1, 4, hidden_size]);
    assert_eq!(logits.dims(), &[1, 4, vocab_size]);
    let max_abs: f32 = logits.abs().unwrap().max_all().unwrap().to_dtype(DType::F32).unwrap().to_scalar().unwrap();
    assert!(max_abs.is_finite(), "logits must be finite, got max_abs={max_abs}");

    // Decode-step shape check: single-token continuation with seqlen_offset.
    model.clear_kv_cache();
    let _ = model.forward(&embeds, 0).expect("prefill for decode check");
    let next_id = Tensor::from_vec(vec![500u32], (1, 1), &device).unwrap();
    let next_embed = model.embed_text_tokens(&next_id).expect("embed next token");
    let (_, decode_logits) = model.forward(&next_embed, 4).expect("decode step forward");
    assert_eq!(decode_logits.dims(), &[1, 1, vocab_size]);

    // Acoustic tokenizer encode/decode roundtrip on a short synthetic waveform.
    let waveform = Tensor::rand(-0.1f32, 0.1f32, (1, 1, 24_000), &device).unwrap();
    let latents = model.encode_acoustic(&waveform).expect("encode_acoustic");
    assert_eq!(latents.dim(0).unwrap(), 1);
    assert_eq!(latents.dim(1).unwrap(), model.config.acoustic_vae_dim);
    let latent_max: f32 = latents.abs().unwrap().max_all().unwrap().to_dtype(DType::F32).unwrap().to_scalar().unwrap();
    assert!(latent_max.is_finite());

    let recon = model.decode_acoustic(&latents).expect("decode_acoustic");
    assert_eq!(recon.dim(0).unwrap(), 1);
    assert_eq!(recon.dim(1).unwrap(), 1);
    let recon_max: f32 = recon.abs().unwrap().max_all().unwrap().to_dtype(DType::F32).unwrap().to_scalar().unwrap();
    assert!(recon_max.is_finite());

    // Semantic tokenizer encode.
    let semantic = model.encode_semantic(&waveform).expect("encode_semantic");
    assert_eq!(semantic.dim(1).unwrap(), model.config.semantic_vae_dim);

    // One diffusion-sampling call (no CFG — see `sample_speech_latents`'s
    // doc comment) using a real decoder hidden state as condition.
    let condition = hidden.narrow(1, 3, 1).unwrap().squeeze(1).unwrap(); // [1, hidden_size]
    let speech_latent = model.sample_speech_latents(&condition).expect("sample_speech_latents");
    assert_eq!(speech_latent.dims(), &[1, model.config.diffusion_head_config.latent_size]);
    let sl_max: f32 = speech_latent.abs().unwrap().max_all().unwrap().to_dtype(DType::F32).unwrap().to_scalar().unwrap();
    assert!(sl_max.is_finite());
}
