//! DSpark drafter vs the cache-free PyTorch reference
//! (`ai/scripts47/dspark_ref.py ... dump`): identical context features and
//! anchor must give the same block hidden states, draft tokens and confidences.
//!
//! Gated by `CRANE_DSPARK_DIR` (HF DSpark dir), `CRANE_MINICPM_V47_DIR` (target
//! checkpoint, for its embedding table and `lm_head`) and `CRANE_DSPARK_DUMP`
//! (prefix of the converted dump: `<prefix>.safetensors` + `<prefix>.json`).

use candle_core::{DType, Device, Tensor};
use crane_core::candle_core;
use crane_core::candle_nn::VarBuilder;
use crane_core::models::dspark::DSparkDrafter;

fn cosine(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap().to_dtype(DType::F32).unwrap();
    let b = b.flatten_all().unwrap().to_dtype(DType::F32).unwrap();
    let dot = (&a * &b)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let n = |t: &Tensor| {
        t.sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt()
    };
    dot / (n(&a) * n(&b))
}

#[test]
#[ignore = "needs CRANE_DSPARK_DIR, CRANE_MINICPM_V47_DIR, CRANE_DSPARK_DUMP"]
fn drafter_matches_pytorch_reference() {
    let draft_dir = std::env::var("CRANE_DSPARK_DIR").expect("set CRANE_DSPARK_DIR");
    check_against_reference(8, 0.995, |device, dtype| {
        DSparkDrafter::from_dir(&draft_dir, device, dtype).unwrap()
    });
}

/// The GGUF drafter dequantizes to bf16 at load; quantization noise moves
/// logits slightly, so the bar is lower than for the bf16 safetensors.
#[test]
#[ignore = "needs CRANE_DSPARK_GGUF, CRANE_MINICPM_V47_DIR, CRANE_DSPARK_DUMP"]
fn gguf_drafter_matches_pytorch_reference() {
    let path = std::env::var("CRANE_DSPARK_GGUF").expect("set CRANE_DSPARK_GGUF");
    check_against_reference(5, 0.97, |device, dtype| {
        DSparkDrafter::from_gguf(&path, device, dtype).unwrap()
    });
}

fn check_against_reference(
    min_leading: usize,
    min_cosine: f32,
    load: impl FnOnce(&Device, DType) -> DSparkDrafter,
) {
    let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("set {k}"));
    let (target_dir, dump) = (env("CRANE_MINICPM_V47_DIR"), env("CRANE_DSPARK_DUMP"));
    #[cfg(feature = "cuda")]
    let (device, dtype) = (Device::new_cuda(0).unwrap(), DType::BF16);
    #[cfg(not(feature = "cuda"))]
    let (device, dtype) = (Device::Cpu, DType::F32);

    let tensors = candle_core::safetensors::load(format!("{dump}.safetensors"), &device).unwrap();
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{dump}.json")).unwrap()).unwrap();
    let anchor = meta["anchor"].as_u64().unwrap() as u32;
    let want_toks: Vec<u32> = serde_json::from_value(meta["toks"].clone()).unwrap();
    let want_conf: Vec<f32> = serde_json::from_value(meta["conf"].clone()).unwrap();

    let mut drafter = load(&device, dtype);
    let (blk, mask) = (
        drafter.config().block_size,
        drafter.config().mask_token_id(),
    );
    drafter.append_context(&tensors["ctx_taps"]).unwrap();
    assert_eq!(drafter.ctx_len(), meta["q0"].as_u64().unwrap() as usize);

    // The target's embedding table and (untied) lm_head.
    let files = vec![std::path::Path::new(&target_dir).join("model.safetensors")];
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device) }.unwrap();
    let vocab = drafter.config().vocab_size;
    let hidden = drafter.config().hidden_size;
    let embed = vb
        .get((vocab, hidden), "model.language_model.embed_tokens.weight")
        .unwrap();
    let lm_head = vb.get((vocab, hidden), "lm_head.weight").unwrap();

    let mut ids = vec![anchor];
    ids.extend(std::iter::repeat_n(mask, blk - 1));
    let ids = Tensor::from_vec(ids, blk, &device).unwrap();
    let block_embeds = embed.index_select(&ids, 0).unwrap();

    let hid = drafter.block_hidden(&block_embeds).unwrap();
    let want_hid = tensors["hid"].to_dtype(DType::F32).unwrap();
    for k in 0..blk {
        let c = cosine(
            &hid.narrow(0, k, 1).unwrap(),
            &want_hid.narrow(0, k, 1).unwrap(),
        );
        assert!(c > min_cosine, "block hidden row {k}: cosine {c}");
    }

    let u = hid
        .matmul(&lm_head.t().unwrap())
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap();
    let c_u = cosine(&u, &tensors["U"]);
    assert!(c_u > min_cosine, "base logits cosine {c_u}");

    let draft = drafter.markov_draft(&u, &hid, anchor).unwrap();
    let matching = draft
        .tokens
        .iter()
        .zip(&want_toks)
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "tokens: {:?}\nwant:   {:?}\nleading match: {matching}/{blk}",
        draft.tokens, want_toks
    );
    println!("conf:   {:?}\nwant:   {:?}", draft.confidence, want_conf);
    assert!(
        matching >= min_leading,
        "only {matching} leading draft tokens match PyTorch"
    );
    for (k, (g, w)) in draft.confidence.iter().zip(&want_conf).enumerate().take(8) {
        assert!(
            (g - w).abs() < if min_cosine > 0.99 { 0.08 } else { 0.2 },
            "confidence[{k}]: {g} vs {w}"
        );
    }

    // Where does a drafting round's time go? (warm clocks, device-synchronised)
    let time = |label: &str, f: &mut dyn FnMut()| {
        let warm = std::time::Instant::now();
        while warm.elapsed().as_secs_f64() < 0.5 {
            f();
            device.synchronize().unwrap();
        }
        let t = std::time::Instant::now();
        for _ in 0..30 {
            f();
        }
        device.synchronize().unwrap();
        println!(
            "{label:<28} {:.2} ms",
            t.elapsed().as_secs_f64() * 1000.0 / 30.0
        );
    };
    time("block_hidden (5 layers)", &mut || {
        drafter.block_hidden(&block_embeds).unwrap();
    });
    time("lm_head over block", &mut || {
        hid.matmul(&lm_head.t().unwrap())
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
    });
    time("markov_draft (total)", &mut || {
        drafter.markov_draft(&u, &hid, anchor).unwrap();
    });
    time("  candidates (topk+gather)", &mut || {
        drafter.markov_candidates(&u).unwrap();
    });
    let cands = drafter.markov_candidates(&u).unwrap();
    time("  chain (15 sequential)", &mut || {
        drafter.markov_chain(&cands, anchor).unwrap();
    });
}
