//! Speculative-decoding plumbing on a real Qwen 3.5-family GGUF: a
//! `forward_spec` block with a mid-block GDN checkpoint, followed by
//! `commit_spec`, must leave the model exactly where a plain token-by-token
//! decode would be.
//!
//! Gated by `CRANE_QWEN35_GGUF` (any `qwen35` GGUF, e.g. the MiniCPM-V-4.7
//! LLM quant), `#[ignore]`d otherwise.

use candle_core::{DType, Device, Tensor};
use crane_core::candle_core;
use crane_core::models::qwen3_5::Qwen3_5TextModel;

fn device() -> Device {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return d;
    }
    Device::Cpu
}

fn cosine(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap().to_dtype(DType::F32).unwrap();
    let b = b.flatten_all().unwrap().to_dtype(DType::F32).unwrap();
    let dot = (&a * &b)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let na = a
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    let nb = b
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
        .sqrt();
    dot / (na * nb)
}

fn argmax(t: &Tensor) -> u32 {
    t.flatten_all()
        .unwrap()
        .argmax(0)
        .unwrap()
        .to_scalar::<u32>()
        .unwrap()
}

fn positions(device: &Device, start: usize, n: usize) -> Tensor {
    let flat: Vec<u32> = (0..n)
        .flat_map(|i| std::iter::repeat_n((start + i) as u32, 3))
        .collect();
    // [3, n]: three identical rows, laid out row-major.
    let mut rows = Vec::with_capacity(3 * n);
    for _ in 0..3 {
        rows.extend((0..n).map(|i| (start + i) as u32));
    }
    drop(flat);
    Tensor::from_vec(rows, (3, n), device).unwrap()
}

fn embed(m: &Qwen3_5TextModel, device: &Device, ids: &[u32]) -> Tensor {
    let t = Tensor::from_vec(ids.to_vec(), (1, ids.len()), device).unwrap();
    m.embed_only(&t).unwrap()
}

/// Logits after consuming `ids` one token at a time from a fresh state.
fn sequential_logits(
    m: &mut Qwen3_5TextModel,
    device: &Device,
    prefill: &[u32],
    rest: &[u32],
) -> Vec<Tensor> {
    m.reset_gdn_caches().unwrap();
    let h = embed(m, device, prefill);
    let mut logits = vec![
        m.forward_embeds(&h, &positions(device, 0, prefill.len()), 0, None)
            .unwrap(),
    ];
    let mut pos = prefill.len();
    for &tok in rest {
        let h = embed(m, device, &[tok]);
        logits.push(
            m.forward_embeds(&h, &positions(device, pos, 1), pos, None)
                .unwrap(),
        );
        pos += 1;
    }
    logits
}

#[test]
#[ignore = "needs CRANE_QWEN35_GGUF"]
fn spec_block_then_commit_matches_sequential_decode() {
    let path = std::env::var("CRANE_QWEN35_GGUF").expect("set CRANE_QWEN35_GGUF");
    let device = device();
    let mmap = crane_core::quantized::gguf_file::mmap_gguf_file(&path).unwrap();
    let (ct, _) = crane_core::quantized::extended_gguf::read_content(mmap.as_ref()).unwrap();
    let mut cursor = std::io::Cursor::new(mmap.as_ref());
    let mut m = Qwen3_5TextModel::from_gguf(ct, &mut cursor, &device).unwrap();

    // Arbitrary but fixed token ids: only state equivalence matters here.
    let ids: Vec<u32> = (0..40u32).map(|i| 1000 + (i * 37) % 5000).collect();
    let (prefill_len, block_len, ckpt) = (24usize, 12usize, 5usize);
    let (prefill, rest) = (&ids[..prefill_len], &ids[prefill_len..]);

    // Reference: logits after each of prefill, then every following token.
    let reference = sequential_logits(&mut m, &device, prefill, rest);

    // Speculative: prefill, one block with a checkpoint at `ckpt`, commit.
    m.reset_gdn_caches().unwrap();
    let h = embed(&m, &device, prefill);
    m.forward_embeds(&h, &positions(&device, 0, prefill_len), 0, None)
        .unwrap();

    let block = &ids[prefill_len..prefill_len + block_len];
    let hb = embed(&m, &device, block);
    let n_logits = block_len;

    // Baseline for the split: the very same block with no mid-block
    // checkpoint (same dtype, same kernels), then back to the prefill state.
    let (base_logits, _) = m
        .forward_spec(
            &hb,
            &positions(&device, prefill_len, block_len),
            prefill_len,
            0,
            &[3],
            n_logits,
            true,
        )
        .unwrap();
    m.commit_spec(prefill_len);
    let (logits, taps) = m
        .forward_spec(
            &hb,
            &positions(&device, prefill_len, block_len),
            prefill_len,
            ckpt,
            &[3, 7, 11],
            n_logits,
            true,
        )
        .unwrap();
    assert_eq!(logits.dims(), &[n_logits, logits.dim(1).unwrap()]);
    assert_eq!(taps.dims()[0], block_len);
    assert_eq!(taps.dims()[1], 3 * h.dim(2).unwrap());

    // Block logits row j = logits after consuming block[..=j] = reference[j + 1].
    // Splitting the recurrence at the checkpoint must not change the block's
    // own outputs: same dtype and kernels, so (near-)identical.
    for j in 0..block_len {
        let a = base_logits.narrow(0, j, 1).unwrap();
        let b = logits.narrow(0, j, 1).unwrap();
        let c = cosine(&a, &b);
        assert!(c > 0.9999, "split vs unsplit block, row {j}: cosine {c}");
        assert_eq!(argmax(&a), argmax(&b), "split vs unsplit argmax, row {j}");
    }

    // The block runs in bf16 activations, the reference in the f32 decode
    // path, so rows differ by rounding noise; what verification needs is the
    // argmax.
    let mut min_cos = 1.0f32;
    let mut argmax_misses = 0;
    for j in 0..block_len {
        let want = &reference[j + 1];
        let got = logits.narrow(0, j, 1).unwrap();
        let c = cosine(&got, want);
        min_cos = min_cos.min(c);
        if argmax(&got) != argmax(want) {
            argmax_misses += 1;
        }
        println!(
            "block row {j}: cosine {c:.5} argmax {}",
            if argmax(&got) == argmax(want) {
                "same"
            } else {
                "DIFF"
            }
        );
    }
    assert!(min_cos > 0.99, "block logits drifted: min cosine {min_cos}");
    assert!(
        argmax_misses <= 1,
        "{argmax_misses} of {block_len} block argmaxes differ"
    );

    // Commit at the checkpoint, then replay block[ckpt..] token by token: the
    // restored recurrent state must reproduce the sequential logits.
    m.commit_spec(prefill_len + ckpt);
    let mut pos = prefill_len + ckpt;
    for (k, &tok) in ids[pos..].iter().enumerate() {
        let hk = embed(&m, &device, &[tok]);
        let got = m
            .forward_embeds(&hk, &positions(&device, pos, 1), pos, None)
            .unwrap();
        let want = &reference[ckpt + k + 1];
        let c = cosine(&got, want);
        assert!(c > 0.999, "post-commit step {k} (pos {pos}): cosine {c}");
        assert_eq!(
            argmax(&got),
            argmax(want),
            "argmax differs at post-commit step {k}"
        );
        pos += 1;
    }
}

/// Cost of a verification block vs a plain decode step (prints, no asserts).
/// Run with `CRANE_GDN_SERIAL_PREFILL=1` to time the serial kernel.
#[test]
#[ignore = "needs CRANE_QWEN35_GGUF; a timing probe"]
fn spec_forward_cost() {
    let path = std::env::var("CRANE_QWEN35_GGUF").expect("set CRANE_QWEN35_GGUF");
    let device = device();
    let mmap = crane_core::quantized::gguf_file::mmap_gguf_file(&path).unwrap();
    let (ct, _) = crane_core::quantized::extended_gguf::read_content(mmap.as_ref()).unwrap();
    let mut cursor = std::io::Cursor::new(mmap.as_ref());
    let mut m = Qwen3_5TextModel::from_gguf(ct, &mut cursor, &device).unwrap();
    let ids: Vec<u32> = (0..64u32).map(|i| 1000 + (i * 37) % 5000).collect();
    let prefill = 40usize;
    let taps = [3usize, 7, 11, 15, 19];

    let time = |label: &str, iters: usize, f: &mut dyn FnMut()| {
        // Warm up for ~0.5 s: the card idles at 210 MHz and ramps slowly.
        let warm = std::time::Instant::now();
        while warm.elapsed().as_secs_f64() < 0.5 {
            f();
            device.synchronize().unwrap();
        }
        let t = std::time::Instant::now();
        for _ in 0..iters {
            f();
        }
        device.synchronize().unwrap();
        println!(
            "{label:<42} {:.2} ms",
            t.elapsed().as_secs_f64() * 1000.0 / iters as f64
        );
    };

    m.reset_gdn_caches().unwrap();
    let h = embed(&m, &device, &ids[..prefill]);
    m.forward_embeds(&h, &positions(&device, 0, prefill), 0, None)
        .unwrap();
    let one = embed(&m, &device, &ids[prefill..prefill + 1]);
    let p1 = positions(&device, prefill, 1);
    // decode step: roll the state back each time so the cache stays put
    time("decode step S=1 (forward_embeds)", 30, &mut || {
        m.forward_embeds(&one, &p1, prefill, None).unwrap();
        m.commit_spec(prefill);
    });
    for (s, ck) in [(16usize, 16usize), (16, 4), (20, 20), (20, 5), (24, 24)] {
        let hb = embed(&m, &device, &ids[prefill..prefill + s]);
        let pb = positions(&device, prefill, s);
        time(
            &format!("forward_spec S={s} ckpt={ck} (taps+logits {})", s.min(16)),
            30,
            &mut || {
                let (l, _) = m
                    .forward_spec(&hb, &pb, prefill, ck, &taps, s.min(16), true)
                    .unwrap();
                l.argmax(1).unwrap().to_vec1::<u32>().unwrap();
                m.commit_spec(prefill);
            },
        );
    }
}
