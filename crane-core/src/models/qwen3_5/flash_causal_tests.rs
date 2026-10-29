// SPDX-License-Identifier: MIT
//! `enable_flash_causal()` against the default attention path on a real GPU.
//!
//! Both models are built from the same deterministic weights. The opted-in
//! logits must stay close to the default ones, and must not be bitwise
//! equal. Equal logits would mean the vendored kernel never ran.

use super::*;
use crate::models::flash_test_support::{compare_logits, hash_weights};
use crate::ops::fused_ops::fattn::test_support::test_gpu_device;

/// Holds off tests that set `CRANE_PREFILL_CHUNK`, so both models of a
/// comparison are prefilled with the same chunking.
fn chunk_env_guard() -> std::sync::MutexGuard<'static, ()> {
    crate::models::qwen3_5::prefill::CHUNK_ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Two full-attention layers among four, at `head_dim` 128 so the vendored
/// kernels have an instantiation.
fn tiny_config() -> Config {
    serde_json::from_str(
        r#"{"text_config":{"head_dim":128,"vocab_size":64,"hidden_size":256,
        "intermediate_size":128,"num_hidden_layers":4,"num_attention_heads":4,
        "num_key_value_heads":2,"hidden_act":"silu","max_position_embeddings":256,
        "rms_norm_eps":1e-6,
        "rope_parameters":{"rope_theta":10000.0,"mrope_section":[],
        "partial_rotary_factor":0.5,"mrope_interleaved":true},
        "full_attention_interval":2,"linear_conv_kernel_dim":4,
        "linear_key_head_dim":16,"linear_value_head_dim":16,
        "linear_num_key_heads":2,"linear_num_value_heads":4,
        "tie_word_embeddings":true,"attn_output_gate":true}}"#,
    )
    .unwrap()
}

fn build(dev: &Device, dtype: DType) -> Qwen3_5TextModel {
    let vb = hash_weights(dev, dtype);
    Qwen3_5TextModel::new(&tiny_config(), &vb, dev, dtype, None).unwrap()
}

fn tokens(dev: &Device, start: usize, len: usize) -> Tensor {
    let ids: Vec<u32> = (start..start + len)
        .map(|i| ((i * 7 + 3) % 64) as u32)
        .collect();
    Tensor::new(ids.as_slice(), dev)
        .unwrap()
        .unsqueeze(0)
        .unwrap()
}

// A prefill, a continuation prefill against a warm cache and a decode step,
// with the flag on, must track the default path. The prefills run the
// kernel and the decode step does not, so decode only moves through the
// cache the prefills filled.
#[test]
fn flash_causal_tracks_default_attention() {
    let _env = chunk_env_guard();
    let dev = test_gpu_device();
    for (dtype, tol) in [(DType::BF16, 0.12f32), (DType::F16, 0.02)] {
        let mut default = build(&dev, dtype);
        let mut flash = build(&dev, dtype);
        flash.enable_flash_causal();

        let mut start = 0;
        for (step, len) in [40usize, 17, 1].into_iter().enumerate() {
            let ids = tokens(&dev, start, len);
            let want = default.forward(&ids, start, None).unwrap();
            let got = flash.forward(&ids, start, None).unwrap();
            let (max, equal) = compare_logits(&got, &want);
            println!("{dtype:?} step {step} len {len}: max_abs={max:.5} bitwise_equal={equal}");
            assert!(max <= tol, "{dtype:?} step {step}: max_abs {max} > {tol}");
            if len > 1 {
                assert!(!equal, "{dtype:?} step {step}: kernel did not run");
            }
            start += len;
        }
    }
}

// With the flag off the model is deterministic, so two instances agree
// bitwise. This pins the comparison above to the flag and not to noise.
#[test]
fn default_attention_is_deterministic() {
    let _env = chunk_env_guard();
    let dev = test_gpu_device();
    let mut a = build(&dev, DType::BF16);
    let mut b = build(&dev, DType::BF16);
    let ids = tokens(&dev, 0, 40);
    let (max, equal) = compare_logits(
        &a.forward(&ids, 0, None).unwrap(),
        &b.forward(&ids, 0, None).unwrap(),
    );
    assert!(
        equal,
        "default path differs between instances (max_abs {max})"
    );
}

// A real checkpoint, prefilled once with the flag off and once with it on.
// Set `CRANE_QWEN35_MODEL` to a BF16 Qwen3.5 directory:
// `cargo test -p crane-core --no-default-features --features
// "rocm,onnx" --release -- --ignored --exact
// models::qwen3_5::model::flash_causal_tests::flash_causal_real_checkpoint_logits_agree
// --nocapture`
#[test]
#[ignore = "needs a real checkpoint, set CRANE_QWEN35_MODEL"]
fn flash_causal_real_checkpoint_logits_agree() {
    let _env = chunk_env_guard();
    let path = std::env::var("CRANE_QWEN35_MODEL").expect("set CRANE_QWEN35_MODEL");
    let dev = test_gpu_device();
    let mut model = Model::new(&path, &dev, &DType::BF16).unwrap();
    let ids: Vec<u32> = (0..1024u32).map(|i| (i * 31 + 7) % 1000 + 100).collect();

    let want = model.forward_step(&ids, 0).unwrap();
    model.clear_kv_cache().unwrap();
    model.enable_flash_causal();
    let got = model.forward_step(&ids, 0).unwrap();

    let flat = |t: &Tensor| {
        t.to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let (want, got) = (flat(&want), flat(&got));
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    let diff: f64 = want
        .iter()
        .zip(&got)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum();
    let norm: f64 = want.iter().map(|a| f64::from(*a).powi(2)).sum();
    let rel_l2 = (diff / norm).sqrt();
    println!(
        "relative L2 {rel_l2:.6}, argmax {} vs {}",
        argmax(&want),
        argmax(&got)
    );
    assert!(rel_l2 < 0.05, "relative L2 {rel_l2}");
    assert_eq!(argmax(&want), argmax(&got), "top-1 token changed");
}
