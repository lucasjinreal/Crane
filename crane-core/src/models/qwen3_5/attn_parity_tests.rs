//! Numerical parity: PR #181's fused flash-attention kernel path vs the
//! pre-PR matmul-SDPA path, on a real Qwen3.5 BF16 checkpoint.
//!
//! `attn_dispatch::matmul_sdpa_forced` (gated by `CRANE_FORCE_MATMUL_SDPA`)
//! is cached once per process, so a single process can only ever exercise
//! one of the two paths. These two tests bridge across a safetensors dump
//! instead of toggling the flag mid-process — run them in order:
//!
//! ```sh
//! CRANE_QWEN35_MODEL=/path/to/qwen3.5-bf16 \
//! CRANE_ATTN_PARITY_DUMP=/tmp/attn_parity.safetensors \
//! cargo test -p crane-core --features cuda \
//!   --exact models::qwen3_5::model::attn_parity_tests::attn_parity_dump_matmul_reference \
//!   -- --ignored --nocapture
//!
//! CRANE_QWEN35_MODEL=/path/to/qwen3.5-bf16 \
//! CRANE_ATTN_PARITY_DUMP=/tmp/attn_parity.safetensors \
//! cargo test -p crane-core --features cuda \
//!   --exact models::qwen3_5::model::attn_parity_tests::attn_kernel_vs_matmul_sdpa \
//!   -- --ignored --nocapture
//! ```
//!
//! A nonzero diff here may come from the new fused kernel, from the
//! matmul-SDPA fallback's own softmax-precision change (native dtype
//! instead of an F32 upcast, see `attn_dispatch`'s module doc), or both —
//! this doesn't distinguish between the two.
use super::*;

#[derive(Debug, Default)]
struct ErrorStats {
    max_abs: f64,
    mean_abs: f64,
    relative_l2: f64,
}

fn error_stats(actual: &Tensor, reference: &Tensor) -> Result<ErrorStats> {
    anyhow::ensure!(actual.dims() == reference.dims(), "tensor shape mismatch");
    let a = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let b = reference
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    anyhow::ensure!(!a.is_empty(), "empty comparison");
    let (mut max, mut sum, mut squared, mut norm) = (0f64, 0f64, 0f64, 0f64);
    for (a, b) in a.into_iter().zip(&b) {
        anyhow::ensure!(a.is_finite() && b.is_finite(), "nonfinite tensor");
        let d = f64::from(a) - f64::from(*b);
        max = max.max(d.abs());
        sum += d.abs();
        squared += d * d;
        norm += f64::from(*b).powi(2);
    }
    Ok(ErrorStats {
        max_abs: max,
        mean_abs: sum / b.len() as f64,
        relative_l2: squared.sqrt() / norm.sqrt().max(1e-12),
    })
}

/// Prefill lengths checked by both tests; `prompt_tokens` requires at least
/// this many tokens.
const PREFIXES: [usize; 2] = [64, 256];

#[cfg(feature = "cuda")]
fn gpu_device() -> Result<Device> {
    Ok(Device::new_cuda(0)?)
}

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
fn gpu_device() -> Result<Device> {
    Ok(Device::new_rocm(0)?)
}

/// Tokenizes the same fixed prompt `fla_decode_tests` uses, long enough to
/// cover every length in [`PREFIXES`].
fn prompt_tokens(model_path: &str) -> Result<Vec<u32>> {
    let tokenizer = Tokenizer::from_file(std::path::Path::new(model_path).join("tokenizer.json"))
        .map_err(E::msg)?;
    let text = "Explain how a computer processes a sentence, stores intermediate results, and generates the next word. Use a concrete example and describe each calculation carefully. ".repeat(40);
    let encoding = tokenizer.encode(text, false).map_err(E::msg)?;
    let tokens = encoding.get_ids().to_vec();
    let need = PREFIXES.into_iter().max().unwrap_or(0);
    anyhow::ensure!(tokens.len() >= need, "need at least {need} tokens");
    Ok(tokens)
}

/// Runs one prefill and returns `(hidden state, last-position logits)`.
fn prefill(model: &mut Model, tokens: &[u32], prefix: usize) -> Result<(Tensor, Tensor)> {
    model.inner.reset_gdn_caches()?;
    let dev = model.device.clone();
    let ids = Tensor::new(&tokens[..prefix], &dev)?.unsqueeze(0)?;
    let mask = super::super::prefill::causal_mask(prefix, 0, &dev, model.inner.attention_dtype())?;
    let hidden = model.inner.forward_layers(&ids, 0, Some(&mask))?;
    let last = hidden.narrow(1, prefix - 1, 1)?.contiguous()?;
    let logits = model.inner.head(&last)?;
    Ok((hidden, logits))
}

/// Dumps the matmul-SDPA path's hidden states and logits for each prefix in
/// [`PREFIXES`] to `CRANE_ATTN_PARITY_DUMP`. Run this first; see this
/// module's doc comment for the full two-step invocation.
#[test]
#[ignore = "needs CUDA/ROCm, CRANE_QWEN35_MODEL, CRANE_ATTN_PARITY_DUMP; run before attn_kernel_vs_matmul_sdpa"]
fn attn_parity_dump_matmul_reference() -> Result<()> {
    // Safe: nothing in this process has read `CRANE_FORCE_MATMUL_SDPA` yet,
    // so this is the value `attn_dispatch::matmul_sdpa_forced`'s `OnceLock`
    // will cache.
    unsafe {
        std::env::set_var("CRANE_FORCE_MATMUL_SDPA", "1");
    }
    let path = std::env::var("CRANE_QWEN35_MODEL").context("set CRANE_QWEN35_MODEL")?;
    let dump_path =
        std::env::var("CRANE_ATTN_PARITY_DUMP").context("set CRANE_ATTN_PARITY_DUMP")?;
    let dev = gpu_device()?;
    let mut model = Model::new(&path, &dev, &DType::BF16)?;
    let tokens = prompt_tokens(&path)?;

    let mut dump = std::collections::HashMap::new();
    for prefix in PREFIXES {
        let (hidden, logits) = prefill(&mut model, &tokens, prefix)?;
        dump.insert(format!("hidden_{prefix}"), hidden);
        dump.insert(format!("logits_{prefix}"), logits);
    }
    candle_core::safetensors::save(&dump, &dump_path)?;
    println!("wrote matmul-SDPA reference dump to {dump_path}");
    Ok(())
}

/// Compares the fused flash-attention kernel path (default — no
/// `CRANE_FORCE_MATMUL_SDPA` set) against
/// [`attn_parity_dump_matmul_reference`]'s dump. Prints `max_abs`/
/// `mean_abs`/`relative_l2` per prefix for hidden state and logits; a human
/// reading the output judges whether the gap is acceptable; there's no
/// established tolerance for this particular comparison yet. Only asserts
/// every stat is finite.
#[test]
#[ignore = "needs CUDA/ROCm, CRANE_QWEN35_MODEL, and CRANE_ATTN_PARITY_DUMP from a prior attn_parity_dump_matmul_reference run"]
fn attn_kernel_vs_matmul_sdpa() -> Result<()> {
    let path = std::env::var("CRANE_QWEN35_MODEL").context("set CRANE_QWEN35_MODEL")?;
    let dump_path =
        std::env::var("CRANE_ATTN_PARITY_DUMP").context("set CRANE_ATTN_PARITY_DUMP")?;
    let dev = gpu_device()?;
    let dump = candle_core::safetensors::load(&dump_path, &dev)?;
    let mut model = Model::new(&path, &dev, &DType::BF16)?;
    let tokens = prompt_tokens(&path)?;

    for prefix in PREFIXES {
        let (hidden, logits) = prefill(&mut model, &tokens, prefix)?;
        let want_hidden = dump.get(&format!("hidden_{prefix}")).with_context(|| {
            format!("dump is missing hidden_{prefix}; was it built with the same PREFIXES?")
        })?;
        let want_logits = dump.get(&format!("logits_{prefix}")).with_context(|| {
            format!("dump is missing logits_{prefix}; was it built with the same PREFIXES?")
        })?;

        let hidden_stats = error_stats(&hidden, want_hidden)?;
        let logits_stats = error_stats(&logits, want_logits)?;
        println!("prefix={prefix}: hidden=({hidden_stats:?}) logits=({logits_stats:?})");
        for stats in [&hidden_stats, &logits_stats] {
            anyhow::ensure!(
                stats.max_abs.is_finite()
                    && stats.mean_abs.is_finite()
                    && stats.relative_l2.is_finite(),
                "non-finite error stat at prefix={prefix}: {stats:?}"
            );
        }
    }
    Ok(())
}
