// SPDX-License-Identifier: MIT
//! Micro-benchmark for the `KvCache` backends (Fp/Int8/Int4).
//!
//! Isolates the per-decode-step cache append (buffer growth plus, for the
//! quantized backends, quantize-on-write/dequantize-on-read) from the rest
//! of the model, at a few pre-filled context depths. `QuantKvCache` currently
//! dequantizes the *whole* filled cache on every append. See
//! `models/modules/kv_cache.rs`'s module docs: the fused dequant-in-attention
//! kernel is still a follow-up. Its per-step cost should therefore grow with
//! depth while `Fp`'s stays flat, which is why this sweeps depth instead of
//! reporting a single number.
//!
//! Usage: `kv_cache_bench [batch] [kv_heads] [head_dim] [iters]`
//! Defaults model a `Qwen3`-style GQA layer: `batch=1`, `kv_heads=8`, `head_dim=128`,
//! 50 decode-step iterations per (kind, depth) pair. Depths swept: 0, 4096,
//! 16384, 32768 pre-filled tokens.
//!
//! This measures the cache backend alone, not a full model's end-to-end
//! decode tok/s. Attention/FFN cost isn't included, so use it to compare
//! Fp/Int8/Int4 relative cost, not as a substitute for a real server
//! benchmark.
//!
//! Device: picks whichever GPU backend the binary was built with (`cuda` >
//! `rocm`), falling back to CPU otherwise. `KvCache` has no custom kernel of
//! its own, just candle tensor ops, so CPU is a fully supported target
//! rather than a stub.

use candle_core::{DType, Device, Tensor};
use crane_core::candle_core;
use crane_core::models::modules::kv_cache::{KvCache, KvCacheKind};
use std::time::Instant;

const DEPTHS: [usize; 4] = [0, 4096, 16384, 32768];
/// Forward appends run before timing starts, to warm up allocations.
const WARMUP: usize = 5;

fn arg(i: usize, default: usize) -> usize {
    std::env::args()
        .nth(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "cuda")]
fn device() -> candle_core::Result<Device> {
    Device::new_cuda(0)
}

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
fn device() -> candle_core::Result<Device> {
    Device::new_rocm(0)
}

// All `device()` arms share this `Result`-returning signature because callers
// use `device()?` uniformly regardless of which cfg arm is compiled; this arm
// alone never actually errors.
#[cfg(not(any(feature = "cuda", feature = "rocm")))]
#[allow(clippy::unnecessary_wraps)]
fn device() -> candle_core::Result<Device> {
    Ok(Device::Cpu)
}

/// Pre-fills a fresh cache of `kind` to `depth` tokens with one prefill-shaped
/// append, then times `iters` single-token decode-shaped appends. Returns
/// (ms/step, cache byte size after the final append).
fn measure(
    kind: KvCacheKind,
    depth: usize,
    batch: usize,
    kv_heads: usize,
    head_dim: usize,
    iters: usize,
    device: &Device,
) -> anyhow::Result<(f64, usize)> {
    let mut cache = KvCache::new(kind);
    if depth > 0 {
        let k = Tensor::randn(0f32, 1.0, (batch, kv_heads, depth, head_dim), device)?
            .to_dtype(DType::F16)?;
        let v = Tensor::randn(0f32, 1.0, (batch, kv_heads, depth, head_dim), device)?
            .to_dtype(DType::F16)?;
        cache.append(&k, &v)?;
    }

    let step_k =
        Tensor::randn(0f32, 1.0, (batch, kv_heads, 1, head_dim), device)?.to_dtype(DType::F16)?;
    let step_v =
        Tensor::randn(0f32, 1.0, (batch, kv_heads, 1, head_dim), device)?.to_dtype(DType::F16)?;

    for _ in 0..WARMUP {
        std::hint::black_box(cache.append(&step_k, &step_v)?);
    }
    device.synchronize()?;

    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(cache.append(&step_k, &step_v)?);
    }
    device.synchronize()?;

    // `iters` is a small CLI-supplied iteration count, far below f64's 52-bit
    // mantissa limit, so this cast loses no precision.
    #[allow(clippy::cast_precision_loss)]
    let iters_f64 = iters as f64;
    let ms_per_step = start.elapsed().as_secs_f64() * 1000.0 / iters_f64;
    Ok((ms_per_step, cache.byte_size()))
}

fn main() -> anyhow::Result<()> {
    let batch = arg(1, 1);
    let kv_heads = arg(2, 8);
    let head_dim = arg(3, 128);
    let iters = arg(4, 50);
    let device = device()?;

    println!(
        "KvCache append benchmark  batch={batch} kv_heads={kv_heads} head_dim={head_dim} iters={iters} device={device:?}"
    );
    println!(
        "{:<6} {:>8} {:>10} {:>10} {:>12}",
        "kind", "depth", "ms/step", "tok/s", "bytes"
    );

    for (label, kind) in [
        ("fp", KvCacheKind::Fp),
        ("int8", KvCacheKind::Int8),
        ("int4", KvCacheKind::Int4),
    ] {
        for depth in DEPTHS {
            let (ms, bytes) = measure(kind, depth, batch, kv_heads, head_dim, iters, &device)?;
            let tok_s = if ms > 0.0 { 1000.0 / ms } else { f64::NAN };
            println!("{label:<6} {depth:>8} {ms:>10.4} {tok_s:>10.1} {bytes:>12}");
        }
    }
    Ok(())
}
