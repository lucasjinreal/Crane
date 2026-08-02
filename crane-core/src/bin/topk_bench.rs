//! Micro-benchmark for the GPU top-k kernel (CUDA or ROCm).
//!
//! Times the device kernel against the host sort it replaces, on the same
//! logits, so a kernel change can be judged without running a whole decode.
//! Usage:
//!   topk_bench [N] [K] [iters]
//! Defaults model Qwen3.5 sampling: N=248320 (vocab), K=40, 200 iterations.
//!
//! Both arms synchronise the device on each side of the timed region, so the
//! numbers are kernel time and not queue time. The first run of the process
//! also pays a `hipcc` compile on ROCm, so a warmup is discarded.
//!
//! Needs `--features cuda` or `--features rocm`; `required-features` cannot
//! express that (it is an AND), so the no-backend build is a stub that errors.

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("topk_bench needs a GPU backend: build with --features cuda or --features rocm")
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
use candle_core::{Device, Tensor};
#[cfg(any(feature = "cuda", feature = "rocm"))]
use std::time::Instant;

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn arg(i: usize, default: usize) -> usize {
    std::env::args()
        .nth(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Mean and standard deviation of a set of per-iteration times, in µs.
#[cfg(any(feature = "cuda", feature = "rocm"))]
fn stats(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
fn main() -> anyhow::Result<()> {
    let n = arg(1, 248_320);
    let k = arg(2, 40);
    let iters = arg(3, 200);

    #[cfg(feature = "cuda")]
    let dev = Device::new_cuda(0)?;
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    let dev = Device::new_rocm(0)?;

    let logits = Tensor::randn(0f32, 4.0, n, &dev)?;

    let run = |label: &str, f: &dyn Fn() -> candle_core::Result<Tensor>| -> anyhow::Result<()> {
        for _ in 0..20 {
            let _ = f()?;
        }
        dev.synchronize()?;

        let mut samples = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = Instant::now();
            let out = f()?;
            dev.synchronize()?;
            samples.push(t.elapsed().as_secs_f64() * 1e6);
            drop(out);
        }
        let (mean, sd) = stats(&samples);
        println!("{label:<22} N={n} K={k}  ->  {mean:8.1} µs ± {sd:.1}");
        Ok(())
    };

    run("topk_indices (gpu)", &|| {
        crane_core::ops::topk_indices(&logits, k)
    })?;
    run("portable (host sort)", &|| {
        crane_core::ops::portable::topk_indices(&logits, k)
    })?;
    Ok(())
}
