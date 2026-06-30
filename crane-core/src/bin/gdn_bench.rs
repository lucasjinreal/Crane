//! Micro-benchmark for the fused GDN recurrence CUDA kernel.
//!
//! Isolates the kernel from the rest of the model so kernel changes can be
//! measured directly. Usage:
//!   gdn_bench [BH] [S] [K] [V] [iters]
//! Defaults model Qwen3.5-0.8B single-sequence prefill: BH=16 S=512 K=128 V=128.

use candle_core::{DType, Device, Tensor};
use crane_core::ops::gdn::gdn_recurrence_cuda;
use std::time::Instant;

fn arg(i: usize, default: usize) -> usize {
    std::env::args()
        .nth(i)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() -> anyhow::Result<()> {
    let bh = arg(1, 16);
    let s = arg(2, 512);
    let k = arg(3, 128);
    let v = arg(4, 128);
    let iters = arg(5, 100);

    let dev = Device::new_cuda(0)?;
    let q = Tensor::randn(0f32, 1.0, (bh, s, k), &dev)?;
    let kt = Tensor::randn(0f32, 1.0, (bh, s, k), &dev)?;
    let vt = Tensor::randn(0f32, 1.0, (bh, s, v), &dev)?;
    // g: small negative log-decay; beta in (0,1).
    let g = Tensor::randn(0f32, 1.0, (bh, s), &dev)?.affine(0.01, -0.05)?;
    let beta = candle_nn::ops::sigmoid(&Tensor::randn(0f32, 1.0, (bh, s), &dev)?)?;
    let state = Tensor::zeros((bh, k, v), DType::F32, &dev)?;

    for _ in 0..10 {
        let _ = gdn_recurrence_cuda(&q, &kt, &vt, &g, &beta, &state)?;
    }
    dev.synchronize()?;

    let t = Instant::now();
    for _ in 0..iters {
        let (y, _st) = gdn_recurrence_cuda(&q, &kt, &vt, &g, &beta, &state)?;
        std::hint::black_box(&y);
    }
    dev.synchronize()?;
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // FLOPs: per timestep per (bh,v) column ~ 4*K (two fused loops). Total
    // ~ bh * s * v * 4 * k.
    let flops = (bh * s * v * 4 * k) as f64;
    let gflops = flops / (ms / 1000.0) / 1e9;
    println!("GDN recurrence  BH={bh} S={s} K={k} V={v}  ->  {ms:.3} ms/iter   ({gflops:.0} GFLOP/s)");
    Ok(())
}
