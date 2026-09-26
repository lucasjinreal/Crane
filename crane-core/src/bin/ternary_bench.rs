//! Micro-benchmark for the Prism PTQ1_0/PQ2_0 `TernaryLinear` CUDA path.
//!
//! Usage: `ternary_bench [output_rows] [cols] [input_rows] [iters]`
//! Defaults to a representative Bonsai-27B decode projection: 5120x5120,
//! one input row, 100 iterations. Both encodings are measured with and
//! without the 1024-wide normalized Hadamard transform.

#[cfg(not(feature = "cuda"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("ternary_bench needs --features cuda")
}

#[cfg(feature = "cuda")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "cuda")]
use crane_core::candle_core;
#[cfg(feature = "cuda")]
use crane_core::quantized::ternary::{HadamardMode, TernaryEncoding, TernaryLinear, TernaryWeight};
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
fn arg(index: usize, default: usize) -> usize {
    std::env::args()
        .nth(index)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "cuda")]
fn measure(
    device: &Device,
    encoding: TernaryEncoding,
    output_rows: usize,
    cols: usize,
    input_rows: usize,
    iters: usize,
    hadamard: bool,
    dtype: DType,
) -> anyhow::Result<f64> {
    let bytes = output_rows * (cols / 128) * encoding.block_bytes();
    let mut packed = vec![0u8; bytes];
    // Use a finite, non-zero scale. The packed codes themselves are not
    // intended to represent a trained matrix; only kernel work and traffic
    // matter to this benchmark.
    let scale = half::f16::from_f32(1.0).to_bits().to_le_bytes();
    for block in packed.chunks_exact_mut(encoding.block_bytes()) {
        match encoding {
            TernaryEncoding::Pq2_0 => block[..2].copy_from_slice(&scale),
            TernaryEncoding::Ptq1_0 => block[26..28].copy_from_slice(&scale),
        }
    }
    let weight = Arc::new(TernaryWeight::new(
        encoding,
        packed,
        output_rows,
        cols,
        device,
    )?);
    let signs = Arc::new(vec![1.0f32; cols]);
    let mode = if hadamard {
        HadamardMode::Forward
    } else {
        HadamardMode::None
    };
    let linear = TernaryLinear::new(weight, signs, 1024, mode, None)?;
    let input = Tensor::randn(0f32, 1.0, (input_rows, cols), device)?.to_dtype(dtype)?;

    for _ in 0..10 {
        std::hint::black_box(linear.forward(&input)?);
    }
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(linear.forward(&input)?);
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1000.0 / iters as f64)
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    let output_rows = arg(1, 5120);
    let cols = arg(2, 5120);
    let input_rows = arg(3, 1);
    let iters = arg(4, 100);
    let device = Device::new_cuda(0)?;
    println!(
        "TernaryLinear CUDA  output_rows={output_rows} cols={cols} input_rows={input_rows} iters={iters}"
    );
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        for encoding in [TernaryEncoding::Ptq1_0, TernaryEncoding::Pq2_0] {
            for hadamard in [false, true] {
                let ms = measure(
                    &device,
                    encoding,
                    output_rows,
                    cols,
                    input_rows,
                    iters,
                    hadamard,
                    dtype,
                )?;
                let label = if hadamard { "+hadamard" } else { "matvec" };
                let gib = (output_rows * (cols / 128) * encoding.block_bytes() * input_rows) as f64
                    / (1024.0 * 1024.0 * 1024.0);
                let bandwidth = gib / (ms / 1000.0);
                println!(
                    "  {dtype:?} {encoding:?} {label:<10} {ms:8.3} ms   {bandwidth:7.1} GiB/s weights"
                );
            }
        }
    }
    Ok(())
}
